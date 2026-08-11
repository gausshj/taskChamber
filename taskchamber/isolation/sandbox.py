"""OS-level sandboxing adapters for isolated agent runtimes.

The :class:`Sandbox` port is vendor-neutral: a runtime stages a filtered
workspace through it and receives a ``cli_path`` wrapper that runs the agent
executable inside an OS sandbox. The runtime never inspects platform-specific
flags, so a new platform only adds one small adapter.

Boundary statement:

* :class:`NoSandbox` still stages a filtered copy and launches the agent CLI
  with a clean, allowlisted environment, but applies no OS process enforcement.
* :class:`BubblewrapSandbox` (Linux) mounts the host filesystem read-only,
  hides both host-home sources, isolates host process namespaces, keeps the
  staged workspace read-only, and makes only per-task config writable.
* :class:`MacOSSandboxExecSandbox` (macOS) hides both host-home sources, denies
  other-process inspection, and denies writes outside per-task config. It
  remains a best-effort development boundary because Seatbelt is deprecated
  and reads outside those homes plus network access remain available.

Neither sandbox isolates network: the task must still reach its provider, and the
read-only tool preset (no Bash/WebFetch/WebSearch) is the network-exposure
control. Real OS-level process isolation is defense in depth on top of the
application-level :class:`~taskchamber.core.policy.WorkspaceGuard`.
"""

from __future__ import annotations

import inspect
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..core.contracts import ExecutionPolicy
from ..core.policy import staged_workspace

DEFAULT_MAX_TOTAL_BYTES = 50_000_000
DEFAULT_MAX_SNAPSHOT_FILES = 10_000
DEFAULT_CHILD_PATH = "/usr/bin:/bin"
POSIX_LAUNCHER_SUPPORTED = os.name != "nt"
CLI_LAUNCH_OBSERVATION_FILE = "cli-main.started"
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OBSERVED_CLI_SCRIPT = """\
executable=$1
marker=$2
shift 2
if [ "$#" -eq 1 ] && [ "$1" = "-v" ]; then
    exec "$executable" "$@"
fi
umask 077
set -C
: > "$marker" || exit 125
set +C
exec "$executable" "$@"
"""


def _host_home_directories() -> tuple[Path, ...]:
    """Return both environment-selected and account-database home paths."""

    homes = {Path(os.path.expanduser("~")).resolve()}
    try:
        import pwd

        homes.add(Path(pwd.getpwuid(os.getuid()).pw_dir).expanduser().resolve())
    except (ImportError, KeyError, OSError):
        pass
    return tuple(sorted(homes - {Path("/")}, key=lambda path: len(path.parts), reverse=True))


@dataclass(frozen=True)
class IsolatedWorkspace:
    """The directory tree an agent runtime may touch for one task."""

    root: Path
    allowed_paths: tuple[Path, ...]


class Sandbox:
    """Vendor-neutral isolation boundary implemented per host platform.

    The base behavior stages a filtered workspace via :func:`staged_workspace`.
    OS-specific subclasses may additionally render a POSIX wrapper script that
    ``exec``\\ s the agent executable inside the platform sandbox.
    """

    name = "sandbox"
    os_isolated = False
    secure_cli_launcher = False
    launch_observation_inside_os_sandbox = False
    operational_preflight = False

    @property
    def available(self) -> bool:
        """Whether this sandbox's OS tool is installed and usable."""

        return True

    def preflight(self) -> bool:
        """Return whether the configured boundary can start on this host."""

        return self.available

    def validate_readable_paths(self, paths: tuple[Path, ...]) -> None:
        """Reject forwarded path settings hidden by this boundary."""

        _reject_masked_paths(paths, ())

    @contextmanager
    def isolate(self, policy: ExecutionPolicy) -> Iterator[IsolatedWorkspace]:
        """Yield the workspace the runtime should use as its working directory.

        Even the default implementation uses a filtered temporary copy. This is
        application-level exposure control, not OS process enforcement.
        """

        with staged_workspace(
            policy.workspace_root,
            source_paths=policy.allowed_paths,
            max_file_bytes=policy.max_file_bytes,
            max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
            max_files=DEFAULT_MAX_SNAPSHOT_FILES,
        ) as snapshot:
            yield IsolatedWorkspace(root=snapshot.root, allowed_paths=snapshot.allowed_paths)

    def prepare_wrapper(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path | None = None,
        environment_keys: tuple[str, ...] = (),
    ) -> Path:
        """Return the clean-environment ``cli_path`` the SDK should launch.

        The current contract uses a generated launcher so the SDK cannot leak
        arbitrary parent-process environment variables into the CLI.
        OS-isolating subclasses additionally enter their native process
        sandbox. A call using only the original three arguments preserves the
        pre-hardening return-the-executable behavior for external callers;
        runtimes should use :meth:`prepare_cli_launcher`.
        """

        if launcher_dir is None and not environment_keys:
            # Preserve the original public method contract for external callers.
            # The Claude runtime uses prepare_cli_launcher(), which always opts
            # into the hardened launcher path below.
            return Path(executable)

        effective_launcher_dir = self._launcher_directory(config_dir, launcher_dir)
        resolved_executable = str(Path(executable).expanduser().resolve())
        return self._write_wrapper(
            _observed_cli_argv(
                resolved_executable,
                config_dir / CLI_LAUNCH_OBSERVATION_FILE,
            ),
            workspace=workspace,
            config_dir=config_dir,
            launcher_dir=effective_launcher_dir,
            environment_keys=environment_keys,
        )

    def prepare_cli_launcher(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path,
        environment_keys: tuple[str, ...] = (),
    ) -> Path:
        """Prepare the hardened launcher while preserving legacy adapters.

        Sandboxes implementing the current contract opt in with
        ``secure_cli_launcher``. Older third-party adapters are called through
        their original ``prepare_wrapper`` signature, then placed behind a
        clean-environment outer launcher. Their native-isolation state remains
        conservatively unobserved until they migrate.
        """

        if self.uses_secure_cli_launcher:
            return self.prepare_wrapper(
                workspace,
                executable=executable,
                config_dir=config_dir,
                launcher_dir=launcher_dir,
                environment_keys=environment_keys,
            )
        legacy_wrapper = self.prepare_wrapper(
            workspace,
            executable=executable,
            config_dir=config_dir,
        )
        return Sandbox._write_wrapper(
            _observed_cli_argv(
                str(Path(legacy_wrapper).expanduser().resolve()),
                config_dir / CLI_LAUNCH_OBSERVATION_FILE,
            ),
            workspace=workspace,
            config_dir=config_dir,
            launcher_dir=self._launcher_directory(config_dir, launcher_dir),
            environment_keys=environment_keys,
        )

    @property
    def uses_secure_cli_launcher(self) -> bool:
        """Whether this instance implements the current launcher signature."""

        if type(self).prepare_wrapper is Sandbox.prepare_wrapper:
            return True
        if not self.secure_cli_launcher:
            return False
        try:
            parameters = inspect.signature(self.prepare_wrapper).parameters.values()
        except (TypeError, ValueError):
            return False
        names = {parameter.name for parameter in parameters}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        return accepts_kwargs or {"launcher_dir", "environment_keys"}.issubset(names)

    @staticmethod
    def _launcher_directory(config_dir: Path, launcher_dir: Path | None) -> Path:
        """Keep legacy callers safe without placing launchers in writable config."""

        selected = launcher_dir or config_dir.with_name(f"{config_dir.name}-launcher")
        resolved_config = config_dir.resolve()
        resolved_launcher = selected.resolve()
        if (
            resolved_launcher == resolved_config
            or resolved_launcher.is_relative_to(resolved_config)
            or resolved_config.is_relative_to(resolved_launcher)
        ):
            raise ValueError("launcher directory must be outside writable CLI config")
        return selected

    @staticmethod
    def _write_wrapper(
        argv_prefix: list[str],
        *,
        workspace: IsolatedWorkspace,
        config_dir: Path,
        launcher_dir: Path,
        environment_keys: tuple[str, ...],
    ) -> Path:
        """Write a launcher containing environment names but never values."""

        if not POSIX_LAUNCHER_SUPPORTED:
            raise ValueError("the clean CLI launcher currently requires a POSIX host")
        config_dir.mkdir(parents=True, exist_ok=True)
        launcher_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = config_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for directory in (config_dir, launcher_dir, temp_dir):
            os.chmod(directory, 0o700)

        fixed_names = {"HOME", "PATH", "PWD", "TEMP", "TMP", "TMPDIR"}
        keys = tuple(dict.fromkeys(key for key in environment_keys if key not in fixed_names))
        invalid = tuple(key for key in keys if _ENVIRONMENT_NAME.fullmatch(key) is None)
        if invalid:
            raise ValueError("CLI environment allowlist contains an invalid variable name")

        fixed_environment = {
            "HOME": str(config_dir),
            "PATH": DEFAULT_CHILD_PATH,
            "PWD": str(workspace.root),
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
        }
        helper = launcher_dir / "agent-cli-exec.py"
        helper.write_text(
            "import os\n"
            "import sys\n\n"
            f"COMMAND = {tuple(argv_prefix)!r}\n"
            f"FIXED_ENVIRONMENT = {fixed_environment!r}\n"
            f"FORWARDED_KEYS = {keys!r}\n"
            f"WORKSPACE = {str(workspace.root)!r}\n\n"
            "arguments = sys.argv[1:]\n"
            "environment = dict(FIXED_ENVIRONMENT)\n"
            "if arguments != ['-v']:\n"
            "    environment.update(\n"
            "        (key, os.environ[key]) for key in FORWARDED_KEYS if key in os.environ\n"
            "    )\n"
            "os.chdir(WORKSPACE)\n"
            "os.execve(COMMAND[0], [*COMMAND, *arguments], environment)\n",
            encoding="utf-8",
        )
        os.chmod(helper, 0o400)

        quoted_python = shlex.quote(sys.executable)
        quoted_helper = shlex.quote(str(helper))
        script = f'#!/bin/sh\nexec {quoted_python} -I -S {quoted_helper} "$@"\n'
        wrapper = launcher_dir / "agent-cli-launcher.sh"
        wrapper.write_text(script, encoding="utf-8")
        os.chmod(wrapper, 0o500)
        return wrapper

    @staticmethod
    def _write_legacy_wrapper(argv_prefix: list[str], *, config_dir: Path) -> Path:
        """Preserve the original inherited-environment wrapper contract."""

        config_dir.mkdir(parents=True, exist_ok=True)
        quoted = " ".join(shlex.quote(part) for part in argv_prefix)
        wrapper = config_dir / "sandbox-wrapper.sh"
        wrapper.write_text(f'#!/bin/sh\nexec {quoted} "$@"\n', encoding="utf-8")
        os.chmod(wrapper, 0o700)
        return wrapper


class NoSandbox(Sandbox):
    """Filtered temporary workspace with no OS-level process enforcement."""

    name = "none"
    os_isolated = False
    secure_cli_launcher = True

    def __init__(
        self,
        *,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_snapshot_files: int = DEFAULT_MAX_SNAPSHOT_FILES,
    ) -> None:
        self.max_total_bytes = max_total_bytes
        self.max_snapshot_files = max_snapshot_files

    @contextmanager
    def isolate(self, policy: ExecutionPolicy) -> Iterator[IsolatedWorkspace]:
        with staged_workspace(
            policy.workspace_root,
            source_paths=policy.allowed_paths,
            max_file_bytes=policy.max_file_bytes,
            max_total_bytes=self.max_total_bytes,
            max_files=self.max_snapshot_files,
        ) as snapshot:
            yield IsolatedWorkspace(root=snapshot.root, allowed_paths=snapshot.allowed_paths)


class _StagingSandbox(Sandbox):
    """Shared staging behavior for sandboxes that build a filtered copy."""

    os_isolated = True

    def __init__(
        self,
        *,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_snapshot_files: int = DEFAULT_MAX_SNAPSHOT_FILES,
    ) -> None:
        self.max_total_bytes = max_total_bytes
        self.max_snapshot_files = max_snapshot_files

    @contextmanager
    def isolate(self, policy: ExecutionPolicy) -> Iterator[IsolatedWorkspace]:
        with staged_workspace(
            policy.workspace_root,
            source_paths=policy.allowed_paths,
            max_file_bytes=policy.max_file_bytes,
            max_total_bytes=self.max_total_bytes,
            max_files=self.max_snapshot_files,
        ) as snapshot:
            yield IsolatedWorkspace(root=snapshot.root, allowed_paths=snapshot.allowed_paths)

    def prepare_wrapper(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path | None = None,
        environment_keys: tuple[str, ...] = (),
    ) -> Path:
        if launcher_dir is None and not environment_keys:
            argv = self._sandbox_argv(
                workspace,
                executable=executable,
                config_dir=config_dir,
                launcher_dir=config_dir,
            )
            return self._write_legacy_wrapper(argv, config_dir=config_dir)
        effective_launcher_dir = self._launcher_directory(config_dir, launcher_dir)
        argv = self._sandbox_argv(
            workspace,
            executable=executable,
            config_dir=config_dir,
            launcher_dir=effective_launcher_dir,
        )
        return self._write_wrapper(
            argv,
            workspace=workspace,
            config_dir=config_dir,
            launcher_dir=effective_launcher_dir,
            environment_keys=environment_keys,
        )

    def _sandbox_argv(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path,
    ) -> list[str]:
        raise NotImplementedError


class BubblewrapSandbox(_StagingSandbox):
    """Linux isolation via bubblewrap user/mount namespaces.

    The host filesystem is mounted read-only, ``$HOME`` is replaced with an empty
    tmpfs, and the staged workspace remains read-only. Only the per-task config
    directory is writable. Network is intentionally not unshared so the provider
    call works.
    """

    name = "bwrap"
    secure_cli_launcher = True
    launch_observation_inside_os_sandbox = True
    operational_preflight = True

    def __init__(
        self,
        *,
        bwrap: str | None = None,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_snapshot_files: int = DEFAULT_MAX_SNAPSHOT_FILES,
    ) -> None:
        super().__init__(
            max_total_bytes=max_total_bytes,
            max_snapshot_files=max_snapshot_files,
        )
        selected = bwrap or shutil.which("bwrap") or "bwrap"
        self._bwrap = shutil.which(selected) or selected

    @property
    def available(self) -> bool:
        return sys.platform.startswith("linux") and shutil.which(self._bwrap) is not None

    def preflight(self) -> bool:
        if not self.available:
            return False
        return _preflight_wrapper(self, executable="/bin/true")

    def validate_readable_paths(self, paths: tuple[Path, ...]) -> None:
        _reject_masked_paths(paths, (Path("/tmp"), *_host_home_directories()))

    def _sandbox_argv(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path,
    ) -> list[str]:
        del launcher_dir
        resolved_executable = Path(executable).expanduser().resolve()
        argv: list[str] = [
            self._bwrap,
            "--die-with-parent",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",
            "/tmp",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
        ]
        masked_homes = _host_home_directories()
        for home in reversed(masked_homes):
            argv += ["--tmpfs", str(home)]

        # Hiding HOME and /tmp can also hide the selected executable. Expose only
        # its canonical file below the most specific masked root. Empty parent
        # directories reveal no adjacent content. The anchor and every component
        # below it must resist replacement by another local user: a writable
        # masked root needs the sticky bit (the root-owned /tmp shape), and no
        # component below it may be foreign-owned, group/world-writable, or a
        # post-canonicalization symlink.
        masked_roots = tuple(
            sorted(
                {Path("/tmp"), *masked_homes} - {Path("/")},
                key=lambda path: len(path.parts),
                reverse=True,
            )
        )
        masked_root = next(
            (root for root in masked_roots if resolved_executable.is_relative_to(root)),
            None,
        )
        if masked_root is not None:
            _verify_rebindable_executable(resolved_executable, masked_root)
            directory = masked_root
            for part in resolved_executable.parent.relative_to(masked_root).parts:
                directory /= part
                argv += ["--dir", str(directory)]
            argv += [
                "--ro-bind",
                str(resolved_executable),
                str(resolved_executable),
            ]

        argv += [
            "--bind",
            str(config_dir),
            str(config_dir),
            "--ro-bind",
            str(workspace.root),
            str(workspace.root),
            "--setenv",
            "HOME",
            str(config_dir),
            "--chdir",
            str(workspace.root),
        ]

        argv += [
            "--",
            *_observed_cli_argv(
                str(resolved_executable),
                config_dir / CLI_LAUNCH_OBSERVATION_FILE,
            ),
        ]
        return argv


class MacOSSandboxExecSandbox(_StagingSandbox):
    """macOS isolation via ``sandbox-exec`` (Seatbelt).

    The profile denies reads of the host home directory except for the exact
    configured executable and denies writes outside the per-task config
    directory. Reads elsewhere and network access remain available, so this is
    best-effort development hardening rather than a production container
    boundary.
    """

    name = "sandbox-exec"
    secure_cli_launcher = True
    launch_observation_inside_os_sandbox = True
    operational_preflight = True

    def __init__(
        self,
        *,
        sandbox_exec: str | None = None,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_snapshot_files: int = DEFAULT_MAX_SNAPSHOT_FILES,
    ) -> None:
        super().__init__(
            max_total_bytes=max_total_bytes,
            max_snapshot_files=max_snapshot_files,
        )
        selected = sandbox_exec or shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        self._sandbox_exec = shutil.which(selected) or selected

    @property
    def available(self) -> bool:
        return sys.platform == "darwin" and shutil.which(self._sandbox_exec) is not None

    def preflight(self) -> bool:
        if not self.available:
            return False
        return _preflight_wrapper(self, executable="/usr/bin/true")

    def validate_readable_paths(self, paths: tuple[Path, ...]) -> None:
        _reject_masked_paths(paths, _host_home_directories())

    @staticmethod
    def _profile_literal(path: Path) -> str:
        value = str(path)
        if "\n" in value or "\r" in value:
            raise ValueError("sandbox profile paths must not contain line breaks")
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _profile(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
    ) -> str:
        resolved_executable = self._profile_literal(Path(executable).expanduser().resolve())
        workspace_root = self._profile_literal(workspace.root.resolve())
        runtime_config = self._profile_literal(config_dir.resolve())
        rules = [
            "(version 1)",
            "(allow default)",
            "(deny process-info*)",
            "(allow process-info* (target self))",
        ]
        rules.extend(
            f'(deny file-read* (subpath "{self._profile_literal(home)}"))'
            for home in _host_home_directories()
        )
        rules.extend(
            [
                f'(allow file-read* (literal "{resolved_executable}"))',
                f'(allow file-read* (subpath "{workspace_root}"))',
                f'(allow file-read* (subpath "{runtime_config}"))',
                "(deny file-write*)",
                '(allow file-write* (literal "/dev/null"))',
                f'(allow file-write* (subpath "{runtime_config}"))',
            ]
        )
        return "\n".join(rules) + "\n"

    def _sandbox_argv(
        self,
        workspace: IsolatedWorkspace,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path,
    ) -> list[str]:
        launcher_dir.mkdir(parents=True, exist_ok=True)
        profile_path = launcher_dir / "sandbox-exec.sb"
        profile_path.write_text(
            self._profile(workspace, executable=executable, config_dir=config_dir),
            encoding="utf-8",
        )
        os.chmod(profile_path, 0o400)
        return [
            self._sandbox_exec,
            "-f",
            str(profile_path),
            *_observed_cli_argv(
                str(Path(executable).expanduser().resolve()),
                config_dir / CLI_LAUNCH_OBSERVATION_FILE,
            ),
        ]


def select_sandbox(mode: str = "auto") -> Sandbox:
    """Pick a sandbox for the composition root from a mode string.

    ``auto`` selects the platform's native sandbox only after its generated
    launcher passes an operational probe, and falls back to :class:`NoSandbox`
    otherwise. ``required`` selects the native sandbox and fails when the same
    probe fails. Explicit native modes also fail closed.
    """

    normalized = mode.strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return NoSandbox()
    if normalized in {"bwrap", "bubblewrap"}:
        return _require_available(BubblewrapSandbox())
    if normalized in {"sandbox-exec", "mac", "macos", "seatbelt"}:
        return _require_available(MacOSSandboxExecSandbox())
    required = normalized in {"required", "require", "on", "true", "1"}
    if normalized != "auto" and not required:
        raise ValueError(f"unsupported TASKCHAMBER_SANDBOX: {mode!r}")

    if sys.platform.startswith("linux"):
        candidate: Sandbox = BubblewrapSandbox()
    elif sys.platform == "darwin":
        candidate = MacOSSandboxExecSandbox()
    elif required:
        raise ValueError("OS sandboxing is required but unsupported on this platform")
    else:
        return NoSandbox()
    if required:
        return _require_available(candidate)
    return candidate if candidate.preflight() else NoSandbox()


def _require_available(sandbox: Sandbox) -> Sandbox:
    if not sandbox.preflight():
        raise ValueError(f"requested OS sandbox {sandbox.name!r} is unavailable")
    return sandbox


def _observed_cli_argv(executable: str, marker: Path) -> list[str]:
    """Run the main CLI after recording entry into the effective boundary."""

    return [
        "/bin/sh",
        "-c",
        _OBSERVED_CLI_SCRIPT,
        "taskchamber-cli",
        executable,
        str(marker),
    ]


def _verify_rebindable_executable(executable: Path, masked_root: Path) -> None:
    """Fail closed when a masked-root executable can be swapped by another user.

    ``executable`` is already canonical. The masked root itself must be a real
    directory owned by the effective user or root; a group/world-writable root
    is only acceptable with the sticky bit, which is what makes a root-owned
    ``/tmp`` a safe anchor while a misconfigured non-sticky ``0777`` home is
    rejected. Every component below the root must be a real filesystem object
    owned by the effective user or root and must not be group/world-writable.
    A symlink component after canonicalization means the path changed between
    resolution and launch. Same-uid replacement remains possible by
    construction; the boundary enforced here is against other local users.
    """

    euid = os.geteuid()
    try:
        root_metadata = masked_root.lstat()
    except OSError as exc:
        raise ValueError("the masked root is unreadable") from exc
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("the masked root changed after canonicalization")
    if root_metadata.st_uid not in {euid, 0}:
        raise ValueError("the masked root is owned by another user")
    if root_metadata.st_mode & 0o022 and not root_metadata.st_mode & stat.S_ISVTX:
        raise ValueError("the masked root is writable without a sticky bit")

    components = [executable]
    parent = executable.parent
    while parent != masked_root:
        components.append(parent)
        parent = parent.parent
    for component in components:
        try:
            metadata = component.lstat()
        except OSError as exc:
            raise ValueError("the selected CLI is unreadable below a masked root") from exc
        _check_rebind_metadata(metadata, euid=euid)


def _check_rebind_metadata(metadata: os.stat_result, *, euid: int) -> None:
    """Reject swappable components below a masked root."""

    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("the selected CLI path changed after canonicalization")
    if metadata.st_uid not in {euid, 0}:
        raise ValueError("the selected CLI is owned by another user below a masked root")
    if metadata.st_mode & 0o022:
        raise ValueError("the selected CLI is writable by group or others below a masked root")


def _reject_masked_paths(paths: tuple[Path, ...], masked_roots: tuple[Path, ...]) -> None:
    # Canonicalize the masked roots before comparison so a host layout where the
    # root is itself a symlink (e.g. macOS ``/tmp`` -> ``/private/tmp``) cannot
    # let a resolved candidate slip past an unresolved literal. ``strict=False``
    # keeps rejection working even when a configured root does not currently
    # exist on this host, and matches the kernel by resolving the root the same
    # way as the candidate.
    resolved_roots = tuple(root.resolve(strict=False) for root in masked_roots)
    for path in paths:
        if not path.is_absolute():
            raise ValueError("forwarded CLI path must be absolute")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("forwarded CLI path is not readable") from exc
        if any(resolved == root or resolved.is_relative_to(root) for root in resolved_roots):
            raise ValueError("forwarded CLI path is hidden by the selected sandbox")


def _preflight_command(argv: list[str]) -> bool:
    """Probe a native sandbox without inheriting credentials or emitting output."""

    try:
        completed = subprocess.run(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env={"PATH": DEFAULT_CHILD_PATH},
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _preflight_wrapper(sandbox: Sandbox, *, executable: str) -> bool:
    """Exercise the same generated launcher and native policy used by tasks."""

    try:
        with TemporaryDirectory(prefix="taskchamber-sandbox-preflight-") as temp_dir:
            root = Path(temp_dir)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            wrapper = sandbox.prepare_cli_launcher(
                IsolatedWorkspace(root=workspace_root, allowed_paths=(workspace_root,)),
                executable=executable,
                config_dir=root / "config",
                launcher_dir=root / "launcher",
            )
            if not _preflight_command([str(wrapper), "--taskchamber-preflight"]):
                return False
            marker = root / "config" / CLI_LAUNCH_OBSERVATION_FILE
            return marker.is_file() and marker.stat().st_mode & 0o077 == 0
    except (OSError, ValueError):
        return False

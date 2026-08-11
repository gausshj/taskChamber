import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from claude_agent_sdk import ResultMessage

from taskchamber.core.contracts import (
    ExecutionPolicy,
    TaskKind,
    TaskRequest,
    TaskStatus,
)
from taskchamber.isolation import (
    BubblewrapSandbox,
    IsolatedWorkspace,
    MacOSSandboxExecSandbox,
    NoSandbox,
    Sandbox,
    select_sandbox,
)
from taskchamber.runtimes.claude import ClaudeAgentSdkRuntime
from taskchamber.runtimes.claude.cli import resolve_claude_cli


def _policy(workspace_root: Path) -> ExecutionPolicy:
    return ExecutionPolicy(
        workspace_root=workspace_root,
        allowed_paths=(workspace_root,),
        system_prompt="Read only.",
        allowed_tools=("Read", "Glob", "Grep"),
        disallowed_tools=("Bash", "Edit", "Write"),
        max_turns=1,
        max_budget_usd=0.5,
        timeout_seconds=1.0,
        max_output_chars=1_000,
        max_file_bytes=1_000_000,
    )


def _request() -> TaskRequest:
    return TaskRequest(
        run_id="run-sb",
        kind=TaskKind.RESEARCH,
        prompt="Inspect the workspace.",
        provider="glm",
        max_turns=1,
    )


def _helper_source(wrapper: Path) -> str:
    return (wrapper.parent / "agent-cli-exec.py").read_text(encoding="utf-8")


def _complete_mock_preflight(argv: list[str]) -> bool:
    wrapper = Path(argv[0])
    marker = wrapper.parent.parent / "config" / "cli-main.started"
    marker.write_text("", encoding="utf-8")
    marker.chmod(0o600)
    return True


def test_select_sandbox_maps_explicit_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox.shutil.which",
        lambda name: str(name) if Path(name).is_absolute() else f"/usr/bin/{name}",
    )
    assert isinstance(select_sandbox("none"), NoSandbox)

    monkeypatch.setattr("taskchamber.isolation.sandbox.sys.platform", "linux")
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox._preflight_command", _complete_mock_preflight
    )
    assert isinstance(select_sandbox("bwrap"), BubblewrapSandbox)
    with pytest.raises(ValueError, match="unavailable"):
        select_sandbox("sandbox-exec")

    monkeypatch.setattr("taskchamber.isolation.sandbox.sys.platform", "darwin")
    assert isinstance(select_sandbox("sandbox-exec"), MacOSSandboxExecSandbox)
    with pytest.raises(ValueError, match="unavailable"):
        select_sandbox("bwrap")


def test_select_sandbox_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        select_sandbox("firejail")


def test_sandbox_availability_reflects_the_tool_being_installed() -> None:
    assert BubblewrapSandbox(bwrap="/does/not/exist/bwrap").available is False
    assert MacOSSandboxExecSandbox(sandbox_exec="/does/not/exist/se").available is False
    # NoSandbox is always usable; it is the hermetic fallback.
    assert NoSandbox().available is True


@pytest.mark.parametrize("mode", ("bwrap", "sandbox-exec", "required"))
def test_explicit_sandbox_modes_fail_closed_when_unavailable(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("taskchamber.isolation.sandbox.shutil.which", lambda _name: None)

    with pytest.raises(ValueError, match="(unavailable|unsupported)"):
        select_sandbox(mode)


def test_auto_select_falls_back_to_no_sandbox_without_a_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("taskchamber.isolation.sandbox.shutil.which", lambda _name: None)
    sandbox = select_sandbox("auto")
    assert isinstance(sandbox, NoSandbox)


def test_auto_select_uses_native_tool_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox._preflight_command", _complete_mock_preflight
    )

    sandbox = select_sandbox("auto")

    if sys.platform.startswith("linux"):
        assert isinstance(sandbox, BubblewrapSandbox)
    elif sys.platform == "darwin":
        assert isinstance(sandbox, MacOSSandboxExecSandbox)
    else:
        assert isinstance(sandbox, NoSandbox)


def test_bwrap_preflight_exercises_the_generated_operational_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("taskchamber.isolation.sandbox.sys.platform", "linux")
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox.shutil.which",
        lambda name: str(name) if Path(name).is_absolute() else f"/usr/bin/{name}",
    )

    def observe(argv: list[str]) -> bool:
        wrapper = Path(argv[0])
        captured["argv"] = argv
        captured["helper"] = _helper_source(wrapper)
        return _complete_mock_preflight(argv)

    monkeypatch.setattr("taskchamber.isolation.sandbox._preflight_command", observe)

    assert BubblewrapSandbox(bwrap="bwrap").preflight() is True
    assert captured["argv"][-1] == "--taskchamber-preflight"  # type: ignore[index]
    helper = str(captured["helper"])
    assert "/usr/bin/bwrap" in helper
    assert "--tmpfs" in helper
    assert "--unshare-pid" in helper
    assert "--chdir" in helper


def test_macos_preflight_exercises_the_generated_operational_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("taskchamber.isolation.sandbox.sys.platform", "darwin")
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox.shutil.which",
        lambda name: str(name) if Path(name).is_absolute() else f"/usr/bin/{name}",
    )

    def observe(argv: list[str]) -> bool:
        wrapper = Path(argv[0])
        captured["argv"] = argv
        captured["helper"] = _helper_source(wrapper)
        captured["profile"] = (wrapper.parent / "sandbox-exec.sb").read_text(encoding="utf-8")
        return _complete_mock_preflight(argv)

    monkeypatch.setattr("taskchamber.isolation.sandbox._preflight_command", observe)

    assert MacOSSandboxExecSandbox(sandbox_exec="sandbox-exec").preflight() is True
    assert captured["argv"][-1] == "--taskchamber-preflight"  # type: ignore[index]
    assert "/usr/bin/sandbox-exec" in str(captured["helper"])
    profile = str(captured["profile"])
    assert "(deny process-info*)" in profile
    assert "(deny file-write*)" in profile


def test_no_sandbox_stages_a_filtered_workspace_without_a_wrapper(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("stay", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret", encoding="utf-8")

    with NoSandbox().isolate(_policy(tmp_path)) as workspace:
        assert workspace.root != tmp_path
        assert workspace.allowed_paths == (workspace.root,)
        assert (workspace.root / "note.txt").read_text(encoding="utf-8") == "stay"
        assert not (workspace.root / ".env.local").exists()


def test_staging_isolates_a_filtered_copy_and_drops_secrets(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("public", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=x", encoding="utf-8")
    sandbox = BubblewrapSandbox(bwrap="/fake/bwrap")  # isolate() does not invoke bwrap

    with sandbox.isolate(_policy(tmp_path)) as workspace:
        assert workspace.root != tmp_path
        assert (workspace.root / "doc.md").read_text(encoding="utf-8") == "public"
        assert not (workspace.root / ".env").exists()


def test_bwrap_wrapper_binds_staged_workspace_and_hides_home(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    launcher_dir = tmp_path / "launcher"
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(tmp_path)) as workspace:
        wrapper = sandbox.prepare_wrapper(
            workspace,
            executable="/opt/claude/bin/claude",
            config_dir=config_dir,
            launcher_dir=launcher_dir,
        )

    script = wrapper.read_text(encoding="utf-8")
    helper = _helper_source(wrapper)
    assert wrapper.stat().st_mode & 0o111  # executable
    assert "/usr/bin/bwrap" in helper
    assert "--ro-bind" in helper
    assert str(workspace.root) in helper
    assert str(config_dir) in helper
    assert wrapper.parent == launcher_dir
    assert str(launcher_dir) not in helper
    assert "--setenv" in helper
    assert "/opt/claude/bin/claude" in helper
    assert "--unshare-pid" in helper
    assert "--unshare-ipc" in helper
    assert "--unshare-uts" in helper
    assert "--new-session" in helper
    assert "--cap-drop" in helper
    assert "--chdir" in helper
    assert '"$@"' in script
    assert "--unshare-net" not in helper  # provider call must work


def test_bwrap_wrapper_rebinds_only_a_home_installed_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    resolved_cli = home / ".local" / "share" / "claude" / "versions" / "2.1.206"
    resolved_cli.parent.mkdir(parents=True)
    resolved_cli.write_text("binary placeholder", encoding="utf-8")
    cli_link = home / ".local" / "bin" / "claude"
    cli_link.parent.mkdir(parents=True)
    cli_link.symlink_to(resolved_cli)
    monkeypatch.setenv("HOME", str(home))

    source = tmp_path / "source"
    source.mkdir()
    config_dir = tmp_path / "config"
    launcher_dir = tmp_path / "launcher"
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(source)) as workspace:
        wrapper = sandbox.prepare_wrapper(
            workspace,
            executable=str(cli_link),
            config_dir=config_dir,
            launcher_dir=launcher_dir,
        )

    helper = _helper_source(wrapper)
    assert str(home) in helper
    assert str(resolved_cli.parent) in helper
    assert str(resolved_cli) in helper
    assert str(cli_link) not in helper


def _prepare_masked_root_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode_dir: int = 0o755,
    mode_file: int = 0o644,
) -> tuple[Path, Path, Path]:
    """Stage a canonical CLI plus a sibling below a fake masked root."""

    masked_root = tmp_path / "masked-tmp"
    tool_dir = masked_root / "tools"
    tool_dir.mkdir(parents=True)
    tool_dir.chmod(mode_dir)
    cli = tool_dir / "claude"
    cli.write_text("binary placeholder", encoding="utf-8")
    cli.chmod(mode_file)
    sibling = tool_dir / "sibling-canary"
    sibling.write_text("must-stay-hidden", encoding="utf-8")
    sibling.chmod(0o644)
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox._host_home_directories",
        lambda: (masked_root,),
    )
    return masked_root, cli, sibling


def test_bwrap_wrapper_rebinds_only_the_canonical_executable_below_a_masked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masked_root, cli, sibling = _prepare_masked_root_rebind(tmp_path, monkeypatch)
    cli_link = tmp_path / "bin" / "claude"
    cli_link.parent.mkdir()
    cli_link.symlink_to(cli)

    source = tmp_path / "source"
    source.mkdir()
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(source)) as workspace:
        wrapper = sandbox.prepare_wrapper(
            workspace,
            executable=str(cli_link),
            config_dir=tmp_path / "config",
            launcher_dir=tmp_path / "launcher",
        )

    helper = _helper_source(wrapper)
    assert str(cli) in helper
    assert str(cli_link) not in helper
    assert str(sibling) not in helper
    assert f"--dir', '{masked_root / 'tools'}" in helper


@pytest.mark.parametrize(
    ("mode_dir", "mode_file"),
    [(0o777, 0o644), (0o755, 0o664), (0o755, 0o666)],
)
def test_bwrap_wrapper_rejects_a_writable_rebind_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode_dir: int,
    mode_file: int,
) -> None:
    _, cli, _ = _prepare_masked_root_rebind(
        tmp_path, monkeypatch, mode_dir=mode_dir, mode_file=mode_file
    )
    source = tmp_path / "source"
    source.mkdir()
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(source)) as workspace:
        with pytest.raises(ValueError, match="writable by group or others"):
            sandbox.prepare_wrapper(
                workspace,
                executable=str(cli),
                config_dir=tmp_path / "config",
                launcher_dir=tmp_path / "launcher",
            )


def test_bwrap_wrapper_rejects_a_rebind_owned_by_another_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cli, _ = _prepare_masked_root_rebind(tmp_path, monkeypatch)
    # Simulate foreign ownership without chown: from the process's perspective
    # the euid no longer matches the masked root owner, so the anchor check
    # fails before any component is trusted.
    monkeypatch.setattr("taskchamber.isolation.sandbox.os.geteuid", lambda: 1_000_001)
    source = tmp_path / "source"
    source.mkdir()
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(source)) as workspace:
        with pytest.raises(ValueError, match="owned by another user"):
            sandbox.prepare_wrapper(
                workspace,
                executable=str(cli),
                config_dir=tmp_path / "config",
                launcher_dir=tmp_path / "launcher",
            )


def test_bwrap_wrapper_rejects_a_non_sticky_world_writable_masked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A misconfigured home-style root (0o777 without the sticky bit) lets any
    # local user replace the first component below it after validation, so the
    # rebind must fail closed even when every component below is clean.
    masked_root, cli, _ = _prepare_masked_root_rebind(
        tmp_path, monkeypatch, mode_dir=0o755, mode_file=0o700
    )
    masked_root.chmod(0o777)
    source = tmp_path / "source"
    source.mkdir()
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(source)) as workspace:
        with pytest.raises(ValueError, match="writable without a sticky bit"):
            sandbox.prepare_wrapper(
                workspace,
                executable=str(cli),
                config_dir=tmp_path / "config",
                launcher_dir=tmp_path / "launcher",
            )


def test_bwrap_wrapper_accepts_a_sticky_world_writable_masked_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The sticky bit is what makes a root-owned /tmp a safe anchor: other users
    # cannot rename or replace entries they do not own.
    masked_root, cli, _ = _prepare_masked_root_rebind(
        tmp_path, monkeypatch, mode_dir=0o755, mode_file=0o700
    )
    masked_root.chmod(0o1777)
    source = tmp_path / "source"
    source.mkdir()
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")

    with sandbox.isolate(_policy(source)) as workspace:
        wrapper = sandbox.prepare_wrapper(
            workspace,
            executable=str(cli),
            config_dir=tmp_path / "config",
            launcher_dir=tmp_path / "launcher",
        )

    assert str(cli) in _helper_source(wrapper)


def _stat_result(mode: int, uid: int) -> os.stat_result:
    return os.stat_result((mode, 0, 0, 0, uid, 0, 0, 0, 0, 0))


def test_check_rebind_metadata_accepts_and_rejects_expected_components() -> None:
    import stat as stat_module

    from taskchamber.isolation.sandbox import _check_rebind_metadata

    euid = os.geteuid()
    # Acceptable: euid-owned non-writable file and root-owned system directory.
    _check_rebind_metadata(_stat_result(stat_module.S_IFREG | 0o700, euid), euid=euid)
    _check_rebind_metadata(_stat_result(stat_module.S_IFDIR | 0o755, 0), euid=euid)

    symlinked = _stat_result(stat_module.S_IFLNK | 0o777, euid)
    with pytest.raises(ValueError, match="changed after canonicalization"):
        _check_rebind_metadata(symlinked, euid=euid)
    foreign_owned = _stat_result(stat_module.S_IFREG | 0o700, 1_000_001)
    with pytest.raises(ValueError, match="owned by another user"):
        _check_rebind_metadata(foreign_owned, euid=euid)
    world_writable = _stat_result(stat_module.S_IFREG | 0o666, euid)
    with pytest.raises(ValueError, match="writable by group or others"):
        _check_rebind_metadata(world_writable, euid=euid)


def test_rebindable_executable_rejects_a_symlink_component_after_canonicalization(
    tmp_path: Path,
) -> None:
    # Simulates a swap between resolve() and launch: a component that was a real
    # directory at canonicalization time is a symlink when the launch is built.
    from taskchamber.isolation.sandbox import _verify_rebindable_executable

    masked_root = tmp_path / "masked-tmp"
    masked_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "claude").write_text("binary placeholder", encoding="utf-8")
    (masked_root / "tools").symlink_to(outside)

    with pytest.raises(ValueError, match="changed after canonicalization"):
        _verify_rebindable_executable(masked_root / "tools" / "claude", masked_root)


def test_rebindable_executable_rejects_a_missing_component(tmp_path: Path) -> None:
    from taskchamber.isolation.sandbox import _verify_rebindable_executable

    masked_root = tmp_path / "masked-tmp"
    masked_root.mkdir()

    with pytest.raises(ValueError, match="unreadable below a masked root"):
        _verify_rebindable_executable(masked_root / "gone" / "claude", masked_root)


def test_rebindable_executable_rejects_an_unreadable_masked_root(tmp_path: Path) -> None:
    from taskchamber.isolation.sandbox import _verify_rebindable_executable

    missing_root = tmp_path / "gone"

    with pytest.raises(ValueError, match="the masked root is unreadable"):
        _verify_rebindable_executable(missing_root / "tools" / "claude", missing_root)


def test_rebindable_executable_rejects_a_symlink_masked_root(tmp_path: Path) -> None:
    from taskchamber.isolation.sandbox import _verify_rebindable_executable

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="the masked root changed after canonicalization"):
        _verify_rebindable_executable(link / "tools" / "claude", link)


def test_rebindable_executable_rejects_a_regular_file_masked_root(tmp_path: Path) -> None:
    # A home-style masked root misconfigured as the executable file itself must
    # fail closed immediately instead of walking parents forever.
    from taskchamber.isolation.sandbox import _verify_rebindable_executable

    not_a_directory = tmp_path / "claude"
    not_a_directory.write_text("binary placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="the masked root is not a directory"):
        _verify_rebindable_executable(not_a_directory, not_a_directory)


def test_rebindable_executable_bounds_the_walk_to_the_masked_root(tmp_path: Path) -> None:
    from taskchamber.isolation.sandbox import _verify_rebindable_executable

    masked_root = tmp_path / "masked-tmp"
    masked_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    cli = outside / "claude"
    cli.write_text("binary placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="not below the masked root"):
        _verify_rebindable_executable(cli, masked_root)


def test_native_tool_basenames_are_canonicalized_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = {
        "bwrap": "/trusted/bin/bwrap",
        "sandbox-exec": "/trusted/bin/sandbox-exec",
    }
    monkeypatch.setattr(
        "taskchamber.isolation.sandbox.shutil.which",
        lambda name: resolved.get(str(name), str(name) if Path(name).is_absolute() else None),
    )
    workspace = IsolatedWorkspace(root=tmp_path / "workspace", allowed_paths=())
    workspace.root.mkdir()

    bwrap_wrapper = BubblewrapSandbox(bwrap="bwrap").prepare_wrapper(
        workspace,
        executable="/opt/claude",
        config_dir=tmp_path / "bwrap-config",
        launcher_dir=tmp_path / "bwrap-launcher",
    )
    mac_wrapper = MacOSSandboxExecSandbox(sandbox_exec="sandbox-exec").prepare_wrapper(
        workspace,
        executable="/opt/claude",
        config_dir=tmp_path / "mac-config",
        launcher_dir=tmp_path / "mac-launcher",
    )

    assert "/trusted/bin/bwrap" in _helper_source(bwrap_wrapper)
    assert "/trusted/bin/sandbox-exec" in _helper_source(mac_wrapper)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bwrap is Linux-only")
def test_bwrap_wrapper_rebinds_an_executable_hidden_by_tmpfs(tmp_path: Path) -> None:
    cli = tmp_path / "bin" / "claude"
    cli.parent.mkdir()
    cli.write_text("binary placeholder", encoding="utf-8")
    config_dir = tmp_path / "config"
    launcher_dir = tmp_path / "launcher"
    workspace = IsolatedWorkspace(root=tmp_path / "workspace", allowed_paths=())
    workspace.root.mkdir()

    wrapper = BubblewrapSandbox(bwrap="/usr/bin/bwrap").prepare_wrapper(
        workspace,
        executable=str(cli),
        config_dir=config_dir,
        launcher_dir=launcher_dir,
    )

    helper = _helper_source(wrapper)
    assert str(cli.parent) in helper
    assert str(cli) in helper


def test_macos_profile_literal_escapes_quotes_and_rejects_line_breaks() -> None:
    escaped = MacOSSandboxExecSandbox._profile_literal(Path('/tmp/a"b'))

    assert escaped == '/tmp/a\\"b'
    with pytest.raises(ValueError, match="line breaks"):
        MacOSSandboxExecSandbox._profile_literal(Path("/tmp/a\nb"))


def test_macos_wrapper_confines_home_reads_and_writes(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    launcher_dir = tmp_path / "launcher"
    sandbox = MacOSSandboxExecSandbox(sandbox_exec="/usr/bin/sandbox-exec")

    with sandbox.isolate(_policy(tmp_path)) as workspace:
        wrapper = sandbox.prepare_wrapper(
            workspace,
            executable="/opt/claude/bin/claude",
            config_dir=config_dir,
            launcher_dir=launcher_dir,
        )

    profile = (launcher_dir / "sandbox-exec.sb").read_text(encoding="utf-8")
    assert "(version 1)" in profile
    assert "(allow default)" in profile
    assert "(deny process-info*)" in profile
    assert "(allow process-info* (target self))" in profile
    assert f'(deny file-read* (subpath "{Path.home().resolve()}"))' in profile
    assert '(allow file-read* (literal "/opt/claude/bin/claude"))' in profile
    assert "(deny file-write*)" in profile
    assert f'(allow file-write* (subpath "{config_dir.resolve()}"))' in profile
    assert str(launcher_dir.resolve()) not in profile
    wrapper_script = wrapper.read_text(encoding="utf-8")
    helper = _helper_source(wrapper)
    assert "/usr/bin/sandbox-exec" in helper
    assert "sandbox-exec.sb" in helper
    assert "/opt/claude/bin/claude" in helper
    assert '"$@"' in wrapper_script


def test_macos_profile_hides_environment_and_account_homes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pwd

    environment_home = tmp_path / "selected-home"
    environment_home.mkdir()
    monkeypatch.setenv("HOME", str(environment_home))
    workspace = IsolatedWorkspace(root=tmp_path / "workspace", allowed_paths=())
    workspace.root.mkdir()
    sandbox = MacOSSandboxExecSandbox(sandbox_exec="/usr/bin/sandbox-exec")

    profile = sandbox._profile(
        workspace,
        executable="/opt/claude",
        config_dir=tmp_path / "config",
    )

    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert f'(deny file-read* (subpath "{environment_home.resolve()}"))' in profile
    assert f'(deny file-read* (subpath "{account_home}"))' in profile


def test_clean_environment_wrapper_drops_parent_secrets_and_cleans_version_probe(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    launcher_dir = tmp_path / "launcher"
    probe = tmp_path / "probe"
    probe.write_text("#!/bin/sh\n/usr/bin/env\n", encoding="utf-8")
    probe.chmod(0o700)
    wrapper = NoSandbox().prepare_wrapper(
        IsolatedWorkspace(root=workspace, allowed_paths=(workspace,)),
        executable=str(probe),
        config_dir=config_dir,
        launcher_dir=launcher_dir,
        environment_keys=("ANTHROPIC_AUTH_TOKEN", "HTTPS_PROXY"),
    )
    inherited = {
        "ANTHROPIC_AUTH_TOKEN": "selected-token",
        "HOST_SECRET_CANARY": "must-not-cross",
        "HTTPS_PROXY": "https://proxy.example",
    }

    main = subprocess.run(
        [str(wrapper), "probe"],
        capture_output=True,
        check=True,
        env=inherited,
        text=True,
    )
    version = subprocess.run(
        [str(wrapper), "-v"],
        capture_output=True,
        check=True,
        env=inherited,
        text=True,
    )

    assert "ANTHROPIC_AUTH_TOKEN=selected-token" in main.stdout
    assert "HTTPS_PROXY=https://proxy.example" in main.stdout
    assert "HOST_SECRET_CANARY" not in main.stdout
    assert f"HOME={config_dir}" in main.stdout
    assert f"PWD={workspace}" in main.stdout
    assert "PATH=/usr/bin:/bin" in main.stdout
    assert "ANTHROPIC_AUTH_TOKEN" not in version.stdout
    assert "HTTPS_PROXY" not in version.stdout
    assert "selected-token" not in wrapper.read_text(encoding="utf-8")
    assert "selected-token" not in _helper_source(wrapper)
    assert "must-not-cross" not in _helper_source(wrapper)
    assert "/usr/bin/env -i" not in wrapper.read_text(encoding="utf-8")
    assert "/usr/bin/env -i" not in _helper_source(wrapper)
    assert wrapper.parent == launcher_dir
    assert wrapper.stat().st_mode & 0o222 == 0


def test_clean_launcher_fails_closed_on_an_unsupported_windows_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("taskchamber.isolation.sandbox.POSIX_LAUNCHER_SUPPORTED", False)
    isolated_workspace = IsolatedWorkspace(root=workspace, allowed_paths=(workspace,))
    config_dir = tmp_path / "config"
    launcher_dir = tmp_path / "launcher"
    sandbox = NoSandbox()

    with pytest.raises(ValueError, match="POSIX"):
        sandbox.prepare_cli_launcher(
            isolated_workspace,
            executable="C:/claude.exe",
            config_dir=config_dir,
            launcher_dir=launcher_dir,
        )


def test_native_sandboxes_reject_forwarded_paths_under_hidden_home() -> None:
    hidden = Path.home().resolve()
    bubblewrap = BubblewrapSandbox(bwrap="/usr/bin/bwrap")
    sandbox_exec = MacOSSandboxExecSandbox(sandbox_exec="/usr/bin/sandbox-exec")

    NoSandbox().validate_readable_paths((hidden,))
    with pytest.raises(ValueError, match="hidden"):
        bubblewrap.validate_readable_paths((hidden,))
    with pytest.raises(ValueError, match="hidden"):
        sandbox_exec.validate_readable_paths((hidden,))


def test_bwrap_rejects_forwarded_paths_under_the_host_tmp_root(
    tmp_path: Path,
) -> None:
    # SonarCloud python:S5443 flags the /tmp masked root used by
    # BubblewrapSandbox.validate_readable_paths. That root is only a rejection
    # boundary: it is never created, read, or forwarded. The shared _reject_masked_paths
    # helper is exercised directly with a hermetic root so the assertions never
    # depend on the host owning a writable real /tmp.
    from taskchamber.isolation.sandbox import _reject_masked_paths

    masked_root = tmp_path / "masked-tmp"
    masked_root.mkdir()

    # An absolute, readable path directly below the masked root is rejected.
    direct = masked_root / "token"
    direct.write_text("canary", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden"):
        _reject_masked_paths((direct,), (masked_root,))

    # A symlink whose canonical target resolves below the masked root is also
    # rejected: resolve(strict=True) canonicalizes before the root check, so a
    # link cannot smuggle a masked target past validation.
    link_parent = tmp_path / "links"
    link_parent.mkdir()
    link = link_parent / "points-into-tmp"
    link.symlink_to(direct)
    with pytest.raises(ValueError, match="hidden"):
        _reject_masked_paths((link,), (masked_root,))

    # The masked root itself may be a symlink to a different canonical directory
    # (macOS exposes /tmp -> /private/tmp). The candidate resolves to the real
    # directory while the configured root is the symlink literal, so the roots
    # must be canonicalized too or the candidate slips past the unresolved
    # literal. This reproduces the regression that motivated the hardening.
    real_dir = tmp_path / "real-tmp"
    real_dir.mkdir()
    root_symlink = tmp_path / "symlinked-tmp"
    root_symlink.symlink_to(real_dir)
    target = real_dir / "ca.pem"
    target.write_text("canary", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden"):
        _reject_masked_paths((target,), (root_symlink,))

    # A sibling path outside the masked root is accepted: only masked targets
    # are rejected, so legitimate forwarded paths keep working.
    outside = tmp_path / "outside"
    outside.write_text("ok", encoding="utf-8")
    _reject_masked_paths((outside,), (masked_root,))  # no exception

    # Missing paths fail closed and the message leaks neither the path nor any
    # host detail.
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="readable") as exc_info:
        _reject_masked_paths((missing,), (masked_root,))
    assert str(missing) not in str(exc_info.value)

    # Non-absolute paths fail closed before any resolution attempt.
    with pytest.raises(ValueError, match="absolute"):
        _reject_masked_paths((Path("relative/secret"),), (masked_root,))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bwrap is Linux-only")
def test_bwrap_validate_readable_paths_rejects_a_real_host_tmp_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the production entry point BubblewrapSandbox.validate_readable_paths
    # against a real path below the host /tmp on Linux (where /tmp is normally a
    # real directory). A forwarded CA path below the masked root must fail
    # closed through the public method, not only through the private helper.
    sandbox = BubblewrapSandbox(bwrap="/usr/bin/bwrap")
    host_tmp = Path("/tmp")
    resolved_host_tmp = host_tmp.resolve()
    link_base = Path.cwd().resolve()
    if link_base.is_relative_to(resolved_host_tmp):
        link_base = Path.home().resolve(strict=True)
    assert not link_base.is_relative_to(resolved_host_tmp)
    monkeypatch.setattr("taskchamber.isolation.sandbox._host_home_directories", lambda: ())

    # Use securely randomized directories for the publicly writable host /tmp
    # probe and keep the link itself outside /tmp so the symlink assertion
    # depends on its canonical target rather than its lexical location.
    with (
        TemporaryDirectory(prefix="taskchamber-s5443-", dir=host_tmp) as probe_directory,
        TemporaryDirectory(prefix=".taskchamber-s5443-link-", dir=link_base) as link_directory,
    ):
        probe = Path(probe_directory) / "ca.pem"
        probe.write_text("canary", encoding="utf-8")
        with pytest.raises(ValueError, match="hidden"):
            sandbox.validate_readable_paths((probe,))

        link = Path(link_directory) / "ca-link"
        link.symlink_to(probe)
        with pytest.raises(ValueError, match="hidden"):
            sandbox.validate_readable_paths((link,))


def test_bwrap_revalidation_rejects_a_path_retargeted_into_the_masked_root(
    tmp_path: Path,
) -> None:
    # Path validation is a point-in-time availability check, not the OS security
    # boundary. Revalidation must observe a changed symlink target; the native
    # bwrap probe separately proves that /tmp remains masked if the target changes
    # after validation but before the sandboxed command uses it.
    from taskchamber.isolation.sandbox import _reject_masked_paths

    masked_root = tmp_path / "masked-tmp"
    masked_root.mkdir()
    safe_target = tmp_path / "legit-ca.pem"
    safe_target.write_text("safe", encoding="utf-8")
    attacker_target = masked_root / "evil.pem"
    attacker_target.write_text("evil", encoding="utf-8")

    # A link that currently points at the safe target is accepted.
    link = tmp_path / "ca-link"
    link.symlink_to(safe_target)
    _reject_masked_paths((link,), (masked_root,))  # accepted

    # If the same link is repointed at the masked root, a subsequent validation
    # canonicalizes the new target and rejects it.
    link.unlink()
    link.symlink_to(attacker_target)
    with pytest.raises(ValueError, match="hidden"):
        _reject_masked_paths((link,), (masked_root,))


def test_version_probe_cannot_create_the_main_launch_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = tmp_path / "config"
    executable = tmp_path / "claude"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    wrapper = NoSandbox().prepare_wrapper(
        IsolatedWorkspace(root=workspace, allowed_paths=(workspace,)),
        executable=str(executable),
        config_dir=config_dir,
        launcher_dir=tmp_path / "launcher",
    )
    marker = config_dir / "cli-main.started"

    subprocess.run([str(wrapper), "-v"], check=True)
    assert not marker.exists()

    subprocess.run([str(wrapper), "--output-format", "stream-json"], check=True)
    assert marker.is_file()
    assert marker.stat().st_mode & 0o077 == 0


def test_legacy_base_prepare_wrapper_call_preserves_the_executable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executable = tmp_path / "claude"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    config_dir = tmp_path / "config"

    wrapper = NoSandbox().prepare_wrapper(
        IsolatedWorkspace(root=workspace, allowed_paths=(workspace,)),
        executable=str(executable),
        config_dir=config_dir,
    )

    assert wrapper == executable
    assert not config_dir.exists()


def test_legacy_native_prepare_wrapper_preserves_inherited_environment_contract(
    tmp_path: Path,
) -> None:
    workspace = IsolatedWorkspace(root=tmp_path / "workspace", allowed_paths=())
    workspace.root.mkdir()
    config_dir = tmp_path / "config"

    wrapper = BubblewrapSandbox(bwrap="/usr/bin/bwrap").prepare_wrapper(
        workspace,
        executable="/opt/claude",
        config_dir=config_dir,
    )

    script = wrapper.read_text(encoding="utf-8")
    assert wrapper == config_dir / "sandbox-wrapper.sh"
    assert "exec /usr/bin/bwrap" in script
    assert "agent-cli-exec.py" not in script
    assert '"$@"' in script


def test_legacy_builtin_subclass_uses_the_compatibility_outer_launcher(
    tmp_path: Path,
) -> None:
    class LegacyBubblewrap(BubblewrapSandbox):
        def prepare_wrapper(
            self,
            workspace: IsolatedWorkspace,
            *,
            executable: str,
            config_dir: Path,
        ) -> Path:
            del workspace, config_dir
            return Path(executable)

    workspace = IsolatedWorkspace(root=tmp_path / "workspace", allowed_paths=())
    workspace.root.mkdir()
    executable = tmp_path / "claude"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    config_dir = tmp_path / "config"
    sandbox = LegacyBubblewrap(bwrap="/usr/bin/bwrap")

    assert sandbox.uses_secure_cli_launcher is False
    wrapper = sandbox.prepare_cli_launcher(
        workspace,
        executable=str(executable),
        config_dir=config_dir,
        launcher_dir=tmp_path / "launcher",
    )
    subprocess.run([str(wrapper), "main"], check=True)

    assert (config_dir / "cli-main.started").is_file()


def test_launcher_directory_cannot_be_nested_in_writable_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executable = tmp_path / "claude"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    config_dir = tmp_path / "config"
    isolated_workspace = IsolatedWorkspace(root=workspace, allowed_paths=(workspace,))
    nested_launcher = config_dir / "launcher"
    launcher_root = tmp_path / "launcher-root"
    launcher_root_config = launcher_root / "config"
    executable_path = str(executable)
    sandbox = NoSandbox()

    with pytest.raises(ValueError, match="outside writable CLI config"):
        sandbox.prepare_wrapper(
            isolated_workspace,
            executable=executable_path,
            config_dir=config_dir,
            launcher_dir=nested_launcher,
        )

    with pytest.raises(ValueError, match="outside writable CLI config"):
        sandbox.prepare_wrapper(
            isolated_workspace,
            executable=executable_path,
            config_dir=launcher_root_config,
            launcher_dir=launcher_root,
        )


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("TASKCHAMBER_RUN_NATIVE_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled native macOS sandbox probe",
)
def test_macos_native_profile_starts_bundled_cli_and_enforces_file_boundary(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve()
    if not source.is_relative_to(Path.home().resolve()):
        pytest.skip("the read-denial canary must be inside the host home")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = IsolatedWorkspace(root=workspace_root, allowed_paths=(workspace_root,))
    sandbox = MacOSSandboxExecSandbox()
    assert sandbox.available
    assert sandbox.preflight()

    cli_config = tmp_path / "cli-config"
    cli_launcher = tmp_path / "cli-launcher"
    cli_wrapper = sandbox.prepare_wrapper(
        workspace,
        executable=str(resolve_claude_cli().path),
        config_dir=cli_config,
        launcher_dir=cli_launcher,
    )
    version = subprocess.run(
        [str(cli_wrapper), "-v"],
        capture_output=True,
        check=True,
        text=True,
        timeout=20,
    )
    assert "Claude Code" in version.stdout

    probe = tmp_path / "boundary-probe"
    process_canary = "TASKCHAMBER_PARENT_ENV_MUST_STAY_HIDDEN"
    outside_write = tmp_path / "outside-write"
    allowed_read = workspace_root / "allowed-read"
    allowed_read.write_text("allowed", encoding="utf-8")
    probe.write_text(
        "#!/bin/sh\n"
        '/bin/cat "$1" >/dev/null || exit 40\n'
        'if /bin/cat "$2" >/dev/null 2>&1; then exit 41; fi\n'
        'if /usr/bin/printf bad > "$3" 2>/dev/null; then exit 42; fi\n'
        'if /bin/ps -p "$4" -o command= 2>/dev/null | /usr/bin/grep -q "$5"; then exit 43; fi\n'
        '/usr/bin/printf allowed > "$HOME/allowed-write"\n',
        encoding="utf-8",
    )
    probe.chmod(0o700)
    probe_config = tmp_path / "probe-config"
    probe_launcher = tmp_path / "probe-launcher"
    probe_wrapper = sandbox.prepare_wrapper(
        workspace,
        executable=str(probe),
        config_dir=probe_config,
        launcher_dir=probe_launcher,
    )

    host_process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import time; time.sleep(20)",
            process_canary,
        ],
    )
    try:
        visible = subprocess.run(
            ["/bin/ps", "-p", str(host_process.pid), "-o", "command="],
            capture_output=True,
            check=True,
            text=True,
        )
        assert process_canary in visible.stdout
        subprocess.run(
            [
                str(probe_wrapper),
                str(allowed_read),
                str(source),
                str(outside_write),
                str(host_process.pid),
                process_canary,
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )
    finally:
        host_process.terminate()
        host_process.wait(timeout=5)

    assert not outside_write.exists()
    assert (probe_config / "allowed-write").read_text(encoding="utf-8") == "allowed"


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or os.environ.get("TASKCHAMBER_RUN_NATIVE_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled native Linux bubblewrap probe",
)
def test_bwrap_native_boundary_hides_host_processes_and_denies_workspace_writes(
    tmp_path: Path,
) -> None:
    with TemporaryDirectory(prefix="taskchamber-native-", dir="/tmp") as host_tmp_directory:
        host_tmp_canary = Path(host_tmp_directory) / "host-canary"
        host_tmp_canary.write_text("must-stay-hidden", encoding="utf-8")

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        denied_write = workspace_root / "denied-write"
        workspace = IsolatedWorkspace(root=workspace_root, allowed_paths=(workspace_root,))
        sandbox = BubblewrapSandbox()
        assert sandbox.preflight()

        # Validate an outside target, then retarget the same workspace link to a
        # host /tmp canary before launch. The application-level check is inherently
        # point-in-time; bwrap's private /tmp must remain the enforcement boundary.
        forwarded_link = workspace_root / "forwarded-ca"
        forwarded_link.symlink_to(Path("/bin/sh").resolve(strict=True))
        sandbox.validate_readable_paths((forwarded_link,))
        forwarded_link.unlink()
        forwarded_link.symlink_to(host_tmp_canary)

        probe = tmp_path / "boundary-probe"
        probe.write_text(
            "#!/bin/sh\n"
            'if [ -e "/proc/$1" ]; then exit 40; fi\n'
            'if /usr/bin/printf bad > "$2" 2>/dev/null; then exit 41; fi\n'
            'if /bin/cat "$3" >/dev/null 2>&1; then exit 42; fi\n'
            '/usr/bin/printf allowed > "$HOME/allowed-write"\n',
            encoding="utf-8",
        )
        probe.chmod(0o700)
        config_dir = tmp_path / "config"
        wrapper = sandbox.prepare_wrapper(
            workspace,
            executable=str(probe),
            config_dir=config_dir,
            launcher_dir=tmp_path / "launcher",
        )
        host_process = subprocess.Popen(["/bin/sleep", "20"])
        try:
            subprocess.run(
                [
                    str(wrapper),
                    str(host_process.pid),
                    str(denied_write),
                    str(forwarded_link),
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=20,
            )
        finally:
            host_process.terminate()
            host_process.wait(timeout=5)

    assert not denied_write.exists()
    assert (config_dir / "allowed-write").read_text(encoding="utf-8") == "allowed"


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or os.environ.get("TASKCHAMBER_RUN_NATIVE_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled native Linux bubblewrap probe",
)
def test_bwrap_mounts_a_private_tmpfs_over_host_tmp(tmp_path: Path) -> None:
    # SonarCloud python:S5443 flags the bwrap "--tmpfs /tmp" mount. That mount is
    # the isolation control, not an unsafe shared-directory use: it hides the
    # shared host /tmp behind a task-private tmpfs inside the mount namespace.
    # This explicitly enabled native probe is the authoritative evidence.
    sandbox = BubblewrapSandbox()
    assert sandbox.preflight()

    with TemporaryDirectory(prefix="taskchamber-s5443-", dir="/tmp") as host_tmp_directory:
        host_tmp = Path(host_tmp_directory)
        host_canary = host_tmp / "host-canary"
        host_canary.write_text("must-stay-hidden", encoding="utf-8")

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        workspace = IsolatedWorkspace(root=workspace_root, allowed_paths=(workspace_root,))
        sandbox_marker = host_tmp / "sandbox-marker"

        # Launch A: the host canary must be invisible inside the sandbox and a
        # write into the sandbox /tmp must succeed for that task.
        probe = tmp_path / "tmp-isolation-probe"
        probe.write_text(
            "#!/bin/sh\n"
            'if [ -e "$1" ]; then exit 40; fi\n'
            '/bin/mkdir -p "$2" || exit 41\n'
            '/usr/bin/printf private >"$2/sandbox-marker" || exit 42\n',
            encoding="utf-8",
        )
        probe.chmod(0o700)
        subprocess.run(
            [
                str(
                    sandbox.prepare_wrapper(
                        workspace,
                        executable=str(probe),
                        config_dir=tmp_path / "config-a",
                        launcher_dir=tmp_path / "launcher-a",
                    )
                ),
                str(host_canary),
                str(host_tmp),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )

        # The marker written inside sandbox /tmp must not have leaked to the host.
        assert not sandbox_marker.exists()
        # A second, independent launch must not observe launch A's marker: each
        # task gets a fresh private tmpfs, so /tmp contents are not shared across
        # TaskChamber executions.
        verification = tmp_path / "verify-probe"
        verification.write_text(
            '#!/bin/sh\nif [ -e "$1/sandbox-marker" ]; then exit 43; fi\n',
            encoding="utf-8",
        )
        verification.chmod(0o700)
        subprocess.run(
            [
                str(
                    sandbox.prepare_wrapper(
                        workspace,
                        executable=str(verification),
                        config_dir=tmp_path / "config-b",
                        launcher_dir=tmp_path / "launcher-b",
                    )
                ),
                str(host_tmp),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or os.environ.get("TASKCHAMBER_RUN_NATIVE_SANDBOX_TESTS") != "1",
    reason="requires an explicitly enabled native Linux bubblewrap probe",
)
def test_bwrap_rebinds_only_the_executable_below_masked_tmp(tmp_path: Path) -> None:
    # SonarCloud python:S5443 flags the executable rebind below the masked /tmp
    # root. This explicitly enabled native probe is the authoritative evidence
    # that only the exact canonical executable crosses the boundary.
    sandbox = BubblewrapSandbox()
    assert sandbox.preflight()

    with TemporaryDirectory(prefix="taskchamber-rebind-", dir="/tmp") as host_tool_directory:
        tool_dir = Path(host_tool_directory)
        tool_dir.chmod(0o700)
        tool = tool_dir / "probe-cli"
        sibling = tool_dir / "sibling-canary"
        sibling.write_text("must-stay-hidden", encoding="utf-8")
        sibling.chmod(0o600)

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        workspace = IsolatedWorkspace(root=workspace_root, allowed_paths=(workspace_root,))

        tool.write_text(
            "#!/bin/sh\n"
            # The sibling below the same host directory must stay hidden.
            'if [ -e "$1" ]; then exit 40; fi\n'
            # The exact executable itself is rebound and runnable.
            'if [ ! -x "$2" ]; then exit 41; fi\n'
            # Empty recreated parents expose no other adjacent host content.
            'if [ "$(/bin/ls -A "$3" | /usr/bin/wc -l)" -ne 1 ]; then exit 42; fi\n',
            encoding="utf-8",
        )
        tool.chmod(0o700)
        subprocess.run(
            [
                str(
                    sandbox.prepare_wrapper(
                        workspace,
                        executable=str(tool),
                        config_dir=tmp_path / "config",
                        launcher_dir=tmp_path / "launcher",
                    )
                ),
                str(sibling),
                str(tool),
                str(tool_dir),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )

        # A group/world-writable ancestor below the masked root fails closed
        # before any sandbox launch.
        tool_dir.chmod(0o777)
        try:
            with pytest.raises(ValueError, match="writable by group or others"):
                sandbox.prepare_wrapper(
                    workspace,
                    executable=str(tool),
                    config_dir=tmp_path / "config-denied",
                    launcher_dir=tmp_path / "launcher-denied",
                )
        finally:
            tool_dir.chmod(0o700)


@pytest.mark.anyio
async def test_runtime_threads_staged_workspace_and_wrapper_into_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "doc.md").write_text("public", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=x", encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_query(*, prompt: str, options: object) -> object:
        captured["options"] = options
        captured["prompt"] = prompt
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="ok",
        )

    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)
    monkeypatch.setattr("taskchamber.isolation.sandbox.sys.platform", "linux")
    monkeypatch.setattr(BubblewrapSandbox, "preflight", lambda _self: True)
    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "token"},
        query_function=fake_query,
        sandbox=BubblewrapSandbox(bwrap="/usr/bin/env"),
        configured_cli_path=str(cli),
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    options = captured["options"]
    cli_path = options.cli_path  # type: ignore[attr-defined]
    cwd = options.cwd  # type: ignore[attr-defined]
    assert cli_path is not None
    assert "agent-cli-launcher" in str(cli_path)
    assert cwd != tmp_path  # the SDK runs against the staged copy, not the source root
    assert result.execution is not None
    assert result.execution.sandbox == "bwrap"
    assert result.execution.workspace_staged is True
    # The injected query never starts the generated CLI path, so telemetry is
    # deliberately conservative even though the wrapper and preflight exist.
    assert result.execution.os_isolated is False
    assert result.execution.cli_wrapper_active is True
    assert result.execution.cli_launch_observed is False
    assert result.execution.sandbox_preflight_passed is True
    assert result.execution.isolation_scope == "none"
    assert result.execution.runtime_process_isolated is False
    assert result.execution.cli_environment_sanitized is False
    assert result.execution.cli_executable_source == "configured"


@pytest.mark.anyio
async def test_default_runtime_stages_with_a_clean_environment_wrapper(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    (tmp_path / "doc.md").write_text("public", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")

    async def fake_query(*, prompt: str, options: object) -> object:
        captured["options"] = options
        captured["dotenv_visible"] = (options.cwd / ".env").exists()  # type: ignore[attr-defined]
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="ok",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "token"}, query_function=fake_query
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    options = captured["options"]
    assert options.cli_path is not None  # type: ignore[attr-defined]
    assert options.cwd != tmp_path  # type: ignore[attr-defined]
    assert captured["dotenv_visible"] is False
    assert result.execution is not None
    assert result.execution.sandbox == "none"
    assert result.execution.workspace_staged is True
    assert result.execution.os_isolated is False
    assert result.execution.cli_wrapper_active is True
    assert result.execution.cli_launch_observed is False
    assert result.execution.sandbox_preflight_passed is None
    assert result.execution.isolation_scope == "none"
    assert result.execution.runtime_process_isolated is False
    assert result.execution.cli_environment_sanitized is False
    assert result.execution.cli_executable_source == "bundled"


@pytest.mark.anyio
async def test_runtime_wraps_a_legacy_sandbox_adapter_without_new_kwargs(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class LegacySandbox(Sandbox):
        name = "legacy"
        os_isolated = True

        def prepare_wrapper(
            self,
            workspace: IsolatedWorkspace,
            *,
            executable: str,
            config_dir: Path,
        ) -> Path:
            del workspace, config_dir
            return Path(executable)

    async def fake_query(*, prompt: str, options: object) -> object:
        del prompt
        captured["options"] = options
        subprocess.run(
            [str(options.cli_path), "--output-format", "stream-json"],  # type: ignore[attr-defined]
            check=True,
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="ok",
        )

    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)
    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "token"},
        query_function=fake_query,
        sandbox=LegacySandbox(),
        configured_cli_path=str(cli),
    )

    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.execution is not None
    assert result.execution.cli_wrapper_active is True
    assert result.execution.cli_launch_observed is True
    assert result.execution.cli_environment_sanitized is True
    assert result.execution.sandbox_preflight_passed is None
    assert result.execution.os_isolated is False
    assert result.execution.isolation_scope == "none"


@pytest.mark.anyio
async def test_deprecated_cli_resolver_remains_a_fail_closed_compatibility_shim(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)

    async def fake_query(*, prompt: str, options: object) -> object:
        del prompt, options
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="ok",
        )

    with pytest.warns(DeprecationWarning, match="cli_resolver"):
        runtime = ClaudeAgentSdkRuntime(
            environment={"Z_AI_API_KEY": "token"},
            query_function=fake_query,
            cli_resolver=lambda name: str(cli) if name == "claude" else None,
        )

    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.execution is not None
    assert result.execution.cli_executable_source == "configured"


@pytest.mark.anyio
async def test_runtime_fails_closed_when_cli_cannot_be_resolved(tmp_path: Path) -> None:
    called = False

    async def fake_query(*, prompt: str, options: object) -> object:
        nonlocal called
        called = True
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="ok",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "token"},
        query_function=fake_query,
        sandbox=NoSandbox(),
        bundled_cli_resolver=lambda: None,
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == "cli_unavailable"
    assert called is False
    assert result.execution is not None
    assert result.execution.sandbox == "none"
    assert result.execution.workspace_staged is False
    assert result.execution.os_isolated is False
    assert result.execution.cli_wrapper_active is False
    assert result.execution.cli_launch_observed is False
    assert result.execution.cli_environment_sanitized is False


@pytest.mark.anyio
async def test_runtime_rechecks_requested_sandbox_availability(tmp_path: Path) -> None:
    called = False

    async def fake_query(*, prompt: str, options: object) -> object:
        nonlocal called
        called = True
        yield object()

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "token"},
        query_function=fake_query,
        sandbox=BubblewrapSandbox(bwrap="/does/not/exist/bwrap"),
    )

    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "sandbox_unavailable"
    assert called is False
    assert result.execution is not None
    assert result.execution.sandbox == "bwrap"
    assert result.execution.os_isolated is False
    assert result.execution.cli_launch_observed is False
    assert result.execution.sandbox_preflight_passed is False
    assert result.execution.runtime_process_isolated is False


@pytest.mark.anyio
async def test_runtime_rejects_a_ca_path_hidden_by_native_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_query(*, prompt: str, options: object) -> object:
        nonlocal called
        called = True
        yield object()

    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)
    monkeypatch.setenv("SSL_CERT_DIR", str(Path.home().resolve()))
    monkeypatch.setattr(MacOSSandboxExecSandbox, "preflight", lambda _self: True)
    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "token"},
        query_function=fake_query,
        sandbox=MacOSSandboxExecSandbox(sandbox_exec="/usr/bin/sandbox-exec"),
        configured_cli_path=str(cli),
    )

    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "cli_environment_invalid"
    assert called is False
    assert str(Path.home()) not in (result.error_message or "")


def test_isolated_workspace_is_immutable(tmp_path: Path) -> None:
    from taskchamber.isolation import IsolatedWorkspace

    workspace = IsolatedWorkspace(root=tmp_path, allowed_paths=(tmp_path,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        workspace.root = tmp_path / "other"  # type: ignore[misc]

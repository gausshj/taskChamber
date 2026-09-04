"""Read-only deployment checks that never call an agent provider."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from .application.composition import create_runtime_registry
from .config import load_configuration, load_project_policy
from .isolation import InsecureCliPathError, select_sandbox
from .runtimes.registry import RuntimeFactoryContext

CLI_PATH_REMEDIATION = (
    "set TASKCHAMBER_CLAUDE_CLI_PATH to an absolute owner-only executable "
    "whose parent directories are not group/world-writable"
)


def deployment_report(
    *,
    config_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
    working_directory: Path | None = None,
) -> dict[str, Any]:
    """Return deployment readiness without starting a task or provider request."""

    launch_directory = (working_directory or Path.cwd()).expanduser().resolve()
    report: dict[str, Any] = {
        "schema": "taskchamber.doctor.v1",
        "ok": False,
        "taskchamber": _installation_identity(),
        "checks": {},
    }
    checks: dict[str, dict[str, Any]] = report["checks"]

    try:
        configuration = load_configuration(
            environment=environment,
            working_directory=launch_directory,
        )
        loaded_policy = load_project_policy(
            configuration,
            working_directory=launch_directory,
            config_file=config_file,
        )
    except (OSError, ValueError) as exc:
        checks["configuration"] = _failed("configuration_invalid", exc)
        return report

    checks["configuration"] = {
        "ok": True,
        "env_file": str(configuration.env_file) if configuration.env_file else None,
        "policy_file": str(loaded_policy.path) if loaded_policy.path else None,
        "workspace_root": (
            str(loaded_policy.workspace_root) if loaded_policy.workspace_root else None
        ),
    }

    runtime_name = configuration.values.get("TASKCHAMBER_RUNTIME", "claude") or "claude"
    runtime_name = runtime_name.strip().lower()
    sandbox_mode = configuration.values.get("TASKCHAMBER_SANDBOX", "auto") or "auto"
    sandbox_mode = sandbox_mode.strip().lower()

    try:
        sandbox = select_sandbox(sandbox_mode)
        preflight_passed = sandbox.preflight()
        if not preflight_passed:
            raise ValueError(f"selected sandbox {sandbox.name!r} failed its operational preflight")
    except (OSError, ValueError) as exc:
        checks["sandbox"] = {
            **_failed("sandbox_unavailable", exc),
            "requested": sandbox_mode,
        }
        return report

    checks["sandbox"] = {
        "ok": True,
        "requested": sandbox_mode,
        "selected": sandbox.name,
        "os_isolated": sandbox.os_isolated,
        "preflight_passed": preflight_passed,
    }

    try:
        registry = create_runtime_registry()
        runtime = registry.create(
            runtime_name,
            RuntimeFactoryContext(configuration=configuration, sandbox=sandbox),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        checks["runtime"] = {
            **_failed("runtime_unavailable", exc),
            "name": runtime_name,
        }
        return report

    checks["runtime"] = {
        "ok": True,
        "name": runtime_name,
        "adapter": runtime.name,
        "default_profile": runtime.default_profile,
    }

    if runtime_name != "claude":
        checks["agent_cli"] = {
            "ok": True,
            "skipped": True,
            "reason": "the selected runtime does not use the built-in Claude CLI adapter",
        }
        report["ok"] = True
        return report

    try:
        from .runtimes.claude.cli import ClaudeCliUnavailableError, resolve_claude_cli
    except ImportError as exc:
        checks["agent_cli"] = _failed("claude_cli_unavailable", exc)
        return report

    try:
        executable = resolve_claude_cli(configuration.values.get("TASKCHAMBER_CLAUDE_CLI_PATH"))
        sandbox.validate_cli_executable(executable.path)
    except InsecureCliPathError as exc:
        checks["agent_cli"] = {
            **_failed("sandbox_cli_path_insecure", exc),
            "remediation": CLI_PATH_REMEDIATION,
        }
        return report
    except (ClaudeCliUnavailableError, OSError, ValueError) as exc:
        checks["agent_cli"] = _failed("claude_cli_unavailable", exc)
        return report

    checks["agent_cli"] = {
        "ok": True,
        "source": executable.source,
        "path": str(executable.path),
        "sandbox_compatible": True,
    }
    report["ok"] = True
    return report


def _installation_identity() -> dict[str, str]:
    try:
        package_version = metadata.version("taskchamber")
    except metadata.PackageNotFoundError:
        package_version = "source-tree"

    invoked = Path(sys.argv[0]).expanduser()
    if not invoked.is_absolute():
        located = shutil.which(str(invoked))
        if located is not None:
            invoked = Path(located)
    try:
        entrypoint = str(invoked.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        entrypoint = str(invoked)

    return {
        "version": package_version,
        "entrypoint": entrypoint,
        "package_root": str(Path(__file__).resolve().parent),
        "python": str(Path(sys.executable).resolve()),
    }


def _failed(code: str, exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": code,
        "message": str(exc),
    }


__all__ = ["deployment_report"]

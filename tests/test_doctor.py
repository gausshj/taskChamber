import json
import sys
from pathlib import Path

import pytest

from taskchamber import doctor as doctor_module
from taskchamber.cli import main
from taskchamber.doctor import deployment_report
from taskchamber.isolation import InsecureCliPathError, NoSandbox
from taskchamber.runtimes.fake.runtime import FakeRuntime


class _InsecureCliSandbox(NoSandbox):
    name = "test-isolated"
    os_isolated = True

    def validate_cli_executable(self, _executable: Path) -> None:
        raise InsecureCliPathError("configure an owner-only executable")


class _UnavailableSandbox(NoSandbox):
    name = "test-unavailable"

    def preflight(self) -> bool:
        return False


class _ClaudeRuntime(FakeRuntime):
    name = "claude"


class _ClaudeRuntimeRegistry:
    def create(self, _name: str, _context: object) -> FakeRuntime:
        return _ClaudeRuntime()


def test_doctor_checks_fake_runtime_without_provider_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TASKCHAMBER_RUNTIME", "fake")
    monkeypatch.setenv("TASKCHAMBER_SANDBOX", "none")
    monkeypatch.delenv("TASKCHAMBER_CONFIG_FILE", raising=False)
    monkeypatch.delenv("TASKCHAMBER_ENV_FILE", raising=False)

    main(["doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "taskchamber.doctor.v1"
    assert payload["ok"] is True
    assert payload["checks"]["configuration"]["ok"] is True
    assert payload["checks"]["runtime"]["name"] == "fake"
    assert payload["checks"]["sandbox"] == {
        "ok": True,
        "requested": "none",
        "selected": "none",
        "os_isolated": False,
        "preflight_passed": True,
    }
    assert payload["checks"]["agent_cli"]["skipped"] is True
    assert Path(payload["taskchamber"]["package_root"]).name == "taskchamber"


def test_doctor_reports_invalid_project_configuration(tmp_path: Path) -> None:
    config = tmp_path / "taskchamber.toml"
    config.write_text("not valid toml =", encoding="utf-8")

    payload = deployment_report(
        config_file=config,
        environment={"TASKCHAMBER_RUNTIME": "fake", "TASKCHAMBER_SANDBOX": "none"},
        working_directory=tmp_path,
    )

    assert payload["ok"] is False
    assert payload["checks"]["configuration"]["error_code"] == "configuration_invalid"


def test_doctor_reports_failed_sandbox_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "taskchamber.doctor.select_sandbox",
        lambda _mode: _UnavailableSandbox(),
    )

    payload = deployment_report(
        environment={"TASKCHAMBER_RUNTIME": "fake", "TASKCHAMBER_SANDBOX": "required"},
        working_directory=tmp_path,
    )

    assert payload["ok"] is False
    assert payload["checks"]["sandbox"] == {
        "ok": False,
        "error_code": "sandbox_unavailable",
        "message": "selected sandbox 'test-unavailable' failed its operational preflight",
        "requested": "required",
    }


def test_doctor_reports_an_unavailable_runtime(tmp_path: Path) -> None:
    payload = deployment_report(
        environment={"TASKCHAMBER_RUNTIME": "missing", "TASKCHAMBER_SANDBOX": "none"},
        working_directory=tmp_path,
    )

    assert payload["ok"] is False
    assert payload["checks"]["runtime"]["ok"] is False
    assert payload["checks"]["runtime"]["error_code"] == "runtime_unavailable"
    assert payload["checks"]["runtime"]["name"] == "missing"


def test_doctor_reports_an_unavailable_claude_cli_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor_module,
        "create_runtime_registry",
        lambda: _ClaudeRuntimeRegistry(),
    )
    monkeypatch.setitem(sys.modules, "taskchamber.runtimes.claude.cli", None)

    payload = deployment_report(
        environment={"TASKCHAMBER_RUNTIME": "claude", "TASKCHAMBER_SANDBOX": "none"},
        working_directory=tmp_path,
    )

    assert payload["ok"] is False
    assert payload["checks"]["agent_cli"]["error_code"] == "claude_cli_unavailable"


def test_installation_identity_falls_back_for_a_source_tree_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def package_not_found(_distribution: str) -> str:
        raise doctor_module.metadata.PackageNotFoundError

    monkeypatch.setattr(doctor_module.metadata, "version", package_not_found)
    monkeypatch.setattr(sys, "argv", ["missing-taskchamber"])
    monkeypatch.setattr(doctor_module.shutil, "which", lambda _command: None)

    identity = doctor_module._installation_identity()

    assert identity["version"] == "source-tree"
    assert identity["entrypoint"] == "missing-taskchamber"


def test_installation_identity_resolves_an_entrypoint_from_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = tmp_path / "taskchamber"
    entrypoint.touch()
    monkeypatch.setattr(sys, "argv", ["taskchamber"])
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda _command: str(entrypoint),
    )

    identity = doctor_module._installation_identity()

    assert identity["entrypoint"] == str(entrypoint.resolve())


def test_doctor_validates_configured_claude_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = tmp_path / "cli" / "claude"
    cli.parent.mkdir()
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)
    environment = {
        "TASKCHAMBER_RUNTIME": "claude",
        "TASKCHAMBER_SANDBOX": "none",
        "TASKCHAMBER_CLAUDE_CLI_PATH": str(cli),
    }

    payload = deployment_report(environment=environment, working_directory=tmp_path)

    assert payload["ok"] is True
    assert payload["checks"]["runtime"]["name"] == "claude"
    assert payload["checks"]["agent_cli"] == {
        "ok": True,
        "source": "configured",
        "path": str(cli.resolve()),
        "sandbox_compatible": True,
    }


def test_doctor_returns_machine_readable_failure_for_bad_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TASKCHAMBER_RUNTIME", "claude")
    monkeypatch.setenv("TASKCHAMBER_SANDBOX", "none")
    monkeypatch.setenv("TASKCHAMBER_CLAUDE_CLI_PATH", "relative/claude")
    monkeypatch.delenv("TASKCHAMBER_CONFIG_FILE", raising=False)
    monkeypatch.delenv("TASKCHAMBER_ENV_FILE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        main(["doctor"])

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["checks"]["agent_cli"]["error_code"] == "claude_cli_unavailable"
    assert "not absolute" in payload["checks"]["agent_cli"]["message"]


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    (
        (
            "TASKCHAMBER_MCP_TEXT_MODE",
            "bogus",
            "must be one of",
        ),
        (
            "TASKCHAMBER_MAX_SINGLE_PASS_DOCUMENT_BYTES",
            "bogus",
            "must be a positive integer",
        ),
        (
            "TASKCHAMBER_DOCUMENT_SOURCES",
            "INVALID-NAME",
            "must contain lowercase source names",
        ),
    ),
)
def test_doctor_rejects_configuration_that_would_prevent_server_startup(
    tmp_path: Path,
    setting: str,
    value: str,
    message: str,
) -> None:
    environment = {
        "TASKCHAMBER_RUNTIME": "fake",
        "TASKCHAMBER_SANDBOX": "none",
        setting: value,
    }

    payload = deployment_report(environment=environment, working_directory=tmp_path)

    assert payload["ok"] is False
    assert payload["checks"]["configuration"]["ok"] is False
    assert payload["checks"]["configuration"]["error_code"] == "configuration_invalid"
    assert message in payload["checks"]["configuration"]["message"]


def test_doctor_reports_sandbox_incompatible_cli_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = tmp_path / "cli" / "claude"
    cli.parent.mkdir()
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)
    monkeypatch.setattr(
        "taskchamber.doctor.select_sandbox",
        lambda _mode: _InsecureCliSandbox(),
    )
    environment = {
        "TASKCHAMBER_RUNTIME": "claude",
        "TASKCHAMBER_SANDBOX": "required",
        "TASKCHAMBER_CLAUDE_CLI_PATH": str(cli),
    }

    payload = deployment_report(environment=environment, working_directory=tmp_path)

    assert payload["ok"] is False
    assert payload["checks"]["agent_cli"] == {
        "ok": False,
        "error_code": "sandbox_cli_path_insecure",
        "message": "configure an owner-only executable",
        "remediation": (
            "set TASKCHAMBER_CLAUDE_CLI_PATH to an absolute owner-only executable "
            "whose parent directories are not group/world-writable"
        ),
    }

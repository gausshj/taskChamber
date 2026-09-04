from collections.abc import Sequence

import pytest

from taskchamber.cli import main


def _help_output(
    arguments: Sequence[str],
    capsys: pytest.CaptureFixture[str],
) -> str:
    with pytest.raises(SystemExit) as exc_info:
        main(arguments)

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.err == ""
    return captured.out


def test_root_help_explains_server_and_management_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help_output(["--help"], capsys)

    assert "Run bounded, isolated agent tasks as a standard MCP server." in output
    assert "Run without a command to start the stdio server." in output
    assert "taskchamber serve" in output
    assert "taskchamber config init" in output
    assert "taskchamber policy validate" in output
    assert "taskchamber doctor" in output


def test_serve_help_documents_stdio_contract_without_starting_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from taskchamber.transport import mcp

    def fail_if_started() -> None:
        raise AssertionError("serve help started the MCP server")

    monkeypatch.setattr(mcp, "main", fail_if_started)

    output = _help_output(["serve", "-h"], capsys)

    assert "Run TaskChamber as a stdio MCP server." in output
    assert "equivalent to running taskchamber with no arguments" in output
    assert "research, summarize, and review" in output
    assert "Standard output is reserved for MCP protocol traffic." in output
    assert "taskchamber.toml" in output
    assert ".env" in output
    assert "TASKCHAMBER_CONFIG_FILE" in output
    assert "TASKCHAMBER_ENV_FILE" in output
    assert "TASKCHAMBER_RUNTIME=fake taskchamber serve" in output
    assert "taskchamber doctor" in output


@pytest.mark.parametrize(
    ("arguments", "expected_text"),
    [
        (
            ["config", "--help"],
            (
                "Create TaskChamber project configuration.",
                "taskchamber config init",
            ),
        ),
        (
            ["config", "init", "--help"],
            (
                "Create a documented TaskChamber project-policy template.",
                "configuration file to create",
                "replace the target file if it already exists",
            ),
        ),
        (
            ["doctor", "--help"],
            (
                "Validate configuration, runtime, agent CLI, and sandbox readiness",
                "without starting the MCP server or calling a provider",
                "explicit project policy file",
            ),
        ),
        (
            ["policy", "--help"],
            (
                "Inspect or edit the TaskChamber project policy.",
                "print the effective non-secret policy as JSON",
                "load and validate the project policy",
                "allow capabilities for one task",
                "deny capabilities for one task",
                "replace the default capabilities for one task",
            ),
        ),
        (
            ["policy", "show", "--help"],
            (
                "Print the effective non-secret project policy as JSON.",
                "explicit project policy file",
            ),
        ),
        (
            ["policy", "validate", "--help"],
            (
                "Validate the project policy without changing it.",
                "explicit project policy file",
            ),
        ),
        (
            ["policy", "allow", "--help"],
            (
                "Add capabilities to one task's allowed set.",
                "task policy to edit",
                "one or more provider-neutral capability names",
            ),
        ),
        (
            ["policy", "deny", "--help"],
            (
                "Remove capabilities from one task's allowed and default sets.",
                "task policy to edit",
                "one or more provider-neutral capability names",
            ),
        ),
        (
            ["policy", "set-default", "--help"],
            (
                "Replace one task's default capability set.",
                "task policy to edit",
                "one or more provider-neutral capability names",
            ),
        ),
    ],
)
def test_management_help_describes_each_command_and_argument(
    arguments: list[str],
    expected_text: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help_output(arguments, capsys)

    for text in expected_text:
        assert text in output

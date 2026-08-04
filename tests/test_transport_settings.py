from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from taskchamber.config import load_configuration
from taskchamber.transport.mcp import create_server
from taskchamber.transport.settings import (
    TEXT_MODE_VARIABLE,
    MCPTextMode,
    MCPTransportSettings,
)


def test_text_mode_defaults_to_full_without_setting(tmp_path: Path) -> None:
    configuration = load_configuration(environment={}, working_directory=tmp_path)

    assert MCPTransportSettings.from_configuration(configuration).text_mode is MCPTextMode.FULL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("full", MCPTextMode.FULL),
        ("metadata_only", MCPTextMode.METADATA_ONLY),
        (" metadata_only ", MCPTextMode.METADATA_ONLY),
    ],
)
def test_text_mode_parses_supported_values(raw: str, expected: MCPTextMode) -> None:
    configuration = load_configuration(environment={TEXT_MODE_VARIABLE: raw})

    assert MCPTransportSettings.from_configuration(configuration).text_mode is expected


def test_process_environment_overrides_dotenv_text_mode(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"{TEXT_MODE_VARIABLE}=metadata_only\n",
        encoding="utf-8",
    )

    configuration = load_configuration(
        environment={TEXT_MODE_VARIABLE: "full"},
        working_directory=tmp_path,
    )

    assert MCPTransportSettings.from_configuration(configuration).text_mode is MCPTextMode.FULL


def test_dotenv_text_mode_applies_without_process_override(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"{TEXT_MODE_VARIABLE}=metadata_only\n",
        encoding="utf-8",
    )

    configuration = load_configuration(environment={}, working_directory=tmp_path)

    settings = MCPTransportSettings.from_configuration(configuration)

    assert settings.text_mode is MCPTextMode.METADATA_ONLY


def test_invalid_text_mode_fails_instead_of_falling_back(tmp_path: Path) -> None:
    configuration = load_configuration(
        environment={TEXT_MODE_VARIABLE: "compact"},
        working_directory=tmp_path,
    )

    with pytest.raises(ValueError, match=TEXT_MODE_VARIABLE):
        MCPTransportSettings.from_configuration(configuration)


def test_create_server_rejects_invalid_text_mode_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "server.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("TASKCHAMBER_ENV_FILE", str(env_file))
    monkeypatch.setenv("TASKCHAMBER_RUNTIME", "fake")
    monkeypatch.setenv("TASKCHAMBER_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(TEXT_MODE_VARIABLE, "bogus")

    with pytest.raises(ValueError, match=TEXT_MODE_VARIABLE):
        create_server()


@pytest.mark.anyio
async def test_create_server_reads_text_mode_from_loaded_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "server.env"
    env_file.write_text(f"{TEXT_MODE_VARIABLE}=metadata_only\n", encoding="utf-8")
    monkeypatch.setenv("TASKCHAMBER_ENV_FILE", str(env_file))
    monkeypatch.setenv("TASKCHAMBER_RUNTIME", "fake")
    monkeypatch.setenv("TASKCHAMBER_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv(TEXT_MODE_VARIABLE, raising=False)

    server = create_server()

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("research", {"question": "q"})

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["output"] == "fake research result"
    assert "fake research result" not in result.content[0].text
    assert result.model_dump_json().count("fake research result") == 1

import json
import sys
import tempfile
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client


@pytest.mark.anyio
async def test_stdio_server_runs_with_fake_runtime(tmp_path: Path) -> None:
    env_file = tmp_path / "server.env"
    env_file.write_text(
        "TASKCHAMBER_DEFAULT_PROFILE=fake-profile\n",
        encoding="utf-8",
    )
    environment = {
        "TASKCHAMBER_ENV_FILE": str(env_file),
        "TASKCHAMBER_RUNTIME": "fake",
        "TASKCHAMBER_WORKSPACE_ROOT": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "taskchamber"],
        cwd=tmp_path,
        env=environment,
    )

    with tempfile.TemporaryFile(mode="w+t") as errlog:
        async with stdio_client(parameters, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                initialization = await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                result = await session.call_tool(
                    "research",
                    {"question": "Smoke test"},
                )
                error_result = await session.call_tool(
                    "summarize",
                    {"file_path": "../outside.txt", "provider": "fake-profile"},
                )

    assert initialization.serverInfo.name == "taskchamber"
    assert tools == {"research", "summarize", "review"}
    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["runtime"] == "fake"
    assert result.structuredContent["model"] == "fake-runtime"
    assert result.structuredContent["provider"] == "fake-profile"
    assert error_result.isError is True
    assert error_result.structuredContent is not None
    assert error_result.structuredContent["status"] == "policy_denied"


@pytest.mark.anyio
async def test_stdio_server_prepares_cli_document_source_without_workspace_copy(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_file = tmp_path / "server.env"
    argv = json.dumps(
        [
            sys.executable,
            str(repository_root / "scripts" / "document_fixture_cli.py"),
            "--query",
            "{query}",
        ],
        separators=(",", ":"),
    )
    env_file.write_text(
        "\n".join(
            [
                "TASKCHAMBER_DEFAULT_PROFILE=fake-profile",
                "TASKCHAMBER_DOCUMENT_SOURCES=fixture_cli",
                "TASKCHAMBER_DOCUMENT_SOURCE__FIXTURE_CLI__KIND=command",
                f"TASKCHAMBER_DOCUMENT_SOURCE__FIXTURE_CLI__ARGV={argv}",
                "TASKCHAMBER_DOCUMENT_SOURCE__FIXTURE_CLI__OUTPUT_FORMAT=json",
            ]
        ),
        encoding="utf-8",
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "taskchamber"],
        cwd=repository_root,
        env={
            "TASKCHAMBER_ENV_FILE": str(env_file),
            "TASKCHAMBER_RUNTIME": "fake",
            "TASKCHAMBER_WORKSPACE_ROOT": str(workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )

    with tempfile.TemporaryFile(mode="w+t") as errlog:
        async with stdio_client(parameters, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "research",
                    {
                        "question": "Return the CLI canary",
                        "document_sources": ["fixture_cli"],
                        "include_workspace": False,
                    },
                )

    assert result.isError is False
    assert result.structuredContent is not None
    execution = result.structuredContent["execution"]
    assert execution["allowed_tools"] == []
    assert execution["document_sources"] == ["fixture_cli"]
    assert list(workspace.iterdir()) == []

"""Install a built TaskChamber wheel twice and smoke-test its stdio MCP command."""

from __future__ import annotations

import argparse
import asyncio
import email
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

INSTALL_ENV_ALLOWLIST = (
    "ALL_PROXY",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "UV_CACHE_DIR",
)


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


async def _smoke_mcp(command: Path, workspace: Path, home: Path) -> None:
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TASKCHAMBER_RUNTIME": "fake",
        "TASKCHAMBER_SANDBOX": "none",
        "TASKCHAMBER_WORKSPACE_ROOT": str(workspace),
    }
    parameters = StdioServerParameters(
        command=str(command),
        args=[],
        cwd=workspace,
        env=environment,
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(tool.name for tool in (await session.list_tools()).tools)
            if tools != ["research", "review", "summarize"]:
                raise RuntimeError(f"unexpected installed MCP tools: {tools}")
            result = await session.call_tool(
                "research",
                {"question": "Distribution smoke test", "max_turns": 1},
            )
    structured = result.structuredContent or {}
    if (
        result.isError
        or structured.get("status") != "success"
        or structured.get("provider") != "fake"
    ):
        raise RuntimeError(f"installed MCP call failed: {structured}")


def _tool_python(tool_directory: Path) -> Path:
    if os.name == "nt":
        return tool_directory / "taskchamber" / "Scripts" / "python.exe"
    return tool_directory / "taskchamber" / "bin" / "python"


def _wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"expected one METADATA file in {wheel.name}")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
    version = metadata.get("Version")
    if not version:
        raise RuntimeError(f"wheel has no package version: {wheel.name}")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Built TaskChamber wheel to install.")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"wheel does not exist: {wheel}")
    expected_version = _wheel_version(wheel)

    with tempfile.TemporaryDirectory(prefix="taskchamber-tool-test-") as temporary:
        root = Path(temporary)
        tool_directory = root / "tools"
        bin_directory = root / "bin"
        workspace = root / "workspace"
        home = root / "home"
        workspace.mkdir()
        home.mkdir()

        environment = {key: os.environ[key] for key in INSTALL_ENV_ALLOWLIST if key in os.environ}
        environment.update(
            {
                "UV_TOOL_BIN_DIR": str(bin_directory),
                "UV_TOOL_DIR": str(tool_directory),
            }
        )
        requirement = f"taskchamber[claude] @ {wheel.as_uri()}"
        install_command = [
            "uv",
            "tool",
            "install",
            "--no-config",
            "--python",
            "3.11",
            requirement,
        ]

        _run(install_command, cwd=root, environment=environment)
        _run(install_command, cwd=root, environment=environment)

        executable = bin_directory / ("taskchamber.exe" if os.name == "nt" else "taskchamber")
        python = _tool_python(tool_directory)
        if not executable.is_file() or not python.is_file():
            raise RuntimeError("uv did not create the expected isolated tool environment")

        _run(
            [
                str(python),
                "-c",
                "import os; import claude_agent_sdk; import taskchamber; "
                "from importlib.metadata import version; "
                "from taskchamber.runtimes.claude.cli import resolve_claude_cli; "
                "cli = resolve_claude_cli(); "
                "assert cli.source == 'bundled'; "
                "assert cli.path.is_file() and os.access(cli.path, os.X_OK); "
                f"assert version('taskchamber') == {json.dumps(expected_version)}",
            ],
            cwd=root,
            environment=environment,
        )
        asyncio.run(_smoke_mcp(executable, workspace, home))

    print("PASS uv tool installed taskchamber[claude] twice in one isolated environment")
    print("PASS installed taskchamber completed initialize, tools/list, and tools/call over stdio")


if __name__ == "__main__":
    main()

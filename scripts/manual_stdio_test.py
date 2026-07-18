"""Run one visible MCP call through the real stdio server for user acceptance."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

SAFE_PARENT_ENV = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TMPDIR",
    "USER",
)


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected one JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected one JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch taskchamber over stdio and make one research call.",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", default="claude", choices=("claude", "fake"))
    parser.add_argument("--provider", help="Override the runtime's default provider profile.")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-output-chars", type=int)
    parser.add_argument(
        "--document-mode",
        default="agentic",
        choices=("agentic", "single_pass"),
    )
    parser.add_argument("--sandbox", default="auto")
    parser.add_argument(
        "--document-source",
        action="append",
        default=[],
        help="Select one configured virtual document source; repeat for multiple sources.",
    )
    parser.add_argument(
        "--document-request",
        action="append",
        type=_json_object,
        default=[],
        help='Structured source JSON, for example {"source":"record_detail",...}.',
    )
    parser.add_argument(
        "--workspace-path",
        action="append",
        default=[],
        help="Narrow the staged workspace with a relative path/glob; repeat as needed.",
    )
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Request one provider-neutral capability; repeat as needed.",
    )
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Expose only selected virtual documents and stage an empty workspace.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Use an explicit ignored dotenv file for providers and document sources.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Use an explicit taskchamber.toml project policy.",
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        help="Override ~/.claude/settings.json for the Claude Code fallback.",
    )
    parser.add_argument(
        "--question",
        default="List the top-level Python package directories in this workspace.",
    )
    return parser.parse_args()


def server_environment(args: argparse.Namespace) -> dict[str, str]:
    """Pass a minimal host environment; the server reads credentials itself."""

    environment = {key: os.environ[key] for key in SAFE_PARENT_ENV if key in os.environ}
    environment.update(
        {
            "TASKCHAMBER_RUNTIME": args.runtime,
            "TASKCHAMBER_SANDBOX": args.sandbox,
            "TASKCHAMBER_WORKSPACE_ROOT": str(args.workspace.expanduser().resolve()),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if args.settings_file is not None:
        environment["TASKCHAMBER_CLAUDE_SETTINGS_FILE"] = str(
            args.settings_file.expanduser().resolve()
        )
    if args.env_file is not None:
        environment["TASKCHAMBER_ENV_FILE"] = str(args.env_file.expanduser().resolve())
    if args.config_file is not None:
        environment["TASKCHAMBER_CONFIG_FILE"] = str(args.config_file.expanduser().resolve())
    return environment


async def run(args: argparse.Namespace) -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "taskchamber"],
        cwd=repository_root,
        env=server_environment(args),
    )
    tool_arguments: dict[str, object] = {
        "question": args.question,
        "max_turns": args.max_turns,
        "document_mode": args.document_mode,
    }
    if args.max_output_chars is not None:
        tool_arguments["max_output_chars"] = args.max_output_chars
    if args.provider:
        tool_arguments["provider"] = args.provider
    if args.document_source:
        tool_arguments["document_sources"] = args.document_source
    if args.document_request:
        tool_arguments["document_requests"] = args.document_request
    if args.workspace_path:
        tool_arguments["workspace_paths"] = args.workspace_path
    if args.capability:
        tool_arguments["requested_capabilities"] = args.capability
    if args.no_workspace:
        tool_arguments["include_workspace"] = False

    async with stdio_client(parameters, errlog=sys.stderr) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(tool.name for tool in (await session.list_tools()).tools)
            print(f"MCP tools: {', '.join(tools)}")
            result = await session.call_tool("research", tool_arguments)

    print("\nStructured result:")
    print(json.dumps(result.structuredContent, ensure_ascii=False, indent=2))
    print("\nText result:")
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            print(text)
    return 1 if result.isError else 0


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()

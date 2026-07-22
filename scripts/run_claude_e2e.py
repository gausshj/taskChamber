"""Run the outer Claude Code agent against an already installed local MCP server."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=Path)
    return parser.parse_args()


def main() -> None:
    test_root = parse_args().test_root.expanduser().resolve()
    workspace = test_root / "workspace"
    prompt_file = test_root / "outer-prompt.txt"
    if not workspace.is_dir() or not prompt_file.is_file():
        raise SystemExit("not an taskchamber E2E fixture")
    claude = shutil.which("claude")
    if claude is None:
        raise SystemExit("Claude Code CLI was not found")

    command = [
        claude,
        "-p",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--max-turns",
        "4",
        "--tools",
        "",
        "--allowedTools",
        "mcp__taskchamber__research",
    ]
    result = subprocess.run(
        command,
        cwd=workspace,
        env=os.environ.copy(),
        input=prompt_file.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        check=False,
    )
    (test_root / "outer-result.json").write_text(result.stdout, encoding="utf-8")
    (test_root / "outer-stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

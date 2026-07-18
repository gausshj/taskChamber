import os
from pathlib import Path

import pytest

from taskchamber.runtimes.claude.cli import (
    ClaudeCliUnavailableError,
    bundled_claude_cli_path,
    resolve_claude_cli,
)


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def test_default_cli_resolution_uses_the_pinned_sdk_bundle() -> None:
    bundled = bundled_claude_cli_path()

    assert bundled is not None
    resolved = resolve_claude_cli()
    assert resolved.source == "bundled"
    assert resolved.path == bundled.resolve()
    assert os.access(resolved.path, os.X_OK)


def test_configured_cli_path_is_explicit_and_canonical(tmp_path: Path) -> None:
    target = _executable(tmp_path / "claude-real")
    link = tmp_path / "claude"
    link.symlink_to(target)

    resolved = resolve_claude_cli(str(link), bundled_resolver=lambda: None)

    assert resolved.source == "configured"
    assert resolved.path == target.resolve()


@pytest.mark.parametrize(
    "configured",
    ("relative/claude", "/does/not/exist/claude", "/tmp/invalid\0claude"),
)
def test_invalid_configured_cli_fails_closed(configured: str) -> None:
    with pytest.raises(ClaudeCliUnavailableError):
        resolve_claude_cli(configured, bundled_resolver=lambda: None)


def test_missing_bundle_does_not_fall_back_to_ambient_path() -> None:
    with pytest.raises(ClaudeCliUnavailableError):
        resolve_claude_cli(bundled_resolver=lambda: None)


def test_non_executable_cli_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "claude"
    candidate.write_text("not executable", encoding="utf-8")
    candidate.chmod(0o600)

    with pytest.raises(ClaudeCliUnavailableError):
        resolve_claude_cli(str(candidate))

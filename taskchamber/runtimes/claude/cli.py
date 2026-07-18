"""Deterministic Claude CLI discovery owned by the Claude runtime adapter."""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import claude_agent_sdk

ClaudeCliSource = Literal["bundled", "configured"]


class ClaudeCliUnavailableError(RuntimeError):
    """The configured or SDK-bundled Claude CLI cannot be executed."""


@dataclass(frozen=True)
class ClaudeCliExecutable:
    """A validated executable and its non-sensitive provenance."""

    path: Path
    source: ClaudeCliSource


def bundled_claude_cli_path() -> Path | None:
    """Return the CLI shipped with the pinned SDK wheel, when present.

    The SDK itself uses this package-relative layout. Keeping the lookup here
    lets TaskChamber wrap the exact executable before the SDK can perform its
    own fallback discovery.
    """

    package_file = getattr(claude_agent_sdk, "__file__", None)
    if not package_file:
        return None
    cli_name = "claude.exe" if platform.system() == "Windows" else "claude"
    candidate = Path(package_file).resolve().parent / "_bundled" / cli_name
    return candidate if candidate.is_file() else None


def resolve_claude_cli(
    configured_path: str | None = None,
    *,
    bundled_resolver: Callable[[], Path | None] = bundled_claude_cli_path,
) -> ClaudeCliExecutable:
    """Resolve one exact CLI without falling back to ambient ``PATH``.

    A server administrator may opt into another executable with
    ``TASKCHAMBER_CLAUDE_CLI_PATH``. Otherwise the CLI bundled with the pinned
    SDK wheel is required, keeping the SDK and CLI versions deterministic.
    """

    configured = configured_path.strip() if configured_path else ""
    if configured:
        raw_path = Path(configured).expanduser()
        if not raw_path.is_absolute():
            raise ClaudeCliUnavailableError("the configured Claude CLI path is not absolute")
        source: ClaudeCliSource = "configured"
    else:
        bundled = bundled_resolver()
        if bundled is None:
            raise ClaudeCliUnavailableError("the SDK-bundled Claude CLI is unavailable")
        raw_path = bundled
        source = "bundled"

    try:
        path = raw_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ClaudeCliUnavailableError("the selected Claude CLI is unavailable") from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ClaudeCliUnavailableError("the selected Claude CLI is not executable")
    return ClaudeCliExecutable(path=path, source=source)


__all__ = [
    "ClaudeCliExecutable",
    "ClaudeCliUnavailableError",
    "bundled_claude_cli_path",
    "resolve_claude_cli",
]

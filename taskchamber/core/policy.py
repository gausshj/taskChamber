"""Workspace and request policy shared across all runtime adapters."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


class RequestValidationError(ValueError):
    """The caller supplied an invalid tool argument."""


class PolicyDeniedError(ValueError):
    """The request would exceed the server's configured safety boundary."""


_PROTECTED_PARTS = frozenset(
    {
        ".aws",
        ".env",
        ".gnupg",
        ".git",
        ".kube",
        ".ssh",
        ".credentials.json",
        "credentials",
        "id_ed25519",
        "id_rsa",
        "secrets",
        "taskchamber.toml",
    }
)
_EXCLUDED_DIRECTORIES = _PROTECTED_PARTS | frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "node_modules"}
)
_PROTECTED_SUFFIXES = frozenset({".jks", ".kdbx", ".key", ".p12", ".pem", ".pfx"})
_MAX_TOOL_PATTERN_CHARS = 4_000


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """An ephemeral, filtered workspace that a runtime may read."""

    root: Path
    allowed_paths: tuple[Path, ...]


@dataclass(frozen=True)
class WorkspaceGuard:
    """Resolve and validate every workspace path against one fixed root."""

    root: Path
    max_file_bytes: int
    allowed_tools: tuple[str, ...]
    allowed_paths: tuple[Path, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())
        if not self.root.is_dir():
            raise ValueError(f"workspace root is not a directory: {self.root}")
        allowed = self.allowed_paths or (self.root,)
        normalized = tuple(self._normalize_allowed_path(path) for path in allowed)
        object.__setattr__(self, "allowed_paths", normalized)

    def resolve_file(self, value: str) -> Path:
        """Return a regular file within ``root`` or reject the request."""

        path = self._resolve_path(value)
        if not path.exists():
            raise RequestValidationError("file_path does not exist")
        if not path.is_file():
            raise RequestValidationError("file_path must identify a regular file")
        if path.stat().st_size > self.max_file_bytes:
            raise PolicyDeniedError("file_path exceeds the configured size limit")
        return path

    def relative_file(self, value: str) -> Path:
        """Return a validated path relative to the workspace root."""

        return self.resolve_file(value).relative_to(self.root)

    def validate_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """Enforce the runtime's read-only workspace boundary before tool use."""

        if tool_name not in self.allowed_tools:
            raise PolicyDeniedError(f"tool {tool_name!r} is not available")

        if tool_name == "Read":
            value = tool_input.get("file_path")
            if not isinstance(value, str) or not value:
                raise PolicyDeniedError("Read requires a file_path")
            self.resolve_file(value)
            return

        if tool_name == "Glob":
            self._validate_optional_tool_path(tool_input)
            self._validate_path_pattern(tool_input.get("pattern"), field="Glob pattern")
            return

        if tool_name == "Grep":
            self._validate_optional_tool_path(tool_input)
            self._validate_content_pattern(tool_input.get("pattern"))
            glob = tool_input.get("glob")
            if glob is not None:
                self._validate_path_pattern(glob, field="Grep glob")
            return

        raise PolicyDeniedError(f"tool {tool_name!r} has no workspace policy adapter")

    def _validate_optional_tool_path(self, tool_input: dict[str, Any]) -> None:
        path = tool_input.get("path")
        if path is None:
            return
        if not isinstance(path, str) or not path:
            raise PolicyDeniedError("tool path must be a non-empty string")
        self._resolve_path(path)

    @staticmethod
    def _validate_content_pattern(value: object) -> None:
        """Validate a Grep content regex without treating it as a file path."""

        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or len(value) > _MAX_TOOL_PATTERN_CHARS
        ):
            raise PolicyDeniedError("Grep pattern must be a bounded non-empty string")

    @classmethod
    def _validate_path_pattern(cls, value: object, *, field: str) -> None:
        """Reject escaping or explicitly protected Glob-style file patterns.

        Broad patterns such as ``**/*`` remain useful because the staged tree is
        already filtered. Explicit requests for protected names and suffixes are
        denied here as a second, tool-call-level boundary.
        """

        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or len(value) > _MAX_TOOL_PATTERN_CHARS
        ):
            raise PolicyDeniedError(f"{field} must be a bounded non-empty string")
        if cls._has_path_escape(value):
            raise PolicyDeniedError(f"{field} may not escape the workspace")
        if cls._pattern_explicitly_targets_protected_path(value):
            raise PolicyDeniedError(f"{field} targets a protected workspace path")

    @staticmethod
    def _pattern_explicitly_targets_protected_path(value: str) -> bool:
        normalized = value.casefold().replace("\\", "/")
        for part in (item for item in normalized.split("/") if item):
            if part.startswith(".env."):
                return True
            for protected in _PROTECTED_PARTS:
                if protected in part and fnmatchcase(protected, part):
                    return True
            for suffix in _PROTECTED_SUFFIXES:
                index = part.rfind(suffix)
                if index < 0:
                    continue
                remainder = part[index + len(suffix) :]
                if not remainder or remainder[0] in "*?[":
                    return True
        return False

    def _resolve_path(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PolicyDeniedError("path is outside the configured workspace") from exc
        self._assert_allowed(resolved)
        self._assert_not_protected(resolved)
        return resolved

    def _normalize_allowed_path(self, value: Path) -> Path:
        candidate = value.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("allowed paths must remain inside the workspace") from exc
        return resolved

    def _assert_allowed(self, path: Path) -> None:
        if any(self._is_within(path, allowed) for allowed in self.allowed_paths or ()):
            return
        raise PolicyDeniedError("path is outside this task's allowed workspace area")

    def _assert_not_protected(self, path: Path) -> None:
        relative = path.relative_to(self.root)
        if self.is_protected_relative(relative):
            raise PolicyDeniedError("path is excluded by the workspace safety policy")

    @classmethod
    def is_protected_relative(cls, relative: Path) -> bool:
        parts = tuple(part.casefold() for part in relative.parts)
        if any(
            part in _PROTECTED_PARTS or part == ".env" or part.startswith(".env.") for part in parts
        ):
            return True
        return relative.suffix.casefold() in _PROTECTED_SUFFIXES

    @staticmethod
    def _is_within(path: Path, allowed: Path) -> bool:
        try:
            path.relative_to(allowed)
        except ValueError:
            return False
        return True

    @staticmethod
    def _has_path_escape(value: str) -> bool:
        path = Path(value)
        return value.startswith("~") or path.is_absolute() or ".." in path.parts


@contextmanager
def staged_workspace(
    source_root: Path,
    *,
    source_paths: tuple[Path, ...],
    max_file_bytes: int,
    max_total_bytes: int,
    max_files: int,
) -> Iterator[WorkspaceSnapshot]:
    """Yield a filtered temporary copy instead of exposing the source workspace.

    The copy preserves workspace-relative paths, excludes common secret-bearing
    locations, skips symlinks, and applies file and aggregate size limits.
    """

    root = source_root.expanduser().resolve()
    guard = WorkspaceGuard(root, max_file_bytes=max_file_bytes, allowed_tools=())
    selections = tuple(guard._resolve_path(str(path)) for path in source_paths)
    with TemporaryDirectory(prefix="taskchamber-workspace-") as directory:
        snapshot_root = Path(directory)
        copied_files = 0
        copied_bytes = 0

        def copy_file(source: Path, *, required: bool) -> Path | None:
            nonlocal copied_files, copied_bytes
            if source.is_symlink() or not source.is_file():
                if required:
                    raise PolicyDeniedError("requested file cannot be staged safely")
                return None
            relative = source.relative_to(root)
            if WorkspaceGuard.is_protected_relative(relative):
                if required:
                    raise PolicyDeniedError("requested file is excluded by workspace policy")
                return None
            size = source.stat().st_size
            if size > max_file_bytes:
                if required:
                    raise PolicyDeniedError("requested file exceeds the configured size limit")
                return None
            if copied_files + 1 > max_files or copied_bytes + size > max_total_bytes:
                raise PolicyDeniedError("workspace snapshot exceeds the configured limit")
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied_files += 1
            copied_bytes += size
            return destination

        allowed_snapshot_paths: list[Path] = []
        for selection in selections:
            if selection.is_file():
                destination = copy_file(selection, required=True)
                assert destination is not None
                allowed_snapshot_paths.append(destination)
                continue
            if selection != root:
                raise PolicyDeniedError("task workspace selection must be a file or root")
            for directory_path, directory_names, file_names in os.walk(root, followlinks=False):
                current = Path(directory_path)
                directory_names[:] = [
                    name
                    for name in directory_names
                    if name.casefold() not in _EXCLUDED_DIRECTORIES
                    and not (current / name).is_symlink()
                ]
                for file_name in file_names:
                    copy_file(current / file_name, required=False)
            allowed_snapshot_paths.append(snapshot_root)

        # The snapshot contains only the selected, filtered files. Allowing the
        # runtime to address the snapshot root lets Glob/Grep work across a
        # caller-selected file set without granting access to unselected source
        # files, because those files were never copied into the task workspace.
        if allowed_snapshot_paths:
            allowed_snapshot_paths = [snapshot_root]

        yield WorkspaceSnapshot(
            root=snapshot_root,
            allowed_paths=tuple(allowed_snapshot_paths),
        )


def validate_text(value: str, *, field: str, maximum: int = 4_000) -> str:
    """Reject empty or oversized free-text MCP inputs before invoking a model."""

    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{field} must not be empty")
    if len(value) > maximum:
        raise RequestValidationError(f"{field} exceeds the {maximum}-character limit")
    return value.strip()

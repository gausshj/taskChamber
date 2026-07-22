"""Root-anchored glob matching for provider-neutral relative paths."""

from __future__ import annotations

from fnmatch import fnmatchcase
from functools import cache


def path_glob_matches(relative_path: str, pattern: str) -> bool:
    """Return whether a safe POSIX relative path matches a rooted glob.

    Normal glob tokens match within one path segment. A segment that is exactly
    ``**`` may match zero or more complete segments. Matching consumes the full
    path and pattern, so a pattern such as ``src/**/*.py`` cannot match a path
    rooted below ``other/src``.
    """

    path_parts = _relative_parts(relative_path)
    pattern_parts = _relative_parts(pattern)
    if path_parts is None or pattern_parts is None:
        return False

    @cache
    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)

        token = pattern_parts[pattern_index]
        if token == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )

        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], token)
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)


def _relative_parts(value: str) -> tuple[str, ...] | None:
    if not value or value.startswith(("/", "~")) or "\\" in value or "\x00" in value:
        return None
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return parts


__all__ = ["path_glob_matches"]

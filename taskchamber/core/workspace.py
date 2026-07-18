"""Resolve caller-selected relative paths below a project workspace ceiling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from .capabilities import WorkspaceAccessPolicy
from .path_globs import path_glob_matches
from .policy import PolicyDeniedError, RequestValidationError, WorkspaceGuard

MAX_SELECTED_FILES = 10_000


@dataclass(frozen=True)
class WorkspaceSelector:
    """Expand paths and globs without exposing files outside project policy."""

    root: Path
    policy: WorkspaceAccessPolicy
    max_file_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def resolve(
        self,
        values: Sequence[str] | None,
        *,
        include_default: bool,
    ) -> tuple[Path, ...]:
        if values is None:
            if not include_default:
                return ()
            if self.policy.include == ("**/*",) and not self.policy.exclude:
                return (self.root,)
            values = self.policy.include
        if isinstance(values, (str, bytes)):
            raise RequestValidationError("workspace_paths must be a list")
        if len(values) > self.policy.max_requested_paths:
            raise RequestValidationError(
                "workspace_paths exceeds the configured request limit of "
                f"{self.policy.max_requested_paths}"
            )

        selected: dict[Path, None] = {}
        for raw in values:
            if not isinstance(raw, str) or not raw.strip():
                raise RequestValidationError("workspace_paths must contain non-empty strings")
            value = raw.strip()
            self._validate_selector(value)
            has_glob = any(character in value for character in "*?[")
            if has_glob and not self.policy.allow_globs:
                raise PolicyDeniedError("workspace glob selection is disabled by project policy")
            candidates = self._candidates(value, has_glob=has_glob)
            matched = 0
            had_existing_match = False
            for candidate in candidates:
                had_existing_match = had_existing_match or candidate.exists()
                if not self._file_is_allowed(candidate):
                    continue
                selected[candidate] = None
                matched += 1
                if len(selected) > MAX_SELECTED_FILES:
                    raise PolicyDeniedError(
                        f"workspace selection exceeds the {MAX_SELECTED_FILES}-file limit"
                    )
            if matched == 0:
                if had_existing_match:
                    raise PolicyDeniedError(
                        f"workspace selection {value!r} is outside the project workspace policy"
                    )
                suggestions = self._suggestions(value)
                suffix = f"; suggestions: {', '.join(suggestions)}" if suggestions else ""
                raise RequestValidationError(
                    f"workspace selection {value!r} matched no allowed files{suffix}. "
                    "Retry with a suggested relative path and do not fall back to shell execution."
                )
        return tuple(sorted(selected, key=lambda path: path.as_posix()))

    def _candidates(self, value: str, *, has_glob: bool) -> tuple[Path, ...]:
        if has_glob:
            try:
                return tuple(self.root.glob(value))
            except (OSError, ValueError) as exc:
                raise RequestValidationError(
                    f"workspace selection {value!r} is not a valid glob"
                ) from exc
        guard = WorkspaceGuard(
            root=self.root,
            max_file_bytes=self.max_file_bytes,
            allowed_tools=(),
        )
        candidate = (self.root / value).resolve(strict=False)
        if candidate.is_dir() and not candidate.is_symlink():
            return tuple(candidate.rglob("*"))
        if candidate.exists():
            return (guard.resolve_file(value),)
        return (candidate,)

    def _file_is_allowed(self, path: Path) -> bool:
        if path.is_symlink() or not path.is_file():
            return False
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(self.root)
        except (OSError, ValueError):
            return False
        if WorkspaceGuard.is_protected_relative(relative):
            return False
        try:
            if resolved.stat().st_size > self.max_file_bytes:
                raise PolicyDeniedError("workspace selection contains an oversized file")
        except OSError:
            return False
        document_id = relative.as_posix()
        if not any(path_glob_matches(document_id, pattern) for pattern in self.policy.include):
            return False
        return not any(path_glob_matches(document_id, pattern) for pattern in self.policy.exclude)

    def _suggestions(self, value: str) -> tuple[str, ...]:
        candidates: list[str] = []
        for index, path in enumerate(self.root.rglob("*")):
            if index >= MAX_SELECTED_FILES:
                break
            if self._file_is_allowed(path):
                candidates.append(path.relative_to(self.root).as_posix())
        return tuple(get_close_matches(value, candidates, n=3, cutoff=0.45))

    @staticmethod
    def _validate_selector(value: str) -> None:
        path = Path(value)
        if (
            "\x00" in value
            or len(value) > 1_000
            or value.startswith("~")
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise PolicyDeniedError(
                "workspace selections must be relative paths or globs inside the project"
            )


__all__ = ["MAX_SELECTED_FILES", "WorkspaceSelector"]

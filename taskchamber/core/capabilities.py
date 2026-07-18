"""Provider-neutral capabilities and project-owned task policy."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from types import MappingProxyType

from .contracts import TaskKind
from .policy import PolicyDeniedError, RequestValidationError

WORKSPACE_LIST = "workspace.list"
WORKSPACE_READ = "workspace.read"
WORKSPACE_SEARCH = "workspace.search"
DOCUMENTS_LIST = "documents.list"
DOCUMENTS_READ = "documents.read"
DOCUMENTS_SEARCH = "documents.search"

WORKSPACE_CAPABILITIES = (WORKSPACE_LIST, WORKSPACE_READ, WORKSPACE_SEARCH)
DOCUMENT_CAPABILITIES = (DOCUMENTS_LIST, DOCUMENTS_READ, DOCUMENTS_SEARCH)
KNOWN_CAPABILITIES = WORKSPACE_CAPABILITIES + DOCUMENT_CAPABILITIES

WORKSPACE_TOOL_BY_CAPABILITY: Mapping[str, str] = MappingProxyType(
    {
        WORKSPACE_READ: "Read",
        WORKSPACE_LIST: "Glob",
        WORKSPACE_SEARCH: "Grep",
    }
)
DOCUMENT_TOOL_BY_CAPABILITY: Mapping[str, str] = MappingProxyType(
    {
        DOCUMENTS_LIST: "DocumentList",
        DOCUMENTS_READ: "DocumentRead",
        DOCUMENTS_SEARCH: "DocumentSearch",
    }
)

_CAPABILITY_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "glob": WORKSPACE_LIST,
        "grep": WORKSPACE_SEARCH,
        "read": WORKSPACE_READ,
        "document.list": DOCUMENTS_LIST,
        "document.read": DOCUMENTS_READ,
        "document.search": DOCUMENTS_SEARCH,
        "documents.list_documents": DOCUMENTS_LIST,
        "documents.read_document": DOCUMENTS_READ,
        "documents.search_documents": DOCUMENTS_SEARCH,
    }
)
_SEPARATORS = re.compile(r"[\s_-]+")


@dataclass(frozen=True)
class WorkspaceAccessPolicy:
    """The project-configured ceiling for caller-selected workspace files."""

    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    allow_globs: bool = True
    max_requested_paths: int = 64


@dataclass(frozen=True)
class TaskCapabilityPolicy:
    """Capabilities one public task may use below the project ceiling."""

    allowed: tuple[str, ...]
    defaults: tuple[str, ...]
    max_turns: int

    def __post_init__(self) -> None:
        if not set(self.defaults).issubset(self.allowed):
            raise ValueError("task default capabilities must be allowed by that task")
        if self.max_turns < 1:
            raise ValueError("task max_turns must be positive")


@dataclass(frozen=True)
class ProjectPolicy:
    """Maximum project authority plus caller-facing task defaults."""

    allowed_capabilities: tuple[str, ...]
    default_capabilities: tuple[str, ...]
    workspace: WorkspaceAccessPolicy
    tasks: Mapping[TaskKind, TaskCapabilityPolicy]
    max_document_sources: int = 16

    def __post_init__(self) -> None:
        known = set(KNOWN_CAPABILITIES)
        if not set(self.allowed_capabilities).issubset(known):
            raise ValueError("project policy contains an unknown capability")
        if not set(self.default_capabilities).issubset(self.allowed_capabilities):
            raise ValueError("project default capabilities must be allowed")
        if self.max_document_sources < 1:
            raise ValueError("max_document_sources must be positive")
        for kind in TaskKind:
            if kind not in self.tasks:
                raise ValueError(f"project policy is missing task {kind.value!r}")
            task = self.tasks[kind]
            if not set(task.allowed).issubset(self.allowed_capabilities):
                raise ValueError(f"task {kind.value!r} exceeds the project capability ceiling")
        object.__setattr__(self, "tasks", MappingProxyType(dict(self.tasks)))

    def effective_capabilities(
        self,
        kind: TaskKind,
        requested: Sequence[str] | None,
    ) -> tuple[str, ...]:
        """Return a normalized caller-selected subset of one task's allowance."""

        task = self.tasks[kind]
        if requested is None:
            return task.defaults
        if isinstance(requested, (str, bytes)):
            raise RequestValidationError("requested_capabilities must be a list of names")
        if len(requested) > len(KNOWN_CAPABILITIES):
            raise RequestValidationError("too many requested_capabilities were supplied")

        selected: list[str] = []
        for raw in requested:
            capability = normalize_capability(raw)
            if capability not in task.allowed:
                allowed = ", ".join(task.allowed) or "none"
                raise PolicyDeniedError(
                    f"capability {capability!r} is outside the {kind.value} policy; "
                    f"allowed values: {allowed}"
                )
            if capability not in selected:
                selected.append(capability)
        return tuple(selected)


def default_project_policy() -> ProjectPolicy:
    """Return broad read-only defaults that projects may narrow in TOML."""

    all_read = KNOWN_CAPABILITIES
    return ProjectPolicy(
        allowed_capabilities=all_read,
        default_capabilities=WORKSPACE_CAPABILITIES,
        workspace=WorkspaceAccessPolicy(),
        tasks={
            TaskKind.RESEARCH: TaskCapabilityPolicy(
                allowed=all_read,
                defaults=all_read,
                max_turns=25,
            ),
            TaskKind.SUMMARIZE: TaskCapabilityPolicy(
                allowed=(
                    WORKSPACE_READ,
                    WORKSPACE_SEARCH,
                    DOCUMENTS_LIST,
                    DOCUMENTS_READ,
                    DOCUMENTS_SEARCH,
                ),
                defaults=(
                    WORKSPACE_READ,
                    DOCUMENTS_LIST,
                    DOCUMENTS_READ,
                    DOCUMENTS_SEARCH,
                ),
                max_turns=15,
            ),
            TaskKind.REVIEW: TaskCapabilityPolicy(
                allowed=all_read,
                defaults=WORKSPACE_CAPABILITIES,
                max_turns=20,
            ),
        },
    )


def normalize_capability(value: object) -> str:
    """Normalize safe aliases and suggest canonical capabilities on mistakes."""

    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError("requested_capabilities must contain non-empty names")
    raw = value.strip().casefold()
    normalized = _SEPARATORS.sub(".", raw).strip(".")
    canonical = _CAPABILITY_ALIASES.get(normalized, normalized)
    if canonical in KNOWN_CAPABILITIES:
        return canonical
    suggestions = get_close_matches(canonical, KNOWN_CAPABILITIES, n=3, cutoff=0.45)
    suffix = f"; suggestions: {', '.join(suggestions)}" if suggestions else ""
    raise RequestValidationError(
        f"unknown capability {value!r}{suffix}; allowed values: "
        f"{', '.join(KNOWN_CAPABILITIES)}. Retry with a listed value and do not "
        "fall back to shell execution."
    )


def workspace_tools(capabilities: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        tool
        for capability, tool in WORKSPACE_TOOL_BY_CAPABILITY.items()
        if capability in capabilities
    )


def document_tools(capabilities: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        tool
        for capability, tool in DOCUMENT_TOOL_BY_CAPABILITY.items()
        if capability in capabilities
    )


__all__ = [
    "DOCUMENTS_LIST",
    "DOCUMENTS_READ",
    "DOCUMENTS_SEARCH",
    "DOCUMENT_CAPABILITIES",
    "DOCUMENT_TOOL_BY_CAPABILITY",
    "KNOWN_CAPABILITIES",
    "ProjectPolicy",
    "TaskCapabilityPolicy",
    "WORKSPACE_CAPABILITIES",
    "WORKSPACE_LIST",
    "WORKSPACE_READ",
    "WORKSPACE_SEARCH",
    "WORKSPACE_TOOL_BY_CAPABILITY",
    "WorkspaceAccessPolicy",
    "default_project_policy",
    "document_tools",
    "normalize_capability",
    "workspace_tools",
]

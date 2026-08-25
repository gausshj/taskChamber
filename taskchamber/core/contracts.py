"""Provider-neutral contracts shared by MCP and runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, computed_field

from .documents import DocumentCatalog


class TaskKind(StrEnum):
    """The stable task-shaped capabilities exposed through MCP."""

    RESEARCH = "research"
    SUMMARIZE = "summarize"
    REVIEW = "review"


class DocumentMode(StrEnum):
    """How selected virtual documents are supplied to the runtime."""

    AGENTIC = "agentic"
    SINGLE_PASS = "single_pass"


class TaskStatus(StrEnum):
    """Normalized outcomes that do not leak a runtime's internal errors."""

    SUCCESS = "success"
    INVALID_REQUEST = "invalid_request"
    POLICY_DENIED = "policy_denied"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    TURN_LIMIT_EXCEEDED = "turn_limit_exceeded"
    FAILED = "failed"


class ToolCallDecision(StrEnum):
    """Outcome of one runtime tool call observed by the policy hook."""

    ALLOWED = "allowed"
    DENIED = "denied"


class AgentCapabilities(BaseModel, frozen=True):
    """Capabilities an adapter can prove to the orchestration layer."""

    read_workspace: bool = False
    read_documents: bool = False
    cancellation: bool = False
    progress: bool = False
    structured_output: bool = False


class TaskRequest(BaseModel, frozen=True):
    """A runtime-independent task after MCP input has been validated."""

    run_id: str = Field(min_length=1)
    kind: TaskKind
    prompt: str = Field(min_length=1)
    provider: str = Field(min_length=1, max_length=64)
    max_turns: int = Field(ge=1, le=100)


class TokenUsage(BaseModel, frozen=True):
    """Provider-reported token counts normalized across runtime adapters.

    ``None`` means the provider did not report that counter. Values are never
    estimated from text length, price tables, plans, or subscription tiers.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ToolCallRecord(BaseModel, frozen=True):
    """Sanitized audit record; tool inputs and paths are deliberately omitted."""

    tool: str = Field(min_length=1, max_length=100)
    decision: ToolCallDecision


class DocumentSourceSelection(BaseModel, frozen=True):
    """A named document source plus bounded, schema-validated string parameters."""

    source: str = Field(min_length=1, max_length=200)
    parameters: dict[str, str] = Field(default_factory=dict)


class ExecutionTelemetry(BaseModel, frozen=True):
    """Evidence about the effective policy and isolation used for one run."""

    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    sandbox: str | None = None
    workspace_staged: bool | None = None
    # ``cli_wrapper_active`` means a generated launcher was selected as the
    # SDK's cli_path. ``cli_launch_observed`` additionally requires the
    # one-shot main-call marker written inside the effective CLI boundary.
    # These are runtime observations, not kernel attestation.
    os_isolated: bool | None = None
    cli_wrapper_active: bool | None = None
    cli_launch_observed: bool | None = None
    sandbox_preflight_passed: bool | None = None
    isolation_scope: str | None = None
    runtime_process_isolated: bool | None = None
    cli_environment_sanitized: bool | None = None
    cli_executable_source: str | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    document_sources: tuple[str, ...] = ()
    document_tools: tuple[str, ...] = ()


class SinglePassDocumentTooLargeDetails(BaseModel, frozen=True):
    """Safe, typed details for a single-pass document that exceeded the byte limit.

    Only public virtual identifiers and server-owned byte counts are carried.
    Host paths, argv, environment values, credentials, and command stderr are
    never included. Clients must not parse ``error_message`` for these values.
    """

    type: Literal["single_pass_document_too_large"] = "single_pass_document_too_large"
    document_mode: Literal["single_pass"] = "single_pass"
    source: str = Field(min_length=1, max_length=200)
    document_id: str = Field(min_length=1, max_length=1_000)
    observed_utf8_bytes: int = Field(ge=0)
    effective_limit_bytes: int = Field(ge=1)
    absolute_limit_bytes: int = Field(ge=1)
    retryable: Literal[False] = False


class TaskResult(BaseModel, frozen=True):
    """A safe, structured result returned by every runtime adapter."""

    run_id: str
    kind: TaskKind
    status: TaskStatus
    output: str = ""
    runtime: str | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: int = Field(default=0, ge=0)
    num_turns: int | None = Field(default=None, ge=0)
    usage: TokenUsage | None = None
    model_usage: dict[str, TokenUsage] | None = None
    execution: ExecutionTelemetry | None = None
    # Provider-reported reference only; pricing plans and subscriptions differ.
    cost_usd: float | None = Field(default=None, ge=0)
    partial: bool = False
    truncated: bool = False
    effective_max_output_chars: int | None = Field(default=None, ge=1)
    error_code: str | None = None
    error_message: str | None = None
    error_details: SinglePassDocumentTooLargeDetails | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_error(self) -> bool:
        """Whether the normalized task status represents a failed call."""

        return self.status is not TaskStatus.SUCCESS


@dataclass(frozen=True)
class ExecutionPolicy:
    """Non-negotiable execution limits handed to a runtime adapter."""

    workspace_root: Path
    allowed_paths: tuple[Path, ...]
    system_prompt: str
    allowed_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...]
    max_turns: int
    # None disables monetary enforcement for the runtime adapter.
    max_budget_usd: float | None
    timeout_seconds: float
    max_output_chars: int
    max_file_bytes: int
    document_catalog: DocumentCatalog | None = None
    document_sources: tuple[str, ...] = ()
    document_tools: tuple[str, ...] = ()


@runtime_checkable
class AgentRuntime(Protocol):
    """Port implemented by Claude, Codex, API, or fake agent runtimes."""

    name: str
    default_profile: str
    capabilities: AgentCapabilities

    async def run(self, request: TaskRequest, policy: ExecutionPolicy) -> TaskResult:
        """Execute one fresh task without exposing runtime-specific state."""

        ...

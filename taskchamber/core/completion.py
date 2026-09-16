"""Provider-neutral structured completion contracts and orchestration service.

A structured completion is a tool-free task: the caller supplies a system
instruction, task text, and a JSON Schema, and receives parsed structured
output plus auditable invocation metadata. It is an optional runtime
capability, separate from the research/summarize/review task pipeline and
from the MCP transport; runtimes that cannot honor a schema must fail
explicitly instead of returning unconstrained prose.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from .contracts import AgentRuntime, ExecutionPolicy, TaskStatus, TokenUsage
from .service import ServerSettings

DEFAULT_COMPLETION_MAX_TURNS = 10
"""Host ceiling for one completion; SDK-internal retries may use several."""


class StructuredCompletionRequest(BaseModel, frozen=True):
    """One caller-supplied structured completion, free of any task preset."""

    request_id: str = Field(min_length=1, max_length=128)
    system_prompt: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    json_schema: dict[str, Any]
    provider: str | None = None
    max_turns: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_output_chars: int | None = Field(default=None, ge=1)

    @field_validator("json_schema")
    @classmethod
    def _schema_must_not_be_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("json_schema must not be empty")
        return value


class StructuredCompletionResult(BaseModel, frozen=True):
    """The auditable outcome of one structured completion invocation."""

    request_id: str
    status: TaskStatus
    runtime: str
    provider: str
    model: str | None = None
    # Parsed structured output exactly as reported by the runtime; null on
    # failure. Never a reserialization labeled as original model text.
    output: Any = None
    # The raw result text envelope when available; bounded by max_output_chars.
    raw_result_text: str | None = None
    usage: TokenUsage | None = None
    model_usage: dict[str, TokenUsage] | None = None
    # Observable invocation shape: SDK-internal turns/retries, not one request.
    num_turns: int | None = None
    # Provider-reported reference only; see ServerSettings.max_budget_usd.
    cost_usd: float | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    partial: bool = False
    truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None
    sdk_version: str | None = None
    # Host-bounded limits actually applied, stamped by the service so the
    # invocation stays auditable regardless of the runtime adapter.
    effective_max_turns: int | None = None
    effective_timeout_seconds: float | None = None
    effective_max_output_chars: int | None = None


@runtime_checkable
class StructuredCompletionRuntime(Protocol):
    """Optional runtime capability for tool-free structured completion."""

    async def complete_structured(
        self,
        request: StructuredCompletionRequest,
        policy: ExecutionPolicy,
    ) -> StructuredCompletionResult:
        """Execute one fresh completion honoring the request's JSON Schema."""

        ...


class StructuredCompletionService:
    """Validate requests, apply host ceilings, and invoke one capable runtime."""

    def __init__(self, runtime: AgentRuntime, settings: ServerSettings) -> None:
        self.runtime = runtime
        self.settings = settings

    async def complete(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        provider = request.provider or self.runtime.default_profile
        runtime = self.runtime
        if not runtime.capabilities.structured_output or not isinstance(
            runtime, StructuredCompletionRuntime
        ):
            return StructuredCompletionResult(
                request_id=request.request_id,
                status=TaskStatus.FAILED,
                runtime=runtime.name,
                provider=provider,
                error_code="structured_output_unsupported",
                error_message=("The selected runtime does not support structured completion."),
            )
        policy = ExecutionPolicy(
            workspace_root=self.settings.workspace_root,
            allowed_paths=(),
            system_prompt=request.system_prompt,
            allowed_tools=(),
            disallowed_tools=(),
            max_turns=self._narrow_turns(request.max_turns),
            max_budget_usd=self.settings.max_budget_usd,
            timeout_seconds=self._narrow_timeout(request.timeout_seconds),
            max_output_chars=self._narrow_output_chars(request.max_output_chars),
            max_file_bytes=self.settings.max_file_bytes,
        )
        result = await runtime.complete_structured(request, policy)
        return result.model_copy(
            update={
                "effective_max_turns": policy.max_turns,
                "effective_timeout_seconds": policy.timeout_seconds,
                "effective_max_output_chars": policy.max_output_chars,
            }
        )

    @staticmethod
    def _narrow_turns(requested: int | None) -> int:
        if requested is None:
            return DEFAULT_COMPLETION_MAX_TURNS
        return min(requested, DEFAULT_COMPLETION_MAX_TURNS)

    def _narrow_timeout(self, requested: float | None) -> float:
        if requested is None:
            return self.settings.timeout_seconds
        return min(requested, self.settings.timeout_seconds)

    def _narrow_output_chars(self, requested: int | None) -> int:
        if requested is None:
            return self.settings.max_output_chars
        return min(requested, self.settings.max_output_chars)


__all__ = [
    "DEFAULT_COMPLETION_MAX_TURNS",
    "StructuredCompletionRequest",
    "StructuredCompletionResult",
    "StructuredCompletionRuntime",
    "StructuredCompletionService",
]

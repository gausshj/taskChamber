"""A deterministic runtime for contract tests and safe MCP demonstrations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ...core.completion import (
    StructuredCompletionRequest,
    StructuredCompletionResult,
)
from ...core.contracts import (
    AgentCapabilities,
    ExecutionPolicy,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

FakeHandler = Callable[[TaskRequest, ExecutionPolicy], Awaitable[TaskResult]]
FakeCompletionHandler = Callable[
    [StructuredCompletionRequest, ExecutionPolicy],
    Awaitable[StructuredCompletionResult],
]


class FakeRuntime:
    """Return deterministic results without spawning a CLI or contacting a model."""

    name = "fake"
    default_profile = "fake"
    capabilities = AgentCapabilities(
        read_workspace=True,
        read_documents=True,
        cancellation=True,
        progress=False,
        structured_output=True,
    )

    def __init__(
        self,
        handler: FakeHandler | None = None,
        completion_handler: FakeCompletionHandler | None = None,
    ) -> None:
        self._handler = handler
        self._completion_handler = completion_handler
        self.requests: list[TaskRequest] = []
        self.completion_requests: list[StructuredCompletionRequest] = []
        self.policies: list[ExecutionPolicy] = []

    async def run(self, request: TaskRequest, policy: ExecutionPolicy) -> TaskResult:
        self.requests.append(request)
        self.policies.append(policy)
        if self._handler is not None:
            return await self._handler(request, policy)
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=f"fake {request.kind.value} result",
            runtime=self.name,
            provider=request.provider,
            model="fake-runtime",
        )

    async def complete_structured(
        self,
        request: StructuredCompletionRequest,
        policy: ExecutionPolicy,
    ) -> StructuredCompletionResult:
        self.completion_requests.append(request)
        self.policies.append(policy)
        if self._completion_handler is not None:
            return await self._completion_handler(request, policy)
        return StructuredCompletionResult(
            request_id=request.request_id,
            status=TaskStatus.SUCCESS,
            runtime=self.name,
            provider=request.provider or self.default_profile,
            model="fake-runtime",
            output={"fake": True},
            raw_result_text='{"fake": true}',
        )

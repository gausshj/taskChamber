"""A deterministic runtime for contract tests and safe MCP demonstrations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ...core.contracts import (
    AgentCapabilities,
    ExecutionPolicy,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

FakeHandler = Callable[[TaskRequest, ExecutionPolicy], Awaitable[TaskResult]]


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

    def __init__(self, handler: FakeHandler | None = None) -> None:
        self._handler = handler
        self.requests: list[TaskRequest] = []
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

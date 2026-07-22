"""Factory entry point for the fake runtime adapter."""

from ...core.contracts import AgentRuntime
from ..registry import RuntimeFactoryContext


def create_runtime(_context: RuntimeFactoryContext) -> AgentRuntime:
    from .runtime import FakeRuntime

    return FakeRuntime()


__all__ = ["create_runtime"]

"""A provider-neutral MCP adapter for isolated agent runtimes."""

from .config import ProviderProfile, SecretProvider
from .core.contracts import (
    AgentCapabilities,
    AgentRuntime,
    DocumentMode,
    ExecutionPolicy,
    ExecutionTelemetry,
    TaskKind,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TokenUsage,
    ToolCallDecision,
    ToolCallRecord,
)
from .runtimes.registry import RuntimeFactoryContext, RuntimeRegistry

__all__ = [
    "AgentCapabilities",
    "AgentRuntime",
    "DocumentMode",
    "ExecutionTelemetry",
    "ExecutionPolicy",
    "ProviderProfile",
    "RuntimeFactoryContext",
    "RuntimeRegistry",
    "SecretProvider",
    "TaskKind",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    "TokenUsage",
    "ToolCallDecision",
    "ToolCallRecord",
]

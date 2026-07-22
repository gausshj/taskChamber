"""Claude Agent SDK runtime adapter."""

from typing import TYPE_CHECKING

from .profiles import DEFAULT_PROVIDER, DEFAULT_PROVIDER_PROFILES, PROVIDERS

if TYPE_CHECKING:
    from .runtime import ClaudeAgentSdkRuntime, ProviderSelectionError

__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_PROVIDER_PROFILES",
    "PROVIDERS",
    "ClaudeAgentSdkRuntime",
    "ProviderSelectionError",
]


def __getattr__(name: str) -> object:
    if name in {"ClaudeAgentSdkRuntime", "ProviderSelectionError"}:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(name)

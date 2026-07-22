"""Declarative lazy registration of runtime adapters bundled with this package."""

from types import MappingProxyType

from .registry import RuntimeRegistry

BUILTIN_RUNTIME_TARGETS = MappingProxyType(
    {
        "fake": "taskchamber.runtimes.fake.factory:create_runtime",
        "claude": "taskchamber.runtimes.claude.factory:create_runtime",
    }
)


def register_builtin_runtimes(registry: RuntimeRegistry) -> None:
    """Register bundled adapters without importing their implementation modules."""

    for name, target in BUILTIN_RUNTIME_TARGETS.items():
        registry.register_lazy(name, target)


__all__ = ["BUILTIN_RUNTIME_TARGETS", "register_builtin_runtimes"]

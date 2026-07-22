"""Lazy runtime factories and optional third-party runtime discovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import TypeAlias, cast

from ..config import ConfigurationBundle
from ..core.contracts import AgentRuntime
from ..isolation import Sandbox

RUNTIME_ENTRY_POINT_GROUP = "taskchamber.runtimes"


@dataclass(frozen=True)
class RuntimeFactoryContext:
    """Vendor-neutral dependencies supplied to an installed runtime factory."""

    configuration: ConfigurationBundle
    sandbox: Sandbox


RuntimeFactory: TypeAlias = Callable[[RuntimeFactoryContext], AgentRuntime]


class RuntimeRegistry:
    """Resolve built-in or installed runtime factories without eager SDK imports."""

    def __init__(self) -> None:
        self._factories: dict[str, RuntimeFactory] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def register(self, name: str, factory: RuntimeFactory) -> None:
        normalized = _runtime_name(name)
        if normalized in self._factories:
            raise ValueError(f"runtime {normalized!r} is already registered")
        self._factories[normalized] = factory

    def register_lazy(self, name: str, target: str) -> None:
        """Register ``module:factory`` without importing the adapter SDK yet."""

        module_name, separator, attribute = target.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("lazy runtime target must use 'module:factory' syntax")

        def load(context: RuntimeFactoryContext) -> AgentRuntime:
            loaded = getattr(import_module(module_name), attribute)
            if not callable(loaded):
                raise TypeError(f"runtime factory target {target!r} is not callable")
            factory = cast(RuntimeFactory, loaded)
            return factory(context)

        self.register(name, load)

    def discover(self) -> None:
        """Load factories installed under the documented Python entry-point group."""

        for entry_point in metadata.entry_points(group=RUNTIME_ENTRY_POINT_GROUP):
            loaded = entry_point.load()
            if not callable(loaded):
                raise TypeError(f"runtime entry point {entry_point.name!r} is not callable")
            self.register(entry_point.name, loaded)

    def create(self, name: str, context: RuntimeFactoryContext) -> AgentRuntime:
        normalized = _runtime_name(name)
        try:
            factory = self._factories[normalized]
        except KeyError as exc:
            available = ", ".join(self.names) or "none"
            raise ValueError(
                f"unsupported TASKCHAMBER_RUNTIME {normalized!r}; available runtimes: {available}"
            ) from exc
        return factory(context)


def _runtime_name(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("runtime name must not be empty")
    return normalized

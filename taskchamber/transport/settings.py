"""Transport-owned MCP response envelope settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import ConfigurationBundle

TEXT_MODE_VARIABLE = "TASKCHAMBER_MCP_TEXT_MODE"


class MCPTextMode(str, Enum):
    """How much of a successful result is repeated in the text content."""

    FULL = "full"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class MCPTransportSettings:
    """Server-side response envelope policy; never caller-controlled per call."""

    text_mode: MCPTextMode = MCPTextMode.FULL

    @classmethod
    def from_configuration(cls, configuration: ConfigurationBundle) -> MCPTransportSettings:
        raw = configuration.values.get(TEXT_MODE_VARIABLE)
        if raw is None or not raw.strip():
            return cls()
        normalized = raw.strip().lower()
        try:
            return cls(text_mode=MCPTextMode(normalized))
        except ValueError:
            allowed = ", ".join(mode.value for mode in MCPTextMode)
            raise ValueError(
                f"{TEXT_MODE_VARIABLE} must be one of: {allowed} (got {raw!r})"
            ) from None


__all__ = ["MCPTextMode", "MCPTransportSettings", "TEXT_MODE_VARIABLE"]

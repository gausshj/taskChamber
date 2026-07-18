"""MCP transport adapters over the provider-neutral task service."""

from .mcp import create_server, get_default_server, main

__all__ = ["create_server", "get_default_server", "main"]

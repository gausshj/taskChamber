"""Source-tree entry point for the TaskChamber stdio MCP server.

Installed environments should normally use the ``taskchamber`` console command
or ``python -m taskchamber``. This wrapper is convenient for repository-local
experiments and keeps transport startup separate from implementation modules.
"""

from typing import TYPE_CHECKING

from taskchamber.transport.mcp import get_default_server, main

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from taskchamber.config import ProviderProfile

    DEFAULT_PROVIDER: str
    PROVIDERS: dict[str, ProviderProfile]
    mcp: FastMCP

__all__ = ["DEFAULT_PROVIDER", "PROVIDERS", "ProviderProfile", "main", "mcp"]


def __getattr__(name: str) -> object:
    if name == "mcp":
        return get_default_server()
    if name in {"DEFAULT_PROVIDER", "PROVIDERS", "ProviderProfile"}:
        from taskchamber.runtimes import claude as claude_runtime

        return getattr(claude_runtime, name)
    raise AttributeError(name)


if __name__ == "__main__":
    main()

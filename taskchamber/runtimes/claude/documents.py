"""Claude SDK bridge for the provider-neutral virtual document catalog."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from mcp.types import ToolAnnotations

from ...core.documents import DocumentCatalog, DocumentError

DOCUMENT_MCP_SERVER = "documents"
CLAUDE_DOCUMENT_TOOL_NAMES = (
    "mcp__documents__list_documents",
    "mcp__documents__read_document",
    "mcp__documents__search_documents",
)
CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY = {
    "DocumentList": CLAUDE_DOCUMENT_TOOL_NAMES[0],
    "DocumentRead": CLAUDE_DOCUMENT_TOOL_NAMES[1],
    "DocumentSearch": CLAUDE_DOCUMENT_TOOL_NAMES[2],
}

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)


def create_document_mcp_server(catalog: DocumentCatalog) -> McpSdkServerConfig:
    """Expose one task-scoped catalog as in-process, read-only SDK tools."""

    @tool(
        "list_documents",
        "List metadata for virtual documents from selected server-configured sources.",
        {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "additionalProperties": False,
        },
        annotations=_READ_ONLY,
    )
    async def list_documents(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            documents = await catalog.list_documents(
                source=arguments.get("source"),
                pattern=arguments.get("pattern"),
                limit=arguments.get("limit", 100),
            )
            return _content([document.as_dict() for document in documents])
        except DocumentError as exc:
            return _error(exc)

    @tool(
        "read_document",
        "Read a bounded line range from one virtual document by source and document ID.",
        {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "document_id": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["source", "document_id"],
            "additionalProperties": False,
        },
        annotations=_READ_ONLY,
    )
    async def read_document(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            page = await catalog.read_document(
                source=arguments["source"],
                document_id=arguments["document_id"],
                start_line=arguments.get("start_line", 1),
                max_lines=arguments.get("max_lines", 200),
            )
            return _content(page.as_dict())
        except (DocumentError, KeyError) as exc:
            return _error(exc)

    @tool(
        "search_documents",
        "Search virtual documents and return bounded line-level matches.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source": {"type": "string"},
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        annotations=_READ_ONLY,
    )
    async def search_documents(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            matches = await catalog.search_documents(
                query=arguments["query"],
                source=arguments.get("source"),
                pattern=arguments.get("pattern"),
                limit=arguments.get("limit", 50),
            )
            return _content([match.as_dict() for match in matches])
        except (DocumentError, KeyError) as exc:
            return _error(exc)

    return create_sdk_mcp_server(
        name=DOCUMENT_MCP_SERVER,
        version="1.0.0",
        tools=[list_documents, read_document, search_documents],
    )


def _content(value: object) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            }
        ]
    }


def _error(error: Exception) -> dict[str, Any]:
    message = str(error) if isinstance(error, DocumentError) else "invalid document request"
    return {
        "content": [{"type": "text", "text": message}],
        "is_error": True,
    }


__all__ = [
    "CLAUDE_DOCUMENT_TOOL_NAMES",
    "CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY",
    "DOCUMENT_MCP_SERVER",
    "create_document_mcp_server",
]

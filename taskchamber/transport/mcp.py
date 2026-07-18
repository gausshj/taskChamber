"""FastMCP transport layer for the provider-neutral task service."""

from __future__ import annotations

import json
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from ..application.composition import create_default_service, create_runtime_from_environment
from ..core.contracts import DocumentMode, DocumentSourceSelection, TaskResult, TaskStatus
from ..core.service import TaskService
from .legacy import render_legacy_result

SERVER_NAME = "taskchamber"
INSTRUCTIONS = (
    "Isolated task-shaped sub-agent tools. Each call delegates read-only "
    "research, summarization, or review to one configured agent runtime and "
    "returns a bounded result. Tasks may select named, server-configured "
    "virtual document sources and caller-narrowed workspace paths/capabilities. "
    "File paths must remain inside the server-owned "
    "workspace; providers, document commands, and credentials are selected by "
    "server policy."
)


def to_call_tool_result(result: TaskResult) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=render_legacy_result(result))],
        structuredContent=result.model_dump(mode="json"),
        isError=result.status is not TaskStatus.SUCCESS,
    )


def create_server(service: TaskService | None = None) -> FastMCP:
    """Build a server whose tools only depend on ``TaskService`` contracts."""

    task_service = service or create_default_service()
    default_profile = task_service.settings.default_profile
    instructions = INSTRUCTIONS
    if task_service.document_sources is not None:
        names = task_service.document_sources.available_names
        if names:
            instructions += f" Configured document_sources: {', '.join(names)}."
    capability_names = task_service.project_policy.allowed_capabilities
    if capability_names:
        instructions += (
            " Requestable capabilities within this project policy: "
            f"{', '.join(capability_names)}. Callers may only narrow this boundary. "
            "On an invalid name, retry only with a listed suggestion; never fall back "
            "to shell execution."
        )
    mcp = FastMCP(SERVER_NAME, instructions=instructions)

    @mcp.resource("taskchamber://capabilities")
    def capabilities() -> str:
        """Describe effective task capabilities and document inputs without secrets."""

        return json.dumps(
            task_service.capability_catalog(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @mcp.tool()
    async def research(
        question: str,
        scope: str | None = None,
        provider: str = default_profile,
        max_turns: int | None = None,
        max_output_chars: int | None = None,
        document_mode: DocumentMode = DocumentMode.AGENTIC,
        document_sources: list[str] | None = None,
        document_requests: list[DocumentSourceSelection] | None = None,
        include_workspace: bool = True,
        workspace_paths: list[str] | None = None,
        requested_capabilities: list[str] | None = None,
    ) -> Annotated[CallToolResult, TaskResult]:
        """Investigate workspace and/or virtual documents with a read-only agent.

        Args:
            question: The investigation to perform.
            scope: Optional prompt hint, not an authorization boundary.
            provider: A server-configured execution profile.
            max_turns: May only reduce the preset's server-side turn limit.
            max_output_chars: May only reduce the server-side output limit.
            document_mode: Use single_pass for exactly one bounded virtual document.
            document_sources: Optional configured virtual document source names.
            document_requests: Named sources with structured, schema-validated parameters.
            include_workspace: Whether to also expose the staged project workspace.
            workspace_paths: Optional relative paths/globs that narrow workspace exposure.
            requested_capabilities: Optional provider-neutral capability subset.
        """

        return to_call_tool_result(
            await task_service.research(
                question=question,
                scope=scope,
                provider=provider,
                max_turns=max_turns,
                max_output_chars=max_output_chars,
                document_mode=document_mode,
                document_sources=document_sources,
                document_requests=document_requests,
                include_workspace=include_workspace,
                workspace_paths=workspace_paths,
                requested_capabilities=requested_capabilities,
            )
        )

    @mcp.tool()
    async def summarize(
        file_path: str | None = None,
        focus: str | None = None,
        provider: str = default_profile,
        max_turns: int | None = None,
        max_output_chars: int | None = None,
        document_mode: DocumentMode = DocumentMode.AGENTIC,
        requested_capabilities: list[str] | None = None,
        document_sources: list[str] | None = None,
        document_requests: list[DocumentSourceSelection] | None = None,
    ) -> Annotated[CallToolResult, TaskResult]:
        """Summarize a workspace file and/or selected virtual documents.

        ``file_path`` may be relative to the configured workspace root or an
        absolute path within that root. Document commands and paths remain
        server-owned; callers select names and structured parameters only.
        """

        return to_call_tool_result(
            await task_service.summarize(
                file_path=file_path,
                focus=focus,
                provider=provider,
                max_turns=max_turns,
                max_output_chars=max_output_chars,
                document_mode=document_mode,
                requested_capabilities=requested_capabilities,
                document_sources=document_sources,
                document_requests=document_requests,
            )
        )

    @mcp.tool()
    async def review(
        file_path: str | None = None,
        provider: str = default_profile,
        max_turns: int | None = None,
        max_output_chars: int | None = None,
        document_mode: DocumentMode = DocumentMode.AGENTIC,
        workspace_paths: list[str] | None = None,
        requested_capabilities: list[str] | None = None,
        document_sources: list[str] | None = None,
        document_requests: list[DocumentSourceSelection] | None = None,
    ) -> Annotated[CallToolResult, TaskResult]:
        """Review selected workspace files and/or virtual documents with a read-only agent.

        ``file_path`` preserves the original single-file contract. Use
        ``workspace_paths`` for additional relative paths or globs. Every match
        must remain within the project-configured workspace policy.
        """

        return to_call_tool_result(
            await task_service.review(
                file_path=file_path,
                provider=provider,
                max_turns=max_turns,
                max_output_chars=max_output_chars,
                document_mode=document_mode,
                workspace_paths=workspace_paths,
                requested_capabilities=requested_capabilities,
                document_sources=document_sources,
                document_requests=document_requests,
            )
        )

    return mcp


_default_mcp: FastMCP | None = None


def get_default_server() -> FastMCP:
    """Build the default server lazily so importing core does not import an SDK."""

    global _default_mcp
    if _default_mcp is None:
        _default_mcp = create_server()
    return _default_mcp


def __getattr__(name: str) -> object:
    if name == "mcp":
        return get_default_server()
    raise AttributeError(name)


def main() -> None:
    """Start the default stdio server."""

    get_default_server().run()


__all__ = [
    "create_default_service",
    "create_runtime_from_environment",
    "create_server",
    "get_default_server",
    "main",
    "render_legacy_result",
    "to_call_tool_result",
]

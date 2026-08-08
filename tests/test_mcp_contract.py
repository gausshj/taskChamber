import json
import sys
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from taskchamber.application.documents import (
    CommandDocumentSource,
    DirectoryDocumentSource,
    DocumentSourceRegistry,
)
from taskchamber.config import (
    CommandDocumentSourceConfig,
    DirectoryDocumentSourceConfig,
    DocumentParameterConfig,
)
from taskchamber.core.contracts import TaskResult, TaskStatus, TokenUsage
from taskchamber.core.service import ServerSettings, TaskService
from taskchamber.runtimes.fake import FakeRuntime
from taskchamber.transport.mcp import MCPTextMode, create_server


@pytest.mark.anyio
async def test_mcp_contract_publishes_tools_and_structured_results(tmp_path: Path) -> None:
    async def result_with_usage(request, _policy):
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output="done",
            usage=TokenUsage(input_tokens=12, output_tokens=3, total_tokens=15),
        )

    runtime = FakeRuntime(handler=result_with_usage)
    service = TaskService(
        runtime,
        ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
    )
    server = create_server(service)

    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert set(tools) == {"research", "summarize", "review"}
        for tool in tools.values():
            assert tool.outputSchema is not None
            assert {"run_id", "kind", "status"} <= set(tool.outputSchema["required"])
        assert tools["research"].inputSchema["properties"]["provider"]["default"] == "fake-profile"
        assert tools["research"].inputSchema["properties"]["include_workspace"]["default"] is True
        assert "document_sources" in tools["research"].inputSchema["properties"]
        assert "document_requests" in tools["research"].inputSchema["properties"]
        assert "workspace_paths" in tools["research"].inputSchema["properties"]
        assert "document_mode" in tools["research"].inputSchema["properties"]
        assert "max_output_chars" in tools["research"].inputSchema["properties"]
        assert "requested_capabilities" in tools["review"].inputSchema["properties"]
        # No per-call parameter lets a caller raise the single-pass input ceiling.
        assert "max_single_pass_document_bytes" not in tools["research"].inputSchema["properties"]
        # The typed error_details is published as an optional output field.
        assert "error_details" in tools["research"].outputSchema["properties"]  # type: ignore[index]

        resources = (await session.list_resources()).resources
        assert [str(resource.uri) for resource in resources] == ["taskchamber://capabilities"]
        capability_resource = await session.read_resource(AnyUrl("taskchamber://capabilities"))
        catalog = json.loads(capability_resource.contents[0].text)
        assert "workspace.read" in catalog["capabilities"]
        assert "executable" not in capability_resource.contents[0].text
        assert catalog["single_pass"] == {
            "max_documents": 1,
            "max_turns": 1,
            "effective_max_document_bytes": 64_000,
            "host_absolute_max_document_bytes": 2_097_152,
            "caller_can_raise": False,
            "oversize_behavior": "error",
        }
        # The capabilities resource must not reveal the configuration source.
        assert "TASKCHAMBER_MAX_SINGLE_PASS" not in capability_resource.contents[0].text

        result = await session.call_tool(
            "research",
            {"question": "Find the entry point"},
        )

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["status"] == "success"
    assert result.structuredContent["kind"] == "research"
    assert result.structuredContent["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "reasoning_output_tokens": None,
        "total_tokens": 15,
    }
    assert result.structuredContent["execution"]["allowed_tools"] == [
        "Read",
        "Glob",
        "Grep",
    ]
    assert "allowed=Read,Glob,Grep" in result.content[0].text
    assert "disallowed=Bash,Edit,Write" in result.content[0].text
    assert result.content[0].text.startswith("[provider=fake-profile status=success]")
    assert "error_code=none" in result.content[0].text
    assert "[tokens input=12 output=3 total=15]" in result.content[0].text
    assert runtime.requests[0].provider == "fake-profile"


@pytest.mark.anyio
async def test_mcp_contract_returns_structured_policy_error(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("not allowed", encoding="utf-8")
    service = TaskService(FakeRuntime(), ServerSettings(workspace_root=tmp_path))
    server = create_server(service)

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "summarize",
            {"file_path": str(outside), "provider": "fake-profile"},
        )

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["status"] == "policy_denied"


@pytest.mark.anyio
async def test_mcp_failure_renders_error_before_incomplete_output(tmp_path: Path) -> None:
    async def turn_limited(request, _policy):
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.TURN_LIMIT_EXCEEDED,
            output="unfinished conclusion",
            num_turns=1,
            partial=True,
            error_code="turn_limit_exceeded",
            error_message="The agent exceeded the configured turn limit.",
        )

    server = create_server(
        TaskService(FakeRuntime(handler=turn_limited), ServerSettings(workspace_root=tmp_path))
    )
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("research", {"question": "fail clearly"})

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["is_error"] is True
    assert result.structuredContent["error_code"] == "turn_limit_exceeded"
    text = result.content[0].text
    assert "is_error=true" in text
    assert "error_code=turn_limit_exceeded" in text
    assert text.index("The agent exceeded") < text.index("unfinished conclusion")
    assert "[incomplete partial output]" in text


@pytest.mark.anyio
async def test_mcp_research_selects_named_virtual_documents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    (external / "manual.md").write_text("virtual fact", encoding="utf-8")

    async def read_virtual_document(request, policy):
        assert policy.document_catalog is not None
        page = await policy.document_catalog.read_document(
            source="manuals", document_id="manual.md"
        )
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=page.content,
        )

    service = TaskService(
        FakeRuntime(handler=read_virtual_document),
        ServerSettings(workspace_root=workspace, default_profile="fake-profile"),
        document_sources=DocumentSourceRegistry(
            {
                "manuals": DirectoryDocumentSource(
                    DirectoryDocumentSourceConfig(name="manuals", root=external)
                )
            }
        ),
    )
    server = create_server(service)
    assert "Configured document_sources: manuals" in server._mcp_server.instructions

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "research",
            {
                "question": "What is the fact?",
                "document_sources": ["manuals"],
                "include_workspace": False,
            },
        )

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["execution"]["document_sources"] == ["manuals"]
    assert "document_sources=manuals" in result.content[0].text
    assert result.content[0].text.endswith("virtual fact")


@pytest.mark.parametrize("tool_name", ["summarize", "review"])
@pytest.mark.anyio
async def test_mcp_file_tools_run_single_pass_without_document_tools(
    tmp_path: Path,
    tool_name: str,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "record.txt").write_text("bounded record body", encoding="utf-8")

    async def complete_once(request, policy):
        assert request.kind.value == tool_name
        assert request.max_turns == 1
        assert "bounded record body" in request.prompt
        assert policy.allowed_tools == ()
        assert policy.document_tools == ()
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=f"{tool_name} complete",
            num_turns=1,
        )

    runtime = FakeRuntime(handler=complete_once)
    server = create_server(
        TaskService(
            runtime,
            ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
            document_sources=DocumentSourceRegistry(
                {
                    "record": DirectoryDocumentSource(
                        DirectoryDocumentSourceConfig(name="record", root=documents)
                    )
                }
            ),
        )
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            tool_name,
            {
                "document_mode": "single_pass",
                "document_sources": ["record"],
                "requested_capabilities": ["documents.read"],
                "max_turns": 1,
            },
        )

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["num_turns"] == 1
    assert result.structuredContent["execution"]["document_sources"] == ["record"]
    assert result.structuredContent["execution"]["document_tools"] == []


@pytest.mark.anyio
async def test_mcp_single_pass_oversize_returns_typed_error_details(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "record.txt").write_text("x" * 100, encoding="utf-8")

    runtime = FakeRuntime()
    server = create_server(
        TaskService(
            runtime,
            ServerSettings(
                workspace_root=tmp_path,
                default_profile="fake-profile",
                max_single_pass_document_bytes=4,
            ),
            document_sources=DocumentSourceRegistry(
                {
                    "record": DirectoryDocumentSource(
                        DirectoryDocumentSourceConfig(name="record", root=documents)
                    )
                }
            ),
        )
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "research",
            {
                "question": "Summarize the record",
                "document_mode": "single_pass",
                "document_sources": ["record"],
                "include_workspace": False,
                "requested_capabilities": ["documents.read"],
                "max_turns": 1,
            },
        )

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["status"] == "invalid_request"
    assert result.structuredContent["error_code"] == "single_pass_document_too_large"
    details = result.structuredContent["error_details"]
    assert details["type"] == "single_pass_document_too_large"
    assert details["document_mode"] == "single_pass"
    assert details["source"] == "record"
    assert details["document_id"] == "record.txt"
    assert details["observed_utf8_bytes"] == 100
    assert details["effective_limit_bytes"] == 4
    assert details["absolute_limit_bytes"] == 2_097_152
    assert details["retryable"] is False
    # The host workspace path must not leak into the structured response.
    assert str(tmp_path) not in json.dumps(result.structuredContent)


@pytest.mark.anyio
async def test_mcp_review_accepts_multiple_workspace_globs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("ONE = 1\n", encoding="utf-8")
    (tmp_path / "src" / "two.py").write_text("TWO = 2\n", encoding="utf-8")
    runtime = FakeRuntime()
    server = create_server(
        TaskService(
            runtime,
            ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
        )
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "review",
            {
                "workspace_paths": ["src/*.py"],
                "requested_capabilities": ["read", "grep"],
            },
        )

    assert result.isError is False
    assert runtime.policies[0].allowed_tools == ("Read", "Grep")
    assert len(runtime.policies[0].allowed_paths) == 2


@pytest.mark.anyio
async def test_mcp_research_uses_structured_command_document_request(tmp_path: Path) -> None:
    script = "import json,sys; print(json.dumps({'result.txt':sys.argv[1]}))"

    async def read_record(request, policy):
        assert policy.document_catalog is not None
        page = await policy.document_catalog.read_document(
            source="record_detail",
            document_id="result.txt",
        )
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=page.content,
        )

    service = TaskService(
        FakeRuntime(handler=read_record),
        ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
        document_sources=DocumentSourceRegistry(
            {
                "record_detail": CommandDocumentSource(
                    CommandDocumentSourceConfig(
                        name="record_detail",
                        aliases=("记录详情",),
                        argv=(sys.executable, "-c", script, "{record_id}"),
                        parameters=(
                            DocumentParameterConfig(
                                name="record_id",
                                pattern=r"^rec_[A-Za-z0-9_-]+$",
                            ),
                        ),
                        output_format="json",
                    ),
                    environment={},
                )
            }
        ),
    )
    server = create_server(service)

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "research",
            {
                "question": "Analyze the record",
                "include_workspace": False,
                "document_requests": [
                    {
                        "source": "记录详情",
                        "parameters": {"record_id": "rec_demo"},
                    }
                ],
                "requested_capabilities": ["documents.read"],
            },
        )

    assert result.isError is False
    assert result.content[0].text.endswith("rec_demo")
    assert result.structuredContent is not None
    assert result.structuredContent["execution"]["document_sources"] == ["record_detail"]
    assert result.structuredContent["execution"]["document_tools"] == ["DocumentRead"]


SENTINEL_BODY = "唯一正文-你好-Δ"


@pytest.mark.anyio
async def test_mcp_full_text_mode_keeps_body_in_text_and_structured_content(
    tmp_path: Path,
) -> None:
    async def succeed(request, _policy):
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=SENTINEL_BODY,
        )

    server = create_server(
        TaskService(
            FakeRuntime(handler=succeed),
            ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
        ),
        text_mode=MCPTextMode.FULL,
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("research", {"question": "q"})

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["output"] == SENTINEL_BODY
    assert result.content[0].text.endswith(SENTINEL_BODY)
    assert result.model_dump_json().count(SENTINEL_BODY) == 2


@pytest.mark.anyio
async def test_mcp_metadata_only_mode_serializes_success_body_exactly_once(
    tmp_path: Path,
) -> None:
    async def succeed(request, _policy):
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=SENTINEL_BODY,
            usage=TokenUsage(input_tokens=12, output_tokens=3, total_tokens=15),
        )

    server = create_server(
        TaskService(
            FakeRuntime(handler=succeed),
            ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
        ),
        text_mode=MCPTextMode.METADATA_ONLY,
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("research", {"question": "q"})

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["output"] == SENTINEL_BODY
    text = result.content[0].text
    assert SENTINEL_BODY not in text
    assert result.model_dump_json().count(SENTINEL_BODY) == 1
    assert "status=success" in text
    assert "partial=false" in text
    assert "truncated=false" in text
    assert "[tokens input=12 output=3 total=15]" in text
    assert "[execution " in text


@pytest.mark.anyio
async def test_mcp_metadata_only_mode_keeps_full_error_text(tmp_path: Path) -> None:
    async def fail_with_partial(request, _policy):
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.FAILED,
            output=f"partial {SENTINEL_BODY}",
            partial=True,
            error_code="failed",
            error_message="The provider stopped mid-task.",
        )

    server = create_server(
        TaskService(
            FakeRuntime(handler=fail_with_partial),
            ServerSettings(workspace_root=tmp_path, default_profile="fake-profile"),
        ),
        text_mode=MCPTextMode.METADATA_ONLY,
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("research", {"question": "q"})

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["output"] == f"partial {SENTINEL_BODY}"
    assert result.structuredContent["error_message"] == "The provider stopped mid-task."
    text = result.content[0].text
    assert "The provider stopped mid-task." in text
    assert "[incomplete partial output]" in text
    assert f"partial {SENTINEL_BODY}" in text


@pytest.mark.anyio
async def test_mcp_metadata_only_mode_keeps_server_truncation_contract(tmp_path: Path) -> None:
    async def verbose(request, _policy):
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=TaskStatus.SUCCESS,
            output=SENTINEL_BODY * 40,
        )

    server = create_server(
        TaskService(
            FakeRuntime(handler=verbose),
            ServerSettings(
                workspace_root=tmp_path,
                default_profile="fake-profile",
                max_output_chars=200,
            ),
        ),
        text_mode=MCPTextMode.METADATA_ONLY,
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("research", {"question": "q"})

    assert result.isError is False
    assert result.structuredContent is not None
    assert len(result.structuredContent["output"]) <= 200
    assert result.structuredContent["output"].endswith("[output truncated by server policy]")
    assert result.structuredContent["truncated"] is True
    assert result.structuredContent["partial"] is True
    assert result.structuredContent["effective_max_output_chars"] == 200
    text = result.content[0].text
    assert SENTINEL_BODY not in text
    assert "partial=true" in text
    assert "truncated=true" in text
    assert "effective_max_output_chars=200" in text

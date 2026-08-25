import asyncio
import dataclasses
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from mcp.shared.memory import create_connected_server_and_client_session

from taskchamber.application.documents import DirectoryDocumentSource
from taskchamber.config import ProviderProfile
from taskchamber.config.documents import DirectoryDocumentSourceConfig
from taskchamber.core.contracts import (
    ExecutionPolicy,
    TaskKind,
    TaskRequest,
    TaskStatus,
    ToolCallDecision,
)
from taskchamber.core.documents import DocumentCatalog
from taskchamber.runtimes.claude import ClaudeAgentSdkRuntime
from taskchamber.runtimes.claude.documents import CLAUDE_DOCUMENT_TOOL_NAMES


def _policy(workspace_root: Path, *, timeout_seconds: float = 1.0) -> ExecutionPolicy:
    return ExecutionPolicy(
        workspace_root=workspace_root,
        allowed_paths=(workspace_root,),
        system_prompt="Read only.",
        allowed_tools=("Read", "Glob", "Grep"),
        disallowed_tools=("Bash", "Edit", "Write"),
        max_turns=3,
        max_budget_usd=0.5,
        timeout_seconds=timeout_seconds,
        max_output_chars=1_000,
        max_file_bytes=1_000,
    )


def _request() -> TaskRequest:
    return TaskRequest(
        run_id="run-1",
        kind=TaskKind.RESEARCH,
        prompt="Inspect the workspace.",
        provider="glm",
        max_turns=3,
    )


def test_claude_options_enforce_read_only_boundary(tmp_path: Path) -> None:
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})
    options = runtime.build_options(_request(), _policy(tmp_path), config_dir=tmp_path / "config")

    assert options.tools == ["Read", "Glob", "Grep"]
    assert options.allowed_tools == ["Read", "Glob", "Grep"]
    assert set(options.disallowed_tools) >= {"Bash", "Edit", "Write"}
    assert options.permission_mode == "dontAsk"
    assert options.cwd == tmp_path
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {}
    assert options.setting_sources == []
    assert options.skills == []
    assert options.extra_args == {"no-session-persistence": None}
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == "test-token"
    assert options.env["ANTHROPIC_API_KEY"] == ""
    assert options.env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "config")


def test_claude_options_disable_budget_unless_the_host_opts_in(tmp_path: Path) -> None:
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})

    opted_out = runtime.build_options(
        _request(),
        dataclasses.replace(_policy(tmp_path), max_budget_usd=None),
        config_dir=tmp_path / "config",
    )
    assert opted_out.max_budget_usd is None

    opted_in = runtime.build_options(
        _request(),
        dataclasses.replace(_policy(tmp_path), max_budget_usd=0.25),
        config_dir=tmp_path / "config",
    )
    assert opted_in.max_budget_usd == 0.25


def test_dynamic_provider_only_forwards_the_selected_credential(tmp_path: Path) -> None:
    reference = "TASKCHAMBER_PROFILE__CUSTOM__API_KEY"
    profile = ProviderProfile(
        name="custom",
        runtime="claude",
        api_format="anthropic",
        base_url="https://provider.example/anthropic",
        model="custom-model",
        credential_ref=reference,
        api_key_field="auth_token",
    )
    runtime = ClaudeAgentSdkRuntime(
        providers={"custom": profile},
        environment={reference: "selected-token", "UNRELATED_SECRET": "must-not-be-copied"},
    )
    request = _request().model_copy(update={"provider": "custom"})

    options = runtime.build_options(request, _policy(tmp_path), config_dir=tmp_path / "config")

    assert options.model == "custom-model"
    assert options.env[reference] == ""
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == "selected-token"
    assert options.env["ANTHROPIC_BASE_URL"] == "https://provider.example/anthropic"
    assert "UNRELATED_SECRET" not in options.env


@pytest.mark.anyio
async def test_claude_pre_tool_hook_denies_workspace_escape_and_protected_patterns(
    tmp_path: Path,
) -> None:
    source = tmp_path / "safe.py"
    source.write_text("print('safe')\n", encoding="utf-8")
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})
    options = runtime.build_options(_request(), _policy(tmp_path), config_dir=tmp_path / "config")
    hook = options.hooks["PreToolUse"][0].hooks[0]

    allowed = await hook(
        {"tool_name": "Read", "tool_input": {"file_path": "safe.py"}},
        None,
        {},
    )
    denied = await hook(
        {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}},
        None,
        {},
    )
    denied_glob = await hook(
        {"tool_name": "Glob", "tool_input": {"pattern": "**/.env*"}},
        None,
        {},
    )
    denied_grep = await hook(
        {
            "tool_name": "Grep",
            "tool_input": {"pattern": "PRIVATE", "glob": "**/*.pem"},
        },
        None,
        {},
    )
    allowed_grep = await hook(
        {
            "tool_name": "Grep",
            "tool_input": {"pattern": r"\.\./api|/v1/messages", "glob": "**/*.py"},
        },
        None,
        {},
    )

    assert allowed == {}
    assert allowed_grep == {}
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert denied_glob["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert denied_grep["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.anyio
async def test_claude_options_expose_only_task_scoped_document_mcp_tools(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    workspace = tmp_path / "workspace"
    external.mkdir()
    workspace.mkdir()
    (external / "manual.md").write_text("line one\nvirtual fact\n", encoding="utf-8")
    prepared = await DirectoryDocumentSource(
        DirectoryDocumentSourceConfig(name="manuals", root=external)
    ).prepare(query="fact")
    catalog = DocumentCatalog({"manuals": prepared})
    policy = dataclasses.replace(
        _policy(workspace),
        allowed_paths=(),
        allowed_tools=(),
        document_catalog=catalog,
        document_sources=("manuals",),
        document_tools=("DocumentList", "DocumentRead", "DocumentSearch"),
    )
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})

    options = runtime.build_options(_request(), policy, config_dir=tmp_path / "config")

    assert options.tools == list(CLAUDE_DOCUMENT_TOOL_NAMES)
    assert options.allowed_tools == list(CLAUDE_DOCUMENT_TOOL_NAMES)
    assert set(options.mcp_servers) == {"documents"}
    hook = options.hooks["PreToolUse"][0].hooks[0]
    allowed = await hook(
        {
            "tool_name": "mcp__documents__read_document",
            "tool_input": {"source": "manuals", "document_id": "manual.md"},
        },
        None,
        {},
    )
    denied = await hook(
        {"tool_name": "mcp__untrusted__read", "tool_input": {}},
        None,
        {},
    )
    assert allowed == {}
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    document_server = options.mcp_servers["documents"]["instance"]
    async with create_connected_server_and_client_session(document_server) as session:
        tools = {item.name for item in (await session.list_tools()).tools}
        assert tools == {"list_documents", "read_document", "search_documents"}
        result = await session.call_tool(
            "read_document",
            {"source": "manuals", "document_id": "manual.md"},
        )
    assert result.isError is False
    assert "virtual fact" in result.content[0].text


@pytest.mark.anyio
async def test_claude_document_only_run_stages_empty_workspace_but_reads_catalog(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    (workspace / "must-not-be-staged.txt").write_text("private", encoding="utf-8")
    (external / "manual.md").write_text("available virtually", encoding="utf-8")
    prepared = await DirectoryDocumentSource(
        DirectoryDocumentSourceConfig(name="manuals", root=external)
    ).prepare(query="virtually")
    policy = dataclasses.replace(
        _policy(workspace),
        allowed_paths=(),
        allowed_tools=(),
        document_catalog=DocumentCatalog({"manuals": prepared}),
        document_sources=("manuals",),
        document_tools=("DocumentList", "DocumentRead", "DocumentSearch"),
    )

    async def fake_query(*, prompt, options):
        del prompt
        assert list(options.cwd.iterdir()) == []
        document_server = options.mcp_servers["documents"]["instance"]
        async with create_connected_server_and_client_session(document_server) as session:
            response = await session.call_tool(
                "read_document",
                {"source": "manuals", "document_id": "manual.md"},
            )
        assert "available virtually" in response.content[0].text
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="document-only",
            total_cost_usd=0,
            usage={},
            model_usage={},
            result="read virtual document",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )
    result = await runtime.run(_request(), policy)

    assert result.status is TaskStatus.SUCCESS
    assert result.execution is not None
    assert result.execution.workspace_staged is True
    assert result.execution.allowed_tools == CLAUDE_DOCUMENT_TOOL_NAMES
    assert result.execution.document_sources == ("manuals",)


@pytest.mark.anyio
async def test_pre_tool_hook_enforces_per_task_allowed_paths(tmp_path: Path) -> None:
    """A file task's hook allows the target file but denies a workspace sibling."""

    target = tmp_path / "target.py"
    target.write_text("print('target')\n", encoding="utf-8")
    sibling = tmp_path / "sibling.py"
    sibling.write_text("print('sibling')\n", encoding="utf-8")

    policy = dataclasses.replace(_policy(tmp_path), allowed_paths=(target.resolve(),))
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})
    options = runtime.build_options(_request(), policy, config_dir=tmp_path / "config")
    hook = options.hooks["PreToolUse"][0].hooks[0]

    allow_target = await hook(
        {"tool_name": "Read", "tool_input": {"file_path": str(target)}},
        None,
        {},
    )
    deny_sibling = await hook(
        {"tool_name": "Read", "tool_input": {"file_path": str(sibling)}},
        None,
        {},
    )

    assert allow_target == {}
    assert deny_sibling["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.anyio
async def test_missing_provider_credential_does_not_start_sdk(tmp_path: Path) -> None:
    called = False

    async def fake_query(**_kwargs):
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(environment={}, query_function=fake_query)
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == "missing_credential"
    assert called is False


@pytest.mark.anyio
async def test_unknown_provider_is_a_safe_runtime_result(tmp_path: Path) -> None:
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})
    request = _request().model_copy(update={"provider": "not-a-provider"})

    result = await runtime.run(request, _policy(tmp_path))

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == "unknown_provider"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("runtime_name", "api_format", "expected_code"),
    [
        ("codex", "responses", "incompatible_runtime"),
        ("claude", "openai_responses", "unsupported_api_format"),
    ],
)
async def test_incompatible_provider_protocol_is_rejected_without_translation(
    tmp_path: Path,
    runtime_name: str,
    api_format: str,
    expected_code: str,
) -> None:
    called = False

    async def fake_query(**_kwargs):
        nonlocal called
        called = True
        yield

    profile = ProviderProfile(
        name="custom",
        runtime=runtime_name,
        api_format=api_format,
        base_url="https://provider.example/api",
        model="model",
        credential_ref="CUSTOM_API_KEY",
    )
    runtime = ClaudeAgentSdkRuntime(
        providers={"custom": profile},
        environment={"CUSTOM_API_KEY": "token"},
        query_function=fake_query,
    )
    request = _request().model_copy(update={"provider": "custom"})

    result = await runtime.run(request, _policy(tmp_path))

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == expected_code
    assert called is False


@pytest.mark.anyio
async def test_claude_runtime_normalizes_successful_result(tmp_path: Path) -> None:
    async def fake_query(**_kwargs):
        yield AssistantMessage(content=[TextBlock(text="intermediate")], model="glm-5.2")
        yield ResultMessage(
            subtype="success",
            duration_ms=7,
            duration_api_ms=5,
            is_error=False,
            num_turns=2,
            session_id="session-1",
            total_cost_usd=0.0123,
            usage={
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 10,
            },
            model_usage={
                "glm-5.2": {
                    "inputTokens": 120,
                    "outputTokens": 30,
                    "cacheReadInputTokens": 80,
                    "cacheCreationInputTokens": 10,
                    "costUSD": 0.0123,
                }
            },
            result="final answer",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.output == "final answer"
    assert result.num_turns == 2
    assert result.cost_usd == pytest.approx(0.0123)
    assert result.usage is not None
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 30
    assert result.usage.cache_read_input_tokens == 80
    assert result.usage.cache_creation_input_tokens == 10
    assert result.usage.total_tokens is None
    assert result.model_usage is not None
    assert result.model_usage["glm-5.2"].input_tokens == 120


def test_claude_single_pass_options_expose_no_tools(tmp_path: Path) -> None:
    runtime = ClaudeAgentSdkRuntime(environment={"Z_AI_API_KEY": "test-token"})
    policy = _policy(tmp_path)
    policy = dataclasses.replace(
        policy,
        allowed_paths=(),
        allowed_tools=(),
        max_turns=1,
        document_catalog=None,
        document_tools=(),
    )

    options = runtime.build_options(
        _request().model_copy(update={"max_turns": 1}),
        policy,
        config_dir=tmp_path / "config",
    )

    assert options.max_turns == 1
    assert options.tools == []
    assert options.allowed_tools == []
    assert options.mcp_servers == {}


@pytest.mark.anyio
async def test_claude_runtime_ignores_unreported_or_invalid_token_counters(
    tmp_path: Path,
) -> None:
    async def fake_query(**_kwargs):
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-usage",
            usage={
                "input_tokens": True,
                "output_tokens": -1,
                "total_tokens": "12",
            },
            model_usage={"glm-5.2": {"inputTokens": None}},
            result="done",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.usage is None
    assert result.model_usage is None


@pytest.mark.anyio
async def test_claude_runtime_records_sanitized_tool_policy_decisions(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("public", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=hidden", encoding="utf-8")

    async def fake_query(*, options, **_kwargs):
        hook = options.hooks["PreToolUse"][0].hooks[0]
        await hook(
            {"tool_name": "Read", "tool_input": {"file_path": "safe.txt"}},
            None,
            {},
        )
        await hook(
            {"tool_name": "Read", "tool_input": {"file_path": ".env"}},
            None,
            {},
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-audit",
            result="done",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.execution is not None
    assert [(record.tool, record.decision) for record in result.execution.tool_calls] == [
        ("Read", ToolCallDecision.ALLOWED),
        ("Read", ToolCallDecision.DENIED),
    ]


@pytest.mark.anyio
async def test_claude_runtime_maps_timeout_without_provider_call(tmp_path: Path) -> None:
    async def slow_query(**_kwargs):
        await asyncio.Event().wait()
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=slow_query,
    )
    result = await runtime.run(_request(), _policy(tmp_path, timeout_seconds=0.01))

    assert result.status is TaskStatus.TIMED_OUT
    assert result.error_code == "timeout"


@pytest.mark.anyio
async def test_claude_runtime_closes_the_stream_after_timeout(tmp_path: Path) -> None:
    closed = False

    async def slow_query(**_kwargs):
        nonlocal closed
        try:
            await asyncio.Event().wait()
            yield
        finally:
            closed = True

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=slow_query,
    )
    result = await runtime.run(_request(), _policy(tmp_path, timeout_seconds=0.01))

    assert result.status is TaskStatus.TIMED_OUT
    assert closed is True


@pytest.mark.anyio
async def test_claude_runtime_hides_unexpected_runtime_errors(tmp_path: Path) -> None:
    async def failing_query(**_kwargs):
        if False:
            yield
        raise RuntimeError("sensitive internal detail")

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=failing_query,
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "runtime_error"
    assert "sensitive" not in result.error_message


@pytest.mark.anyio
async def test_claude_runtime_maps_missing_final_result(tmp_path: Path) -> None:
    async def empty_query(**_kwargs):
        if False:
            yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=empty_query,
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "missing_result"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("subtype", "expected"),
    [
        ("error_max_turns", TaskStatus.TURN_LIMIT_EXCEEDED),
        ("error_max_budget_usd", TaskStatus.BUDGET_EXCEEDED),
    ],
)
async def test_claude_runtime_maps_limit_results(
    tmp_path: Path,
    subtype: str,
    expected: TaskStatus,
) -> None:
    async def limited_query(**_kwargs):
        yield ResultMessage(
            subtype=subtype,
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=3,
            session_id="session-1",
            result="partial",
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=limited_query,
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is expected
    assert result.output == "partial"
    assert result.error_code == expected.value
    assert result.is_error is True
    assert result.partial is True


@pytest.mark.anyio
async def test_claude_runtime_maps_limit_result_when_sdk_raises_after_it(tmp_path: Path) -> None:
    """The SDK raises after yielding a turn-limit ResultMessage; keep the mapping."""

    async def query_then_raise(**_kwargs):
        yield ResultMessage(
            subtype="error_max_turns",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=3,
            session_id="session-1",
            result="partial",
        )
        raise RuntimeError("Reached maximum number of turns")

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=query_then_raise,
    )
    result = await runtime.run(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.TURN_LIMIT_EXCEEDED
    assert result.output == "partial"
    assert result.error_code == "turn_limit_exceeded"
    assert result.partial is True

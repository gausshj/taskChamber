"""Contract tests for the tool-free structured completion capability."""

import asyncio
import json
from pathlib import Path

import pytest
from claude_agent_sdk import ResultMessage

from taskchamber.cli import main as cli_main
from taskchamber.core.completion import (
    DEFAULT_COMPLETION_MAX_TURNS,
    StructuredCompletionRequest,
    StructuredCompletionResult,
    StructuredCompletionService,
)
from taskchamber.core.contracts import (
    AgentCapabilities,
    ExecutionPolicy,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from taskchamber.core.service import ServerSettings
from taskchamber.runtimes.claude import ClaudeAgentSdkRuntime
from taskchamber.runtimes.fake import FakeRuntime

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _request(**overrides: object) -> StructuredCompletionRequest:
    payload: dict[str, object] = {
        "request_id": "req-1",
        "system_prompt": "Return JSON only.",
        "prompt": "Give me the data.",
        "json_schema": SCHEMA,
        "provider": "glm",
    }
    payload.update(overrides)
    return StructuredCompletionRequest.model_validate(payload)


def _policy(
    workspace_root: Path,
    *,
    timeout_seconds: float = 1.0,
    max_output_chars: int = 1_000,
) -> ExecutionPolicy:
    return ExecutionPolicy(
        workspace_root=workspace_root,
        allowed_paths=(),
        system_prompt="Return JSON only.",
        allowed_tools=(),
        disallowed_tools=(),
        max_turns=3,
        max_budget_usd=None,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        max_file_bytes=1_000,
    )


def _result_message(**overrides: object) -> ResultMessage:
    payload: dict[str, object] = {
        "subtype": "success",
        "duration_ms": 3,
        "duration_api_ms": 2,
        "is_error": False,
        "num_turns": 1,
        "session_id": "completion-session",
        "total_cost_usd": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "model_usage": {},
        "result": '{"answer": "structured"}',
    }
    payload.update(overrides)
    return ResultMessage(**payload)  # type: ignore[arg-type]


class _UnsupportedRuntime:
    """A runtime without the structured completion capability."""

    name = "unsupported"
    default_profile = "unsupported"
    capabilities = AgentCapabilities(structured_output=False)

    async def run(self, request: TaskRequest, policy: ExecutionPolicy) -> TaskResult:
        raise NotImplementedError


@pytest.mark.anyio
async def test_service_returns_result_and_stamps_effective_limits(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    service = StructuredCompletionService(runtime, ServerSettings(workspace_root=tmp_path))

    result = await service.complete(_request(provider=None))

    assert result.status is TaskStatus.SUCCESS
    assert result.request_id == "req-1"
    assert result.output == {"fake": True}
    assert result.provider == "fake"
    assert result.effective_max_turns == DEFAULT_COMPLETION_MAX_TURNS
    assert result.effective_timeout_seconds == 120.0
    assert result.effective_max_output_chars == 12_000


@pytest.mark.anyio
async def test_service_narrows_caller_limits_to_host_ceilings(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    service = StructuredCompletionService(runtime, ServerSettings(workspace_root=tmp_path))

    await service.complete(_request(max_turns=99, timeout_seconds=999.0, max_output_chars=99_999))
    policy = runtime.policies[0]
    assert policy.max_turns == DEFAULT_COMPLETION_MAX_TURNS
    assert policy.timeout_seconds == 120.0
    assert policy.max_output_chars == 12_000

    await service.complete(_request(max_turns=2, timeout_seconds=5.0, max_output_chars=100))
    narrowed = runtime.policies[1]
    assert narrowed.max_turns == 2
    assert narrowed.timeout_seconds == 5.0
    assert narrowed.max_output_chars == 100


@pytest.mark.anyio
async def test_service_exposes_no_workspace_document_or_tool_access(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    service = StructuredCompletionService(runtime, ServerSettings(workspace_root=tmp_path))

    await service.complete(_request())

    policy = runtime.policies[0]
    assert policy.allowed_paths == ()
    assert policy.allowed_tools == ()
    assert policy.document_catalog is None
    assert policy.document_tools == ()


@pytest.mark.anyio
async def test_service_fails_explicitly_when_the_runtime_is_incapable(
    tmp_path: Path,
) -> None:
    service = StructuredCompletionService(
        _UnsupportedRuntime(),  # type: ignore[arg-type]
        ServerSettings(workspace_root=tmp_path),
    )

    result = await service.complete(_request(provider=None))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "structured_output_unsupported"
    assert result.output is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"json_schema": {}},
        {"prompt": ""},
        {"system_prompt": ""},
        {"request_id": ""},
    ],
)
def test_request_validation_rejects_empty_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _request(**overrides)


@pytest.mark.anyio
async def test_schema_and_instructions_reach_the_sdk_unchanged(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def fake_query(**kwargs: object) -> object:
        captured.update(kwargs)
        yield _result_message(structured_output={"answer": "structured"})

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    options = captured["options"]
    assert options.output_format == {"type": "json_schema", "schema": SCHEMA}
    assert options.system_prompt == "Return JSON only."
    assert captured["prompt"] == "Give me the data."
    assert options.tools == []
    assert options.allowed_tools == []
    assert options.mcp_servers == {}
    assert options.extra_args == {"no-session-persistence": None}
    assert options.setting_sources == []
    assert options.skills == []


@pytest.mark.anyio
async def test_nested_and_null_fields_survive_without_rewriting(tmp_path: Path) -> None:
    structured = {"outer": {"items": [1, None, {"leaf": None}]}, "explicit_null": None}

    async def fake_query(**kwargs: object) -> object:
        yield _result_message(
            structured_output=structured,
            num_turns=2,
            total_cost_usd=0.0042,
            model_usage={"model-x": {"input_tokens": 10, "output_tokens": 5}},
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.output == structured
    assert result.raw_result_text == '{"answer": "structured"}'
    assert result.num_turns == 2
    assert result.cost_usd == 0.0042
    assert result.usage is not None and result.usage.input_tokens == 10
    assert result.model_usage is not None and "model-x" in result.model_usage
    assert result.sdk_version is not None
    assert result.error_code is None


@pytest.mark.anyio
async def test_success_result_without_structured_output_is_explicit_failure(
    tmp_path: Path,
) -> None:
    async def fake_query(**kwargs: object) -> object:
        yield _result_message(structured_output=None, result="unconstrained prose")

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "structured_output_missing"
    assert result.output is None


@pytest.mark.anyio
async def test_sdk_error_result_never_appears_as_success(tmp_path: Path) -> None:
    async def fake_query(**kwargs: object) -> object:
        yield _result_message(
            subtype="error_during_execution",
            is_error=True,
            structured_output=None,
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "agent_result_error"
    assert result.output is None


@pytest.mark.anyio
async def test_turn_limit_result_maps_to_the_stable_failure_code(tmp_path: Path) -> None:
    async def fake_query(**kwargs: object) -> object:
        yield _result_message(
            subtype="error_max_turns",
            is_error=True,
            structured_output=None,
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.TURN_LIMIT_EXCEEDED
    assert result.error_code == "turn_limit_exceeded"


@pytest.mark.anyio
async def test_timeout_is_an_explicit_failure(tmp_path: Path) -> None:
    async def slow_query(**kwargs: object) -> object:
        await asyncio.sleep(5)
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=slow_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path, timeout_seconds=0.01))

    assert result.status is TaskStatus.TIMED_OUT
    assert result.error_code == "timeout"


@pytest.mark.anyio
async def test_missing_credential_is_an_explicit_failure(tmp_path: Path) -> None:
    called = False

    async def fake_query(**kwargs: object) -> object:
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(environment={}, query_function=fake_query)

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == "missing_credential"
    assert called is False


@pytest.mark.anyio
async def test_raw_text_is_bounded_without_touching_structured_output(
    tmp_path: Path,
) -> None:
    structured = {"answer": "x" * 500}

    async def fake_query(**kwargs: object) -> object:
        yield _result_message(structured_output=structured, result="y" * 500)

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path, max_output_chars=100))

    assert result.status is TaskStatus.SUCCESS
    assert result.truncated is True
    assert result.raw_result_text is not None
    assert len(result.raw_result_text) == 100
    assert result.output == structured


@pytest.mark.anyio
async def test_two_completions_share_no_conversation_state(tmp_path: Path) -> None:
    invocations: list[object] = []

    async def fake_query(**kwargs: object) -> object:
        invocations.append(kwargs["options"])
        yield _result_message(structured_output={"answer": "ok"})

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    first = await runtime.complete_structured(_request(request_id="req-a"), _policy(tmp_path))
    second = await runtime.complete_structured(_request(request_id="req-b"), _policy(tmp_path))

    assert first.status is TaskStatus.SUCCESS
    assert second.status is TaskStatus.SUCCESS
    assert first.request_id == "req-a"
    assert second.request_id == "req-b"
    assert len(invocations) == 2
    for options in invocations:
        assert options.extra_args == {"no-session-persistence": None}


def test_cli_complete_prints_json_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TASKCHAMBER_RUNTIME", "fake")
    monkeypatch.setenv("TASKCHAMBER_WORKSPACE_ROOT", str(tmp_path))
    request_file = tmp_path / "request.json"
    request_file.write_text(
        json.dumps(
            {
                "request_id": "cli-1",
                "system_prompt": "Return JSON only.",
                "prompt": "Give me the data.",
                "json_schema": SCHEMA,
            }
        ),
        encoding="utf-8",
    )

    cli_main(["complete", "--request", str(request_file)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["request_id"] == "cli-1"
    assert payload["output"] == {"fake": True}
    assert payload["effective_max_turns"] == DEFAULT_COMPLETION_MAX_TURNS


def test_cli_complete_rejects_an_invalid_request_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_file = tmp_path / "broken.json"
    request_file.write_text('{"request_id": "x"}', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["complete", "--request", str(request_file)])

    assert excinfo.value.code == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.anyio
async def test_completion_result_serializes_with_generic_field_names() -> None:
    result = StructuredCompletionResult(
        request_id="req-1",
        status=TaskStatus.SUCCESS,
        runtime="fake",
        provider="fake",
        output={"nested": {"null_field": None}},
    )

    payload = result.model_dump(mode="json")

    assert payload["status"] == "success"
    assert payload["output"] == {"nested": {"null_field": None}}
    assert payload["error_code"] is None
    assert set(payload) >= {
        "request_id",
        "status",
        "runtime",
        "provider",
        "model",
        "output",
        "raw_result_text",
        "usage",
        "model_usage",
        "num_turns",
        "cost_usd",
        "duration_ms",
        "partial",
        "truncated",
        "error_code",
        "error_message",
        "sdk_version",
        "effective_max_turns",
        "effective_timeout_seconds",
        "effective_max_output_chars",
    }

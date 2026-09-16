"""Contract tests for the tool-free structured completion capability."""

import asyncio
import json
from importlib import metadata
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock

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
from taskchamber.isolation import InsecureCliPathError, NoSandbox
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


class _UnavailableSandbox(NoSandbox):
    os_isolated = True

    def preflight(self) -> bool:
        return False


class _InsecureValidatingSandbox(NoSandbox):
    def validate_cli_executable(self, executable: Path) -> None:
        raise InsecureCliPathError(
            "the selected CLI is writable by group or others below a masked root"
        )


class _InvalidEnvironmentSandbox(NoSandbox):
    def validate_readable_paths(self, paths: tuple[Path, ...]) -> None:
        raise ValueError("forwarded CLI path is hidden by the selected sandbox")


class _InsecureLauncherSandbox(NoSandbox):
    def prepare_cli_launcher(
        self,
        workspace: object,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path,
        environment_keys: tuple[str, ...] = (),
    ) -> Path:
        raise InsecureCliPathError(
            "the selected CLI is writable by group or others below a masked root"
        )


class _FailingLauncherSandbox(NoSandbox):
    def prepare_cli_launcher(
        self,
        workspace: object,
        *,
        executable: str,
        config_dir: Path,
        launcher_dir: Path,
        environment_keys: tuple[str, ...] = (),
    ) -> Path:
        raise ValueError("boundary construction failed")


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
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.model_usage is not None
    assert "model-x" in result.model_usage
    assert result.sdk_version is not None
    assert result.error_code is None


@pytest.mark.anyio
async def test_success_result_without_structured_output_is_explicit_failure(
    tmp_path: Path,
) -> None:
    async def fake_query(**kwargs: object) -> object:
        yield _result_message(
            structured_output=None,
            result="unconstrained prose",
            num_turns=3,
            total_cost_usd=0.0123,
            usage={"input_tokens": 123, "output_tokens": 45},
            model_usage={"model-x": {"input_tokens": 123, "output_tokens": 45}},
        )

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "structured_output_missing"
    assert result.output is None
    # The invocation data already produced is preserved for cost accounting
    # and compatibility debugging.
    assert result.raw_result_text == "unconstrained prose"
    assert result.num_turns == 3
    assert result.cost_usd == 0.0123
    assert result.usage is not None
    assert result.usage.input_tokens == 123
    assert result.usage.output_tokens == 45
    assert result.model_usage is not None
    assert "model-x" in result.model_usage


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


@pytest.mark.anyio
async def test_unknown_provider_is_an_explicit_failure(tmp_path: Path) -> None:
    called = False

    async def fake_query(**kwargs: object) -> object:
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(
        _request(provider="not-a-provider"), _policy(tmp_path)
    )

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == "unknown_provider"
    assert called is False


@pytest.mark.anyio
async def test_unavailable_cli_is_an_explicit_failure(tmp_path: Path) -> None:
    called = False

    async def fake_query(**kwargs: object) -> object:
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
        bundled_cli_resolver=lambda: None,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.PROVIDER_UNAVAILABLE
    assert result.error_code == "cli_unavailable"
    assert called is False


@pytest.mark.anyio
async def test_unavailable_sandbox_is_an_explicit_failure(tmp_path: Path) -> None:
    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        sandbox=_UnavailableSandbox(),
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "sandbox_unavailable"


@pytest.mark.anyio
async def test_insecure_cli_path_is_an_explicit_failure(tmp_path: Path) -> None:
    called = False

    async def fake_query(**kwargs: object) -> object:
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
        sandbox=_InsecureValidatingSandbox(),
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "sandbox_cli_path_insecure"
    assert "TASKCHAMBER_CLAUDE_CLI_PATH" in (result.error_message or "")
    assert called is False


@pytest.mark.anyio
async def test_hidden_certificate_path_is_an_explicit_failure(tmp_path: Path) -> None:
    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        sandbox=_InvalidEnvironmentSandbox(),
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "cli_environment_invalid"


@pytest.mark.anyio
async def test_launcher_rebind_rejection_maps_to_the_specific_error(
    tmp_path: Path,
) -> None:
    called = False

    async def fake_query(**kwargs: object) -> object:
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
        sandbox=_InsecureLauncherSandbox(),
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "sandbox_cli_path_insecure"
    assert called is False


@pytest.mark.anyio
async def test_launcher_construction_failure_maps_to_sandbox_setup(
    tmp_path: Path,
) -> None:
    called = False

    async def fake_query(**kwargs: object) -> object:
        nonlocal called
        called = True
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
        sandbox=_FailingLauncherSandbox(),
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "sandbox_setup_failed"
    assert called is False


@pytest.mark.anyio
async def test_sdk_exception_without_a_result_is_a_runtime_error(
    tmp_path: Path,
) -> None:
    async def failing_query(**kwargs: object) -> object:
        raise RuntimeError("sdk failure")
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=failing_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "runtime_error"
    assert result.output is None


@pytest.mark.anyio
async def test_stream_without_a_result_message_is_an_explicit_failure(
    tmp_path: Path,
) -> None:
    async def empty_query(**kwargs: object) -> object:
        if False:
            yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=empty_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "missing_result"


@pytest.mark.anyio
async def test_service_uses_an_injected_completion_handler(tmp_path: Path) -> None:
    async def handler(
        request: StructuredCompletionRequest,
        policy: ExecutionPolicy,
    ) -> StructuredCompletionResult:
        return StructuredCompletionResult(
            request_id=request.request_id,
            status=TaskStatus.FAILED,
            runtime="fake",
            provider="fake",
            error_code="custom_failure",
            error_message="injected handler outcome",
        )

    service = StructuredCompletionService(
        FakeRuntime(completion_handler=handler),
        ServerSettings(workspace_root=tmp_path),
    )

    result = await service.complete(_request(provider=None))

    assert result.status is TaskStatus.FAILED
    assert result.error_code == "custom_failure"
    assert result.effective_max_turns == DEFAULT_COMPLETION_MAX_TURNS


def test_cli_complete_rejects_malformed_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_file = tmp_path / "not-json.json"
    request_file.write_text("this is not json", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["complete", "--request", str(request_file)])

    assert excinfo.value.code == 2
    assert "error" in capsys.readouterr().err


def test_cli_complete_exits_one_on_a_failed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FailingService:
        async def complete(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            return StructuredCompletionResult(
                request_id=request.request_id,
                status=TaskStatus.FAILED,
                runtime="fake",
                provider="fake",
                error_code="custom_failure",
                error_message="completion failed",
            )

    monkeypatch.setattr(
        "taskchamber.application.composition.create_default_completion_service",
        lambda: _FailingService(),
    )
    request_file = tmp_path / "request.json"
    request_file.write_text(
        json.dumps(
            {
                "request_id": "cli-fail",
                "system_prompt": "Return JSON only.",
                "prompt": "Give me the data.",
                "json_schema": SCHEMA,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["complete", "--request", str(request_file)])

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "custom_failure"


@pytest.mark.anyio
async def test_completion_succeeds_without_sdk_version_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(package_name: str) -> str:
        assert package_name == "claude-agent-sdk"
        raise metadata.PackageNotFoundError(package_name)

    async def fake_query(**kwargs: object) -> object:
        yield _result_message(structured_output={"answer": "ok"})

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )
    monkeypatch.setattr(
        "taskchamber.runtimes.claude.runtime.metadata.version",
        missing_version,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.output == {"answer": "ok"}
    assert result.sdk_version is None
    assert result.error_code is None
    assert result.usage is not None
    assert result.usage.input_tokens == 10


@pytest.mark.anyio
async def test_system_messages_do_not_interrupt_structured_completion(tmp_path: Path) -> None:
    async def fake_query(**kwargs: object) -> object:
        yield SystemMessage(subtype="init", data={"session_id": "completion-session"})
        yield AssistantMessage(
            content=[TextBlock(text="intermediate")],
            model="observed-model",
        )
        yield SystemMessage(subtype="status", data={"status": None})
        yield _result_message(structured_output={"answer": "ok"})

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.request_id == "req-1"
    assert result.output == {"answer": "ok"}
    assert result.model == "observed-model"
    assert result.error_code is None


@pytest.mark.anyio
async def test_observed_model_comes_from_assistant_messages(tmp_path: Path) -> None:
    async def fake_query(**kwargs: object) -> object:
        yield AssistantMessage(
            content=[TextBlock(text="intermediate")],
            model="actually-observed-model",
        )
        yield _result_message(structured_output={"answer": "ok"})

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    # The observed model wins over the configured profile model.
    assert result.model == "actually-observed-model"


@pytest.mark.anyio
async def test_completion_hook_allows_internal_structured_output_only(
    tmp_path: Path,
) -> None:
    hook_decisions: list[dict[str, object]] = []

    async def fake_query(**kwargs: object) -> object:
        options = kwargs["options"]
        hook = options.hooks["PreToolUse"][0].hooks[0]
        # The SDK delivers output_format results through this internal call;
        # it must survive the zero-tool guard.
        hook_decisions.append(
            await hook(
                {
                    "tool_name": "StructuredOutput",
                    "tool_input": {"answer": "ok"},
                },
                None,
                {},
            )
        )
        # Real tools stay denied in the tool-free completion mode.
        hook_decisions.append(
            await hook(
                {"tool_name": "Read", "tool_input": {"file_path": "x.py"}},
                None,
                {},
            )
        )
        yield _result_message(structured_output={"answer": "ok"})

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=fake_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert hook_decisions[0] == {}
    assert hook_decisions[1]["hookSpecificOutput"]["permissionDecision"] == "deny"  # type: ignore[index]


@pytest.mark.anyio
async def test_cancellation_propagates_instead_of_becoming_a_result(
    tmp_path: Path,
) -> None:
    async def cancelled_query(**kwargs: object) -> object:
        raise asyncio.CancelledError
        yield

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=cancelled_query,
    )

    request = _request()
    policy = _policy(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await runtime.complete_structured(request, policy)


@pytest.mark.anyio
async def test_result_arriving_before_an_sdk_exception_is_still_mapped(
    tmp_path: Path,
) -> None:
    async def raising_query(**kwargs: object) -> object:
        yield _result_message(structured_output={"answer": "ok"})
        raise RuntimeError("sdk failure after the final result")

    runtime = ClaudeAgentSdkRuntime(
        environment={"Z_AI_API_KEY": "test-token"},
        query_function=raising_query,
    )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS
    assert result.output == {"answer": "ok"}


@pytest.mark.anyio
async def test_legacy_cli_resolver_path_is_honored(tmp_path: Path) -> None:
    cli = tmp_path / "bin" / "claude"
    cli.parent.mkdir()
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)

    async def fake_query(**kwargs: object) -> object:
        yield _result_message(structured_output={"answer": "ok"})

    with pytest.warns(DeprecationWarning):
        runtime = ClaudeAgentSdkRuntime(
            environment={"Z_AI_API_KEY": "test-token"},
            query_function=fake_query,
            cli_resolver=lambda _name: str(cli),
        )

    result = await runtime.complete_structured(_request(), _policy(tmp_path))

    assert result.status is TaskStatus.SUCCESS

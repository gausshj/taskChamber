from taskchamber.core.contracts import (
    ExecutionTelemetry,
    TaskKind,
    TaskResult,
    TaskStatus,
    TokenUsage,
    ToolCallDecision,
    ToolCallRecord,
)
from taskchamber.transport.mcp import render_legacy_result, render_metadata_only_result


def test_legacy_result_renders_minimal_success_exactly() -> None:
    result = TaskResult(
        run_id="run-minimal",
        kind=TaskKind.RESEARCH,
        status=TaskStatus.SUCCESS,
        output="done",
    )

    assert render_legacy_result(result) == (
        "[provider=unknown status=success]\n"
        "[result is_error=false partial=false num_turns=unknown duration_ms=0 "
        "input_tokens=unknown output_tokens=unknown "
        "effective_max_output_chars=unknown truncated=false error_code=none]\n\n"
        "done"
    )


def test_legacy_result_renders_complete_telemetry_exactly() -> None:
    result = TaskResult(
        run_id="run-complete",
        kind=TaskKind.REVIEW,
        status=TaskStatus.SUCCESS,
        output="complete",
        provider="provider-name",
        duration_ms=1_234,
        num_turns=2,
        usage=TokenUsage(
            input_tokens=11,
            output_tokens=5,
            cache_read_input_tokens=7,
            cache_creation_input_tokens=3,
            reasoning_output_tokens=2,
            total_tokens=28,
        ),
        execution=ExecutionTelemetry(
            allowed_tools=("Read", "Grep"),
            disallowed_tools=("Bash",),
            sandbox="bwrap",
            workspace_staged=False,
            os_isolated=True,
            cli_wrapper_active=True,
            cli_launch_observed=True,
            sandbox_preflight_passed=True,
            isolation_scope="agent_cli",
            runtime_process_isolated=False,
            cli_environment_sanitized=True,
            cli_executable_source="bundled",
            tool_calls=(
                ToolCallRecord(tool="Read", decision=ToolCallDecision.ALLOWED),
                ToolCallRecord(tool="Bash", decision=ToolCallDecision.DENIED),
            ),
            document_sources=("docs", "api"),
            document_tools=("DocumentRead",),
        ),
        cost_usd=0.012345,
        partial=True,
        truncated=True,
        effective_max_output_chars=800,
    )

    assert render_legacy_result(result) == (
        "[provider=provider-name status=success cost=$0.0123]\n"
        "[result is_error=false partial=true num_turns=2 duration_ms=1234 "
        "input_tokens=11 output_tokens=5 effective_max_output_chars=800 "
        "truncated=true error_code=none]\n"
        "[tokens input=11 output=5 cache_read=7 cache_create=3 reasoning=2 total=28]\n"
        "[execution sandbox=bwrap os_isolated=true staged=false wrapper=true "
        "cli_launch_observed=true sandbox_preflight_passed=true "
        "isolation_scope=agent_cli runtime_process_isolated=false "
        "cli_environment_sanitized=true cli_executable_source=bundled "
        "allowed=Read,Grep disallowed=Bash document_sources=docs,api "
        "document_tools=DocumentRead tool_calls=2 denied=1]\n\n"
        "complete"
    )


def test_legacy_result_renders_error_before_incomplete_output_exactly() -> None:
    result = TaskResult(
        run_id="run-error",
        kind=TaskKind.SUMMARIZE,
        status=TaskStatus.TURN_LIMIT_EXCEEDED,
        output="unfinished conclusion",
        num_turns=1,
        partial=True,
        error_code="turn_limit_exceeded",
        error_message="The agent exceeded the configured turn limit.",
    )

    assert render_legacy_result(result) == (
        "[provider=unknown status=turn_limit_exceeded]\n"
        "[result is_error=true partial=true num_turns=1 duration_ms=0 "
        "input_tokens=unknown output_tokens=unknown "
        "effective_max_output_chars=unknown truncated=false "
        "error_code=turn_limit_exceeded]\n\n"
        "The agent exceeded the configured turn limit.\n\n"
        "[incomplete partial output]\n"
        "unfinished conclusion"
    )


def test_legacy_result_uses_error_fallbacks() -> None:
    result = TaskResult(
        run_id="run-failed",
        kind=TaskKind.RESEARCH,
        status=TaskStatus.FAILED,
    )

    rendered = render_legacy_result(result)

    assert "error_code=failed" in rendered
    assert rendered.endswith("The task did not complete successfully.")


def test_legacy_result_omits_empty_optional_metadata_sections() -> None:
    result = TaskResult(
        run_id="run-empty-metadata",
        kind=TaskKind.RESEARCH,
        status=TaskStatus.SUCCESS,
        usage=TokenUsage(),
        execution=ExecutionTelemetry(),
    )

    rendered = render_legacy_result(result)

    assert "[tokens " not in rendered
    assert "[execution " not in rendered


def test_legacy_result_preserves_zero_and_false_telemetry() -> None:
    result = TaskResult(
        run_id="run-zero",
        kind=TaskKind.RESEARCH,
        status=TaskStatus.SUCCESS,
        output="done",
        provider="provider-name",
        num_turns=0,
        usage=TokenUsage(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            reasoning_output_tokens=0,
            total_tokens=0,
        ),
        execution=ExecutionTelemetry(
            sandbox="none",
            workspace_staged=False,
            os_isolated=False,
            cli_wrapper_active=False,
            tool_calls=(ToolCallRecord(tool="Read", decision=ToolCallDecision.ALLOWED),),
        ),
        cost_usd=0,
        effective_max_output_chars=1,
        error_code="ignored-on-success",
        error_message="ignored on success",
    )

    rendered = render_legacy_result(result)

    assert "cost=$0.0000" in rendered
    assert "num_turns=0" in rendered
    assert "error_code=none" in rendered
    assert "[tokens input=0 output=0 cache_read=0 cache_create=0 reasoning=0 total=0]" in rendered
    assert (
        "[execution sandbox=none os_isolated=false staged=false wrapper=false "
        "tool_calls=1 denied=0]"
    ) in rendered
    assert "ignored" not in rendered


def test_metadata_only_result_omits_success_body_but_keeps_metadata() -> None:
    result = TaskResult(
        run_id="run-compact",
        kind=TaskKind.RESEARCH,
        status=TaskStatus.SUCCESS,
        output="唯一正文-你好-Δ",
        provider="provider-name",
        num_turns=2,
        duration_ms=42,
        usage=TokenUsage(input_tokens=11, output_tokens=5, total_tokens=16),
        execution=ExecutionTelemetry(allowed_tools=("Read",)),
        truncated=True,
        partial=True,
        effective_max_output_chars=800,
    )

    rendered = render_metadata_only_result(result)

    assert "唯一正文-你好-Δ" not in rendered
    assert rendered == (
        "[provider=provider-name status=success]\n"
        "[result is_error=false partial=true num_turns=2 duration_ms=42 "
        "input_tokens=11 output_tokens=5 effective_max_output_chars=800 "
        "truncated=true error_code=none]\n"
        "[tokens input=11 output=5 total=16]\n"
        "[execution allowed=Read]"
    )


def test_metadata_only_result_reuses_the_same_metadata_lines_as_full_text() -> None:
    result = TaskResult(
        run_id="run-shared",
        kind=TaskKind.REVIEW,
        status=TaskStatus.SUCCESS,
        output="body",
        provider="provider-name",
        usage=TokenUsage(input_tokens=1),
        execution=ExecutionTelemetry(allowed_tools=("Read", "Grep")),
    )

    full = render_legacy_result(result)
    compact = render_metadata_only_result(result)

    assert full == f"{compact}\n\nbody"

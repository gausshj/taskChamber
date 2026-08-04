"""Compatibility text rendering for MCP clients without structured output."""

from __future__ import annotations

from ..core.contracts import ExecutionTelemetry, TaskResult, TokenUsage


def render_legacy_result(result: TaskResult) -> str:
    """Render the stable text result alongside the canonical structured result."""

    is_error = result.is_error
    metadata_text = render_result_metadata(result)
    return f"{metadata_text}\n\n{_render_body(result, is_error=is_error)}".rstrip()


def render_metadata_only_result(result: TaskResult) -> str:
    """Render only the audit/status metadata lines, without the generated body."""

    return render_result_metadata(result)


def render_result_metadata(result: TaskResult) -> str:
    """Render the header and metadata lines shared by both text modes."""

    is_error = result.is_error
    metadata = [
        _render_header(result),
        _render_result_metadata(result, is_error=is_error),
    ]
    optional_metadata = (
        _render_token_metadata(result.usage),
        _render_execution_metadata(result.execution),
    )
    metadata.extend(line for line in optional_metadata if line is not None)
    return "\n".join(metadata)


def _render_header(result: TaskResult) -> str:
    provider = result.provider or "unknown"
    header = f"[provider={provider} status={result.status.value}"
    if result.cost_usd is not None:
        header += f" cost=${result.cost_usd:.4f}"
    return header + "]"


def _render_result_metadata(result: TaskResult, *, is_error: bool) -> str:
    usage = result.usage
    input_tokens = (
        usage.input_tokens if usage is not None and usage.input_tokens is not None else "unknown"
    )
    output_tokens = (
        usage.output_tokens if usage is not None and usage.output_tokens is not None else "unknown"
    )
    output_limit = result.effective_max_output_chars or "unknown"
    error_code = (result.error_code or result.status.value) if is_error else "none"
    fields = [
        f"is_error={str(is_error).lower()}",
        f"partial={str(result.partial).lower()}",
        f"num_turns={result.num_turns if result.num_turns is not None else 'unknown'}",
        f"duration_ms={result.duration_ms}",
        f"input_tokens={input_tokens}",
        f"output_tokens={output_tokens}",
        f"effective_max_output_chars={output_limit}",
        f"truncated={str(result.truncated).lower()}",
        f"error_code={error_code}",
    ]
    return f"[result {' '.join(fields)}]"


def _render_token_metadata(usage: TokenUsage | None) -> str | None:
    if usage is None:
        return None
    fields = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens,
        "cache_create": usage.cache_creation_input_tokens,
        "reasoning": usage.reasoning_output_tokens,
        "total": usage.total_tokens,
    }
    reported = " ".join(f"{name}={value}" for name, value in fields.items() if value is not None)
    return f"[tokens {reported}]" if reported else None


def _render_execution_metadata(execution: ExecutionTelemetry | None) -> str | None:
    if execution is None:
        return None
    fields: list[str] = []
    if execution.sandbox is not None:
        fields.append(f"sandbox={execution.sandbox}")
    if execution.os_isolated is not None:
        fields.append(f"os_isolated={str(execution.os_isolated).lower()}")
    if execution.workspace_staged is not None:
        fields.append(f"staged={str(execution.workspace_staged).lower()}")
    if execution.cli_wrapper_active is not None:
        fields.append(f"wrapper={str(execution.cli_wrapper_active).lower()}")
    if execution.cli_launch_observed is not None:
        fields.append(f"cli_launch_observed={str(execution.cli_launch_observed).lower()}")
    if execution.sandbox_preflight_passed is not None:
        fields.append(f"sandbox_preflight_passed={str(execution.sandbox_preflight_passed).lower()}")
    if execution.isolation_scope is not None:
        fields.append(f"isolation_scope={execution.isolation_scope}")
    if execution.runtime_process_isolated is not None:
        fields.append(f"runtime_process_isolated={str(execution.runtime_process_isolated).lower()}")
    if execution.cli_environment_sanitized is not None:
        fields.append(
            f"cli_environment_sanitized={str(execution.cli_environment_sanitized).lower()}"
        )
    if execution.cli_executable_source is not None:
        fields.append(f"cli_executable_source={execution.cli_executable_source}")
    if execution.allowed_tools:
        fields.append(f"allowed={','.join(execution.allowed_tools)}")
    if execution.disallowed_tools:
        fields.append(f"disallowed={','.join(execution.disallowed_tools)}")
    if execution.document_sources:
        fields.append(f"document_sources={','.join(execution.document_sources)}")
    if execution.document_tools:
        fields.append(f"document_tools={','.join(execution.document_tools)}")
    if execution.tool_calls:
        denied = sum(call.decision.value == "denied" for call in execution.tool_calls)
        fields.append(f"tool_calls={len(execution.tool_calls)}")
        fields.append(f"denied={denied}")
    return f"[execution {' '.join(fields)}]" if fields else None


def _render_body(result: TaskResult, *, is_error: bool) -> str:
    if not is_error:
        return result.output
    body = result.error_message or "The task did not complete successfully."
    if result.output:
        body += "\n\n[incomplete partial output]\n" + result.output
    return body


__all__ = ["render_legacy_result", "render_metadata_only_result", "render_result_metadata"]

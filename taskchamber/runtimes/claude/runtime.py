"""Claude Agent SDK implementation of the provider-neutral runtime port."""

from __future__ import annotations

import asyncio
import os
import stat
import time
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    TextBlock,
    query,
)

from ...config import (
    MappingSecretProvider,
    ProviderProfile,
    SecretProvider,
    secret_references,
)
from ...core.contracts import (
    AgentCapabilities,
    ExecutionPolicy,
    ExecutionTelemetry,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TokenUsage,
    ToolCallDecision,
    ToolCallRecord,
)
from ...core.policy import PolicyDeniedError, WorkspaceGuard
from ...isolation import (
    CLI_LAUNCH_OBSERVATION_FILE,
    IsolatedWorkspace,
    NoSandbox,
    Sandbox,
)
from .cli import (
    ClaudeCliExecutable,
    ClaudeCliUnavailableError,
    bundled_claude_cli_path,
    resolve_claude_cli,
)
from .documents import (
    CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY,
    DOCUMENT_MCP_SERVER,
    create_document_mcp_server,
)
from .profiles import (
    CLAUDE_CODE_DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_PROVIDER_PROFILES,
)

QueryFunction = Callable[..., AsyncIterator[object]]
BundledCliResolver = Callable[[], Path | None]

CLAUDE_CLI_ENVIRONMENT_ALLOWLIST = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "CLAUDE_AGENT_SDK_VERSION",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CONFIG_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NODE_EXTRA_CA_CERTS",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TRACEPARENT",
    "TRACESTATE",
    "TZ",
)
_CLI_FILE_PATH_ENVIRONMENT_KEYS = (
    "NODE_EXTRA_CA_CERTS",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
)
_CLI_DIRECTORY_PATH_ENVIRONMENT_KEYS = ("SSL_CERT_DIR",)


@dataclass(frozen=True)
class _CliLaunch:
    path: Path
    executable: ClaudeCliExecutable
    observation_path: Path
    observation_inside_os_sandbox: bool
    wrapper_active: bool = True
    environment_sanitized: bool = True


class ProviderSelectionError(PolicyDeniedError):
    """A profile cannot be used by this runtime without protocol translation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ClaudeAgentSdkRuntime:
    """Run one constrained Claude Agent SDK query for each MCP task.

    The SDK merges ``options.env`` with the parent process environment. This
    adapter therefore launches the CLI through an allowlisting wrapper. The
    Python SDK itself still runs in the hosting process, which must use a minimal
    environment and account if SDK supply-chain compromise is in scope.
    """

    name = "claude-agent-sdk"
    default_profile = DEFAULT_PROVIDER
    capabilities = AgentCapabilities(
        read_workspace=True,
        read_documents=True,
        cancellation=True,
        progress=False,
        structured_output=False,
    )

    def __init__(
        self,
        *,
        providers: Mapping[str, ProviderProfile] | None = None,
        secrets: SecretProvider | None = None,
        environment: Mapping[str, str] | None = None,
        query_function: QueryFunction = query,
        sandbox: Sandbox | None = None,
        configured_cli_path: str | None = None,
        bundled_cli_resolver: BundledCliResolver = bundled_claude_cli_path,
        cli_resolver: Callable[[str], str | None] | None = None,
        default_profile: str = DEFAULT_PROVIDER,
    ) -> None:
        if secrets is not None and environment is not None:
            raise ValueError("pass either secrets or environment, not both")
        if configured_cli_path is not None and cli_resolver is not None:
            raise ValueError("configured_cli_path and cli_resolver are mutually exclusive")
        if cli_resolver is not None:
            warnings.warn(
                "cli_resolver is deprecated; configure an absolute "
                "TASKCHAMBER_CLAUDE_CLI_PATH instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self._providers = dict(providers or DEFAULT_PROVIDER_PROFILES)
        self._secrets = secrets or MappingSecretProvider(
            environment if environment is not None else os.environ
        )
        self._query_function = query_function
        self._sandbox = sandbox or NoSandbox()
        self._configured_cli_path = configured_cli_path
        self._bundled_cli_resolver = bundled_cli_resolver
        self._legacy_cli_resolver = cli_resolver
        self.default_profile = default_profile

    def build_options(
        self,
        request: TaskRequest,
        policy: ExecutionPolicy,
        *,
        config_dir: Path,
        workspace: IsolatedWorkspace | None = None,
        cli_path: str | None = None,
        tool_audit: list[ToolCallRecord] | None = None,
    ) -> ClaudeAgentOptions:
        """Construct the SDK options in one inspectable, testable location.

        When a sandbox supplies ``workspace``, the read-only guard and the SDK
        working directory point at the staged tree rather than the source root,
        so the PreToolUse hook validates paths exactly as the sandboxed process
        sees them.
        """

        profile = self._provider_for(request.provider)
        active = workspace or IsolatedWorkspace(
            root=policy.workspace_root,
            allowed_paths=policy.allowed_paths,
        )
        guard = WorkspaceGuard(
            root=active.root,
            max_file_bytes=policy.max_file_bytes,
            allowed_tools=policy.allowed_tools,
            allowed_paths=active.allowed_paths,
        )
        document_tools: tuple[str, ...] = ()
        mcp_servers: dict[str, Any] = {}
        if policy.document_catalog is not None:
            document_tools = tuple(
                CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY[name]
                for name in policy.document_tools
                if name in CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY
            )
            mcp_servers[DOCUMENT_MCP_SERVER] = create_document_mcp_server(policy.document_catalog)
        effective_tools = list(policy.allowed_tools + document_tools)

        return ClaudeAgentOptions(
            system_prompt=policy.system_prompt,
            model=self._model_for(profile),
            tools=effective_tools,
            allowed_tools=effective_tools,
            disallowed_tools=list(policy.disallowed_tools),
            permission_mode="dontAsk",
            max_turns=policy.max_turns,
            max_budget_usd=policy.max_budget_usd,
            cwd=active.root,
            cli_path=cli_path,
            setting_sources=[],
            skills=[],
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            env=self.environment_for(profile, config_dir=config_dir),
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[
                            self._pre_tool_use(
                                guard,
                                tool_audit,
                                document_tools=frozenset(document_tools),
                            )
                        ],
                    )
                ]
            },
            extra_args={"no-session-persistence": None},
            stderr=lambda _message: None,
        )

    def environment_for(self, profile: ProviderProfile, *, config_dir: Path) -> dict[str, str]:
        """Build a per-call environment without exposing other provider keys."""

        token = self._secrets.get(profile.credential_ref) or ""
        env = {reference: "" for reference in secret_references(tuple(self._providers.values()))}
        env.update(
            {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_BASE_URL": "",
                "ANTHROPIC_MODEL": self._model_for(profile) or "",
                "CLAUDE_CONFIG_DIR": str(config_dir),
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            }
        )
        if profile.base_url is not None:
            env["ANTHROPIC_BASE_URL"] = profile.base_url
        if profile.api_key_field == "api_key":
            env["ANTHROPIC_API_KEY"] = token
        else:
            env["ANTHROPIC_AUTH_TOKEN"] = token
        return env

    def _prepare_cli_launch(
        self,
        workspace: IsolatedWorkspace,
        *,
        config_dir: Path,
        launcher_dir: Path,
        executable: ClaudeCliExecutable,
    ) -> _CliLaunch:
        """Build one immutable, clean-environment launcher for the exact CLI."""

        path = self._sandbox.prepare_cli_launcher(
            workspace,
            executable=str(executable.path),
            config_dir=config_dir,
            launcher_dir=launcher_dir,
            environment_keys=CLAUDE_CLI_ENVIRONMENT_ALLOWLIST,
        )
        return _CliLaunch(
            path=path,
            executable=executable,
            observation_path=config_dir / CLI_LAUNCH_OBSERVATION_FILE,
            observation_inside_os_sandbox=(
                self._sandbox.launch_observation_inside_os_sandbox
                and self._sandbox.uses_secure_cli_launcher
            ),
        )

    async def run(self, request: TaskRequest, policy: ExecutionPolicy) -> TaskResult:
        """Execute a fresh SDK query and normalize its final result."""

        try:
            profile = self._provider_for(request.provider)
        except ProviderSelectionError as exc:
            return self._failure(
                request,
                status=TaskStatus.PROVIDER_UNAVAILABLE,
                error_code=exc.code,
                message=str(exc),
            )

        if self._secrets.get(profile.credential_ref) is None:
            return self._failure(
                request,
                status=TaskStatus.PROVIDER_UNAVAILABLE,
                error_code="missing_credential",
                message="The requested provider credential is not available.",
                model=self._model_for(profile),
            )

        sandbox_preflight_passed: bool | None = None
        if self._sandbox.os_isolated:
            sandbox_available = self._sandbox.preflight()
            if self._sandbox.operational_preflight and self._sandbox.uses_secure_cli_launcher:
                sandbox_preflight_passed = sandbox_available
            if not sandbox_available:
                return self._failure(
                    request,
                    status=TaskStatus.FAILED,
                    error_code="sandbox_unavailable",
                    message="The requested OS sandbox is unavailable.",
                    model=self._model_for(profile),
                    execution=self._unavailable_execution_telemetry(
                        policy,
                        sandbox_preflight_passed=(
                            False
                            if self._sandbox.operational_preflight
                            and self._sandbox.uses_secure_cli_launcher
                            else None
                        ),
                    ),
                )

        try:
            if self._legacy_cli_resolver is not None:
                executable = resolve_claude_cli(
                    self._legacy_cli_resolver("claude"),
                    bundled_resolver=lambda: None,
                )
            else:
                executable = resolve_claude_cli(
                    self._configured_cli_path,
                    bundled_resolver=self._bundled_cli_resolver,
                )
        except (ClaudeCliUnavailableError, OSError, RuntimeError, ValueError):
            return self._failure(
                request,
                status=TaskStatus.PROVIDER_UNAVAILABLE,
                error_code="cli_unavailable",
                message=(
                    "The Claude CLI is unavailable; install the pinned SDK wheel or "
                    "configure an explicit executable."
                ),
                model=self._model_for(profile),
                execution=self._unavailable_execution_telemetry(
                    policy,
                    sandbox_preflight_passed=sandbox_preflight_passed,
                ),
            )

        try:
            self._sandbox.validate_readable_paths(self._forwarded_environment_paths())
        except ValueError:
            return self._failure(
                request,
                status=TaskStatus.FAILED,
                error_code="cli_environment_invalid",
                message=(
                    "A forwarded CLI certificate path is unavailable inside the selected boundary."
                ),
                model=self._model_for(profile),
                execution=self._unavailable_execution_telemetry(
                    policy,
                    sandbox_preflight_passed=sandbox_preflight_passed,
                ),
            )

        started = time.monotonic()
        final: ResultMessage | None = None
        partial_text: list[str] = []
        observed_model: str | None = None
        tool_audit: list[ToolCallRecord] = []

        with TemporaryDirectory(prefix="taskchamber-") as temp_dir:
            task_dir = Path(temp_dir)
            config_dir = task_dir / "config"
            launcher_dir = task_dir / "launcher"
            with self._sandbox.isolate(policy) as workspace:
                try:
                    launch = self._prepare_cli_launch(
                        workspace,
                        config_dir=config_dir,
                        launcher_dir=launcher_dir,
                        executable=executable,
                    )
                except (OSError, ValueError):
                    return self._failure(
                        request,
                        status=TaskStatus.FAILED,
                        error_code="sandbox_setup_failed",
                        message="The runtime could not establish the requested CLI boundary.",
                        model=self._model_for(profile),
                        duration_ms=self._elapsed_ms(started),
                        max_output_chars=policy.max_output_chars,
                        execution=self._execution_telemetry(
                            policy,
                            workspace,
                            None,
                            tool_audit,
                            sandbox_preflight_passed=sandbox_preflight_passed,
                        ),
                    )
                options = self.build_options(
                    request,
                    policy,
                    config_dir=config_dir,
                    workspace=workspace,
                    cli_path=str(launch.path),
                    tool_audit=tool_audit,
                )
                stream = self._query_function(prompt=request.prompt, options=options)
                try:
                    async with asyncio.timeout(policy.timeout_seconds):
                        async for message in stream:
                            if isinstance(message, AssistantMessage):
                                observed_model = message.model or observed_model
                                partial_text.extend(
                                    block.text
                                    for block in message.content
                                    if isinstance(block, TextBlock)
                                )
                            elif isinstance(message, ResultMessage):
                                final = message
                except TimeoutError:
                    return self._failure(
                        request,
                        status=TaskStatus.TIMED_OUT,
                        error_code="timeout",
                        message="The agent task exceeded the configured time limit.",
                        model=self._model_for(profile) or observed_model,
                        duration_ms=self._elapsed_ms(started),
                        output="".join(partial_text),
                        partial=bool(partial_text),
                        max_output_chars=policy.max_output_chars,
                        execution=self._execution_telemetry(
                            policy,
                            workspace,
                            launch,
                            tool_audit,
                            sandbox_preflight_passed=sandbox_preflight_passed,
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if final is None:
                        return self._failure(
                            request,
                            status=TaskStatus.FAILED,
                            error_code="runtime_error",
                            message="The agent runtime could not complete the task.",
                            model=self._model_for(profile) or observed_model,
                            duration_ms=self._elapsed_ms(started),
                            output="".join(partial_text),
                            partial=bool(partial_text),
                            max_output_chars=policy.max_output_chars,
                            execution=self._execution_telemetry(
                                policy,
                                workspace,
                                launch,
                                tool_audit,
                                sandbox_preflight_passed=sandbox_preflight_passed,
                            ),
                        )
                    # A final ResultMessage arrived before the SDK raised (it raises
                    # after a turn-limit or budget result); map that result below
                    # instead of discarding it as a generic runtime error.
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()

                execution = self._execution_telemetry(
                    policy,
                    workspace,
                    launch,
                    tool_audit,
                    sandbox_preflight_passed=sandbox_preflight_passed,
                )

        if final is None:
            return self._failure(
                request,
                status=TaskStatus.FAILED,
                error_code="missing_result",
                message="The agent runtime ended without a final result.",
                model=self._model_for(profile) or observed_model,
                duration_ms=self._elapsed_ms(started),
                output="".join(partial_text),
                partial=bool(partial_text),
                max_output_chars=policy.max_output_chars,
                execution=execution,
            )

        output = final.result or "".join(partial_text)
        status = self._status_for(final)
        truncated = len(output) > policy.max_output_chars
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=status,
            output=self._truncate(output, policy),
            runtime=self.name,
            provider=profile.name,
            model=self._model_for(profile) or observed_model,
            duration_ms=max(final.duration_ms, self._elapsed_ms(started)),
            num_turns=final.num_turns,
            usage=self._token_usage(final.usage),
            model_usage=self._model_token_usage(final.model_usage),
            execution=execution,
            cost_usd=final.total_cost_usd,
            partial=(status is not TaskStatus.SUCCESS and bool(output)) or truncated,
            truncated=truncated,
            effective_max_output_chars=policy.max_output_chars,
            error_code=None if status is TaskStatus.SUCCESS else self._error_code_for(status),
            error_message=(
                None if status is TaskStatus.SUCCESS else self._error_message_for(status)
            ),
        )

    def _provider_for(self, name: str) -> ProviderProfile:
        try:
            profile = self._providers[name]
        except KeyError as exc:
            raise ProviderSelectionError(
                "unknown_provider",
                "The requested provider is not configured for this runtime.",
            ) from exc
        if profile.runtime not in {"claude", "claude-agent-sdk"}:
            raise ProviderSelectionError(
                "incompatible_runtime",
                "The requested provider belongs to a different agent runtime.",
            )
        if profile.api_format != "anthropic":
            raise ProviderSelectionError(
                "unsupported_api_format",
                "The Claude runtime requires an Anthropic-compatible provider; "
                "translation is disabled.",
            )
        return profile

    @staticmethod
    def _token_usage(raw: object) -> TokenUsage | None:
        if not isinstance(raw, Mapping):
            return None
        usage = TokenUsage(
            input_tokens=ClaudeAgentSdkRuntime._counter(raw, "input_tokens", "inputTokens"),
            output_tokens=ClaudeAgentSdkRuntime._counter(
                raw,
                "output_tokens",
                "outputTokens",
            ),
            cache_read_input_tokens=ClaudeAgentSdkRuntime._counter(
                raw,
                "cache_read_input_tokens",
                "cacheReadInputTokens",
            ),
            cache_creation_input_tokens=ClaudeAgentSdkRuntime._counter(
                raw,
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
            ),
            reasoning_output_tokens=ClaudeAgentSdkRuntime._counter(
                raw,
                "reasoning_output_tokens",
                "reasoningOutputTokens",
                "reasoning_tokens",
                "reasoningTokens",
            ),
            total_tokens=ClaudeAgentSdkRuntime._counter(
                raw,
                "total_tokens",
                "totalTokens",
            ),
        )
        return usage if any(value is not None for value in usage.model_dump().values()) else None

    @staticmethod
    def _model_token_usage(raw: object) -> dict[str, TokenUsage] | None:
        if not isinstance(raw, Mapping):
            return None
        normalized: dict[str, TokenUsage] = {}
        for model, model_data in raw.items():
            if not isinstance(model, str) or not model:
                continue
            usage = ClaudeAgentSdkRuntime._token_usage(model_data)
            if usage is not None:
                normalized[model] = usage
        return normalized or None

    @staticmethod
    def _counter(raw: Mapping[object, object], *keys: str) -> int | None:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    @staticmethod
    def _model_for(profile: ProviderProfile) -> str | None:
        if profile.model == CLAUDE_CODE_DEFAULT_MODEL:
            return None
        return profile.model

    def _execution_telemetry(
        self,
        policy: ExecutionPolicy,
        workspace: IsolatedWorkspace,
        launch: _CliLaunch | None,
        tool_audit: list[ToolCallRecord],
        *,
        sandbox_preflight_passed: bool | None,
    ) -> ExecutionTelemetry:
        document_tools = tuple(
            CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY[name]
            for name in policy.document_tools
            if name in CLAUDE_DOCUMENT_TOOL_BY_CAPABILITY
        )
        cli_launch_observed = self._launch_observed(launch)
        os_isolated = bool(
            self._sandbox.os_isolated
            and sandbox_preflight_passed
            and launch is not None
            and launch.observation_inside_os_sandbox
            and cli_launch_observed
        )
        environment_sanitized = bool(
            launch is not None and launch.environment_sanitized and cli_launch_observed
        )
        return ExecutionTelemetry(
            allowed_tools=policy.allowed_tools + document_tools,
            disallowed_tools=policy.disallowed_tools,
            sandbox=self._sandbox.name,
            workspace_staged=workspace.root.resolve() != policy.workspace_root.resolve(),
            os_isolated=os_isolated,
            cli_wrapper_active=launch.wrapper_active if launch is not None else False,
            cli_launch_observed=cli_launch_observed,
            sandbox_preflight_passed=sandbox_preflight_passed,
            isolation_scope=("agent_cli" if os_isolated else "none"),
            runtime_process_isolated=False,
            cli_environment_sanitized=environment_sanitized,
            cli_executable_source=(launch.executable.source if launch is not None else None),
            tool_calls=tuple(tool_audit),
            document_sources=policy.document_sources,
            document_tools=document_tools,
        )

    @staticmethod
    def _forwarded_environment_paths() -> tuple[Path, ...]:
        """Collect path-valued CLI settings without returning their values."""

        values = [os.environ[key] for key in _CLI_FILE_PATH_ENVIRONMENT_KEYS if os.environ.get(key)]
        for key in _CLI_DIRECTORY_PATH_ENVIRONMENT_KEYS:
            value = os.environ.get(key)
            if value:
                values.extend(part for part in value.split(os.pathsep) if part)
        return tuple(Path(value).expanduser() for value in values)

    @staticmethod
    def _launch_observed(launch: _CliLaunch | None) -> bool:
        """Recognize the one-shot marker written inside the effective boundary."""

        if launch is None:
            return False
        try:
            marker = launch.observation_path.lstat()
        except OSError:
            return False
        return (
            stat.S_ISREG(marker.st_mode)
            and marker.st_uid == os.getuid()
            and marker.st_mode & 0o077 == 0
        )

    def _unavailable_execution_telemetry(
        self,
        policy: ExecutionPolicy,
        *,
        sandbox_preflight_passed: bool | None = None,
    ) -> ExecutionTelemetry:
        return ExecutionTelemetry(
            allowed_tools=policy.allowed_tools,
            disallowed_tools=policy.disallowed_tools,
            sandbox=self._sandbox.name,
            workspace_staged=False,
            os_isolated=False,
            cli_wrapper_active=False,
            cli_launch_observed=False,
            sandbox_preflight_passed=sandbox_preflight_passed,
            isolation_scope="none",
            runtime_process_isolated=False,
            cli_environment_sanitized=False,
            document_sources=policy.document_sources,
        )

    @staticmethod
    def _pre_tool_use(
        guard: WorkspaceGuard,
        tool_audit: list[ToolCallRecord] | None = None,
        *,
        document_tools: frozenset[str] = frozenset(),
    ) -> Callable[[Any, str | None, Any], Any]:
        async def enforce(
            input_data: Any,
            _tool_use_id: str | None,
            _context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
            tool = tool_name if isinstance(tool_name, str) and tool_name else "unknown"
            if tool in document_tools:
                if tool_audit is not None:
                    tool_audit.append(ToolCallRecord(tool=tool, decision=ToolCallDecision.ALLOWED))
                return {}
            try:
                guard.validate_tool_call(
                    input_data["tool_name"],
                    input_data["tool_input"],
                )
            except Exception:
                if tool_audit is not None:
                    tool_audit.append(ToolCallRecord(tool=tool, decision=ToolCallDecision.DENIED))
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "workspace policy rejected this tool call",
                    }
                }
            if tool_audit is not None:
                tool_audit.append(ToolCallRecord(tool=tool, decision=ToolCallDecision.ALLOWED))
            return {}

        return enforce

    @staticmethod
    def _status_for(result: ResultMessage) -> TaskStatus:
        subtype = result.subtype.lower()
        if "max_budget" in subtype:
            return TaskStatus.BUDGET_EXCEEDED
        if "max_turn" in subtype:
            return TaskStatus.TURN_LIMIT_EXCEEDED
        if result.is_error or subtype != "success":
            return TaskStatus.FAILED
        return TaskStatus.SUCCESS

    @staticmethod
    def _error_code_for(status: TaskStatus) -> str:
        if status in {TaskStatus.TURN_LIMIT_EXCEEDED, TaskStatus.BUDGET_EXCEEDED}:
            return status.value
        return "agent_result_error"

    @staticmethod
    def _error_message_for(status: TaskStatus) -> str:
        if status is TaskStatus.TURN_LIMIT_EXCEEDED:
            return "The agent exceeded the configured turn limit."
        if status is TaskStatus.BUDGET_EXCEEDED:
            return "The agent exceeded the configured budget limit."
        return "The agent runtime reported an unsuccessful result."

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1_000))

    @staticmethod
    def _truncate(value: str, policy: ExecutionPolicy) -> str:
        if len(value) <= policy.max_output_chars:
            return value
        marker = "\n\n[output truncated by server policy]"
        if policy.max_output_chars <= len(marker):
            return value[: policy.max_output_chars]
        return value[: policy.max_output_chars - len(marker)] + marker

    def _failure(
        self,
        request: TaskRequest,
        *,
        status: TaskStatus,
        error_code: str,
        message: str,
        model: str | None = None,
        duration_ms: int = 0,
        output: str = "",
        partial: bool = False,
        max_output_chars: int | None = None,
        execution: ExecutionTelemetry | None = None,
    ) -> TaskResult:
        truncated = False
        if max_output_chars is not None and len(output) > max_output_chars:
            truncated = True
            marker = "\n\n[output truncated by server policy]"
            if max_output_chars <= len(marker):
                output = output[:max_output_chars]
            else:
                output = output[: max_output_chars - len(marker)] + marker
        return TaskResult(
            run_id=request.run_id,
            kind=request.kind,
            status=status,
            output=output,
            runtime=self.name,
            provider=request.provider,
            model=model,
            duration_ms=duration_ms,
            execution=execution,
            partial=partial or truncated,
            truncated=truncated,
            effective_max_output_chars=max_output_chars,
            error_code=error_code,
            error_message=message,
        )

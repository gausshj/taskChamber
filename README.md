# TaskChamber

**Isolated agent tasks over MCP.**

`taskchamber` exposes bounded sub-agent tasks as a standard stdio MCP server.
It keeps the MCP contract independent from the agent runtime that performs the
work, so Claude Agent SDK is the first adapter rather than a permanent
dependency of the server core.

## Installation

Install the local MCP executable and the current Claude runtime adapter in one
isolated, persistent uv tool environment:

```bash
uv tool install --python 3.11 'taskchamber[claude]'
```

This exposes the `taskchamber` command. It does not modify Claude Code, Codex,
or another MCP client, and it does not copy provider credentials. Client
registration and host-owned runtime configuration remain separate steps; see
[`docs/MCP_CLIENT_SETUP.md`](https://github.com/gausshj/taskChamber/blob/main/docs/MCP_CLIENT_SETUP.md).

For reproducible deployments, pin a released version:

```bash
uv tool install --python 3.11 'taskchamber[claude]==0.1.0'
```

The bundled Claude adapter pins the exact Agent SDK version exercised by the
release checks. SDK upgrades are shipped as reviewed TaskChamber releases
rather than silently changing an existing adapter installation.

## Architecture

```text
MCP client (Claude Code, Codex, or another client)
    -> FastMCP transport and stable task tools
    -> TaskService: validation, policy, limits, normalized results
    -> named DocumentSource registry (optional directories or fixed CLI argv)
    -> RuntimeRegistry -> AgentRuntime adapter
       -> ClaudeAgentSdkRuntime | FakeRuntime | future adapter

taskchamber.toml -> project capability ceiling + task defaults + document sources
Local .env / process env -> provider profiles + SecretProvider -> runtime factory
```

The implementation follows those boundaries in the package layout:

```text
taskchamber/
├── application/          # composition root; selects configured ports
├── config/               # provider profiles and secret sources
├── core/                 # contracts, policy, and TaskService
├── isolation/            # workspace staging and OS sandbox adapters
├── runtimes/
│   ├── registry.py       # runtime factory contract and plugin discovery
│   ├── builtins.py       # lazy built-in adapter targets
│   ├── claude/           # Claude SDK: factory, profiles, runtime
│   └── fake/             # credential-free test adapter
└── transport/            # stdio FastMCP adapter
```

Implementation modules live only in the package paths shown above. The
repository-level `taskchamber_mcp.py` is a convenience entry point for
source-tree experiments; installed environments can use the `taskchamber`
console command or `python -m taskchamber`.

The public tools remain deliberately task-shaped:

| Tool | Purpose |
| --- | --- |
| `research` | Investigate a focused question in the workspace and/or selected virtual document sources. |
| `summarize` | Summarize an allowed workspace file and/or selected virtual documents. |
| `review` | Review one or more selected workspace files and optional virtual documents. |

Each response keeps the legacy text header such as
`[provider=claude_code status=success]` and also returns a structured `TaskResult` in
MCP `structuredContent`.

By default (`TASKCHAMBER_MCP_TEXT_MODE=full`) the text block repeats the complete
output below the metadata header, so clients without structured-result support
keep working unchanged. Clients that consume both representations (for example,
clients that merge text and `structuredContent` into one tool message) can set
`TASKCHAMBER_MCP_TEXT_MODE=metadata_only` to receive the generated body exactly
once: successful responses then carry only the provider/status/result/tokens/
execution metadata lines in text, while `structuredContent` remains the
canonical complete `TaskResult` including `output`. Error responses always keep
the full legacy text, including `error_message` and any incomplete partial
output, because some clients ignore structured content on error paths. See
[`docs/MCP_CLIENT_SETUP.md`](https://github.com/gausshj/taskChamber/blob/main/docs/MCP_CLIENT_SETUP.md)
for client guidance.

`max_output_chars` bounds `TaskResult.output` only. In the default `full` mode
the wire envelope also contains the legacy text copy, so the total response
size is not limited to `max_output_chars`; use `metadata_only` when the
client-visible envelope size must track the configured output bound.

The validated effective limit is also passed to the runtime before generation:
the server-owned system prompt asks the model for a complete response within
that budget (`research`, `summarize`, and `review` alike). This is generation
guidance, not enforcement — the hard character cap still applies after
generation, and over-limit output keeps the `partial=true` / `truncated=true`
signals (plus the `[output truncated by server policy]` marker when the
effective limit is large enough to hold it).

All three tasks can use the project-configured capability policy. `research`
can select named external directory or CLI/API-backed document sources and set
`include_workspace=false`; `summarize` and `review` may use them as selected
inputs. These documents are unified behind a read-only virtual catalog and are
not copied into the staged workspace. MCP callers cannot submit arbitrary host
paths or commands. See
[`docs/DOCUMENT_SOURCES.md`](https://github.com/gausshj/taskChamber/blob/main/docs/DOCUMENT_SOURCES.md) for configuration, CLI
JSON output, limits, and a credential-free fixture.

Provider-reported token counters are normalized into `usage` and optional
per-model `model_usage`. A `[tokens ...]` text line is also emitted for MCP
clients that do not expose structured results. `cost_usd` remains available for
compatibility, but it is only provider-reported reference metadata: pricing,
subscriptions, pricing arrangements, and gateways are not comparable.

## Local setup

```bash
uv sync --all-groups
uv run pre-commit install
uv run pytest -q
```

Run the same repository checks without installing a Git hook with
`uv run pre-commit run --all-files`.

For a runtime-only installation, install the adapter explicitly:

```bash
uv sync --no-dev --extra claude
```

Start a no-network, deterministic server for protocol experiments:

```bash
TASKCHAMBER_RUNTIME=fake uv run python taskchamber_mcp.py
```

The equivalent module and installed console entry points are
`uv run python -m taskchamber` and `uv run taskchamber`.

## Project policy and caller-selected scope

TaskChamber provides broad read-only functionality; a concrete project performs
a second, manual configuration step to establish its maximum boundary:

```bash
taskchamber config init
taskchamber policy validate
taskchamber policy show
```

The resulting `taskchamber.toml` contains no credential values. It controls the
project capability ceiling, per-task allowed/default capabilities, relative
workspace include/exclude patterns, and named document sources. The management
CLI edits that same TOML atomically:

```bash
taskchamber policy deny review documents.search
taskchamber policy set-default review workspace.read workspace.search
```

An MCP caller may then provide `workspace_paths`, `requested_capabilities`, and
structured `document_requests` to narrow one call. It cannot broaden the project
or task policy, submit an absolute host path, or supply a command. The redacted
`taskchamber://capabilities` MCP resource publishes canonical names, aliases,
parameter formats, and examples so an outer agent does not need to guess CLI
syntax. Invalid approximate names return suggestions but are never executed
silently.

See [`docs/POLICY_CONFIGURATION.md`](https://github.com/gausshj/taskChamber/blob/main/docs/POLICY_CONFIGURATION.md)
for the complete schema, fixed-command parameter example, fuzzy correction
behavior, and network boundary.

## Provider profiles and local secrets

Runtime and provider are separate concepts. `claude` is an agent runtime;
`custom_provider` below is an API provider profile that speaks the Anthropic
wire format required by that runtime. The server does not translate between
Anthropic, OpenAI Responses, Chat Completions, or other agent protocols.

Provider selection follows this precedence:

1. `TASKCHAMBER_DEFAULT_PROFILE` and profiles explicitly configured by the
   project;
2. the single project profile when exactly one is declared;
3. the built-in `glm` profile when `Z_AI_API_KEY` is injected directly;
4. otherwise, the sanitized `claude_code` fallback loaded from the active
   `~/.claude/settings.json`.

The fallback imports only `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, the top-level
model, and `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`. It does not load user
plugins, hooks, MCP servers, tools, permissions, skills, or transcripts. OAuth
or Keychain-only Claude authentication is not copied; the fallback requires an
API credential in the settings file.

To override the Claude Code default for this project, copy `.env.example` to
the ignored `.env` file and define a profile:

```dotenv
TASKCHAMBER_RUNTIME=claude
TASKCHAMBER_DEFAULT_PROFILE=custom_provider
TASKCHAMBER_PROFILES=custom_provider

TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__RUNTIME=claude
TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__API_FORMAT=anthropic
TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__BASE_URL=https://provider.example/api/anthropic
TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__MODEL=provider-model-name
TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__API_KEY_FIELD=auth_token
TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__API_KEY=replace-locally
```

Profile names use lowercase letters, digits, and underscores. Process
environment values override `.env`; an explicit file can be selected with
`TASKCHAMBER_ENV_FILE`. The loader uses `dotenv_values()` and does not mutate
the parent process environment. Provider URLs must use HTTPS, except localhost
development endpoints, and credentials in URLs are rejected.

`.env` is still plaintext. It is for local experiments only, is ignored by git,
and is excluded from every staged agent workspace. CI uses no provider secrets;
production deployments should inject an environment or secret-manager-backed
`SecretProvider` from a minimal launch environment.

## Claude runtime

The default runtime is `ClaudeAgentSdkRuntime`. Built-in compatibility profiles
remain available for `glm`, `deepseek`, and `anthropic`, while `.env` can add or
override profiles without changing Python code. With no project selection, its
default profile is the sanitized `claude_code` fallback described above.
Claude profiles must declare
`API_FORMAT=anthropic`; incompatible formats fail before the SDK starts.

For a real local experiment, set a workspace root and only the credential for
the provider you have authorized. Do not store credential values in an MCP
configuration committed to this repository.

```bash
TASKCHAMBER_WORKSPACE_ROOT=/absolute/path/to/safe-fixture \
Z_AI_API_KEY=... \
uv run python taskchamber_mcp.py
```

The Claude adapter explicitly restricts its built-in tool set, uses a fixed
working directory, validates paths before tool use, disables external child
MCP configuration, and uses a temporary Claude config directory per task. It
also launches the Claude CLI through a wrapper that replaces the inherited
environment with a small allowlist. The wrapper contains environment variable
names, never credential values, and the SDK's separate `-v` probe receives no
provider credential or proxy variables.

This does **not** sandbox the Python Agent SDK imported into the TaskChamber
server process. The current OS adapters cover only the Claude CLI and processes
it starts. TaskChamber, the SDK, document catalog callbacks, and configured
document-source commands remain in the trusted host process. Consequently this
boundary is designed for an untrusted prompt/model with a trusted TaskChamber,
pinned SDK, and selected CLI. If SDK supply-chain compromise is in scope, run
the service from a minimal environment and isolated account/container; a future
worker-process boundary is required to remove the SDK from the trusted parent.

## OS-level sandboxing

On top of the application-level boundary, the Claude runtime can run each task
inside an OS sandbox through a vendor-neutral `Sandbox` port:

| Adapter | Platform | What it enforces |
| --- | --- | --- |
| `BubblewrapSandbox` | Linux | Host filesystem read-only, environment-selected and account-database home directories hidden, staged workspace read-only, host PID/IPC/UTS namespaces hidden, capabilities dropped, and only the per-task CLI config writable; an executable below a hidden root is mounted back as one exact read-only file. |
| `MacOSSandboxExecSandbox` | macOS | Hides both host home paths except for the exact CLI and staged inputs, denies other-process inspection and writes outside per-task CLI config, but still permits reads outside those homes and network; best-effort development hardening because `sandbox-exec` is deprecated. |
| `NoSandbox` | POSIX (Linux/macOS) | Filtered temporary workspace and clean CLI environment, but no OS process enforcement — the portable fallback when no native sandbox is usable. |

The sandbox stages a **filtered copy** of the allowed project workspace (secrets
such as `.env` and `*.pem` are excluded at copy time) and points the SDK at it
via a `cli_path` launcher. The launcher first rebuilds a minimal environment and
then `exec`s `bwrap … claude` (Linux), `sandbox-exec … claude` (macOS), or the
CLI directly (`NoSandbox`). Launcher code and writable CLI config use disjoint
directories. Native adapters enforce the launcher as read-only; `NoSandbox`
provides separation and a clean environment but no same-user write boundary.
The SDK's `-v` probe cannot create the one-shot marker used to observe a main
CLI launch.

TaskChamber resolves the CLI before invoking the SDK. By default it requires the
executable bundled with the pinned `claude-agent-sdk` wheel; it never silently
falls back to an ambient `PATH` entry. An administrator may deliberately select
another reviewed executable with an absolute `TASKCHAMBER_CLAUDE_CLI_PATH`.
That override must be self-contained: the clean launcher fixes
`PATH=/usr/bin:/bin`, and native sandboxes expose only the exact executable
below a hidden home or `/tmp`, not adjacent npm/Homebrew modules or
interpreters. A missing, non-executable, or incompatible override fails instead
of falling back to ambient PATH; prefer the pinned bundle.

Virtual document sources are separate from this snapshot. Directory documents
are read in place by the MCP server through bounded catalog operations; CLI
stdout is held as bounded task-local data. Neither is materialized in the
agent's workspace tree.

`TASKCHAMBER_SANDBOX` selects OS process enforcement from the composition root.
All built-in modes stage a filtered workspace first:

- `auto` (default) — use the platform's native sandbox only after the real
  generated launcher and policy pass an operational probe; otherwise fall back
  to `NoSandbox`.
- `required` — require the current platform's native sandbox and fail when its
  operational probe fails. This requires adapter activation; it does not turn
  macOS's deprecated Seatbelt adapter into production-grade containment.
- `none` — disable OS isolation (application-level guards only).
- `bwrap` / `sandbox-exec` — require that specific adapter; a missing tool or
  failed operational probe is an error rather than a silent downgrade.

```bash
TASKCHAMBER_SANDBOX=auto uv run python taskchamber_mcp.py   # Linux: bwrap if present
TASKCHAMBER_SANDBOX=required uv run python taskchamber_mcp.py
TASKCHAMBER_SANDBOX=none  uv run python taskchamber_mcp.py   # app-level guards only
```

Path-valued certificate settings forwarded to the CLI
(`NODE_EXTRA_CA_CERTS`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, and
`SSL_CERT_DIR`) must be absolute and exist. A native sandbox rejects values
under a hidden home directory (and under `/tmp` for Bubblewrap) with
`cli_environment_invalid`; place deployment CA material in a reviewed
system-readable location instead of widening the home-directory mount.

Boundary statement: **neither sandbox isolates network, and neither contains
the Python SDK process.** A task must still reach its provider, and the read-only
tool preset (no `Bash`, `WebFetch`, or `WebSearch`) remains an application-level
network-exposure control for a trusted CLI. OS isolation is defense in depth on
top of the workspace guard, not protection against a compromised SDK. Execution
telemetry therefore reports `isolation_scope=agent_cli` and
`runtime_process_isolated=false` separately. `wrapper=true` means a launcher
was selected, while `cli_launch_observed=true` means its main-call marker was
created after entering the effective boundary. Native runs also require
`sandbox_preflight_passed=true` before `os_isolated=true` can be reported.

The sandbox adapter layer is unit-tested without a provider (argv/profile
generation, staging, runtime wiring). Confirming that a real `claude` process
actually starts under the sandbox requires a live call with a provider
credential — run it once before trusting the boundary.

## Adding another agent runtime

New runtimes implement the small `AgentRuntime` protocol in
`taskchamber/core/contracts.py`:

```python
class MyRuntime:
    name = "my-runtime"
    capabilities = AgentCapabilities(read_workspace=True, read_documents=True)

    async def run(
        self,
        request: TaskRequest,
        policy: ExecutionPolicy,
    ) -> TaskResult:
        ...
```

`read_documents=True` means the adapter can expose an agentic task's
`policy.document_catalog` through equivalent bounded read-only tools; omit it
if the runtime has not implemented that bridge. In `single_pass` mode,
`TaskService` embeds the one selected document in the prompt and supplies no
catalog or document tools to the adapter. The adapter owns vendor-specific
model calls, tool-loop behavior, and provider credentials. The MCP server,
workspace policy, tool schemas, and result format stay unchanged. `FakeRuntime`
demonstrates this boundary and powers the test suite without model calls or
credentials.

Built-ins are registered lazily so importing MCP core does not import a vendor
SDK. An independently installed adapter exposes a callable factory through the
Python entry-point group `taskchamber.runtimes`; the factory receives a
`RuntimeFactoryContext` containing the parsed configuration and selected
`Sandbox`. Vendor SDK packages therefore remain optional dependencies.

```toml
[project.entry-points."taskchamber.runtimes"]
my_runtime = "my_runtime_package:create_runtime"
```

```python
def create_runtime(context: RuntimeFactoryContext) -> AgentRuntime:
    return MyRuntime(configuration=context.configuration, sandbox=context.sandbox)
```

A bundled adapter uses the same internal shape as Claude:

```text
runtimes/codex/
├── __init__.py   # lightweight public API; do not eagerly import the SDK
├── factory.py    # RuntimeFactoryContext -> AgentRuntime
├── profiles.py   # Codex-compatible provider defaults, if any
└── runtime.py    # SDK-specific execution and result normalization
```

The prospective Codex adapter belongs in that package and must implement its
native protocol directly. It must not be routed through the Claude adapter or a
translation layer. See `docs/RUNTIME_ADAPTERS.md` for the complete extension
contract.

## Verification

The test suite covers workspace escape and symlink rejection, dotenv precedence
and secret redaction, provider/runtime protocol compatibility, runtime plugin
discovery, Claude SDK option hardening through an injected query function,
in-memory MCP contracts, and a real stdio client/server smoke test using
`FakeRuntime`. GitHub Actions runs the locked environment without credentials
or provider calls across CPython 3.11 through 3.14 on both Ubuntu and macOS.
Repository quality and release-artifact checks run once on the Python 3.11
baseline instead of being duplicated in every matrix job.

For a user-visible real stdio call, follow `docs/USER_TESTING.md` or run
`uv run python scripts/manual_stdio_test.py`. This is an actual provider request
and may consume quota.

To expose the server to Claude Code or another outer agent as MCP tools, see
`docs/MCP_CLIENT_SETUP.md`.

For the disposable, adversarial real-world acceptance test covering Claude Code
installation, outer-agent invocation, workspace staging, OS sandbox activation,
tool allow/deny policy, secret canaries, and write attempts, follow
`docs/REAL_E2E_TEST.md`.

Release maintainers should follow
[`docs/RELEASING.md`](https://github.com/gausshj/taskChamber/blob/main/docs/RELEASING.md).

## License

TaskChamber is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

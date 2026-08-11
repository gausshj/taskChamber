# Runtime adapter extension contract

An agent runtime is an execution engine, while a provider profile is an API
endpoint, model, wire format, and credential reference usable by that engine.
Keep those concepts separate: selecting a provider must never silently switch
the runtime or trigger protocol translation.

## Stable core boundary

Every adapter implements `AgentRuntime` from
`taskchamber.core.contracts`. Its only operation is a fresh, stateless
`run(TaskRequest, ExecutionPolicy) -> TaskResult` call. The adapter must declare
capabilities, declare its `default_profile`, and return normalized failures
without leaking SDK exceptions, environment values, or transcripts.

The adapter receives host-owned limits in `ExecutionPolicy`. Unsupported
capabilities are an explicit failure; they are not permission to weaken the
policy.

`RuntimeFactoryContext.sandbox` is a host-supplied capability, not a core-level
process supervisor. The built-in Claude adapter uses it, but an installed
runtime factory is trusted Python code and can ignore it. Do not claim that an
arbitrary `AgentRuntime` is isolated merely because a `Sandbox` object was
passed to its factory. A future boundary against compromised runtime code must
launch the entire adapter in a separately supervised worker/container.

`read_workspace` and `read_documents` are independent capabilities. For an
agentic document task, `ExecutionPolicy.document_catalog` contains the selected
provider-neutral catalog. An adapter declaring `read_documents=True` must map
that catalog to equivalent bounded list/read/search tools without exposing raw
host paths, commands, or an unrestricted shell. Claude does this with a
task-scoped in-process SDK MCP server; another runtime may use its native custom
tool mechanism. In `single_pass` mode, `TaskService` reads the one selected
document, embeds it in the isolated prompt, and passes `document_catalog=None`;
the adapter receives no document tools and performs one model turn.

## Bundled adapter layout

A bundled adapter uses one isolated package:

```text
taskchamber/runtimes/<name>/
├── __init__.py
├── factory.py
├── profiles.py
└── runtime.py
```

- `runtime.py` is the only place that imports and interprets the vendor SDK.
- `profiles.py` declares only endpoints compatible with that SDK's native
  protocol.
- `factory.py` accepts `RuntimeFactoryContext`, resolves adapter-owned profiles
  and secrets, and constructs the runtime.
- `__init__.py` stays lightweight so registry discovery does not eagerly import
  an optional vendor dependency.

Add its lazy target to `BUILTIN_RUNTIME_TARGETS` in
`taskchamber.runtimes.builtins`. Do not add SDK-specific branches to the MCP
transport, `TaskService`, configuration loader, or composition root.

For a future bundled Codex adapter, use `runtimes/codex/` with this same shape.
Its runtime must call the Codex SDK or CLI through the protocol that the chosen
provider actually supports. It must not reuse Claude message parsing or convert
Responses-style events into Anthropic events.

## External adapter package

An independently distributed adapter does not require a core change. Export a
callable factory through the entry-point group:

```toml
[project.entry-points."taskchamber.runtimes"]
codex = "taskchamber_codex.factory:create_runtime"
```

```python
from taskchamber.core.contracts import AgentRuntime
from taskchamber.runtimes.registry import RuntimeFactoryContext


def create_runtime(context: RuntimeFactoryContext) -> AgentRuntime:
    return CodexRuntime(
        configuration=context.configuration,
        sandbox=context.sandbox,
    )
```

Installed entry points are trusted executable Python code. Duplicate runtime
names are rejected rather than resolved by order.

## Sandbox adapter migration

The hardened Claude path calls `Sandbox.prepare_cli_launcher(...)`. A legacy
third-party sandbox that only overrides the original
`prepare_wrapper(workspace, *, executable, config_dir)` signature remains
callable: TaskChamber places its result behind a clean-environment outer
launcher, but conservatively does not claim native OS isolation. To implement
the current contract, generate the marker inside the native boundary, accept
the separate `launcher_dir` and environment-name allowlist, and opt in with
`secure_cli_launcher = True`, `operational_preflight = True`, and
`launch_observation_inside_os_sandbox = True` only after equivalent tests.
Direct callers of the original three-argument built-in `prepare_wrapper`
contract retain its inherited-environment behavior. They do not receive the
new clean-environment guarantee; migrate runtime integrations to
`prepare_cli_launcher` explicitly.

`ClaudeAgentSdkRuntime(cli_resolver=...)` remains as a deprecated compatibility
shim for explicit callers. It never runs silently by default and a missing or
non-absolute resolver result fails closed. New code should configure an
absolute `TASKCHAMBER_CLAUDE_CLI_PATH` or use the pinned SDK bundle.

When the canonical CLI resolves below a root the sandbox masks (a host home or
`/tmp`), the Bubblewrap adapter recreates empty parent directories and
read-only binds only the exact executable. The masked root itself must be a
real directory owned by the effective user or root, and a group/world-writable
root is accepted only with the sticky bit — the shape that makes a root-owned
`/tmp` safe while a misconfigured non-sticky `0777` home is rejected. Every
path component below the root must likewise be owned by the effective user or
root, must not be group/world-writable, and must not be a symlink after
canonicalization; otherwise launcher preparation fails closed with
`sandbox_setup_failed`. This guards the rebind against replacement by other
local users between canonicalization and launch; a same-uid process is already
inside the trust domain and is not the boundary being enforced.

## Isolation telemetry

Runtime telemetry must distinguish configuration from execution.
`cli_wrapper_active=true` means a generated launcher was passed as `cli_path`;
it is not execution evidence. `cli_launch_observed=true` requires the one-shot
main-call marker written after the clean-environment launcher enters its
effective boundary. For a native adapter, `sandbox_preflight_passed=true` means
the same generated wrapper and policy completed an operational probe.
`os_isolated=true` and `isolation_scope=agent_cli` require both observations and
mean only the Claude CLI and its descendants entered Bubblewrap or Seatbelt.
The Python SDK remains in TaskChamber, so `runtime_process_isolated=false`.
These fields are runtime evidence, not kernel attestation, and say nothing about
environment access by the imported SDK.

## Adapter verification

Each SDK adapter needs focused tests for:

1. lazy core import without its vendor dependency;
2. compatible and incompatible provider profiles;
3. missing credentials and sanitized SDK failures;
4. effective tools, working directory, timeout, cancellation, and cleanup;
   document-capable adapters must also prove catalog tool mapping and rejection
   of unselected sources;
5. normalized success, cost, partial output, and error results;
6. real stdio MCP interoperability through `TaskService`;
7. one live SDK-plus-CLI routing check before claiming third-party provider
   compatibility.

Use an injected SDK/query function for deterministic unit tests and
`FakeRuntime` for credential-free MCP/CI coverage.

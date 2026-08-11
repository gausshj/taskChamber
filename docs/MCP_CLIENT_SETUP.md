# Use taskchamber from an outer agent

The outer agent is the MCP client. It discovers three stable tools — `research`,
`summarize`, and `review` — and calls them like any other MCP tool. Each call
starts one fresh inner `AgentRuntime` task and returns a normalized `TaskResult`.

```text
Claude Code or another outer agent
    -> MCP research / summarize / review
    -> stdio FastMCP server
    -> TaskService policy and workspace boundary
    -> selected runtime (Claude now; Codex later)
    -> provider
    -> result + token usage back to the outer agent
```

The inner Claude runtime always uses `strict_mcp_config=True`. It sets
`mcp_servers={}` when no agentic document catalog is needed, including for
`single_pass` document calls. An agentic document call mounts exactly one
task-scoped, in-process, read-only document MCP server for bounded
list/read/search operations. It never mounts TaskChamber itself or inherits the
outer agent's MCP servers, so it cannot recursively call this server. Calls are
stateless and do not reuse the outer agent's conversation.

For a concrete project, establish the maximum boundary before registration:

```bash
taskchamber config init
taskchamber policy validate
taskchamber policy show
```

The outer agent can read `taskchamber://capabilities` for redacted canonical
capabilities, document aliases, parameter schemas, and examples. It may select
relative files/globs and a capability subset for each call, but cannot broaden
the `taskchamber.toml` policy.

## Temporary Claude Code experiment

From the repository root, create the configuration as an in-memory JSON string:

```bash
export REPO=/absolute/path/to/taskChamber
cd "$REPO"

MCP_CONFIG="$(jq -nc \
  --arg python "$PWD/.venv/bin/python" \
  --arg workspace "$PWD" \
  '{mcpServers:{"taskchamber":{
    type:"stdio",
    command:$python,
    args:["-m","taskchamber"],
    env:{
      TASKCHAMBER_RUNTIME:"claude",
      TASKCHAMBER_SANDBOX:"auto",
      TASKCHAMBER_WORKSPACE_ROOT:$workspace
    }
  }}}')"

claude --mcp-config "$MCP_CONFIG" --strict-mcp-config
```

This does not write a user or project MCP registration. `--strict-mcp-config`
keeps the experiment limited to this server. The subprocess inherits `HOME`, so
the trusted server process can read the sanitized Claude Code settings fallback;
no API key appears in the MCP JSON. The inner CLI does not inherit that HOME or
the complete server environment: TaskChamber launches the SDK-bundled CLI
through a clean-environment wrapper.

If `TASKCHAMBER_CLAUDE_CLI_PATH` is set, it must point to a reviewed,
self-contained executable compatible with `PATH=/usr/bin:/bin`. Native
sandboxes do not expose adjacent modules below a hidden home or `/tmp`; use the
pinned SDK bundle unless an override has been verified under the selected
sandbox.

`auto` is suitable for this portable experiment. On a deployment where native
process isolation is mandatory, set `TASKCHAMBER_SANDBOX` to `required` (or the
specific `bwrap` / `sandbox-exec` adapter). Those modes fail when the generated
native launcher cannot pass its operational preflight. They isolate the inner
CLI and its descendants, not the Python Agent SDK in the server process.
On macOS, `required` requires the deprecated Seatbelt adapter to activate; it
does not make that best-effort development boundary suitable for production.
Use a container or VM when stronger containment is required.
If the deployment forwards a custom CA path, keep it absolute and outside the
hidden account home (`/tmp` is hidden by Bubblewrap as well); incompatible path
settings fail explicitly with `cli_environment_invalid`.

Inside Claude Code, request the tool explicitly for the first check:

```text
请调用 taskchamber 的 research 工具，workspace_paths 只传入 README.md，
requested_capabilities 只使用 workspace.read；
返回答案后同时告诉我 structured result 中的 provider、model、usage 和 model_usage。
```

The outer agent receives the textual result, a `[tokens ...]` line when usage is
reported, and the full structured object. It can use token counts when deciding
whether another isolated task is worthwhile. Current token reporting is
post-call telemetry, not a hard pre-call token budget. `usage` is the aggregate
counter set; `model_usage` breaks the same provider-reported counters down per
model. TaskChamber does not emit a per-call provider invocation trace.

## Response envelope text mode

Every tool response carries two representations of the same `TaskResult`: the
legacy text block in `content` and the canonical object in `structuredContent`.
The server-side `TASKCHAMBER_MCP_TEXT_MODE` setting controls how much of a
successful result the text block repeats:

- `full` (default, backward compatible): the text block ends with the complete
  generated output. Clients that read only the text content keep working
  unchanged, and clients that merge text with `structuredContent` see the body
  twice.
- `metadata_only`: successful responses keep only the provider/status/result/
  tokens/execution metadata lines in text; the generated body is served exactly
  once, through `structuredContent.output`, which remains the canonical
  complete `TaskResult`. Enable this only for clients that consume
  `structuredContent` (including clients that merge both representations).
  Clients that read only the text content must not enable this mode, because
  the success body no longer appears in text.

Error responses are identical in both modes: the text block always keeps the
full legacy rendering, including `error_message` and any `[incomplete partial
output]` section, because some clients ignore `structuredContent` on error
paths.

`max_output_chars` bounds `TaskResult.output` (and therefore
`structuredContent.output`). In `full` mode the wire envelope additionally
contains the legacy text copy, so the total response size is not limited to
`max_output_chars`; in `metadata_only` mode the client-visible success envelope
tracks the configured output bound.

The validated effective limit (host ceiling, optionally reduced per call) is
also given to the runtime before generation as a server-owned system-prompt
instruction asking for a complete response within that budget. That guidance
encourages compact, cleanly finished answers, but model compliance is not
guaranteed: the post-hoc character cap remains the authoritative boundary, and
over-limit results still return `partial=true` and `truncated=true`.

## Persistent project registration

Only after the temporary test succeeds, generate a server object and ask Claude
Code to store it at project scope:

```bash
SERVER_JSON="$(jq -nc \
  --arg python "$PWD/.venv/bin/python" \
  --arg workspace "$PWD" \
  '{type:"stdio",command:$python,args:["-m","taskchamber"],env:{
    TASKCHAMBER_RUNTIME:"claude",
    TASKCHAMBER_SANDBOX:"auto",
    TASKCHAMBER_WORKSPACE_ROOT:$workspace
  }}')"

claude mcp add-json --scope project taskchamber "$SERVER_JSON"
claude mcp get taskchamber
```

This registration contains absolute local paths but no credential. Review the
resulting project configuration before committing it; a machine-specific path
usually should remain local. Remove it with:

```bash
claude mcp remove --scope project taskchamber
```

## Project provider override

The MCP configuration selects the runtime and workspace, not the credential.
Provider precedence remains:

1. project `.env` profile;
2. directly injected supported profile credential;
3. sanitized Claude Code settings defaults.

Never place an API key in MCP JSON. For a project-specific provider, use the
ignored `.env` documented by `.env.example` or a production `SecretProvider`.

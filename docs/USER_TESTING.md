# User acceptance test

This test launches the real stdio MCP server, performs `initialize`, lists its
tools, and calls `research` once through the selected agent runtime.

The default command makes a real provider request and may consume quota. It
does not modify Claude Code, `~/.claude`, or project files.

## 1. Preconditions

Choose the repository checkout and confirm whether a project provider override
is active:

```bash
export REPO=/absolute/path/to/taskChamber
cd "$REPO"
test ! -f .env && echo "using Claude Code default" || echo "project .env overrides default"
```

When testing the Claude Code settings fallback, inspect only non-secret provider
fields. The inner runtime uses the CLI bundled with the pinned SDK wheel rather
than an ambient `claude` from `PATH`:

```bash
uv run python -c 'import subprocess; from taskchamber.runtimes.claude.cli import bundled_claude_cli_path as p; subprocess.run([str(p()), "-v"], check=True)'
jq '{base_url: .env.ANTHROPIC_BASE_URL, model: (.model // .env.ANTHROPIC_MODEL), credential_present: ((.env.ANTHROPIC_AUTH_TOKEN // .env.ANTHROPIC_API_KEY // "") | length > 0)}' ~/.claude/settings.json
```

Expected: the configured HTTPS endpoint and model, plus
`credential_present: true`. The command never prints the credential value.

## 2. Prepare the locked environment

```bash
uv sync --locked --all-groups
uv run pytest -q
```

## 3. Make one real MCP call

```bash
uv run python scripts/manual_stdio_test.py \
  --workspace "$REPO" \
  --max-turns 3 \
  --question "只读取 README.md，用三点概括这个项目的架构。"
```

The client deliberately launches the server with a minimal environment. The
server reads the credential from Claude Code settings into a private
`SecretProvider`; it does not copy the credential into the client command.

Expected output includes:

```text
MCP tools: research, review, summarize

Structured result:
{
  "status": "success",
  "runtime": "claude-agent-sdk",
  "provider": "claude_code",
  "model": "provider-model-name",
  "usage": {
    "input_tokens": 123,
    "output_tokens": 45,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "reasoning_output_tokens": null,
    "total_tokens": null
  }
}
```

Only counters actually reported by the provider are populated; the example
numbers are illustrative. `cost_usd` may also be present but is only a provider
reference because subscription and pricing arrangements differ. A non-zero
command exit means the MCP result was an error; keep the structured `status` and
`error_code` for diagnosis, but do not paste settings or credential values.
The structured `execution` object should report `cli_wrapper_active: true`,
`cli_launch_observed: true`, and `cli_environment_sanitized: true`. Native OS
sandbox runs additionally report `sandbox_preflight_passed: true`,
`os_isolated: true`, and `isolation_scope: "agent_cli"`.
`runtime_process_isolated: false` remains deliberate: the Python SDK stays in
the trusted server process.

## Project override test

When `.env` explicitly declares a provider and
`TASKCHAMBER_DEFAULT_PROFILE`, the same command uses that project profile
instead of Claude Code defaults. `.env.example` documents the fields. Remove or
rename the local `.env` to return to the Claude Code settings fallback; never
commit it.

For a credential-free protocol-only check:

```bash
uv run python scripts/manual_stdio_test.py --runtime fake
```

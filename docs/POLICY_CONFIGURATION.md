# Project policy and per-call delegation

TaskChamber ships a broad read-only capability vocabulary. A concrete project
creates `taskchamber.toml` to define the maximum boundary for that installation,
then an MCP caller may select a smaller set of files, virtual documents, and
capabilities for one fresh task.

The effective policy is always the intersection of:

```text
hard-coded safety invariants
    ∩ project policy.allowed_capabilities and workspace filters
    ∩ task allowed_capabilities
    ∩ requested_capabilities and resource selections from this MCP call
```

The caller cannot raise `max_turns` or `max_output_chars`, add a host path,
enable an unconfigured document command, or request a capability outside the
project/task ceiling.
The existing `scope` argument remains prompt text only; it is never used as an
authorization boundary.

## Create and inspect a project policy

From the project root:

```bash
taskchamber config init
taskchamber policy validate
taskchamber policy show
```

`taskchamber policy show` prints the effective non-secret policy. It does not
print command argv, host document roots, environment values, or credentials.

The TOML file is the canonical configuration. The management CLI edits the same
file atomically:

```bash
taskchamber policy deny review documents.search
taskchamber policy allow review documents.search
taskchamber policy set-default review workspace.read workspace.search
```

An `allow` operation may only add a capability already present in
`policy.allowed_capabilities`. Policy changes are loaded when the stdio server
starts; restart the MCP server after editing the file.

Select another file explicitly with `TASKCHAMBER_CONFIG_FILE` or the management
commands' `--config` option. `taskchamber` with no arguments, `taskchamber serve`,
and `python -m taskchamber` continue to run the stdio server without printing a
banner to stdout.

## Capability vocabulary

The public MCP contract uses provider-neutral names. Runtime adapters map these
names to their concrete SDK tools:

| Capability | Claude adapter tool | Meaning |
| --- | --- | --- |
| `workspace.list` | `Glob` | list staged workspace matches |
| `workspace.read` | `Read` | read staged workspace files |
| `workspace.search` | `Grep` | search staged workspace files |
| `documents.list` | `mcp__documents__list_documents` | list virtual documents |
| `documents.read` | `mcp__documents__read_document` | read bounded virtual pages |
| `documents.search` | `mcp__documents__search_documents` | search virtual documents |

`Bash`, write/edit tools, web tools, and recursive Task/MCP delegation are not
members of this vocabulary. Adding them is a separate security and API change,
not a TOML toggle.

Each task has independent `allowed_capabilities` and `default_capabilities`.
Defaults apply when the caller omits `requested_capabilities`; an explicit list
selects a subset of the task allowance.

## Workspace selection

The project policy defines relative include/exclude patterns:

```toml
[policy.workspace]
root = "."
include = ["src/**/*.py", "tests/**/*.py", "pyproject.toml"]
exclude = ["tests/private/**"]
allow_globs = true
max_requested_paths = 64
```

The caller can narrow a task further:

```json
{
  "workspace_paths": [
    "src/runtime/**/*.py",
    "tests/test_runtime*.py"
  ],
  "requested_capabilities": [
    "workspace.read",
    "workspace.search"
  ]
}
```

Caller selections must be relative paths or globs. Absolute paths, `~`, `..`,
symlink escapes, protected files, excluded patterns, oversized files, and
selections beyond the configured count are rejected. Only matched files are
copied into the filtered task workspace. Glob/Grep may address the staged root,
but unselected source files do not exist there.

The canonical `taskchamber.toml` filename is also a protected workspace file so
an inner agent cannot read configured command templates. If an explicit config
uses a different filename, place it outside the readable workspace or exclude it
in `policy.workspace.exclude`.

`review` accepts the legacy `file_path` and the additive `workspace_paths` list.
`research` uses `workspace_paths` when `include_workspace=true`. `summarize` and
`review` can also use selected virtual document sources as supporting input.

## Discovery and correction hints

The server publishes a redacted MCP resource:

```text
taskchamber://capabilities
```

It contains task capabilities, source names, descriptions, aliases, parameter
formats, and examples. It never contains executable paths, argv, host roots, or
secret references. The same canonical capability/source names are summarized in
the server and tool descriptions for clients that do not read MCP resources.

Safe normalization and configured aliases may resolve automatically, for
example `Workspace-Read` to `workspace.read` or `record lookup` to
`record_detail`. An approximate spelling is never executed silently.
Invalid capabilities, files, sources, and parameter names return bounded
candidate suggestions plus explicit guidance to retry with a listed value and
not fall back to shell execution. Ambiguous aliases are rejected at startup.

## Fixed command document capability

A project can expose one exact CLI operation without exposing the rest of the
CLI:

```toml
[document_sources.record_detail]
kind = "command"
description = "Retrieve one record from an internal API"
aliases = ["record detail", "record lookup"]
executable = "/opt/records/bin/records-cli"
args = ["records", "get", "{record_id}"]
env_refs = ["RECORDS_API_TOKEN"]
output_format = "json_document"
document_id = "record.json"
timeout_seconds = 20

[document_sources.record_detail.parameters.record_id]
description = "Record identifier"
aliases = ["record", "id"]
pattern = "^rec_[A-Za-z0-9._-]{1,120}$"
example = "rec_20260715_a"
```

The MCP caller submits semantic data, not a command:

```json
{
  "document_requests": [
    {
      "source": "record_detail",
      "parameters": {
        "record_id": "rec_20260715_a"
      }
    }
  ]
}
```

TaskChamber executes only the fixed argv template with direct process execution.
Every declared parameter must occupy one whole argv item, match its configured
regular expression and length, and not begin with `-`. Unknown, missing, or
duplicate parameters fail before the process starts. The caller cannot replace
the executable, subcommands, cwd, environment allowlist, or output bounds.

Command sources still run with the MCP server account's host and network access;
the inner agent sandbox does not contain this connector process. Treat the
executable as trusted deployment code, use source-scoped credentials, and run
production connectors under an appropriate OS/container/egress policy. A
hostname network allowlist is intentionally not accepted by this TOML schema
until TaskChamber has an adapter that can enforce it.

Legacy `.env` directory and fixed `{query}` command sources remain supported for
compatibility. TOML is preferred for project policy, aliases, and typed command
parameters; real secret values remain in the ignored `.env` or another
`SecretProvider`.

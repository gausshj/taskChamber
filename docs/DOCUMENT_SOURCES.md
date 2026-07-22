# Virtual document sources

`research` can combine the staged project workspace with documents that live in
other directories or are returned by a CLI. External documents are exposed as a
read-only virtual catalog; they are **not copied into the staged workspace**.
`summarize` and `review` may select the same sources as supporting or primary
input. Project TOML is preferred; the environment format remains compatible.

The call path is:

```text
named server configuration
    -> DirectoryDocumentSource | CommandDocumentSource
    -> task-scoped DocumentCatalog
    -> runtime adapter's read-only document tools
    -> research result
```

For the Claude adapter, the catalog becomes one in-process SDK MCP server with
three tools:

- `list_documents`: list source/document IDs and bounded metadata;
- `read_document`: read a bounded line range by source and document ID;
- `search_documents`: return bounded line matches across selected sources.

These are inner, task-scoped agent tools. The public stdio MCP surface remains
`research`, `summarize`, and `review`.

## Caller contract

`research` has two additive arguments:

```json
{
  "question": "Find the retention rule and recent failures",
  "document_sources": ["product_docs", "service_logs", "remote_docs"],
  "include_workspace": false
}
```

The caller may select configured source names only. It cannot supply a raw host
path, command, URL, or environment variable. `include_workspace=false` gives
the agent an empty staged workspace and only the selected virtual documents.
The default remains `true` for compatibility with existing calls.

Parameterized sources use the additive `document_requests` argument:

```json
{
  "question": "Analyze this record",
  "document_requests": [
    {
      "source": "record_detail",
      "parameters": {"record_id": "rec_20260715_a"}
    }
  ],
  "include_workspace": false,
  "requested_capabilities": ["documents.list", "documents.read"]
}
```

For one already-identified document, use bounded single-pass execution:

```json
{
  "question": "Summarize the record detail",
  "document_sources": ["record_detail"],
  "include_workspace": false,
  "document_mode": "single_pass",
  "max_output_chars": 2000,
  "requested_capabilities": ["documents.read"]
}
```

The server requires exactly one selected virtual document, reads it before the
runtime starts, embeds it in the initial prompt, exposes no workspace or
document tools, and forces one model turn. The effective caller capabilities
must include `documents.read`; listing or searching authority alone cannot
expose the document body. The default server limit is 64,000 UTF-8 bytes.
Multiple or oversized documents fail explicitly and never fall back to agentic
execution. `max_output_chars` may only reduce the server's configured output
cap. Larger catalogs should retain the default `agentic` mode so the inner
runtime can list, search, and page documents.

## Multiple directory sources

Configure each root independently, then select any combination in one research
call:

```dotenv
TASKCHAMBER_DOCUMENT_SOURCES=product_docs,service_logs

TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__KIND=directory
TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__ROOT=/absolute/path/to/product/docs
TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__INCLUDE=**/*.md,**/*.txt
TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__EXCLUDE=**/drafts/**

TASKCHAMBER_DOCUMENT_SOURCE__SERVICE_LOGS__KIND=directory
TASKCHAMBER_DOCUMENT_SOURCE__SERVICE_LOGS__ROOT=/absolute/path/to/service/logs
TASKCHAMBER_DOCUMENT_SOURCE__SERVICE_LOGS__INCLUDE=**/*.log
```

Directory files are indexed by virtual workspace-independent IDs such as
`guides/setup.md` or `2026-07/service.log`. Reads happen against the configured
root in the MCP server process. Symlinks, `.env*`, credential directories, PEM
and key files, binary files, oversized files, and configured exclusions are not
published. Absolute host paths are not returned to the agent.

Optional per-source bounds are:

```dotenv
TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__MAX_FILE_BYTES=1000000
TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__MAX_TOTAL_BYTES=50000000
TASKCHAMBER_DOCUMENT_SOURCE__PRODUCT_DOCS__MAX_FILES=10000
```

## CLI or API-backed source

A command source runs one server-owned argv for each fresh `research` call. It
does not enable `Bash` for the agent and does not invoke a shell. `{query}` is
substituted as one argv value, so shell metacharacters in the question are not
executed.

For a typed fixed operation, configure TOML so the caller never has to provide
or reconstruct the command:

```toml
[document_sources.record_detail]
kind = "command"
description = "Retrieve one record from an internal API"
aliases = ["record detail", "record lookup"]
executable = "/absolute/path/to/records-cli"
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

The executable and `records get` subcommands are immutable server policy.
Only the standalone parameter item is substituted. It must match the configured
full regular expression and size, and may not begin with `-`, so this source
cannot call other `records-cli` operations or inject additional options.

The older `.env` form supports a fixed `{query}` argument:

```dotenv
TASKCHAMBER_DOCUMENT_SOURCES=remote_docs
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__KIND=command
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__ARGV=["/absolute/path/to/doc-cli","fetch","--format","json","--query","{query}"]
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__CWD=/absolute/path/to/safe/working-directory
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__OUTPUT_FORMAT=json
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__ENV_ALLOW=DOC_API_TOKEN
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__TIMEOUT_SECONDS=30
```

`ARGV` must be a JSON string array. The executable and every argument are fixed
by local server configuration; only a standalone `{query}` argv item is
supported. Known command-shell executables are rejected. Do not route the
placeholder into an interpreter expression through a wrapper CLI.
The child gets a minimal environment (`PATH`, locale, and temp directory when
present) plus the names in `ENV_ALLOW`. Other parent-process secrets are not
forwarded. Put an API credential in the ignored `.env` or inject it into the
server process, then explicitly list only its variable name in `ENV_ALLOW`.

The command may call any authorized document API internally. This keeps API
transport, authentication, pagination, and vendor response formats inside the
dedicated CLI while the MCP/runtime contract remains provider-neutral.

### JSON output

The preferred stdout format is:

```json
{
  "documents": [
    {
      "id": "api/guide.md",
      "title": "Guide",
      "media_type": "text/markdown",
      "content": "Document body"
    }
  ]
}
```

`id` (or `path`) and `content` are required strings. `title` and `media_type`
are optional. A top-level array, one document object, or an object mapping IDs
to string contents is also accepted. Stdout must contain only the payload;
diagnostics belong on stderr. Non-zero exit, timeout, malformed UTF-8/JSON,
unsafe or duplicate IDs, and size-limit failures become a redacted MCP error.

For a CLI that emits one plain-text document instead:

```dotenv
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__OUTPUT_FORMAT=text
TASKCHAMBER_DOCUMENT_SOURCE__REMOTE_DOCS__DOCUMENT_ID=cli/output.txt
```

Additional command bounds are `MAX_OUTPUT_BYTES`, `MAX_DOCUMENT_BYTES`, and
`MAX_DOCUMENTS`.

If stdout is arbitrary business JSON rather than a document collection, set
`output_format = "json_document"` (or the equivalent environment value). The
complete JSON value—object, array, string, number, boolean, or null—is exposed
as one `application/json` virtual document using pretty-printed, Unicode-safe
text. The existing `json` collection mode is unchanged.

## Credential-free smoke configuration

The repository includes a deterministic CLI fixture. Use absolute paths in the
real `.env`:

```dotenv
TASKCHAMBER_RUNTIME=fake
TASKCHAMBER_DOCUMENT_SOURCES=fixture_cli
TASKCHAMBER_DOCUMENT_SOURCE__FIXTURE_CLI__KIND=command
TASKCHAMBER_DOCUMENT_SOURCE__FIXTURE_CLI__ARGV=["/absolute/path/to/taskchamber/.venv/bin/python","/absolute/path/to/taskchamber/scripts/document_fixture_cli.py","--query","{query}"]
TASKCHAMBER_DOCUMENT_SOURCE__FIXTURE_CLI__OUTPUT_FORMAT=json
```

Then call `research` with `document_sources=["fixture_cli"]` and
`include_workspace=false`. The fake runtime verifies configuration and MCP
plumbing without a model call; the Claude runtime is required to observe an
agent actually choosing `list_documents`, `read_document`, or
`search_documents`.

The bundled manual stdio client accepts those arguments directly:

```bash
uv run python scripts/manual_stdio_test.py \
  --runtime fake \
  --env-file .env \
  --document-source fixture_cli \
  --no-workspace \
  --question 'Return the virtual-document-cli-ok canary'
```

## Security boundary

- Source definitions are host policy, not MCP inputs.
- External directory contents are never added to the staged filesystem.
- The CLI runs outside the agent process with direct argv execution and a
  minimal environment.
- The agent still receives document contents and may send them to the selected
  model provider. Configure only sources whose contents are authorized for that
  provider.
- A command source can access the network or other resources available to the
  MCP server account. Treat its executable and configuration as trusted server
  code.
- The runtime agent sandbox does not wrap the command source process. A declared
  network hostname restriction would therefore be descriptive rather than
  enforceable, so the current schema does not accept one. Use an OS/container or
  egress policy for production connectors.

The `DocumentSource` and `PreparedDocumentSource` protocols are runtime-neutral.
A future native HTTP/API source can implement the same interface without
changing the public MCP tool or runtime adapters.

# Security Policy

TaskChamber is an MCP server that launches isolated agent tasks. Reports about
sandbox escapes, credential or secret leakage, policy bypasses, and unsafe
command execution are treated with the highest priority.

## Supported versions

Security fixes are applied to the latest published release and to `main`.
Older release lines do not receive backports; upgrade to the current release
before reporting problems that may already be fixed.

| Version | Supported |
| ------- | --------- |
| Latest release | Yes |
| Older releases | No |

## Reporting a vulnerability

Use GitHub Private Vulnerability Reporting:

<https://github.com/gausshj/taskChamber/security/advisories/new>

This creates a private advisory visible only to the maintainers. Do not open a
public issue, pull request, or discussion for a suspected vulnerability.

### What to include

A useful report contains:

- the TaskChamber version or commit SHA, Python version, and operating system
  (generic names only, for example `Ubuntu 24.04 x86_64`);
- the affected boundary, such as the MCP stdio transport, capability policy,
  workspace staging, OS sandbox, document sources, or a runtime adapter;
- the smallest configuration and steps needed to reproduce the behavior,
  expressed with neutral placeholder values;
- the expected and actual security-relevant behavior, and the impact you
  believe is possible.

### What to leave out

Never include real secrets or private data in a report. Replace them with
placeholders:

- API keys, tokens, `.env` contents, and credential-bearing provider URLs;
- personal absolute paths, internal hostnames, and private deployment details;
- source documents, prompts, customer data, and full agent transcripts.

Sanitized metadata such as `status`, `error_code`, `partial`, `truncated`, and
sandbox activation booleans is usually enough to explain an issue.

## Coordination expectations

- Reports are acknowledged as soon as a maintainer has reviewed them.
- Assessment and fixing happen privately inside the security advisory. A fix
  is published through a normal release before any public disclosure.
- Credit is given to reporters in the published advisory unless they prefer to
  remain anonymous.
- Please give the maintainers a reasonable window to investigate and release a
  fix before disclosing the issue anywhere else.

## Scope notes

The following are documented design properties, not vulnerabilities on their
own: OS-level sandboxing is defense in depth and does not isolate network
access; the Claude Agent SDK merges its child environment with the parent
process, so production hosts are expected to provide a minimal launch
environment or an isolated worker. Reports are still welcome when a concrete
behavior contradicts these documented guarantees.

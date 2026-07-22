"""Administrative CLI plus the backward-compatible stdio server entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import tomlkit

from .config import DEFAULT_CONFIG_FILE, load_configuration, load_project_policy
from .core.capabilities import KNOWN_CAPABILITIES, normalize_capability
from .core.contracts import TaskKind

HELP_FORMATTER = argparse.RawDescriptionHelpFormatter

ROOT_DESCRIPTION = """\
Run bounded, isolated agent tasks as a standard MCP server.

Run without a command to start the stdio server. Management commands configure
or inspect project policy without starting the server.
"""

ROOT_EPILOG = """\
examples:
  taskchamber
  taskchamber serve
  taskchamber config init
  taskchamber policy validate
  taskchamber policy show
"""

SERVE_DESCRIPTION = """\
Run TaskChamber as a stdio MCP server.

This command is equivalent to running taskchamber with no arguments. It exposes
the research, summarize, and review tools to an MCP client; it does not open an
interactive shell. Standard output is reserved for MCP protocol traffic.
"""

SERVE_EPILOG = """\
configuration:
  taskchamber.toml  Project policy, task defaults, and named document sources.
  .env              Local runtime and provider settings; keep credentials untracked.
  environment       Overrides .env settings for the server process.

Use TASKCHAMBER_CONFIG_FILE or TASKCHAMBER_ENV_FILE to select another file.

before serving:
  taskchamber config init
  taskchamber policy validate

example:
  TASKCHAMBER_RUNTIME=fake taskchamber serve
"""

CONFIG_DESCRIPTION = """\
Create TaskChamber project configuration.

The generated taskchamber.toml defines the maximum project policy and contains
no credential values. Runtime and provider settings belong in .env or the
server process environment.
"""

CONFIG_EPILOG = """\
example:
  taskchamber config init
"""

POLICY_DESCRIPTION = """\
Inspect or edit the TaskChamber project policy.

Policy commands operate on taskchamber.toml and never start the MCP server.
Edits take effect the next time the server starts.
"""

POLICY_EPILOG = """\
examples:
  taskchamber policy validate
  taskchamber policy show
  taskchamber policy deny review documents.search
  taskchamber policy set-default review workspace.read workspace.search
"""

DEFAULT_CONFIG_TEMPLATE = """\
schema_version = 1

[policy]
allowed_capabilities = [
  "workspace.list",
  "workspace.read",
  "workspace.search",
  "documents.list",
  "documents.read",
  "documents.search",
]
default_capabilities = ["workspace.read", "workspace.search"]
max_document_sources = 16

[policy.workspace]
root = "."
include = ["**/*"]
exclude = ["taskchamber.toml"]
allow_globs = true
max_requested_paths = 64

[tasks.research]
allowed_capabilities = [
  "workspace.list",
  "workspace.read",
  "workspace.search",
  "documents.list",
  "documents.read",
  "documents.search",
]
default_capabilities = [
  "workspace.list",
  "workspace.read",
  "workspace.search",
  "documents.list",
  "documents.read",
  "documents.search",
]
max_turns = 25

[tasks.summarize]
allowed_capabilities = [
  "workspace.read",
  "workspace.search",
  "documents.list",
  "documents.read",
  "documents.search",
]
default_capabilities = [
  "workspace.read",
  "documents.list",
  "documents.read",
  "documents.search",
]
max_turns = 15

[tasks.review]
allowed_capabilities = [
  "workspace.list",
  "workspace.read",
  "workspace.search",
  "documents.list",
  "documents.read",
  "documents.search",
]
default_capabilities = ["workspace.list", "workspace.read", "workspace.search"]
max_turns = 20

# Example fixed command source. Secrets stay in .env or another SecretProvider;
# only their reference names belong here.
# [document_sources.record_detail]
# kind = "command"
# description = "Retrieve one record from an internal API"
# aliases = ["record detail", "record lookup"]
# executable = "/absolute/path/to/records-cli"
# args = ["records", "get", "{record_id}"]
# env_refs = ["RECORDS_API_TOKEN"]
# output_format = "json_document"
# document_id = "record.json"
# timeout_seconds = 20
#
# [document_sources.record_detail.parameters.record_id]
# description = "Record identifier"
# aliases = ["record", "id"]
# pattern = "^rec_[A-Za-z0-9._-]{1,120}$"
# example = "rec_20260715_a"
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taskchamber",
        description=ROOT_DESCRIPTION,
        epilog=ROOT_EPILOG,
        formatter_class=HELP_FORMATTER,
    )
    commands = parser.add_subparsers(dest="command")

    commands.add_parser(
        "serve",
        help="run the stdio MCP server",
        description=SERVE_DESCRIPTION,
        epilog=SERVE_EPILOG,
        formatter_class=HELP_FORMATTER,
    )

    config = commands.add_parser(
        "config",
        help="create project configuration",
        description=CONFIG_DESCRIPTION,
        epilog=CONFIG_EPILOG,
        formatter_class=HELP_FORMATTER,
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    init = config_commands.add_parser(
        "init",
        help="create a documented taskchamber.toml",
        description="Create a documented TaskChamber project-policy template.",
        epilog="example:\n  taskchamber config init --path taskchamber.toml",
        formatter_class=HELP_FORMATTER,
    )
    init.add_argument(
        "--path",
        type=Path,
        default=Path(DEFAULT_CONFIG_FILE),
        help=f"configuration file to create (default: {DEFAULT_CONFIG_FILE})",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="replace the target file if it already exists",
    )

    policy = commands.add_parser(
        "policy",
        help="inspect or edit project policy",
        description=POLICY_DESCRIPTION,
        epilog=POLICY_EPILOG,
        formatter_class=HELP_FORMATTER,
    )
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    show = policy_commands.add_parser(
        "show",
        help="print the effective non-secret policy as JSON",
        description="Print the effective non-secret project policy as JSON.",
        formatter_class=HELP_FORMATTER,
    )
    _add_config_argument(show)
    validate = policy_commands.add_parser(
        "validate",
        help="load and validate the project policy",
        description="Validate the project policy without changing it.",
        formatter_class=HELP_FORMATTER,
    )
    _add_config_argument(validate)

    edit_commands = {
        "allow": (
            "allow capabilities for one task",
            "Add capabilities to one task's allowed set.",
        ),
        "deny": (
            "deny capabilities for one task",
            "Remove capabilities from one task's allowed and default sets.",
        ),
        "set-default": (
            "replace the default capabilities for one task",
            "Replace one task's default capability set.",
        ),
    }
    for name, (help_text, description) in edit_commands.items():
        subcommand = policy_commands.add_parser(
            name,
            help=help_text,
            description=description,
            formatter_class=HELP_FORMATTER,
        )
        subcommand.add_argument(
            "task",
            choices=[kind.value for kind in TaskKind],
            help="task policy to edit",
        )
        subcommand.add_argument(
            "capabilities",
            nargs="+",
            metavar="CAPABILITY",
            help="one or more provider-neutral capability names",
        )
        _add_config_argument(subcommand)

    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "explicit project policy file "
            f"(default: TASKCHAMBER_CONFIG_FILE or ./{DEFAULT_CONFIG_FILE})"
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one management command, or stdio MCP when no command is supplied."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["serve"]:
        from .transport.mcp import main as serve

        serve()
        return
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "config" and args.config_command == "init":
            _config_init(args.path, force=args.force)
            return
        if args.command == "policy":
            if args.policy_command == "show":
                _policy_show(args.config)
                return
            if args.policy_command == "validate":
                _policy_validate(args.config)
                return
            if args.policy_command in {"allow", "deny", "set-default"}:
                _policy_edit(
                    args.config,
                    task=TaskKind(args.task),
                    operation=args.policy_command,
                    capabilities=args.capabilities,
                )
                return
        parser.error("a command is required")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _config_init(path: Path, *, force: bool) -> None:
    target = path.expanduser().resolve()
    if target.exists() and not force:
        raise ValueError(f"config file already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    print(f"Created {target}")


def _load(config_file: Path | None) -> tuple[Any, Path | None]:
    working_directory = Path.cwd().resolve()
    configuration = load_configuration(
        environment=os.environ,
        working_directory=working_directory,
    )
    loaded = load_project_policy(
        configuration,
        working_directory=working_directory,
        config_file=config_file,
    )
    return loaded, loaded.path


def _policy_show(config_file: Path | None) -> None:
    loaded, path = _load(config_file)
    policy = loaded.policy
    payload = {
        "config_file": str(path) if path is not None else None,
        "allowed_capabilities": list(policy.allowed_capabilities),
        "default_capabilities": list(policy.default_capabilities),
        "workspace": {
            "root": str(loaded.workspace_root) if loaded.workspace_root is not None else None,
            "include": list(policy.workspace.include),
            "exclude": list(policy.workspace.exclude),
            "allow_globs": policy.workspace.allow_globs,
            "max_requested_paths": policy.workspace.max_requested_paths,
        },
        "tasks": {
            kind.value: {
                "allowed_capabilities": list(task.allowed),
                "default_capabilities": list(task.defaults),
                "max_turns": task.max_turns,
            }
            for kind, task in policy.tasks.items()
        },
        "document_sources": {
            name: {
                "kind": "command" if hasattr(source, "argv") else "directory",
                "description": source.description,
                "aliases": list(source.aliases),
                "parameters": [parameter.name for parameter in getattr(source, "parameters", ())],
            }
            for name, source in loaded.document_sources.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _policy_validate(config_file: Path | None) -> None:
    loaded, path = _load(config_file)
    source = str(path) if path is not None else "built-in defaults"
    print(
        f"Valid TaskChamber policy: {source} "
        f"({len(loaded.policy.allowed_capabilities)} capabilities, "
        f"{len(loaded.document_sources)} document sources)"
    )


def _policy_edit(
    config_file: Path | None,
    *,
    task: TaskKind,
    operation: str,
    capabilities: Sequence[str],
) -> None:
    loaded, selected = _load(config_file)
    if selected is None:
        raise ValueError("no taskchamber.toml exists; run 'taskchamber config init' first")
    requested = tuple(dict.fromkeys(normalize_capability(value) for value in capabilities))
    if any(value not in KNOWN_CAPABILITIES for value in requested):
        raise ValueError("unknown capability")
    if operation != "deny" and not set(requested).issubset(loaded.policy.allowed_capabilities):
        raise ValueError("the edit would exceed policy.allowed_capabilities")

    document = tomlkit.parse(selected.read_text(encoding="utf-8"))
    tasks = document.get("tasks")
    if tasks is None:
        tasks = tomlkit.table()
        document["tasks"] = tasks
    task_table = tasks.get(task.value)
    if task_table is None:
        task_table = tomlkit.table()
        tasks[task.value] = task_table

    effective = loaded.policy.tasks[task]
    allowed = list(effective.allowed)
    defaults = list(effective.defaults)
    if operation == "allow":
        for capability in requested:
            if capability not in allowed:
                allowed.append(capability)
    elif operation == "deny":
        allowed = [capability for capability in allowed if capability not in requested]
        defaults = [capability for capability in defaults if capability not in requested]
    else:
        if not set(requested).issubset(allowed):
            raise ValueError("default capabilities must already be allowed for the task")
        defaults = list(requested)

    task_table["allowed_capabilities"] = allowed
    task_table["default_capabilities"] = defaults
    _atomic_write(
        selected,
        tomlkit.dumps(document),
        validate=lambda candidate: _load(candidate),
    )
    print(f"Updated {task.value} policy in {selected}")


def _atomic_write(
    path: Path,
    content: str,
    *,
    validate: Callable[[Path], object] | None = None,
) -> None:
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        if validate is not None:
            validate(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["DEFAULT_CONFIG_TEMPLATE", "build_parser", "main"]

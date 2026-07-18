"""Load the non-secret project policy from ``taskchamber.toml``."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from ..core.capabilities import (
    KNOWN_CAPABILITIES,
    ProjectPolicy,
    TaskCapabilityPolicy,
    WorkspaceAccessPolicy,
    default_project_policy,
    normalize_capability,
)
from ..core.contracts import TaskKind
from .documents import (
    CommandDocumentSourceConfig,
    DirectoryDocumentSourceConfig,
    DocumentParameterConfig,
    DocumentSourceConfig,
    validate_argv,
)
from .loader import ConfigurationBundle

CONFIG_FILE_VARIABLE = "TASKCHAMBER_CONFIG_FILE"
DEFAULT_CONFIG_FILE = "taskchamber.toml"
_SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class LoadedProjectPolicy:
    """An effective policy and the TOML file that supplied it, if any."""

    policy: ProjectPolicy
    path: Path | None
    document_sources: Mapping[str, DocumentSourceConfig]
    workspace_root: Path | None


def load_project_policy(
    configuration: ConfigurationBundle,
    *,
    working_directory: Path,
    config_file: Path | None = None,
) -> LoadedProjectPolicy:
    """Load a strict project policy or return broad built-in read-only defaults."""

    selected = _select_config_file(
        configuration,
        working_directory=working_directory,
        explicit=config_file,
    )
    if selected is None:
        return LoadedProjectPolicy(
            default_project_policy(),
            None,
            MappingProxyType({}),
            None,
        )
    try:
        with selected.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"could not read TaskChamber config file: {selected}") from exc
    return LoadedProjectPolicy(
        _parse_project_policy(raw),
        selected,
        _parse_document_sources(raw.get("document_sources", {}), base_directory=selected.parent),
        _parse_workspace_root(raw, base_directory=selected.parent),
    )


def _parse_workspace_root(raw: Mapping[str, Any], *, base_directory: Path) -> Path | None:
    policy = _table(raw.get("policy", {}), field="policy")
    workspace = _table(policy.get("workspace", {}), field="policy.workspace")
    value = workspace.get("root")
    if value is None:
        return None
    return _resolve_path(
        _string(value, field="policy.workspace.root"), base_directory=base_directory
    )


def _select_config_file(
    configuration: ConfigurationBundle,
    *,
    working_directory: Path,
    explicit: Path | None,
) -> Path | None:
    configured = configuration.values.get(CONFIG_FILE_VARIABLE)
    requested = explicit or (Path(configured) if configured else None)
    if requested is None:
        candidate = (working_directory / DEFAULT_CONFIG_FILE).resolve()
        return candidate if candidate.is_file() else None
    path = requested.expanduser()
    if not path.is_absolute():
        path = working_directory / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"TaskChamber config file does not exist: {path}")
    return path


def _parse_project_policy(raw: Mapping[str, Any]) -> ProjectPolicy:
    _reject_unknown(raw, {"schema_version", "policy", "tasks", "document_sources"}, "root")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError("schema_version must be 1")

    defaults = default_project_policy()
    policy_table = _table(raw.get("policy", {}), field="policy")
    _reject_unknown(
        policy_table,
        {
            "allowed_capabilities",
            "default_capabilities",
            "max_document_sources",
            "workspace",
        },
        "policy",
    )
    allowed = _capabilities(
        policy_table.get("allowed_capabilities"),
        default=defaults.allowed_capabilities,
        field="policy.allowed_capabilities",
    )
    default_capabilities = _capabilities(
        policy_table.get("default_capabilities"),
        default=tuple(
            capability for capability in defaults.default_capabilities if capability in allowed
        ),
        field="policy.default_capabilities",
    )
    if not set(default_capabilities).issubset(allowed):
        raise ValueError("policy.default_capabilities must be allowed by the project policy")

    workspace_table = _table(policy_table.get("workspace", {}), field="policy.workspace")
    _reject_unknown(
        workspace_table,
        {"root", "include", "exclude", "allow_globs", "max_requested_paths"},
        "policy.workspace",
    )
    workspace = WorkspaceAccessPolicy(
        include=_patterns(
            workspace_table.get("include"),
            default=defaults.workspace.include,
            field="policy.workspace.include",
        ),
        exclude=_patterns(
            workspace_table.get("exclude"),
            default=defaults.workspace.exclude,
            field="policy.workspace.exclude",
        ),
        allow_globs=_boolean(
            workspace_table.get("allow_globs"),
            default=defaults.workspace.allow_globs,
            field="policy.workspace.allow_globs",
        ),
        max_requested_paths=_positive_int(
            workspace_table.get("max_requested_paths"),
            default=defaults.workspace.max_requested_paths,
            field="policy.workspace.max_requested_paths",
            maximum=1_000,
        ),
    )

    tasks_table = _table(raw.get("tasks", {}), field="tasks")
    _reject_unknown(tasks_table, {kind.value for kind in TaskKind}, "tasks")
    tasks: dict[TaskKind, TaskCapabilityPolicy] = {}
    for kind in TaskKind:
        built_in = defaults.tasks[kind]
        task_table = _table(tasks_table.get(kind.value, {}), field=f"tasks.{kind.value}")
        _reject_unknown(
            task_table,
            {"allowed_capabilities", "default_capabilities", "max_turns"},
            f"tasks.{kind.value}",
        )
        task_allowed = _capabilities(
            task_table.get("allowed_capabilities"),
            default=tuple(capability for capability in built_in.allowed if capability in allowed),
            field=f"tasks.{kind.value}.allowed_capabilities",
        )
        if not set(task_allowed).issubset(allowed):
            raise ValueError(f"tasks.{kind.value}.allowed_capabilities exceeds the project policy")
        built_in_defaults = tuple(
            capability for capability in built_in.defaults if capability in task_allowed
        )
        task_defaults = _capabilities(
            task_table.get("default_capabilities"),
            default=built_in_defaults,
            field=f"tasks.{kind.value}.default_capabilities",
        )
        tasks[kind] = TaskCapabilityPolicy(
            allowed=task_allowed,
            defaults=task_defaults,
            max_turns=_positive_int(
                task_table.get("max_turns"),
                default=built_in.max_turns,
                field=f"tasks.{kind.value}.max_turns",
                maximum=built_in.max_turns,
            ),
        )

    return ProjectPolicy(
        allowed_capabilities=allowed,
        default_capabilities=default_capabilities,
        workspace=workspace,
        tasks=MappingProxyType(tasks),
        max_document_sources=_positive_int(
            policy_table.get("max_document_sources"),
            default=defaults.max_document_sources,
            field="policy.max_document_sources",
            maximum=100,
        ),
    )


def _parse_document_sources(
    value: object,
    *,
    base_directory: Path,
) -> Mapping[str, DocumentSourceConfig]:
    table = _table(value, field="document_sources")
    sources: dict[str, DocumentSourceConfig] = {}
    for name, raw_source in table.items():
        if not _SOURCE_NAME.fullmatch(name):
            raise ValueError("document source names must be lowercase identifiers")
        source = _table(raw_source, field=f"document_sources.{name}")
        kind = source.get("kind")
        if kind == "directory":
            _reject_unknown(
                source,
                {
                    "kind",
                    "root",
                    "description",
                    "aliases",
                    "include",
                    "exclude",
                    "max_file_bytes",
                    "max_total_bytes",
                    "max_files",
                },
                f"document_sources.{name}",
            )
            root_value = _string(source.get("root"), field=f"document_sources.{name}.root")
            sources[name] = DirectoryDocumentSourceConfig(
                name=name,
                root=_resolve_path(root_value, base_directory=base_directory),
                description=_optional_string(
                    source.get("description"),
                    field=f"document_sources.{name}.description",
                ),
                aliases=_strings(
                    source.get("aliases"),
                    default=(),
                    field=f"document_sources.{name}.aliases",
                ),
                include=_patterns(
                    source.get("include"),
                    default=("**/*",),
                    field=f"document_sources.{name}.include",
                ),
                exclude=_patterns(
                    source.get("exclude"),
                    default=(),
                    field=f"document_sources.{name}.exclude",
                ),
                max_file_bytes=_positive_int(
                    source.get("max_file_bytes"),
                    default=1_000_000,
                    field=f"document_sources.{name}.max_file_bytes",
                    maximum=1_000_000_000,
                ),
                max_total_bytes=_positive_int(
                    source.get("max_total_bytes"),
                    default=50_000_000,
                    field=f"document_sources.{name}.max_total_bytes",
                    maximum=10_000_000_000,
                ),
                max_files=_positive_int(
                    source.get("max_files"),
                    default=10_000,
                    field=f"document_sources.{name}.max_files",
                    maximum=1_000_000,
                ),
            )
            continue
        if kind == "command":
            _reject_unknown(
                source,
                {
                    "kind",
                    "executable",
                    "args",
                    "description",
                    "aliases",
                    "parameters",
                    "cwd",
                    "env_refs",
                    "output_format",
                    "document_id",
                    "timeout_seconds",
                    "max_output_bytes",
                    "max_document_bytes",
                    "max_documents",
                },
                f"document_sources.{name}",
            )
            executable = _string(
                source.get("executable"),
                field=f"document_sources.{name}.executable",
            )
            executable = _resolve_executable_config(executable, base_directory=base_directory)
            args = source.get("args", [])
            if not isinstance(args, list):
                raise ValueError(f"document_sources.{name}.args must be an array")
            parameters = _document_parameters(
                source.get("parameters", {}),
                field=f"document_sources.{name}.parameters",
            )
            placeholders = frozenset({"query", *(parameter.name for parameter in parameters)})
            argv = validate_argv(
                [executable, *args],
                field=f"document_sources.{name}.args",
                allowed_placeholders=placeholders,
            )
            output_format = source.get("output_format", "text")
            if output_format not in {"text", "json", "json_document"}:
                raise ValueError(
                    f"document_sources.{name}.output_format must be "
                    "'text', 'json', or 'json_document'"
                )
            cwd_value = source.get("cwd")
            if cwd_value is not None and not isinstance(cwd_value, str):
                raise ValueError(f"document_sources.{name}.cwd must be a string")
            env_refs = _strings(
                source.get("env_refs"),
                default=(),
                field=f"document_sources.{name}.env_refs",
            )
            if any(_ENV_NAME.fullmatch(reference) is None for reference in env_refs):
                raise ValueError(
                    f"document_sources.{name}.env_refs contains an invalid environment name"
                )
            sources[name] = CommandDocumentSourceConfig(
                name=name,
                argv=argv,
                description=_optional_string(
                    source.get("description"),
                    field=f"document_sources.{name}.description",
                ),
                aliases=_strings(
                    source.get("aliases"),
                    default=(),
                    field=f"document_sources.{name}.aliases",
                ),
                parameters=parameters,
                cwd=(
                    _resolve_path(cwd_value, base_directory=base_directory) if cwd_value else None
                ),
                env_allow=env_refs,
                output_format=output_format,
                document_id=_optional_string(
                    source.get("document_id"),
                    field=f"document_sources.{name}.document_id",
                )
                or "output.txt",
                timeout_seconds=_positive_number(
                    source.get("timeout_seconds"),
                    default=30.0,
                    field=f"document_sources.{name}.timeout_seconds",
                    maximum=3_600.0,
                ),
                max_output_bytes=_positive_int(
                    source.get("max_output_bytes"),
                    default=5_000_000,
                    field=f"document_sources.{name}.max_output_bytes",
                    maximum=1_000_000_000,
                ),
                max_document_bytes=_positive_int(
                    source.get("max_document_bytes"),
                    default=1_000_000,
                    field=f"document_sources.{name}.max_document_bytes",
                    maximum=1_000_000_000,
                ),
                max_documents=_positive_int(
                    source.get("max_documents"),
                    default=1_000,
                    field=f"document_sources.{name}.max_documents",
                    maximum=1_000_000,
                ),
            )
            continue
        raise ValueError(f"document_sources.{name}.kind must be 'directory' or 'command'")
    return MappingProxyType(sources)


def _document_parameters(value: object, *, field: str) -> tuple[DocumentParameterConfig, ...]:
    table = _table(value, field=field)
    result: list[DocumentParameterConfig] = []
    for name, raw_parameter in table.items():
        parameter = _table(raw_parameter, field=f"{field}.{name}")
        _reject_unknown(
            parameter,
            {"pattern", "description", "aliases", "example", "max_length"},
            f"{field}.{name}",
        )
        example = parameter.get("example")
        if example is not None and not isinstance(example, str):
            raise ValueError(f"{field}.{name}.example must be a string")
        result.append(
            DocumentParameterConfig(
                name=name,
                pattern=_string(parameter.get("pattern"), field=f"{field}.{name}.pattern"),
                description=_optional_string(
                    parameter.get("description"),
                    field=f"{field}.{name}.description",
                ),
                aliases=_strings(
                    parameter.get("aliases"),
                    default=(),
                    field=f"{field}.{name}.aliases",
                ),
                example=example,
                max_length=_positive_int(
                    parameter.get("max_length"),
                    default=200,
                    field=f"{field}.{name}.max_length",
                    maximum=4_000,
                ),
            )
        )
    return tuple(result)


def _capabilities(value: object, *, default: tuple[str, ...], field: str) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result: list[str] = []
    for item in value:
        try:
            capability = normalize_capability(item)
        except ValueError as exc:
            raise ValueError(f"{field}: {exc}") from exc
        if capability not in result:
            result.append(capability)
    if any(capability not in KNOWN_CAPABILITIES for capability in result):
        raise ValueError(f"{field} contains an unknown capability")
    return tuple(result)


def _patterns(value: object, *, default: tuple[str, ...], field: str) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of strings")
    result = tuple(item.strip() for item in value if item.strip())
    if any("\x00" in item or len(item) > 500 for item in result):
        raise ValueError(f"{field} contains an invalid pattern")
    if any(item.startswith(("/", "~")) or ".." in PurePosixPath(item).parts for item in result):
        raise ValueError(f"{field} patterns must remain relative")
    return result


def _strings(value: object, *, default: tuple[str, ...], field: str) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of strings")
    result = tuple(item.strip() for item in value if item.strip())
    if any("\x00" in item or len(item) > 500 for item in result):
        raise ValueError(f"{field} contains an invalid value")
    return result


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or "\x00" in value or len(value) > 4_000:
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _resolve_path(value: str, *, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _resolve_executable_config(value: str, *, base_directory: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if path.parent != Path("."):
        return str((base_directory / path).resolve())
    return value


def _table(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a table")
    return value


def _boolean(value: object, *, default: bool, field: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _positive_int(value: object, *, default: int, field: str, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise ValueError(f"{field} must be an integer between 1 and {maximum}")
    return value


def _positive_number(value: object, *, default: float, field: str, maximum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if result <= 0 or result > maximum:
        raise ValueError(f"{field} must be between 0 and {maximum}")
    return result


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")


__all__ = [
    "CONFIG_FILE_VARIABLE",
    "DEFAULT_CONFIG_FILE",
    "LoadedProjectPolicy",
    "load_project_policy",
]

"""Configuration models for named document sources."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .loader import ConfigurationBundle

DOCUMENT_SOURCES_VARIABLE = "TASKCHAMBER_DOCUMENT_SOURCES"
DOCUMENT_SOURCE_PREFIX = "TASKCHAMBER_DOCUMENT_SOURCE"

_SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PLACEHOLDER = re.compile(r"^\{([a-z][a-z0-9_]{0,63})\}$")
_SHELL_EXECUTABLES = frozenset(
    {"bash", "cmd", "cmd.exe", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "zsh"}
)


@dataclass(frozen=True)
class DirectoryDocumentSourceConfig:
    """A server-owned directory exposed through virtual document tools."""

    name: str
    root: Path
    description: str = ""
    aliases: tuple[str, ...] = ()
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 50_000_000
    max_files: int = 10_000

    def __post_init__(self) -> None:
        _validate_source_metadata(self.name, self.description, self.aliases)


@dataclass(frozen=True)
class DocumentParameterConfig:
    """One typed value accepted by a fixed command document source."""

    name: str
    pattern: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    example: str | None = None
    max_length: int = 200

    def __post_init__(self) -> None:
        if not _PARAMETER_NAME.fullmatch(self.name):
            raise ValueError("document parameter names must be lowercase identifiers")
        if not self.pattern or len(self.pattern) > 1_000:
            raise ValueError(f"document parameter {self.name!r} requires a bounded pattern")
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"document parameter {self.name!r} has an invalid pattern") from exc
        if self.max_length < 1 or self.max_length > 4_000:
            raise ValueError(f"document parameter {self.name!r} has an invalid max_length")
        _validate_aliases(self.aliases, field=f"document parameter {self.name!r}")


@dataclass(frozen=True)
class CommandDocumentSourceConfig:
    """A fixed argv command whose stdout becomes virtual documents."""

    name: str
    argv: tuple[str, ...]
    description: str = ""
    aliases: tuple[str, ...] = ()
    parameters: tuple[DocumentParameterConfig, ...] = ()
    cwd: Path | None = None
    env_allow: tuple[str, ...] = ()
    output_format: str = "text"
    document_id: str = "output.txt"
    timeout_seconds: float = 30.0
    max_output_bytes: int = 5_000_000
    max_document_bytes: int = 1_000_000
    max_documents: int = 1_000

    def __post_init__(self) -> None:
        _validate_source_metadata(self.name, self.description, self.aliases)
        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(f"document source {self.name!r} has duplicate parameters")
        validate_argv(
            list(self.argv),
            field=f"document source {self.name!r} argv",
            allowed_placeholders=frozenset({"query", *parameter_names}),
        )


DocumentSourceConfig = DirectoryDocumentSourceConfig | CommandDocumentSourceConfig


def load_document_source_configs(
    configuration: ConfigurationBundle,
    *,
    base_directory: Path,
) -> Mapping[str, DocumentSourceConfig]:
    """Load named directory and command sources from ordered configuration."""

    names = _source_names(configuration.values.get(DOCUMENT_SOURCES_VARIABLE, "") or "")
    sources: dict[str, DocumentSourceConfig] = {}
    for name in names:
        token = name.upper()
        prefix = f"{DOCUMENT_SOURCE_PREFIX}__{token}__"
        kind = (configuration.values.get(prefix + "KIND", "") or "").strip().lower()
        if kind == "directory":
            root_value = _required(configuration, prefix + "ROOT")
            root = _resolve_path(root_value, base_directory=base_directory)
            sources[name] = DirectoryDocumentSourceConfig(
                name=name,
                root=root,
                include=_patterns(
                    configuration.values.get(prefix + "INCLUDE"),
                    default=("**/*",),
                ),
                exclude=_patterns(configuration.values.get(prefix + "EXCLUDE"), default=()),
                max_file_bytes=_positive_int(
                    configuration.values.get(prefix + "MAX_FILE_BYTES"),
                    default=1_000_000,
                    field=prefix + "MAX_FILE_BYTES",
                ),
                max_total_bytes=_positive_int(
                    configuration.values.get(prefix + "MAX_TOTAL_BYTES"),
                    default=50_000_000,
                    field=prefix + "MAX_TOTAL_BYTES",
                ),
                max_files=_positive_int(
                    configuration.values.get(prefix + "MAX_FILES"),
                    default=10_000,
                    field=prefix + "MAX_FILES",
                ),
            )
            continue
        if kind == "command":
            argv = _argv(
                _required(configuration, prefix + "ARGV"),
                field=prefix + "ARGV",
                allowed_placeholders=frozenset({"query"}),
            )
            cwd_value = configuration.values.get(prefix + "CWD")
            output_format = (
                (configuration.values.get(prefix + "OUTPUT_FORMAT", "text") or "text")
                .strip()
                .lower()
            )
            if output_format not in {"text", "json", "json_document"}:
                raise ValueError(
                    f"{prefix}OUTPUT_FORMAT must be 'text', 'json', or 'json_document'"
                )
            document_id = (
                configuration.values.get(prefix + "DOCUMENT_ID", "output.txt") or "output.txt"
            ).strip()
            _validate_document_id(document_id, field=prefix + "DOCUMENT_ID")
            sources[name] = CommandDocumentSourceConfig(
                name=name,
                argv=argv,
                cwd=(
                    _resolve_path(cwd_value, base_directory=base_directory) if cwd_value else None
                ),
                env_allow=_env_names(configuration.values.get(prefix + "ENV_ALLOW")),
                output_format=output_format,
                document_id=document_id,
                timeout_seconds=_positive_float(
                    configuration.values.get(prefix + "TIMEOUT_SECONDS"),
                    default=30.0,
                    field=prefix + "TIMEOUT_SECONDS",
                ),
                max_output_bytes=_positive_int(
                    configuration.values.get(prefix + "MAX_OUTPUT_BYTES"),
                    default=5_000_000,
                    field=prefix + "MAX_OUTPUT_BYTES",
                ),
                max_document_bytes=_positive_int(
                    configuration.values.get(prefix + "MAX_DOCUMENT_BYTES"),
                    default=1_000_000,
                    field=prefix + "MAX_DOCUMENT_BYTES",
                ),
                max_documents=_positive_int(
                    configuration.values.get(prefix + "MAX_DOCUMENTS"),
                    default=1_000,
                    field=prefix + "MAX_DOCUMENTS",
                ),
            )
            continue
        raise ValueError(f"{prefix}KIND must be 'directory' or 'command'")
    return sources


def _source_names(raw: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in raw.split(","):
        name = value.strip().lower()
        if not name:
            continue
        if not _SOURCE_NAME.fullmatch(name):
            raise ValueError("TASKCHAMBER_DOCUMENT_SOURCES must contain lowercase source names")
        if name in result:
            raise ValueError(f"duplicate document source {name!r}")
        result.append(name)
    return tuple(result)


def _validate_source_metadata(name: str, description: str, aliases: tuple[str, ...]) -> None:
    if not _SOURCE_NAME.fullmatch(name):
        raise ValueError("document source names must be lowercase identifiers")
    if len(description) > 4_000 or "\x00" in description:
        raise ValueError(f"document source {name!r} has an invalid description")
    _validate_aliases(aliases, field=f"document source {name!r}")


def _validate_aliases(aliases: tuple[str, ...], *, field: str) -> None:
    if any(not alias.strip() or len(alias) > 200 or "\x00" in alias for alias in aliases):
        raise ValueError(f"{field} contains an invalid alias")


def _required(configuration: ConfigurationBundle, name: str) -> str:
    value = configuration.values.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _resolve_path(value: str, *, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _patterns(raw: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return default
    patterns = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not patterns:
        return default
    if any("\x00" in pattern or len(pattern) > 500 for pattern in patterns):
        raise ValueError("document source patterns must be at most 500 characters")
    return patterns


def validate_argv(
    value: object,
    *,
    field: str,
    allowed_placeholders: frozenset[str],
) -> tuple[str, ...]:
    """Validate a direct argv list and standalone, declared placeholders."""

    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError(f"{field} must contain between 1 and 64 argv strings")
    argv: list[str] = []
    used_placeholders: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 4_000 or "\x00" in item:
            raise ValueError(f"{field} contains an invalid argv item")
        match = _PLACEHOLDER.fullmatch(item)
        if match:
            placeholder = match.group(1)
            if placeholder not in allowed_placeholders:
                raise ValueError(
                    f"{field} only supports declared placeholders; {{{placeholder}}} is undeclared"
                )
            used_placeholders.add(placeholder)
        elif any(f"{{{name}}}" in item for name in allowed_placeholders):
            raise ValueError(f"{field} requires placeholders to be standalone argv items")
        argv.append(item)
    unused = allowed_placeholders - used_placeholders - {"query"}
    if unused:
        raise ValueError(f"{field} does not use declared parameters: {', '.join(sorted(unused))}")
    if Path(argv[0]).name.casefold() in _SHELL_EXECUTABLES:
        raise ValueError(f"{field} may not invoke a command shell")
    return tuple(argv)


def _argv(
    raw: str,
    *,
    field: str,
    allowed_placeholders: frozenset[str],
) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must be a JSON argv array") from exc
    return validate_argv(
        value,
        field=field,
        allowed_placeholders=allowed_placeholders,
    )


def _env_names(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    if any(not _ENV_NAME.fullmatch(name) for name in names):
        raise ValueError("document command ENV_ALLOW contains an invalid environment name")
    return names


def _positive_int(raw: str | None, *, default: int, field: str) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_float(raw: str | None, *, default: float, field: str) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be positive") from exc
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _validate_document_id(value: str, *, field: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or ".." in path.parts
        or value.startswith("~")
        or "\\" in value
        or "\x00" in value
        or len(value) > 1_000
    ):
        raise ValueError(f"{field} must be a safe virtual relative path")


__all__ = [
    "DOCUMENT_SOURCE_PREFIX",
    "DOCUMENT_SOURCES_VARIABLE",
    "CommandDocumentSourceConfig",
    "DirectoryDocumentSourceConfig",
    "DocumentSourceConfig",
    "load_document_source_configs",
]

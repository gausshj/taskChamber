"""Application adapters for server-configured virtual document sources."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path, PurePosixPath

from ..config import (
    CommandDocumentSourceConfig,
    ConfigurationBundle,
    DirectoryDocumentSourceConfig,
    DocumentParameterConfig,
    DocumentSourceConfig,
)
from ..core.documents import (
    DocumentCatalog,
    DocumentInfo,
    DocumentPage,
    DocumentRequestError,
    DocumentSearchHit,
    DocumentSource,
    DocumentSourceError,
    PreparedDocumentSource,
)
from ..core.path_globs import path_glob_matches
from ..core.policy import WorkspaceGuard

_BASE_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "TMPDIR")
_SOURCE_SEPARATORS = re.compile(r"[\s.-]+")


class DocumentSourceRegistry:
    """Select named sources without accepting commands or paths from MCP callers."""

    def __init__(self, sources: Mapping[str, DocumentSource]) -> None:
        self._sources = dict(sources)
        aliases: dict[str, str] = {}
        for name, source in self._sources.items():
            for value in (name, *getattr(source, "aliases", ())):
                normalized = _normalize_source_hint(value)
                existing = aliases.get(normalized)
                if existing is not None and existing != name:
                    raise ValueError(
                        f"document source alias {value!r} is ambiguous between "
                        f"{existing!r} and {name!r}"
                    )
                aliases[normalized] = name
        self._aliases = aliases

    @property
    def available_names(self) -> tuple[str, ...]:
        return tuple(self._sources)

    @property
    def public_catalog(self) -> Mapping[str, object]:
        """Return redacted discovery metadata without commands, paths, or secrets."""

        result: dict[str, object] = {}
        for name, source in self._sources.items():
            parameters = {
                parameter.name: {
                    "description": parameter.description,
                    "aliases": list(parameter.aliases),
                    "example": parameter.example,
                    "pattern": parameter.pattern,
                    "max_length": parameter.max_length,
                }
                for parameter in getattr(source, "parameters", ())
            }
            result[name] = {
                "description": getattr(source, "description", ""),
                "aliases": list(getattr(source, "aliases", ())),
                "parameters": parameters,
            }
        return result

    async def open(
        self,
        names: tuple[str, ...],
        *,
        query: str,
        parameters: Mapping[str, Mapping[str, str]] | None = None,
    ) -> DocumentCatalog:
        if not names:
            raise DocumentRequestError("at least one document source must be selected")
        if len(names) > 16:
            raise DocumentRequestError("at most 16 document sources may be selected")

        selected: dict[str, PreparedDocumentSource] = {}
        for raw_name in names:
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise DocumentRequestError("document source names must not be empty")
            name = self._resolve_name(raw_name)
            if name in selected:
                raise DocumentRequestError(f"duplicate document source {name!r}")
            try:
                source = self._sources[name]
            except KeyError as exc:
                raise DocumentRequestError(f"document source {name!r} is not configured") from exc
            selected[name] = await source.prepare(
                query=query,
                parameters=(parameters or {}).get(raw_name),
            )
        return DocumentCatalog(selected)

    def _resolve_name(self, value: str) -> str:
        normalized = _normalize_source_hint(value)
        name = self._aliases.get(normalized)
        if name is not None:
            return name
        matches = get_close_matches(normalized, self._aliases, n=5, cutoff=0.4)
        suggestions = tuple(dict.fromkeys(self._aliases[match] for match in matches))[:3]
        suffix = f"; suggestions: {', '.join(suggestions)}" if suggestions else ""
        allowed = ", ".join(self.available_names) or "none"
        raise DocumentRequestError(
            f"document source {value!r} is not configured{suffix}; allowed values: {allowed}. "
            "Retry with a listed value and do not fall back to shell execution."
        )


class DirectoryDocumentSource:
    """Expose an external directory in place, without staging or copying it."""

    def __init__(self, config: DirectoryDocumentSourceConfig) -> None:
        self.name = config.name
        self.description = config.description
        self.aliases = config.aliases
        self.parameters: tuple[DocumentParameterConfig, ...] = ()
        self._config = config

    async def prepare(
        self,
        *,
        query: str,
        parameters: Mapping[str, str] | None = None,
    ) -> PreparedDocumentSource:
        del query
        if parameters:
            raise DocumentRequestError(
                f"directory document source {self.name!r} does not accept parameters"
            )
        root = self._config.root.expanduser().resolve()
        if not root.is_dir():
            raise DocumentSourceError(f"directory document source {self.name!r} is unavailable")
        documents = await asyncio.to_thread(self._index, root)
        return _DirectoryPreparedSource(self.name, root, documents, self._config)

    def _index(self, root: Path) -> Mapping[str, Path]:
        documents: dict[str, Path] = {}
        total_bytes = 0
        try:
            for path in root.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                relative = path.relative_to(root)
                document_id = relative.as_posix()
                if WorkspaceGuard.is_protected_relative(relative):
                    continue
                if not _matches(document_id, self._config.include):
                    continue
                if self._config.exclude and _matches(document_id, self._config.exclude):
                    continue
                size = path.stat().st_size
                if size > self._config.max_file_bytes or not _looks_textual(path):
                    continue
                if len(documents) + 1 > self._config.max_files:
                    raise DocumentSourceError(
                        f"directory document source {self.name!r} exceeds its file limit"
                    )
                if total_bytes + size > self._config.max_total_bytes:
                    raise DocumentSourceError(
                        f"directory document source {self.name!r} exceeds its byte limit"
                    )
                documents[document_id] = path
                total_bytes += size
        except DocumentSourceError:
            raise
        except OSError as exc:
            raise DocumentSourceError(
                f"directory document source {self.name!r} could not be indexed"
            ) from exc
        return documents


class _DirectoryPreparedSource:
    def __init__(
        self,
        name: str,
        root: Path,
        documents: Mapping[str, Path],
        config: DirectoryDocumentSourceConfig,
    ) -> None:
        self.name = name
        self._root = root
        self._documents = dict(documents)
        self._config = config

    async def list_documents(self, *, pattern: str | None, limit: int) -> tuple[DocumentInfo, ...]:
        results: list[DocumentInfo] = []
        for document_id, path in self._documents.items():
            if pattern and not path_glob_matches(document_id, pattern):
                continue
            results.append(self._info(document_id, path))
            if len(results) >= limit:
                break
        return tuple(results)

    async def read_document(
        self, document_id: str, *, start_line: int, max_lines: int
    ) -> DocumentPage:
        path = self._path_for(document_id)
        content = await asyncio.to_thread(
            _read_text_file,
            path,
            root=self._root,
            maximum=self._config.max_file_bytes,
            source=self.name,
        )
        return _page(self._info(document_id, path), content, start_line, max_lines)

    async def search_documents(
        self, query: str, *, pattern: str | None, limit: int
    ) -> tuple[DocumentSearchHit, ...]:
        results: list[DocumentSearchHit] = []
        needle = query.casefold()
        for document_id, path in self._documents.items():
            if pattern and not path_glob_matches(document_id, pattern):
                continue
            content = await asyncio.to_thread(
                _read_text_file,
                path,
                root=self._root,
                maximum=self._config.max_file_bytes,
                source=self.name,
            )
            for line_number, line in enumerate(content.splitlines(), start=1):
                if needle not in line.casefold():
                    continue
                results.append(
                    DocumentSearchHit(
                        source=self.name,
                        document_id=document_id,
                        line=line_number,
                        text=line[:1_000],
                    )
                )
                if len(results) >= limit:
                    return tuple(results)
        return tuple(results)

    def _path_for(self, document_id: str) -> Path:
        _validate_virtual_id(document_id)
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise DocumentRequestError(
                f"document {document_id!r} does not exist in source {self.name!r}"
            ) from exc

    def _info(self, document_id: str, path: Path) -> DocumentInfo:
        try:
            _resolved, size = _safe_file(
                path,
                root=self._root,
                maximum=self._config.max_file_bytes,
                source=self.name,
            )
        except DocumentSourceError as exc:
            raise DocumentSourceError(
                f"document {document_id!r} in source {self.name!r} is unavailable"
            ) from exc
        media_type = mimetypes.guess_type(document_id)[0] or "text/plain"
        return DocumentInfo(
            source=self.name,
            document_id=document_id,
            title=path.name,
            media_type=media_type,
            size_bytes=size,
            provenance=f"directory:{self.name}",
        )


class CommandDocumentSource:
    """Execute one fixed argv command and expose bounded stdout as documents."""

    def __init__(
        self,
        config: CommandDocumentSourceConfig,
        *,
        environment: Mapping[str, str],
    ) -> None:
        self.name = config.name
        self.description = config.description
        self.aliases = config.aliases
        self.parameters = config.parameters
        self._config = config
        self._environment = dict(environment)

    async def prepare(
        self,
        *,
        query: str,
        parameters: Mapping[str, str] | None = None,
    ) -> PreparedDocumentSource:
        executable = _resolve_executable(
            self._config.argv[0],
            cwd=self._config.cwd,
            path=self._environment.get("PATH"),
            source=self.name,
        )
        parameter_values = _command_parameters(self._config.parameters, parameters)

        def substitute(item: str) -> str:
            if item == "{query}":
                return query
            if item.startswith("{") and item.endswith("}"):
                return parameter_values[item[1:-1]]
            return item

        argv = (executable,) + tuple(substitute(item) for item in self._config.argv[1:])
        cwd = self._config.cwd
        if cwd is not None and not cwd.is_dir():
            raise DocumentSourceError(
                f"command document source {self.name!r} has an unavailable working directory"
            )

        stdout = await self._run(argv, cwd=cwd)
        documents = _parse_command_output(self._config, stdout)
        return _MemoryPreparedSource(self.name, documents, provenance=f"command:{self.name}")

    async def _run(self, argv: tuple[str, ...], *, cwd: Path | None) -> bytes:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=self._environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise DocumentSourceError(
                f"command document source {self.name!r} could not be started"
            ) from exc

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(
            _read_stream(process.stdout, self._config.max_output_bytes)
        )
        stderr_task = asyncio.create_task(_read_stream(process.stderr, 65_536))
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                stdout, _stderr = await asyncio.gather(stdout_task, stderr_task)
                return_code = await process.wait()
        except TimeoutError as exc:
            await _stop_process(process, stdout_task, stderr_task)
            raise DocumentSourceError(
                f"command document source {self.name!r} exceeded its time limit"
            ) from exc
        except _OutputLimitError as exc:
            await _stop_process(process, stdout_task, stderr_task)
            raise DocumentSourceError(
                f"command document source {self.name!r} exceeded its output limit"
            ) from exc
        except BaseException:
            await _stop_process(process, stdout_task, stderr_task)
            raise
        if return_code != 0:
            raise DocumentSourceError(
                f"command document source {self.name!r} failed with exit code {return_code}"
            )
        return stdout


@dataclass(frozen=True)
class _MemoryDocument:
    document_id: str
    title: str
    media_type: str
    content: str


class _MemoryPreparedSource:
    def __init__(
        self,
        name: str,
        documents: tuple[_MemoryDocument, ...],
        *,
        provenance: str,
    ) -> None:
        self.name = name
        self._documents = {document.document_id: document for document in documents}
        self._provenance = provenance

    async def list_documents(self, *, pattern: str | None, limit: int) -> tuple[DocumentInfo, ...]:
        results: list[DocumentInfo] = []
        for document in self._documents.values():
            if pattern and not path_glob_matches(document.document_id, pattern):
                continue
            results.append(self._info(document))
            if len(results) >= limit:
                break
        return tuple(results)

    async def read_document(
        self, document_id: str, *, start_line: int, max_lines: int
    ) -> DocumentPage:
        document = self._document(document_id)
        return _page(self._info(document), document.content, start_line, max_lines)

    async def search_documents(
        self, query: str, *, pattern: str | None, limit: int
    ) -> tuple[DocumentSearchHit, ...]:
        results: list[DocumentSearchHit] = []
        needle = query.casefold()
        for document in self._documents.values():
            if pattern and not path_glob_matches(document.document_id, pattern):
                continue
            for line_number, line in enumerate(document.content.splitlines(), start=1):
                if needle not in line.casefold():
                    continue
                results.append(
                    DocumentSearchHit(
                        source=self.name,
                        document_id=document.document_id,
                        line=line_number,
                        text=line[:1_000],
                    )
                )
                if len(results) >= limit:
                    return tuple(results)
        return tuple(results)

    def _document(self, document_id: str) -> _MemoryDocument:
        _validate_virtual_id(document_id)
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise DocumentRequestError(
                f"document {document_id!r} does not exist in source {self.name!r}"
            ) from exc

    def _info(self, document: _MemoryDocument) -> DocumentInfo:
        return DocumentInfo(
            source=self.name,
            document_id=document.document_id,
            title=document.title,
            media_type=document.media_type,
            size_bytes=len(document.content.encode("utf-8")),
            provenance=self._provenance,
        )


def build_document_source_registry(
    configs: Mapping[str, DocumentSourceConfig],
    configuration: ConfigurationBundle,
) -> DocumentSourceRegistry:
    """Build adapters while forwarding only explicitly allowlisted command env."""

    sources: dict[str, DocumentSource] = {}
    for name, config in configs.items():
        if isinstance(config, DirectoryDocumentSourceConfig):
            sources[name] = DirectoryDocumentSource(config)
            continue
        environment: dict[str, str] = {}
        for variable in _BASE_ENVIRONMENT + config.env_allow:
            value = configuration.secrets.get(variable)
            if value is not None:
                environment[variable] = value
        sources[name] = CommandDocumentSource(config, environment=environment)
    return DocumentSourceRegistry(sources)


def _normalize_source_hint(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise DocumentRequestError("document source names and aliases must be non-empty strings")
    return _SOURCE_SEPARATORS.sub("_", value.strip().casefold()).strip("_")


def _command_parameters(
    configs: tuple[DocumentParameterConfig, ...],
    supplied: Mapping[str, str] | None,
) -> Mapping[str, str]:
    configured = {config.name: config for config in configs}
    aliases: dict[str, str] = {}
    for config in configs:
        for alias in (config.name, *config.aliases):
            normalized_alias = _normalize_source_hint(alias)
            existing = aliases.get(normalized_alias)
            if existing is not None and existing != config.name:
                raise DocumentRequestError(f"document parameter alias {alias!r} is ambiguous")
            aliases[normalized_alias] = config.name
    values: dict[str, str] = {}
    for raw_name, value in (supplied or {}).items():
        normalized_name = _normalize_source_hint(raw_name)
        name = aliases.get(normalized_name)
        if name is None:
            matches = get_close_matches(normalized_name, aliases, n=5, cutoff=0.4)
            suggestions = tuple(dict.fromkeys(aliases[match] for match in matches))[:3]
            suffix = f"; suggestions: {', '.join(suggestions)}" if suggestions else ""
            allowed = ", ".join(configured) or "none"
            raise DocumentRequestError(
                f"unknown document parameter {raw_name!r}{suffix}; allowed values: {allowed}. "
                "Retry with a listed value and do not fall back to shell execution."
            )
        if name in values:
            raise DocumentRequestError(f"duplicate document parameter {name!r}")
        values[name] = value
    missing = [name for name in configured if name not in values]
    if missing:
        examples = ", ".join(
            f"{name}={configured[name].example!r}"
            for name in missing
            if configured[name].example is not None
        )
        suffix = f"; examples: {examples}" if examples else ""
        raise DocumentRequestError(f"missing document parameters: {', '.join(missing)}{suffix}")

    normalized: dict[str, str] = {}
    for name, config in configured.items():
        value = values[name]
        if not isinstance(value, str) or not value or len(value) > config.max_length:
            raise DocumentRequestError(
                f"document parameter {name!r} must be a non-empty string of at most "
                f"{config.max_length} characters"
            )
        if value.startswith("-"):
            raise DocumentRequestError(f"document parameter {name!r} may not begin with '-'")
        if re.fullmatch(config.pattern, value) is None:
            example = f"; example: {config.example}" if config.example else ""
            raise DocumentRequestError(
                f"document parameter {name!r} does not match its configured format{example}"
            )
        normalized[name] = value
    return normalized


def _matches(document_id: str, patterns: tuple[str, ...]) -> bool:
    return any(path_glob_matches(document_id, pattern) for pattern in patterns)


def _looks_textual(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return b"\x00" not in stream.read(4_096)
    except OSError:
        return False


def _read_text_file(path: Path, *, root: Path, maximum: int, source: str) -> str:
    try:
        resolved, _size = _safe_file(path, root=root, maximum=maximum, source=source)
        raw = resolved.read_bytes()
    except DocumentSourceError:
        raise
    except (OSError, ValueError) as exc:
        raise DocumentSourceError(f"document in source {source!r} is unavailable") from exc
    if b"\x00" in raw:
        raise DocumentSourceError(f"binary document in source {source!r} cannot be read")
    if len(raw) > maximum:
        raise DocumentSourceError(f"document in source {source!r} exceeds its limit")
    return raw.decode("utf-8", errors="replace")


def _safe_file(path: Path, *, root: Path, maximum: int, source: str) -> tuple[Path, int]:
    try:
        if path.is_symlink():
            raise DocumentSourceError(f"document in source {source!r} is no longer safe")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        metadata = resolved.stat()
    except DocumentSourceError:
        raise
    except (OSError, ValueError) as exc:
        raise DocumentSourceError(f"document in source {source!r} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DocumentSourceError(f"document in source {source!r} is not a regular file")
    if metadata.st_size > maximum:
        raise DocumentSourceError(f"document in source {source!r} exceeds its limit")
    return resolved, metadata.st_size


def _page(
    info: DocumentInfo,
    content: str,
    start_line: int,
    max_lines: int,
) -> DocumentPage:
    lines = content.splitlines()
    total = len(lines)
    start_index = min(start_line - 1, total)
    selected = lines[start_index : start_index + max_lines]
    return DocumentPage(
        document=info,
        start_line=start_line,
        end_line=start_index + len(selected),
        total_lines=total,
        content="\n".join(selected),
    )


def _resolve_executable(
    value: str,
    *,
    cwd: Path | None,
    path: str | None,
    source: str,
) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        if not candidate.is_absolute():
            candidate = (cwd or Path.cwd()) / candidate
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
    else:
        resolved_name = shutil.which(value, path=path)
        if resolved_name:
            return resolved_name
    raise DocumentSourceError(f"command document source {source!r} executable is unavailable")


class _OutputLimitError(RuntimeError):
    pass


async def _read_stream(stream: asyncio.StreamReader, maximum: int) -> bytes:
    result = bytearray()
    while True:
        chunk = await stream.read(min(65_536, maximum + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > maximum:
            raise _OutputLimitError


async def _stop_process(
    process: asyncio.subprocess.Process,
    *tasks: asyncio.Task[bytes],
) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _parse_command_output(
    config: CommandDocumentSourceConfig,
    stdout: bytes,
) -> tuple[_MemoryDocument, ...]:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentSourceError(
            f"command document source {config.name!r} returned non-UTF-8 output"
        ) from exc
    if config.output_format == "text":
        return (
            _memory_document(
                config,
                document_id=config.document_id,
                content=text,
                title=config.document_id,
                media_type="text/plain",
            ),
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentSourceError(
            f"command document source {config.name!r} returned invalid JSON"
        ) from exc
    if config.output_format == "json_document":
        return (
            _memory_document(
                config,
                document_id=config.document_id,
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                title=config.document_id,
                media_type="application/json",
            ),
        )

    items: object
    if isinstance(payload, dict) and "documents" in payload:
        items = payload["documents"]
    elif isinstance(payload, dict) and "content" in payload:
        items = [payload]
    elif isinstance(payload, dict):
        items = [
            {"id": document_id, "content": content} for document_id, content in payload.items()
        ]
    else:
        items = payload
    if not isinstance(items, list) or len(items) > config.max_documents:
        raise DocumentSourceError(
            f"command document source {config.name!r} returned an invalid document collection"
        )

    documents: list[_MemoryDocument] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise DocumentSourceError(
                f"command document source {config.name!r} returned an invalid document"
            )
        document_id = item.get("id", item.get("path"))
        content = item.get("content")
        title = item.get("title", document_id)
        media_type = item.get("media_type", "text/plain")
        if (
            not isinstance(document_id, str)
            or not isinstance(content, str)
            or not isinstance(title, str)
            or not isinstance(media_type, str)
        ):
            raise DocumentSourceError(
                f"command document source {config.name!r} returned invalid document fields"
            )
        if document_id in seen:
            raise DocumentSourceError(
                f"command document source {config.name!r} returned duplicate document ids"
            )
        documents.append(
            _memory_document(
                config,
                document_id=document_id,
                content=content,
                title=title,
                media_type=media_type,
            )
        )
        seen.add(document_id)
    return tuple(documents)


def _memory_document(
    config: CommandDocumentSourceConfig,
    *,
    document_id: str,
    content: str,
    title: str,
    media_type: str,
) -> _MemoryDocument:
    try:
        _validate_virtual_id(document_id)
    except DocumentRequestError as exc:
        raise DocumentSourceError(
            f"command document source {config.name!r} returned an unsafe document id"
        ) from exc
    if len(content.encode("utf-8")) > config.max_document_bytes:
        raise DocumentSourceError(
            f"document {document_id!r} from source {config.name!r} exceeds its limit"
        )
    if not title or len(title) > 1_000 or not media_type or len(media_type) > 200:
        raise DocumentSourceError(
            f"document {document_id!r} from source {config.name!r} has invalid metadata"
        )
    return _MemoryDocument(document_id, title, media_type, content)


def _validate_virtual_id(document_id: str) -> None:
    path = PurePosixPath(document_id)
    if (
        not document_id
        or document_id.startswith(("/", "~"))
        or "\\" in document_id
        or ".." in path.parts
        or "\x00" in document_id
        or len(document_id) > 1_000
    ):
        raise DocumentRequestError("document_id must be a safe virtual relative path")


__all__ = [
    "CommandDocumentSource",
    "DirectoryDocumentSource",
    "DocumentSourceRegistry",
    "build_document_source_registry",
]

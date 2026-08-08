"""Provider-neutral document catalog contracts.

Document sources stay outside the agent workspace.  Runtime adapters expose
this catalog through read-only tools, so a directory, CLI, or future API source
does not need to be copied into the staged filesystem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

DOCUMENT_TOOL_NAMES = ("DocumentList", "DocumentRead", "DocumentSearch")


class DocumentError(RuntimeError):
    """Base class for safe document-source failures."""


class DocumentRequestError(DocumentError):
    """The caller or model supplied an invalid document request."""


class SinglePassDocumentTooLargeError(DocumentRequestError):
    """A single-pass document exceeded the effective server byte limit.

    Carries only public virtual identifiers and the server-owned effective
    limit so the service layer can build safe, typed error details. The host
    absolute guardrail is intentionally not known here: the document catalog
    only enforces the effective admission limit and never depends on host
    deployment configuration.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str,
        document_id: str,
        observed_utf8_bytes: int,
        effective_limit_bytes: int,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.document_id = document_id
        self.observed_utf8_bytes = observed_utf8_bytes
        self.effective_limit_bytes = effective_limit_bytes


class DocumentSourceError(DocumentError):
    """A configured document source could not produce content."""


@dataclass(frozen=True)
class DocumentInfo:
    """Metadata for one virtual document."""

    source: str
    document_id: str
    title: str
    media_type: str
    size_bytes: int
    provenance: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "document_id": self.document_id,
            "title": self.title,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class DocumentPage:
    """A bounded line range read from one virtual document."""

    document: DocumentInfo
    start_line: int
    end_line: int
    total_lines: int
    content: str

    def as_dict(self) -> dict[str, object]:
        return {
            "document": self.document.as_dict(),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "total_lines": self.total_lines,
            "content": self.content,
        }


@dataclass(frozen=True)
class DocumentSearchHit:
    """One line-level search result."""

    source: str
    document_id: str
    line: int
    text: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "document_id": self.document_id,
            "line": self.line,
            "text": self.text,
        }


@runtime_checkable
class PreparedDocumentSource(Protocol):
    """One task-scoped, read-only document source."""

    name: str

    async def list_documents(self, *, pattern: str | None, limit: int) -> tuple[DocumentInfo, ...]:
        """List bounded metadata without returning document bodies."""

        ...

    async def read_document(
        self, document_id: str, *, start_line: int, max_lines: int
    ) -> DocumentPage:
        """Read a bounded line range from one document."""

        ...

    async def search_documents(
        self, query: str, *, pattern: str | None, limit: int
    ) -> tuple[DocumentSearchHit, ...]:
        """Search this source and return bounded line-level hits."""

        ...


@runtime_checkable
class DocumentSource(Protocol):
    """Factory for a fresh task-scoped document source."""

    name: str

    async def prepare(
        self,
        *,
        query: str,
        parameters: Mapping[str, str] | None = None,
    ) -> PreparedDocumentSource:
        """Resolve the source for one stateless research task."""

        ...


@runtime_checkable
class DocumentSourceResolver(Protocol):
    """Resolve caller-selected, server-configured sources for one task."""

    @property
    def available_names(self) -> tuple[str, ...]:
        """Return the configured public source identifiers."""

        ...

    @property
    def public_catalog(self) -> Mapping[str, object]:
        """Return redacted names, aliases, descriptions, and parameter schemas."""

        ...

    async def open(
        self,
        names: tuple[str, ...],
        *,
        query: str,
        parameters: Mapping[str, Mapping[str, str]] | None = None,
    ) -> DocumentCatalog:
        """Prepare a fresh catalog containing exactly the selected sources."""

        ...


class DocumentCatalog:
    """A unified view over selected named document sources."""

    def __init__(self, sources: Mapping[str, PreparedDocumentSource]) -> None:
        if not sources:
            raise ValueError("a document catalog requires at least one source")
        self._sources = dict(sources)

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(self._sources)

    async def list_documents(
        self,
        *,
        source: str | None = None,
        pattern: str | None = None,
        limit: int = 100,
    ) -> tuple[DocumentInfo, ...]:
        selected = self._select(source)
        effective_limit = self._limit(limit, maximum=500)
        normalized_pattern = self._optional_text(pattern, field="pattern", maximum=500)
        results: list[DocumentInfo] = []
        for item in selected:
            remaining = effective_limit - len(results)
            if remaining <= 0:
                break
            results.extend(await item.list_documents(pattern=normalized_pattern, limit=remaining))
        return tuple(results)

    async def read_single_document(self, *, max_bytes: int) -> tuple[DocumentInfo, str]:
        """Return the only selected document in full within a server-owned byte limit."""

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        documents = await self.list_documents(limit=2)
        if len(documents) != 1:
            raise DocumentRequestError(
                "single_pass document mode requires exactly one selected document"
            )
        document = documents[0]
        if document.size_bytes > max_bytes:
            raise SinglePassDocumentTooLargeError(
                "The selected single-pass document exceeds the configured byte limit.",
                source=document.source,
                document_id=document.document_id,
                observed_utf8_bytes=document.size_bytes,
                effective_limit_bytes=max_bytes,
            )
        pages: list[str] = []
        start_line = 1
        while True:
            page = await self.read_document(
                source=document.source,
                document_id=document.document_id,
                start_line=start_line,
                max_lines=500,
            )
            pages.append(page.content)
            if page.end_line >= page.total_lines:
                break
            start_line = page.end_line + 1
        content = "\n".join(pages)
        observed = len(content.encode("utf-8"))
        if observed > max_bytes:
            raise SinglePassDocumentTooLargeError(
                "The selected single-pass document exceeds the configured byte limit.",
                source=document.source,
                document_id=document.document_id,
                observed_utf8_bytes=observed,
                effective_limit_bytes=max_bytes,
            )
        return document, content

    async def read_document(
        self,
        *,
        source: str,
        document_id: str,
        start_line: int = 1,
        max_lines: int = 200,
    ) -> DocumentPage:
        selected = self._select(self._text(source, field="source", maximum=64))
        identifier = self._text(document_id, field="document_id", maximum=1_000)
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
            raise DocumentRequestError("start_line must be a positive integer")
        effective_lines = self._limit(max_lines, maximum=500)
        return await selected[0].read_document(
            identifier,
            start_line=start_line,
            max_lines=effective_lines,
        )

    async def search_documents(
        self,
        *,
        query: str,
        source: str | None = None,
        pattern: str | None = None,
        limit: int = 50,
    ) -> tuple[DocumentSearchHit, ...]:
        selected = self._select(source)
        normalized_query = self._text(query, field="query", maximum=1_000)
        normalized_pattern = self._optional_text(pattern, field="pattern", maximum=500)
        effective_limit = self._limit(limit, maximum=200)
        results: list[DocumentSearchHit] = []
        for item in selected:
            remaining = effective_limit - len(results)
            if remaining <= 0:
                break
            results.extend(
                await item.search_documents(
                    normalized_query,
                    pattern=normalized_pattern,
                    limit=remaining,
                )
            )
        return tuple(results)

    def _select(self, source: str | None) -> tuple[PreparedDocumentSource, ...]:
        if source is None or not source.strip():
            return tuple(self._sources.values())
        normalized = self._text(source, field="source", maximum=64)
        try:
            return (self._sources[normalized],)
        except KeyError as exc:
            raise DocumentRequestError(f"document source {normalized!r} is not selected") from exc

    @staticmethod
    def _text(value: str, *, field: str, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise DocumentRequestError(f"{field} must not be empty")
        normalized = value.strip()
        if len(normalized) > maximum:
            raise DocumentRequestError(f"{field} exceeds the {maximum}-character limit")
        return normalized

    @classmethod
    def _optional_text(cls, value: str | None, *, field: str, maximum: int) -> str | None:
        if value is None or not value.strip():
            return None
        return cls._text(value, field=field, maximum=maximum)

    @staticmethod
    def _limit(value: int, *, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DocumentRequestError("limit must be a positive integer")
        return min(value, maximum)


__all__ = [
    "DOCUMENT_TOOL_NAMES",
    "DocumentCatalog",
    "DocumentError",
    "DocumentInfo",
    "DocumentPage",
    "DocumentRequestError",
    "DocumentSearchHit",
    "DocumentSource",
    "DocumentSourceError",
    "DocumentSourceResolver",
    "PreparedDocumentSource",
    "SinglePassDocumentTooLargeError",
]

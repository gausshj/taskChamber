"""Application composition for configured runtime and service instances."""

from .composition import (
    create_default_service,
    create_runtime_from_configuration,
    create_runtime_from_environment,
    create_runtime_registry,
)
from .documents import (
    CommandDocumentSource,
    DirectoryDocumentSource,
    DocumentSourceRegistry,
    build_document_source_registry,
)

__all__ = [
    "CommandDocumentSource",
    "DirectoryDocumentSource",
    "DocumentSourceRegistry",
    "build_document_source_registry",
    "create_default_service",
    "create_runtime_from_configuration",
    "create_runtime_from_environment",
    "create_runtime_registry",
]

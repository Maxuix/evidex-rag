"""Native Docling parser runtime."""

from rag_kb.adapters.parser.docling.artifacts import (
    ArtifactManifestError,
    DoclingArtifactManifest,
    verify_docling_artifacts,
)
from rag_kb.adapters.parser.docling.factory import (
    ALLOWED_FORMATS,
    build_docling_converter,
)
from rag_kb.adapters.parser.docling.parser import (
    DOCLING_DOCUMENT_VERSION,
    DoclingParser,
)

__all__ = [
    "ALLOWED_FORMATS",
    "ArtifactManifestError",
    "DoclingArtifactManifest",
    "DOCLING_DOCUMENT_VERSION",
    "DoclingParser",
    "build_docling_converter",
    "verify_docling_artifacts",
]

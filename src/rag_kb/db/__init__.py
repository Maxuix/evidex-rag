"""Persistence models, migrations, and read-only database validation."""

from rag_kb.db.compatibility import (
    DatabaseCompatibility,
    DatabaseCompatibilityError,
    validate_database_compatibility,
)
from rag_kb.db.models import Base
from rag_kb.db.readiness import RuntimeReadiness, validate_runtime_readiness
from rag_kb.db.session import (
    DatabaseProcess,
    DatabaseResources,
    create_database_resources,
)

__all__ = [
    "Base",
    "DatabaseCompatibility",
    "DatabaseCompatibilityError",
    "DatabaseProcess",
    "DatabaseResources",
    "RuntimeReadiness",
    "create_database_resources",
    "validate_database_compatibility",
    "validate_runtime_readiness",
]

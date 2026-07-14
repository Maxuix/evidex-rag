"""Persistence models, migrations, and read-only database validation."""

from rag_kb.db.compatibility import (
    DatabaseCompatibility,
    DatabaseCompatibilityError,
    validate_database_compatibility,
)
from rag_kb.db.models import Base

__all__ = [
    "Base",
    "DatabaseCompatibility",
    "DatabaseCompatibilityError",
    "validate_database_compatibility",
]

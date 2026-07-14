"""Typed configuration and deployment-profile boundary."""

from rag_kb.config.profiles import DeploymentProfile
from rag_kb.config.settings import Settings, load_settings
from rag_kb.config.validation import (
    StartupConfigurationError,
    StartupValidation,
    validate_startup_environment,
)

__all__ = [
    "DeploymentProfile",
    "Settings",
    "StartupConfigurationError",
    "StartupValidation",
    "load_settings",
    "validate_startup_environment",
]

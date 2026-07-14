"""Deployment profiles enabled by the current delivery boundary."""

from enum import StrEnum


class DeploymentProfile(StrEnum):
    """Known profiles; only development is currently executable."""

    DEVELOPMENT = "development"
    DEPARTMENT = "department"

"""Provider-neutral contracts for publishable intermediate artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ARTIFACT_SCHEMA_VERSION: Literal["aspr_artifact"] = "aspr_artifact"


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for a release manifest."""
    return datetime.now(timezone.utc)


class ArtifactModel(BaseModel):
    """Strict base model used by all exchange-layer documents."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal["aspr_artifact"] = ARTIFACT_SCHEMA_VERSION


class ArtifactFile(ArtifactModel):
    """One immutable file within a release."""

    path: str
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def relative_path_only(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact file paths must be non-empty and relative")
        return path.as_posix()


class ArtifactReference(ArtifactModel):
    """Pinned dependency on another release, never a mutable directory path."""

    producer: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    artifact: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    release: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ArtifactManifest(ArtifactModel):
    """Self-contained manifest for one publish-once artifact release."""

    contract: Literal["aspr_artifact_manifest"] = "aspr_artifact_manifest"
    producer: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    artifact: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    release: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    created_at: datetime = Field(default_factory=utc_now)
    files: list[ArtifactFile] = Field(min_length=1)
    dependencies: list[ArtifactReference] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("files")
    @classmethod
    def unique_file_paths(cls, value: list[ArtifactFile]) -> list[ArtifactFile]:
        paths = [file.path for file in value]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact file paths must be unique")
        return value

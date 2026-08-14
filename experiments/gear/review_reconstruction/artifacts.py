"""Explicit publication boundary for reconstructed human reference reviews."""

from __future__ import annotations

from pathlib import Path

from artifact_store import ArtifactReference, ArtifactStore
from artifact_store.catalog import validate_dependency


def publish_reference_dataset(
    *,
    store: ArtifactStore,
    release: str,
    dataset_dir: Path,
    source_dataset: ArtifactReference,
) -> ArtifactReference:
    """Publish sealed one-pass reconstruction records as a reference dataset."""
    validate_dependency("review_reconstruction", source_dataset.producer)
    manifest = Path(dataset_dir) / "batch_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"reconstruction batch manifest is missing: {manifest}")
    return store.publish_directory(
        producer="review_reconstruction",
        artifact="reference_reviews",
        release=release,
        source=dataset_dir,
        dependencies=[source_dataset],
        metadata={"contract": "reconstruction_session_response"},
    )


__all__ = ["publish_reference_dataset"]

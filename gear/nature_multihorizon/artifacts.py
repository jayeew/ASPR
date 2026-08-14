"""Publication boundary for reusable corpus and calibration releases."""

from __future__ import annotations

from pathlib import Path

from artifact_store import ArtifactReference, ArtifactStore


def publish_dataset_release(
    *,
    store: ArtifactStore,
    release: str,
    dataset_dir: Path,
) -> ArtifactReference:
    """Publish a sealed dataset or calibration release for downstream modules."""
    if not Path(dataset_dir).is_dir():
        raise NotADirectoryError(dataset_dir)
    return store.publish_directory(
        producer="datasets",
        artifact="dataset_release",
        release=release,
        source=dataset_dir,
        metadata={"contract": "dataset_release"},
    )


__all__ = ["publish_dataset_release"]

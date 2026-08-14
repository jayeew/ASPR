"""Explicit publication boundary for completed GEAR agent-review runs."""

from __future__ import annotations

from pathlib import Path

from artifact_store import ArtifactReference, ArtifactStore
from artifact_store.catalog import validate_dependency


def publish_review_run(
    *,
    store: ArtifactStore,
    release: str,
    run_dir: Path,
    calibration: ArtifactReference,
) -> ArtifactReference:
    """Publish a verified agent-review run without exposing runtime internals."""
    validate_dependency("gear_agent", calibration.producer)
    required = {"review_bundle.json", "review.json", "run_manifest.json"}
    missing = sorted(name for name in required if not (Path(run_dir) / name).is_file())
    if missing:
        raise FileNotFoundError(f"review run is incomplete: {missing}")
    return store.publish_directory(
        producer="gear_agent",
        artifact="review_run",
        release=release,
        source=run_dir,
        dependencies=[calibration],
        metadata={"contract": "aspr_gear_review_bundle"},
    )


__all__ = ["publish_review_run"]

"""Explicit publication boundary for the current Fig.1–Fig.10 suite."""

from __future__ import annotations

from pathlib import Path

from artifact_store import ArtifactReference, ArtifactStore
from artifact_store.catalog import validate_dependency


def publish_figure_result(
    *,
    store: ArtifactStore,
    release: str,
    figure_dir: Path,
    dataset: ArtifactReference,
) -> ArtifactReference:
    """Publish one audited figure directory using only a pinned data release."""
    validate_dependency("figures", dataset.producer)
    required = {"run_manifest.json", "output_inventory.json"}
    missing = sorted(
        name for name in required if not (Path(figure_dir) / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"figure result is incomplete: {missing}")
    return store.publish_directory(
        producer="figures",
        artifact="figure_result",
        release=release,
        source=figure_dir,
        dependencies=[dataset],
        metadata={"contract": "figure_run_manifest"},
    )


__all__ = ["publish_figure_result"]

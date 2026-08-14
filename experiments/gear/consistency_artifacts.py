"""Publication boundary for agent-versus-human consistency evaluation."""

from __future__ import annotations

from pathlib import Path

from artifact_store import ArtifactReference, ArtifactStore
from artifact_store.catalog import validate_dependency


def publish_consistency_evaluation(
    *,
    store: ArtifactStore,
    release: str,
    results_dir: Path,
    reference_reviews: ArtifactReference,
    agent_reviews: ArtifactReference,
) -> ArtifactReference:
    """Publish blinded comparison metrics with both inputs pinned by manifest hash."""
    validate_dependency("consistency_evaluation", reference_reviews.producer)
    validate_dependency("consistency_evaluation", agent_reviews.producer)
    required = {"summary.json", "corpus_metrics.json", "sample_metrics.jsonl"}
    missing = sorted(
        name for name in required if not (Path(results_dir) / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"consistency result is incomplete: {missing}")
    return store.publish_directory(
        producer="consistency_evaluation",
        artifact="agent_human_agreement",
        release=release,
        source=results_dir,
        dependencies=[reference_reviews, agent_reviews],
        metadata={"contract": "structured_review_corpus_metrics"},
    )


__all__ = ["publish_consistency_evaluation"]

"""Offline consistency checks for v2 saved runs."""

from __future__ import annotations

from pathlib import Path

from gear.artifacts import read_model
from gear.trace import EvidenceStore, sha256_value

from .contracts import AnalysisResult, ClaimSet


def validate_run(root: Path) -> dict[str, object]:
    shared = read_model(root / "shared" / "claims.json", ClaimSet)
    expected = {x.claim_id: x.normalized_claim_text for x in shared.claims}
    errors: list[str] = []
    checked = 0
    for mode in ("gear", "graph", "fusion"):
        path = root / mode / "analysis.json"
        if not path.exists():
            errors.append(f"{mode}:missing_result")
            continue
        result = read_model(path, AnalysisResult)
        if (
            result.claim_fingerprint != sha256_value(shared)
            or result.paper_id != shared.paper_id
        ):
            errors.append(f"{mode}:identity_mismatch")
        ids = [x.claim_id for x in result.assessments]
        if len(ids) != len(set(ids)):
            errors.append(f"{mode}:duplicate_claims")
        for row in result.assessments:
            checked += 1
            if expected.get(row.claim_id) != row.claim_text:
                errors.append(f"{mode}:{row.claim_id}:text_mismatch")
            store = EvidenceStore(root / mode / row.claim_id.rsplit("::", 1)[-1])
            known = set(store._evidence)
            for finding in row.findings:
                if not finding.evidence_keys or not set(finding.evidence_keys).issubset(
                    known
                ):
                    errors.append(f"{mode}:{row.claim_id}:unbound_evidence")
        if set(expected) != set(ids):
            errors.append(f"{mode}:incomplete_claim_coverage")
    return {"valid": not errors, "checked_assessments": checked, "errors": errors}


def validate_assets(root: Path) -> dict[str, object]:
    import sqlite3

    import numpy as np

    required = [
        "claim_graph_index.sqlite",
        "paper_graph_index.sqlite",
        "claim_graph_runtime_statistics.sqlite",
        "claim_nodes.parquet",
        "claim_embedding_matrix.npy",
        "community_centroid_matrix.npy",
        "community_centroid_index.parquet",
        "canonical_target_works.parquet",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {"valid": False, "missing": missing}
    try:
        with sqlite3.connect(
            f"file:{(root/required[0]).resolve()}?mode=ro&immutable=1", uri=True
        ) as connection:
            count, maximum = connection.execute(
                "SELECT COUNT(*),MAX(claim_row) FROM claim_nodes"
            ).fetchone()
        matrix = np.load(root / "claim_embedding_matrix.npy", mmap_mode="r")
        valid = matrix.ndim == 2 and matrix.shape[0] == count and maximum < count
        return {
            "valid": valid,
            "claim_count": count,
            "embedding_shape": list(matrix.shape),
        }
    except (OSError, ValueError, sqlite3.Error) as exc:
        return {"valid": False, "error": str(exc)}

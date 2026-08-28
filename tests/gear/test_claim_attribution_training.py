from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.gear.evaluation.train_claim_attribution_head import (
    train_and_gate_claim_attribution,
)
from gear.claim_attribution import FEATURE_SCHEMA_VERSION, T0_FEATURE_NAMES


def _frame(holdout_offset: float = 0.0) -> pd.DataFrame:
    rows = []
    splits = ["development"] * 24 + ["domain_holdout"] * 2 + ["temporal_holdout"] * 2
    for paper_index, split in enumerate(splits):
        for claim_index, centrality in enumerate((0.2, 0.6, 1.0)):
            features = dict.fromkeys(T0_FEATURE_NAMES, 0.0)
            features["claim_centrality"] = centrality
            features["claim_type__method_claim"] = 1.0
            features["anatomy_role__substantive_innovation"] = 1.0
            features["pathway__local_method_adoption"] = 1.0
            rows.append(
                {
                    "paper_id": f"W{paper_index:03d}",
                    "claim_id": f"C{claim_index}",
                    "outer_fold_id": paper_index // 8,
                    "domain12": f"domain-{paper_index % 3}",
                    "publication_year": 2000 + paper_index // 6,
                    "integration_split": split,
                    "future_adoption": centrality
                    + (holdout_offset if split != "development" else 0.0),
                    "claim_t0_schema_version": FEATURE_SCHEMA_VERSION,
                    **features,
                }
            )
    return pd.DataFrame(rows)


def test_training_is_development_only_and_exactly_replayable(tmp_path: Path) -> None:
    frame = _frame()
    temporal = tmp_path / "temporal.parquet"
    domain = tmp_path / "domain.parquet"
    frame.to_parquet(temporal, index=False)
    frame.to_parquet(domain, index=False)
    counts = {"development": 24, "domain_holdout": 2, "temporal_holdout": 2}
    first = train_and_gate_claim_attribution(
        temporal,
        domain,
        tmp_path / "first",
        release_id="candidate-1",
        bootstrap_replicates=100,
        expected_split_counts=counts,
    )
    changed = _frame(1000.0)
    changed.to_parquet(temporal, index=False)
    changed.to_parquet(domain, index=False)
    second = train_and_gate_claim_attribution(
        temporal,
        domain,
        tmp_path / "second",
        release_id="candidate-2",
        bootstrap_replicates=100,
        expected_split_counts=counts,
    )

    assert first["status"] == "promoted"
    assert second["status"] == "promoted"
    assert (tmp_path / "first" / "claim_attribution_linear_head.json").read_bytes() == (
        tmp_path / "second" / "claim_attribution_linear_head.json"
    ).read_bytes()


def test_candidate_fails_closed_on_old_mixed_schema(tmp_path: Path) -> None:
    frame = _frame().drop(columns=["claim_type__method_claim"])
    temporal = tmp_path / "temporal.parquet"
    domain = tmp_path / "domain.parquet"
    frame.to_parquet(temporal, index=False)
    frame.to_parquet(domain, index=False)
    with pytest.raises(ValueError, match="exact T0 columns missing"):
        train_and_gate_claim_attribution(
            temporal,
            domain,
            tmp_path / "out",
            release_id="blocked",
            bootstrap_replicates=10,
            expected_split_counts={
                "development": 24,
                "domain_holdout": 2,
                "temporal_holdout": 2,
            },
        )
    assert not (tmp_path / "out" / "release.json").exists()

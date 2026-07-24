from __future__ import annotations

from pathlib import Path

import pandas as pd

from aspr.nature_multihorizon.sealed_v6 import (
    SEALED_OUTCOME_COLUMNS,
    assemble_locked_sealed_features,
)


def test_sealed_feature_lock_reads_no_outcome_columns(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "horizon": 5,
                "publication_year": 2014,
                "domain12": "physics",
                "venue_family": "nature",
                "cohort_member": 1,
                "future_uptake": 1,
                "conditional_diffusion_member": 1,
            },
            {
                "paper_id": "P2",
                "horizon": 5,
                "publication_year": 2014,
                "domain12": "physics",
                "venue_family": "nature",
                "cohort_member": 0,
                "future_uptake": 0,
                "conditional_diffusion_member": 0,
            },
        ]
    ).to_parquet(tmp_path / "cohort_membership.parquet", index=False)
    pd.DataFrame(
        [{"paper_id": "P1", "field_variety": 2.0}]
    ).to_parquet(tmp_path / "innovation_features.parquet", index=False)
    pd.DataFrame(
        [{"paper_id": "P1", "log_reference_count": 3.0}]
    ).to_parquet(tmp_path / "control_features.parquet", index=False)
    pd.DataFrame(
        [{"paper_id": "P1", "bc_component_share_t0": 0.2}]
    ).to_parquet(tmp_path / "opportunity_features.parquet", index=False)

    locked = assemble_locked_sealed_features(
        tmp_path,
        horizon=5,
        start_year=2014,
        end_year=2017,
    )

    assert locked["paper_id"].tolist() == ["P1"]
    assert not (set(locked.columns) & SEALED_OUTCOME_COLUMNS)

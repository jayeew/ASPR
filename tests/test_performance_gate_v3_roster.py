from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.performance_gate_v3_roster import extract_run_metrics, strictly_improves


def _write_run(root: Path, *, learned: float, equal: float, latest: float, deltas: int) -> None:
    root.mkdir(parents=True)
    (root / "fig3_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "learned_oof_spearman": learned,
                "equal_weight_oof_spearman": equal,
                "n_contributing_graph_deltas": deltas,
                "contributing_graph_deltas": [f"d{i}" for i in range(deltas)],
                "data_profile": {"domains": ["crispr", "magnetic_properties_of_thin_films"]},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"fold": 1, "test_spearman": 0.1},
            {"fold": 2, "test_spearman": latest},
        ]
    ).to_csv(root / "fig3_cv_summary.csv", index=False)


def test_strictly_improves_requires_oof_latest_fold_and_contributing_deltas(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_run(baseline_dir, learned=0.30, equal=0.10, latest=0.20, deltas=4)
    _write_run(candidate_dir, learned=0.31, equal=0.10, latest=0.21, deltas=4)

    baseline = extract_run_metrics(baseline_dir, corpus)
    candidate = extract_run_metrics(candidate_dir, corpus)

    checks = strictly_improves(candidate, baseline)

    assert checks == {
        "improves_oof": True,
        "improves_latest_fold": True,
        "improves_contributing_deltas": False,
    }

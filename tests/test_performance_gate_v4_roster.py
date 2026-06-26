from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts import performance_gate_v4_roster as mod


def _write_run(root: Path, *, learned: float, equal: float, latest: float, deltas: int, enrichment: float) -> None:
    root.mkdir(parents=True)
    domains = [f"domain_{idx}" for idx in range(8)]
    (root / "fig3_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "learned_oof_spearman": learned,
                "equal_weight_oof_spearman": equal,
                "learned_vs_equal_delta": learned - equal,
                "n_contributing_graph_deltas": deltas,
                "contributing_graph_deltas": [f"d{idx}" for idx in range(deltas)],
                "data_profile": {"domains": domains},
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
    (root / "fig3_effect_summary.json").write_text(
        json.dumps({"top_vs_bottom_score_decile_rgpm_top10_enrichment": enrichment}),
        encoding="utf-8",
    )


def _domain_screen() -> pd.DataFrame:
    families = [
        "biology_biomedicine",
        "biology_biomedicine",
        "materials_chemistry",
        "materials_chemistry",
        "physics_astronomy",
        "physics_astronomy",
        "methods_computing",
        "methods_computing",
    ]
    return pd.DataFrame(
        [
            {
                "domain": f"domain_{idx}",
                "display_name": f"Domain {idx}",
                "field_name": family,
                "family": family,
                "quality_pass": 1,
                "quality_score": 1.0,
                "failure_reasons": "",
            }
            for idx, family in enumerate(families)
        ]
    )


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "min_oof": 0.45,
        "min_learned_vs_equal": 0.03,
        "min_latest_fold": 0.35,
        "ideal_latest_fold": 0.40,
        "min_contributing_deltas": 5,
        "min_enrichment": 5.0,
        "max_family_share": 0.50,
        "min_domains": 8,
        "max_domains": 12,
        "baseline_run_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_final_gate_requires_oof_latest_deltas_enrichment_and_balance(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    weak = tmp_path / "weak"
    _write_run(weak, learned=0.44, equal=0.20, latest=0.50, deltas=5, enrichment=6.0)

    result = mod.evaluate_final_gate(weak, corpus, _domain_screen(), _args())

    assert result["final_pass"] is False
    assert result["checks"]["learned_oof_spearman"] is False
    assert result["checks"]["latest_fold"] is True
    assert result["checks"]["contributing_graph_deltas"] is True

    low_deltas = tmp_path / "low_deltas"
    _write_run(low_deltas, learned=0.46, equal=0.20, latest=0.50, deltas=4, enrichment=6.0)

    result = mod.evaluate_final_gate(low_deltas, corpus, _domain_screen(), _args())

    assert result["final_pass"] is False
    assert result["checks"]["learned_oof_spearman"] is True
    assert result["checks"]["contributing_graph_deltas"] is False


def test_failed_final_gate_does_not_write_target_roster(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    target_roster = tmp_path / "publication_target_domains.json"
    domain_screen = _domain_screen()
    final_eval = {"final_pass": False, "status": "pending_final_fig3_recompute"}
    args = argparse.Namespace(
        out_dir=out_dir,
        top_screen_domains=4,
        min_oof=0.45,
        min_learned_vs_equal=0.03,
        min_latest_fold=0.35,
        ideal_latest_fold=0.40,
        min_contributing_deltas=5,
        min_enrichment=5.0,
        max_family_share=0.50,
        min_domains=8,
        max_domains=12,
        registry_csv=tmp_path / "registry.csv",
        target_roster_path=target_roster,
    )
    pd.DataFrame().to_csv(args.registry_csv, index=False)

    decision = mod.write_outputs(domain_screen, final_eval, args)

    assert decision["final_pass"] is False
    assert (out_dir / "performance_gate_decision_v4.json").exists()
    assert (out_dir / "candidate_domains.csv").exists()
    assert not target_roster.exists()

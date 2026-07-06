from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig3.fig3_empirical_weight_learning import (  # noqa: E402
    fig3_existing_selected_outputs_reuse_ready,
)


def _args(out_dir: Path, n_weight_samples: int = 5000) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=Path("outputs/redraw_v6a_best_fig2/fig2_strong_input"),
        out_dir=out_dir,
        run_mode="multi_domain",
        panel="all",
        tau=8,
        cv_mode="time_block",
        delta_variant="matched_control_v3",
        min_refs=4,
        min_controls=50,
        max_papers=8000,
        n_weight_samples=n_weight_samples,
        skip_sensitivity=True,
        fig1_corpus_source="selected",
        formats=["png", "svg"],
        export_tables=True,
        audit_only=False,
    )


def _write_reusable_fixture(out_dir: Path, domains: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fig3_selected_weight_learning_full.png").write_bytes(b"not-a-real-png-but-present")
    (out_dir / "fig3_selected_weight_learning_full.svg").write_text("<svg></svg>", encoding="utf-8")
    (out_dir / "figure_quality_report.json").write_text(
        json.dumps(
            {
                "figure": "fig3",
                "overall_pass": True,
                "quality_gates": {
                    "overall_pass": True,
                    "status_label": "strong predictive evidence",
                    "checks": {
                        "learned_oof_spearman_ge_0_45": 1,
                        "top_decile_enrichment_ge_5x": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "figure": "fig3",
                "domains": domains,
                "inputs": {
                    "data_dir": "outputs/redraw_v6a_best_fig2/fig2_strong_input",
                    "run_mode": "multi_domain",
                    "panel": "all",
                    "tau": 8,
                    "cv_mode": "time_block",
                    "delta_variant": "matched_control_v3",
                    "min_refs": 4,
                    "min_controls": 50,
                    "fig1_corpus_source": "selected",
                    "run_name": "selected:multi_domain",
                },
                "command_argv": [
                    "fig3_empirical_weight_learning.py",
                    "--data-dir",
                    "outputs/redraw_v6a_best_fig2/fig2_strong_input",
                    "--run-mode",
                    "multi_domain",
                    "--panel",
                    "all",
                    "--tau",
                    "8",
                    "--min-refs",
                    "4",
                    "--min-controls",
                    "50",
                    "--max-papers",
                    "8000",
                    "--n-weight-samples",
                    "5000",
                    "--skip-sensitivity",
                    "--formats",
                    "png",
                    "svg",
                ],
            }
        ),
        encoding="utf-8",
    )


class Fig3ReuseContractTests(unittest.TestCase):
    def test_reuse_valid_current_accepts_matching_strong_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            domains = ["crispr", "graphene"]
            _write_reusable_fixture(out_dir, domains)

            decision = fig3_existing_selected_outputs_reuse_ready(_args(out_dir), domains)

            self.assertTrue(decision["reusable"], decision)
            self.assertEqual("reuse_valid_current", decision["status"])

    def test_reuse_valid_current_rejects_mismatched_weight_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            domains = ["crispr", "graphene"]
            _write_reusable_fixture(out_dir, domains)

            decision = fig3_existing_selected_outputs_reuse_ready(
                _args(out_dir, n_weight_samples=7000),
                domains,
            )

            self.assertFalse(decision["reusable"])
            self.assertIn("n_weight_samples", decision["reason"])

    def test_reuse_valid_current_rejects_missing_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            domains = ["crispr", "graphene"]
            _write_reusable_fixture(out_dir, domains)
            (out_dir / "fig3_selected_weight_learning_full.svg").unlink()

            decision = fig3_existing_selected_outputs_reuse_ready(_args(out_dir), domains)

            self.assertFalse(decision["reusable"])
            self.assertIn("missing selected output", decision["reason"])


if __name__ == "__main__":
    unittest.main()

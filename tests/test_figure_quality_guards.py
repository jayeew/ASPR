from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import normalize_reference_closure_report, strict_main_figure_failed
from experiments.kg_perturbation_fig1.fig1_knowledge_perturbation_v3 import (
    dominant_parameter_table,
    dominant_parameter_trajectories,
)
from experiments.kg_perturbation_fig2.fig2_empirical_panels import build_quality_gates


def _toy_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rolling_start": [2000, 2005],
            "rolling_end": [2004, 2009],
            "B": [0.2, 0.8],
        }
    )


class FigureQualityGuardTests(unittest.TestCase):
    def test_manual_dominant_parameter_fails_in_main_mode(self) -> None:
        cfg = {
            "plot": {
                "dominant_parameters": [
                    {"key": "B", "label": "B", "source": "B", "values": [0.0, 1.0]},
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "manual_schematic"):
            dominant_parameter_trajectories(_toy_metrics(), cfg, allow_manual=False)

    def test_computed_dominant_parameter_table_records_provenance(self) -> None:
        cfg = {
            "plot": {
                "dominant_parameters": [
                    {"key": "B", "label": "B", "source": "B"},
                ]
            }
        }
        table = dominant_parameter_table(_toy_metrics(), cfg, allow_manual=False)
        self.assertEqual(set(table["provenance"]), {"computed"})
        self.assertEqual(set(table["source_column"]), {"B"})

    def test_raw_zero_reference_closure_is_not_measured_or_passed(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "domain": "crispr",
                    "raw_records": 0,
                    "coverage_materialized": 1.0,
                    "status": "audit_only_no_online_closure",
                }
            ]
        )
        normalized = normalize_reference_closure_report(report)
        row = normalized.iloc[0]
        self.assertEqual(row["coverage_status"], "not_measured")
        self.assertEqual(int(row["coverage_measured"]), 0)
        self.assertTrue(math.isnan(float(row["coverage_materialized"])))
        self.assertEqual(int(row["quality_gate_pass"]), 0)

    def test_strict_main_figure_detects_failed_quality_gate(self) -> None:
        self.assertTrue(strict_main_figure_failed({"overall_pass": False}))
        self.assertFalse(strict_main_figure_failed({"overall_pass": True}))

    def test_fig2_quality_gate_fails_unmeasured_closure_and_relaxed_controls(self) -> None:
        quality = build_quality_gates(
            paper_metrics=pd.DataFrame(
                {
                    "domain": ["a", "b", "c", "d"] * 2,
                    "paper_id": [f"p{i}" for i in range(8)],
                }
            ),
            graph_delta_diagnostics_df=pd.DataFrame(
                {
                    "delta": ["future_bridge", "future_reach", "future_entropy", "future_path", "future_mix"],
                    "active": [1, 1, 1, 1, 1],
                }
            ),
            reference_closure_report=pd.DataFrame(
                [
                    {
                        "domain": "a",
                        "raw_records": 0,
                        "coverage_materialized": 1.0,
                        "status": "audit_only_no_online_closure",
                    }
                ]
            ),
            matched_controls=pd.DataFrame({"control_tier": ["field_all_years"] * 8}),
            bootstrap=pd.DataFrame(columns=["metric", "future_outcome", "ci_low", "rho"]),
            composite_rho=0.25,
        )
        self.assertEqual(quality["checks"]["reference_closure_coverage_min80pct"], 0)
        self.assertEqual(quality["checks"]["relaxed_control_tier_ratio_max25pct"], 0)
        self.assertFalse(quality["overall_pass"])


if __name__ == "__main__":
    unittest.main()

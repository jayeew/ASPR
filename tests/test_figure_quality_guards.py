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
from experiments.fig01.old.fig1_knowledge_perturbation import (
    dominant_parameter_table,
    dominant_parameter_trajectories,
)
from experiments.fig02.old.fig2_empirical_panels import ComputedData, _panel_b_design_payload, build_quality_gates


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

    def test_panel_b_payload_centers_screening_gates_not_secondary_data_flow(self) -> None:
        comp = ComputedData(
            paper_metrics=pd.DataFrame(
                {
                    "paper_id": [f"p{i}" for i in range(8)],
                    "domain": ["a", "b"] * 4,
                }
            ),
            candidate_metrics=pd.DataFrame(),
            graph_deltas=pd.DataFrame(),
            redundancy_corr=pd.DataFrame(),
            indicator_delta_corr=pd.DataFrame(),
            percentile_long=pd.DataFrame(),
            landmark_summary=pd.DataFrame(),
            metric_standardization_diagnostics=pd.DataFrame(),
            graph_delta_diagnostics=pd.DataFrame(
                {
                    "delta": [
                        "community_reach",
                        "field_entropy",
                        "cross_community_adoption",
                        "path_shortening",
                        "partition_change",
                        "boundary_mixing",
                        "hub_formation",
                    ],
                    "active": [1, 1, 1, 1, 1, 1, 1],
                }
            ),
            input_audit=pd.DataFrame(
                {
                    "domain": ["a", "b"],
                    "raw_papers": [10, 20],
                    "citation_edges": [50, 70],
                }
            ),
            reference_closure_report=pd.DataFrame({"coverage_materialized": [1.0, 1.0]}),
            quality_gates={"significant_expected_links": 12},
            evidence_mode="strong",
        )

        payload = _panel_b_design_payload(comp)

        self.assertEqual([92, 67, 49, 29, 12, 7], list(payload["screening_counts"].values()))
        self.assertEqual(
            [
                "Future-impact signals",
                "Prestige/context signals",
                "Non-reference signals",
                "Generic graph controls",
                "Redundant variants",
            ],
            [item["category"] for item in payload["rejection_bins"]],
        )
        self.assertEqual(["B", "RS", "ΔQ0", "Uzzi-style", "RTD", "Burt IP", "PDE"], payload["final_basis"])
        self.assertNotIn("gate_rows", payload)
        self.assertNotIn("indicator_groups", payload)
        self.assertNotIn("audit_badges", payload)
        self.assertNotIn("data_flow", payload)


if __name__ == "__main__":
    unittest.main()

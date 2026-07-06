from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.kg_perturbation_fig2.fig2_empirical_panels import add_reference_closure_nodes, build_fig2_quality_gates
from experiments.kg_perturbation_fig2.build_fig2_reference_closure import build_reference_closure
from experiments.kg_perturbation_fig2.build_fig2_strong_inputs import (
    add_matching_columns,
    build_strong_inputs,
    compute_control_tier_count_maps,
)


class Fig2ReferenceClosureTests(unittest.TestCase):
    def test_fig2_strong_gates_require_eligible_controls_and_closure(self) -> None:
        report = build_fig2_quality_gates(
            n_domains=10,
            total_eligible_papers=8000,
            active_future_outcomes=["a", "b", "c", "d", "e"],
            relaxed_control_tier_ratio=0.24,
            reference_closure_measured_all_domains=True,
            min_reference_closure_coverage=0.81,
            significant_expected_links=4,
            mechanism_composite_partial_spearman=0.21,
        )

        self.assertTrue(report["overall_pass"])
        self.assertEqual(1, report["checks"]["total_eligible_papers_min8000"])
        self.assertEqual(1, report["checks"]["relaxed_control_tier_ratio_max25pct"])
        self.assertEqual(1, report["checks"]["reference_closure_coverage_min80pct"])

    def test_reference_closure_materializes_json_and_delimited_references(self) -> None:
        works = pd.DataFrame(
            [
                {
                    "id": "W1",
                    "paper_id": "W1",
                    "domain": "crispr",
                    "referenced_works": json.dumps(["R1", "R2"]),
                },
                {
                    "id": "W2",
                    "paper_id": "W2",
                    "domain": "crispr",
                    "referenced_works": "R2;R3",
                },
                {
                    "id": "W3",
                    "paper_id": "W3",
                    "domain": "graphene_2d_materials",
                    "referenced_works": "",
                },
            ]
        )

        closure, report = build_reference_closure(works)

        self.assertEqual(4, len(closure))
        self.assertEqual({"R1", "R2", "R3"}, set(closure["referenced_work_id"]))
        crispr_report = report[report["domain"] == "crispr"].iloc[0]
        self.assertEqual(4, int(crispr_report["referenced_works_count"]))
        self.assertEqual(1.0, float(crispr_report["coverage_materialized"]))
        self.assertEqual(1, int(crispr_report["coverage_measured"]))
        graphene_report = report[report["domain"] == "graphene_2d_materials"].iloc[0]
        self.assertEqual(0, int(graphene_report["quality_gate_pass"]))

    def test_build_strong_inputs_writes_cutoff_closure_and_control_audit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_fig2_strong_") as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            out_dir = Path(tmp) / "out"
            works = pd.DataFrame(
                [
                    {
                        "id": f"W{i}",
                        "title": f"Paper {i}",
                        "year": 2010 + (i % 3),
                        "domain": "crispr" if i < 6 else "graphene_2d_materials",
                        "primary_field": "Biology",
                        "document_type": "article",
                        "is_landmark": int(i in {0, 6}),
                        "reference_count": 10 + (i % 2),
                        "cited_by_count": 100 - i,
                        "source_dataset": "unit",
                        "referenced_works": json.dumps([f"R{i}", f"R{i+1}"]),
                    }
                    for i in range(24)
                ]
            )
            works.to_csv(source / "works.csv", index=False)
            pd.DataFrame({"source": ["W1", "W2"], "target": ["W0", "W1"]}).to_csv(source / "citations.csv", index=False)
            pd.DataFrame({"community": [1], "label": ["Topic"], "x": [0.0], "y": [1.0], "domain": ["crispr"], "topic_id": ["T1"]}).to_csv(
                source / "topics.csv",
                index=False,
            )
            pd.DataFrame({"source_community": [1], "target_community": [1], "weight": [1]}).to_csv(source / "topic_edges.csv", index=False)

            summary = build_strong_inputs(
                source=source,
                out_dir=out_dir,
                pre_cutoff_max_year=2018,
                future_window_start=2019,
                future_window_end=2025,
                min_total_eligible=10,
                min_controls=1,
            )

            self.assertEqual(24, summary["eligible_papers"])
            self.assertTrue((out_dir / "multi_domain" / "works.csv").exists())
            self.assertTrue((out_dir / "reference_closure_table.csv").exists())
            self.assertTrue((out_dir / "fig2_reference_closure_report.csv").exists())
            self.assertTrue((out_dir / "fig2_control_tier_audit.csv").exists())
            closure_report = pd.read_csv(out_dir / "fig2_reference_closure_report.csv")
            self.assertEqual(1, int(closure_report["quality_gate_pass"].min()))
            control_audit = pd.read_csv(out_dir / "fig2_control_tier_audit.csv")
            self.assertIn("control_tier", control_audit.columns)
            self.assertIn("relaxed_control_tier_ratio", summary)
            self.assertGreaterEqual(float(summary["relaxed_control_tier_ratio"]), 0.0)
            self.assertLessEqual(float(summary["relaxed_control_tier_ratio"]), 1.0)

    def test_control_tier_count_maps_exclude_landmarks_and_self(self) -> None:
        eligible = add_matching_columns(
            pd.DataFrame(
                [
                    {"id": "L1", "domain": "d", "year": 2018, "document_type": "article", "reference_count": 10, "source_dataset": "v", "is_landmark": 1},
                    {"id": "P1", "domain": "d", "year": 2018, "document_type": "article", "reference_count": 10, "source_dataset": "v", "is_landmark": 0},
                    {"id": "P2", "domain": "d", "year": 2018, "document_type": "article", "reference_count": 10, "source_dataset": "v", "is_landmark": 0},
                    {"id": "P3", "domain": "d", "year": 2018, "document_type": "article", "reference_count": 10, "source_dataset": "v", "is_landmark": 0},
                ]
            )
        )

        count_maps = compute_control_tier_count_maps(eligible)
        exact_key = ("d", 2018, "article", 0, "v")

        self.assertEqual(3, count_maps["exact"][exact_key])

    def test_reference_count_bins_are_configurable_for_control_audit(self) -> None:
        eligible = add_matching_columns(
            pd.DataFrame(
                [
                    {"id": f"P{i}", "domain": "d", "year": 2018, "document_type": "article", "reference_count": i + 1, "source_dataset": "v", "is_landmark": 0}
                    for i in range(12)
                ]
            ),
            reference_count_bins=4,
        )

        self.assertLessEqual(eligible["reference_count_decile"].nunique(), 4)

    def test_add_reference_closure_uses_cached_referenced_works_when_fig1_raw_missing(self) -> None:
        class Raw:
            pass

        raw = Raw()
        raw.works = pd.DataFrame(
            [
                {"id": "W1", "domain": "crispr", "referenced_works": json.dumps(["R1", "R2"])},
                {"id": "W2", "domain": "crispr", "referenced_works": "W1;R3"},
            ]
        )
        raw.citations = pd.DataFrame({"source": ["W2"], "target": ["W1"]})
        raw.topics = pd.DataFrame()
        raw.topic_edges = pd.DataFrame()

        with tempfile.TemporaryDirectory(prefix="aspr_fig2_closure_") as tmp:
            _, report = add_reference_closure_nodes(
                raw=raw,
                fig1_dir=Path(tmp),
                domain="crispr",
                reference_closure="auto",
                online_expand=False,
                closure_cap=50000,
                closure_coverage_target=0.80,
                openalex_api_key=None,
                openalex_api_keys=None,
                email=None,
                progress=False,
            )

        self.assertEqual("cached_referenced_works_materialized", report["status"])
        self.assertEqual(4, report["total_reference_mentions"])
        self.assertEqual(1, report["internal_reference_mentions"])
        self.assertEqual(3, report["materialized_closure_unique_references"])
        self.assertGreaterEqual(float(report["coverage_materialized"]), 0.8)


if __name__ == "__main__":
    unittest.main()

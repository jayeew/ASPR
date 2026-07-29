from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object used by the frozen package."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return value


def sha256_file(path: Path) -> str:
    """Hash one generated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RegistryTests(unittest.TestCase):
    """Validate construct, evidence, and feature-registry invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(ROOT / "protocol.json")
        cls.dimensions = read_json(ROOT / "dimensions.json")["dimensions"]
        cls.evidence = read_json(ROOT / "literature_evidence.json")["records"]
        cls.registry = read_json(ROOT / "feature_registry.json")
        cls.features = cls.registry["features"]
        cls.rules = read_json(ROOT / "screening_rules.json")

    def test_unique_identifiers_and_resolved_links(self) -> None:
        dimension_ids = [row["dimension_id"] for row in self.dimensions]
        source_ids = [row["source_id"] for row in self.evidence]
        feature_ids = [row["feature_id"] for row in self.features]
        self.assertEqual(len(dimension_ids), len(set(dimension_ids)))
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertEqual(len(feature_ids), len(set(feature_ids)))
        self.assertFalse(
            {
                feature["dimension_id"] for feature in self.features
            }
            - set(dimension_ids)
        )
        self.assertFalse(
            {
                source_id
                for feature in self.features
                for source_id in feature["source_ids"]
            }
            - set(source_ids)
        )

    def test_no_numeric_quota_or_outcome_selection(self) -> None:
        outcome = self.protocol["outcome_separation"]
        self.assertFalse(outcome["future_outcomes_used_to_define_features"])
        self.assertFalse(outcome["future_outcomes_used_to_select_features"])
        self.assertFalse(outcome["prediction_performance_used_to_select_features"])
        reproducibility = self.protocol["reproducibility"]
        self.assertTrue(
            reproducibility["selection_has_no_dimension_or_feature_quota"]
        )
        self.assertTrue(
            self.rules["family_deduplication"]["no_per_dimension_quota"]
        )
        self.assertNotIn("maximum_features", self.rules)
        self.assertNotIn("target_feature_count", self.rules)

    def test_evidence_includes_support_and_adverse_results(self) -> None:
        directions = {record["evidence_direction"] for record in self.evidence}
        self.assertIn("adverse", directions)
        self.assertIn("null_or_inconsistent", directions)
        self.assertIn("mixed", directions)
        self.assertGreaterEqual(len(self.evidence), 40)
        self.assertGreaterEqual(
            len({record["research_group"] for record in self.evidence}),
            35,
        )

    def test_candidate_census_is_broad(self) -> None:
        self.assertGreaterEqual(len(self.features), 60)
        represented_dimensions = {
            feature["dimension_id"] for feature in self.features
        }
        self.assertEqual(
            represented_dimensions,
            {dimension["dimension_id"] for dimension in self.dimensions},
        )
        blocks = {feature["block"] for feature in self.features}
        self.assertEqual(
            blocks,
            {
                "innovation_evidence",
                "substantive_potential",
                "opportunity_visibility",
                "context_control",
                "candidate_unready",
            },
        )


class OutputTests(unittest.TestCase):
    """Validate the generated selection and audit trail."""

    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "select_features.py")],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "audit_local_data.py")],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        cls.summary = read_json(OUTPUTS / "selection_summary.json")
        cls.final_dimensions = read_json(
            OUTPUTS / "final_dimensions.json"
        )["dimensions"]
        cls.final_features = read_json(
            OUTPUTS / "final_features.json"
        )["features"]
        cls.feature_sets = read_json(
            OUTPUTS / "training_feature_sets.json"
        )["sets"]

    def test_generated_counts_are_frozen(self) -> None:
        self.assertEqual(self.summary["candidate_feature_families"], 64)
        self.assertEqual(self.summary["literature_evidence_records"], 43)
        self.assertEqual(self.summary["selected_predictor_dimensions"], 8)
        self.assertEqual(self.summary["selected_context_dimensions"], 1)
        self.assertEqual(self.summary["selected_feature_families"], 26)
        self.assertEqual(self.summary["search_snapshot_records"], 1000)
        self.assertEqual(
            self.summary["evidence_sources_found_in_snapshot"],
            26,
        )
        self.assertEqual(
            self.summary[
                "evidence_sources_from_chaining_or_direct_verification"
            ],
            17,
        )

    def test_local_data_provenance_and_schema_pass(self) -> None:
        audit = read_json(OUTPUTS / "local_data_audit.json")
        self.assertTrue(audit["all_sources_pass"])
        self.assertEqual(audit["source_count"], 8)
        self.assertEqual(audit["missing_materialized_columns"], [])
        self.assertEqual(audit["primary_row_count_mismatches"], [])

    def test_selected_dimension_structure(self) -> None:
        selected = {
            row["dimension_id"]
            for row in self.final_dimensions
            if row["selected"]
        }
        self.assertEqual(
            selected,
            {
                "D01_RECOMBINATIONAL_NOVELTY",
                "D02_COMBINATION_PROFILE",
                "D03_KNOWLEDGE_DIVERSITY",
                "D04_PRIOR_KNOWLEDGE_SEARCH",
                "D05_TOPIC_MOMENTUM",
                "D06_TEAM_REACH",
                "D07_KNOWLEDGE_NETWORK_POSITION",
                "D08_PUBLICATION_VISIBILITY",
                "C01_FIELD_TIME_CONTEXT",
            },
        )
        self.assertNotIn("X01_OPEN_REPRODUCIBILITY", selected)
        self.assertNotIn("X02_METHOD_EVIDENCE_STRENGTH", selected)
        for row in self.final_dimensions:
            if row["selected"] and row["block"] != "context_control":
                self.assertGreaterEqual(row["independent_group_count"], 2)
                self.assertGreaterEqual(row["selected_feature_count"], 1)

    def test_innovation_core_remains_direct_and_small(self) -> None:
        self.assertEqual(
            self.feature_sets["innovation_core"],
            [
                "first_time_source_pair_share",
                "reference_overlap_novelty_t0",
            ],
        )
        by_name = {feature["name"]: feature for feature in self.final_features}
        for name in self.feature_sets["innovation_core"]:
            self.assertEqual(by_name[name]["block"], "innovation_evidence")

    def test_feature_sets_are_nested(self) -> None:
        default = set(self.feature_sets["full_t0_default"])
        extended = set(self.feature_sets["full_t0_extended"])
        sensitivity = set(self.feature_sets["full_t0_with_sensitivity"])
        self.assertTrue(default < extended)
        self.assertTrue(extended < sensitivity)
        self.assertEqual(len(default), 19)
        self.assertEqual(len(extended), 23)
        self.assertEqual(len(sensitivity), 26)

    def test_no_selected_future_or_forbidden_feature(self) -> None:
        defaults = read_json(ROOT / "feature_registry.json")[
            "candidate_defaults"
        ]
        raw_by_id = {
            feature["feature_id"]: feature
            for feature in read_json(ROOT / "feature_registry.json")[
                "features"
            ]
        }
        for selected in self.final_features:
            merged = dict(defaults)
            merged.update(raw_by_id[selected["feature_id"]])
            self.assertFalse(merged["requires_future"])
            self.assertTrue(merged["t0_computable"])
            self.assertNotIn(
                merged["bias_policy"],
                {
                    "forbidden_status",
                    "forbidden_protected",
                    "forbidden_citation_bias",
                },
            )
            self.assertFalse(merged["uses_outcome_for_selection"])

    def test_critical_exclusions_are_explicit(self) -> None:
        with (OUTPUTS / "feature_decisions.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            decisions = {
                row["feature_id"]: row for row in csv.DictReader(handle)
            }
        self.assertIn(
            "G06_NO_FUTURE_INFORMATION",
            decisions["F007_FUTURE_DISRUPTION_INDEX"]["failed_gates"],
        )
        self.assertIn(
            "G08_BIAS_GUARDRAIL",
            decisions["F050_CURRENT_JOURNAL_IMPACT_FACTOR"]["failed_gates"],
        )
        self.assertIn(
            "G08_BIAS_GUARDRAIL",
            decisions["F062_POSITIVE_RESULT_STATUS"]["failed_gates"],
        )
        self.assertIn(
            "G08_BIAS_GUARDRAIL",
            decisions["F064_PROTECTED_AUTHOR_ATTRIBUTE"]["failed_gates"],
        )
        self.assertEqual(
            decisions["F022_REFERENCE_AGE_MEDIAN"]["final_role"],
            "excluded_redundant",
        )

    def test_every_candidate_has_all_gate_results(self) -> None:
        gate_ids = set(read_json(ROOT / "screening_rules.json")["hard_gates"])
        with (OUTPUTS / "feature_decisions.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 64)
        for row in rows:
            for gate_id in gate_ids:
                self.assertIn(row[gate_id], {"True", "False"})
        roles = Counter(row["final_role"] for row in rows)
        self.assertEqual(sum(roles.values()), 64)

    def test_search_snapshot_has_complete_ranked_pairs(self) -> None:
        with (ROOT / "search_snapshot.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        pair_counts = Counter(
            (row["provider"], row["query_id"]) for row in rows
        )
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len(pair_counts), 40)
        self.assertEqual(set(pair_counts.values()), {25})
        errors = read_json(ROOT / "search_errors.json")["errors"]
        self.assertEqual(errors, [])

    def test_output_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory:
            with tempfile.TemporaryDirectory() as second_directory:
                first = Path(first_directory)
                second = Path(second_directory)
                for output in (first, second):
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "select_features.py"),
                            "--output-dir",
                            str(output),
                        ],
                        cwd=ROOT.parent,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                first_files = sorted(path.name for path in first.iterdir())
                second_files = sorted(path.name for path in second.iterdir())
                self.assertEqual(first_files, second_files)
                for name in first_files:
                    self.assertEqual(
                        sha256_file(first / name),
                        sha256_file(second / name),
                        name,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

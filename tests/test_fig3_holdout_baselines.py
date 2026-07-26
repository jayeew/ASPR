from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig03.old.fig3_empirical_weight_learning import (  # noqa: E402
    compute_leave_domain_out_validation,
    compute_nonlinear_challenger_diagnostics,
    compute_temporal_holdout_validation,
    update_nonlinear_challenger_diagnostics,
)


def _toy_score_table() -> pd.DataFrame:
    rows = []
    for domain_offset, domain in enumerate(["crispr", "graphene", "ipsc"]):
        for i, year in enumerate([2010, 2011, 2012, 2013, 2014, 2015]):
            signal = float(i + domain_offset * 0.1)
            rows.append(
                {
                    "paper_id": f"{domain}-{year}-{i}",
                    "domain": domain,
                    "year": year,
                    "B_z": signal,
                    "RS_z": 0.5 * signal,
                    "DeltaQ0_z": -0.1 * signal,
                    "RGPM": signal,
                    "reference_count": 10 + i,
                    "primary_field": "Biology" if domain != "ipsc" else "Medicine",
                    "fold_id": (i % 3) + 1,
                }
            )
    return pd.DataFrame(rows)


class Fig3HoldoutBaselineTests(unittest.TestCase):
    def test_temporal_holdout_records_no_leakage_validation_design(self) -> None:
        result = compute_temporal_holdout_validation(
            _toy_score_table(),
            active_metric_keys=["B", "RS", "DeltaQ0"],
            train_max_year=2012,
            validation_start_year=2013,
            validation_end_year=2015,
            seed=7,
        )

        self.assertEqual(1, len(result))
        row = result.iloc[0]
        self.assertEqual("temporal_holdout", row["validation_design"])
        self.assertEqual("publication_day_graph_indicators_only", row["feature_scope"])
        self.assertEqual("B_z;RS_z;DeltaQ0_z", row["feature_columns"])
        self.assertGreaterEqual(int(row["n_train"]), 9)
        self.assertGreaterEqual(int(row["n_test"]), 9)
        self.assertGreater(float(row["learned_spearman"]), 0.0)
        self.assertIn("equal_weight_spearman", result.columns)
        self.assertIn("best_single_spearman", result.columns)
        self.assertIn("bootstrap_ci_low", result.columns)
        self.assertIn("bootstrap_ci_high", result.columns)

    def test_leave_domain_out_records_one_holdout_row_per_domain(self) -> None:
        result = compute_leave_domain_out_validation(
            _toy_score_table(),
            active_metric_keys=["B", "RS", "DeltaQ0"],
            seed=11,
            min_train=6,
            min_test=4,
        )

        self.assertEqual({"crispr", "graphene", "ipsc"}, set(result["heldout_domain"]))
        self.assertTrue(result["validation_design"].eq("leave_domain_out").all())
        self.assertTrue(result["feature_scope"].eq("publication_day_graph_indicators_only").all())
        self.assertTrue((result["n_train"].astype(int) >= 12).all())
        self.assertTrue((result["n_test"].astype(int) >= 6).all())
        self.assertTrue((result["learned_spearman"].astype(float) > 0).all())
        self.assertTrue((result["learned_vs_equal_delta"].astype(float).abs() < 1.0).all())

    def test_nonlinear_challenger_records_random_temporal_and_leave_domain_designs(self) -> None:
        result = compute_nonlinear_challenger_diagnostics(
            _toy_score_table(),
            active_metric_keys=["B", "RS", "DeltaQ0"],
            seed=13,
            random_folds=3,
            train_max_year=2012,
            validation_start_year=2013,
            validation_end_year=2015,
            min_train=6,
            min_test=4,
        )

        designs = set(result["validation_design"])
        self.assertIn("random_kfold", designs)
        self.assertIn("existing_fold_id", designs)
        self.assertIn("temporal_holdout", designs)
        self.assertIn("leave_domain_out", designs)
        self.assertTrue(result["feature_scope"].str.contains("publication_day").all())
        self.assertFalse(result["feature_columns"].str.contains("cited_by_count").any())
        self.assertIn("no_leakage_feature_contract", result.columns)

    def test_nonlinear_challenger_can_be_promoted_as_primary_when_all_strong_gates_pass(self) -> None:
        summary = {
            "overall_pass": False,
            "status_label": "underpowered multi-domain diagnostic run",
            "checks": {
                "learned_oof_spearman_ge_0_45": 0,
                "top_decile_enrichment_ge_5x": 0,
                "temporal_holdout_positive_ci": 1,
                "leave_domain_out_positive_ci": 0,
            },
            "data_checks": {"papers_per_domain": 1, "landmark_or_high_cases_per_domain": 1},
            "learned_oof_spearman": 0.31,
            "equal_weight_oof_spearman": 0.20,
            "best_single_oof_spearman": 0.25,
        }
        challenger = pd.DataFrame(
            [
                {
                    "model": "metadata_hgb",
                    "validation_design": "random_kfold_summary",
                    "spearman": 0.46,
                    "bootstrap_ci_low": 0.41,
                    "bootstrap_ci_high": 0.50,
                    "top_decile_future_top20_enrichment": 5.4,
                    "no_leakage_feature_contract": "excludes_cited_by_count_and_future_outcome_columns",
                },
                {
                    "model": "metadata_hgb",
                    "validation_design": "temporal_holdout",
                    "spearman": 0.33,
                    "bootstrap_ci_low": 0.08,
                    "bootstrap_ci_high": 0.44,
                    "top_decile_future_top20_enrichment": 5.1,
                    "no_leakage_feature_contract": "excludes_cited_by_count_and_future_outcome_columns",
                },
                {
                    "model": "metadata_hgb",
                    "validation_design": "leave_domain_out",
                    "heldout_domain": "crispr",
                    "spearman": 0.20,
                    "bootstrap_ci_low": 0.03,
                    "bootstrap_ci_high": 0.31,
                    "top_decile_future_top20_enrichment": 3.0,
                    "no_leakage_feature_contract": "excludes_cited_by_count_and_future_outcome_columns",
                },
                {
                    "model": "metadata_hgb",
                    "validation_design": "leave_domain_out",
                    "heldout_domain": "graphene",
                    "spearman": 0.24,
                    "bootstrap_ci_low": 0.04,
                    "bootstrap_ci_high": 0.35,
                    "top_decile_future_top20_enrichment": 4.0,
                    "no_leakage_feature_contract": "excludes_cited_by_count_and_future_outcome_columns",
                },
            ]
        )

        updated = update_nonlinear_challenger_diagnostics(summary, challenger)

        self.assertTrue(updated["overall_pass"])
        self.assertEqual("metadata_hgb_no_leakage", updated["primary_model"]["model"])
        self.assertEqual(1, updated["checks"]["learned_oof_spearman_ge_0_45"])
        self.assertEqual(1, updated["checks"]["top_decile_enrichment_ge_5x"])
        self.assertEqual(1, updated["checks"]["leave_domain_out_positive_ci"])
        self.assertEqual("strong predictive evidence", updated["status_label"])
        self.assertEqual(0.31, updated["simplex_learned_oof_spearman"])
        self.assertEqual("promoted_primary_model", updated["holdout_validation"]["source"])
        self.assertEqual(1, updated["holdout_validation"]["leave_domain_out_positive_ci"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig10.build_fig10_ablation import (  # noqa: E402
    METRICS,
    VARIANTS,
    ablate_case_metrics,
    build_error_taxonomy,
    build_preference_results,
    derive_full_aspr_case_metrics,
    summarize_ablation,
)


def _toy_fig4_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paper_id": ["p1", "p2", "p3"],
            "structured_semantic_consistency_mean": [4.0, 4.5, 3.5],
            "soft_claim_recall": [0.6, 0.8, 0.5],
            "novelty_semantic_coverage": [0.5, 0.75, 0.25],
            "prior_art_semantic_coverage": [0.4, 0.5, 0.2],
            "contradiction_rate": [0.0, 0.05, 0.1],
            "overclaiming_flag": [0, 0, 1],
            "peer_flesch_reading_ease": [38.0, 42.0, 35.0],
            "agent_flesch_reading_ease": [45.0, 44.0, 39.0],
            "agent_grammar_errors_per_5000": [0.0, 2.0, 5.0],
            "agent_spelling_errors_per_5000": [0.0, 0.0, 3.0],
            "claim_evidence_coverage": [0.6, 0.8, 0.5],
            "total_peer_aspects": [6, 8, 5],
            "covered_peer_aspects": [5, 7, 4],
        }
    )


class Fig10AblationTests(unittest.TestCase):
    def test_ablation_contract_contains_required_variants_and_metrics(self) -> None:
        full_cases = derive_full_aspr_case_metrics(_toy_fig4_metrics())
        case_scores = ablate_case_metrics(full_cases)
        summary, forest = summarize_ablation(case_scores)

        self.assertTrue(set(VARIANTS).issubset(set(summary["variant"])))
        self.assertTrue({metric for metric, _, _ in METRICS}.issubset(set(summary["metric"])))

        full = float(forest.loc[forest["variant"].eq("full ASPR"), "mean"].iloc[0])
        generic = float(forest.loc[forest["variant"].eq("generic LLM-only baseline"), "mean"].iloc[0])
        self.assertLess(generic, full)

    def test_preference_and_error_tables_are_pipeline_labeled(self) -> None:
        full_cases = derive_full_aspr_case_metrics(_toy_fig4_metrics())
        case_scores = ablate_case_metrics(full_cases)
        _, forest = summarize_ablation(case_scores)
        preference = build_preference_results(forest)
        errors = build_error_taxonomy(case_scores)

        self.assertEqual(set(preference["evaluator_type"]), {"LLM-as-judge"})
        self.assertTrue(preference["source"].str.contains("no_human_scores_available").all())
        self.assertGreaterEqual(len(errors["error_type"].unique()), 8)
        self.assertTrue(errors["error_rate"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()

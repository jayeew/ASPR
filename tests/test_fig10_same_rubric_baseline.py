from __future__ import annotations

import os
import unittest

from experiments.kg_perturbation_fig10.build_fig10_same_rubric_baseline import (
    classify_evaluable_cases,
    generic_response_to_fig4_label,
    match_paper_same_rubric,
    same_rubric_status_from_counts,
    summarize_same_rubric_results,
)


def _generic_response() -> dict[str, object]:
    return {
        "scores_1_5": {
            "novelty": 4,
            "significance": 4,
            "prior_art": 3,
            "evidence_rigor": 2,
            "limitations": 2,
            "future_work": 3,
            "overall": 3,
            "unsupported_or_overclaiming_risk": 2,
        },
        "review_points": {
            "novelty": ["The mechanism is tested in a useful setting."],
            "significance": ["The study may matter for the field."],
            "prior_art": ["Prior work is only partly identifiable from the abstract."],
            "evidence_rigor": ["The assay details need careful validation."],
            "limitations": ["The abstract does not establish all controls."],
            "future_work": ["Additional perturbation tests would help."],
        },
        "recommendation": "major revision",
        "confidence": 0.55,
    }


def _peer_label() -> dict[str, object]:
    return {
        "paper_id": "paper-1",
        "kind": "peer_review",
        "aspects": {
            "novelty": {"points": ["The mechanism is tested in a useful setting."], "quotes": []},
            "significance": {"points": ["The study may matter for the field."], "quotes": []},
            "prior_art_comparison": {"points": ["Prior work is only partly identifiable from the abstract."], "quotes": []},
            "evidence_rigor": {"points": ["The assay details need careful validation."], "quotes": []},
            "limitations": {"points": ["The abstract does not establish all controls."], "quotes": []},
            "future_work": {"points": ["Additional perturbation tests would help."], "quotes": []},
        },
    }


class Fig10SameRubricBaselineTests(unittest.TestCase):
    def test_generic_response_maps_to_fig4_label_schema(self) -> None:
        label = generic_response_to_fig4_label("paper-1", _generic_response())

        self.assertEqual("generic_llm_baseline", label["kind"])
        self.assertIn("prior_art_comparison", label["aspects"])
        self.assertEqual(3.0, label["aspects"]["prior_art_comparison"]["score_1_5"])
        self.assertEqual(2.0, label["aspects"]["evidence_rigor"]["score_1_5"])

    def test_match_paper_same_rubric_reuses_fig4_relation_fields(self) -> None:
        old = os.environ.get("FIG4_USE_EMBEDDING_MATCH")
        os.environ["FIG4_USE_EMBEDDING_MATCH"] = "0"
        try:
            label = generic_response_to_fig4_label("paper-1", _generic_response())
            rows = match_paper_same_rubric(
                paper_id="paper-1",
                title="A careful mechanistic study",
                peer_label=_peer_label(),
                generic_label=label,
            )
        finally:
            if old is None:
                os.environ.pop("FIG4_USE_EMBEDDING_MATCH", None)
            else:
                os.environ["FIG4_USE_EMBEDDING_MATCH"] = old

        self.assertGreaterEqual(len(rows), 6)
        self.assertTrue({"paper_id", "aspect", "relation", "score", "bge_only_relation", "refined_relation"}.issubset(rows[0]))
        self.assertTrue(all(row["scoring_protocol"] == "same_fig4_semantic_matcher" for row in rows))
        self.assertTrue(any(row["relation"] == "entailed" for row in rows))

    def test_summarize_same_rubric_results_emits_fig10_metric_schema(self) -> None:
        old = os.environ.get("FIG4_USE_EMBEDDING_MATCH")
        os.environ["FIG4_USE_EMBEDDING_MATCH"] = "0"
        try:
            label = generic_response_to_fig4_label("paper-1", _generic_response())
            rows = match_paper_same_rubric(
                paper_id="paper-1",
                title="A careful mechanistic study",
                peer_label=_peer_label(),
                generic_label=label,
            )
        finally:
            if old is None:
                os.environ.pop("FIG4_USE_EMBEDDING_MATCH", None)
            else:
                os.environ["FIG4_USE_EMBEDDING_MATCH"] = old

        summary = summarize_same_rubric_results(rows, expected_case_count=1)

        self.assertEqual("same_fig4_semantic_matcher", summary.iloc[0]["scoring_protocol"])
        self.assertEqual("observed_generic_llm_run", summary.iloc[0]["source"])
        for metric in [
            "semantic_agreement",
            "novelty_coverage",
            "prior_art_accuracy",
            "factuality",
            "unsupported_claim_rate",
            "evidence_trace_completeness",
            "review_structure_coverage",
        ]:
            self.assertIn(metric, summary.columns)
            self.assertGreaterEqual(float(summary.iloc[0][metric]), 0.0)
            self.assertLessEqual(float(summary.iloc[0][metric]), 1.0)

    def test_zero_peer_point_cases_are_documented_as_evaluable_sample_exclusions(self) -> None:
        manifest = [
            {"paper_id": "paper-1", "title": "paper with peer points"},
            {"paper_id": "paper-2", "title": "paper without peer points"},
        ]
        labels = {
            ("paper-1", "peer_review"): _peer_label(),
            ("paper-2", "peer_review"): {"paper_id": "paper-2", "kind": "peer_review", "aspects": {}},
        }
        raw_by_paper = {
            "paper-1": {"paper_id": "paper-1", "run_status": "ok"},
            "paper-2": {"paper_id": "paper-2", "run_status": "ok"},
        }

        exclusions, evaluable_ids = classify_evaluable_cases(
            manifest=manifest,
            labels=labels,
            raw_by_paper=raw_by_paper,
            max_points_per_aspect=4,
        )

        self.assertEqual(["paper-1"], evaluable_ids)
        self.assertEqual(1, len(exclusions))
        self.assertEqual("paper-2", exclusions[0]["case_id"])
        self.assertEqual("zero_peer_review_points", exclusions[0]["exclusion_reason"])
        self.assertEqual(0, exclusions[0]["peer_point_count"])
        self.assertEqual("pre_specified_evaluable_sample_exclusion", exclusions[0]["exclusion_policy"])
        self.assertEqual(
            "observed_generic_llm_run_same_rubric_evaluable_complete",
            same_rubric_status_from_counts(case_count=1, expected_case_count=2, excluded_case_count=1),
        )


if __name__ == "__main__":
    unittest.main()

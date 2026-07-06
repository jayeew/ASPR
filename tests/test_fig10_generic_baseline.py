from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.kg_perturbation_fig10.build_fig10_ablation import METRICS
import experiments.kg_perturbation_fig10.build_fig10_generic_baseline as generic_baseline
from experiments.kg_perturbation_fig10.build_fig10_generic_baseline import (
    build_generic_prompt,
    parse_json_response,
    run_generic_baseline,
    score_generic_response,
    write_generic_baseline_results,
)


def _toy_paper() -> pd.Series:
    return pd.Series(
        {
            "paper_id": "paper-1",
            "title": "A careful mechanistic study",
            "abstract": "We combine perturbation assays and imaging to test a proposed mechanism.",
            "peer_novelty": 4.0,
            "peer_significance": 4.0,
            "peer_rigor": 2.0,
            "peer_limitations": 2.0,
            "peer_future_work": 3.0,
            "graph_confidence": 0.99,
            "top_mechanisms": "SHOULD_NOT_APPEAR_IN_PROMPT",
        }
    )


def _toy_response() -> dict[str, object]:
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


class Fig10GenericBaselineTests(unittest.TestCase):
    def test_prompt_uses_only_manuscript_metadata_not_peer_or_graph(self) -> None:
        prompt = build_generic_prompt(_toy_paper())

        self.assertIn("A careful mechanistic study", prompt)
        self.assertIn("perturbation assays", prompt)
        self.assertNotIn("SHOULD_NOT_APPEAR_IN_PROMPT", prompt)
        self.assertNotIn("peer_novelty", prompt)
        self.assertNotIn("graph_confidence", prompt)

    def test_parse_json_response_extracts_object_from_model_text(self) -> None:
        payload = _toy_response()
        text = "Here is the JSON:\n```json\n" + json.dumps(payload) + "\n```"

        parsed = parse_json_response(text)

        self.assertEqual("major revision", parsed["recommendation"])
        self.assertEqual(4, parsed["scores_1_5"]["novelty"])

    def test_score_generic_response_emits_required_fig10_metrics(self) -> None:
        scored = score_generic_response(_toy_paper(), _toy_response(), model_name="qwen3:8b", prompt_hash="abc123")

        metric_keys = {metric for metric, _, _ in METRICS}
        self.assertTrue(metric_keys.issubset(scored.keys()))
        self.assertEqual("observed_generic_llm_run", scored["source"])
        self.assertEqual("title_abstract_only_no_graph_no_peer_review", scored["input_scope"])
        for metric in metric_keys:
            self.assertGreaterEqual(float(scored[metric]), 0.0)
            self.assertLessEqual(float(scored[metric]), 1.0)

    def test_write_generic_baseline_results_records_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = [
                score_generic_response(_toy_paper(), _toy_response(), model_name="qwen3:8b", prompt_hash="abc123")
            ]

            result = write_generic_baseline_results(out_dir, rows=rows, expected_case_count=1, model_name="qwen3:8b")

            self.assertTrue((out_dir / "fig10_generic_llm_baseline_results.csv").exists())
            self.assertTrue((out_dir / "fig10_generic_llm_baseline_manifest.json").exists())
            self.assertEqual("observed_generic_llm_run", result["status"])
            self.assertEqual(1, result["case_count"])

    def test_run_generic_baseline_skip_existing_reuses_ok_case_without_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fig4_metrics = out_dir / "fig4_metrics.csv"
            pd.DataFrame([_toy_paper().to_dict()]).to_csv(fig4_metrics, index=False)
            existing_raw = {
                "paper_id": "paper-1",
                "model_name": "qwen3:8b",
                "prompt_hash": "abc123",
                "input_scope": "title_abstract_only_no_graph_no_peer_review",
                "run_status": "ok",
                "raw_response": json.dumps(_toy_response()),
                "parsed_response": _toy_response(),
            }
            (out_dir / "fig10_generic_llm_baseline_outputs.jsonl").write_text(
                json.dumps(existing_raw) + "\n",
                encoding="utf-8",
            )
            scored = score_generic_response(_toy_paper(), _toy_response(), model_name="qwen3:8b", prompt_hash="abc123")
            pd.DataFrame([scored]).to_csv(out_dir / "fig10_generic_llm_baseline_results.csv", index=False)

            original_call = generic_baseline.call_ollama

            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("Ollama should not be called for existing ok case")

            generic_baseline.call_ollama = fail_if_called
            try:
                result = run_generic_baseline(
                    fig4_metrics=fig4_metrics,
                    out_dir=out_dir,
                    model_name="qwen3:8b",
                    max_cases=1,
                    resume=True,
                    skip_existing=True,
                )
            finally:
                generic_baseline.call_ollama = original_call

            self.assertEqual("observed_generic_llm_run", result["status"])
            self.assertEqual(1, result["case_count"])
            outputs = (out_dir / "fig10_generic_llm_baseline_outputs.jsonl").read_text(encoding="utf-8")
            self.assertIn('"run_status": "ok"', outputs)

    def test_run_generic_baseline_resume_filters_stale_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fig4_metrics = out_dir / "fig4_metrics.csv"
            current = _toy_paper().to_dict()
            stale = _toy_paper().to_dict()
            stale["paper_id"] = "stale-paper"
            pd.DataFrame([current]).to_csv(fig4_metrics, index=False)

            current_raw = {
                "paper_id": "paper-1",
                "model_name": "qwen3:8b",
                "prompt_hash": "abc123",
                "input_scope": "title_abstract_only_no_graph_no_peer_review",
                "run_status": "ok",
                "raw_response": json.dumps(_toy_response()),
                "parsed_response": _toy_response(),
            }
            stale_raw = dict(current_raw)
            stale_raw["paper_id"] = "stale-paper"
            (out_dir / "fig10_generic_llm_baseline_outputs.jsonl").write_text(
                json.dumps(current_raw) + "\n" + json.dumps(stale_raw) + "\n",
                encoding="utf-8",
            )
            current_scored = score_generic_response(
                pd.Series(current),
                _toy_response(),
                model_name="qwen3:8b",
                prompt_hash="abc123",
            )
            stale_scored = score_generic_response(
                pd.Series(stale),
                _toy_response(),
                model_name="qwen3:8b",
                prompt_hash="abc123",
            )
            pd.DataFrame([current_scored, stale_scored]).to_csv(
                out_dir / "fig10_generic_llm_baseline_results.csv",
                index=False,
            )

            original_call = generic_baseline.call_ollama

            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("Ollama should not be called for existing ok case")

            generic_baseline.call_ollama = fail_if_called
            try:
                result = run_generic_baseline(
                    fig4_metrics=fig4_metrics,
                    out_dir=out_dir,
                    model_name="qwen3:8b",
                    resume=True,
                    skip_existing=True,
                )
            finally:
                generic_baseline.call_ollama = original_call

            self.assertEqual(1, result["case_count"])
            self.assertEqual(1, result["ok_count"])
            outputs = (out_dir / "fig10_generic_llm_baseline_outputs.jsonl").read_text(encoding="utf-8")
            self.assertIn("paper-1", outputs)
            self.assertNotIn("stale-paper", outputs)


if __name__ == "__main__":
    unittest.main()

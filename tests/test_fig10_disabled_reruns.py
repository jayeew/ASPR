from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.fig10.old.build_fig10_ablation import METRICS
from experiments.fig10.old.build_fig10_ablation import (
    build_fig10_true_rerun_completion_audit,
    import_observed_disabled_module_rerun_sidecar,
)
from experiments.fig10.old.build_fig10_disabled_reruns import (
    build_disabled_variant_prompt,
    run_disabled_module_reruns,
    score_disabled_response,
)
import experiments.fig10.old.build_fig10_disabled_reruns as disabled_reruns


def _toy_paper() -> pd.Series:
    return pd.Series(
        {
            "paper_id": "paper-1",
            "title": "A careful perturbation study",
            "abstract": "We combine perturbation assays and imaging to test a proposed mechanism.",
            "peer_novelty": 4.0,
            "peer_significance": 4.0,
            "peer_rigor": 3.0,
            "peer_limitations": 2.0,
            "peer_future_work": 3.0,
        }
    )


def _toy_response() -> dict[str, object]:
    return {
        "scores_1_5": {
            "novelty": 3,
            "significance": 4,
            "prior_art": 2,
            "evidence_rigor": 3,
            "limitations": 2,
            "future_work": 3,
            "overall": 3,
            "unsupported_or_overclaiming_risk": 3,
        },
        "review_points": {
            "novelty": ["The study tests a useful mechanism."],
            "significance": ["The result may matter for the field."],
            "prior_art": ["Prior work is only partly available in this disabled run."],
            "evidence_rigor": ["The evidence should be checked against the assay controls."],
            "limitations": ["The abstract leaves some boundary conditions unclear."],
            "future_work": ["Additional perturbation experiments would help."],
        },
        "recommendation": "major revision",
        "confidence": 0.55,
    }


class Fig10DisabledRerunTests(unittest.TestCase):
    def test_no_graph_prompt_excludes_graph_evidence(self) -> None:
        prompt = build_disabled_variant_prompt(
            _toy_paper(),
            variant="no graph agent",
            retrieved_papers=[{"title": "Prior paper", "abstract": "Prior-art context."}],
            graph_evidence="GRAPH_SIGNAL_SHOULD_NOT_APPEAR",
        )

        self.assertIn("no graph agent", prompt)
        self.assertIn("Prior paper", prompt)
        self.assertNotIn("GRAPH_SIGNAL_SHOULD_NOT_APPEAR", prompt)

    def test_no_retrieval_prompt_excludes_prior_art_context(self) -> None:
        prompt = build_disabled_variant_prompt(
            _toy_paper(),
            variant="no prior-art retrieval",
            retrieved_papers=[{"title": "PRIOR_TITLE_SHOULD_NOT_APPEAR", "abstract": "PRIOR_ABSTRACT_SHOULD_NOT_APPEAR"}],
            graph_evidence="Graph evidence may remain for this variant.",
        )

        self.assertIn("no prior-art retrieval", prompt)
        self.assertIn("Graph evidence may remain", prompt)
        self.assertNotIn("PRIOR_TITLE_SHOULD_NOT_APPEAR", prompt)
        self.assertNotIn("PRIOR_ABSTRACT_SHOULD_NOT_APPEAR", prompt)

    def test_disabled_response_scores_to_fig10_sidecar_schema(self) -> None:
        scored = score_disabled_response(
            _toy_paper(),
            _toy_response(),
            variant="no graph agent",
            model_name="qwen3:8b",
            prompt_hash="abc123",
            runtime_seconds=2.5,
            review_text_path="review.txt",
            evidence_trace_path="trace.json",
        )

        self.assertEqual("true_disabled_module_rerun", scored["source"])
        self.assertEqual("no graph agent", scored["variant"])
        self.assertEqual("ok", scored["run_status"])
        for metric, _, _ in METRICS:
            self.assertIn(metric, scored)
            self.assertGreaterEqual(float(scored[metric]), 0.0)
            self.assertLessEqual(float(scored[metric]), 1.0)

    def test_run_disabled_module_reruns_writes_completed_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fig4_metrics = out_dir / "fig4_metrics.csv"
            pd.DataFrame([_toy_paper().to_dict()]).to_csv(fig4_metrics, index=False)

            original_call = disabled_reruns.call_ollama

            def fake_call_ollama(*_args, **_kwargs) -> str:
                return json.dumps(_toy_response())

            disabled_reruns.call_ollama = fake_call_ollama
            try:
                manifest = run_disabled_module_reruns(
                    fig4_metrics=fig4_metrics,
                    out_dir=out_dir,
                    variants=["no graph agent"],
                    model_name="qwen3:8b",
                    max_cases=1,
                    timeout=1,
                )
            finally:
                disabled_reruns.call_ollama = original_call

            sidecar = pd.read_csv(out_dir / "fig10_completed_disabled_module_reruns.csv")

        self.assertEqual("observed_disabled_module_rerun_partial", manifest["status"])
        self.assertEqual(1, int(manifest["ok_count"]))
        self.assertEqual({"no graph agent"}, set(sidecar["variant"]))
        self.assertEqual({"true_disabled_module_rerun"}, set(sidecar["source"]))

    def test_run_disabled_module_reruns_sidecar_imports_into_true_rerun_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            out_dir = Path("relative_fig10_out")
            fig4_metrics = out_dir / "fig4_metrics.csv"
            out_dir.mkdir()
            pd.DataFrame([_toy_paper().to_dict()]).to_csv(fig4_metrics, index=False)

            original_call = disabled_reruns.call_ollama

            def fake_call_ollama(*_args, **_kwargs) -> str:
                return json.dumps(_toy_response())

            disabled_reruns.call_ollama = fake_call_ollama
            try:
                run_disabled_module_reruns(
                    fig4_metrics=fig4_metrics,
                    out_dir=out_dir,
                    variants=["no graph agent"],
                    model_name="qwen3:8b",
                    max_cases=1,
                    timeout=1,
                )
            finally:
                disabled_reruns.call_ollama = original_call
                os.chdir(original_cwd)

            imported = import_observed_disabled_module_rerun_sidecar(
                expected_ids=["paper-1"],
                sidecar_path=tmp_path / out_dir / "fig10_completed_disabled_module_reruns.csv",
                out_dir=tmp_path / out_dir,
            )
            audit = build_fig10_true_rerun_completion_audit(tmp_path / out_dir, expected_ids=["paper-1"])

        self.assertEqual(1, len(imported))
        no_graph_audit = audit[audit["variant"].eq("no graph agent")].iloc[0]
        self.assertEqual(1, int(no_graph_audit["observed_case_variant_pairs"]))
        self.assertEqual(0, int(no_graph_audit["invalid_artifact_pairs"]))


if __name__ == "__main__":
    unittest.main()

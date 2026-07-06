from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig10.build_fig10_ablation import (  # noqa: E402
    METRICS,
    PREFERENCE_DIMENSIONS,
    VARIANTS,
    ablate_case_metrics,
    build_fig10_blinded_preference_completion_audit,
    build_fig10_true_rerun_completion_audit,
    build_fig10_blinded_preference_package,
    build_evidence_provenance,
    build_error_taxonomy,
    build_preference_results,
    build_replacement_gates,
    build_true_rerun_results_template,
    build_true_rerun_contract,
    derive_full_aspr_case_metrics,
    human_preference_status,
    load_observed_generic_baseline,
    load_observed_true_module_reruns,
    materialize_fig4_full_aspr_true_rerun_results,
    materialize_observed_generic_llm_true_rerun_results,
    merge_fig10_evaluator_preference_returns,
    import_observed_disabled_module_rerun_sidecar,
    quality_gates,
    summarize_ablation,
    true_module_rerun_status,
)


def _toy_fig4_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paper_id": ["p1", "p2", "p3"],
            "title": ["Paper one", "Paper two", "Paper three"],
            "abstract": ["Abstract one", "Abstract two", "Abstract three"],
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

    def test_provenance_and_replacement_gates_block_strong_claims(self) -> None:
        provenance = build_evidence_provenance(
            fig4_metrics=PROJECT_ROOT / "outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv",
            out_dir=PROJECT_ROOT / "outputs/kg_perturbation_fig10",
        )
        gates = build_replacement_gates(provenance)

        self.assertIn("ASPR-Qwen checkpoint run", set(provenance["evidence_item"]))
        self.assertIn("blinded human preference", set(provenance["evidence_item"]))
        self.assertTrue(provenance["allowed_main_text_claim"].str.len().gt(0).all())
        self.assertTrue(provenance["forbidden_claim"].str.len().gt(0).all())
        self.assertFalse(gates["pass_for_nature_strong_claim"].all())
        self.assertTrue(gates["pass_for_pipeline_figure"].all())
        self.assertIn("checkpoint_run_observed", set(provenance["evidence_status"]))
        self.assertNotIn("missing_checkpoint", set(provenance["evidence_status"]))
        self.assertIn("missing_human_scores", set(provenance["evidence_status"]))
        generic_gate = gates[gates["gate_id"].eq("current_generic_llm_baseline")].iloc[0]
        generic_status = provenance[provenance["evidence_item"].eq("current generic LLM baseline")]["evidence_status"].iloc[0]
        if generic_status in {
            "observed_generic_llm_run_same_rubric",
            "observed_generic_llm_run_same_rubric_evaluable_complete",
        }:
            self.assertTrue(bool(generic_gate["pass_for_nature_strong_claim"]))
        else:
            self.assertFalse(bool(generic_gate["pass_for_nature_strong_claim"]))

    def test_quality_gate_accepts_provenance_audit_after_checkpoint_is_observed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "fig10_panel_text.json").write_text("{}", encoding="utf-8")
            for name in [
                "fig10_true_module_rerun_contract.csv",
                "fig10_true_rerun_completion_audit.csv",
                "fig10_blinded_preference_packet.csv",
                "fig10_blinded_preference_completion_audit.csv",
                "fig10_evidence_provenance.csv",
                "fig10_replacement_gates.csv",
            ]:
                (out_dir / name).write_text("ok\n", encoding="utf-8")
            figure = out_dir / "fig10_full.png"
            figure.write_bytes(b"0" * 10001)
            ablation_summary = pd.DataFrame(
                {
                    "variant": VARIANTS,
                    "metric": [METRICS[idx % len(METRICS)][0] for idx in range(len(VARIANTS))],
                    "source": ["observed_real_fig4_metrics"] + ["pipeline_estimate_formula_from_fig4"] * (len(VARIANTS) - 1),
                }
            )
            for metric, _, _ in METRICS:
                ablation_summary = pd.concat(
                    [
                        ablation_summary,
                        pd.DataFrame(
                            {
                                "variant": ["full ASPR"],
                                "metric": [metric],
                                "source": ["observed_real_fig4_metrics"],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            provenance = pd.DataFrame(
                {
                    "evidence_item": [
                        "missing-module ablation estimates",
                        "blinded human preference",
                        "ASPR-Qwen checkpoint run",
                        "current generic LLM baseline",
                        "full ASPR automatic metrics",
                    ],
                    "evidence_status": [
                        "pipeline_estimate_formula_from_fig4",
                        "missing_human_scores",
                        "checkpoint_run_observed",
                        "observed_generic_llm_run_same_rubric",
                        "observed_real_fig4_metrics",
                    ],
                }
            )
            replacement_gates = pd.DataFrame(
                {
                    "pass_for_nature_strong_claim": [0, 0, 1, 1, 1],
                    "pass_for_pipeline_figure": [1, 1, 1, 1, 1],
                }
            )
            preference = pd.DataFrame(
                {
                    "source": ["llm_judge_pipeline_ready_no_human_scores_available"],
                }
            )
            error_taxonomy = pd.DataFrame({"error_type": [f"e{i}" for i in range(8)]})

            gates = quality_gates(
                out_dir,
                ablation_summary=ablation_summary,
                preference=preference,
                error_taxonomy=error_taxonomy,
                provenance=provenance,
                replacement_gates=replacement_gates,
                figures=[figure],
            )

            self.assertEqual(1, gates["checks"]["provenance_audit_exists"])
            self.assertEqual(0, gates["nature_strong_claim_ready"])

    def test_observed_generic_baseline_replaces_pipeline_estimate_when_complete(self) -> None:
        full_cases = derive_full_aspr_case_metrics(_toy_fig4_metrics())
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "fig10_generic_llm_baseline_results.csv"
            rows = []
            for _, case in full_cases.iterrows():
                row = {
                    "case_id": case["case_id"],
                    "source": "observed_generic_llm_run",
                    "run_status": "ok",
                }
                for metric, _, _ in METRICS:
                    row[metric] = 0.25
                rows.append(row)
            pd.DataFrame(rows).to_csv(baseline_path, index=False)

            observed = load_observed_generic_baseline(baseline_path, expected_case_ids=full_cases["case_id"])
            case_scores = ablate_case_metrics(full_cases, observed_generic_baseline=observed)

        generic = case_scores[case_scores["variant"].eq("generic LLM-only baseline")]
        self.assertEqual({"observed_generic_llm_run"}, set(generic["source"]))
        self.assertTrue(generic["score"].eq(0.25).all())

    def test_preference_results_allow_observed_comparator_to_beat_full(self) -> None:
        forest = pd.DataFrame(
            {
                "variant": [
                    "full ASPR",
                    "generic LLM-only baseline",
                    "no graph agent",
                    "no ASPR-Qwen",
                    "no prior-art retrieval",
                    "no fusion",
                    "no verifier",
                ],
                "mean": [0.60, 0.70, 0.50, 0.50, 0.50, 0.50, 0.50],
            }
        )

        preference = build_preference_results(forest)
        generic = preference[preference["comparison"].eq("full ASPR vs generic LLM-only baseline")].iloc[0]

        self.assertLess(generic["full_aspr_win_rate"], generic["comparator_win_rate"])

    def test_same_rubric_partial_generic_baseline_updates_provenance_without_strong_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "fig10_generic_llm_same_rubric_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "observed_generic_llm_run_same_rubric_partial",
                        "case_count": 48,
                        "expected_case_count": 50,
                        "scoring_protocol": "same_fig4_semantic_matcher",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "case_id": ["p1"],
                    "source": ["observed_generic_llm_run"],
                    "run_status": ["ok"],
                    "scoring_protocol": ["same_fig4_semantic_matcher"],
                }
            ).to_csv(out_dir / "fig10_generic_llm_same_rubric_results.csv", index=False)

            provenance = build_evidence_provenance(
                fig4_metrics=PROJECT_ROOT / "outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv",
                out_dir=out_dir,
            )
            gates = build_replacement_gates(provenance)

        generic_status = provenance[provenance["evidence_item"].eq("current generic LLM baseline")]["evidence_status"].iloc[0]
        generic_gate = gates[gates["gate_id"].eq("current_generic_llm_baseline")].iloc[0]
        self.assertEqual("observed_generic_llm_run_same_rubric_partial", generic_status)
        self.assertFalse(bool(generic_gate["pass_for_nature_strong_claim"]))

    def test_same_rubric_evaluable_complete_generic_baseline_passes_baseline_gate_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "fig10_generic_llm_same_rubric_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "observed_generic_llm_run_same_rubric_evaluable_complete",
                        "case_count": 48,
                        "expected_case_count": 50,
                        "evaluable_case_count": 48,
                        "excluded_case_count": 2,
                        "scoring_protocol": "same_fig4_semantic_matcher",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "case_id": ["p1"],
                    "source": ["observed_generic_llm_run"],
                    "run_status": ["ok"],
                    "scoring_protocol": ["same_fig4_semantic_matcher"],
                }
            ).to_csv(out_dir / "fig10_generic_llm_same_rubric_results.csv", index=False)

            provenance = build_evidence_provenance(
                fig4_metrics=PROJECT_ROOT / "outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv",
                out_dir=out_dir,
            )
            gates = build_replacement_gates(provenance)

        generic_status = provenance[provenance["evidence_item"].eq("current generic LLM baseline")]["evidence_status"].iloc[0]
        generic_gate = gates[gates["gate_id"].eq("current_generic_llm_baseline")].iloc[0]
        self.assertEqual("observed_generic_llm_run_same_rubric_evaluable_complete", generic_status)
        self.assertTrue(bool(generic_gate["pass_for_nature_strong_claim"]))
        self.assertFalse(gates["pass_for_nature_strong_claim"].all())

    def test_true_ablation_human_preference_and_checkpoint_can_pass_replacement_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fig4_metrics = out_dir / "fig4_metrics.csv"
            _toy_fig4_metrics().to_csv(fig4_metrics, index=False)
            rows = []
            for variant in VARIANTS:
                for case_id in ["p1", "p2", "p3"]:
                    review_path = out_dir / "reviews" / case_id / variant.replace(" ", "_") / "review.txt"
                    trace_path = out_dir / "traces" / case_id / variant.replace(" ", "_") / "trace.json"
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text("review", encoding="utf-8")
                    trace_path.write_text("{}", encoding="utf-8")
                    row = {
                        "variant": variant,
                        "case_id": case_id,
                        "source": "true_disabled_module_rerun",
                        "run_status": "ok",
                        "review_text_path": str(review_path.relative_to(out_dir)),
                        "evidence_trace_path": str(trace_path.relative_to(out_dir)),
                        "runtime_seconds": 12.0,
                        "failure_reason": "",
                    }
                    for metric, _, _ in METRICS:
                        row[metric] = 0.75 if variant == "full ASPR" else 0.55
                    rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "fig10_true_module_rerun_results.csv", index=False)
            human_rows = []
            for evaluator_id in ["r1", "r2", "r3"]:
                for case_id in ["p1", "p2"]:
                    for dimension in ["novelty", "prior_art", "evidence_grounding", "usefulness", "factuality"]:
                        human_rows.append(
                            {
                                "comparison": "full ASPR vs generic LLM-only",
                                "case_id": case_id,
                                "dimension": dimension,
                                "evaluator_id": evaluator_id,
                                "blind_setting": "system_names_hidden",
                                "preference": "full ASPR",
                            }
                        )
            pd.DataFrame(human_rows).to_csv(out_dir / "fig10_human_preference.csv", index=False)
            (out_dir / "fig10_generic_llm_same_rubric_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "observed_generic_llm_run_same_rubric",
                        "case_count": 3,
                        "expected_case_count": 3,
                        "scoring_protocol": "same_fig4_semantic_matcher",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame({"case_id": ["p1", "p2", "p3"], "source": ["observed_generic_llm_run"] * 3, "run_status": ["ok"] * 3}).to_csv(
                out_dir / "fig10_generic_llm_same_rubric_results.csv",
                index=False,
            )
            fig9_dir = out_dir.parent / "kg_perturbation_fig9"
            fig9_dir.mkdir(exist_ok=True)
            (fig9_dir / "fig9_aspr_qwen_output.json").write_text(
                json.dumps(
                    {
                        "checkpoint_invoked": True,
                        "model_hash": "abc123",
                        "training_config": {"method": "lora"},
                        "data_version": "v1",
                        "prompt": "review",
                        "decoding_config": {"temperature": 0.2},
                        "seed": 20260630,
                        "runtime": {"seconds": 4.2},
                    }
                ),
                encoding="utf-8",
            )

            provenance = build_evidence_provenance(
                fig4_metrics=fig4_metrics,
                out_dir=out_dir,
            )
            gates = build_replacement_gates(provenance)
            observed = load_observed_true_module_reruns(
                out_dir / "fig10_true_module_rerun_results.csv",
                expected_ids=["p1", "p2", "p3"],
            )

        for gate_id in ["true_disabled_module_reruns", "blinded_human_preference", "checkpoint_generated_aspr_qwen"]:
            row = gates[gates["gate_id"].eq(gate_id)].iloc[0]
            self.assertTrue(bool(row["pass_for_nature_strong_claim"]), gate_id)
        self.assertFalse(observed.empty)
        self.assertEqual({"observed_true_module_rerun_full_aspr", "observed_true_module_rerun"}, set(observed["source"]))

    def test_true_rerun_gate_requires_full_case_coverage_and_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            for variant in VARIANTS:
                row = {
                    "variant": variant,
                    "case_id": "p1",
                    "source": "true_disabled_module_rerun",
                    "run_status": "ok",
                    "review_text_path": "missing_review.txt",
                    "evidence_trace_path": "missing_trace.json",
                    "runtime_seconds": 12.0,
                    "failure_reason": "",
                }
                for metric, _, _ in METRICS:
                    row[metric] = 0.5
                rows.append(row)
            path = out_dir / "fig10_true_module_rerun_results.csv"
            pd.DataFrame(rows).to_csv(path, index=False)

            status = true_module_rerun_status(path, expected_ids=["p1", "p2"])

        self.assertEqual("true_module_rerun_missing_expected_cases", status)

    def test_true_rerun_contract_lists_every_case_variant_pair(self) -> None:
        contract = build_true_rerun_contract(_toy_fig4_metrics())

        self.assertEqual(len(VARIANTS) * 3, len(contract))
        self.assertTrue({"case_id", "variant", "required_metric_columns", "acceptance_rule"}.issubset(contract.columns))

    def test_true_rerun_results_template_uses_contract_paths_and_required_columns(self) -> None:
        contract = build_true_rerun_contract(_toy_fig4_metrics())

        template = build_true_rerun_results_template(contract)

        required_columns = {
            "variant",
            "case_id",
            "source",
            "run_status",
            "review_text_path",
            "evidence_trace_path",
            "runtime_seconds",
            "failure_reason",
            *{metric for metric, _, _ in METRICS},
        }
        self.assertEqual(len(contract), len(template))
        self.assertTrue(required_columns.issubset(template.columns))
        self.assertTrue(template["review_text_path"].astype(str).str.startswith("true_reruns/").all())
        self.assertEqual({"true_disabled_module_rerun"}, set(template["source"].astype(str)))
        self.assertTrue(template["runtime_seconds"].astype(str).eq("").all())

    def test_materialized_fig4_full_aspr_reruns_flag_fallback_without_counting_it_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            agent_outputs = out_dir / "fig4_agent_outputs.jsonl"
            agent_outputs.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "paper_id": "p1",
                                "success": True,
                                "innovation_evaluation": "Observed full ASPR review for p1.",
                                "agent_runtime_seconds": 12.5,
                                "retrieval_source": "openalex",
                                "retrieved_papers_count": 7,
                                "graph_metric_evidence": {"confidence": 0.8},
                            }
                        ),
                        json.dumps(
                            {
                                "paper_id": "p2",
                                "success": False,
                                "innovation_evaluation": "Fallback review for p2.",
                                "agent_runtime_seconds": 3.0,
                                "retrieval_source": "openalex",
                                "failure_reason": "lats_failed_lightweight_fallback:No module named 'langgraph'",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            materialized = materialize_fig4_full_aspr_true_rerun_results(
                _toy_fig4_metrics().iloc[:2],
                agent_outputs_path=agent_outputs,
                out_dir=out_dir,
            )
            audit = build_fig10_true_rerun_completion_audit(out_dir, expected_ids=["p1", "p2"])

            self.assertEqual({"full ASPR"}, set(materialized["variant"]))
            self.assertEqual({"p1", "p2"}, set(materialized["case_id"]))
            ok_row = materialized[materialized["case_id"].eq("p1")].iloc[0]
            fallback_row = materialized[materialized["case_id"].eq("p2")].iloc[0]
            self.assertEqual("ok", ok_row["run_status"])
            self.assertIn("observed_full_aspr_rerun", ok_row["source"])
            self.assertNotEqual("ok", fallback_row["run_status"])
            self.assertIn("not_true_rerun", fallback_row["source"])
            for _, row in materialized.iterrows():
                self.assertTrue((out_dir / row["review_text_path"]).exists())
                self.assertTrue((out_dir / row["evidence_trace_path"]).exists())
                self.assertGreater(float(row["runtime_seconds"]), 0.0)
                for metric, _, _ in METRICS:
                    self.assertGreaterEqual(float(row[metric]), 0.0)
                    self.assertLessEqual(float(row[metric]), 1.0)
            full_audit = audit[audit["variant"].eq("full ASPR")].iloc[0]
            self.assertEqual(1, int(full_audit["observed_case_variant_pairs"]))
            self.assertEqual(1, int(full_audit["invalid_artifact_pairs"]))

    def test_materialized_generic_llm_baseline_counts_as_observed_generic_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            baseline_results = out_dir / "fig10_generic_llm_baseline_results.csv"
            baseline_outputs = out_dir / "fig10_generic_llm_baseline_outputs.jsonl"
            rows = []
            for case_id in ["p1", "p2"]:
                row = {
                    "case_id": case_id,
                    "paper_id": case_id,
                    "source": "observed_generic_llm_run",
                    "run_status": "ok",
                    "runtime_seconds": 4.5,
                }
                for metric, _, _ in METRICS:
                    row[metric] = 0.42
                rows.append(row)
            pd.DataFrame(rows).to_csv(baseline_results, index=False)
            baseline_outputs.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "paper_id": case_id,
                            "run_status": "ok",
                            "raw_response": f"Generic observed review for {case_id}.",
                            "parsed_response": {
                                "review_points": {
                                    "novelty": [f"Generic novelty for {case_id}."],
                                    "prior_art": [f"Generic prior-art for {case_id}."],
                                }
                            },
                            "model_name": "qwen3:8b",
                        }
                    )
                    for case_id in ["p1", "p2"]
                )
                + "\n",
                encoding="utf-8",
            )

            materialized = materialize_observed_generic_llm_true_rerun_results(
                expected_ids=["p1", "p2"],
                baseline_results_path=baseline_results,
                baseline_outputs_path=baseline_outputs,
                out_dir=out_dir,
            )
            audit = build_fig10_true_rerun_completion_audit(out_dir, expected_ids=["p1", "p2"])

            self.assertEqual({"generic LLM-only baseline"}, set(materialized["variant"]))
            self.assertEqual({"observed_true_module_rerun_generic_llm"}, set(materialized["source"]))
            for _, row in materialized.iterrows():
                self.assertTrue((out_dir / row["review_text_path"]).exists())
                self.assertTrue((out_dir / row["evidence_trace_path"]).exists())
                self.assertGreater(float(row["runtime_seconds"]), 0.0)
            generic_audit = audit[audit["variant"].eq("generic LLM-only baseline")].iloc[0]
            self.assertEqual(2, int(generic_audit["observed_case_variant_pairs"]))
            self.assertEqual(0, int(generic_audit["invalid_artifact_pairs"]))

    def test_import_observed_disabled_module_sidecar_counts_real_artifacts(self) -> None:
        full_cases = derive_full_aspr_case_metrics(_toy_fig4_metrics().iloc[:2])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            source_dir = out_dir / "incoming_runs"
            source_dir.mkdir()
            sidecar_rows = []
            for case_id in full_cases["case_id"]:
                review_path = source_dir / f"{case_id}_no_graph_review.txt"
                trace_path = source_dir / f"{case_id}_no_graph_trace.json"
                review_path.write_text(f"Observed disabled graph-agent review for {case_id}.\n", encoding="utf-8")
                trace_path.write_text(json.dumps({"case_id": case_id, "variant": "no graph agent"}) + "\n", encoding="utf-8")
                row = {
                    "case_id": case_id,
                    "variant": "no graph agent",
                    "source": "true_disabled_module_rerun",
                    "run_status": "ok",
                    "review_text_path": str(review_path),
                    "evidence_trace_path": str(trace_path),
                    "runtime_seconds": 12.5,
                    "failure_reason": "",
                }
                for metric, _, _ in METRICS:
                    row[metric] = 0.5
                sidecar_rows.append(row)
            sidecar_path = out_dir / "fig10_completed_disabled_module_reruns.csv"
            pd.DataFrame(sidecar_rows).to_csv(sidecar_path, index=False)

            imported = import_observed_disabled_module_rerun_sidecar(
                expected_ids=full_cases["case_id"],
                sidecar_path=sidecar_path,
                out_dir=out_dir,
            )
            audit = build_fig10_true_rerun_completion_audit(out_dir, expected_ids=full_cases["case_id"])

            self.assertEqual(2, len(imported))
            self.assertEqual({"observed_true_module_rerun_disabled_module"}, set(imported["source"]))
            for path_value in imported["review_text_path"].tolist() + imported["evidence_trace_path"].tolist():
                self.assertTrue((out_dir / path_value).exists())
            no_graph_audit = audit[audit["variant"].eq("no graph agent")].iloc[0]
            self.assertEqual(2, int(no_graph_audit["observed_case_variant_pairs"]))
            self.assertEqual(0, int(no_graph_audit["missing_case_variant_pairs"]))
            self.assertEqual(0, int(no_graph_audit["invalid_artifact_pairs"]))
            self.assertEqual(1, int(no_graph_audit["pass"]))

    def test_true_rerun_completion_audit_reports_missing_case_variant_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            audit = build_fig10_true_rerun_completion_audit(out_dir, expected_ids=["p1", "p2"])
            audit_written = (out_dir / "fig10_true_rerun_completion_audit.csv").exists()

        overall = audit[audit["audit_item"].eq("overall_true_rerun_ready")].iloc[0]
        missing = audit[audit["audit_item"].eq("variant_completion")]
        self.assertFalse(bool(overall["pass"]))
        self.assertEqual(len(VARIANTS) * 2, int(overall["missing_case_variant_pairs"]))
        self.assertEqual(set(VARIANTS), set(missing["variant"]))
        self.assertTrue(audit_written)

    def test_true_rerun_completion_audit_accepts_pandas_series_expected_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            audit = build_fig10_true_rerun_completion_audit(out_dir, expected_ids=pd.Series(["p1", "p2"]))

        overall = audit[audit["audit_item"].eq("overall_true_rerun_ready")].iloc[0]
        self.assertEqual(len(VARIANTS) * 2, int(overall["missing_case_variant_pairs"]))

    def test_true_rerun_completion_audit_passes_complete_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            expected_cases = ["p1", "p2"]
            for variant in VARIANTS:
                for case_id in expected_cases:
                    review_path = out_dir / "true_reruns" / case_id / variant.replace(" ", "_") / "review.txt"
                    trace_path = out_dir / "true_reruns" / case_id / variant.replace(" ", "_") / "evidence_trace.json"
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text("review", encoding="utf-8")
                    trace_path.write_text("{}", encoding="utf-8")
                    row = {
                        "variant": variant,
                        "case_id": case_id,
                        "source": "true_disabled_module_rerun",
                        "run_status": "ok",
                        "review_text_path": str(review_path.relative_to(out_dir)),
                        "evidence_trace_path": str(trace_path.relative_to(out_dir)),
                        "runtime_seconds": 12.0,
                        "failure_reason": "",
                    }
                    for metric, _, _ in METRICS:
                        row[metric] = 0.7
                    rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "fig10_true_module_rerun_results.csv", index=False)

            audit = build_fig10_true_rerun_completion_audit(out_dir, expected_ids=expected_cases)

        overall = audit[audit["audit_item"].eq("overall_true_rerun_ready")].iloc[0]
        self.assertTrue(bool(overall["pass"]))
        self.assertEqual(0, int(overall["missing_case_variant_pairs"]))
        self.assertEqual(0, int(overall["invalid_artifact_pairs"]))

    def test_blinded_preference_packet_hides_system_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            full_outputs = out_dir / "fig4_agent_outputs.jsonl"
            generic_outputs = out_dir / "fig10_generic_llm_baseline_outputs.jsonl"
            full_outputs.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "paper_id": case_id,
                            "title": f"Title {case_id}",
                            "innovation_evaluation": f"Full review text for {case_id}.",
                        }
                    )
                    for case_id in ["p1", "p2", "p3"]
                )
                + "\n",
                encoding="utf-8",
            )
            generic_outputs.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "paper_id": case_id,
                            "run_status": "ok",
                            "parsed_response": {
                                "review_points": {
                                    "novelty": [f"Generic novelty point for {case_id}."],
                                    "prior_art": [f"Generic prior-art point for {case_id}."],
                                },
                                "recommendation": "major revision",
                                "confidence": 0.6,
                            },
                        }
                    )
                    for case_id in ["p1", "p2", "p3"]
                )
                + "\n",
                encoding="utf-8",
            )

            packet, answer_key, protocol = build_fig10_blinded_preference_package(
                _toy_fig4_metrics(),
                full_outputs_path=full_outputs,
                generic_outputs_path=generic_outputs,
                out_dir=out_dir,
                seed=17,
            )
            completed_template = pd.read_csv(out_dir / "fig10_completed_blinded_preferences_template.csv")
            evaluator_templates = {
                evaluator_id: pd.read_csv(out_dir / f"fig10_completed_blinded_preferences_{evaluator_id}.csv")
                for evaluator_id in ["evaluator_1", "evaluator_2", "evaluator_3"]
            }

        self.assertEqual(3, len(packet))
        self.assertEqual(3, len(answer_key))
        self.assertNotIn("paper_id", packet.columns)
        self.assertNotIn("system_a", packet.columns)
        self.assertNotIn("system_b", packet.columns)
        self.assertTrue({"system_a", "system_b", "paper_id"}.issubset(answer_key.columns))
        self.assertEqual({"full ASPR", "generic LLM-only baseline"}, set(answer_key["system_a"]) | set(answer_key["system_b"]))
        self.assertTrue(protocol["blinding_rule"].startswith("System names"))
        self.assertIn("completed_blinded_preference_path", protocol)
        self.assertEqual("fig10_completed_blinded_preferences.csv", protocol["completed_blinded_preference_path"])
        self.assertEqual(
            {
                "evaluator_1": "fig10_completed_blinded_preferences_evaluator_1.csv",
                "evaluator_2": "fig10_completed_blinded_preferences_evaluator_2.csv",
                "evaluator_3": "fig10_completed_blinded_preferences_evaluator_3.csv",
            },
            protocol["evaluator_template_paths"],
        )
        self.assertEqual(3 * len(PREFERENCE_DIMENSIONS) * 3, len(completed_template))
        self.assertNotIn("paper_id", completed_template.columns)
        self.assertNotIn("system_a", completed_template.columns)
        self.assertNotIn("system_b", completed_template.columns)
        self.assertTrue(
            {
                "blinded_case_id",
                "dimension",
                "evaluator_id",
                "blind_setting",
                "preferred_system",
                "evaluator_type",
                "preference_source",
                "rationale",
            }.issubset(completed_template.columns)
        )
        self.assertEqual(
            {"evaluator_1", "evaluator_2", "evaluator_3"},
            set(completed_template["evaluator_id"].astype(str)),
        )
        self.assertEqual({"blinded human"}, set(completed_template["evaluator_type"].astype(str)))
        self.assertEqual(
            {"external_blinded_human_panel"},
            set(completed_template["preference_source"].astype(str)),
        )
        for evaluator_id, evaluator_template in evaluator_templates.items():
            self.assertEqual(3 * len(PREFERENCE_DIMENSIONS), len(evaluator_template))
            self.assertEqual({evaluator_id}, set(evaluator_template["evaluator_id"].astype(str)))
            self.assertNotIn("paper_id", evaluator_template.columns)
            self.assertNotIn("system_a", evaluator_template.columns)
            self.assertNotIn("system_b", evaluator_template.columns)

    def test_blinded_preference_completion_audit_requires_human_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame(
                {
                    "blinded_case_id": ["F10P-001", "F10P-002"],
                    "title": ["Paper one", "Paper two"],
                    "abstract": ["A", "B"],
                    "review_a": ["A review", "A review"],
                    "review_b": ["B review", "B review"],
                    "rubric_dimensions": [";".join(PREFERENCE_DIMENSIONS)] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_packet.csv", index=False)
            pd.DataFrame(
                {
                    "blinded_case_id": ["F10P-001", "F10P-002"],
                    "paper_id": ["p1", "p2"],
                    "system_a": ["full ASPR", "generic LLM-only baseline"],
                    "system_b": ["generic LLM-only baseline", "full ASPR"],
                }
            ).to_csv(out_dir / "fig10_blinded_preference_answer_key.csv", index=False)

            audit = build_fig10_blinded_preference_completion_audit(out_dir)

        overall = audit[audit["audit_item"].eq("overall_blinded_preference_ready")].iloc[0]
        self.assertFalse(bool(overall["pass"]))
        self.assertGreater(int(overall["missing_judgements"]), 0)

    def test_blinded_preference_completion_audit_passes_complete_unblinded_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cases = ["F10P-001", "F10P-002"]
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "title": ["Paper one", "Paper two"],
                    "abstract": ["A", "B"],
                    "review_a": ["A review", "A review"],
                    "review_b": ["B review", "B review"],
                    "rubric_dimensions": [";".join(PREFERENCE_DIMENSIONS)] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_packet.csv", index=False)
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "paper_id": ["p1", "p2"],
                    "system_a": ["full ASPR", "generic LLM-only baseline"],
                    "system_b": ["generic LLM-only baseline", "full ASPR"],
                }
            ).to_csv(out_dir / "fig10_blinded_preference_answer_key.csv", index=False)
            human_rows = []
            for evaluator_id in ["r1", "r2", "r3"]:
                for blinded_case_id in cases:
                    for dimension in PREFERENCE_DIMENSIONS:
                        human_rows.append(
                            {
                                "comparison": "full ASPR vs generic LLM-only baseline",
                                "blinded_case_id": blinded_case_id,
                                "dimension": dimension,
                                "evaluator_id": evaluator_id,
                                "blind_setting": "system_names_hidden",
                                "preference": "full ASPR",
                            }
                        )
            pd.DataFrame(human_rows).to_csv(out_dir / "fig10_human_preference.csv", index=False)

            audit = build_fig10_blinded_preference_completion_audit(out_dir)

        overall = audit[audit["audit_item"].eq("overall_blinded_preference_ready")].iloc[0]
        self.assertTrue(bool(overall["pass"]))
        self.assertEqual(0, int(overall["missing_judgements"]))

    def test_completed_blinded_preference_sidecar_materializes_human_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cases = ["F10P-001", "F10P-002"]
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "title": ["Paper one", "Paper two"],
                    "abstract": ["A", "B"],
                    "review_a": ["A review", "A review"],
                    "review_b": ["B review", "B review"],
                    "rubric_dimensions": [";".join(PREFERENCE_DIMENSIONS)] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_packet.csv", index=False)
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "paper_id": ["p1", "p2"],
                    "system_a": ["generic LLM-only baseline", "generic LLM-only baseline"],
                    "system_b": ["full ASPR", "full ASPR"],
                    "comparison": ["full ASPR vs generic LLM-only baseline"] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_answer_key.csv", index=False)
            sidecar_rows = []
            for evaluator_id in ["r1", "r2", "r3"]:
                for blinded_case_id in cases:
                    for dimension in PREFERENCE_DIMENSIONS:
                        sidecar_rows.append(
                            {
                                "blinded_case_id": blinded_case_id,
                                "dimension": dimension,
                                "evaluator_id": evaluator_id,
                                "preferred_system": "system_b",
                                "blind_setting": "system_names_hidden",
                                "rationale": "B is more grounded.",
                            }
                        )
            pd.DataFrame(sidecar_rows).to_csv(out_dir / "fig10_completed_blinded_preferences.csv", index=False)

            audit = build_fig10_blinded_preference_completion_audit(out_dir)
            materialized_path = out_dir / "fig10_human_preference.csv"
            self.assertTrue(materialized_path.exists())
            materialized = pd.read_csv(materialized_path)
            status = human_preference_status(materialized_path)

        overall = audit[audit["audit_item"].eq("overall_blinded_preference_ready")].iloc[0]
        self.assertTrue(bool(overall["pass"]))
        self.assertEqual("human_preference_observed", status)
        self.assertEqual({"full ASPR"}, set(materialized["preference"].astype(str)))
        self.assertEqual({"p1", "p2"}, set(materialized["case_id"].astype(str)))

    def test_evaluator_specific_preference_returns_merge_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cases = ["F10P-001", "F10P-002"]
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "title": ["Paper one", "Paper two"],
                    "abstract": ["A", "B"],
                    "review_a": ["A review", "A review"],
                    "review_b": ["B review", "B review"],
                    "rubric_dimensions": [";".join(PREFERENCE_DIMENSIONS)] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_packet.csv", index=False)
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "paper_id": ["p1", "p2"],
                    "system_a": ["generic LLM-only baseline", "generic LLM-only baseline"],
                    "system_b": ["full ASPR", "full ASPR"],
                    "comparison": ["full ASPR vs generic LLM-only baseline"] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_answer_key.csv", index=False)
            for evaluator_id in ["evaluator_1", "evaluator_2", "evaluator_3"]:
                rows = []
                for blinded_case_id in cases:
                    for dimension in PREFERENCE_DIMENSIONS:
                        rows.append(
                            {
                                "comparison": "full ASPR vs generic LLM-only baseline",
                                "blinded_case_id": blinded_case_id,
                                "dimension": dimension,
                                "evaluator_id": evaluator_id,
                                "blind_setting": "system_names_hidden",
                                "preferred_system": "system_b",
                                "evaluator_type": "blinded human",
                                "preference_source": "external_blinded_human_panel",
                                "rationale": "B is better grounded.",
                            }
                        )
                pd.DataFrame(rows).to_csv(
                    out_dir / f"fig10_completed_blinded_preferences_{evaluator_id}.csv",
                    index=False,
                )

            merge_audit = merge_fig10_evaluator_preference_returns(out_dir)
            combined = pd.read_csv(out_dir / "fig10_completed_blinded_preferences.csv")
            completion_audit = build_fig10_blinded_preference_completion_audit(out_dir)

        overall = merge_audit[merge_audit["audit_item"].eq("overall_evaluator_return_merge_ready")].iloc[0]
        self.assertTrue(bool(overall["pass"]))
        self.assertEqual(2 * len(PREFERENCE_DIMENSIONS) * 3, len(combined))
        self.assertEqual({"system_b"}, set(combined["preferred_system"].astype(str)))
        self.assertEqual(
            {"evaluator_1", "evaluator_2", "evaluator_3"},
            set(combined["evaluator_id"].astype(str)),
        )
        self.assertTrue(
            bool(completion_audit[completion_audit["audit_item"].eq("overall_blinded_preference_ready")].iloc[0]["pass"])
        )

    def test_evaluator_specific_preference_returns_do_not_merge_blank_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cases = ["F10P-001", "F10P-002"]
            pd.DataFrame(
                {
                    "blinded_case_id": cases,
                    "title": ["Paper one", "Paper two"],
                    "abstract": ["A", "B"],
                    "review_a": ["A review", "A review"],
                    "review_b": ["B review", "B review"],
                    "rubric_dimensions": [";".join(PREFERENCE_DIMENSIONS)] * 2,
                }
            ).to_csv(out_dir / "fig10_blinded_preference_packet.csv", index=False)
            for evaluator_id in ["evaluator_1", "evaluator_2", "evaluator_3"]:
                rows = []
                for blinded_case_id in cases:
                    for dimension in PREFERENCE_DIMENSIONS:
                        rows.append(
                            {
                                "comparison": "full ASPR vs generic LLM-only baseline",
                                "blinded_case_id": blinded_case_id,
                                "dimension": dimension,
                                "evaluator_id": evaluator_id,
                                "blind_setting": "system_names_hidden",
                                "preferred_system": "",
                                "evaluator_type": "blinded human",
                                "preference_source": "external_blinded_human_panel",
                                "rationale": "",
                            }
                        )
                pd.DataFrame(rows).to_csv(
                    out_dir / f"fig10_completed_blinded_preferences_{evaluator_id}.csv",
                    index=False,
                )

            merge_audit = merge_fig10_evaluator_preference_returns(out_dir)

        overall = merge_audit[merge_audit["audit_item"].eq("overall_evaluator_return_merge_ready")].iloc[0]
        self.assertFalse(bool(overall["pass"]))
        self.assertIn("incomplete", str(overall["failure_reason"]))
        self.assertFalse((out_dir / "fig10_completed_blinded_preferences.csv").exists())


if __name__ == "__main__":
    unittest.main()

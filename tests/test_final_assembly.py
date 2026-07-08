from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_final_assembly.build_final_assembly import (  # noqa: E402
    CAPTIONS,
    build_gap_list,
    build_final_assembly,
    fig10_same_rubric_note,
)


class FinalAssemblyTests(unittest.TestCase):
    def test_final_assembly_contract(self) -> None:
        self.assertEqual(len(CAPTIONS), 10)
        with tempfile.TemporaryDirectory(prefix="aspr_final_assembly_") as tmp:
            out_dir = Path(tmp)
            result = build_final_assembly(out_dir)

            self.assertTrue(result["quality_gates"]["overall_pass"])
            self.assertTrue((out_dir / "fig6_fig10_multi_round_consistency_report.md").exists())
            self.assertTrue((out_dir / "fig6_fig10_three_round_consistency_report.md").exists())
            self.assertTrue((out_dir / "fig1_fig10_caption_drafts.md").exists())
            checks = result["quality_gates"]["checks"]
            self.assertEqual(1, checks["multi_round_protocol_recorded"])
            self.assertEqual(1, checks["max_main_iterations_eq_6"])
            self.assertEqual(1, checks["auto_iteration_no_user_choice"])
            audit = (out_dir / "fig1_fig10_cross_figure_audit.csv").read_text(encoding="utf-8")
            captions = (out_dir / "fig1_fig10_caption_drafts.md").read_text(encoding="utf-8")
            self.assertIn("Full-rerun robustness and boundary conditions", audit)
            self.assertNotIn("Cached robustness and boundary conditions", audit)
            self.assertIn("construction-matched OpenAlex full-graph rebuild", captions)
            self.assertNotIn("Cached/proxy probes are complemented", captions)
            self.assertNotIn("explicitly assumed ASPR-Qwen draft", captions)
            self.assertNotIn("not representative checkpoint performance", captions)
            self.assertIn("checkpoint-generated ASPR-Qwen", captions)
            self.assertNotIn("Blinded human preference remain governed", captions)
            self.assertIn("Blinded human preference remains governed", captions)
            self.assertNotIn("human-like", captions.lower())
            self.assertIn("AI/AI-enabled science frontier", captions)
            self.assertIn("point-cloud rows", captions)
            gaps = (out_dir / "fig1_fig10_pipeline_ready_gaps.csv").read_text(encoding="utf-8")
            self.assertIn("Fig.7", gaps)
            self.assertIn("Replacement gates are not passed", gaps)
            self.assertNotIn("Fig.5,", gaps)
            self.assertNotIn("ASPR-Qwen output is assumed", gaps)
            self.assertNotIn("ASPR-Qwen checkpoint outputs", gaps)
            self.assertNotIn("Fig.6,A fresh OpenAlex", gaps)
            self.assertNotIn("rank stability remains below", gaps)
            collection_path = out_dir / "fig1_fig10_strict_evidence_collection_checklist.csv"
            self.assertTrue(collection_path.exists())
            collection = collection_path.read_text(encoding="utf-8")
            self.assertIn("fig4_completed_blinded_labels.csv", collection)
            self.assertIn("fig4_completed_blinded_labels_template.csv", collection)
            self.assertIn("fig4_completed_blinded_labels_labeler_1.csv", collection)
            self.assertIn("fig4_completed_blinded_labels_labeler_2.csv", collection)
            self.assertIn("fig4_completed_blinded_labels_labeler_3.csv", collection)
            self.assertIn("non-LLM/non-synthetic label_source", collection)
            self.assertIn("fig9_checkpoint_metadata_template.json", collection)
            self.assertIn("summary_judgement", collection)
            self.assertIn("major_strengths", collection)
            self.assertIn("major_concerns", collection)
            self.assertIn("fig10_true_module_rerun_results_template.csv", collection)
            self.assertIn("fig10_completed_blinded_preferences.csv", collection)
            self.assertIn("fig10_completed_blinded_preferences_template.csv", collection)
            self.assertIn("fig10_completed_blinded_preferences_evaluator_1.csv", collection)
            self.assertIn("fig10_completed_blinded_preferences_evaluator_2.csv", collection)
            self.assertIn("fig10_completed_blinded_preferences_evaluator_3.csv", collection)
            self.assertIn("evaluator_type/preference_source provenance", collection)
            self.assertIn("non-LLM/non-synthetic", collection)
            self.assertIn("evidence_grounding/prior_art/usefulness", collection)
            handoff_path = out_dir / "fig1_fig10_external_evidence_handoff.md"
            self.assertTrue(handoff_path.exists())
            handoff = handoff_path.read_text(encoding="utf-8")
            self.assertIn("# Fig.1-Fig.10 Strict External Evidence Handoff", handoff)
            self.assertIn("main_claim_ready: 1", handoff)
            self.assertIn("strict_all_figures_ready: 0", handoff)
            self.assertIn("make figures-external-evidence-intake", handoff)
            self.assertIn("make fig4-merge-blinded-labels", handoff)
            self.assertIn("make fig10-merge-blinded-preferences", handoff)
            self.assertIn("fig4_completed_blinded_labels.csv", handoff)
            self.assertIn("fig4_completed_blinded_labels_labeler_1.csv", handoff)
            self.assertIn("fig4_completed_blinded_labels_labeler_2.csv", handoff)
            self.assertIn("fig4_completed_blinded_labels_labeler_3.csv", handoff)
            self.assertIn("fig10_completed_blinded_preferences.csv", handoff)
            self.assertIn("strict external-evidence check", handoff)
            self.assertIn("Fig.4", handoff)
            self.assertIn("fig4_completed_blinded_labels.csv", handoff)
            self.assertIn("fig4_completed_blinded_labels_template.csv", handoff)
            self.assertIn("non-LLM/non-synthetic label_source", handoff)
            self.assertIn("Fig.9", handoff)
            self.assertIn("fig9_checkpoint_metadata.json", handoff)
            self.assertIn("Fig.10", handoff)
            self.assertIn("fig10_true_module_rerun_results.csv", handoff)
            self.assertIn("fig10_completed_blinded_preferences.csv", handoff)
            self.assertIn("fig10_completed_blinded_preferences_template.csv", handoff)
            self.assertIn("fig10_completed_blinded_preferences_evaluator_1.csv", handoff)
            self.assertIn("fig10_completed_blinded_preferences_evaluator_2.csv", handoff)
            self.assertIn("fig10_completed_blinded_preferences_evaluator_3.csv", handoff)
            self.assertIn("evaluator_type/preference_source provenance", handoff)
            layout_audit_path = out_dir / "fig1_fig10_layout_readability_audit.csv"
            self.assertTrue(layout_audit_path.exists())
            layout_audit = layout_audit_path.read_text(encoding="utf-8")
            self.assertIn("Fig.5", layout_audit)
            self.assertIn("redraw_handoff_ready", layout_audit)
            self.assertNotIn("needs_layout_redesign", layout_audit)
            layout_df = pd.read_csv(layout_audit_path)
            self.assertEqual(0, int(layout_df["layout_redesign_needed"].sum()))
            layout_status = dict(zip(layout_df["figure"], layout_df["reading_pass_status"]))
            self.assertEqual("ready", layout_status["Fig.6"])
            self.assertEqual("ready_with_caveat", layout_status["Fig.7"])
            self.assertEqual("ready", layout_status["Fig.9"])
            self.assertEqual("ready_with_caveat", layout_status["Fig.10"])
            packet_index_path = out_dir / "fig1_fig10_external_evidence_packet_index.csv"
            self.assertTrue(packet_index_path.exists())
            packet_index = pd.read_csv(packet_index_path)
            self.assertTrue(
                {
                    "figure",
                    "artifact_path",
                    "artifact_role",
                    "recipient",
                    "blinded",
                    "contains_answer_key_or_unblinded_mapping",
                    "required_for_strict_gate",
                    "exists",
                }.issubset(packet_index.columns)
            )
            fig4_template = packet_index[
                packet_index["artifact_path"].eq("outputs/kg_perturbation_fig4_full50/fig4_completed_blinded_labels_template.csv")
            ].iloc[0]
            self.assertEqual("return_template", fig4_template["artifact_role"])
            self.assertEqual("human_labeler", fig4_template["recipient"])
            self.assertEqual(1, int(fig4_template["blinded"]))
            self.assertEqual(0, int(fig4_template["contains_answer_key_or_unblinded_mapping"]))
            fig4_labeler_1 = packet_index[
                packet_index["artifact_path"].eq("outputs/kg_perturbation_fig4_full50/fig4_completed_blinded_labels_labeler_1.csv")
            ].iloc[0]
            self.assertEqual("return_template", fig4_labeler_1["artifact_role"])
            self.assertEqual("human_labeler", fig4_labeler_1["recipient"])
            self.assertEqual(1, int(fig4_labeler_1["blinded"]))
            fig4_answer_key = packet_index[
                packet_index["artifact_path"].eq("outputs/kg_perturbation_fig4_full50/fig4_blinded_labeling_answer_key.csv")
            ].iloc[0]
            self.assertEqual("coordinator", fig4_answer_key["recipient"])
            self.assertEqual(1, int(fig4_answer_key["contains_answer_key_or_unblinded_mapping"]))
            fig9_output = packet_index[
                packet_index["artifact_path"].eq("outputs/kg_perturbation_fig9/fig9_aspr_qwen_output.json")
            ].iloc[0]
            self.assertEqual("checkpoint_runner", fig9_output["recipient"])
            self.assertEqual("required_return", fig9_output["artifact_role"])
            fig10_return = packet_index[
                packet_index["artifact_path"].eq("outputs/kg_perturbation_fig10/fig10_completed_blinded_preferences.csv")
            ].iloc[0]
            self.assertEqual("required_return", fig10_return["artifact_role"])
            self.assertEqual("human_preference_panel", fig10_return["recipient"])
            self.assertEqual(1, int(fig10_return["required_for_strict_gate"]))
            distribution_manifest_path = out_dir / "fig1_fig10_external_evidence_distribution_manifest.csv"
            self.assertTrue(distribution_manifest_path.exists())
            distribution_manifest = pd.read_csv(distribution_manifest_path)
            self.assertTrue(
                {
                    "package_name",
                    "recipient",
                    "package_path",
                    "file_count",
                    "contains_answer_key_or_unblinded_mapping",
                    "zip_bytes",
                    "zip_sha256",
                }.issubset(distribution_manifest.columns)
            )
            fig4_zip = out_dir / "external_evidence_distribution" / "fig4_human_labeler_packet.zip"
            fig10_zip = out_dir / "external_evidence_distribution" / "fig10_human_preference_panel_packet.zip"
            coordinator_zip = out_dir / "external_evidence_distribution" / "coordinator_private_evidence_packet.zip"
            self.assertTrue(fig4_zip.exists())
            self.assertTrue(fig10_zip.exists())
            self.assertTrue(coordinator_zip.exists())
            for _, package_row in distribution_manifest.iterrows():
                package_path = out_dir / str(package_row["package_path"])
                self.assertTrue(package_path.exists())
                package_bytes = package_path.read_bytes()
                self.assertEqual(len(package_bytes), int(package_row["zip_bytes"]))
                self.assertEqual(hashlib.sha256(package_bytes).hexdigest(), package_row["zip_sha256"])
            with zipfile.ZipFile(fig4_zip) as archive:
                fig4_names = archive.namelist()
                fig4_readme = archive.read("README.md").decode("utf-8")
            self.assertTrue(any(name.endswith("fig4_blinded_labeling_packet.csv") for name in fig4_names))
            self.assertTrue(any(name.endswith("fig4_completed_blinded_labels_template.csv") for name in fig4_names))
            self.assertTrue(any(name.endswith("fig4_completed_blinded_labels_labeler_1.csv") for name in fig4_names))
            self.assertTrue(any(name.endswith("fig4_completed_blinded_labels_labeler_2.csv") for name in fig4_names))
            self.assertTrue(any(name.endswith("fig4_completed_blinded_labels_labeler_3.csv") for name in fig4_names))
            self.assertFalse(any("answer_key" in name or "replacement_manifest" in name for name in fig4_names))
            self.assertIn("fig4_completed_blinded_labels.csv", fig4_readme)
            self.assertIn("three labeler-specific templates", fig4_readme)
            self.assertIn("Do not use LLM", fig4_readme)
            self.assertIn("Do not request or inspect answer keys", fig4_readme)
            with zipfile.ZipFile(fig10_zip) as archive:
                fig10_names = archive.namelist()
                fig10_readme = archive.read("README.md").decode("utf-8")
            self.assertTrue(any(name.endswith("fig10_blinded_preference_packet.csv") for name in fig10_names))
            self.assertTrue(any(name.endswith("fig10_completed_blinded_preferences_template.csv") for name in fig10_names))
            self.assertTrue(any(name.endswith("fig10_completed_blinded_preferences_evaluator_1.csv") for name in fig10_names))
            self.assertTrue(any(name.endswith("fig10_completed_blinded_preferences_evaluator_2.csv") for name in fig10_names))
            self.assertTrue(any(name.endswith("fig10_completed_blinded_preferences_evaluator_3.csv") for name in fig10_names))
            self.assertFalse(any("answer_key" in name for name in fig10_names))
            self.assertIn("fig10_completed_blinded_preferences.csv", fig10_readme)
            self.assertIn("750 blinded judgements", fig10_readme)
            self.assertIn("Do not use LLM", fig10_readme)
            self.assertIn("Do not request or inspect answer keys", fig10_readme)
            with zipfile.ZipFile(coordinator_zip) as archive:
                coordinator_names = archive.namelist()
            self.assertTrue(any("fig4_blinded_labeling_answer_key.csv" in name for name in coordinator_names))
            self.assertTrue(any("fig10_blinded_preference_answer_key.csv" in name for name in coordinator_names))
            readiness_path = out_dir / "fig1_fig10_submission_readiness.json"
            self.assertTrue(readiness_path.exists())
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            self.assertEqual(1, readiness["main_claim_ready"])
            self.assertEqual(0, readiness["strict_all_figures_ready"])
            self.assertEqual(0, readiness["strict_external_evidence_ready"])
            self.assertEqual(["Fig.4", "Fig.10"], readiness["strict_failed_figures"])
            self.assertIn("Fig.10", readiness["strict_evidence_missing_figures"])
            self.assertEqual(1, result["quality_gates"]["checks"]["main_claim_nature_check_pass"])
            self.assertEqual(0, result["quality_gates"]["checks"]["strict_all_figures_nature_check_pass"])
            self.assertEqual(0, result["quality_gates"]["checks"]["strict_external_evidence_check_pass"])
            self.assertEqual(1, result["quality_gates"]["checks"]["external_evidence_distribution_checksums_written"])
            self.assertEqual("main_claim_ready_with_strict_evidence_gaps", result["quality_gates"]["submission_status"])

    def test_checkpoint_ready_fig9_removes_assumed_storyboard_language(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_final_fig9_ready_") as tmp:
            root = Path(tmp)
            fig9_dir = root / "outputs" / "kg_perturbation_fig9"
            fig10_dir = root / "outputs" / "kg_perturbation_fig10"
            fig9_dir.mkdir(parents=True)
            fig10_dir.mkdir(parents=True)
            (fig9_dir / "fig9_quality_report.json").write_text(
                json.dumps(
                    {
                        "aspr_qwen_boundary": "checkpoint-generated ASPR-Qwen output with saved model metadata",
                        "replacement_gate": "checkpoint output and metadata are present for this case",
                        "notes": ["ASPR-Qwen output is checkpoint-generated and metadata-complete."],
                    }
                ),
                encoding="utf-8",
            )
            (fig10_dir / "figure_quality_report.json").write_text(
                json.dumps({"quality_gates": {"nature_strong_claim_ready": 0}}),
                encoding="utf-8",
            )
            (fig10_dir / "fig10_generic_llm_same_rubric_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "observed_generic_llm_run_same_rubric",
                        "case_count": 50,
                        "expected_case_count": 50,
                        "evaluable_case_count": 50,
                        "excluded_case_count": 0,
                        "match_count": 298,
                    }
                ),
                encoding="utf-8",
            )

            captions = "\n".join(build_final_assembly(root / "assembly", project_root=root)["captions"].values())
            gaps = build_gap_list(root)
            audit = (root / "assembly" / "fig1_fig10_cross_figure_audit.csv").read_text(encoding="utf-8")

        self.assertIn("checkpoint-generated ASPR-Qwen", captions)
        self.assertNotIn("assumed ASPR-Qwen", captions)
        self.assertNotIn("not representative checkpoint performance", captions)
        self.assertNotIn("Fig.9", gaps["figure"].tolist())
        self.assertNotIn("assumed ASPR-Qwen lane", audit)
        self.assertIn("checkpoint-generated ASPR-Qwen lane", audit)
        fig10_gap = gaps.loc[gaps["figure"].eq("Fig.10"), "gap"].iloc[0]
        self.assertNotIn("ASPR-Qwen checkpoint outputs", fig10_gap)

    def test_fig10_same_rubric_note_uses_manifest_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_fig10_manifest_") as tmp:
            root = Path(tmp)
            manifest = root / "outputs" / "kg_perturbation_fig10" / "fig10_generic_llm_same_rubric_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                (
                    "{\n"
                    '  "status": "observed_generic_llm_run_same_rubric",\n'
                    '  "case_count": 50,\n'
                    '  "expected_case_count": 50,\n'
                    '  "evaluable_case_count": 50,\n'
                    '  "excluded_case_count": 0,\n'
                    '  "match_count": 299\n'
                    "}\n"
                ),
                encoding="utf-8",
            )
            summary = root / "outputs" / "kg_perturbation_fig10" / "fig10_generic_llm_same_rubric_summary.csv"
            summary.write_text(
                "\n".join(
                    [
                        "metric,mean,std,count",
                        "semantic_agreement,0.225,0.13,50",
                        "prior_art_accuracy,0.23,0.25,50",
                        "unsupported_claim_rate,0.312,0.12,50",
                    ]
                ),
                encoding="utf-8",
            )

            note = fig10_same_rubric_note(root)

        self.assertIn("50/50", note)
        self.assertIn("299", note)
        self.assertIn("semantic agreement mean=0.225", note)
        self.assertIn("prior-art accuracy mean=0.230", note)
        self.assertNotIn("48/50", note)
        self.assertNotIn("near-zero", note)
        self.assertNotIn("zero-peer-point exclusions", note)

    def test_gap_list_is_derived_from_current_quality_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_gap_root_") as tmp:
            root = Path(tmp)
            (root / "outputs" / "kg_perturbation_fig4_full50").mkdir(parents=True)
            (root / "outputs" / "kg_perturbation_fig5").mkdir(parents=True)
            (root / "outputs" / "kg_perturbation_fig6").mkdir(parents=True)
            (root / "outputs" / "kg_perturbation_fig7").mkdir(parents=True)
            (root / "outputs" / "kg_perturbation_fig9").mkdir(parents=True)
            (root / "outputs" / "kg_perturbation_fig10").mkdir(parents=True)

            (root / "outputs" / "kg_perturbation_fig4_full50" / "figure_quality_report.json").write_text(
                json.dumps(
                    {
                        "overall_pass": False,
                        "global_score_coverage_audit": {
                            "additional_fixed_cases_needed": {"low": 10, "middle": 10, "high": 0}
                        },
                        "external_validation_replacement_manifest": {
                            "manifest_path": "outputs/kg_perturbation_fig4_full50/fig4_external_validation_replacement_manifest.csv",
                            "additional_ready_labels_needed": {"low": 10, "middle": 10, "high": 10},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig5" / "figure_quality_report.json").write_text(
                json.dumps(
                    {
                        "overall_pass": True,
                        "quality_gates": {
                            "checks": {
                                "backtest_table_present": 1,
                                "mean_precision_delta_nonnegative": 0,
                                "mean_ndcg_delta_positive": 0,
                            },
                            "mean_precision_at_10": 0.175,
                            "mean_baseline_precision_at_10": 0.367,
                            "mean_ndcg_at_10": 0.176,
                            "mean_baseline_ndcg_at_10": 0.364,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig5" / "fig5_alignment_metrics.csv").write_text(
                "metric_group,metric,value,denominator\n"
                "retrospective_backtest,precision_at_10,0.175,12\n"
                "retrospective_backtest,baseline_precision_at_10,0.367,3\n"
                "retrospective_backtest,ndcg_at_10,0.176,12\n"
                "retrospective_backtest,baseline_ndcg_at_10,0.364,3\n",
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig6" / "figure_quality_report.json").write_text(
                json.dumps({"quality_gates": {"nature_strong_claim_ready": 1}}),
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig7" / "figure_quality_report.json").write_text(
                json.dumps(
                    {
                        "quality_gates": {
                            "headline_point_estimate_supported": True,
                            "strict_claim_supported": False,
                            "checks": {"nature_rank": 1, "strict_interval_separation": 0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig9" / "fig9_quality_report.json").write_text(
                json.dumps({"aspr_qwen_boundary": "assumed pipeline-ready placeholder"}),
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig10" / "figure_quality_report.json").write_text(
                json.dumps({"quality_gates": {"nature_strong_claim_ready": 0}}),
                encoding="utf-8",
            )
            (root / "outputs" / "kg_perturbation_fig10" / "fig10_generic_llm_same_rubric_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "observed_generic_llm_run_same_rubric",
                        "case_count": 50,
                        "expected_case_count": 50,
                        "evaluable_case_count": 50,
                        "excluded_case_count": 0,
                        "match_count": 299,
                    }
                ),
                encoding="utf-8",
            )

            gaps = build_gap_list(root)

        self.assertEqual(["Fig.4", "Fig.5", "Fig.7", "Fig.9", "Fig.10"], gaps["figure"].tolist())
        self.assertNotIn("Fig.6", set(gaps["figure"]))
        fig4 = gaps.loc[gaps["figure"].eq("Fig.4")].iloc[0]
        self.assertIn("replacement manifest", fig4["next_replacement"])
        self.assertIn("low: 10, middle: 10, high: 10", fig4["next_replacement"])
        self.assertIn("precision@10 0.175 vs 0.367", gaps.loc[gaps["figure"].eq("Fig.5"), "gap"].iloc[0])
        fig7 = gaps.loc[gaps["figure"].eq("Fig.7")].iloc[0]
        self.assertEqual("claim-scope caveat", fig7["severity"])
        self.assertIn("point-estimate claim", fig7["gap"])


if __name__ == "__main__":
    unittest.main()

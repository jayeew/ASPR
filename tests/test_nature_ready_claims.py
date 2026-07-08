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

from experiments.nature_ready_checks import (  # noqa: E402
    _strict_fig4_blinded_labels_ready,
    _strict_fig10_blinded_preferences_ready,
    _strict_fig10_true_reruns_ready,
    build_claim_ledger,
    build_nature_check_report,
    build_strict_evidence_check_report,
    fig4_peer_review_alignment_status,
)


FIG10_VARIANTS = [
    "full ASPR",
    "no graph agent",
    "no ASPR-Qwen",
    "no prior-art retrieval",
    "no evidence trace",
    "no fusion",
    "no verifier",
    "generic LLM-only baseline",
]
FIG10_METRIC_COLUMNS = [
    "semantic_agreement",
    "novelty_coverage",
    "prior_art_accuracy",
    "factuality",
    "readability",
    "unsupported_claim_rate",
    "evidence_trace_completeness",
    "review_structure_coverage",
]


def write_valid_fig10_true_reruns(fig10_dir: Path, case_ids: list[str]) -> None:
    """Write a syntactically valid Fig.10 true-rerun artifact for tests."""
    true_rerun_rows = []
    for case_id in case_ids:
        for variant in FIG10_VARIANTS:
            variant_slug = variant.replace(" ", "_").replace("/", "_")
            review_path = Path("true_reruns") / case_id / variant_slug / "review.txt"
            evidence_path = Path("true_reruns") / case_id / variant_slug / "evidence_trace.json"
            (fig10_dir / review_path).parent.mkdir(parents=True, exist_ok=True)
            (fig10_dir / review_path).write_text("review text\n", encoding="utf-8")
            (fig10_dir / evidence_path).write_text("{}\n", encoding="utf-8")
            row = {
                "case_id": case_id,
                "variant": variant,
                "source": "true_disabled_module_rerun",
                "run_status": "ok",
                "review_text_path": str(review_path),
                "evidence_trace_path": str(evidence_path),
                "runtime_seconds": 1.0,
                "failure_reason": "",
            }
            for metric in FIG10_METRIC_COLUMNS:
                row[metric] = 0.1 if metric == "unsupported_claim_rate" else 0.8
            true_rerun_rows.append(row)
    true_rerun = pd.DataFrame(true_rerun_rows)
    true_rerun.to_csv(fig10_dir / "fig10_true_module_rerun_results.csv", index=False)
    true_rerun.head(0).to_csv(fig10_dir / "fig10_true_module_rerun_results_template.csv", index=False)
    true_rerun.head(0).to_csv(fig10_dir / "fig10_true_module_rerun_contract.csv", index=False)


def write_fig10_completed_preferences(
    fig10_dir: Path,
    preferred_system: str,
    *,
    evaluator_type: str = "blinded human",
    preference_source: str = "external_blinded_human_panel",
) -> None:
    """Write a complete 50-case blinded preference packet and sidecar."""
    dimensions = ["novelty", "prior_art", "evidence_grounding", "usefulness", "factuality"]
    packet_rows = []
    key_rows = []
    preference_rows = []
    for case_idx in range(50):
        blinded_id = f"F10P-{case_idx + 1:03d}"
        packet_rows.append({"blinded_case_id": blinded_id, "title": f"Case {case_idx + 1}"})
        key_rows.append(
            {
                "blinded_case_id": blinded_id,
                "paper_id": f"case-{case_idx + 1:02d}",
                "system_a": "full ASPR",
                "system_b": "generic LLM-only baseline",
                "comparison": "full ASPR vs generic LLM-only baseline",
            }
        )
        for dimension in dimensions:
            for evaluator_id in ["evaluator_1", "evaluator_2", "evaluator_3"]:
                preference_rows.append(
                    {
                        "comparison": "full ASPR vs generic LLM-only baseline",
                        "blinded_case_id": blinded_id,
                        "dimension": dimension,
                        "evaluator_id": evaluator_id,
                        "blind_setting": "system_names_hidden",
                                "preferred_system": preferred_system,
                                "evaluator_type": evaluator_type,
                                "preference_source": preference_source,
                                "rationale": "valid test preference",
                            }
                        )
    pd.DataFrame(packet_rows).to_csv(fig10_dir / "fig10_blinded_preference_packet.csv", index=False)
    pd.DataFrame(key_rows).to_csv(fig10_dir / "fig10_blinded_preference_answer_key.csv", index=False)
    preference = pd.DataFrame(preference_rows)
    preference.to_csv(fig10_dir / "fig10_completed_blinded_preferences_template.csv", index=False)
    for evaluator_id, evaluator_rows in preference.groupby("evaluator_id", sort=True):
        evaluator_rows.to_csv(
            fig10_dir / f"fig10_completed_blinded_preferences_{evaluator_id}.csv",
            index=False,
        )
    preference.to_csv(fig10_dir / "fig10_completed_blinded_preferences.csv", index=False)
    (fig10_dir / "fig10_blinded_preference_protocol.md").write_text("protocol\n", encoding="utf-8")


class NatureReadyClaimTests(unittest.TestCase):
    def test_claim_ledger_contains_forced_downgrade_rules(self) -> None:
        ledger = build_claim_ledger(PROJECT_ROOT)

        required = {
            "figure",
            "claim_id",
            "main_text_role",
            "required_gate",
            "current_status",
            "allowed_claim",
            "forbidden_claim",
            "quality_gate_path",
            "required_artifacts",
            "main_or_extended_data",
            "required_action",
        }
        self.assertTrue(required.issubset(set(ledger.columns)))
        self.assertEqual(10, ledger["figure"].nunique())
        publication_claim_text = "\n".join(
            ledger[["allowed_claim", "forbidden_claim", "required_action"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )
        self.assertNotIn("human-like", publication_claim_text.lower())

        main_blockers = set(ledger.loc[ledger["main_or_extended_data"].eq("main"), "figure"])
        self.assertEqual({"Fig.1", "Fig.2", "Fig.3", "Fig.6"}, main_blockers)

        fig2 = ledger[ledger["figure"].eq("Fig.2")].iloc[0]
        self.assertIn("multi-domain empirical validation", fig2["allowed_claim"])
        self.assertNotIn("diagnostic association", fig2["allowed_claim"])

        fig3 = ledger[ledger["figure"].eq("Fig.3")].iloc[0]
        self.assertIn("no-leakage", fig3["allowed_claim"])
        self.assertIn("temporal holdout", fig3["allowed_claim"])
        self.assertIn("leave-domain-out", fig3["allowed_claim"])
        self.assertNotIn("diagnostic corpus", fig3["allowed_claim"])

        fig4 = ledger[ledger["figure"].eq("Fig.4")].iloc[0]
        self.assertEqual("extended", fig4["main_or_extended_data"])
        self.assertIn("range-restricted", fig4["allowed_claim"])
        self.assertIn("global external validation", fig4["forbidden_claim"])

        fig6 = ledger[ledger["figure"].eq("Fig.6")].iloc[0]
        self.assertIn("full graph-rerun robustness", fig6["allowed_claim"])
        self.assertEqual("fig6_full_rerun_robustness", fig6["required_gate"])

        fig5 = ledger[ledger["figure"].eq("Fig.5")].iloc[0]
        self.assertEqual("source-backed AI frontier handoff", fig5["main_text_role"])
        self.assertIn("2024-2026 AI/AI-enabled science frontier", fig5["allowed_claim"])
        self.assertIn("unverified buzzwords", fig5["forbidden_claim"])

        fig7 = ledger[ledger["figure"].eq("Fig.7")].iloc[0]
        self.assertEqual(1, int(fig7["quality_gate_pass"]))
        self.assertIn("point-estimate", fig7["main_text_role"])
        self.assertIn("strict dominance remains unsupported", fig7["required_action"])

        fig10 = ledger[ledger["figure"].eq("Fig.10")].iloc[0]
        self.assertEqual("extended", fig10["main_or_extended_data"])
        self.assertIn("pipeline audit", fig10["main_text_role"])
        self.assertIn("completed causal module reruns", fig10["forbidden_claim"])

        fig9 = ledger[ledger["figure"].eq("Fig.9")].iloc[0]
        self.assertEqual(1, int(fig9["quality_gate_pass"]))
        self.assertIn("checkpoint-generated", fig9["current_status"])
        self.assertIn("checkpoint-generated ASPR-Qwen", fig9["allowed_claim"])
        self.assertNotIn("assumed", fig9["allowed_claim"].lower())
        self.assertNotIn("until checkpoint output is saved", fig9["forbidden_claim"])

    def test_source_controlled_figure_docs_avoid_human_like_performance_wording(self) -> None:
        doc_paths = [
            PROJECT_ROOT / "experiments" / "kg_perturbation_fig8" / "README.md",
            PROJECT_ROOT / "experiments" / "kg_perturbation_fig9" / "README.md",
        ]
        docs = "\n".join(path.read_text(encoding="utf-8") for path in doc_paths)

        self.assertNotIn("human-like", docs.lower())

    def test_nature_check_report_flags_current_package_not_strong_ready(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_check_") as tmp:
            out_dir = Path(tmp)
            report = build_nature_check_report(PROJECT_ROOT, out_dir)

            self.assertTrue(report["overall_pass"])
            self.assertTrue((out_dir / "fig1_fig10_claim_ledger.csv").exists())
            self.assertTrue((out_dir / "fig1_fig10_nature_check_summary.csv").exists())
            failures = {row["check_id"] for row in report["checks"] if not row["passed"]}
            self.assertNotIn("fig1_fig3_strong_gates", failures)
            self.assertNotIn("fig4_external_validation", failures)
            self.assertNotIn("fig6_full_rerun_robustness", failures)
            self.assertIn("fig10_replacement_gates", failures)

    def test_fig1_fig3_check_reports_only_current_failed_subgates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_check_") as tmp:
            report = build_nature_check_report(PROJECT_ROOT, Path(tmp))

        checks = {row["check_id"]: row for row in report["checks"]}
        action = checks["fig1_fig3_strong_gates"]["required_action"]
        self.assertTrue(checks["fig1_fig3_strong_gates"]["passed"])
        self.assertIn("Fig.1-Fig.3 gates pass", action)
        self.assertNotIn("Fig.2 sample/control/closure", action)

    def test_nature_check_marks_extended_data_failures_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_check_") as tmp:
            report = build_nature_check_report(PROJECT_ROOT, Path(tmp))

        checks = {row["check_id"]: row for row in report["checks"]}
        self.assertFalse(checks["fig10_replacement_gates"]["passed"])
        self.assertEqual("extended_data_nonblocking", checks["fig10_replacement_gates"]["blocking_scope"])
        self.assertEqual("extended_data_nonblocking", checks["fig4_external_validation"]["blocking_scope"])
        self.assertTrue(checks["fig4_external_validation"]["passed"])

    def test_strict_all_figures_check_blocks_on_extended_data_replacement_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_all_figure_check_") as tmp:
            report = build_nature_check_report(PROJECT_ROOT, Path(tmp), require_all_figures=True)

        self.assertFalse(report["overall_pass"])
        self.assertEqual("all_figures_need_revision_before_nature_submission", report["status_label"])
        checks = {row["check_id"]: row for row in report["checks"]}
        self.assertIn("all_figures_claim_gates", checks)
        self.assertFalse(checks["all_figures_claim_gates"]["passed"])
        self.assertIn("Fig.4", checks["all_figures_claim_gates"]["required_action"])
        self.assertIn("Fig.10", checks["all_figures_claim_gates"]["required_action"])
        self.assertNotIn("Fig.9", checks["all_figures_claim_gates"]["required_action"])

    def test_strict_all_figures_report_does_not_overwrite_main_claim_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_all_figure_check_") as tmp:
            out_dir = Path(tmp)
            build_nature_check_report(PROJECT_ROOT, out_dir)
            build_nature_check_report(PROJECT_ROOT, out_dir, require_all_figures=True)
            main_report = json.loads((out_dir / "fig1_fig10_nature_check_report.json").read_text(encoding="utf-8"))
            strict_report = json.loads(
                (out_dir / "fig1_fig10_all_figures_nature_check_report.json").read_text(encoding="utf-8")
            )

        self.assertFalse(main_report["require_all_figures"])
        self.assertEqual("nature_ready", main_report["status_label"])
        self.assertTrue(strict_report["require_all_figures"])
        self.assertEqual("all_figures_need_revision_before_nature_submission", strict_report["status_label"])

    def test_strict_evidence_check_reports_current_missing_external_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_evidence_") as tmp:
            out_dir = Path(tmp)
            report = build_strict_evidence_check_report(PROJECT_ROOT, out_dir)

            self.assertFalse(report["overall_pass"])
            self.assertEqual("strict_external_evidence_missing", report["status_label"])
            self.assertTrue((out_dir / "fig1_fig10_strict_evidence_check_summary.csv").exists())
            self.assertTrue((out_dir / "fig1_fig10_strict_evidence_check_report.json").exists())
            failed = [row for row in report["checks"] if not row["passed"]]
            self.assertGreaterEqual(len(failed), 1)
            failed_artifacts = " ".join(row["required_submission_artifact"] for row in failed)
            self.assertIn("fig4_completed_blinded_labels.csv", failed_artifacts)
            self.assertNotIn("fig9_checkpoint_metadata.json", failed_artifacts)
            self.assertNotIn("fig10_true_module_rerun_results.csv", failed_artifacts)
            self.assertIn("fig10_completed_blinded_preferences.csv", failed_artifacts)
            fig9_rows = [row for row in report["checks"] if row["figure"] == "Fig.9"]
            self.assertEqual(1, len(fig9_rows))
            self.assertTrue(fig9_rows[0]["passed"])
            self.assertNotIn("assumed placeholder", fig9_rows[0]["blocker"])
            self.assertIn("checkpoint-generated", fig9_rows[0]["blocker"])
            fig10_true_rerun_rows = [
                row
                for row in report["checks"]
                if row["figure"] == "Fig.10"
                and "fig10_true_module_rerun_results.csv" in row["required_submission_artifact"]
            ]
            self.assertEqual(1, len(fig10_true_rerun_rows))
            self.assertTrue(fig10_true_rerun_rows[0]["passed"])

    def test_strict_evidence_check_passes_when_required_artifacts_and_templates_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_evidence_root_") as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "kg_perturbation_final_assembly"
            fig4_dir = root / "outputs" / "kg_perturbation_fig4_full50"
            fig9_dir = root / "outputs" / "kg_perturbation_fig9"
            fig10_dir = root / "outputs" / "kg_perturbation_fig10"
            fig4_dir.mkdir(parents=True)
            fig9_dir.mkdir(parents=True)
            fig10_dir.mkdir(parents=True)

            fig4_packet_rows = []
            fig4_key_rows = []
            fig4_label_rows = []
            tiers = ["low", "middle", "high"]
            for idx in range(30):
                blinded_id = f"F4LV-{idx + 1:04d}"
                tier = tiers[idx // 10]
                fig4_packet_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "assignment_role": "primary_validation_labeling_sample",
                        "title": f"Paper {idx + 1}",
                    }
                )
                fig4_key_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "paper_id": f"paper-{idx + 1}",
                        "global_fig3_tier": tier,
                        "fig3_score_for_validation": {"low": 0.1, "middle": 0.55, "high": 0.95}[tier],
                        "fig3_global_percentile": {"low": 0.25, "middle": 0.65, "high": 0.95}[tier],
                        "assignment_role": "primary_validation_labeling_sample",
                    }
                )
                fig4_label_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "label_novelty_1_5": {"low": 2, "middle": 3, "high": 5}[tier],
                        "label_significance_1_5": {"low": 2, "middle": 4, "high": 5}[tier],
                        "label_prior_art_1_5": 3,
                        "label_confidence_1_5": 5,
                        "label_source": "blinded_human",
                        "labeler_id": "reviewer_a",
                        "label_notes": "valid test label",
                    }
                )
            pd.DataFrame(fig4_packet_rows).to_csv(fig4_dir / "fig4_blinded_labeling_packet.csv", index=False)
            pd.DataFrame(fig4_packet_rows).to_csv(fig4_dir / "fig4_completed_blinded_labels_template.csv", index=False)
            for labeler_id in ["labeler_1", "labeler_2", "labeler_3"]:
                labeler_rows = []
                for row in fig4_label_rows:
                    labeler_row = dict(row)
                    labeler_row["labeler_id"] = labeler_id
                    labeler_rows.append(labeler_row)
                pd.DataFrame(labeler_rows).to_csv(
                    fig4_dir / f"fig4_completed_blinded_labels_{labeler_id}.csv",
                    index=False,
                )
            (fig4_dir / "fig4_blinded_labeling_protocol.md").write_text(
                "Fig.4 blinded labelers must not see answer keys or Fig.3 scores.",
                encoding="utf-8",
            )
            pd.DataFrame(fig4_key_rows).to_csv(fig4_dir / "fig4_blinded_labeling_answer_key.csv", index=False)
            pd.DataFrame(fig4_label_rows).to_csv(fig4_dir / "fig4_completed_blinded_labels.csv", index=False)
            pd.DataFrame(fig4_key_rows).to_csv(fig4_dir / "fig4_external_validation_replacement_manifest.csv", index=False)
            pd.DataFrame({"paper_id": [f"case-{idx + 1:02d}" for idx in range(50)]}).to_csv(
                fig4_dir / "fig4_metrics_summary.csv",
                index=False,
            )

            checkpoint_metadata = {
                "model_hash": "sha256:valid",
                "training_config": {"epochs": 1},
                "data_version": "test-v1",
                "prompt": "review this paper",
                "decoding_config": {"temperature": 0.1},
                "seed": 20260630,
                "runtime_seconds": 12.5,
            }
            checkpoint_output = {
                "case_id": "s41467-023-35844-2",
                "output_origin": "checkpoint_generated_aspr_qwen_output",
                "checkpoint_invoked": True,
                "summary_judgement": "checkpoint-generated review summary",
                "major_strengths": ["strength"],
                "major_concerns": ["concern"],
                "checkpoint_metadata": checkpoint_metadata,
            }
            (fig9_dir / "fig9_aspr_qwen_output.json").write_text(json.dumps(checkpoint_output), encoding="utf-8")
            (fig9_dir / "fig9_checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata), encoding="utf-8")
            (fig9_dir / "fig9_checkpoint_metadata_template.json").write_text(json.dumps(checkpoint_metadata), encoding="utf-8")
            (fig9_dir / "fig9_checkpoint_run_contract.json").write_text(json.dumps({"required_metadata_keys": sorted(checkpoint_metadata)}), encoding="utf-8")

            write_valid_fig10_true_reruns(fig10_dir, [f"case-{idx + 1:02d}" for idx in range(50)])

            write_fig10_completed_preferences(fig10_dir, preferred_system="system_a")

            report = build_strict_evidence_check_report(root, out_dir)

            self.assertTrue(report["overall_pass"])
            self.assertEqual("strict_external_evidence_ready", report["status_label"])
            self.assertEqual(4, len(report["checks"]))
            self.assertTrue(all(row["passed"] for row in report["checks"]))

    def test_strict_evidence_check_rejects_placeholder_external_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_evidence_bad_") as tmp:
            root = Path(tmp)
            out_dir = root / "outputs" / "kg_perturbation_final_assembly"
            for rel_path in [
                "outputs/kg_perturbation_fig4_full50/fig4_completed_blinded_labels.csv",
                "outputs/kg_perturbation_fig4_full50/fig4_blinded_labeling_packet.csv",
                "outputs/kg_perturbation_fig4_full50/fig4_completed_blinded_labels_template.csv",
                "outputs/kg_perturbation_fig4_full50/fig4_blinded_labeling_answer_key.csv",
                "outputs/kg_perturbation_fig4_full50/fig4_external_validation_replacement_manifest.csv",
                "outputs/kg_perturbation_fig9/fig9_aspr_qwen_output.json",
                "outputs/kg_perturbation_fig9/fig9_checkpoint_metadata.json",
                "outputs/kg_perturbation_fig9/fig9_checkpoint_metadata_template.json",
                "outputs/kg_perturbation_fig9/fig9_checkpoint_run_contract.json",
                "outputs/kg_perturbation_fig10/fig10_true_module_rerun_results.csv",
                "outputs/kg_perturbation_fig10/fig10_true_module_rerun_results_template.csv",
                "outputs/kg_perturbation_fig10/fig10_true_module_rerun_contract.csv",
                "outputs/kg_perturbation_fig10/fig10_completed_blinded_preferences.csv",
                "outputs/kg_perturbation_fig10/fig10_blinded_preference_packet.csv",
                "outputs/kg_perturbation_fig10/fig10_blinded_preference_answer_key.csv",
                "outputs/kg_perturbation_fig10/fig10_blinded_preference_protocol.md",
            ]:
                path = root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")

            report = build_strict_evidence_check_report(root, out_dir)

            self.assertFalse(report["overall_pass"])
            self.assertTrue(all(not row["passed"] for row in report["checks"]))
            statuses = {row["status"] for row in report["checks"]}
            self.assertIn("invalid_required_submission_artifact", statuses)

    def test_strict_fig4_rejects_complete_labels_without_positive_external_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_fig4_labels_") as tmp:
            root = Path(tmp)
            fig4_dir = root / "outputs" / "kg_perturbation_fig4_full50"
            fig4_dir.mkdir(parents=True)
            packet_rows = []
            key_rows = []
            label_rows = []
            tiers = ["low", "middle", "high"]
            for idx in range(30):
                blinded_id = f"F4LV-{idx + 1:04d}"
                tier = tiers[idx // 10]
                packet_rows.append({"blinded_case_id": blinded_id, "title": f"Paper {idx + 1}"})
                key_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "paper_id": f"paper-{idx + 1}",
                        "global_fig3_tier": tier,
                        "fig3_score_for_validation": {"low": 0.1, "middle": 0.55, "high": 0.95}[tier],
                        "fig3_global_percentile": {"low": 0.25, "middle": 0.65, "high": 0.95}[tier],
                        "assignment_role": "primary_validation_labeling_sample",
                    }
                )
                label_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "label_novelty_1_5": {"low": 5, "middle": 3, "high": 2}[tier],
                        "label_significance_1_5": {"low": 5, "middle": 3, "high": 2}[tier],
                        "label_prior_art_1_5": 3,
                        "label_confidence_1_5": 5,
                        "label_source": "blinded_human",
                        "labeler_id": "reviewer_a",
                    }
                )
            pd.DataFrame(packet_rows).to_csv(fig4_dir / "fig4_blinded_labeling_packet.csv", index=False)
            pd.DataFrame(key_rows).to_csv(fig4_dir / "fig4_blinded_labeling_answer_key.csv", index=False)
            pd.DataFrame(label_rows).to_csv(fig4_dir / "fig4_completed_blinded_labels.csv", index=False)
            pd.DataFrame(label_rows).to_csv(fig4_dir / "fig4_external_validation_replacement_manifest.csv", index=False)

            status = _strict_fig4_blinded_labels_ready(root)

            self.assertFalse(status["passed"])
            self.assertIn("blinded_external_validation_incomplete", status["detail"])

    def test_strict_fig4_rejects_nonhuman_or_synthetic_label_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_fig4_label_source_") as tmp:
            root = Path(tmp)
            fig4_dir = root / "outputs" / "kg_perturbation_fig4_full50"
            fig4_dir.mkdir(parents=True)
            packet_rows = []
            key_rows = []
            label_rows = []
            tiers = ["low", "middle", "high"]
            for idx in range(30):
                blinded_id = f"F4LV-{idx + 1:04d}"
                tier = tiers[idx // 10]
                packet_rows.append({"blinded_case_id": blinded_id, "title": f"Paper {idx + 1}"})
                key_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "paper_id": f"paper-{idx + 1}",
                        "global_fig3_tier": tier,
                        "fig3_score_for_validation": {"low": 0.1, "middle": 0.55, "high": 0.95}[tier],
                        "fig3_global_percentile": {"low": 0.25, "middle": 0.65, "high": 0.95}[tier],
                        "assignment_role": "primary_validation_labeling_sample",
                    }
                )
                label_rows.append(
                    {
                        "blinded_case_id": blinded_id,
                        "label_novelty_1_5": {"low": 2, "middle": 4, "high": 5}[tier],
                        "label_significance_1_5": {"low": 2, "middle": 4, "high": 5}[tier],
                        "label_prior_art_1_5": 3,
                        "label_confidence_1_5": 5,
                        "label_source": "llm_judge_synthetic",
                        "labeler_id": "model_rater_1",
                    }
                )
            pd.DataFrame(packet_rows).to_csv(fig4_dir / "fig4_blinded_labeling_packet.csv", index=False)
            pd.DataFrame(key_rows).to_csv(fig4_dir / "fig4_blinded_labeling_answer_key.csv", index=False)
            pd.DataFrame(label_rows).to_csv(fig4_dir / "fig4_completed_blinded_labels.csv", index=False)
            pd.DataFrame(label_rows).to_csv(fig4_dir / "fig4_external_validation_replacement_manifest.csv", index=False)

            status = _strict_fig4_blinded_labels_ready(root)

            self.assertFalse(status["passed"])
            self.assertIn("fig4_blinded_labels_nonhuman_or_synthetic_source", status["detail"])

    def test_strict_fig10_true_reruns_must_match_fig4_frozen_case_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_fig10_cases_") as tmp:
            root = Path(tmp)
            fig4_dir = root / "outputs" / "kg_perturbation_fig4_full50"
            fig10_dir = root / "outputs" / "kg_perturbation_fig10"
            fig4_dir.mkdir(parents=True)
            fig10_dir.mkdir(parents=True)
            pd.DataFrame({"paper_id": [f"expected-{idx + 1:02d}" for idx in range(50)]}).to_csv(
                fig4_dir / "fig4_metrics_summary.csv",
                index=False,
            )
            write_valid_fig10_true_reruns(fig10_dir, [f"other-{idx + 1:02d}" for idx in range(50)])

            status = _strict_fig10_true_reruns_ready(root)

            self.assertFalse(status["passed"])
            self.assertIn("missing_expected_cases", status["detail"])

    def test_strict_fig10_blinded_preferences_must_support_full_aspr_on_key_dimensions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_fig10_preference_") as tmp:
            root = Path(tmp)
            fig10_dir = root / "outputs" / "kg_perturbation_fig10"
            fig10_dir.mkdir(parents=True)
            write_fig10_completed_preferences(fig10_dir, preferred_system="system_b")

            status = _strict_fig10_blinded_preferences_ready(root)

            self.assertFalse(status["passed"])
            self.assertIn("full_aspr_preference_not_supported", status["detail"])

    def test_strict_fig10_blinded_preferences_require_human_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_strict_fig10_preference_source_") as tmp:
            root = Path(tmp)
            fig10_dir = root / "outputs" / "kg_perturbation_fig10"
            fig10_dir.mkdir(parents=True)
            write_fig10_completed_preferences(
                fig10_dir,
                preferred_system="system_a",
                evaluator_type="LLM-as-judge",
                preference_source="synthetic_model_panel",
            )

            status = _strict_fig10_blinded_preferences_ready(root)

            self.assertFalse(status["passed"])
            self.assertIn("fig10_blinded_preferences_nonhuman_or_synthetic_source", status["detail"])

    def test_fig4_status_honors_quality_report_when_metric_coverage_is_nonzero(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_fig4_status_") as tmp:
            root = Path(tmp)
            fig4_dir = root / "outputs" / "kg_perturbation_fig4_full50"
            fig4_dir.mkdir(parents=True)
            (fig4_dir / "fig4_metrics_summary.csv").write_text(
                "\n".join(
                    [
                        "soft_claim_recall,claim_evidence_coverage,covered_peer_aspects,missing_peer_point_rate",
                        "0.2,0.2,2,0.8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (fig4_dir / "figure_quality_report.json").write_text(
                json.dumps(
                    {
                        "overall_pass": False,
                        "status_label": "external_validation_blocked",
                        "checks": {
                            "soft_claim_recall_nonzero": 1,
                            "claim_evidence_coverage_nonzero": 1,
                            "covered_peer_aspects_nonzero": 1,
                            "missing_peer_point_rate_below_one": 1,
                            "fig3_peer_novelty_positive": 0,
                            "fig3_peer_significance_positive": 0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            status = fig4_peer_review_alignment_status(root)

        self.assertFalse(status["claim_ready"])
        self.assertEqual("external_validation_blocked", status["status"])
        self.assertIn("fig3_peer_novelty_positive", status["details"])
        self.assertIn("fig3_peer_significance_positive", status["details"])


if __name__ == "__main__":
    unittest.main()

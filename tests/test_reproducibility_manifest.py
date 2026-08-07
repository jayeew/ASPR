from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityManifestTests(unittest.TestCase):
    def test_makefile_exposes_current_and_nature_check_targets(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("figures-current:", text)
        self.assertIn("figures-main-nature:", text)
        self.assertIn("figures-extended:", text)
        self.assertIn("figures-nature-check:", text)
        self.assertIn("figures-evidence-packets:", text)
        self.assertIn("figures-strict-evidence-check:", text)
        self.assertIn("python3 -m experiments.nature_ready_checks", text)
        self.assertIn("--strict-evidence-check", text)
        self.assertIn("fig5_forecast_outcomes", text)

    def test_makefile_has_rerun_safe_targets(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("figures-main-nature:", text)
        self.assertIn("--reuse-audit", text)
        self.assertIn("--prefer-scored-candidate-pool", text)
        self.assertIn("FIG4_REUSE_RETRIEVAL_CACHE=1", text)
        self.assertIn("--refresh-invalid-agent-only", text)
        self.assertIn("FIG4_QUERY_KEYWORD_LIMIT ?= 3", text)
        self.assertIn("FIG4_LATS_MODEL ?= qwen3:8b", text)
        self.assertIn("FIG4_LATS_BASE_URL ?= http://localhost:11434/v1", text)
        self.assertIn("FIG4_AGENT_MAX_ITERATIONS ?= 0", text)
        self.assertIn("ASPR_LATS_CANDIDATES ?= 1", text)
        self.assertIn("ASPR_LATS_BEAM_WIDTH ?= 1", text)
        self.assertIn("ASPR_LATS_PROMPT_PREFIX ?= /no_think", text)
        self.assertIn("ASPR_LATS_SINGLE_PASS ?= 1", text)
        self.assertIn("ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL)", text)
        self.assertIn("ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL)", text)
        self.assertIn("ASPR_LATS_LLM_API_KEY=ollama", text)
        self.assertIn("ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX)", text)
        self.assertIn("ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS)", text)
        self.assertIn("FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS)", text)
        self.assertIn("ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE)", text)
        self.assertIn("$(FIG10_MAX_CASES)", text)
        self.assertIn("figures-extended:", text)

    def test_makefile_uses_fig2_strong_input_builder(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("build_fig2_strong_inputs.py", text)
        self.assertIn("FIG2_FUTURE_TAU ?= 8", text)
        self.assertIn("FIG2_MIN_REFS ?= 4", text)
        self.assertIn("FIG2_MAX_PAPERS ?= 8000", text)
        self.assertIn("--future-tau $(FIG2_FUTURE_TAU)", text)
        self.assertIn("--min-refs $(FIG2_MIN_REFS)", text)
        self.assertIn("--max-papers $(FIG2_MAX_PAPERS)", text)

    def test_makefile_runs_final_hgb_fig3(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("$(PYTHON) -m experiments.fig03.new.run --stage all", text)

    def test_makefile_runs_fig6_full_rerun_when_enabled(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("FIG6_BUILD_FULL_RERUN ?= 1", text)
        self.assertIn("FIG6_FULL_RERUN_MAX_PAPERS ?=", text)
        self.assertIn("FIG6_BUILD_FULL_RERUN=$(FIG6_BUILD_FULL_RERUN)", text)
        self.assertIn("FIG6_FULL_RERUN_MAX_PAPERS=$(FIG6_FULL_RERUN_MAX_PAPERS)", text)

    def test_makefile_refreshes_external_evidence_packets(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("figures-evidence-packets:", text)
        self.assertIn("fig4_completed_blinded_labels.csv", text)
        self.assertIn("fig9_checkpoint_run_contract.json", text)
        self.assertIn("fig10_true_module_rerun_contract.csv", text)
        self.assertIn("fig10_blinded_preference_packet.csv", text)
        self.assertIn("fig1_fig10_external_evidence_packet_index.csv", text)
        self.assertIn("build_final_assembly.py", text)

    def test_makefile_exposes_external_evidence_intake_target(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("figures-external-evidence-intake:", text)
        self.assertIn("fig4_completed_blinded_labels.csv", text)
        self.assertIn("fig4-merge-blinded-labels:", text)
        self.assertIn("merge_fig4_labeler_blinded_label_returns", text)
        self.assertIn("fig10_completed_blinded_preferences.csv", text)
        self.assertIn("fig10-merge-blinded-preferences:", text)
        self.assertIn("merge_fig10_evaluator_preference_returns", text)
        self.assertIn("experiments/fig04/old/main_fig4.py", text)
        self.assertIn("experiments/fig10/old/build_fig10_ablation.py", text)
        self.assertIn("experiments/common/old/final_assembly/build_final_assembly.py", text)
        self.assertIn("--strict-evidence-check", text)

    def test_makefile_exposes_fig9_checkpoint_rerun_target(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        self.assertTrue(makefile.exists())
        text = makefile.read_text(encoding="utf-8")
        self.assertIn("fig9-checkpoint-run:", text)
        self.assertIn("FIG9_CHECKPOINT_PATH ?=", text)
        self.assertIn("FIG9_SEED ?= 20260701", text)
        self.assertIn("run_fig9_checkpoint_inference.py", text)
        self.assertIn("--checkpoint-path $(FIG9_CHECKPOINT_PATH)", text)
        self.assertIn("build_fig9_case.py", text)
        self.assertIn("build_final_assembly.py", text)

    def test_requirements_and_availability_docs_exist(self) -> None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for package in ["networkx", "scipy", "scikit-learn", "pypdf", "pydantic", "matplotlib", "Pillow", "reportlab"]:
            self.assertIn(package, requirements)

        data_sources = (PROJECT_ROOT / "docs" / "data_sources.md").read_text(encoding="utf-8")
        self.assertIn("OpenAlex", data_sources)
        self.assertIn("Semantic Scholar", data_sources)
        self.assertIn("must not redistribute copyrighted full text", data_sources)

        code_availability = (PROJECT_ROOT / "docs" / "code_availability.md").read_text(encoding="utf-8")
        self.assertIn("figures-nature-check", code_availability)
        self.assertIn("figures-external-evidence-intake", code_availability)
        self.assertIn("Code Availability", code_availability)


if __name__ == "__main__":
    unittest.main()

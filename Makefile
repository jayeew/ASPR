.PHONY: figures-current figures-main-nature figures-extended fig4-claim-scope fig5-ai-frontier visual-redesign-handoff nature-iter-audit figures-evidence-packets figures-external-evidence-intake fig4-merge-blinded-labels fig9-checkpoint-run fig10-disabled-reruns fig10-merge-blinded-preferences figures-nature-check figures-all-nature-check figures-strict-evidence-check fig8-current final-assembly test-nature-ready

PYTHON ?= python3
MPLCONFIGDIR ?= /tmp/aspr_mplconfig
export MPLCONFIGDIR
FIG4_MARKDOWN_ROOT ?= /mnt/d/aspr_nature_markdown
FIG10_MODEL ?= qwen3:8b
FIG10_MAX_CASES ?= 0
FIG10_TIMEOUT ?= 120
FIG10_DISABLED_MAX_CASES ?= 0
FIG10_DISABLED_VARIANTS ?=
FIG4_QUERY_KEYWORD_LIMIT ?= 3
FIG4_OPENALEX_PER_PAGE ?= 10
FIG4_OPENALEX_FROM_YEAR ?= 2000
FIG4_LATS_MODEL ?= qwen3:8b
FIG4_LATS_BASE_URL ?= http://localhost:11434/v1
FIG4_AGENT_MAX_ITERATIONS ?= 0
ASPR_LATS_CANDIDATES ?= 1
ASPR_LATS_BEAM_WIDTH ?= 1
ASPR_LATS_MAX_TOKENS ?= 512
ASPR_LATS_PROMPT_PREFIX ?= /no_think
ASPR_LATS_SINGLE_PASS ?= 1
FIG2_FUTURE_TAU ?= 8
FIG2_MIN_REFS ?= 4
FIG2_MIN_CONTROLS ?= 10
FIG2_MAX_PAPERS ?= 8000
FIG2_REFERENCE_COUNT_BINS ?= 4
FIG6_BUILD_FULL_RERUN ?= 1
FIG6_FULL_RERUN_MAX_PAPERS ?= 300
FIG9_CHECKPOINT_PATH ?= /home/jayee/workspace/checkpoint/qwen-0.6b-review
FIG9_MAX_NEW_TOKENS ?= 512
FIG9_TEMPERATURE ?= 0.2
FIG9_TOP_P ?= 0.9
FIG9_SEED ?= 20260701
FIG9_MAX_INPUT_CHARS ?= 8000
NATURE_ITER_ROUND ?= 0
FIG5_AI_END_DATE ?= 2026-07-08
FIG5_AI_PER_QUERY ?= 30

figures-current: figures-main-nature fig5-ai-frontier figures-extended fig4-claim-scope visual-redesign-handoff final-assembly

nature-iter-audit:
	$(PYTHON) experiments/common/old/nature_iteration/build_nature_iteration.py --round $(NATURE_ITER_ROUND)

fig5-ai-frontier:
	$(PYTHON) experiments/fig05/old/build_fig5_ai_frontier.py \
		--local-papers outputs/fig05/old/work/kg_perturbation/plot_data/base/papers_master.csv \
		--out-dir outputs/fig05/old/work/kg_perturbation/ai_frontier \
		--start-date 2024-01-01 --end-date $(FIG5_AI_END_DATE) \
		--per-query $(FIG5_AI_PER_QUERY)

fig4-claim-scope:
	$(PYTHON) experiments/fig04/old/build_fig4_claim_scope.py \
		--fig4-dir outputs/fig04/old/work/full50

visual-redesign-handoff:
	$(PYTHON) experiments/common/old/final_assembly/build_visual_redesign_handoff.py \
		--out-dir outputs/common/old/final_assembly_work/visual_redesign_handoff

figures-main-nature:
	$(PYTHON) experiments/fig01/old/fig1_knowledge_perturbation.py \
		--config experiments/fig01/old/configs/v6a_display_crispr.yaml \
		         experiments/fig01/old/configs/v6a_display_graphene.yaml \
		         experiments/fig01/old/configs/v6a_display_ipsc.yaml \
		         experiments/fig01/old/configs/v6a_display_exoplanets.yaml \
		--out-dir outputs/fig01/old/work/redraw_v6a_best \
		--corpus-dir data/knowledge_corpus/v2_publication_v6a_locked_candidate
	$(PYTHON) experiments/fig02/old/build_fig2_strong_inputs.py \
		--source data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig2 \
		--out-dir outputs/fig02/old/work/redraw_v6a_best/fig2_strong_input \
		--pre-cutoff-max-year 2018 \
		--future-window-start 2019 \
		--future-window-end 2025 \
		--min-total-eligible 8000 \
		--min-controls $(FIG2_MIN_CONTROLS) \
		--reference-count-bins $(FIG2_REFERENCE_COUNT_BINS)
	$(PYTHON) experiments/fig02/old/fig2_empirical_panels.py \
		--data-dir outputs/fig02/old/work/redraw_v6a_best/fig2_strong_input \
		--out-dir outputs/fig02/old/work/redraw_v6a_best \
		--evidence-mode strong \
		--domains crispr,exoplanets,gamma_ray_bursts_and_supernovae,genetics_aging_and_longevity_in_model_organisms,graphene_2d_materials,ipsc_reprogramming,microbiome_metagenomics,perovskite_solar_cells,topological_insulators,ubiquitin_and_proteasome_pathways \
		--panel all --export-tables \
		--fig1-snapshot-dir outputs/fig01/old/work/redraw_v6a_best/crispr \
		--future-tau $(FIG2_FUTURE_TAU) \
		--min-refs $(FIG2_MIN_REFS) \
		--min-controls $(FIG2_MIN_CONTROLS) \
		--max-papers $(FIG2_MAX_PAPERS) \
		--quiet
	$(PYTHON) -m experiments.fig03.new.run --stage all
	FIG4_REUSE_RETRIEVAL_CACHE=1 FIG4_QUERY_KEYWORD_LIMIT=$(FIG4_QUERY_KEYWORD_LIMIT) ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE) ASPR_OPENALEX_FROM_YEAR=$(FIG4_OPENALEX_FROM_YEAR) ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL) ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL) ASPR_LATS_LLM_API_KEY=ollama FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS) ASPR_LATS_CANDIDATES=$(ASPR_LATS_CANDIDATES) ASPR_LATS_BEAM_WIDTH=$(ASPR_LATS_BEAM_WIDTH) ASPR_LATS_MAX_TOKENS=$(ASPR_LATS_MAX_TOKENS) ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX) ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS) $(PYTHON) experiments/fig04/old/main_fig4.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig04/old/work/full50 \
		--sample-size 50 --journal-scope all \
		--retrieval-provider openalex --judge-backend heuristic \
		--reuse-audit --refresh-invalid-agent-only \
		--prefer-scored-candidate-pool \
		--require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet
	$(PYTHON) -m experiments.fig05.old.fig5_forecast_outcomes \
		--fig3-run-dir outputs/fig03/old/work/redraw_v6a_best/multi_domain \
		--fig3-input-dir outputs/fig03/old/work/redraw_v6a_best/fig3_input/multi_domain \
		--out-dir outputs/fig05/old/work/kg_perturbation \
		--backtest-windows 2010:2015 2015:2020 2020:2025 \
		--formats png svg --quiet
	FIG6_BUILD_FULL_RERUN=$(FIG6_BUILD_FULL_RERUN) FIG6_FULL_RERUN_MAX_PAPERS=$(FIG6_FULL_RERUN_MAX_PAPERS) \
		$(PYTHON) experiments/fig06/old/build_fig6_robustness.py

figures-extended:
	$(PYTHON) experiments/fig07/old/build_fig7_venue_contribution.py \
		--fig3-run-dir outputs/fig03/old/work/redraw_v6a_best/multi_domain \
		--works-table data/knowledge_corpus/v2_publication_v6a_locked_candidate/works.csv \
		--out-dir outputs/fig07/old/work/kg_perturbation
	$(MAKE) fig8-current
	$(PYTHON) experiments/fig09/old/build_fig9_case.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig09/old/work/kg_perturbation
	$(PYTHON) experiments/fig10/old/build_fig10_generic_baseline.py \
		--fig4-metrics outputs/fig04/old/work/full50/fig4_metrics_summary.csv \
		--out-dir outputs/fig10/old/work/kg_perturbation \
		--model-name $(FIG10_MODEL) --timeout $(FIG10_TIMEOUT) --max-cases $(FIG10_MAX_CASES) \
		--resume --skip-existing
	$(PYTHON) experiments/fig10/old/build_fig10_same_rubric_baseline.py \
		--fig4-dir outputs/fig04/old/work/full50 \
		--baseline-outputs outputs/fig10/old/work/kg_perturbation/fig10_generic_llm_baseline_outputs.jsonl \
		--out-dir outputs/fig10/old/work/kg_perturbation
	$(PYTHON) experiments/fig10/old/build_fig10_ablation.py \
		--fig4-metrics outputs/fig04/old/work/full50/fig4_metrics_summary.csv \
		--out-dir outputs/fig10/old/work/kg_perturbation

figures-evidence-packets:
	FIG4_REUSE_RETRIEVAL_CACHE=1 FIG4_QUERY_KEYWORD_LIMIT=$(FIG4_QUERY_KEYWORD_LIMIT) ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE) ASPR_OPENALEX_FROM_YEAR=$(FIG4_OPENALEX_FROM_YEAR) ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL) ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL) ASPR_LATS_LLM_API_KEY=ollama FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS) ASPR_LATS_CANDIDATES=$(ASPR_LATS_CANDIDATES) ASPR_LATS_BEAM_WIDTH=$(ASPR_LATS_BEAM_WIDTH) ASPR_LATS_MAX_TOKENS=$(ASPR_LATS_MAX_TOKENS) ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX) ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS) $(PYTHON) experiments/fig04/old/main_fig4.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig04/old/work/full50 \
		--sample-size 50 --journal-scope all \
		--retrieval-provider openalex --judge-backend heuristic \
		--reuse-audit --refresh-invalid-agent-only \
		--prefer-scored-candidate-pool \
		--require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet
	@printf '%s\n' 'Fig4 completed label return file: outputs/fig04/old/work/full50/fig4_completed_blinded_labels.csv'
	$(PYTHON) experiments/fig09/old/build_fig9_case.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig09/old/work/kg_perturbation
	@printf '%s\n' 'Fig9 checkpoint contract: outputs/fig09/old/work/kg_perturbation/fig9_checkpoint_run_contract.json'
	$(PYTHON) experiments/fig10/old/build_fig10_ablation.py \
		--fig4-metrics outputs/fig04/old/work/full50/fig4_metrics_summary.csv \
		--out-dir outputs/fig10/old/work/kg_perturbation
	@printf '%s\n' 'Fig10 true rerun contract: outputs/fig10/old/work/kg_perturbation/fig10_true_module_rerun_contract.csv'
	@printf '%s\n' 'Fig10 blinded preference packet: outputs/fig10/old/work/kg_perturbation/fig10_blinded_preference_packet.csv'
	@printf '%s\n' 'Fig10 completed preference return file: outputs/fig10/old/work/kg_perturbation/fig10_completed_blinded_preferences.csv'
	$(PYTHON) experiments/common/old/final_assembly/build_final_assembly.py
	@printf '%s\n' 'External evidence packet index: outputs/common/old/final_assembly_work/fig1_fig10_external_evidence_packet_index.csv'

figures-external-evidence-intake:
	@printf '%s\n' 'Reading returned Fig4 labels from: outputs/fig04/old/work/full50/fig4_completed_blinded_labels.csv'
	$(MAKE) fig4-merge-blinded-labels
	FIG4_REUSE_RETRIEVAL_CACHE=1 FIG4_QUERY_KEYWORD_LIMIT=$(FIG4_QUERY_KEYWORD_LIMIT) ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE) ASPR_OPENALEX_FROM_YEAR=$(FIG4_OPENALEX_FROM_YEAR) ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL) ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL) ASPR_LATS_LLM_API_KEY=ollama FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS) ASPR_LATS_CANDIDATES=$(ASPR_LATS_CANDIDATES) ASPR_LATS_BEAM_WIDTH=$(ASPR_LATS_BEAM_WIDTH) ASPR_LATS_MAX_TOKENS=$(ASPR_LATS_MAX_TOKENS) ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX) ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS) $(PYTHON) experiments/fig04/old/main_fig4.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig04/old/work/full50 \
		--sample-size 50 --journal-scope all \
		--retrieval-provider openalex --judge-backend heuristic \
		--reuse-audit --refresh-invalid-agent-only \
		--prefer-scored-candidate-pool \
		--require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet
	@printf '%s\n' 'Reading returned Fig10 preferences from: outputs/fig10/old/work/kg_perturbation/fig10_completed_blinded_preferences.csv'
	$(MAKE) fig10-merge-blinded-preferences
	$(PYTHON) experiments/fig10/old/build_fig10_ablation.py \
		--fig4-metrics outputs/fig04/old/work/full50/fig4_metrics_summary.csv \
		--out-dir outputs/fig10/old/work/kg_perturbation
	$(PYTHON) experiments/common/old/final_assembly/build_final_assembly.py
	$(PYTHON) -m experiments.nature_ready_checks \
		--out-dir outputs/common/old/final_assembly_work \
		--strict-evidence-check

fig4-merge-blinded-labels:
	$(PYTHON) -c "from pathlib import Path; from experiments.fig04.old.main_fig4 import merge_fig4_labeler_blinded_label_returns; print(merge_fig4_labeler_blinded_label_returns(Path('outputs/fig04/old/work/full50')).to_json(orient='records'))"

fig10-merge-blinded-preferences:
	$(PYTHON) -c "from pathlib import Path; from experiments.fig10.old.build_fig10_ablation import merge_fig10_evaluator_preference_returns; print(merge_fig10_evaluator_preference_returns(Path('outputs/fig10/old/work/kg_perturbation')).to_json(orient='records'))"

fig9-checkpoint-run:
	$(PYTHON) experiments/fig09/old/run_fig9_checkpoint_inference.py \
		--checkpoint-path $(FIG9_CHECKPOINT_PATH) \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig09/old/work/kg_perturbation \
		--max-new-tokens $(FIG9_MAX_NEW_TOKENS) \
		--temperature $(FIG9_TEMPERATURE) \
		--top-p $(FIG9_TOP_P) \
		--seed $(FIG9_SEED) \
		--max-input-chars $(FIG9_MAX_INPUT_CHARS)
	$(PYTHON) experiments/fig09/old/build_fig9_case.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/fig09/old/work/kg_perturbation
	$(PYTHON) experiments/common/old/final_assembly/build_final_assembly.py

fig10-disabled-reruns:
	$(PYTHON) experiments/fig10/old/build_fig10_disabled_reruns.py \
		--fig4-metrics outputs/fig04/old/work/full50/fig4_metrics_summary.csv \
		--fig4-agent-outputs outputs/fig04/old/work/full50/fig4_agent_outputs.jsonl \
		--out-dir outputs/fig10/old/work/kg_perturbation \
		--model-name $(FIG10_MODEL) --timeout $(FIG10_TIMEOUT) \
		$(if $(filter-out 0,$(FIG10_DISABLED_MAX_CASES)),--max-cases $(FIG10_DISABLED_MAX_CASES),) \
		$(if $(FIG10_DISABLED_VARIANTS),--variants $(FIG10_DISABLED_VARIANTS),) \
		--resume --skip-existing
	$(PYTHON) experiments/fig10/old/build_fig10_ablation.py \
		--fig4-metrics outputs/fig04/old/work/full50/fig4_metrics_summary.csv \
		--out-dir outputs/fig10/old/work/kg_perturbation
	$(PYTHON) experiments/common/old/final_assembly/build_final_assembly.py

fig8-current:
	$(PYTHON) experiments/fig08/old/build_fig8_handoff.py \
		--out-dir outputs/fig08/old/work/kg_perturbation

final-assembly:
	$(PYTHON) experiments/common/old/final_assembly/build_final_assembly.py

figures-nature-check:
	python3 -m experiments.nature_ready_checks \
		--out-dir outputs/common/old/final_assembly_work

figures-all-nature-check:
	python3 -m experiments.nature_ready_checks \
		--out-dir outputs/common/old/final_assembly_work \
		--require-all-figures

figures-strict-evidence-check:
	python3 -m experiments.nature_ready_checks \
		--out-dir outputs/common/old/final_assembly_work \
		--strict-evidence-check

test-nature-ready:
	$(PYTHON) -m unittest \
		tests.test_final_assembly \
		tests.test_nature_ready_claims \
		tests.test_fig8_renderer \
		tests.test_reproducibility_manifest \
		tests.test_no_leakage_features \
		tests.test_fig1_sampling_horizon \
	tests.test_fig2_reference_closure \
	tests.test_fig3_holdout_baselines \
	tests.test_fig3_reuse_contract \
	tests.test_fig4_external_validation \
		tests.test_fig5_forecast_backtest \
	tests.test_fig6_robustness \
	tests.test_fig7_venue_contribution \
	tests.test_fig9_checkpoint_boundary \
tests.test_fig10_generic_baseline \
tests.test_fig10_disabled_reruns \
tests.test_fig10_same_rubric_baseline \
tests.test_fig10_ablation -v

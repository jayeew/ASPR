.PHONY: figures-current figures-main-nature figures-extended figures-evidence-packets figures-external-evidence-intake fig4-merge-blinded-labels fig9-checkpoint-run fig10-disabled-reruns fig10-merge-blinded-preferences figures-nature-check figures-all-nature-check figures-strict-evidence-check fig8-current final-assembly test-nature-ready

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
FIG3_MIN_CONTROLS ?= 50
FIG3_N_WEIGHT_SAMPLES ?= 5000
FIG3_REUSE_VALID ?= 1
FIG3_REUSE_ARGS = $(if $(filter 1 true yes,$(FIG3_REUSE_VALID)),--reuse-valid-current,)
FIG6_BUILD_FULL_RERUN ?= 1
FIG6_FULL_RERUN_MAX_PAPERS ?= 300
FIG9_CHECKPOINT_PATH ?= /home/jayee/workspace/checkpoint/qwen-0.6b-review
FIG9_MAX_NEW_TOKENS ?= 512
FIG9_TEMPERATURE ?= 0.2
FIG9_TOP_P ?= 0.9
FIG9_SEED ?= 20260701
FIG9_MAX_INPUT_CHARS ?= 8000

figures-current: figures-main-nature figures-extended final-assembly

figures-main-nature:
	$(PYTHON) experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py \
		--config experiments/kg_perturbation_fig1/configs/v6a_display_crispr.yaml \
		         experiments/kg_perturbation_fig1/configs/v6a_display_graphene.yaml \
		         experiments/kg_perturbation_fig1/configs/v6a_display_ipsc.yaml \
		         experiments/kg_perturbation_fig1/configs/v6a_display_exoplanets.yaml \
		--out-dir outputs/redraw_v6a_best_fig1 \
		--corpus-dir data/knowledge_corpus/v2_publication_v6a_locked_candidate
	$(PYTHON) experiments/kg_perturbation_fig2/build_fig2_strong_inputs.py \
		--source data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig2 \
		--out-dir outputs/redraw_v6a_best_fig2/fig2_strong_input \
		--pre-cutoff-max-year 2018 \
		--future-window-start 2019 \
		--future-window-end 2025 \
		--min-total-eligible 8000 \
		--min-controls $(FIG2_MIN_CONTROLS) \
		--reference-count-bins $(FIG2_REFERENCE_COUNT_BINS)
	$(PYTHON) experiments/kg_perturbation_fig2/fig2_empirical_panels.py \
		--data-dir outputs/redraw_v6a_best_fig2/fig2_strong_input \
		--out-dir outputs/redraw_v6a_best_fig2 \
		--evidence-mode strong \
		--domains crispr,exoplanets,gamma_ray_bursts_and_supernovae,genetics_aging_and_longevity_in_model_organisms,graphene_2d_materials,ipsc_reprogramming,microbiome_metagenomics,perovskite_solar_cells,topological_insulators,ubiquitin_and_proteasome_pathways \
		--panel all --export-tables \
		--future-tau $(FIG2_FUTURE_TAU) \
		--min-refs $(FIG2_MIN_REFS) \
		--min-controls $(FIG2_MIN_CONTROLS) \
		--max-papers $(FIG2_MAX_PAPERS) \
		--quiet
	$(PYTHON) experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py \
		--data-dir outputs/redraw_v6a_best_fig2/fig2_strong_input \
		--out-dir outputs/redraw_v6a_best_fig3 \
		--run-mode multi_domain \
		--domains crispr exoplanets gamma_ray_bursts_and_supernovae genetics_aging_and_longevity_in_model_organisms graphene_2d_materials ipsc_reprogramming microbiome_metagenomics perovskite_solar_cells topological_insulators ubiquitin_and_proteasome_pathways \
		--panel all --export-tables --diagnostics \
		--tau $(FIG2_FUTURE_TAU) \
		--min-refs $(FIG2_MIN_REFS) \
		--min-controls $(FIG3_MIN_CONTROLS) \
		--max-papers $(FIG2_MAX_PAPERS) \
		--n-weight-samples $(FIG3_N_WEIGHT_SAMPLES) --skip-sensitivity --formats png svg --quiet \
		$(FIG3_REUSE_ARGS)
	FIG4_REUSE_RETRIEVAL_CACHE=1 FIG4_QUERY_KEYWORD_LIMIT=$(FIG4_QUERY_KEYWORD_LIMIT) ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE) ASPR_OPENALEX_FROM_YEAR=$(FIG4_OPENALEX_FROM_YEAR) ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL) ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL) ASPR_LATS_LLM_API_KEY=ollama FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS) ASPR_LATS_CANDIDATES=$(ASPR_LATS_CANDIDATES) ASPR_LATS_BEAM_WIDTH=$(ASPR_LATS_BEAM_WIDTH) ASPR_LATS_MAX_TOKENS=$(ASPR_LATS_MAX_TOKENS) ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX) ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS) $(PYTHON) experiments/kg_perturbation_fig4/main_fig4.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig4_full50 \
		--sample-size 50 --journal-scope all \
		--retrieval-provider openalex --judge-backend heuristic \
		--reuse-audit --refresh-invalid-agent-only \
		--prefer-scored-candidate-pool \
		--require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet
	$(PYTHON) -m experiments.kg_perturbation_fig5.fig5_forecast_outcomes \
		--fig3-run-dir outputs/redraw_v6a_best_fig3/multi_domain \
		--fig3-input-dir outputs/redraw_v6a_best_fig3/fig3_input/multi_domain \
		--out-dir outputs/kg_perturbation_fig5 \
		--backtest-windows 2010:2015 2015:2020 2020:2025 \
		--formats png svg --quiet
	FIG6_BUILD_FULL_RERUN=$(FIG6_BUILD_FULL_RERUN) FIG6_FULL_RERUN_MAX_PAPERS=$(FIG6_FULL_RERUN_MAX_PAPERS) \
		$(PYTHON) experiments/kg_perturbation_fig6/build_fig6_robustness.py

figures-extended:
	$(PYTHON) experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py \
		--fig3-run-dir outputs/redraw_v6a_best_fig3/multi_domain \
		--works-table data/knowledge_corpus/v2_publication_v6a_locked_candidate/works.csv \
		--out-dir outputs/kg_perturbation_fig7
	$(MAKE) fig8-current
	$(PYTHON) experiments/kg_perturbation_fig9/build_fig9_case.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig9
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py \
		--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
		--out-dir outputs/kg_perturbation_fig10 \
		--model-name $(FIG10_MODEL) --timeout $(FIG10_TIMEOUT) --max-cases $(FIG10_MAX_CASES) \
		--resume --skip-existing
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_same_rubric_baseline.py \
		--fig4-dir outputs/kg_perturbation_fig4_full50 \
		--baseline-outputs outputs/kg_perturbation_fig10/fig10_generic_llm_baseline_outputs.jsonl \
		--out-dir outputs/kg_perturbation_fig10
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_ablation.py \
		--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
		--out-dir outputs/kg_perturbation_fig10

figures-evidence-packets:
	FIG4_REUSE_RETRIEVAL_CACHE=1 FIG4_QUERY_KEYWORD_LIMIT=$(FIG4_QUERY_KEYWORD_LIMIT) ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE) ASPR_OPENALEX_FROM_YEAR=$(FIG4_OPENALEX_FROM_YEAR) ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL) ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL) ASPR_LATS_LLM_API_KEY=ollama FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS) ASPR_LATS_CANDIDATES=$(ASPR_LATS_CANDIDATES) ASPR_LATS_BEAM_WIDTH=$(ASPR_LATS_BEAM_WIDTH) ASPR_LATS_MAX_TOKENS=$(ASPR_LATS_MAX_TOKENS) ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX) ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS) $(PYTHON) experiments/kg_perturbation_fig4/main_fig4.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig4_full50 \
		--sample-size 50 --journal-scope all \
		--retrieval-provider openalex --judge-backend heuristic \
		--reuse-audit --refresh-invalid-agent-only \
		--prefer-scored-candidate-pool \
		--require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet
	@printf '%s\n' 'Fig4 completed label return file: outputs/kg_perturbation_fig4_full50/fig4_completed_blinded_labels.csv'
	$(PYTHON) experiments/kg_perturbation_fig9/build_fig9_case.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig9
	@printf '%s\n' 'Fig9 checkpoint contract: outputs/kg_perturbation_fig9/fig9_checkpoint_run_contract.json'
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_ablation.py \
		--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
		--out-dir outputs/kg_perturbation_fig10
	@printf '%s\n' 'Fig10 true rerun contract: outputs/kg_perturbation_fig10/fig10_true_module_rerun_contract.csv'
	@printf '%s\n' 'Fig10 blinded preference packet: outputs/kg_perturbation_fig10/fig10_blinded_preference_packet.csv'
	@printf '%s\n' 'Fig10 completed preference return file: outputs/kg_perturbation_fig10/fig10_completed_blinded_preferences.csv'
	$(PYTHON) experiments/kg_perturbation_final_assembly/build_final_assembly.py
	@printf '%s\n' 'External evidence packet index: outputs/kg_perturbation_final_assembly/fig1_fig10_external_evidence_packet_index.csv'

figures-external-evidence-intake:
	@printf '%s\n' 'Reading returned Fig4 labels from: outputs/kg_perturbation_fig4_full50/fig4_completed_blinded_labels.csv'
	$(MAKE) fig4-merge-blinded-labels
	FIG4_REUSE_RETRIEVAL_CACHE=1 FIG4_QUERY_KEYWORD_LIMIT=$(FIG4_QUERY_KEYWORD_LIMIT) ASPR_OPENALEX_PER_PAGE=$(FIG4_OPENALEX_PER_PAGE) ASPR_OPENALEX_FROM_YEAR=$(FIG4_OPENALEX_FROM_YEAR) ASPR_LATS_LLM_MODEL=$(FIG4_LATS_MODEL) ASPR_LATS_LLM_BASE_URL=$(FIG4_LATS_BASE_URL) ASPR_LATS_LLM_API_KEY=ollama FIG4_AGENT_MAX_ITERATIONS=$(FIG4_AGENT_MAX_ITERATIONS) ASPR_LATS_CANDIDATES=$(ASPR_LATS_CANDIDATES) ASPR_LATS_BEAM_WIDTH=$(ASPR_LATS_BEAM_WIDTH) ASPR_LATS_MAX_TOKENS=$(ASPR_LATS_MAX_TOKENS) ASPR_LATS_PROMPT_PREFIX=$(ASPR_LATS_PROMPT_PREFIX) ASPR_LATS_SINGLE_PASS=$(ASPR_LATS_SINGLE_PASS) $(PYTHON) experiments/kg_perturbation_fig4/main_fig4.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig4_full50 \
		--sample-size 50 --journal-scope all \
		--retrieval-provider openalex --judge-backend heuristic \
		--reuse-audit --refresh-invalid-agent-only \
		--prefer-scored-candidate-pool \
		--require-fixed-sample --forbid-lightweight --forbid-local-retrieval --forbid-lexical-fallback --quiet
	@printf '%s\n' 'Reading returned Fig10 preferences from: outputs/kg_perturbation_fig10/fig10_completed_blinded_preferences.csv'
	$(MAKE) fig10-merge-blinded-preferences
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_ablation.py \
		--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
		--out-dir outputs/kg_perturbation_fig10
	$(PYTHON) experiments/kg_perturbation_final_assembly/build_final_assembly.py
	$(PYTHON) -m experiments.nature_ready_checks \
		--out-dir outputs/kg_perturbation_final_assembly \
		--strict-evidence-check

fig4-merge-blinded-labels:
	$(PYTHON) -c "from pathlib import Path; from experiments.kg_perturbation_fig4.main_fig4 import merge_fig4_labeler_blinded_label_returns; print(merge_fig4_labeler_blinded_label_returns(Path('outputs/kg_perturbation_fig4_full50')).to_json(orient='records'))"

fig10-merge-blinded-preferences:
	$(PYTHON) -c "from pathlib import Path; from experiments.kg_perturbation_fig10.build_fig10_ablation import merge_fig10_evaluator_preference_returns; print(merge_fig10_evaluator_preference_returns(Path('outputs/kg_perturbation_fig10')).to_json(orient='records'))"

fig9-checkpoint-run:
	$(PYTHON) experiments/kg_perturbation_fig9/run_fig9_checkpoint_inference.py \
		--checkpoint-path $(FIG9_CHECKPOINT_PATH) \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig9 \
		--max-new-tokens $(FIG9_MAX_NEW_TOKENS) \
		--temperature $(FIG9_TEMPERATURE) \
		--top-p $(FIG9_TOP_P) \
		--seed $(FIG9_SEED) \
		--max-input-chars $(FIG9_MAX_INPUT_CHARS)
	$(PYTHON) experiments/kg_perturbation_fig9/build_fig9_case.py \
		--markdown-root $(FIG4_MARKDOWN_ROOT) \
		--output-dir outputs/kg_perturbation_fig9
	$(PYTHON) experiments/kg_perturbation_final_assembly/build_final_assembly.py

fig10-disabled-reruns:
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_disabled_reruns.py \
		--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
		--fig4-agent-outputs outputs/kg_perturbation_fig4_full50/fig4_agent_outputs.jsonl \
		--out-dir outputs/kg_perturbation_fig10 \
		--model-name $(FIG10_MODEL) --timeout $(FIG10_TIMEOUT) \
		$(if $(filter-out 0,$(FIG10_DISABLED_MAX_CASES)),--max-cases $(FIG10_DISABLED_MAX_CASES),) \
		$(if $(FIG10_DISABLED_VARIANTS),--variants $(FIG10_DISABLED_VARIANTS),) \
		--resume --skip-existing
	$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_ablation.py \
		--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
		--out-dir outputs/kg_perturbation_fig10
	$(PYTHON) experiments/kg_perturbation_final_assembly/build_final_assembly.py

fig8-current:
	$(PYTHON) -m experiments.kg_perturbation_fig8.render_fig8 \
		--out-dir outputs/kg_perturbation_fig8

final-assembly:
	$(PYTHON) experiments/kg_perturbation_final_assembly/build_final_assembly.py

figures-nature-check:
	python3 -m experiments.nature_ready_checks \
		--out-dir outputs/kg_perturbation_final_assembly

figures-all-nature-check:
	python3 -m experiments.nature_ready_checks \
		--out-dir outputs/kg_perturbation_final_assembly \
		--require-all-figures

figures-strict-evidence-check:
	python3 -m experiments.nature_ready_checks \
		--out-dir outputs/kg_perturbation_final_assembly \
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

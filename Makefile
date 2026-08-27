.PHONY: gear-review gear-test gear-validate gear-lint gear-reconstruction-help gear-reconstruction-test gear-module-help artifact-help dataset-help figures-help

PYTHON ?= python3
GEAR_PAPER ?=
GEAR_METADATA ?=
GEAR_OUTPUT_DIR ?=
GEAR_RUNTIME_LINT = \
	gear/__init__.py gear/cli.py gear/config.py gear/diffusion_forecast.py \
	gear/evidence_policy.py gear/evidence_supervisor.py gear/graph_guidance.py \
	gear/graph_prior.py gear/graph_prior_contracts.py gear/prior_art.py \
	gear/process_diagnostic.py gear/review_compiler.py gear/review_contracts.py \
	gear/review_fusion.py gear/review_pipeline.py gear/review_state.py \
	gear/review_verifier.py experiments/gear/evaluation \
	experiments/gear/review_reconstruction/contracts.py \
	experiments/gear/review_reconstruction/sessions.py \
	scripts/build_gear_diffusion_release.py tests/gear

gear-review:
	@test -n "$(GEAR_PAPER)" || (echo "GEAR_PAPER is required" >&2; exit 2)
	$(PYTHON) -m gear review \
		--paper "$(GEAR_PAPER)" \
		$(if $(GEAR_METADATA),--metadata "$(GEAR_METADATA)",) \
		$(if $(GEAR_OUTPUT_DIR),--output-dir "$(GEAR_OUTPUT_DIR)",)

gear-test:
	TMPDIR=/tmp $(PYTHON) -m pytest -s -q tests/gear

gear-validate:
	$(PYTHON) -m gear validate-assets

gear-lint:
	$(PYTHON) -m black --check $(GEAR_RUNTIME_LINT)
	$(PYTHON) -m ruff check --select E4,E7,E9,F,I $(GEAR_RUNTIME_LINT)
	$(PYTHON) -m mypy --ignore-missing-imports --follow-imports=skip \
		gear/__init__.py gear/cli.py gear/config.py gear/diffusion_forecast.py \
		gear/graph_prior_contracts.py gear/graph_prior.py gear/graph_guidance.py \
		gear/review_contracts.py gear/review_state.py gear/review_fusion.py \
		gear/review_compiler.py gear/evidence_supervisor.py gear/review_pipeline.py \
		gear/review_verifier.py experiments/gear/evaluation/contracts.py \
		experiments/gear/evaluation/graph_ablation.py \
		experiments/gear/evaluation/human_audit.py \
		experiments/gear/evaluation/runner.py \
		experiments/gear/review_reconstruction/contracts.py \
		experiments/gear/review_reconstruction/sessions.py \
		scripts/build_gear_diffusion_release.py

gear-reconstruction-help:
	$(PYTHON) -m experiments.gear.review_reconstruction --help

gear-reconstruction-test:
	TMPDIR=/tmp $(PYTHON) -m pytest -s -q \
		tests/gear/test_review_contracts.py \
		tests/gear/test_runtime.py \
		tests/gear/test_reconstruction.py \
		tests/gear/test_human_audit.py

gear-module-help:
	$(PYTHON) -m gear.module_cli --help

artifact-help:
	$(PYTHON) -m artifact_store --help

dataset-help:
	$(PYTHON) scripts/run_nature_multihorizon.py --help

figures-help:
	$(PYTHON) -m experiments.common.new.run_all --help

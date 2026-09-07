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
	TMPDIR=/tmp $(PYTHON) -m pytest -s -q tests/innovation_v2 tests/gear/test_paper_compiler.py tests/gear/test_prior_art.py tests/gear/test_codex_cli.py tests/gear/test_model_backends.py

gear-validate:
	$(PYTHON) -m gear validate-assets

gear-lint:
	$(PYTHON) -m black --check gear/innovation gear/work_identity.py scripts/innovation_experiment.py tests/innovation_v2
	$(PYTHON) -m ruff check gear/innovation gear/work_identity.py scripts/innovation_experiment.py tests/innovation_v2
	$(PYTHON) -m mypy --ignore-missing-imports --follow-imports=skip gear/innovation scripts/innovation_experiment.py

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

# Archived interface tests remain available for migration diagnostics.
gear-legacy-test:
	$(PYTHON) -m pytest -q tests/gear

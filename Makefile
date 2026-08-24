.PHONY: gear-review gear-test gear-validate gear-lint gear-submission-train gear-calibration-show gear-calibration-promote gear-reconstruction-help gear-reconstruction-test gear-module-help artifact-help dataset-help figures-help

PYTHON ?= python3
GEAR_PAPER ?=
GEAR_METADATA ?=
GEAR_OUTPUT_DIR ?=
GEAR_SUBMISSION_OUTPUT ?= outputs/gear/submission_calibration

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

gear-calibration-show:
	$(PYTHON) -m gear show-calibration --verify

gear-calibration-promote:
	$(PYTHON) -m gear promote-calibration

gear-lint:
	$(PYTHON) -m black --check gear scripts/run_gear_reconstruction_sessions.py scripts/run_gear_revision_audit.py experiments/gear tests/gear
	$(PYTHON) -m ruff check --select E4,E7,E9,F,I gear scripts/run_gear_reconstruction_sessions.py scripts/run_gear_revision_audit.py experiments/gear tests/gear
	$(PYTHON) -m mypy --ignore-missing-imports --follow-imports=skip gear scripts/run_gear_reconstruction_sessions.py scripts/run_gear_revision_audit.py experiments/gear

gear-submission-train:
	$(PYTHON) experiments/gear/train_submission_calibration.py \
		--output-dir "$(GEAR_SUBMISSION_OUTPUT)"

gear-reconstruction-help:
	$(PYTHON) -m experiments.gear.review_reconstruction --help

gear-reconstruction-test:
	TMPDIR=/tmp $(PYTHON) -m pytest -s -q \
		tests/gear/test_review_contracts.py \
		tests/gear/test_runtime.py \
		tests/gear/test_reconstruction.py \
		tests/gear/test_evaluation.py

gear-module-help:
	$(PYTHON) -m gear.module_cli --help

artifact-help:
	$(PYTHON) -m artifact_store --help

dataset-help:
	$(PYTHON) scripts/run_nature_multihorizon.py --help

figures-help:
	$(PYTHON) -m experiments.common.new.run_all --help

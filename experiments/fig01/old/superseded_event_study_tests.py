"""Archived tests for the superseded generic Fig.1 event-study design."""

from pathlib import Path

from experiments.common.new.adapters.tests_support import run_tests


if __name__ == "__main__":
    run_tests(
        1,
        Path(__file__).with_name("superseded_event_study_config.json"),
    )

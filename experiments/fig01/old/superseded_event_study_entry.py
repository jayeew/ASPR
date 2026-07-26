"""Archived entry point for the superseded generic Fig.1 event-study design."""

from pathlib import Path

from experiments.common.new.adapters.runtime import run_figure_cli


if __name__ == "__main__":
    run_figure_cli(
        1,
        Path(__file__).with_name("superseded_event_study_config.json"),
    )

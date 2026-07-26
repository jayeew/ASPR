"""Run Fig.3."""

from pathlib import Path

from experiments.common.new.adapters.runtime import run_figure_cli


if __name__ == "__main__":
    run_figure_cli(3, Path(__file__).with_name("config.json"))

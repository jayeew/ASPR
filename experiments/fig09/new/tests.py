"""Tests for Fig.9."""

from pathlib import Path

from experiments.common.new.adapters.tests_support import run_tests


if __name__ == "__main__":
    run_tests(9, Path(__file__).with_name("config.json"))

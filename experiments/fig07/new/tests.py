"""Tests for Fig.7."""

from pathlib import Path

from experiments.common.new.adapters.tests_support import run_tests


if __name__ == "__main__":
    run_tests(7, Path(__file__).with_name("config.json"))

"""Regression tests for the expanded-data ASPR Fig.3."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .analysis import load_config, resolve_path
from .audit import validate_outputs


class Fig03PerformanceLandscapeTests(unittest.TestCase):
    """Check the frozen statistical and rendering contracts."""

    config: Mapping[str, Any]
    output_dir: Path
    report: Mapping[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.output_dir = resolve_path(str(cls.config["output_dir"]))
        cls.report = validate_outputs(cls.config, cls.output_dir)

    def test_all_acceptance_checks_pass(self) -> None:
        failed = [row for row in self.report["checks"] if not row["passed"]]
        self.assertFalse(failed, failed)

    def test_figure_status_is_complete(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.report["status"], "complete_aspr_performance_landscape")


if __name__ == "__main__":
    unittest.main(verbosity=2)

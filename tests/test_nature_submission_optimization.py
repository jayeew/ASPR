from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.nature_submission_optimization.build_nature_optimization import build_optimization


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class NatureSubmissionOptimizationTest(unittest.TestCase):
    def test_builds_fig1_fig10_optimization_package(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_optimization_") as tmp_dir:
            result = build_optimization(Path(tmp_dir), "latest")
            out_dir = Path(result["out_dir"])

            self.assertTrue(result["quality"]["overall_pass"])
            self.assertEqual(10, len(read_rows(out_dir / "figure_optimization_status.csv")))
            self.assertGreaterEqual(len(read_rows(out_dir / "fig6_fig10_priority_backlog.csv")), 5)
            self.assertTrue((out_dir / "caption_replacement_drafts.md").exists())
            self.assertTrue((out_dir / "submission_claim_boundaries.md").exists())
            self.assertTrue((out_dir / "experiment_upgrade_protocol.md").exists())
            self.assertTrue((out_dir / "verification_log.md").exists())

            claim_boundary = (out_dir / "submission_claim_boundaries.md").read_text(encoding="utf-8")
            self.assertIn("Fig.10", claim_boundary)
            self.assertIn("不能写成真实模块因果重跑", claim_boundary)

            backlog = read_rows(out_dir / "fig6_fig10_priority_backlog.csv")
            self.assertTrue(any(row["fig_id"] == "fig10" and row["action_class"] == "pipeline_ready_gap" for row in backlog))
            self.assertTrue(any(row["fig_id"] == "fig6" and row["action_class"] == "pipeline_ready_gap" for row in backlog))


if __name__ == "__main__":
    unittest.main()

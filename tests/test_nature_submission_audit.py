from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.nature_submission_audit.build_nature_submission_audit import FIGURES, build_audit


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class NatureSubmissionAuditTest(unittest.TestCase):
    def test_builds_complete_chinese_iteration_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = build_audit(Path(tmp_dir), "latest")
            out_dir = Path(result["out_dir"])

            self.assertTrue(result["quality"]["overall_pass"])
            self.assertEqual(10, len(FIGURES))

            for fig in FIGURES:
                fig_dir = out_dir / fig.fig_id
                self.assertTrue((fig_dir / f"{fig.fig_id}_data_forensics.md").exists())
                self.assertTrue((fig_dir / f"{fig.fig_id}_visual_audit.csv").exists())
                self.assertTrue((fig_dir / f"{fig.fig_id}_claim_to_evidence_map.csv").exists())
                self.assertTrue((fig_dir / f"{fig.fig_id}_reviewer_objection_notes.md").exists())
                self.assertTrue((fig_dir / f"{fig.fig_id}_readiness_score.csv").exists())

            required_outputs = [
                "audit_iteration_report.md",
                "iteration_decision_board.csv",
                "updated_gaps.csv",
                "caption_edits.csv",
                "caption_edits.md",
                "reviewer_objection_response_matrix.csv",
                "verification_log.md",
                "contact_sheet.png",
                "figure_quality_report.json",
            ]
            for name in required_outputs:
                self.assertTrue((out_dir / name).exists(), name)

            report_text = (out_dir / "audit_iteration_report.md").read_text(encoding="utf-8")
            self.assertIn("Nature 投稿级一轮审计报告", report_text)
            self.assertIn("pipeline-ready 估计", report_text)

            decision_rows = read_rows(out_dir / "iteration_decision_board.csv")
            self.assertGreaterEqual(len(decision_rows), 20)
            self.assertTrue(any(row["label"] == "Fig.10" for row in decision_rows))

            gap_rows = read_rows(out_dir / "updated_gaps.csv")
            self.assertTrue(any(row["label"] == "Fig.10" for row in gap_rows))
            self.assertTrue(any(row["label"] == "Fig.9" for row in gap_rows))

            caption_rows = read_rows(out_dir / "caption_edits.csv")
            self.assertEqual(10, len(caption_rows))

            reviewer_rows = read_rows(out_dir / "reviewer_objection_response_matrix.csv")
            self.assertEqual(30, len(reviewer_rows))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_final_assembly.build_final_assembly import (  # noqa: E402
    CAPTIONS,
    build_final_assembly,
)


class FinalAssemblyTests(unittest.TestCase):
    def test_final_assembly_contract(self) -> None:
        self.assertEqual(len(CAPTIONS), 10)
        with tempfile.TemporaryDirectory(prefix="aspr_final_assembly_") as tmp:
            out_dir = Path(tmp)
            result = build_final_assembly(out_dir)

            self.assertTrue(result["quality_gates"]["overall_pass"])
            self.assertTrue((out_dir / "fig6_fig10_three_round_consistency_report.md").exists())
            self.assertTrue((out_dir / "fig1_fig10_caption_drafts.md").exists())
            gaps = (out_dir / "fig1_fig10_pipeline_ready_gaps.csv").read_text(encoding="utf-8")
            self.assertIn("Fig.7", gaps)
            self.assertIn("ASPR-Qwen output is assumed", gaps)


if __name__ == "__main__":
    unittest.main()

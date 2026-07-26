from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.common.old.final_assembly.build_visual_redesign_handoff import build_handoff


class VisualRedesignHandoffTests(unittest.TestCase):
    def test_build_handoff_writes_four_redesign_prompts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_visual_redesign_") as tmp:
            out_dir = Path(tmp)

            result = build_handoff(out_dir)

            manifest = pd.read_csv(out_dir / "visual_redesign_handoff_manifest.csv")
            self.assertEqual({"Fig.6", "Fig.7", "Fig.9", "Fig.10"}, set(manifest["figure"]))
            self.assertTrue((out_dir / "fig6_visual_redesign_prompt.md").exists())
            self.assertTrue((out_dir / "fig7_visual_redesign_prompt.md").exists())
            self.assertTrue((out_dir / "fig9_visual_redesign_prompt.md").exists())
            self.assertTrue((out_dir / "fig10_visual_redesign_prompt.md").exists())
            prompt = (out_dir / "fig9_visual_redesign_prompt.md").read_text(encoding="utf-8")
            self.assertIn("one large run-instance visual", prompt)
            self.assertIn("source_artifacts", prompt)
            self.assertIn("Do not invent new data", prompt)
            self.assertIn("four_redesign_figures_covered", result["quality_gates"]["checks"])


if __name__ == "__main__":
    unittest.main()

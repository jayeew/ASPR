from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.kg_perturbation_fig8.build_fig8_handoff import build_handoff


PROMPT_TEXT = """
Input manuscript -> Reference-graph calibration -> Claim cards -> ASPR reflection-guided review search -> Evidence-linked review
B RS DeltaQ0 Uzzi RTD BurtIP PDE
ASPR-Qwen Nature review data Transparent peer review
S_w prior 是 calibration prior，不是论文质量分数
不要让 ASPR-Qwen 直接连接最终 review
只渲染短标签，不要把说明性段落画进图中。
"""


class Fig8HandoffTests(unittest.TestCase):
    def test_build_handoff_writes_quality_report_and_expected_image(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_fig8_") as tmp:
            root = Path(tmp)
            prompt = root / "fig8_prompt.md"
            prompt.write_text(PROMPT_TEXT, encoding="utf-8")
            image = root / "fig8_source.png"
            Image.new("RGB", (1600, 1000), "white").save(image)
            out_dir = root / "out"

            paths = build_handoff(prompt, image, out_dir)

            self.assertTrue(paths["image"].exists())
            self.assertEqual("fig8_full.png", paths["image"].name)
            report = (out_dir / "figure_quality_report.json").read_text(encoding="utf-8")
            self.assertIn("fig8_gpt_image_handoff_ready", report)
            self.assertIn("ASPR-Qwen is shown as a reviewer-style side branch", report)

    def test_makefile_uses_fig8_handoff_builder(self) -> None:
        makefile = Path("Makefile").read_text(encoding="utf-8")

        self.assertIn("experiments/kg_perturbation_fig8/build_fig8_handoff.py", makefile)
        self.assertNotIn("experiments.kg_perturbation_fig8.render_fig8", makefile)


if __name__ == "__main__":
    unittest.main()

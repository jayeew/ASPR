from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.old.nature_iteration.build_nature_iteration import (
    AUTO_CONTINUE_WITHOUT_USER_CHOICE,
    MAX_MAIN_ITERATIONS,
    build_final_patch,
    build_iteration,
)


class NatureIterationProtocolTests(unittest.TestCase):
    def test_iteration_manifest_encodes_multi_round_auto_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_iter_") as tmp:
            out_root = Path(tmp)

            manifest = build_iteration(6, out_root)

            self.assertEqual(6, MAX_MAIN_ITERATIONS)
            self.assertTrue(AUTO_CONTINUE_WITHOUT_USER_CHOICE)
            self.assertEqual("r6", manifest["round_id"])
            self.assertEqual(6, manifest["max_main_iterations"])
            self.assertTrue(manifest["auto_continue_without_user_choice"])
            self.assertFalse(manifest["requires_user_choice_mid_run"])
            self.assertTrue(manifest["fix_list_is_execution_queue"])
            self.assertEqual("auto_continue_default_actions_no_midrun_user_choice", manifest["execution_policy"])

            saved = json.loads((out_root / "r6" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(6, saved["max_main_iterations"])
            self.assertFalse(saved["requires_user_choice_mid_run"])

            reflection = (out_root / "r6" / "round_reflection.md").read_text(encoding="utf-8")
            fix_list = (out_root / "r6" / "next_fix_list.md").read_text(encoding="utf-8")
            self.assertIn("不要求、也不等待用户中途选择", reflection)
            self.assertIn("不输出路线选择题", fix_list)
            self.assertIn("最多 `6` 轮主迭代", fix_list)

    def test_round_zero_writes_explicit_baseline_audit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_iter_") as tmp:
            out_root = Path(tmp)

            build_iteration(0, out_root)

            baseline = out_root / "r0" / "nature_iter_baseline_audit.md"
            self.assertTrue(baseline.exists())
            text = baseline.read_text(encoding="utf-8")
            self.assertIn("# Nature Iteration Baseline Audit", text)
            self.assertIn("不要求用户选择路线", text)

    def test_final_patch_writes_stop_decision_and_cleans_iteration_visuals(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_nature_iter_") as tmp:
            out_root = Path(tmp)
            build_iteration(6, out_root)
            old_visual = out_root / "r5" / "old_panel.svg"
            old_visual.parent.mkdir(parents=True, exist_ok=True)
            old_visual.write_text("<svg></svg>", encoding="utf-8")

            manifest = build_final_patch(out_root)

            final_patch = out_root / "final_patch"
            self.assertFalse(old_visual.exists())
            self.assertEqual("final_patch", manifest["final_patch_id"])
            self.assertFalse(manifest["requires_user_choice_mid_run"])
            self.assertIn("stop_after_round6", manifest["stop_decision"])
            self.assertTrue((final_patch / "final_patch_manifest.json").exists())
            self.assertTrue((final_patch / "completion_audit.csv").exists())
            self.assertTrue((final_patch / "round6_blocker_gap_list.csv").exists())
            notes = (final_patch / "final_patch_notes.md").read_text(encoding="utf-8")
            self.assertIn("不继续循环美化", notes)
            self.assertIn("strict external-evidence claims", notes)


if __name__ == "__main__":
    unittest.main()

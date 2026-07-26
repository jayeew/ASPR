from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.old.v6_1_figures_r1.analysis import (
    ANGLE_ORDER,
    angle_feature_sets,
    load_inputs,
    prediction_deciles,
    primary_records,
)
from experiments.common.old.v6_1_figures_r1.run_all import EXPERIMENT_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AsprV61FigureSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs(
            PROJECT_ROOT,
            PROJECT_ROOT / "configs" / "aspr_v6_1_figures.json",
        )

    def test_eight_primary_metrics_cover_all_five_angles(self) -> None:
        primary = primary_records(self.inputs)
        self.assertEqual(8, len(primary))
        self.assertEqual(set(ANGLE_ORDER), set(primary["angle_id"]))
        self.assertTrue(primary["source_ids"].str.len().gt(0).all())

    def test_angle_ablation_sets_do_not_change_k1_or_primary_registry(self) -> None:
        feature_sets = angle_feature_sets(self.inputs)
        self.assertEqual(10, len(feature_sets))
        k1 = set(self.inputs.source_config["k1_controls"])
        primary = set(primary_records(self.inputs)["feature"])
        for index in range(1, 6):
            added = set(feature_sets[f"k1_plus_a{index}"])
            deleted = set(feature_sets[f"final_minus_a{index}"])
            self.assertTrue(k1.issubset(added))
            self.assertTrue(k1.issubset(deleted))
            self.assertTrue((added - k1).issubset(primary))
            self.assertTrue((deleted - k1).issubset(primary))

    def test_prediction_deciles_are_balanced_and_zero_inclusive(self) -> None:
        rows = []
        for model_id, reverse in (("model_a", False), ("model_b", True)):
            for index in range(100):
                rows.append(
                    {
                        "paper_id": f"P{index:03d}",
                        "horizon": 5,
                        "model_id": model_id,
                        "realized_diffusion_target": float(index // 2),
                        "expected_diffusion_score": float(
                            99 - index if reverse else index
                        ),
                    }
                )
        deciles = prediction_deciles(
            pd.DataFrame(rows), ["model_a", "model_b"]
        )
        self.assertEqual(20, len(deciles))
        self.assertTrue(deciles["n"].eq(10).all())
        self.assertTrue(
            np.isfinite(deciles["observed_high_impact_rate"]).all()
        )
        self.assertEqual(
            10,
            int(
                deciles.loc[
                    deciles["model_id"].eq("model_a"), "high_impact_count"
                ].sum()
            ),
        )

    def test_experiment_map_has_one_question_per_figure(self) -> None:
        self.assertEqual(list(range(1, 11)), [row["experiment"] for row in EXPERIMENT_MAP])
        self.assertEqual(10, len({row["question"] for row in EXPERIMENT_MAP}))


if __name__ == "__main__":
    unittest.main()

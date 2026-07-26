from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.new.base.builders_1_5 import (
    build_fig2,
    build_fig3,
    build_fig4,
)
from experiments.common.new.base.builders_6_10 import (
    build_fig6,
    build_fig7,
    build_fig8,
    build_fig9,
    build_fig10,
)
from experiments.common.new.base.common import (
    ANGLE_ORDER,
    FEATURE_LABELS,
    resolve_suite_paths,
    sha256_file,
)
from experiments.common.new.base.run_all import run_suite


CONFIG_PATH = PROJECT_ROOT / "configs/aspr_v6_1_nature_figures.json"


class AsprV61NatureFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.paths = resolve_suite_paths(
            CONFIG_PATH,
            Path("/tmp/aspr-v6-1-nature-figure-tests"),
        )
        cls.fig2 = build_fig2(cls.config, cls.paths)
        cls.fig3 = build_fig3(cls.config, cls.paths)
        cls.fig4 = build_fig4(cls.config, cls.paths)
        cls.fig6 = build_fig6(cls.config, cls.paths)
        cls.fig7 = build_fig7(cls.config, cls.paths)
        cls.fig8 = build_fig8(cls.config, cls.paths)
        cls.fig9 = build_fig9(cls.config, cls.paths)
        cls.fig10 = build_fig10(cls.config, cls.paths)

    def test_five_angles_and_eight_primary_indicators_are_frozen(self) -> None:
        primary = self.fig2.tables["primary_indicator_map"]
        self.assertEqual(8, len(primary))
        self.assertEqual(set(ANGLE_ORDER), set(primary["angle_id"]))
        self.assertEqual(set(FEATURE_LABELS), set(primary["code_name"]))
        self.assertEqual(50, len(self.fig2.tables["candidate_decisions"]))

    def test_primary_indicators_pass_registered_quality_gates(self) -> None:
        gates = self.fig2.tables["primary_quality_gates"]
        self.assertTrue(gates["overall_coverage"].ge(0.70).all())
        self.assertTrue(gates["minimum_domain_coverage"].ge(0.50).all())
        self.assertTrue(gates["stability_spearman"].ge(0.90).all())
        self.assertEqual(1, self.fig2.chart_contract["traditional_heatmap_count"])

    def test_main_temporal_oof_values_are_the_frozen_release(self) -> None:
        panel = self.fig3.panel_text
        self.assertAlmostEqual(0.7670398790343685, panel["b"]["main_oof_spearman"])
        self.assertAlmostEqual(0.6813462295889368, panel["b"]["k1_spearman"])
        self.assertAlmostEqual(
            0.7073002630551549,
            panel["b"]["innovation_only_spearman"],
        )
        self.assertEqual(101_350, panel["c"]["n"])

    def test_human_evidence_gates_remain_blocked(self) -> None:
        self.assertEqual("draft_labels_incomplete", self.fig4.status)
        self.assertFalse(self.fig4.chart_contract["status_gate"]["passed"])
        self.assertEqual(
            "draft_comparability_and_human_preference_blocked",
            self.fig10.status,
        )
        self.assertFalse(
            self.fig10.chart_contract["human_preference_gate"]["passed"]
        )
        self.assertFalse(self.fig10.chart_contract["automatic_comparison_valid"])

    def test_unrun_robustness_doses_are_not_materialized(self) -> None:
        self.assertEqual(
            [1.0, 0.8],
            self.fig6.panel_text["b"]["available_retention_levels"],
        )
        observed = set(
            self.fig6.tables["registered_reference_resampling"][
                "reference_retention"
            ].unique()
        )
        self.assertEqual({1.0, 0.8}, observed)
        self.assertFalse(observed.intersection({0.75, 0.50, 0.25, 0.10}))

    def test_venue_and_case_boundaries_are_explicit(self) -> None:
        score_contract = self.fig7.chart_contract["score_contract"]
        self.assertFalse(score_contract["contains_venue_family"])
        self.assertEqual("innovation_only", score_contract["model_id"])
        self.assertFalse(self.fig9.chart_contract["current_case_fingerprint_available"])
        self.assertFalse(self.fig9.chart_contract["population_performance_claim"])

    def test_generated_image_assets_are_non_numeric_and_hashed(self) -> None:
        expected = {
            8: "7a4c6fb487f9904d49d11b6c233682653e0c8eac709a126b96cddf4a676ee7a4",
            9: "c582545b380f5baf50cd1d5d3313636dbc3d9fa5288bbf6ceb7d2c7a7d540e00",
            10: "6629d62802f171d35bc76384d67b2792c2207b886a6a285d2366c22744281020",
        }
        for figure_id, bundle in [
            (8, self.fig8),
            (9, self.fig9),
            (10, self.fig10),
        ]:
            asset = Path(bundle.chart_contract["background_asset"])
            self.assertEqual(expected[figure_id], sha256_file(asset))
            if figure_id == 8:
                self.assertFalse(
                    bundle.chart_contract["image_asset_may_render_numeric_values"]
                )

    def test_single_figure_run_writes_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_suite(
                CONFIG_PATH,
                output,
                figure_ids=[8],
                formats=["png"],
                dpi=80,
            )
            figure_dir = output / "fig08"
            required = [
                figure_dir / "panel_text.json",
                figure_dir / "chart_contract.json",
                figure_dir / "run_manifest.json",
                figure_dir / "panel_data/primary_indicators.csv",
                figure_dir / "panels/fig08_a.png",
                figure_dir / "panels/fig08_a.svg",
                figure_dir / "fig08_full.png",
                figure_dir / "image_assets/image_asset_manifest.json",
            ]
            self.assertTrue(all(path.is_file() for path in required))
            manifest = json.loads(
                (figure_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("complete_architecture_only", manifest["status"])


if __name__ == "__main__":
    unittest.main()

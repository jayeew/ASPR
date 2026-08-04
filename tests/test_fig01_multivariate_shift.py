"""Checks for the multivariate feature-space variant of Fig. 1."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "outputs" / "fig01" / "new"
PANEL_DATA = OUTPUT / "panel_data"
EXPECTED_FEATURES = {
    "EF0017",
    "EF0240",
    "EF0309",
    "EF0312",
    "EF0315",
    "EF0318",
}
EXPECTED_DIMENSIONS = {"CD029", "CD031", "CD032"}
EXPECTED_DOMAINS = {
    "crispr",
    "graphene_2d_materials",
    "click_chemistry_cuaac",
    "genome_wide_association_studies",
}


class MultivariateShiftFigureTests(unittest.TestCase):
    """Validate the frozen pool, statistics, and unchanged graph panels."""

    def setUp(self) -> None:
        self.pool = pd.read_csv(
            PANEL_DATA / "multivariate_feature_pool.csv"
        )
        self.displacement = pd.read_csv(
            PANEL_DATA / "multivariate_stage_displacement.csv"
        )
        self.contributions = pd.read_csv(
            PANEL_DATA / "multivariate_dimension_contributions.csv"
        )
        self.placebos = pd.read_csv(
            PANEL_DATA / "multivariate_placebos.csv"
        )
        self.manifest = json.loads(
            (OUTPUT / "analysis_manifest_multivariate.json").read_text(
                encoding="utf-8"
            )
        )

    def test_core_pool_is_fixed_and_not_effect_selected(self) -> None:
        self.assertEqual(set(self.pool["feature_id"]), EXPECTED_FEATURES)
        self.assertEqual(
            set(self.pool["dimension_id"]), EXPECTED_DIMENSIONS
        )
        self.assertEqual(
            set(self.pool["tier"]),
            {
                "source_formula_existing",
                "source_formula_local_surrogate",
            },
        )
        self.assertTrue(
            self.pool["materialization_status"].eq("materialized").all()
        )
        self.assertEqual(
            int(self.pool["tier"].eq("source_formula_existing").sum()), 4
        )
        self.assertEqual(
            int(
                self.pool["tier"]
                .eq("source_formula_local_surrogate")
                .sum()
            ),
            2,
        )
        self.assertFalse(self.manifest["outcome_used_for_selection"])
        self.assertFalse(self.manifest["future_information_used"])

    def test_equal_feature_and_dimension_weights(self) -> None:
        feature_weight_sums = self.pool.groupby("dimension_id")[
            "feature_weight_within_dimension"
        ].sum()
        self.assertTrue(
            np.allclose(feature_weight_sums.to_numpy(dtype=float), 1.0)
        )
        dimension_weights = (
            self.pool[["dimension_id", "dimension_weight"]]
            .drop_duplicates()
            .set_index("dimension_id")["dimension_weight"]
        )
        self.assertEqual(set(dimension_weights.index), EXPECTED_DIMENSIONS)
        self.assertTrue(
            np.allclose(dimension_weights.to_numpy(dtype=float), 1.0 / 3.0)
        )

    def test_stage_rows_and_intervals_are_complete(self) -> None:
        self.assertEqual(set(self.displacement["domain"]), EXPECTED_DOMAINS)
        self.assertEqual(
            set(self.displacement["stage"]),
            {"pre", "landmark", "early", "late"},
        )
        self.assertTrue(
            self.displacement.groupby("domain").size().eq(4).all()
        )
        self.assertTrue(
            (
                self.displacement["ci_low"]
                <= self.displacement["displacement_pp"]
            ).all()
        )
        self.assertTrue(
            (
                self.displacement["displacement_pp"]
                <= self.displacement["ci_high"]
            ).all()
        )
        pre = self.displacement.loc[self.displacement["stage"].eq("pre")]
        self.assertTrue(pre["displacement_pp"].eq(0).all())
        post = self.displacement.loc[~self.displacement["stage"].eq("pre")]
        self.assertTrue(post["placebo_n"].gt(0).all())
        self.assertTrue(
            (
                post["placebo_low"]
                <= post["placebo_median"]
            ).all()
        )
        self.assertTrue(
            (
                post["placebo_median"]
                <= post["placebo_high"]
            ).all()
        )

    def test_dimension_contributions_reconstruct_displacement(self) -> None:
        post = self.contributions.loc[
            ~self.contributions["stage"].eq("pre")
        ]
        shares = post.groupby(["domain", "stage"])[
            "contribution_share"
        ].sum()
        self.assertTrue(np.allclose(shares.to_numpy(dtype=float), 1.0))
        reconstructed = (
            post.assign(square=lambda frame: frame["dimension_rms_pp"] ** 2)
            .groupby(["domain", "stage"])["square"]
            .mean()
            .pow(0.5)
        )
        reported = self.displacement.set_index(["domain", "stage"])[
            "displacement_pp"
        ].loc[reconstructed.index]
        self.assertTrue(
            np.allclose(
                reconstructed.to_numpy(dtype=float),
                reported.to_numpy(dtype=float),
            )
        )
        observed = self.displacement.set_index(["domain", "stage"])[
            "displacement_pp"
        ]
        allocated = (
            post.assign(
                allocated_pp=lambda frame: (
                    frame["contribution_share"]
                    * frame.set_index(["domain", "stage"]).index.map(observed)
                )
            )
            .groupby(["domain", "stage"])["allocated_pp"]
            .sum()
        )
        self.assertTrue(
            np.allclose(
                allocated.to_numpy(dtype=float),
                observed.loc[allocated.index].to_numpy(dtype=float),
            )
        )

    def test_placebo_rows_are_auditable(self) -> None:
        self.assertEqual(set(self.placebos["focal_domain"]), EXPECTED_DOMAINS)
        self.assertTrue(
            set(self.placebos["placebo_kind"]).issubset(
                {
                    "same_domain_historical",
                    "contemporaneous_control",
                }
            )
        )
        self.assertFalse(
            self.placebos.duplicated(
                [
                    "focal_domain",
                    "placebo_domain",
                    "placebo_start_year",
                    "stage",
                ]
            ).any()
        )

    def test_render_bundle_and_network_pixels(self) -> None:
        for extension in ("png", "svg", "pdf"):
            path = OUTPUT / f"figure_full_multivariate_shift.{extension}"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
        canonical = Image.open(OUTPUT / "figure_full.png").convert("RGB")
        revised = Image.open(
            OUTPUT / "figure_full_multivariate_shift.png"
        ).convert("RGB")
        self.assertEqual(canonical.size, revised.size)
        self.assertEqual(revised.size, (4322, 3968))
        crop = (0, 0, int(canonical.width * 0.55), canonical.height)
        difference = ImageChops.difference(
            canonical.crop(crop), revised.crop(crop)
        )
        self.assertIsNone(difference.getbbox())
        render_manifest = json.loads(
            (OUTPUT / "render_manifest_multivariate.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            render_manifest["chart_type"], "decomposed_bullet_forest"
        )
        self.assertEqual(
            render_manifest["design_version"],
            "fig1-multivariate-shift-v8.3",
        )
        self.assertEqual(
            render_manifest["layout_qa"]["out_of_canvas_text_count"], 0
        )


if __name__ == "__main__":
    unittest.main()

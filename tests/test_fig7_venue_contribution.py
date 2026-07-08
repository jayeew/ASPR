from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig7.build_fig7_venue_contribution import (
    build_pairwise_contribution_tests,
    build_portfolio_tables,
    compose_full_figure,
    field_year_zscore,
    map_venue_family,
)


class Fig7VenueContributionTests(unittest.TestCase):
    def test_venue_family_mapping_rules(self) -> None:
        self.assertEqual(
            map_venue_family("Nature Communications", "Springer Nature", ["Springer Nature"]).family,
            "Nature Portfolio",
        )
        self.assertEqual(map_venue_family("Science Advances", "AAAS", []).family, "Science family")
        self.assertEqual(map_venue_family("Cell", "Cell Press", ["Cell Press", "Elsevier BV"]).family, "Cell Press")
        self.assertEqual(map_venue_family("Proceedings of the National Academy of Sciences", "", []).family, "PNAS")
        self.assertEqual(map_venue_family("The Lancet Oncology", "Elsevier BV", ["Elsevier BV"]).family, "Lancet family")
        self.assertEqual(map_venue_family("Journal of Catalysis", "Elsevier BV", ["Elsevier BV"]).family, "Elsevier")

    def test_field_year_zscore_and_enrichment_are_finite(self) -> None:
        df = pd.DataFrame(
            {
                "primary_field": ["Biology"] * 8 + ["Physics"] * 8,
                "year": [2020] * 4 + [2021] * 4 + [2020] * 4 + [2021] * 4,
                "score": np.arange(16, dtype=float),
            }
        )
        z = field_year_zscore(df, "score", min_group=4)
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(float(z.iloc[:4].mean()), 0.0, places=7)

    def test_portfolio_tables_rank_high_signal_family(self) -> None:
        rows = []
        for i in range(30):
            rows.append(
                {
                    "paper_id": f"n{i}",
                    "venue_family_plot": "Nature Portfolio",
                    "source_id": "S1",
                    "article_type": "article",
                    "reference_count": 20,
                    "publication_day_signal_controlled": 1.0 + i / 100.0,
                    "future_impact_controlled": 0.5,
                    "B_z_controlled": 0.2,
                    "RS_z_controlled": 0.2,
                    "DeltaQ0_z_controlled": 0.2,
                    "Uzzi_z_controlled": 0.2,
                    "RTD_z_controlled": 0.2,
                    "BurtIP_z_controlled": 0.2,
                    "PDE_z_controlled": 0.2,
                }
            )
        for i in range(30):
            rows.append(
                {
                    "paper_id": f"o{i}",
                    "venue_family_plot": "Other publishers",
                    "source_id": "S2",
                    "article_type": "article",
                    "reference_count": 20,
                    "publication_day_signal_controlled": -0.5 + i / 200.0,
                    "future_impact_controlled": 0.1,
                    "B_z_controlled": -0.1,
                    "RS_z_controlled": -0.1,
                    "DeltaQ0_z_controlled": -0.1,
                    "Uzzi_z_controlled": -0.1,
                    "RTD_z_controlled": -0.1,
                    "BurtIP_z_controlled": -0.1,
                    "PDE_z_controlled": -0.1,
                }
            )
        portfolio, rankings, enrichment, mechanism, prepost = build_portfolio_tables(pd.DataFrame(rows))
        self.assertEqual(str(portfolio.iloc[0]["venue_family"]), "Nature Portfolio")
        pairwise = build_pairwise_contribution_tests(pd.DataFrame(rows), portfolio, n_boot=200)
        self.assertFalse(pairwise.empty)
        self.assertEqual(str(pairwise.iloc[0]["comparator"]), "Other publishers")
        self.assertGreater(float(pairwise.iloc[0]["aggregate_diff_ci_low"]), 0.0)
        nature_top5 = enrichment.loc[
            enrichment["venue_family"].eq("Nature Portfolio") & enrichment["top_k"].eq("top_5pct")
        ].iloc[0]
        self.assertGreater(float(nature_top5["enrichment_ratio"]), 1.0)
        self.assertFalse(rankings.empty)
        self.assertFalse(mechanism.empty)
        self.assertFalse(prepost.empty)

    def test_compose_full_figure_uses_compact_four_panel_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            panel_paths = []
            for idx in range(6):
                path = out_dir / f"panel_{idx}.png"
                Image.new("RGB", (1400, 920), (255, 255, 255)).save(path)
                panel_paths.append(path)
            audit = pd.DataFrame(
                [
                    {"audit_item": "nature_rank", "status": "pass", "value": "1", "note": "Nature ranks first."},
                    {"audit_item": "strict_interval_separation", "status": "gap", "value": "0", "note": "Intervals overlap."},
                    {"audit_item": "pairwise_aggregate_difference", "status": "gap", "value": "0", "note": "Pairwise gap."},
                    {"audit_item": "field_year_normalization", "status": "pass", "value": "yes", "note": "Controlled."},
                ]
            )
            portfolio = pd.DataFrame(
                [
                    {"venue_family": "Nature Portfolio", "vci": 2.0, "vci_rank": 1, "n_papers": 20},
                    {"venue_family": "Science family", "vci": 1.0, "vci_rank": 2, "n_papers": 15},
                    {"venue_family": "APS", "vci": 0.5, "vci_rank": 3, "n_papers": 12},
                ]
            )
            out = out_dir / "fig7_full.png"

            compose_full_figure(panel_paths, out, headline_supported=True, strict_claim=False, audit=audit, portfolio=portfolio)

            self.assertTrue(out.exists())
            self.assertTrue((out_dir / "fig7_panel_compact_audit.png").exists())
            with Image.open(out) as image:
                self.assertEqual((2902, 2072), image.size)


if __name__ == "__main__":
    unittest.main()

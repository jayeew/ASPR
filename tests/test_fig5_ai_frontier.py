from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.kg_perturbation_fig5.build_fig5_ai_frontier import (
    add_scores_and_positions,
    build_quality_gates,
    build_term_table,
    classify_theme,
    local_frontier_rows,
    render_image2_prompt,
    select_point_cloud,
)


class Fig5AIFrontierTests(unittest.TestCase):
    def test_local_frontier_rows_keep_only_2024_2026_ai_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_fig5_ai_") as tmp:
            path = Path(tmp) / "papers_master.csv"
            pd.DataFrame(
                [
                    {
                        "paper_id": "p1",
                        "title": "Large language model agents for scientific discovery",
                        "year": 2024,
                        "cited_by_count": 10,
                        "domain": "systems",
                        "topic_label": "Foundation models for science",
                    },
                    {
                        "paper_id": "p2",
                        "title": "A non-AI materials synthesis report",
                        "year": 2024,
                        "cited_by_count": 5,
                        "domain": "materials",
                        "topic_label": "Materials synthesis",
                    },
                    {
                        "paper_id": "p3",
                        "title": "Machine learning for old datasets",
                        "year": 2023,
                        "cited_by_count": 5,
                        "domain": "methods",
                        "topic_label": "Machine learning",
                    },
                ]
            ).to_csv(path, index=False)

            rows = local_frontier_rows(path, 2024, 2026)

        self.assertEqual(1, len(rows))
        self.assertEqual("p1", rows[0]["paper_id"])
        self.assertEqual(1, rows[0]["title_topic_ai_match"])

    def test_quality_gate_passes_dense_source_backed_ai_frontier(self) -> None:
        rows = []
        labels = [
            ("large_language_model", "large language model", "AI for scientific discovery"),
            ("foundation_model", "foundation model", "AI-enabled materials discovery"),
            ("generative_ai", "generative AI", "AI-enabled biomedicine"),
            ("diffusion_model", "diffusion model", "AI-enabled chemistry"),
            ("deep_learning", "deep learning", "AI-enabled genomics and cells"),
            ("machine_learning", "machine learning", "AI-enabled astronomy"),
            ("artificial_intelligence", "artificial intelligence", "AI-enabled climate and Earth science"),
            ("neural_network", "neural network", "General-purpose AI methods"),
            ("multimodal_ai", "multimodal AI", "General-purpose AI methods"),
        ]
        for idx in range(90):
            term_id, term_label, theme = labels[idx % len(labels)]
            rows.append(
                {
                    "source": "openalex_works_search",
                    "source_query": term_label,
                    "source_url": f"https://api.openalex.org/works?page={idx}",
                    "paper_id": f"W{idx}",
                    "doi": "",
                    "title": f"{term_label} paper {idx}",
                    "year": 2024 + (idx % 3),
                    "cited_by_count": idx,
                    "domain": "openalex",
                    "topic_label": theme,
                    "ai_term_ids": term_id,
                    "ai_terms": term_label,
                    "theme_id": f"theme_{idx % 6}",
                    "theme_label": theme,
                    "title_topic_ai_match": 1,
                }
            )
        frontier = add_scores_and_positions(pd.DataFrame(rows), 2026)
        point_cloud = frontier.head(80)
        terms = build_term_table(frontier)

        gates = build_quality_gates(frontier, point_cloud, terms)

        self.assertTrue(gates["overall_pass"])
        self.assertEqual("source_backed_ai_frontier_ready", gates["status_label"])
        self.assertEqual(1, gates["checks"]["top_points_ai_relevance_all"])
        self.assertGreaterEqual(gates["counts"]["ai_terms"], 8)

    def test_image_prompt_removes_take_home_footer(self) -> None:
        point_cloud = pd.DataFrame(
            [
                {
                    "title": "Large language model for scientific discovery",
                    "year": 2024,
                    "theme_label": "AI for scientific discovery",
                    "frontier_score": 0.9,
                    "source": "openalex_works_search",
                }
            ]
        )
        terms = pd.DataFrame({"term_label": ["large language model"]})
        gates = {"status_label": "source_backed_ai_frontier_ready"}

        prompt = render_image2_prompt(point_cloud, terms, gates)

        self.assertIn('"no_take_home_footer": true', prompt)
        self.assertIn("dense AI/AI-enabled science frontier point cloud", prompt)

    def test_application_query_fallback_does_not_force_invisible_theme(self) -> None:
        theme_id, theme_label = classify_theme("A survey of large language models Topic Modeling", fallback="chemistry")

        self.assertEqual("general_methods", theme_id)
        self.assertEqual("General-purpose AI methods", theme_label)

    def test_point_cloud_selection_preserves_theme_coverage(self) -> None:
        rows = []
        for theme_idx in range(6):
            for item_idx in range(12):
                rows.append(
                    {
                        "paper_id": f"t{theme_idx}_{item_idx}",
                        "theme_id": f"theme_{theme_idx}",
                        "theme_label": f"Theme {theme_idx}",
                        "frontier_score": 1.0 - theme_idx * 0.1 - item_idx * 0.001,
                        "cited_by_count": 100 - item_idx,
                        "year": 2024,
                    }
                )
        frontier = pd.DataFrame(rows)

        point_cloud = select_point_cloud(frontier, 30)

        self.assertGreaterEqual(point_cloud["theme_id"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()

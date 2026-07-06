from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig1.fig1_knowledge_perturbation_v3 import (  # noqa: E402
    apply_deterministic_hybrid_edge_sampling,
    build_edge_sampling_manifest_row,
    effective_main_cumulative_horizon_years,
)


class Fig1SamplingHorizonTests(unittest.TestCase):
    def test_deterministic_sampling_preserves_direct_edges_and_records_manifest(self) -> None:
        graph = nx.Graph()
        for idx in range(8):
            graph.add_node(f"p{idx}", year=2000 + idx % 3)
        graph.add_edge("p0", "p1", weight=9.0, direct=1)
        graph.add_edge("p1", "p2", weight=8.0, direct=1)
        for idx in range(2, 8):
            graph.add_edge(f"p{idx - 1}", f"p{idx}", weight=1.0 + idx, bibliographic=idx)
        graph.add_edge("p0", "p7", weight=2.0, cocitation=3)

        cfg = {
            "deterministic_hybrid_sampling": True,
            "sampling_target_edges": 5,
            "sampling_seed": 20260630,
        }
        sampled = apply_deterministic_hybrid_edge_sampling(graph, cfg)
        manifest = build_edge_sampling_manifest_row("toy", graph, sampled, cfg)

        self.assertEqual(5, sampled.number_of_edges())
        self.assertTrue(sampled.has_edge("p0", "p1"))
        self.assertTrue(sampled.has_edge("p1", "p2"))
        self.assertEqual(1, manifest["sampling_applied"])
        self.assertEqual(2, manifest["direct_edges_preserved"])
        self.assertLess(float(manifest["hybrid_sampling_fraction"]), 1.0)

        sampled_again = apply_deterministic_hybrid_edge_sampling(graph, cfg)
        self.assertEqual(sorted(sampled.edges()), sorted(sampled_again.edges()))

    def test_effective_main_horizon_uses_explicit_common_horizon(self) -> None:
        cfg = {"metrics": {"common_cumulative_horizon_years": 15}}
        self.assertEqual(15, effective_main_cumulative_horizon_years(cfg, [(1995, 2005), (1995, 2024)]))
        self.assertEqual(30, effective_main_cumulative_horizon_years({}, [(1995, 2005), (1995, 2024)]))


if __name__ == "__main__":
    unittest.main()

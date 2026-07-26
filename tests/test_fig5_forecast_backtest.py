from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig05.old.fig5_forecast_outcomes import (  # noqa: E402
    Fig5Tables,
    add_focus_scores,
    add_backtest_baseline_columns,
    build_fig5_quality_report,
    ndcg_at_k,
    write_outputs,
)


class Fig5ForecastBacktestTests(unittest.TestCase):
    def test_backtest_metrics_add_precision_ndcg_and_baseline_delta(self) -> None:
        table = pd.DataFrame(
            {
                "window": ["2010->2015", "2010->2015", "2010->2015"],
                "method": ["graph_score", "growth_only", "random"],
                "top10_hit_rate": [0.6, 0.4, 0.1],
                "ndcg_at_10": [0.7, 0.5, 0.2],
            }
        )

        enriched = add_backtest_baseline_columns(table)
        graph = enriched[enriched["method"].eq("graph_score")].iloc[0]

        self.assertEqual(0.6, graph["precision_at_10"])
        self.assertEqual("growth_only", graph["baseline_method"])
        self.assertAlmostEqual(0.2, graph["delta_precision_at_10"])
        self.assertAlmostEqual(0.2, graph["delta_ndcg_at_10"])

    def test_ndcg_at_k_rewards_correct_ranked_hits(self) -> None:
        self.assertGreater(ndcg_at_k(["a", "b", "c"], ["a", "x", "c"], 3), 0.0)
        self.assertEqual(0.0, ndcg_at_k(["a", "b", "c"], ["x", "y", "z"], 3))

    def test_growth_baseline_uses_only_pre_cutoff_history(self) -> None:
        focus = pd.DataFrame(
            {
                "historical_size": [10, 10, 10],
                "historical_citations": [5, 15, 30],
                "hist_scored_papers": [10, 10, 10],
                "hist_top_tail_score": [0.1, 0.2, 0.3],
                "hist_landmarks": [0, 1, 2],
                "recent_hist_size": [1, 3, 6],
                "prior_hist_size": [5, 5, 5],
                "future_papers": [100, 1, 1],
                "future_citations": [100, 1, 1],
                "future_rgpm_top_tail": [1, 0, 0],
                "future_landmarks": [1, 0, 0],
            }
        )
        changed_future = focus.copy()
        changed_future["future_papers"] = [1, 100, 1]
        changed_future["future_citations"] = [1, 100, 1]

        add_focus_scores(focus, min_historical_papers=5)
        add_focus_scores(changed_future, min_historical_papers=5)

        self.assertEqual(focus["growth_only_score"].round(12).tolist(), changed_future["growth_only_score"].round(12).tolist())
        self.assertNotEqual(focus["realized_score"].round(12).tolist(), changed_future["realized_score"].round(12).tolist())

    def test_write_outputs_exports_nature_ready_backtest_contract(self) -> None:
        tables = Fig5Tables(
            focus=pd.DataFrame({"focus_id": ["f1"], "predicted_score": [0.9]}),
            predicted_focus=pd.DataFrame({"focus_id": ["f1"], "predicted_rank": [1]}),
            realized_focus=pd.DataFrame({"focus_id": ["f1"], "realized_rank": [1]}),
            alignment=pd.DataFrame({"focus_id": ["f1"], "hit_type": ["exact_hit"]}),
            key_innovations=pd.DataFrame({"innovation_id": ["i1"], "focus_id": ["f1"]}),
            backtest=pd.DataFrame(
                {
                    "window": ["2010:2015"],
                    "method": ["graph_score"],
                    "precision_at_10": [0.7],
                    "ndcg_at_10": [0.8],
                    "baseline_precision_at_10": [0.3],
                }
            ),
            summary={"warnings": [], "backtest_ready": True},
        )
        args = argparse.Namespace(
            fig3_run_dir=Path("fig3"),
            fig3_input_dir=Path("fig3_input"),
            out_dir=Path("unused"),
            domain_filter=None,
            cutoff_year=2020,
            validation_start=2021,
            validation_end=2025,
            top_n=10,
            case_count=4,
            min_historical_papers=5,
            min_future_papers=2,
            backtest_windows=["2010:2015"],
            seed=2028,
        )
        data = argparse.Namespace(score_col="S_w_oof", rgpm_col="RGPM", min_year=2000, max_year=2025)

        with tempfile.TemporaryDirectory(prefix="aspr_fig5_") as tmp:
            out_dir = Path(tmp)
            args.out_dir = out_dir
            write_outputs(out_dir, tables, args, data)

            self.assertTrue((out_dir / "fig5_backtest_focus.csv").exists())
            self.assertTrue((out_dir / "fig5_alignment_metrics.csv").exists())
            self.assertTrue((out_dir / "fig5_failure_cases.csv").exists())
            report = build_fig5_quality_report(out_dir)

        self.assertTrue(report["quality_gates"]["checks"]["backtest_table_present"])
        self.assertTrue(report["quality_gates"]["checks"]["alignment_metrics_present"])


if __name__ == "__main__":
    unittest.main()

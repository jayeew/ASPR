from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.fig06.old.build_fig6_robustness as fig6_module  # noqa: E402
from experiments.fig06.old.build_fig6_robustness import (  # noqa: E402
    FULL_RERUN_INDICATOR_STABILITY,
    FULL_RERUN_MANIFEST,
    FULL_RERUN_PRIMARY_MODEL_STABILITY,
    FULL_RERUN_RANK_STABILITY,
    GRAPH_METRICS,
    build_cache_graph_perturbation_panel,
    build_cross_domain_panel_from_score_table,
    build_fig6_quality_report,
    build_full_graph_rerun_artifacts,
    build_reference_closure_drift_diagnostic,
    fetch_openalex_references_for_sample,
    fit_fig3_primary_model,
    load_full_rerun_artifacts,
    load_primary_model_stability_artifacts,
    merge_online_sample_references_with_cached_neighborhood,
    primary_model_stability_for_rerun,
    recompute_formal_fig3_indicators,
    write_full_rerun_fetch_failure,
    write_full_rerun_failure_cases,
    write_reference_stable_subset_diagnostic,
)
from experiments.fig03.old.fig3_empirical_weight_learning import (  # noqa: E402
    RawData,
    compute_indicator_and_delta_tables,
)


class Fig6RobustnessTests(unittest.TestCase):
    def test_formal_rerun_membership_filter_does_not_rebuild_work_id_set_per_paper(self) -> None:
        source = inspect.getsource(recompute_formal_fig3_indicators)

        self.assertNotIn('for paper_id in paper_ids if paper_id in set(works["id"])', source)
        self.assertIn("work_ids = set(", source)

    def test_fig3_formal_indicator_scan_can_be_limited_to_sample_ids(self) -> None:
        works = pd.DataFrame(
            [
                {"id": "r0", "title": "ref 0", "year": 1999, "primary_field": "biology", "display_community": 1, "domain": "d", "is_landmark": 0},
                {"id": "r1", "title": "ref 1", "year": 1999, "primary_field": "chemistry", "display_community": 2, "domain": "d", "is_landmark": 0},
                {"id": "extra", "title": "extra eligible", "year": 2001, "primary_field": "physics", "display_community": 3, "domain": "d", "is_landmark": 0},
                {"id": "sample", "title": "sample", "year": 2005, "primary_field": "materials", "display_community": 4, "domain": "d", "is_landmark": 0},
                {"id": "future", "title": "future", "year": 2007, "primary_field": "materials", "display_community": 5, "domain": "d", "is_landmark": 0},
            ]
        )
        citations = pd.DataFrame(
            [
                {"source": "extra", "target": "r0"},
                {"source": "extra", "target": "r1"},
                {"source": "sample", "target": "r0"},
                {"source": "sample", "target": "r1"},
                {"source": "sample", "target": "extra"},
                {"source": "future", "target": "sample"},
            ]
        )
        raw = RawData(
            works=works,
            citations=citations,
            topics=pd.DataFrame(),
            topic_edges=pd.DataFrame(),
            analysis_end_year=2010,
        )

        metrics, deltas = compute_indicator_and_delta_tables(
            raw,
            tau=1,
            min_refs=2,
            paper_id_filter={"sample"},
            progress=False,
        )

        self.assertEqual(["sample"], metrics["paper_id"].tolist())
        self.assertEqual(["sample"], deltas["paper_id"].tolist())

    def test_cache_graph_perturbation_panel_uses_indicator_level_inputs(self) -> None:
        n = 48
        rng = np.random.default_rng(20260629)
        indicator_rows = []
        future_rows = []
        for idx in range(n):
            domain = "domain_a" if idx < n // 2 else "domain_b"
            base = idx / n
            row = {
                "paper_id": f"p{idx:03d}",
                "domain": domain,
                "year": 2000 + idx % 6,
            }
            for metric_pos, metric in enumerate(GRAPH_METRICS):
                row[f"{metric}_z"] = base + metric_pos * 0.03 + float(rng.normal(0, 0.01))
            indicator_rows.append(row)
            future_rows.append(
                {
                    "paper_id": f"p{idx:03d}",
                    "RGPM": base + float(rng.normal(0, 0.02)),
                }
            )
        indicators = pd.DataFrame(indicator_rows)
        future = pd.DataFrame(future_rows)
        weights = {metric: 1.0 / len(GRAPH_METRICS) for metric in GRAPH_METRICS}

        panel = build_cache_graph_perturbation_panel(indicators, future, weights, seeds=[1, 2, 3], top_k=8)

        self.assertFalse(panel.empty)
        self.assertIn("drop_metric", set(panel["perturbation_type"]))
        self.assertIn("bootstrap_indicator_noise", set(panel["perturbation_type"]))
        self.assertTrue(panel["source_status"].str.contains("cached_indicator_rerun").all())
        self.assertTrue(panel["rank_spearman_mean"].between(-1.0, 1.0).all())
        self.assertTrue(panel["topk_jaccard_mean"].between(0.0, 1.0).all())
        self.assertTrue(panel["target_spearman_mean"].between(-1.0, 1.0).all())

    def test_cross_domain_panel_falls_back_to_current_fig3_score_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "fig3_score_table.csv"
            rows = []
            for domain_idx, domain in enumerate(["domain_a", "domain_b", "domain_c"]):
                for idx in range(20):
                    score = idx / 20 + domain_idx * 0.03
                    rows.append(
                        {
                            "domain": domain,
                            "S_w": score,
                            "S_w_oof": score + 0.01,
                            "RGPM": score + (idx % 3) * 0.01,
                        }
                    )
            pd.DataFrame(rows).to_csv(score_path, index=False)

            with patch.object(fig6_module, "SCORE_TABLE", score_path):
                panel = build_cross_domain_panel_from_score_table()

        self.assertFalse(panel.empty)
        self.assertTrue({"learned_oof_spearman", "graph_top10_mean", "score_coverage_norm", "n_papers"}.issubset(panel.columns))
        self.assertTrue(panel["source_status"].str.contains("derived_from_fig3_score_table").all())
        self.assertTrue(panel["learned_oof_spearman"].notna().all())

    def test_quality_report_blocks_strong_claim_until_full_graph_rerun(self) -> None:
        panel_review = pd.DataFrame(
            [
                {"panel": "A", "strength": "strong", "keep_decision": "keep"},
                {"panel": "B", "strength": "strong_with_proxy_label", "keep_decision": "keep"},
                {"panel": "G", "strength": "stronger_than_score_table_proxy_but_not_full_rerun", "keep_decision": "supporting_audit_csv"},
            ]
        )
        panel_g = pd.DataFrame(
            {
                "perturbation_type": ["drop_metric"],
                "source_status": ["cached_indicator_rerun"],
            }
        )

        quality = build_fig6_quality_report(panel_review=panel_review, panel_g=panel_g)

        checks = quality["quality_gates"]["checks"]
        self.assertEqual(1, checks["proxy_labels_preserved"])
        self.assertEqual(1, checks["cache_indicator_rerun_present"])
        self.assertEqual(1, checks["full_graph_rerun_gap_declared"])
        self.assertEqual(0, quality["quality_gates"]["nature_strong_claim_ready"])
        self.assertTrue(quality["overall_pass"])

    def test_quality_report_allows_strong_claim_with_full_rerun_stability(self) -> None:
        panel_review = pd.DataFrame(
            [
                {"panel": "A", "strength": "strong", "keep_decision": "keep"},
                {"panel": "G", "strength": "full_online_graph_rerun", "keep_decision": "keep"},
            ]
        )
        panel_g = pd.DataFrame(
            {
                "perturbation_type": ["full_openalex_rerun", "full_s2_rerun"],
                "source_status": ["full_graph_rerun", "full_graph_rerun"],
            }
        )
        full_rerun = pd.DataFrame(
            {
                "rerun_id": [
                    "openalex_seed1_direct",
                    "openalex_seed2_direct_bc",
                    "openalex_seed3_direct_bc_cocite",
                    "openalex_seed4_direct_bc",
                    "openalex_seed5_direct_bc_cocite",
                ],
                "source": ["openalex_api"] * 5,
                "metadata_refresh_mode": ["online_openalex_by_id"] * 5,
                "rerun_scope": ["full_graph_rebuild"] * 5,
                "edge_sampling_seed": [1, 2, 3, 4, 5],
                "graph_construction": [
                    "direct_only",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                ],
                "rank_spearman": [0.91, 0.86, 0.88, 0.90, 0.87],
                "learned_score_direction_preserved": [1, 1, 1, 1, 1],
            }
        )

        quality = build_fig6_quality_report(panel_review=panel_review, panel_g=panel_g, full_rerun=full_rerun)

        gates = quality["quality_gates"]
        self.assertEqual(1, gates["nature_strong_claim_ready"])
        self.assertEqual(1, gates["checks"]["full_graph_rerun_artifacts_present"])
        self.assertEqual(1, gates["checks"]["online_metadata_refresh_present"])
        self.assertEqual(1, gates["checks"]["full_graph_rebuild_scope_present"])
        self.assertEqual(1, gates["checks"]["edge_sampling_seeds_ge_5"])
        self.assertEqual(1, gates["checks"]["graph_construction_variants_ge_3"])
        self.assertEqual(1, gates["checks"]["rank_stability_ge_0_8"])

    def test_quality_report_uses_primary_model_stability_when_fig3_promotes_hgb(self) -> None:
        panel_review = pd.DataFrame(
            [
                {"panel": "A", "strength": "strong", "keep_decision": "keep"},
                {"panel": "G", "strength": "full_online_graph_rerun", "keep_decision": "keep"},
            ]
        )
        panel_g = pd.DataFrame(
            {
                "perturbation_type": ["full_openalex_rerun"],
                "source_status": ["full_graph_rerun"],
            }
        )
        rerun_ids = [f"openalex_seed{i}" for i in range(5)]
        full_rerun = pd.DataFrame(
            {
                "rerun_id": rerun_ids,
                "source": ["openalex_api"] * 5,
                "metadata_refresh_mode": ["online_openalex_by_id"] * 5,
                "rerun_scope": ["full_graph_rebuild"] * 5,
                "edge_sampling_seed": [1, 2, 3, 4, 5],
                "graph_construction": [
                    "direct_only",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                ],
                "rank_spearman": [0.42, 0.43, 0.44, 0.45, 0.46],
                "learned_score_direction_preserved": [1, 1, 1, 1, 1],
            }
        )
        primary = pd.DataFrame(
            {
                "rerun_id": rerun_ids,
                "rank_spearman": [0.91, 0.90, 0.89, 0.88, 0.87],
                "primary_score_direction_preserved": [1, 1, 1, 1, 1],
            }
        )

        quality = build_fig6_quality_report(
            panel_review=panel_review,
            panel_g=panel_g,
            full_rerun=full_rerun,
            primary_model_stability=primary,
            fig3_primary_model="metadata_hgb_no_leakage",
        )

        gates = quality["quality_gates"]
        self.assertEqual(0, gates["checks"]["rank_stability_ge_0_8"])
        self.assertEqual(1, gates["checks"]["primary_model_rank_stability_ge_0_8"])
        self.assertEqual(1, gates["checks"]["robustness_rank_gate_pass"])
        self.assertEqual(1, gates["nature_strong_claim_ready"])

    def test_quality_report_requires_primary_model_artifact_for_hgb_primary(self) -> None:
        panel_review = pd.DataFrame(
            [
                {"panel": "A", "strength": "strong", "keep_decision": "keep"},
                {"panel": "G", "strength": "full_online_graph_rerun", "keep_decision": "keep"},
            ]
        )
        panel_g = pd.DataFrame({"perturbation_type": ["full_openalex_rerun"], "source_status": ["full_graph_rerun"]})
        full_rerun = pd.DataFrame(
            {
                "rerun_id": [f"openalex_seed{i}" for i in range(5)],
                "source": ["openalex_api"] * 5,
                "metadata_refresh_mode": ["online_openalex_by_id"] * 5,
                "rerun_scope": ["full_graph_rebuild"] * 5,
                "edge_sampling_seed": [1, 2, 3, 4, 5],
                "graph_construction": [
                    "direct_only",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                ],
                "rank_spearman": [0.91, 0.90, 0.89, 0.88, 0.87],
                "learned_score_direction_preserved": [1, 1, 1, 1, 1],
            }
        )

        quality = build_fig6_quality_report(
            panel_review=panel_review,
            panel_g=panel_g,
            full_rerun=full_rerun,
            fig3_primary_model="metadata_hgb_no_leakage",
        )

        gates = quality["quality_gates"]
        self.assertEqual(1, gates["checks"]["requires_primary_model_stability"])
        self.assertEqual(0, gates["checks"]["primary_model_stability_artifacts_present"])
        self.assertEqual(0, gates["nature_strong_claim_ready"])

    def test_quality_report_rejects_full_rerun_without_online_rebuild_provenance(self) -> None:
        panel_review = pd.DataFrame(
            [
                {"panel": "A", "strength": "strong", "keep_decision": "keep"},
                {"panel": "G", "strength": "full_online_graph_rerun", "keep_decision": "keep"},
            ]
        )
        panel_g = pd.DataFrame(
            {
                "perturbation_type": ["full_openalex_rerun"],
                "source_status": ["full_graph_rerun"],
            }
        )
        full_rerun = pd.DataFrame(
            {
                "rerun_id": ["cached_seed1"],
                "source": ["openalex_cached_snapshot"],
                "metadata_refresh_mode": ["local_snapshot_only"],
                "rerun_scope": ["score_table_perturbation"],
                "edge_sampling_seed": [1],
                "graph_construction": ["direct_only"],
                "rank_spearman": [0.99],
                "learned_score_direction_preserved": [1],
            }
        )

        quality = build_fig6_quality_report(panel_review=panel_review, panel_g=panel_g, full_rerun=full_rerun)

        gates = quality["quality_gates"]
        self.assertEqual(0, gates["nature_strong_claim_ready"])
        self.assertEqual(0, gates["checks"]["online_metadata_refresh_present"])
        self.assertEqual(0, gates["checks"]["full_graph_rebuild_scope_present"])
        self.assertEqual(0, gates["checks"]["edge_sampling_seeds_ge_5"])
        self.assertEqual(0, gates["checks"]["graph_construction_variants_ge_3"])

    def test_quality_report_labels_online_rerun_as_unstable_when_rank_gate_fails(self) -> None:
        panel_review = pd.DataFrame(
            [
                {"panel": "A", "strength": "strong", "keep_decision": "keep"},
                {"panel": "G", "strength": "full_online_graph_rerun", "keep_decision": "keep"},
            ]
        )
        panel_g = pd.DataFrame({"perturbation_type": ["full_openalex_rerun"], "source_status": ["full_graph_rerun"]})
        full_rerun = pd.DataFrame(
            {
                "rerun_id": [f"openalex_seed{i}" for i in range(5)],
                "source": ["openalex_api"] * 5,
                "metadata_refresh_mode": ["online_openalex_by_id"] * 5,
                "rerun_scope": ["full_graph_rebuild"] * 5,
                "edge_sampling_seed": [1, 2, 3, 4, 5],
                "graph_construction": [
                    "direct_only",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                    "direct_plus_bc",
                    "direct_plus_bc_cocitation",
                ],
                "rank_spearman": [0.42, 0.43, 0.44, 0.45, 0.46],
                "learned_score_direction_preserved": [1, 1, 1, 1, 1],
            }
        )

        quality = build_fig6_quality_report(panel_review=panel_review, panel_g=panel_g, full_rerun=full_rerun)

        self.assertEqual("full_graph_rerun_unstable", quality["status_label"])
        self.assertEqual(0, quality["quality_gates"]["nature_strong_claim_ready"])
        self.assertIn("rank stability", quality["quality_gates"]["replacement_gate"])

    def test_full_rerun_loader_requires_complete_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame(
                {
                    "rerun_id": ["openalex_seed1"],
                    "rank_spearman": [0.91],
                    "top_decile_jaccard": [0.83],
                    "learned_score_direction_preserved": [1],
                }
            ).to_csv(out_dir / "fig6_rank_stability.csv", index=False)

            incomplete = load_full_rerun_artifacts(out_dir)
            self.assertTrue(incomplete.empty)

            pd.DataFrame(
                {
                    "rerun_id": ["openalex_seed1"],
                    "source": ["openalex"],
                    "reference_closure": ["on"],
                    "edge_sampling_seed": [20260630],
                    "graph_construction": ["direct_plus_bc"],
                    "cutoff_year_delta": [0],
                    "metadata_refresh_mode": ["online_openalex_by_id"],
                    "rerun_scope": ["full_graph_rebuild"],
                    "n_sampled_papers": [12],
                    "n_refetched_works": [12],
                    "n_edges": [42],
                    "metadata_fetch_status": ["success"],
                    "graph_build_status": ["success"],
                    "indicator_status": ["success"],
                    "input_hash": ["sha256:abc"],
                }
            ).to_csv(out_dir / "fig6_full_rerun_manifest.csv", index=False)
            pd.DataFrame(
                {
                    "rerun_id": ["openalex_seed1"],
                    "metric": ["B"],
                    "baseline_mean": [0.10],
                    "rerun_mean": [0.12],
                    "delta": [0.02],
                    "direction_preserved": [1],
                }
            ).to_csv(out_dir / "fig6_indicator_stability.csv", index=False)

            complete = load_full_rerun_artifacts(out_dir)
            self.assertEqual(["openalex_seed1"], complete["rerun_id"].tolist())
            self.assertEqual("online_openalex_by_id", complete.loc[0, "metadata_refresh_mode"])
            self.assertEqual("full_graph_rebuild", complete.loc[0, "rerun_scope"])
            self.assertEqual(0.91, float(complete.loc[0, "rank_spearman"]))

    def test_primary_model_stability_loader_requires_successful_rerun_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame(
                {
                    "rerun_id": ["openalex_seed1"],
                    "source": ["openalex"],
                    "reference_closure": ["on"],
                    "edge_sampling_seed": [20260630],
                    "graph_construction": ["direct_plus_bc"],
                    "cutoff_year_delta": [0],
                    "metadata_refresh_mode": ["online_openalex_by_id"],
                    "rerun_scope": ["full_graph_rebuild"],
                    "n_sampled_papers": [12],
                    "n_refetched_works": [12],
                    "n_edges": [42],
                    "metadata_fetch_status": ["success"],
                    "graph_build_status": ["success"],
                    "indicator_status": ["success"],
                    "input_hash": ["sha256:abc"],
                }
            ).to_csv(out_dir / "fig6_full_rerun_manifest.csv", index=False)
            pd.DataFrame(
                {
                    "rerun_id": ["openalex_seed1", "failed_seed"],
                    "primary_model": ["metadata_hgb_no_leakage", "metadata_hgb_no_leakage"],
                    "feature_scope": [
                        "publication_day_graph_indicators_plus_year_reference_domain_field",
                        "publication_day_graph_indicators_plus_year_reference_domain_field",
                    ],
                    "n_scored_papers": [12, 0],
                    "rank_spearman": [0.93, np.nan],
                    "top_decile_jaccard": [0.80, np.nan],
                    "primary_score_direction_preserved": [1, 0],
                    "model_status": ["success", "failed_empty_rerun_metrics"],
                    "score_source": ["fig3_primary_model_contract", "fig3_primary_model_contract"],
                }
            ).to_csv(out_dir / "fig6_primary_model_stability.csv", index=False)

            loaded = load_primary_model_stability_artifacts(out_dir)

            self.assertEqual(["openalex_seed1"], loaded["rerun_id"].tolist())
            self.assertEqual(0.93, float(loaded.loc[0, "rank_spearman"]))
            self.assertEqual("online_openalex_by_id", loaded.loc[0, "metadata_refresh_mode"])

    def test_primary_model_stability_for_identical_rerun_scores_perfectly(self) -> None:
        n = 48
        rows = []
        for idx in range(n):
            row = {
                "paper_id": f"p{idx}",
                "domain": "domain_a" if idx < n // 2 else "domain_b",
                "primary_field": ["biology", "physics", "materials"][idx % 3],
                "year": 2000 + idx % 8,
                "reference_count": 5 + idx % 11,
                "RGPM": float(idx) / n,
            }
            for pos, metric in enumerate(GRAPH_METRICS):
                row[f"{metric}_z"] = float(idx) / n + pos * 0.02
            rows.append(row)
        training = pd.DataFrame(rows)
        fitted = fit_fig3_primary_model(training)
        if fitted.get("model_status") != "success":
            self.skipTest(f"primary model unavailable: {fitted.get('model_status')}")
        baseline = training.drop(columns=["RGPM"]).copy()
        rerun = baseline.copy()

        row, drift_rows = primary_model_stability_for_rerun(
            fitted_model=fitted,
            baseline=baseline,
            rerun_metrics=rerun,
            rerun_id="r1",
            source="openalex_api",
            edge_sampling_seed=20260630,
            graph_construction="direct_only",
        )

        self.assertEqual("success", row["model_status"])
        self.assertEqual(n, int(row["n_scored_papers"]))
        self.assertAlmostEqual(1.0, float(row["rank_spearman"]))
        self.assertEqual(n, len(drift_rows))
        self.assertIn("baseline_primary_score", drift_rows[0])
        self.assertIn("rerun_primary_score", drift_rows[0])

    def test_build_full_graph_rerun_artifacts_writes_online_graph_rebuild_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            works = pd.DataFrame(
                [
                    {
                        "id": f"p{i}",
                        "title": f"paper {i}",
                        "year": 2000 + (i % 4),
                        "domain": "domain_a" if i < 8 else "domain_b",
                        "primary_field": ["biology", "chemistry", "physics", "materials"][i % 4],
                        "display_community": i % 5,
                        "is_landmark": 0,
                    }
                    for i in range(16)
                ]
            )
            citation_rows = []
            for i in range(4, 16):
                for j in range(max(0, i - 4), i):
                    citation_rows.append({"source": f"p{i}", "target": f"p{j}"})
            citations = pd.DataFrame(citation_rows)
            baseline = works.copy().rename(columns={"id": "paper_id"})
            baseline["reference_count"] = baseline["paper_id"].map(citations.groupby("source").size()).fillna(0)
            for pos, metric in enumerate(GRAPH_METRICS):
                baseline[metric] = baseline["reference_count"] + pos * 0.1
                baseline[f"{metric}_z"] = (baseline[metric] - baseline[metric].mean()) / max(float(baseline[metric].std(ddof=0)), 1e-9)
            primary_training = pd.concat([baseline, baseline], ignore_index=True)
            primary_training["paper_id"] = [f"train_{idx}" for idx in range(len(primary_training))]
            primary_training["RGPM"] = np.linspace(0.0, 1.0, len(primary_training))
            weights = {metric: 1.0 / len(GRAPH_METRICS) for metric in GRAPH_METRICS}

            manifest, indicator, rank = build_full_graph_rerun_artifacts(
                works=works,
                citations=citations,
                baseline_indicators=baseline,
                weights=weights,
                out_dir=out_dir,
                source="openalex_api",
                metadata_refresh_mode="online_openalex_by_id",
                seeds=[1, 2, 3, 4, 5],
                graph_constructions=["direct_only", "direct_plus_bc", "direct_plus_bc_cocitation"],
                max_papers=12,
                primary_model_training=primary_training,
            )

            self.assertFalse(manifest.empty)
            self.assertFalse(indicator.empty)
            self.assertFalse(rank.empty)
            self.assertTrue((out_dir / "fig6_full_rerun_manifest.csv").exists())
            self.assertTrue((out_dir / "fig6_indicator_stability.csv").exists())
            self.assertTrue((out_dir / "fig6_rank_stability.csv").exists())
            self.assertTrue((out_dir / "fig6_full_rerun_paper_drift.csv").exists())
            self.assertTrue((out_dir / "fig6_primary_model_stability.csv").exists())
            self.assertTrue((out_dir / "fig6_primary_model_paper_drift.csv").exists())
            paper_drift = pd.read_csv(out_dir / "fig6_full_rerun_paper_drift.csv")
            self.assertFalse(paper_drift.empty)
            for column in [
                "rerun_id",
                "paper_id",
                "baseline_score",
                "rerun_score",
                "score_delta",
                "baseline_rank",
                "rerun_rank",
                "rank_delta",
                "baseline_top_decile",
                "rerun_top_decile",
                "B_delta",
                "PDE_delta",
            ]:
                self.assertIn(column, paper_drift.columns)
            self.assertEqual({"full_graph_rebuild"}, set(manifest["rerun_scope"]))
            self.assertEqual({"online_openalex_by_id"}, set(manifest["metadata_refresh_mode"]))
            self.assertGreaterEqual(manifest["edge_sampling_seed"].nunique(), 5)
            self.assertGreaterEqual(manifest["graph_construction"].nunique(), 3)
            primary_stability = pd.read_csv(out_dir / "fig6_primary_model_stability.csv")
            self.assertFalse(primary_stability.empty)
            self.assertIn("rank_spearman", primary_stability.columns)
            self.assertIn("model_status", primary_stability.columns)

    def test_full_rerun_uses_construction_matched_cached_baseline_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            paper_ids = [f"p{i}" for i in range(24)]
            works = pd.DataFrame(
                {
                    "id": paper_ids,
                    "year": [2000 + i % 4 for i in range(24)],
                    "domain": ["domain_a"] * 12 + ["domain_b"] * 12,
                    "primary_field": ["field_a", "field_b", "field_c", "field_d"] * 6,
                    "display_community": [i % 3 for i in range(24)],
                }
            )
            cached_citations = pd.DataFrame({"source": ["cached"], "target": ["ref"]})
            online_citations = pd.DataFrame({"source": ["online"], "target": ["ref"]})
            baseline = works.rename(columns={"id": "paper_id"}).copy()
            baseline["reference_count"] = 1
            for metric in GRAPH_METRICS:
                baseline[metric] = list(reversed(range(24)))
                baseline[f"{metric}_z"] = np.linspace(1.0, -1.0, 24)
            baseline["RGPM"] = np.linspace(0.0, 1.0, 24)
            weights = {metric: 1.0 / len(GRAPH_METRICS) for metric in GRAPH_METRICS}

            def fake_recompute(_works, citations, sample_ids, *, graph_construction, seed):
                values = np.arange(len(sample_ids), dtype=float)
                if citations is online_citations:
                    values = values + 0.1
                frame = pd.DataFrame(
                    {
                        "paper_id": sample_ids,
                        "year": [2000 + i % 4 for i in range(len(sample_ids))],
                        "domain": ["domain_a" if i < len(sample_ids) / 2 else "domain_b" for i in range(len(sample_ids))],
                        "primary_field": [["field_a", "field_b", "field_c", "field_d"][i % 4] for i in range(len(sample_ids))],
                        "reference_count": [1] * len(sample_ids),
                        "edge_count": [10] * len(sample_ids),
                    }
                )
                for metric in GRAPH_METRICS:
                    frame[metric] = values
                return frame

            with patch(
                "experiments.fig06.old.build_fig6_robustness.recompute_formal_fig3_indicators",
                side_effect=fake_recompute,
            ):
                _manifest, _indicator, rank = build_full_graph_rerun_artifacts(
                    works=works,
                    citations=online_citations,
                    baseline_indicators=baseline,
                    weights=weights,
                    out_dir=out_dir,
                    source="openalex_api",
                    metadata_refresh_mode="online_openalex_by_id",
                    seeds=[20260630],
                    graph_constructions=["direct_plus_bc"],
                    baseline_citations=cached_citations,
                    sample_ids=paper_ids,
                    primary_model_training=baseline,
                )

            self.assertGreater(float(rank.loc[0, "rank_spearman"]), 0.99)
            self.assertIn("baseline=", pd.read_csv(out_dir / "fig6_full_rerun_manifest.csv").loc[0, "input_hash"])

    def test_fetch_failure_preserves_existing_valid_full_rerun_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "rerun_id": "ok_rerun",
                        "source": "openalex_api",
                        "reference_closure": "on",
                        "edge_sampling_seed": 20260630,
                        "graph_construction": "direct_only",
                        "cutoff_year_delta": 0,
                        "metadata_refresh_mode": "online_openalex_by_id",
                        "rerun_scope": "full_graph_rebuild",
                        "n_sampled_papers": 3,
                        "n_refetched_works": 3,
                        "n_edges": 12,
                        "metadata_fetch_status": "success",
                        "graph_build_status": "success",
                        "indicator_status": "success",
                        "input_hash": "sha256:ok",
                    }
                ]
            ).to_csv(out_dir / FULL_RERUN_MANIFEST, index=False)
            pd.DataFrame(
                [
                    {
                        "rerun_id": "ok_rerun",
                        "metric": "B",
                        "baseline_mean": 0.2,
                        "rerun_mean": 0.22,
                        "delta": 0.02,
                        "direction_preserved": 1,
                    }
                ]
            ).to_csv(out_dir / FULL_RERUN_INDICATOR_STABILITY, index=False)
            pd.DataFrame(
                [
                    {
                        "rerun_id": "ok_rerun",
                        "rank_spearman": 0.91,
                        "top_decile_jaccard": 0.5,
                        "learned_score_direction_preserved": 1,
                    }
                ]
            ).to_csv(out_dir / FULL_RERUN_RANK_STABILITY, index=False)
            pd.DataFrame(
                [
                    {
                        "rerun_id": "ok_rerun",
                        "primary_model": "metadata_hgb_no_leakage",
                        "feature_scope": "publication_day_graph_indicators_plus_year_reference_domain_field",
                        "n_scored_papers": 3,
                        "rank_spearman": 0.92,
                        "top_decile_jaccard": 0.5,
                        "primary_score_direction_preserved": 1,
                        "model_status": "success",
                        "score_source": "fig3_primary_model_contract",
                    }
                ]
            ).to_csv(out_dir / FULL_RERUN_PRIMARY_MODEL_STABILITY, index=False)

            before = (out_dir / FULL_RERUN_MANIFEST).read_text(encoding="utf-8")
            write_full_rerun_fetch_failure(
                out_dir=out_dir,
                failure_row={
                    "rerun_id": "openalex_api_fetch_incomplete",
                    "source": "openalex_api",
                    "reference_closure": "on",
                    "edge_sampling_seed": 20260630,
                    "graph_construction": "direct_only",
                    "cutoff_year_delta": 0,
                    "metadata_refresh_mode": "online_openalex_by_id",
                    "rerun_scope": "full_graph_rebuild",
                    "n_sampled_papers": 3,
                    "n_refetched_works": 0,
                    "n_edges": 0,
                    "metadata_fetch_status": "failed_no_records",
                    "graph_build_status": "skipped",
                    "indicator_status": "skipped",
                    "input_hash": "sha256:failed",
                },
                online_citations=pd.DataFrame(columns=["source", "target"]),
            )

            after = (out_dir / FULL_RERUN_MANIFEST).read_text(encoding="utf-8")
            self.assertEqual(before, after)
            self.assertFalse(load_full_rerun_artifacts(out_dir).empty)
            self.assertFalse(load_primary_model_stability_artifacts(out_dir).empty)
            self.assertTrue((out_dir / "fig6_full_rerun_refresh_attempt.csv").exists())

    def test_openalex_fetch_splits_failed_batches_before_skipping_records(self) -> None:
        works = pd.DataFrame(
            {
                "id": ["domain::https://openalex.org/W1", "domain::https://openalex.org/W2"],
                "year": [2000, 2001],
                "domain": ["d", "d"],
                "primary_field": ["biology", "biology"],
                "display_community": [1, 2],
            }
        )
        sample_ids = works["id"].tolist()

        class FakeResponse:
            def __init__(self, status_code: int, payload: dict) -> None:
                self.status_code = status_code
                self._payload = payload

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    import requests

                    raise requests.HTTPError(f"status {self.status_code}")

            def json(self) -> dict:
                return self._payload

        calls: list[str] = []

        def fake_get(_url: str, params: dict, timeout: int) -> FakeResponse:
            filter_value = str(params["filter"])
            calls.append(filter_value)
            if "|" in filter_value:
                return FakeResponse(504, {"error": "Gateway timeout"})
            work_id = filter_value.rsplit(":", 1)[-1]
            return FakeResponse(
                200,
                {
                    "results": [
                        {
                            "id": f"https://openalex.org/{work_id}",
                            "referenced_works": ["https://openalex.org/W1"],
                        }
                    ]
                },
            )

        with patch("experiments.fig06.old.build_fig6_robustness.requests.get", side_effect=fake_get):
            citations, fetched, status = fetch_openalex_references_for_sample(works, sample_ids, batch_size=2)

        self.assertEqual("success", status)
        self.assertEqual(2, fetched)
        self.assertGreater(len(calls), 2)
        self.assertFalse(citations.empty)

    def test_online_references_replace_sample_edges_but_keep_cached_reference_neighborhood(self) -> None:
        cached = pd.DataFrame(
            [
                {"source": "p1", "target": "old_ref"},
                {"source": "ref_a", "target": "ref_b"},
                {"source": "other", "target": "old_ref"},
            ]
        )
        online = pd.DataFrame([{"source": "p1", "target": "new_ref"}])

        merged = merge_online_sample_references_with_cached_neighborhood(
            cached_citations=cached,
            online_sample_citations=online,
            sample_ids=["p1"],
        )

        edges = set(map(tuple, merged[["source", "target"]].to_numpy()))
        self.assertNotIn(("p1", "old_ref"), edges)
        self.assertIn(("p1", "new_ref"), edges)
        self.assertIn(("ref_a", "ref_b"), edges)
        self.assertIn(("other", "old_ref"), edges)

    def test_reference_closure_drift_diagnostic_reports_per_paper_overlap(self) -> None:
        cached = pd.DataFrame(
            [
                {"source": "p1", "target": "r1"},
                {"source": "p1", "target": "r2"},
                {"source": "p2", "target": "r3"},
            ]
        )
        online = pd.DataFrame(
            [
                {"source": "p1", "target": "r1"},
                {"source": "p1", "target": "r4"},
                {"source": "p2", "target": "r3"},
            ]
        )

        drift = build_reference_closure_drift_diagnostic(
            cached_citations=cached,
            online_sample_citations=online,
            sample_ids=["p1", "p2"],
        )

        p1 = drift.set_index("paper_id").loc["p1"]
        p2 = drift.set_index("paper_id").loc["p2"]
        self.assertEqual(2, int(p1["cached_ref_count"]))
        self.assertEqual(2, int(p1["online_ref_count"]))
        self.assertEqual(1, int(p1["overlap_count"]))
        self.assertAlmostEqual(1 / 3, float(p1["reference_jaccard"]))
        self.assertEqual(1.0, float(p2["reference_jaccard"]))

    def test_full_rerun_failure_cases_join_rank_and_reference_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame(
                {
                    "rerun_id": ["r1", "r1", "r2"],
                    "graph_construction": ["direct_only", "direct_only", "direct_plus_bc"],
                    "paper_id": ["p1", "p2", "p3"],
                    "baseline_rank": [100, 10, 50],
                    "rerun_rank": [1, 20, 180],
                    "rank_delta": [-99, 10, 130],
                    "baseline_score": [0.1, 0.8, 0.4],
                    "rerun_score": [1.2, 0.7, -0.2],
                    "score_delta": [1.1, -0.1, -0.6],
                }
            ).to_csv(out_dir / "fig6_full_rerun_paper_drift.csv", index=False)
            pd.DataFrame(
                {
                    "paper_id": ["p1", "p2", "p3"],
                    "reference_jaccard": [0.5, 1.0, 1.0],
                    "reference_count_delta": [0, 0, 2],
                }
            ).to_csv(out_dir / "fig6_reference_closure_drift.csv", index=False)

            failures = write_full_rerun_failure_cases(out_dir, top_n=2)

            self.assertEqual(["p3", "p1"], failures["paper_id"].tolist())
            self.assertEqual("reference_count_changed", failures.iloc[0]["failure_mode"])
            self.assertEqual("reference_closure_changed", failures.iloc[1]["failure_mode"])
            self.assertTrue((out_dir / "fig6_full_rerun_failure_cases.csv").exists())

    def test_reference_stable_subset_diagnostic_filters_rank_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            for idx in range(6):
                rows.append(
                    {
                        "rerun_id": "r1",
                        "graph_construction": "direct_only",
                        "paper_id": f"p{idx}",
                        "baseline_score": float(idx),
                        "rerun_score": float(idx) if idx < 4 else float(10 - idx),
                    }
                )
            pd.DataFrame(rows).to_csv(out_dir / "fig6_full_rerun_paper_drift.csv", index=False)
            pd.DataFrame(
                {
                    "paper_id": [f"p{idx}" for idx in range(6)],
                    "reference_jaccard": [1.0, 1.0, 0.95, 0.91, 0.5, 0.4],
                    "reference_count_delta": [0, 0, 0, 0, 3, 4],
                }
            ).to_csv(out_dir / "fig6_reference_closure_drift.csv", index=False)

            diagnostic = write_reference_stable_subset_diagnostic(out_dir)

            self.assertIn("jaccard_ge_0_9", set(diagnostic["filter"]))
            stable = diagnostic[
                diagnostic["filter"].eq("jaccard_ge_0_9")
                & diagnostic["graph_construction"].eq("direct_only")
            ].iloc[0]
            self.assertEqual(4, int(stable["n_min"]))
            self.assertAlmostEqual(1.0, float(stable["rank_spearman_min"]))
            self.assertTrue((out_dir / "fig6_reference_stable_subset_diagnostic.csv").exists())


if __name__ == "__main__":
    unittest.main()

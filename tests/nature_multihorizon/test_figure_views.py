from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gear.nature_multihorizon.artifact_store import hash_json
from gear.nature_multihorizon.contracts import AUXILIARY_FEATURES, CORE_FEATURES
from gear.nature_multihorizon.figure_views import (
    MECHANISM_COLUMNS,
    _claim_readiness,
    _forecast_score_view,
    _venue_family_summary,
    export_figure_views,
)
from gear.nature_multihorizon.release import build_release_manifest, publish_release
from experiments.common.old.kg_perturbation_v2.run_figure import validate_figure_view
from experiments.common.old.kg_perturbation_v2.renderers import _tables, render_figure
from experiments.common.old.kg_perturbation_v2.render_all_figures import render_all
from experiments.common.old.kg_perturbation_v2.build_final_assembly import (
    build_final_assembly,
)


class FigureViewTests(unittest.TestCase):
    def test_fig3_readiness_matches_all_four_locked_panels(self) -> None:
        metrics = []
        for model_id in (
            "domain_year_only",
            "bibliographic_aux10_ridge",
            "mechanism5_equal_weight",
            "mechanism5_simplex",
            "gam18",
            "hgb18",
            "rank_blend",
        ):
            metrics.append(
                {
                    "horizon": 5,
                    "model_id": model_id,
                    "scope": "development_oof_all_models",
                    "metric": "rho_global_calibrated",
                    "value": 0.5,
                    "sensitivity": "main",
                }
            )
        for horizon in (3, 5, 8):
            for metric in (
                "rho_global_calibrated",
                "rho_global_uncalibrated",
                "rho_domain_macro",
                "rho_conditional",
            ):
                metrics.append(
                    {
                        "horizon": horizon,
                        "model_id": "nested_selector",
                        "scope": "development_oof",
                        "metric": metric,
                        "value": 0.5,
                        "ci_low": 0.4,
                        "ci_high": 0.6,
                        "sensitivity": "main",
                    }
                )
            metrics.extend(
                [
                    {
                        "horizon": horizon,
                        "model_id": "nested_selector",
                        "scope": "development_oof",
                        "metric": "n_finite_oof",
                        "value": 5_000,
                        "sensitivity": "main",
                    },
                    {
                        "horizon": horizon,
                        "model_id": "nested_selector",
                        "scope": "development_oof",
                        "metric": "top_decile_enrichment",
                        "value": 5.0,
                        "ci_low": 4.0,
                        "ci_high": 6.0,
                        "sensitivity": "main",
                    },
                ]
            )
            for scope in (
                "sealed_temporal_holdout",
                "strict_label_availability__sealed_temporal_holdout",
            ):
                metrics.append(
                    {
                        "horizon": horizon,
                        "model_id": "gam18",
                        "scope": scope,
                        "metric": "rho_global_calibrated",
                        "value": 0.4,
                        "ci_low": 0.2,
                        "ci_high": 0.6,
                        "sensitivity": "main",
                    }
                )
        predictions = pd.DataFrame(
            [
                {
                    "paper_id": f"W{horizon}-{index}",
                    "horizon": horizon,
                    "is_selected": True,
                    "prediction_calibrated": index / 5_000,
                    "target_adjusted_oof": index / 5_000,
                }
                for horizon in (3, 5, 8)
                for index in range(5_000)
            ]
        )
        holdout = pd.DataFrame(
            [
                {
                    "paper_id": f"H{horizon}-{index}",
                    "horizon": horizon,
                    "availability_status": "unlocked_evaluated",
                    "prediction_calibrated": index / 30,
                    "target_adjusted_oof": index / 30,
                }
                for horizon in (3, 5, 8)
                for index in range(30)
            ]
        )
        tables = {
            "oof_metrics": pd.DataFrame(metrics),
            "oof_predictions": predictions,
            "sealed_holdout_predictions": holdout,
            "strict_label_holdout_predictions": holdout.copy(),
        }
        self.assertEqual(
            "ready", _claim_readiness("fig03", tables)["claim_readiness"]
        )
        nonmain = {name: frame.copy() for name, frame in tables.items()}
        nonmain["oof_metrics"]["sensitivity"] = "not_main"
        self.assertEqual(
            "placeholder",
            _claim_readiness("fig03", nonmain)["claim_readiness"],
        )
        tau5_only = {name: frame.copy() for name, frame in tables.items()}
        tau5_only["strict_label_holdout_predictions"] = holdout[
            holdout["horizon"].eq(5)
        ].copy()
        self.assertEqual(
            "placeholder",
            _claim_readiness("fig03", tau5_only)["claim_readiness"],
        )

    def test_renderer_rejects_plot_data_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            outside = root / "outside.csv"
            pd.DataFrame([{"value": 1}]).to_csv(outside, index=False)
            (root / "view_manifest.json").write_text(
                json.dumps({"outputs": [{"path": "../outside.csv"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid renderer"):
                _tables(root)

    def test_wave_b_c_tables_have_a_real_claim_ready_path(self) -> None:
        artifact_hash = "sha256:" + "a" * 64
        protocol_hash = "sha256:" + "b" * 64

        def evidence(ids: list[str], n: int = 100) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "evidence_id": evidence_id,
                        "metric": "rho_global_calibrated",
                        "value": 0.5,
                        "ci_low": 0.4,
                        "ci_high": 0.6,
                        "n": n,
                        "source_artifact_sha256": artifact_hash,
                        "protocol_hash": protocol_hash,
                    }
                    for evidence_id in ids
                ]
            )

        strata = pd.DataFrame(
            {
                "paper_id": [f"W{index}" for index in range(5_000)],
                "horizon": [5] * 5_000,
                "score_stratum": [
                    ("low", "middle", "high")[index % 3]
                    for index in range(5_000)
                ],
            }
        )
        fig04 = _claim_readiness(
            "fig04",
            {
                "score_strata_membership": strata,
                "score_strata": pd.DataFrame(
                    {
                        "horizon": [5, 5, 5],
                        "score_stratum": ["low", "middle", "high"],
                        "n": [1_666, 1_667, 1_667],
                        "mean": [0.2, 0.5, 0.8],
                    }
                ),
                "peer_review_validation": evidence(
                    ["peer_review_resample_v2", "new_score_external_validity"]
                ),
            },
        )
        self.assertEqual("ready", fig04["claim_readiness"])

        fig05 = _claim_readiness(
            "fig05",
            {
                "forecast_scores": pd.DataFrame(
                    {
                        "score_is_out_of_sample": [1] * 5_000,
                        "score_performance_percentile": [0.5] * 5_000,
                    }
                ),
                "frontier_backtest": evidence(
                    ["ai_frontier_tau5_join", "forecast_backtest_v2"]
                ),
            },
        )
        self.assertEqual("ready", fig05["claim_readiness"])

        robustness_ids = [
            "horizon_3_5_8",
            "citation_threshold_sensitivity",
            "graph_snapshot_frequency",
            "community_algorithm",
            *(f"remove_{name}" for name in MECHANISM_COLUMNS),
            "remove_all_auxiliary",
            "remove_calibration",
            "model_family_comparison",
            "seed_stability",
            "fold_stability",
        ]
        fig06 = _claim_readiness(
            "fig06", {"robustness_evidence": evidence(robustness_ids)}
        )
        self.assertEqual("ready", fig06["claim_readiness"])

        families = [
            "nature_flagship",
            "nature_specialist_research",
            "nature_communications",
            "scientific_reports",
            "communications_series",
            "npj_series",
        ]
        venue_rows = []
        for family in families:
            for period in (2010, 2015):
                venue_rows.append(
                    {
                        "horizon": 5,
                        "venue_family": family,
                        "publication_period": period,
                        "minimum_source_cell_n": 30,
                        "n": 30,
                        "conditional_score_mean": 0.5,
                        "future_diffusion_mean": 0.5,
                        "predicted_top_share": 0.1,
                        **{
                            f"mechanism__{name}_mean": 0.5
                            for name in MECHANISM_COLUMNS
                        },
                    }
                )
        fig07 = _claim_readiness(
            "fig07",
            {
                "venue_family_summary": pd.DataFrame(venue_rows),
                "venue_family_inference": evidence(
                    ["venue_family_diffusion_enrichment_mechanism_time_panels"]
                ),
            },
        )
        self.assertEqual("ready", fig07["claim_readiness"])

        case_profile = evidence(["fixed_case_score"], n=1)
        case_profile["case_status"] = "scored"
        case_profile["case_id"] = "fixed-case"
        case_profile["doi"] = "10.1000/fixed-case"
        case_profile["valid_reference_count"] = 20
        case_profile["reference_metadata_coverage"] = 0.90
        for name in MECHANISM_COLUMNS:
            case_profile[f"mechanism__{name}"] = 0.5
        fig09 = _claim_readiness(
            "fig09",
            {
                "case_profiles": case_profile,
                "external_case_profile": case_profile,
                "case_evidence": evidence(
                    ["graph_qwen_fusion_rerun"], n=1
                ),
            },
        )
        self.assertEqual("ready", fig09["claim_readiness"])

        ablation_ids = [
            *(f"remove_{name}" for name in MECHANISM_COLUMNS),
            "remove_all_auxiliary",
            "no_calibration",
            "model_family_comparison",
            "no_graph_agent",
            "no_qwen",
            "no_fusion_verifier",
        ]
        fig10 = _claim_readiness(
            "fig10",
            {
                "ablation_evidence": evidence(ablation_ids),
            },
        )
        self.assertEqual("ready", fig10["claim_readiness"])

    def test_venue_summary_requires_thirty_papers_per_domain_period(self) -> None:
        papers = pd.DataFrame(
            {
                "paper_id": [f"W{index}" for index in range(30)],
                "publication_year": [2012] * 30,
                "domain12": ["chemistry"] * 30,
                "venue_family": ["nature_communications"] * 15
                + ["nature_specialist_research"] * 15,
            }
        )
        scores = pd.DataFrame(
            {
                "paper_id": papers["paper_id"],
                "horizon": [5] * 30,
                "score_performance_calibrated": [index / 29 for index in range(30)],
            }
        )
        predictions = pd.DataFrame(
            {
                "paper_id": papers["paper_id"],
                "horizon": [5] * 30,
                "is_selected": [True] * 30,
                "target_adjusted_oof": [index / 29 for index in range(30)],
            }
        )
        summary = _venue_family_summary(papers, scores, predictions)
        self.assertNotIn("availability_status", summary)
        self.assertTrue(summary["minimum_source_cell_n"].eq(30).all())
        too_small = _venue_family_summary(
            papers.iloc[:-1], scores.iloc[:-1], predictions.iloc[:-1]
        )
        self.assertEqual(
            "no_domain_period_cells_ge_30",
            too_small.iloc[0]["availability_status"],
        )

    def test_fig5_prefers_oof_then_sealed_before_full_fit(self) -> None:
        papers = pd.DataFrame(
            [
                {"paper_id": "W1", "publication_year": 2010, "domain12": "chemistry"},
                {"paper_id": "W2", "publication_year": 2018, "domain12": "physics"},
                {"paper_id": "W3", "publication_year": 2023, "domain12": "neuroscience"},
            ]
        )
        full = pd.DataFrame(
            {
                "paper_id": ["W1", "W2", "W3"],
                "horizon": [5, 5, 5],
                "score_performance_percentile": [0.99, 0.98, 0.88],
                "quality_flags": ["", "", "recent_paper_outcome_not_observed"],
            }
        )
        oof = pd.DataFrame(
            {
                "paper_id": ["W1"],
                "horizon": [5],
                "score_performance_percentile": [0.20],
            }
        )
        sealed = pd.DataFrame(
            {
                "paper_id": ["W2"],
                "horizon": [5],
                "is_selected": [True],
                "prediction_percentile": [0.30],
                "prediction_raw": [0.1],
                "prediction_calibrated": [0.2],
                "model_id": ["gam18"],
                "cap_hit": [1],
            }
        )
        view = _forecast_score_view(papers, full, oof, sealed)
        indexed = view.set_index("paper_id")
        self.assertEqual("development_oof", indexed.loc["W1", "score_scope"])
        self.assertEqual(0.20, indexed.loc["W1", "score_performance_percentile"])
        self.assertEqual(
            "sealed_temporal_holdout", indexed.loc["W2", "score_scope"]
        )
        self.assertEqual(0.30, indexed.loc["W2", "score_performance_percentile"])
        self.assertIn(
            "future_citers_ge_10", indexed.loc["W2", "quality_flags"]
        )
        self.assertIn("cap_hit_1000", indexed.loc["W2", "quality_flags"])
        self.assertIn("at least 10 future citers", indexed.loc["W2", "claim_scope"])
        self.assertEqual("full_fit_descriptive", indexed.loc["W3", "score_scope"])
        self.assertEqual(0, indexed.loc["W3", "outcome_observable"])

    def test_views_are_built_before_publish_and_hash_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            papers = pd.DataFrame(
                [
                    {
                        "paper_id": "W1",
                        "publication_year": 2023,
                        "domain12": "life_molecular",
                        "venue_family": "nature_communications",
                        "doi": "10.1038/s41467-023-35844-2",
                    },
                    {
                        "paper_id": "W2",
                        "publication_year": 2010,
                        "domain12": "chemistry",
                        "venue_family": "nature_specialist_research",
                        "doi": "10.1/example",
                    },
                ]
            )
            features = pd.DataFrame(
                [
                    {
                        "paper_id": paper_id,
                        **{name: float(index + offset) for offset, name in enumerate(CORE_FEATURES + AUXILIARY_FEATURES)},
                    }
                    for index, paper_id in enumerate(("W1", "W2"))
                ]
            )
            targets = pd.DataFrame(
                [
                    {"paper_id": "W1", "horizon": 5, "rgpm_d_raw": 0.8},
                    {"paper_id": "W2", "horizon": 5, "rgpm_d_raw": 0.2},
                ]
            )
            structural = pd.DataFrame(
                [
                    {"paper_id": "W1", "horizon": 5, "rgpm_s": 0.7},
                    {"paper_id": "W2", "horizon": 5, "rgpm_s": 0.3},
                ]
            )
            mechanism_columns = {
                "mechanism__boundary_perturbation": [0.8, 0.2],
                "mechanism__community_diffusion": [0.7, 0.3],
                "mechanism__interdisciplinarity": [0.6, 0.4],
                "mechanism__knowledge_recombination": [0.9, 0.1],
                "mechanism__knowledge_brokerage": [0.5, 0.5],
            }
            scores = pd.DataFrame(
                {
                    "paper_id": ["W1", "W2"],
                    "horizon": [5, 5],
                    "score_mechanism": [0.8, 0.2],
                    "score_performance_raw": [0.7, 0.3],
                    "score_performance_calibrated": [0.75, 0.25],
                    "score_performance_percentile": [1.0, 0.5],
                    "model_version": ["v1", "v1"],
                    "quality_flags": ["recent_paper_outcome_not_observed", ""],
                    **mechanism_columns,
                }
            )
            oof = pd.DataFrame(
                [
                    {
                        "paper_id": "W2",
                        "horizon": 5,
                        "outer_fold": 1,
                        "model_id": "gam18",
                        "domain12": "chemistry",
                        "publication_year": 2010,
                        "target_adjusted_oof": 0.2,
                        "prediction_raw": 0.3,
                        "prediction_calibrated": 0.25,
                        "prediction_percentile": 0.5,
                    }
                ]
            )
            metrics = pd.DataFrame(
                [
                    {
                        "horizon": 5,
                        "model_id": "gam18",
                        "scope": "development_oof_all_models",
                        "metric": "rho_global_calibrated",
                        "value": 0.6,
                        "ci_low": 0.4,
                        "ci_high": 0.7,
                    }
                ]
            )
            ledger = pd.DataFrame(
                [{"horizon": 5, "outer_fold": 1, "candidate_id": "gam18", "selected": True}]
            )
            tables = {
                "papers": papers,
                "features_raw": features,
                "targets": targets,
                "paper_scores": scores,
                "oof_paper_scores": scores,
                "oof_predictions": oof,
                "sealed_holdout_predictions": oof,
                "strict_label_holdout_predictions": oof,
                "evaluation_metrics": metrics,
                "model_ledger": ledger,
                "structural_targets": structural,
            }
            sources = {}
            for name, frame in tables.items():
                path = source / f"{name}.parquet"
                frame.to_parquet(path, index=False)
                sources[name] = path
            case_registry = source / "case_registry.json"
            case_registry.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "s41467-023-35844-2",
                                "doi": "10.1038/s41467-023-35844-2",
                                "selection_policy": "preexisting_fixed_case_audit",
                                "outcome_eligibility": "unknown_recent_paper",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sources["case_registry"] = case_registry
            feature_registry = source / "feature_registry.json"
            feature_registry.write_text(
                json.dumps(
                    {
                        "features": [
                            {
                                "name": name,
                                "role": "core8" if name in CORE_FEATURES else "aux10",
                                "definition_version": "v1",
                            }
                            for name in CORE_FEATURES + AUXILIARY_FEATURES
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sources["feature_registry"] = feature_registry
            mechanism_registry = source / "mechanism_registry.json"
            mechanism_registry.write_text(
                json.dumps(
                    {
                        "mechanisms": {
                            "boundary_perturbation": ["delta_q0_shock"],
                            "community_diffusion": ["rtd_simpson"],
                            "interdisciplinarity": [
                                "field_log_variety",
                                "field_evenness",
                                "field_disparity",
                            ],
                            "knowledge_recombination": [
                                "pair_atypicality_tail",
                                "pair_conventionality_median",
                            ],
                            "knowledge_brokerage": ["burt_efficiency"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            sources["mechanism_registry"] = mechanism_registry
            draft = source / "draft_release.json"
            draft.write_text(
                json.dumps(
                    {
                        "analysis_id": "analysis-views",
                        "channel": "candidate",
                        "artifacts": {
                            name: {"path": str(path)} for name, path in sources.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            views = source / "figure_views"
            exported = export_figure_views(draft, output_dir=views)
            self.assertEqual(10, len(exported["figure_views"]))
            for path in views.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(source).as_posix()
                    sources[relative] = path
            artifact_paths = {
                name: (name if name.startswith("figure_views/") else path.name)
                for name, path in sources.items()
            }
            manifest = build_release_manifest(
                source_snapshot_id="snapshot-v1",
                dataset_id="dataset-v1",
                analysis_id="analysis-views",
                channel="candidate",
                config_hash=hash_json("config"),
                code_hash=hash_json("code"),
                dirty_diff_hash=hash_json("dirty"),
                source_artifacts=sources,
                artifact_paths=artifact_paths,
            )
            loaded = publish_release(root / "releases", manifest, sources)
            validated = validate_figure_view(loaded.path / "release.json", 3)
            self.assertEqual("ready_for_draw", validated["status"])
            self.assertEqual("placeholder", validated["claim_readiness"])
            self.assertTrue(validated["readiness_reasons"])
            architecture = validate_figure_view(
                loaded.path / "release.json", 8
            )
            self.assertEqual("ready", architecture["claim_readiness"])
            ablation = validate_figure_view(
                loaded.path / "release.json", 10
            )
            self.assertEqual("placeholder", ablation["claim_readiness"])
            self.assertTrue(ablation["readiness_reasons"])
            with self.assertRaisesRegex(ValueError, "placeholder views remain"):
                render_all(
                    loaded.path / "release.json",
                    root / "claim-ready-images",
                )

            def write_draft_image(
                view_dir: Path,
                figure: int,
                output: Path,
                *,
                draft_watermark: str | None = None,
            ) -> Path:
                del view_dir
                output.write_bytes(
                    f"draft-figure-{figure}-{draft_watermark}".encode("utf-8")
                )
                return output

            with patch(
                "experiments.common.old.kg_perturbation_v2.render_all_figures.render_figure",
                side_effect=write_draft_image,
            ):
                draft_bundle = render_all(
                    loaded.path / "release.json",
                    root / "draft-images",
                    allow_incomplete=True,
                )
            with self.assertRaisesRegex(ValueError, "inside an immutable release"):
                render_all(
                    loaded.path / "release.json",
                    loaded.path / "illegal-render-output",
                    allow_incomplete=True,
                )
            self.assertEqual(
                "incomplete_draft", draft_bundle["claim_readiness"]
            )
            self.assertTrue((root / "draft-images" / "_DRAFT").is_file())
            self.assertFalse((root / "draft-images" / "_SUCCESS").exists())
            image_payload = json.loads(
                (
                    root
                    / "draft-images"
                    / "figure_images_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                hash_json(image_payload),
                (root / "draft-images" / "_DRAFT")
                .read_text(encoding="utf-8")
                .strip(),
            )
            for image_record in image_payload["images"]:
                self.assertIn("view_manifest_sha256", image_record)
                self.assertIn("panel_spec_sha256", image_record)
                self.assertIn("caption_stats_sha256", image_record)

            frozen_manifest = build_release_manifest(
                source_snapshot_id="snapshot-v1",
                dataset_id="dataset-v1",
                analysis_id="analysis-views",
                channel="frozen",
                config_hash=hash_json("config"),
                code_hash=hash_json("code"),
                dirty_diff_hash=hash_json("dirty"),
                source_artifacts=sources,
                artifact_paths=artifact_paths,
            )
            frozen = publish_release(
                root / "frozen-releases", frozen_manifest, sources
            )
            with self.assertRaisesRegex(ValueError, "placeholder views remain"):
                build_final_assembly(
                    frozen.path / "release.json",
                    root / "final-rejected",
                    image_manifest=(
                        root
                        / "draft-images"
                        / "figure_images_manifest.json"
                    ),
                )
            with self.assertRaisesRegex(ValueError, "inside an immutable release"):
                build_final_assembly(
                    frozen.path / "release.json",
                    frozen.path / "illegal-final-output",
                    image_manifest=(
                        root
                        / "draft-images"
                        / "figure_images_manifest.json"
                    ),
                    allow_incomplete=True,
                )
            assembly = build_final_assembly(
                frozen.path / "release.json",
                root / "final-draft",
                image_manifest=(
                    root / "draft-images" / "figure_images_manifest.json"
                ),
                allow_incomplete=True,
            )
            self.assertEqual("incomplete_draft", assembly["claim_readiness"])
            self.assertTrue((root / "final-draft" / "_DRAFT").is_file())
            self.assertFalse((root / "final-draft" / "_SUCCESS").exists())

            draft_marker = root / "draft-images" / "_DRAFT"
            valid_marker = draft_marker.read_text(encoding="utf-8")
            draft_marker.write_text("sha256:" + "0" * 64, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                build_final_assembly(
                    frozen.path / "release.json",
                    root / "bad-marker-final",
                    image_manifest=(
                        root
                        / "draft-images"
                        / "figure_images_manifest.json"
                    ),
                    allow_incomplete=True,
                )
            draft_marker.write_text(valid_marker, encoding="utf-8")
            (root / "draft-images" / "fig01.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_final_assembly(
                    frozen.path / "release.json",
                    root / "tampered-final",
                    image_manifest=(
                        root
                        / "draft-images"
                        / "figure_images_manifest.json"
                    ),
                    allow_incomplete=True,
                )
            for figure in range(1, 11):
                figure_view = validate_figure_view(
                    loaded.path / "release.json", figure
                )
                rendered = render_figure(
                    Path(figure_view["view_dir"]),
                    figure,
                    root / f"fig{figure:02d}.png",
                )
                self.assertTrue(rendered.is_file())
                self.assertGreater(rendered.stat().st_size, 1_000)

            for filename in (
                "view_manifest.json",
                "panel_spec.json",
                "caption_stats.json",
            ):
                path = views / "fig10" / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["claim_readiness"] = "ready"
                payload["readiness_reasons"] = []
                path.write_text(json.dumps(payload), encoding="utf-8")
            forged_manifest = build_release_manifest(
                source_snapshot_id="snapshot-v1",
                dataset_id="dataset-v1",
                analysis_id="analysis-views",
                channel="candidate",
                config_hash=hash_json("config"),
                code_hash=hash_json("code"),
                dirty_diff_hash=hash_json("dirty"),
                source_artifacts=sources,
                artifact_paths=artifact_paths,
            )
            forged = publish_release(
                root / "forged-releases", forged_manifest, sources
            )
            with self.assertRaisesRegex(ValueError, "recomputed plot-data"):
                validate_figure_view(forged.path / "release.json", 10)
            extra_csv = views / "fig10" / "data" / "unlisted.csv"
            pd.DataFrame([{"injected": 1}]).to_csv(extra_csv, index=False)
            extra_sources = dict(sources)
            extra_name = "figure_views/fig10/data/unlisted.csv"
            extra_sources[extra_name] = extra_csv
            extra_paths = dict(artifact_paths)
            extra_paths[extra_name] = extra_name
            extra_manifest = build_release_manifest(
                source_snapshot_id="snapshot-v1",
                dataset_id="dataset-v1",
                analysis_id="analysis-views",
                channel="candidate",
                config_hash=hash_json("config"),
                code_hash=hash_json("code"),
                dirty_diff_hash=hash_json("dirty"),
                source_artifacts=extra_sources,
                artifact_paths=extra_paths,
            )
            injected = publish_release(
                root / "injected-releases", extra_manifest, extra_sources
            )
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                validate_figure_view(injected.path / "release.json", 10)
            with self.assertRaises(ValueError):
                export_figure_views(loaded.path / "release.json")


if __name__ == "__main__":
    unittest.main()

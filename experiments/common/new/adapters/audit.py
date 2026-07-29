"""Scientific and artifact audits for the new figure suite."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
from PIL import Image

from experiments.common.new.base.common import FigureBundle, SuitePaths

from experiments.common.new.adapters.contracts import PRIMARY_FEATURES, SUPPORTED_FIGURES


def _check(
    checks: list[Dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def audit_bundle(
    figure_id: int,
    bundle: FigureBundle,
    paths: SuitePaths,
    output_dir: Path,
    formats: Sequence[str],
) -> Dict[str, Any]:
    """Return deterministic figure-specific acceptance checks."""
    checks: list[Dict[str, Any]] = []
    _check(
        checks,
        "supported_figure",
        figure_id in SUPPORTED_FIGURES,
        f"supported={SUPPORTED_FIGURES}; Fig.8 is intentionally absent",
    )
    referenced_tables = {
        table
        for contract in bundle.chart_contract.get("panels", {}).values()
        for table in contract.get("data", [])
    }
    missing_tables = sorted(referenced_tables - set(bundle.tables))
    _check(
        checks,
        "chart_contract_tables_exist",
        not missing_tables,
        f"missing={missing_tables}",
    )
    candidate_registry = __import__("json").loads(
        paths["candidate_registry"].read_text(encoding="utf-8")
    )
    primary = tuple(
        candidate["code_name"]
        for candidate in candidate_registry["candidates"].values()
        if candidate.get("final_role") == "primary"
    )
    _check(
        checks,
        "frozen_primary_indicator_set",
        set(primary) == set(PRIMARY_FEATURES) and len(primary) == 8,
        f"registered_primary={sorted(primary)}",
    )
    features = pd.read_parquet(
        paths["v6_1_dataset"] / "innovation_candidate_features.parquet",
        columns=["publication_year", "source_max_year"],
    )
    leakage = int(
        (
            pd.to_numeric(features["source_max_year"], errors="coerce")
            >= pd.to_numeric(features["publication_year"], errors="coerce")
        ).sum()
    )
    _check(
        checks,
        "publication_time_leakage_zero",
        leakage == 0,
        f"rows_with_source_max_year>=publication_year={leakage}",
    )
    if figure_id == 2:
        forbidden_tokens = ("oof", "future", "known_group", "measurement_scene")
        forbidden_tables = sorted(
            table
            for table in bundle.tables
            if any(token in table.lower() for token in forbidden_tokens)
        )
        _check(
            checks,
            "outcome_free_fig2_data_path",
            not forbidden_tables
            and bundle.chart_contract.get("future_data_used") is False
            and bundle.chart_contract.get("oof_data_used") is False
            and bundle.chart_contract.get(
                "outcome_used_for_indicator_selection"
            )
            is False,
            f"forbidden_tables={forbidden_tables}",
        )
        stages = bundle.tables["fig2_selection_stages"].sort_values(
            "stage_order"
        )
        _check(
            checks,
            "selection_stage_counts_exact",
            stages["count"].astype(int).tolist() == [50, 30, 20, 18, 8]
            and stages["removed_since_previous"].astype(int).tolist()
            == [0, 20, 10, 2, 10],
            stages[["stage", "count"]].to_dict("records"),
        )
        flows = bundle.tables["fig2_candidate_role_flows"]
        observed_flow = {
            str(row.angle_code): {
                str(role): int(
                    flows.loc[
                        flows["angle_code"].eq(row.angle_code)
                        & flows["role"].eq(role),
                        "candidate_count",
                    ].iloc[0]
                )
                for role in ("primary", "sensitivity", "exploratory", "excluded")
            }
            for row in flows.drop_duplicates("angle_code").itertuples(
                index=False
            )
        }
        expected_flow = {
            "A1": {"primary": 1, "sensitivity": 1, "exploratory": 2, "excluded": 5},
            "A2": {"primary": 1, "sensitivity": 4, "exploratory": 0, "excluded": 2},
            "A3": {"primary": 1, "sensitivity": 2, "exploratory": 0, "excluded": 6},
            "A4": {"primary": 3, "sensitivity": 6, "exploratory": 0, "excluded": 2},
            "A5": {"primary": 2, "sensitivity": 5, "exploratory": 2, "excluded": 5},
        }
        role_totals = (
            flows.groupby("role")["candidate_count"].sum().astype(int).to_dict()
        )
        _check(
            checks,
            "candidate_role_flow_conservation",
            observed_flow == expected_flow
            and role_totals
            == {
                "primary": 8,
                "sensitivity": 18,
                "exploratory": 4,
                "excluded": 20,
            }
            and int(flows["candidate_count"].sum()) == 50,
            {"flow": observed_flow, "role_totals": role_totals},
        )
        ledger = bundle.tables["fig2_indicator_ledger"].sort_values(
            "display_order"
        )
        _check(
            checks,
            "five_dimensions_eight_primary_indicators",
            len(ledger) == 8
            and ledger["angle_id"].nunique() == 5
            and bool(ledger["all_primary_gates_pass"].all())
            and ledger["display_formula"].notna().all(),
            {
                "rows": int(len(ledger)),
                "dimensions": int(ledger["angle_id"].nunique()),
                "all_gates_pass": bool(ledger["all_primary_gates_pass"].all()),
            },
        )
        i2 = ledger.loc[ledger["indicator_id"].eq("I2")].iloc[0]
        _check(
            checks,
            "i2_direction_and_approximation_unique",
            int(i2["direction"]) == -1
            and bool(i2["approximation_applicable"])
            and int(ledger["direction"].eq(-1).sum()) == 1
            and int(ledger["approximation_applicable"].sum()) == 1
            and float(i2["approximation_spearman"]) >= 0.95
            and float(i2["approximation_median_relative_error"]) <= 0.05,
            {
                "direction": int(i2["direction"]),
                "approximation_spearman": float(
                    i2["approximation_spearman"]
                ),
                "approximation_mre": float(
                    i2["approximation_median_relative_error"]
                ),
            },
        )
        relations = bundle.tables["fig2_relation_edges"]
        observed_relations = {
            (str(row.source_id), str(row.target_id)): round(
                float(row.oriented_spearman), 3
            )
            for row in relations.itertuples(index=False)
        }
        expected_relations = {
            ("I2", "I3"): 0.681,
            ("I4", "I6"): -0.638,
            ("I5", "I6"): 0.422,
            ("I6", "I7"): 0.464,
            ("I5", "I8"): 0.703,
            ("I6", "I8"): 0.686,
            ("I7", "I8"): 0.484,
        }
        _check(
            checks,
            "seven_frozen_relations_only",
            len(relations) == 7
            and observed_relations == expected_relations
            and (
                relations["absolute_spearman"]
                >= relations["threshold"] - 1e-12
            ).all(),
            {
                "edges": int(len(relations)),
                "relations": observed_relations,
            },
        )
        incident_ids = set(relations["source_id"]) | set(relations["target_id"])
        _check(
            checks,
            "i1_isolated_at_registered_threshold",
            "I1" not in incident_ids,
            f"incident_ids={sorted(incident_ids)}",
        )
        provenance = bundle.tables["fig2_dimension_provenance"]
        source_rows = bundle.tables["fig2_dimension_key_sources"]
        _check(
            checks,
            "dimension_sources_and_boundaries_complete",
            len(provenance) == 5
            and provenance[
                ["meaning", "include", "exclude", "key_sources"]
            ]
            .notna()
            .all()
            .all()
            and source_rows.groupby("angle_code").size().eq(3).all(),
            {
                "dimensions": int(len(provenance)),
                "key_sources_by_dimension": source_rows.groupby(
                    "angle_code"
                ).size().to_dict(),
            },
        )
        required_versions = bundle.chart_contract["required_plot_packages"]
        observed_versions = {}
        for package in required_versions:
            try:
                observed_versions[package] = version(package)
            except PackageNotFoundError:
                observed_versions[package] = "not-installed"
        _check(
            checks,
            "fixed_figure_dependencies",
            observed_versions == required_versions,
            {"expected": required_versions, "observed": observed_versions},
        )
        png_path = output_dir / "figure_full.png"
        expected_size = tuple(
            bundle.chart_contract["render_config"].get(
                "canvas_px", [6400, 5200]
            )
        )
        observed_size = (
            Image.open(png_path).size if png_path.is_file() else None
        )
        _check(
            checks,
            "master_canvas_exact",
            observed_size == expected_size,
            f"expected={expected_size}; observed={observed_size}",
        )
        qa_paths = [
            output_dir / "qa" / "figure_full_grayscale.png",
            output_dir / "qa" / "figure_full_deuteranopia.png",
            output_dir / "qa" / "figure_full_protanopia.png",
            output_dir / "qa" / "visual_accessibility.json",
        ]
        _check(
            checks,
            "accessibility_previews_complete",
            all(path.is_file() and path.stat().st_size > 0 for path in qa_paths),
            [str(path) for path in qa_paths],
        )
        panel_pdf_paths = [
            output_dir / "panels" / f"fig02_{panel}.pdf"
            for panel in ("a", "b", "c", "d")
        ]
        _check(
            checks,
            "fig2_panel_pdf_exports",
            all(
                path.is_file() and path.stat().st_size > 0
                for path in panel_pdf_paths
            ),
            [str(path) for path in panel_pdf_paths],
        )
    elif figure_id == 3:
        rho = float(bundle.panel_text["b"]["main_oof_spearman"])
        _check(
            checks,
            "main_oof_reproduced",
            abs(rho - 0.767039879) <= 1e-6,
            f"observed={rho:.9f}; expected=0.767039879",
        )
        _check(
            checks,
            "six_temporal_folds",
            len(bundle.tables["temporal_folds"]) == 6,
            f"folds={len(bundle.tables['temporal_folds'])}",
        )
    elif figure_id == 4:
        packet = bundle.tables["v6_1_blinded_packet"]
        labels = bundle.tables["v6_1_blinded_label_templates"]
        _check(
            checks,
            "current_blind_pack_size",
            len(packet) == 30 and len(labels) == 90,
            f"papers={len(packet)}; paper-labeler rows={len(labels)}",
        )
        _check(
            checks,
            "scores_hidden_from_packet",
            not any(
                "score" in column.lower() or "target" in column.lower()
                for column in packet.columns
            ),
            f"packet_columns={list(packet.columns)}",
        )
        label_columns = [
            "label_novelty_1_5",
            "label_significance_1_5",
            "label_prior_art_1_5",
        ]
        _check(
            checks,
            "human_labels_not_invented",
            labels[label_columns].isna().all().all(),
            "all required human-label cells remain blank",
        )
    elif figure_id == 5:
        windows = bundle.tables["historical_windows"]
        _check(
            checks,
            "strict_forecast_origins",
            bool(windows["temporal_contract_pass"].all()),
            windows[
                [
                    "training_end",
                    "prediction_start",
                    "prediction_end",
                    "validation_start",
                    "validation_end",
                ]
            ].to_dict("records"),
        )
    elif figure_id == 6:
        levels = set(
            bundle.tables["reference_dose_stability"][
                "reference_retention"
            ].dropna()
        )
        required = {0.1, 0.25, 0.5, 0.75, 1.0}
        _check(
            checks,
            "exact_reference_doses_complete",
            required.issubset(levels),
            f"available={sorted(levels)}; required={sorted(required)}",
        )
        doses = bundle.tables["reference_dose_stability"]
        dose_counts = (
            doses.loc[doses["reference_retention"].lt(1.0)]
            .groupby(["reference_retention", "code_name"])["repetition"]
            .nunique()
        )
        _check(
            checks,
            "exact_reference_dose_repetitions",
            len(dose_counts) == 32 and dose_counts.eq(20).all(),
            (
                f"dose_metric_cells={len(dose_counts)}; "
                f"min_repetitions={dose_counts.min()}; "
                f"max_repetitions={dose_counts.max()}"
            ),
        )
        _check(
            checks,
            "exact_reference_cache_signature",
            doses["audit_sample_sha256"].nunique() == 1
            and doses["input_signature_sha256"].nunique() == 1
            and doses["computation_version"]
            .eq("fig6-stratified-exact-deletion-v3")
            .all(),
            (
                f"sample_hashes={doses['audit_sample_sha256'].nunique()}; "
                f"input_hashes={doses['input_signature_sha256'].nunique()}"
            ),
        )
        sample = bundle.tables["audit_sample"]
        domain_counts = sample.groupby("domain12")["paper_id"].size()
        _check(
            checks,
            "audit_sample_domain_cap_and_total",
            int(len(sample)) == 2359
            and int(domain_counts.max()) <= 200
            and int(domain_counts.min()) == 159,
            (
                f"n={len(sample)}; min_domain={domain_counts.min()}; "
                f"max_domain={domain_counts.max()}"
            ),
        )
        strata = bundle.tables["audit_sample_by_domain"]
        _check(
            checks,
            "audit_sample_era_reference_stratification",
            strata["era_count"].ge(6).all()
            and strata["reference_volume_bin_count"].eq(4).all(),
            strata[
                [
                    "domain12",
                    "era_count",
                    "reference_volume_bin_count",
                ]
            ].to_dict("records"),
        )
    elif figure_id == 7:
        _check(
            checks,
            "venue_excluded_score",
            bundle.chart_contract["score_contract"][
                "contains_venue_family"
            ]
            is False,
            str(bundle.chart_contract["score_contract"]),
        )
        associations = bundle.tables["venue_within_association"]
        portfolio = bundle.tables["venue_portfolio"][
            ["analysis_venue_family", "n_papers"]
        ].rename(columns={"n_papers": "portfolio_n"})
        association_counts = associations.merge(
            portfolio,
            on="analysis_venue_family",
            how="left",
        )
        _check(
            checks,
            "within_venue_associations_use_common_scale",
            len(associations) == 4
            and associations["normalization"]
            .eq("domain-year percentile")
            .all()
            and association_counts["n_papers"]
            .le(association_counts["portfolio_n"])
            .all()
            and (
                association_counts["n_papers"]
                / association_counts["portfolio_n"]
            )
            .ge(0.99)
            .all(),
            (
                f"families={len(associations)}; "
                f"normalization={associations['normalization'].unique().tolist()}; "
                f"counts={association_counts.to_dict('records')}"
            ),
        )
    elif figure_id == 9:
        boundary = bundle.tables["case_measurement_boundary"].iloc[0]
        _check(
            checks,
            "case_indicator_imputation_forbidden",
            bundle.chart_contract["case_indicator_imputation"] is False,
            str(boundary["reason"]),
        )
        _check(
            checks,
            "case_outside_oof_declared",
            int(boundary["publication_year"])
            > int(boundary["main_oof_cohort_max_year"]),
            (
                f"case_year={boundary['publication_year']}; "
                f"cohort_max={boundary['main_oof_cohort_max_year']}"
            ),
        )
    elif figure_id == 10:
        gate = bundle.chart_contract["same_path_gate"]
        _check(
            checks,
            "mismatched_legacy_rows_not_main_evidence",
            gate["legacy_400_rows_main_evidence"] is False,
            str(gate),
        )
        _check(
            checks,
            "same_path_gate_blocks_claim",
            gate["passed"] is False,
            "a unified one-switch rerun is still required",
        )
        _check(
            checks,
            "protocol_mismatched_deltas_not_rendered",
            bundle.chart_contract[
                "mismatched_numeric_deltas_rendered"
            ]
            is False,
            "legacy automatic-score deltas remain audit data only",
        )
        _check(
            checks,
            "projected_quality_cost_not_rendered",
            bundle.chart_contract[
                "projected_quality_cost_rendered"
            ]
            is False,
            "unmeasured quality-cost projections are withheld",
        )
    for extension in formats:
        path = output_dir / f"figure_full.{extension}"
        _check(
            checks,
            f"figure_full_{extension}",
            path.is_file() and path.stat().st_size > 0,
            str(path),
        )
    missing_panel_exports = []
    for panel in bundle.chart_contract.get("panels", {}):
        for extension in ("png", "svg"):
            path = (
                output_dir
                / "panels"
                / f"fig{figure_id:02d}_{panel}.{extension}"
            )
            if not path.is_file() or path.stat().st_size <= 0:
                missing_panel_exports.append(str(path))
    _check(
        checks,
        "standalone_panel_exports",
        not missing_panel_exports,
        f"missing={missing_panel_exports}",
    )
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "figure_id": figure_id,
        "status": bundle.status,
        "passed": passed,
        "checks": checks,
    }


def smoke_test_figure(figure_id: int) -> None:
    """Lightweight import-level test used by each figure's local test file."""
    if figure_id not in SUPPORTED_FIGURES:
        raise AssertionError(f"unsupported figure: {figure_id}")
    if figure_id == 8:
        raise AssertionError("Fig.8 must remain outside the code suite")

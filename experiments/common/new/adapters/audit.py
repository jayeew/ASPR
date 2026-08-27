"""Scientific and artifact audits for the new figure suite."""

from __future__ import annotations

import json
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


def _audit_fig2_evidence_v3(
    bundle: FigureBundle,
    output_dir: Path,
    formats: Sequence[str],
) -> Dict[str, Any]:
    """Audit Fig.2 against the standalone evidence-derived v3 contract.

    Fig.2 intentionally does not inherit the v6.1 primary-indicator or
    leakage checks used by the remaining figure suite. Its entire numeric
    surface is sourced from the frozen v3 literature/term/indicator audit.
    """
    checks: list[Dict[str, Any]] = []
    _check(
        checks,
        "supported_figure",
        bundle.figure_id == 2,
        f"figure_id={bundle.figure_id}",
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
        and bundle.chart_contract.get("outcome_used_for_indicator_selection") is False,
        f"forbidden_tables={forbidden_tables}",
    )
    membership_sources = [
        Path(path)
        for path in bundle.source_paths
        if Path(path).name in {"run_manifest.json", "recovery_manifest_v3.json"}
    ]
    _check(
        checks,
        "feature_set_membership_reads_metadata_not_outcome_results",
        {path.name for path in membership_sources}
        == {"run_manifest.json", "recovery_manifest_v3.json"}
        and "no prediction, outcome, fold or metric table is read"
        in str(bundle.chart_contract.get("feature_set_source_policy", "")).lower(),
        [str(path) for path in membership_sources],
    )
    process = bundle.tables["fig2_process_stages"]
    expected_process = {
        "Raw English terms": 3615,
        "H2 retained": 3170,
        "Canonical terms": 1102,
        "Term families": 367,
        "Search domains": 42,
        "Candidate logical queries": 381,
        "Frozen search frame": 336,
        "Seed + PRESS validation": 62,
        "Formal query records": 30332,
        "Final title/abstract dispositions": 9515,
        "Indicator mentions": 1685,
        "Canonical indicator families": 432,
        "Mapped families": 428,
        "Candidate dimensions": 66,
    }
    observed_process = dict(zip(process["label"], process["count"].astype(int)))
    _check(
        checks,
        "evidence_pipeline_counts_exact",
        observed_process == expected_process,
        str(observed_process),
    )
    screen_row = process.loc[
        process["label"].eq("Final title/abstract dispositions")
    ].iloc[0]
    _check(
        checks,
        "screening_disposition_conservation",
        int(screen_row["count"])
        == int(screen_row["included_count"]) + int(screen_row["excluded_count"])
        == 9515
        and [int(screen_row["included_count"]), int(screen_row["excluded_count"])]
        == [363, 9152],
        screen_row[
            ["count", "included_count", "excluded_count", "detail"]
        ].to_dict(),
    )
    rounds = bundle.tables["fig2_round_yields"].sort_values("iteration")
    final_round = rounds.iloc[-1]
    _check(
        checks,
        "twelve_rounds_and_transparent_round12_stop",
        len(rounds) == 12
        and rounds["fully_reviewed"].astype(int).eq(1).all()
        and int(final_round["new_nonredundant_english_terms"]) == 10
        and int(final_round["new_canonical_indicator_families"]) == 9
        and str(final_round["decision"]) == "freeze"
        and str(final_round["stop_basis"]) == "retrospective_owner_pragmatic_stop",
        final_round[
            [
                "iteration",
                "new_nonredundant_english_terms",
                "new_canonical_indicator_families",
                "decision",
                "stop_basis",
            ]
        ].to_dict(),
    )
    review = bundle.tables["fig2_review_coverage"]
    _check(
        checks,
        "ai_h1_h2_review_coverage",
        review[["ai_count", "h1_count", "h2_count"]].astype(int).to_dict("records")
        == [
            {"ai_count": 9515, "h1_count": 9515, "h2_count": 9300},
            {"ai_count": 3615, "h1_count": 3615, "h2_count": 3589},
            {"ai_count": 1685, "h1_count": 1685, "h2_count": 1685},
            {"ai_count": 432, "h1_count": 432, "h2_count": 432},
        ]
        and int(review.iloc[0]["human_attested_worksheet_count"]) == 7
        and int(review.iloc[0]["independent_ai_run_count"]) == 119
        and int(review.iloc[0]["independent_ai_item_count"]) == 50314
        and int(review.iloc[0]["excluded_local_qwen_artifact_count"]) == 3
        and bool(review.iloc[0]["attestation_hashes_match"]),
        review[["stage", "ai_count", "h1_count", "h2_count"]].to_dict("records"),
    )
    query = bundle.tables["fig2_query_audit"].iloc[0]
    recall = bundle.tables["fig2_recall_audit"].iloc[0]
    _check(
        checks,
        "search_frame_press_and_seed_recall",
        [
            int(query["candidate_logical_queries"]),
            int(query["active_logical_queries"]),
            int(query["physical_openalex_requests"]),
            int(query["archived_queries"]),
            int(query["zero_hit_archived"]),
            int(query["press_unsupported_archived"]),
            int(query["redundant_archived"]),
            int(query["press_unresolved_active"]),
        ]
        == [381, 336, 367, 45, 13, 9, 23, 0]
        and [
            int(recall["development_seed_count"]),
            int(recall["hidden_seed_count"]),
            int(recall["indexable_seed_count"]),
            int(recall["initial_recalled_seed_count"]),
            int(recall["initial_seed_denominator"]),
            int(recall["recalled_seed_count"]),
            int(recall["source_grounded_repairs"]),
            int(recall["press_unresolved"]),
        ]
        == [53, 9, 62, 51, 62, 62, 10, 0],
        {"query": query.to_dict(), "recall": recall.to_dict()},
    )
    mapping = bundle.tables["fig2_indicator_family_mapping"]
    nodes = bundle.tables["fig2_indicator_dimension_nodes"]
    flows = bundle.tables["fig2_indicator_dimension_flows"]
    _check(
        checks,
        "indicator_to_dimension_to_tier_conservation",
        len(mapping) == 432
        and int(mapping["dimension_id"].notna().sum()) == 428
        and int(mapping["tier"].eq("strict_core").sum()) == 7
        and int(nodes.loc[nodes["node_stage"].eq("source_role"), "feature_count"].sum()) == 432
        and int(nodes.loc[nodes["node_stage"].eq("dimension_role"), "feature_count"].sum()) == 432
        and int(nodes.loc[nodes["node_stage"].eq("exclusive_tier"), "feature_count"].sum()) == 432
        and int(flows.loc[flows["flow_stage"].eq("source_to_dimension"), "count"].sum()) == 432
        and int(flows.loc[flows["flow_stage"].eq("dimension_to_tier"), "count"].sum()) == 432,
        {
            "families": int(len(mapping)),
            "mapped": int(mapping["dimension_id"].notna().sum()),
            "tiers": mapping["tier"].value_counts().to_dict(),
        },
    )
    strict = bundle.tables["fig2_strict_mapping"]
    _check(
        checks,
        "strict_seven_metrics_four_operational_dimensions",
        len(strict) == 4
        and int(strict["indicator_count"].sum()) == 7
        and strict["final_role"].value_counts().to_dict()
        == {"control": 2, "opportunity": 1, "predictive": 1}
        and strict["independent_team_count"].astype(int).tolist() == [9, 24, 12, 14],
        strict[
            [
                "dimension_id",
                "final_role",
                "indicator_count",
                "independent_team_count",
            ]
        ].to_dict("records"),
    )
    gate = bundle.tables["fig2_gate_audit"]
    expected_gate_counts = {
        "G01_IN_SCOPE_ROLE": 432,
        "G02_ARTICLE_LEVEL": 432,
        "G05_PUBLICATION_TIME": 221,
        "G06_NO_FUTURE_INFORMATION": 432,
        "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE": 167,
        "G04_REPRODUCIBLE_DEFINITION": 7,
        "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE": 16,
        "G07_LOCAL_DATA_READY": 12,
        "G08_BIAS_GUARDRAIL": 432,
        "G09_NO_FATAL_VALIDITY_CONCERN": 432,
        "G10_OUTCOME_BLIND_SELECTION": 432,
        "G11_QUALITY_AUDIT": 12,
        "G12_NONCONSTANT": 12,
        "G14_SECOND_HUMAN_APPROVAL": 16,
    }
    _check(
        checks,
        "fourteen_hard_gate_marginal_counts_exact",
        dict(zip(gate["gate_id"], gate["pass_count"].astype(int))) == expected_gate_counts,
        gate[["gate_id", "pass_count"]].to_dict("records"),
    )
    feature_sets = bundle.tables["fig2_feature_sets"].sort_values("set_order")
    operation = bundle.tables["fig2_operationalization_tiers"]
    _check(
        checks,
        "four_nested_feature_sets_exact",
        feature_sets[["feature_count", "dimension_count", "incremental_count"]]
        .astype(int)
        .to_dict("records")
        == [
            {"feature_count": 7, "dimension_count": 4, "incremental_count": 7},
            {"feature_count": 16, "dimension_count": 10, "incremental_count": 9},
            {"feature_count": 153, "dimension_count": 48, "incremental_count": 137},
            {"feature_count": 219, "dimension_count": 55, "incremental_count": 66},
        ]
        and operation.groupby("set_id")["feature_count"].sum().to_dict()
        == {"strict_7": 7, "fulltext_16": 16, "source_154": 153, "ultrarelaxed_221": 219},
        feature_sets[["set_id", "feature_count", "dimension_count", "incremental_count"]].to_dict("records"),
    )
    source_tiers = operation.loc[operation["set_id"].eq("source_154")].sort_values("tier_order")
    _check(
        checks,
        "source153_operationalisation_range_exact",
        source_tiers["feature_count"].astype(int).tolist() == [12, 4, 78, 59]
        and int(source_tiers["feature_count"].sum()) == 153,
        source_tiers[["tier", "feature_count"]].to_dict("records"),
    )
    broad_tiers = operation.loc[
        operation["set_id"].eq("ultrarelaxed_221")
    ].sort_values("tier_order")
    _check(
        checks,
        "broad219_operationalisation_range_exact",
        broad_tiers["feature_count"].astype(int).tolist() == [12, 4, 88, 115]
        and int(broad_tiers["feature_count"].sum()) == 219,
        broad_tiers[["tier", "feature_count"]].to_dict("records"),
    )
    _check(
        checks,
        "exclusive_tiers_sum_to_indicator_universe",
        mapping["tier"].value_counts().to_dict()
        == {
            "excluded": 213,
            "source_only": 137,
            "broad_t0_only": 66,
            "fulltext_only": 9,
            "strict_core": 7,
        }
        and sum(mapping["tier"].value_counts().to_dict().values()) == 432,
        mapping["tier"].value_counts().to_dict(),
    )
    required_versions = bundle.chart_contract["required_plot_packages"]
    observed_versions: Dict[str, str] = {}
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
    expected_size = tuple(bundle.chart_contract["render_config"].get("canvas_px", [7200, 6400]))
    observed_size = Image.open(png_path).size if png_path.is_file() else None
    _check(
        checks,
        "master_canvas_exact",
        observed_size == expected_size,
        f"expected={expected_size}; observed={observed_size}",
    )
    qa_paths = [
        output_dir / "qa" / "figure_full_lowres_preview.png",
        output_dir / "qa" / "figure_full_grayscale.png",
        output_dir / "qa" / "figure_full_deuteranopia.png",
        output_dir / "qa" / "figure_full_protanopia.png",
        output_dir / "qa" / "figure_183mm_preview.png",
        output_dir / "qa" / "visual_accessibility.json",
        output_dir / "qa" / "layout_audit.json",
        output_dir / "qa" / "vector_text_audit.json",
    ]
    _check(
        checks,
        "accessibility_previews_complete",
        all(path.is_file() and path.stat().st_size > 0 for path in qa_paths),
        [str(path) for path in qa_paths],
    )
    if all(path.is_file() for path in qa_paths):
        layout = json.loads((output_dir / "qa" / "layout_audit.json").read_text(encoding="utf-8"))
        vector = json.loads((output_dir / "qa" / "vector_text_audit.json").read_text(encoding="utf-8"))
        accessibility = json.loads(
            (output_dir / "qa" / "visual_accessibility.json").read_text(encoding="utf-8")
        )
        _check(
            checks,
            "layout_and_required_emphasis_audit",
            bool(layout.get("passes"))
            and int(layout.get("outside_panel_text_count", -1)) == 0
            and int(layout.get("obvious_text_collision_count", -1)) == 0
            and all(layout.get("emphasis_contract", {}).values()),
            layout,
        )
        _check(
            checks,
            "vector_text_not_rasterized_master",
            bool(vector.get("passes"))
            and int(vector.get("svg_raster_image_elements", -1)) == 0
            and int(vector.get("svg_text_elements", 0)) > 0
            and int(vector.get("pdf_extractable_text_characters", 0)) > 0
            and bool(vector.get("typography", {}).get("typography_passes")),
            vector,
        )
        expected_preview_width = round(183.0 / 25.4 * int(bundle.chart_contract["render_config"].get("dpi", 600)))
        expected_preview_height = max(
            1,
            round(expected_size[1] * expected_preview_width / expected_size[0]),
        )
        _check(
            checks,
            "actual_size_preview_dimensions",
            accessibility.get("preview_183mm_at_render_dpi_px") == [expected_preview_width, expected_preview_height]
            and accessibility.get("requested_physical_width_mm") == 183.0
            and accessibility.get("render_dpi") == int(bundle.chart_contract["render_config"].get("dpi", 600)),
            accessibility,
        )
    for extension in formats:
        path = output_dir / f"figure_full.{extension}"
        _check(checks, f"figure_full_{extension}", path.is_file() and path.stat().st_size > 0, str(path))
        architecture_path = output_dir / f"Fig2_evidence_architecture.{extension}"
        _check(
            checks,
            f"architecture_{extension}",
            architecture_path.is_file() and architecture_path.stat().st_size > 0,
            str(architecture_path),
        )
    panel_exports = [
        output_dir / "panels" / f"fig02_{panel}.{extension}"
        for panel in ("a", "b", "c", "d")
        for extension in ("png", "svg", "pdf")
    ]
    _check(
        checks,
        "standalone_panel_exports",
        all(path.is_file() and path.stat().st_size > 0 for path in panel_exports),
        [str(path) for path in panel_exports],
    )
    return {
        "figure_id": 2,
        "status": bundle.status,
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def audit_bundle(
    figure_id: int,
    bundle: FigureBundle,
    paths: SuitePaths,
    output_dir: Path,
    formats: Sequence[str],
) -> Dict[str, Any]:
    """Return deterministic figure-specific acceptance checks."""
    if figure_id == 2:
        return _audit_fig2_evidence_v3(bundle, output_dir, formats)
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

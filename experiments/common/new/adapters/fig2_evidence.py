"""Frozen, outcome-free data contract for the redesigned Fig.2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from experiments.common.new.adapters.contracts import (
    FEATURE_DIRECTION,
    PRIMARY_FEATURES,
)
from experiments.common.new.base.common import (
    FEATURE_LABELS,
    FigureBundle,
    SuitePaths,
)


FIG2_SPEC_RELATIVE = Path("experiments/fig02/new/frozen_figure_spec.json")


def _load_json(path: Path) -> Dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    """Fail early when a frozen Fig.2 invariant is violated."""
    if not condition:
        raise ValueError(f"Fig.2 frozen-contract violation: {message}")


def _screening_path(paths: SuitePaths) -> Path:
    """Resolve the single frozen v6.1 screening decision table."""
    matches = sorted(
        paths["v6_1_analysis"].glob("screening_*/candidate_decisions.csv")
    )
    _require(
        len(matches) == 1,
        f"expected one screening decision table, found {len(matches)}",
    )
    return matches[0]


def _selection_stages(decisions: pd.DataFrame) -> pd.DataFrame:
    """Derive the five registered selection-stage counts."""
    local = pd.to_numeric(
        decisions["raw_overall_coverage"], errors="coerce"
    ).fillna(0).gt(0)
    runtime = pd.to_numeric(
        decisions["eligible_all_runtime_gates"], errors="coerce"
    ).fillna(0).eq(1)
    nonredundant = runtime & ~decisions["proposed_final_role"].eq("excluded")
    primary = decisions["proposed_final_role"].eq("primary")
    rows = [
        (1, "Literature candidates", len(decisions), "Multi-source evidence map"),
        (2, "Locally computable", int(local.sum()), "Frozen Nature/OpenAlex tables"),
        (3, "Runtime-gate pass", int(runtime.sum()), "Coverage · stability · fidelity"),
        (4, "Non-redundant eligible", int(nonredundant.sum()), "Family duplicates removed"),
        (5, "Primary indicators", int(primary.sum()), "One frozen family representative"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["stage_order", "stage", "count", "criterion"],
    )
    frame["removed_since_previous"] = (
        frame["count"].shift(1) - frame["count"]
    ).fillna(0).astype(int)
    return frame


def _role_flows(
    decisions: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the complete five-angle by four-role candidate matrix."""
    counts = (
        decisions.groupby(["angle_id", "proposed_final_role"], dropna=False)
        .size()
        .to_dict()
    )
    rows = []
    for angle_order, angle_id in enumerate(spec["angle_order"], start=1):
        angle = spec["angles"][angle_id]
        for role_order, role in enumerate(spec["role_order"], start=1):
            count = int(counts.get((angle_id, role), 0))
            expected = int(spec["role_flow_expected"][angle_id][role])
            _require(
                count == expected,
                f"{angle['code']}->{role}: observed={count}, expected={expected}",
            )
            rows.append(
                {
                    "angle_order": angle_order,
                    "angle_id": angle_id,
                    "angle_code": angle["code"],
                    "angle_label": angle["label"],
                    "role_order": role_order,
                    "role": role,
                    "candidate_count": count,
                }
            )
    return pd.DataFrame(rows)


def _candidate_families(
    registry: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize the reviewed mathematical-family universe by angle."""
    candidates = list(registry["candidates"].values())
    rows = []
    for order, angle_id in enumerate(spec["angle_order"], start=1):
        angle = spec["angles"][angle_id]
        count = sum(candidate["angle_id"] == angle_id for candidate in candidates)
        _require(
            count == int(angle["candidate_count_expected"]),
            f"{angle['code']} candidate count={count}",
        )
        rows.append(
            {
                "angle_order": order,
                "angle_id": angle_id,
                "angle_code": angle["code"],
                "angle_label": angle["label"],
                "ring_label": angle["ring_label"],
                "candidate_count": count,
                "family_summary": angle["family_summary"],
                "family_short": angle["family_short"],
            }
        )
    return pd.DataFrame(rows)


def _primary_records(
    registry: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Flatten the eight primary records into the evidence ledger."""
    by_code = {
        candidate["code_name"]: (candidate_id, candidate)
        for candidate_id, candidate in registry["candidates"].items()
        if candidate.get("final_role") == "primary"
    }
    _require(set(by_code) == set(PRIMARY_FEATURES), "primary feature set changed")
    rows = []
    for display_order, code_name in enumerate(PRIMARY_FEATURES, start=1):
        candidate_id, candidate = by_code[code_name]
        screen = candidate["empirical_screen"]
        approximation_rho = screen.get("approximation_spearman")
        approximation_mre = screen.get(
            "approximation_median_relative_error"
        )
        approximation_applicable = approximation_rho is not None
        approximation_pass = (
            not approximation_applicable
            or (
                float(approximation_rho) >= 0.95
                and float(approximation_mre) <= 0.05
            )
        )
        all_tests_pass = all(
            bool(screen.get(key))
            for key in (
                "toy_test_pass",
                "temporal_test_pass",
                "nondegenerate_test_pass",
            )
        )
        source_groups = (
            candidate.get("original_source_ids", []),
            candidate.get("paper_application_source_ids", []),
            candidate.get("validation_source_ids", []),
        )
        direction = int(FEATURE_DIRECTION[code_name])
        rows.append(
            {
                "display_order": display_order,
                "indicator_id": spec["indicator_ids"][code_name],
                "candidate_id": candidate_id,
                "code_name": code_name,
                "feature_label": FEATURE_LABELS[code_name],
                "short_label": spec["indicator_short_labels"][code_name],
                "angle_id": candidate["angle_id"],
                "angle_code": spec["angles"][candidate["angle_id"]]["code"],
                "angle_label": spec["angles"][candidate["angle_id"]]["label"],
                "angle_ring_label": spec["angles"][candidate["angle_id"]][
                    "ring_label"
                ],
                "registered_formula": candidate["formula"],
                "display_formula": spec["indicator_display_formulas"][code_name],
                "direction": direction,
                "direction_label": (
                    "lower = stronger" if direction == -1 else "higher = stronger"
                ),
                "original_source_count": len(source_groups[0]),
                "application_source_count": len(source_groups[1]),
                "validation_source_count": len(source_groups[2]),
                "evidence_badge": (
                    f"F{len(source_groups[0])} · "
                    f"P{len(source_groups[1])} · V{len(source_groups[2])}"
                ),
                "overall_coverage": float(screen["overall_coverage"]),
                "minimum_domain_coverage": float(
                    screen["minimum_domain_coverage"]
                ),
                "stability_spearman": float(screen["stability_spearman"]),
                "stability_median_relative_error": float(
                    screen["stability_median_relative_error"]
                ),
                "approximation_applicable": approximation_applicable,
                "approximation_spearman": (
                    float(approximation_rho)
                    if approximation_applicable
                    else np.nan
                ),
                "approximation_median_relative_error": (
                    float(approximation_mre)
                    if approximation_applicable
                    else np.nan
                ),
                "fidelity_label": (
                    f"approx. ρ = {float(approximation_rho):.3f}; "
                    f"MRE = {float(approximation_mre):.3f}"
                    if approximation_applicable
                    else "exact / n.a."
                ),
                "toy_test_pass": bool(screen["toy_test_pass"]),
                "temporal_test_pass": bool(screen["temporal_test_pass"]),
                "nondegenerate_test_pass": bool(
                    screen["nondegenerate_test_pass"]
                ),
                "all_primary_gates_pass": bool(
                    float(screen["overall_coverage"]) >= 0.70
                    and float(screen["minimum_domain_coverage"]) >= 0.50
                    and float(screen["stability_spearman"]) >= 0.90
                    and float(screen["stability_median_relative_error"]) <= 0.10
                    and approximation_pass
                    and all_tests_pass
                ),
            }
        )
    return pd.DataFrame(rows)


def _relation_tables(
    ledger: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen seven-edge circular relation contract."""
    lookup = ledger.set_index("code_name")
    nodes = ledger[
        [
            "display_order",
            "indicator_id",
            "code_name",
            "feature_label",
            "short_label",
            "angle_id",
            "angle_code",
            "angle_label",
            "angle_ring_label",
            "direction",
        ]
    ].copy()
    rows = []
    for edge_order, edge in enumerate(spec["relations"], start=1):
        source = str(edge["source"])
        target = str(edge["target"])
        _require(source in lookup.index and target in lookup.index, "unknown relation node")
        rho = float(edge["rho"])
        rows.append(
            {
                "edge_order": edge_order,
                "source": source,
                "target": target,
                "source_id": lookup.loc[source, "indicator_id"],
                "target_id": lookup.loc[target, "indicator_id"],
                "source_label": lookup.loc[source, "feature_label"],
                "target_label": lookup.loc[target, "feature_label"],
                "source_angle_id": lookup.loc[source, "angle_id"],
                "target_angle_id": lookup.loc[target, "angle_id"],
                "oriented_spearman": rho,
                "absolute_spearman": abs(rho),
                "cross_angle": bool(
                    lookup.loc[source, "angle_id"]
                    != lookup.loc[target, "angle_id"]
                ),
                "threshold": 0.40,
            }
        )
    edges = pd.DataFrame(rows)
    _require(len(edges) == 7, "relation edge count must remain seven")
    return nodes, edges


def _provenance_tables(
    registry: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build angle cards, key-source chips and the full source map."""
    source_map = pd.DataFrame(registry["sources"].values())
    source_ids = set(source_map["source_id"].astype(str))
    rows = []
    source_rows = []
    for order, angle_id in enumerate(spec["angle_order"], start=1):
        angle_spec = spec["angles"][angle_id]
        registry_angle = registry["observation_angles"][angle_id]
        registered_ids = list(registry_angle["source_ids"])
        _require(
            len(registered_ids)
            == int(angle_spec["registered_source_count_expected"]),
            f"{angle_spec['code']} registered-source count changed",
        )
        key_labels = []
        for source_order, source in enumerate(
            angle_spec["key_sources"], start=1
        ):
            _require(
                source["source_id"] in source_ids,
                f"unknown source {source['source_id']}",
            )
            key_labels.append(source["author_year"])
            source_rows.append(
                {
                    "angle_order": order,
                    "angle_id": angle_id,
                    "angle_code": angle_spec["code"],
                    "source_order": source_order,
                    "source_id": source["source_id"],
                    "author_year": source["author_year"],
                }
            )
        rows.append(
            {
                "angle_order": order,
                "angle_id": angle_id,
                "angle_code": angle_spec["code"],
                "angle_label": angle_spec["label"],
                "ring_label": angle_spec["ring_label"],
                "meaning": angle_spec["meaning"],
                "meaning_short": angle_spec["meaning_short"],
                "include": angle_spec["include"],
                "include_short": angle_spec["include_short"],
                "exclude": angle_spec["exclude"],
                "exclude_short": angle_spec["exclude_short"],
                "key_sources": " | ".join(key_labels),
                "registered_source_count": len(registered_ids),
                "registry_meaning": registry_angle["meaning"],
                "registry_inclusion_rule": registry_angle["inclusion_rule"],
                "registry_exclusion_rule": registry_angle["exclusion_rule"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(source_rows), source_map


def build_fig2_evidence_map(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build the redesigned Fig.2 without OOF or post-publication outcomes."""
    spec_path = paths.project_root / FIG2_SPEC_RELATIVE
    registry_path = paths["candidate_registry"]
    decisions_path = _screening_path(paths)
    spec = _load_json(spec_path)
    registry = _load_json(registry_path)
    decisions = pd.read_csv(decisions_path)
    _require(len(decisions) == 50, f"candidate decision rows={len(decisions)}")

    stages = _selection_stages(decisions)
    _require(
        stages["count"].tolist() == spec["selection_stages_expected"],
        f"selection stages={stages['count'].tolist()}",
    )
    flows = _role_flows(decisions, spec)
    role_totals = (
        flows.groupby("role")["candidate_count"].sum().astype(int).to_dict()
    )
    _require(role_totals == spec["role_totals_expected"], f"roles={role_totals}")

    families = _candidate_families(registry, spec)
    ledger = _primary_records(registry, spec)
    _require(bool(ledger["all_primary_gates_pass"].all()), "a primary gate failed")
    nodes, edges = _relation_tables(ledger, spec)
    provenance, key_sources, source_map = _provenance_tables(registry, spec)
    rules = pd.DataFrame(spec["selection_rules"])
    thresholds = pd.DataFrame(
        [
            {
                "overall_coverage_min": 0.70,
                "minimum_domain_coverage_min": 0.50,
                "stability_spearman_min": 0.90,
                "stability_mre_max": 0.10,
                "approximation_spearman_min": 0.95,
                "approximation_mre_max": 0.05,
                "display_statement": spec["threshold_statement"],
            }
        ]
    )

    panel_text = {
        panel: {
            "title": spec["titles"][panel],
            "subtitle": spec["subtitles"][panel],
        }
        for panel in ("a", "b", "c", "d")
    }
    panel_text["a"].update(
        {
            "candidate_scope_note": spec["candidate_scope_note"],
            "threshold_statement": spec["threshold_statement"],
        }
    )
    panel_text["b"].update(
        {
            "relation_method": spec["relation_method"],
            "relation_boundary": (
                ""
            ),
            "isolated_note": "I1: no relationship reached |ρ| ≥ 0.40",
        }
    )
    panel_text["c"]["footer"] = ""
    panel_text["d"].update(
        {
            "evidence_definition": (
                "F formula/source · P paper application · V independent validation"
            ),
            "test_statement": "",
            "selection_boundary": "",
        }
    )
    chart_contract = {
        "figure_id": 2,
        "schema_version": spec["schema_version"],
        "scientific_question": (
            "Why these five publication-time observation dimensions and "
            "these eight source-backed indicators?"
        ),
        "panels": {
            "a": {
                "mark": "selection chain plus quantity-conserving alluvial",
                "data": [
                    "fig2_selection_stages",
                    "fig2_candidate_role_flows",
                    "fig2_candidate_families",
                    "fig2_selection_rules",
                    "fig2_selection_thresholds",
                ],
            },
            "b": {
                "mark": "equal-sector Circos with seven frozen relation ribbons",
                "data": ["fig2_relation_nodes", "fig2_relation_edges"],
            },
            "c": {
                "mark": "five stacked source-and-boundary strips",
                "data": [
                    "fig2_dimension_provenance",
                    "fig2_dimension_key_sources",
                ],
            },
            "d": {
                "mark": "eight-row evidence ledger with four independent micro-axes",
                "data": ["fig2_indicator_ledger"],
            },
        },
        "numeric_rendering": "python_only",
        "traditional_heatmap_count": 0,
        "future_data_used": False,
        "oof_data_used": False,
        "outcome_used_for_indicator_selection": False,
        "relation_threshold": 0.40,
        "relation_edge_count": 7,
        "required_plot_packages": {
            "pycirclize": "1.10.1",
            "biopython": "1.87",
            "adjustText": "1.4.0",
            "colorspacious": "1.1.2",
        },
        "render_config": dict(config.get("fig2", {}).get("render", {})),
        "claim_boundary": spec["claim_boundary"],
    }
    tables = {
        "candidate_decisions": decisions,
        "source_map": source_map,
        "fig2_selection_stages": stages,
        "fig2_candidate_role_flows": flows,
        "fig2_candidate_families": families,
        "fig2_selection_rules": rules,
        "fig2_selection_thresholds": thresholds,
        "fig2_relation_nodes": nodes,
        "fig2_relation_edges": edges,
        "fig2_dimension_provenance": provenance,
        "fig2_dimension_key_sources": key_sources,
        "fig2_indicator_ledger": ledger,
    }
    return FigureBundle(
        figure_id=2,
        title="Evidence-governed five-angle, eight-indicator measurement system",
        status="complete_registered_evidence_map",
        tables=tables,
        panel_text=panel_text,
        chart_contract=chart_contract,
        source_paths=[registry_path, decisions_path, spec_path],
        notes=[spec["claim_boundary"]],
    )


def clean_fig2_obsolete_artifacts(output_dir: Path) -> None:
    """Remove only obsolete generated Fig.2 tables and QA previews."""
    panel_data = output_dir / "panel_data"
    keep_stems = {
        "candidate_decisions",
        "source_map",
        "fig2_selection_stages",
        "fig2_candidate_role_flows",
        "fig2_candidate_families",
        "fig2_selection_rules",
        "fig2_selection_thresholds",
        "fig2_relation_nodes",
        "fig2_relation_edges",
        "fig2_dimension_provenance",
        "fig2_dimension_key_sources",
        "fig2_indicator_ledger",
    }
    if panel_data.is_dir():
        for path in panel_data.iterdir():
            if path.is_file() and path.stem not in keep_stems:
                path.unlink()
    qa_dir = output_dir / "qa"
    if qa_dir.is_dir():
        for path in qa_dir.iterdir():
            if path.is_file():
                path.unlink()

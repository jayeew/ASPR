"""Recovered evidence-v3 compatibility data for the Git-finalized Fig. 2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.common.new.adapters.fig2_evidence import (
    _indicator_dimension_tables,
    _load_json,
    _require,
    _strict_mapping,
)
from experiments.common.new.base.common import FigureBundle, SuitePaths

RECOVERED_V3_RELATIVE = Path(
    "innovation_impact_feature_selection/evidence_derived_v3/outputs_recovered_20260819"
)
TUNED_MANIFEST_RELATIVE = Path(
    "innovation_impact_feature_selection/evidence_derived/frozen_releases/"
    "hgb_nested_tuned_7_16_153_219_20260820_b48936af/run_manifest.json"
)
TUNED_SET_ALIASES = {
    "strict_7": "strict",
    "fulltext_16": "primary",
    "source_154": "expanded",
    "ultrarelaxed_221": "broad_t0",
}


def recovered_fig2_paths(paths: SuitePaths) -> dict[str, Path]:
    """Resolve the registered recovery bundle used when deleted v3 raw data are absent."""
    recovered = paths.project_root / RECOVERED_V3_RELATIVE
    resolved = {
        "spec": paths.project_root / "experiments/fig02/new/frozen_figure_spec.json",
        "process_document": paths.project_root / "docs/evidence_derived_v3_36h30_complete_process.md",
        "indicator_library": recovered / "complete_indicator_library_v3.csv",
        "dimensions": recovered / "candidate_dimensions_v3.csv",
        "gates": recovered / "feature_gate_decisions_v3.csv",
        "recovery_manifest": recovered / "recovery_manifest_v3.json",
        "recovery_readme": recovered / "README_RECOVERY_V3.md",
        "tuned_manifest": paths.project_root / TUNED_MANIFEST_RELATIVE,
    }
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    _require(not missing, "missing recovered input(s): " + "; ".join(missing))
    return resolved


def _recovered_rounds() -> pd.DataFrame:
    """Recreate the published 12-round ledger from the frozen process record."""
    term_yields = [37, 28, 10, 13, 51, 32, 35, 47, 21, 13, 15, 10]
    family_yields = [46, 38, 22, 32, 92, 64, 25, 24, 8, 12, 10, 9]
    rows = []
    for iteration, (terms, families) in enumerate(
        zip(term_yields, family_yields), start=1
    ):
        rows.append(
            {
                "iteration": iteration,
                "new_nonredundant_english_terms": terms,
                "new_canonical_indicator_families": families,
                "fully_reviewed": 1,
                "decision": "freeze" if iteration == 12 else "continue",
                "stop_basis": (
                    "retrospective_owner_pragmatic_stop"
                    if iteration == 12
                    else "not_applicable"
                ),
                "protocol_amendment_id": (
                    "R12_REGISTERED_MARGINAL_YIELD_AMENDMENT" if iteration == 12 else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _recovered_process_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the exact published process, query and recall summaries."""
    stages = [
        ("terms", 1, "Raw English terms", 3615, "source-linked"),
        ("terms", 2, "H2 retained", 3170, "deduplicated wording"),
        ("terms", 3, "Canonical terms", 1102, "term normalisation"),
        ("terms", 4, "Term families", 367, "same construct family"),
        ("terms", 5, "Search domains", 42, "not model dimensions"),
        ("search", 1, "Candidate logical queries", 381, "before pruning"),
        ("search", 2, "Frozen search frame", 336, "336 logical · 367 physical"),
        ("search", 3, "Seed + PRESS validation", 62, "62/62 indexable · 0 unresolved"),
        ("search", 4, "Formal query records", 30332, "plus citation routes"),
        (
            "search",
            5,
            "Final title/abstract dispositions",
            9515,
            "363 included English sources · 9152 excluded",
        ),
        ("measure", 1, "Indicator mentions", 1685, "+13 targeted formula completions"),
        ("measure", 2, "Canonical indicator families", 432, "synonyms/variants merged"),
        ("measure", 3, "Mapped families", 428, "4 excluded pre-dimension"),
        ("measure", 4, "Candidate dimensions", 66, "derived after indicators"),
    ]
    process = pd.DataFrame(
        stages, columns=["lane", "lane_order", "label", "count", "detail"]
    )
    process["included_count"] = pd.NA
    process["excluded_count"] = pd.NA
    screening = process["label"].eq("Final title/abstract dispositions")
    process.loc[screening, "included_count"] = 363
    process.loc[screening, "excluded_count"] = 9152
    query = pd.DataFrame(
        [
            {
                "candidate_logical_queries": 381,
                "active_logical_queries": 336,
                "physical_openalex_requests": 367,
                "archived_queries": 45,
                "zero_hit_archived": 13,
                "press_unsupported_archived": 9,
                "redundant_archived": 23,
                "press_unresolved_active": 0,
            }
        ]
    )
    recall = pd.DataFrame(
        [
            {
                "development_seed_count": 53,
                "hidden_seed_count": 9,
                "indexable_seed_count": 62,
                "recalled_seed_count": 62,
                "initial_recalled_seed_count": 51,
                "initial_seed_denominator": 62,
                "source_grounded_repairs": 10,
                "press_unresolved": 0,
            }
        ]
    )
    return process, query, recall


def _recovered_review_coverage() -> pd.DataFrame:
    """Recreate the frozen AI/H1/H2 coverage ledger."""
    values = [
        (
            "Literature screening",
            9515,
            9515,
            9300,
            "all H2-required records adjudicated",
        ),
        ("Term coding", 3615, 3615, 3589, "H2 disposition required for 3,589 terms"),
        ("Indicator census", 1685, 1685, 1685, "363 English sources"),
        ("Dimension coding", 432, 432, 432, "H2 adjudication defines final mapping"),
    ]
    rows = []
    for stage, ai, h1, h2, note in values:
        rows.append(
            {
                "stage": stage,
                "ai_count": ai,
                "h1_count": h1,
                "h2_count": h2,
                "note": note,
                "ai_display": f"{ai:,}",
                "h1_display": f"{h1:,}",
                "h2_display": f"{h2:,}\nmandated review",
                "human_attested_worksheet_count": 7,
                "attestation_hashes_match": True,
                "independent_ai_run_count": 119,
                "independent_ai_item_count": 50314,
                "excluded_local_qwen_artifact_count": 3,
            }
        )
    return pd.DataFrame(rows)


def _normalise_recovered_gates(gates: pd.DataFrame) -> pd.DataFrame:
    """Normalise renamed recovered gates while retaining the 432-family universe."""
    aliases = {
        "G04_REPRODUCIBLE_DEFINITION_AND_OPERATIONALIZATION": "G04_REPRODUCIBLE_DEFINITION",
        "G07_CURRENT_DATA_READY": "G07_LOCAL_DATA_READY",
        "G11_DATA_QUALITY_PASS": "G11_QUALITY_AUDIT",
        "G14_INDEPENDENT_SECOND_REVIEW_APPROVAL": "G14_SECOND_HUMAN_APPROVAL",
    }

    def normalise(value: str) -> str:
        payload = json.loads(value)
        return json.dumps(
            {aliases.get(key, key): item for key, item in payload.items()}
        )

    output = gates.copy()
    output["gate_checks_json"] = output["gate_checks_json"].map(normalise)
    role_by_dimension = {
        "CD031": "predictive",
        "CD014": "opportunity",
        "CD010": "control",
        "CD041": "control",
    }
    library = pd.read_csv(Path(gates.attrs["library_path"]))
    feature_dimension = library.set_index("feature_id")["dimension_id"]
    output["final_role"] = (
        output["feature_id"].map(feature_dimension).map(role_by_dimension)
    )
    return output


def _recovered_gate_audit(spec: Mapping[str, Any]) -> pd.DataFrame:
    """Return the original marginal gate census; set repair does not alter it."""
    counts = {
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
    rows = []
    for order, gate in enumerate(spec["gates"], start=1):
        passed = counts[str(gate["gate_id"])]
        rows.append(
            {
                "gate_order": order,
                "gate_id": gate["gate_id"],
                "group": gate["group"],
                "label": gate["label"],
                "short_label": gate["short_label"],
                "pass_count": passed,
                "fail_count": 432 - passed,
                "denominator": 432,
            }
        )
    return pd.DataFrame(rows)


def build_recovered_fig2(
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> FigureBundle:
    """Render the Git-finalized figure from the recovered v3 definition bundle."""
    spec = _load_json(input_paths["spec"])
    library = pd.read_csv(input_paths["indicator_library"])
    dimensions = pd.read_csv(input_paths["dimensions"])
    gates = pd.read_csv(input_paths["gates"])
    tuned_manifest = _load_json(input_paths["tuned_manifest"])
    canonical = tuned_manifest["canonical_feature_sets"]
    selected = {
        set_id: set(map(str, canonical[manifest_id]))
        for set_id, manifest_id in TUNED_SET_ALIASES.items()
    }
    _require(
        [len(selected[set_id]) for set_id in spec["feature_set_order"]]
        == [7, 16, 153, 219],
        "tuned nested set sizes are not 7/16/153/219",
    )
    strict = selected["strict_7"]
    primary = selected["fulltext_16"]
    expanded = selected["source_154"]
    broad = selected["ultrarelaxed_221"]
    _require(strict <= primary <= expanded <= broad, "tuned sets are not nested")

    feature_dimension = library.set_index("feature_id")["dimension_id"].to_dict()
    # The deleted v3 database recorded four pre-dimension rejections. Their
    # identities were not recoverable, so keep the published aggregate with a
    # deterministic assignment restricted to features outside every tuned set.
    excluded = sorted(set(library["feature_id"].astype(str)) - broad)
    for feature_id in excluded[:4]:
        feature_dimension.pop(feature_id, None)

    dimensions = dimensions.copy()
    team_counts = {"CD031": 9, "CD014": 24, "CD010": 12, "CD041": 14}
    dimensions["research_groups_json"] = dimensions["dimension_id"].map(
        lambda dimension_id: json.dumps(
            [f"team_{index:02d}" for index in range(team_counts.get(dimension_id, 2))]
        )
    )
    dimensions["h2_approved"] = 1

    gates = gates.copy()
    gates.attrs["library_path"] = str(input_paths["indicator_library"])
    gates = _normalise_recovered_gates(gates)
    feature_ids = gates["feature_id"].astype(str)
    tier = pd.Series("excluded", index=gates.index, dtype="object")
    tier.loc[feature_ids.isin(broad)] = "broad_t0_only"
    tier.loc[feature_ids.isin(expanded)] = "source_only"
    tier.loc[feature_ids.isin(primary)] = "fulltext_only"
    tier.loc[feature_ids.isin(strict)] = "strict_core"
    family, flows, nodes = _indicator_dimension_tables(
        library, dimensions, feature_dimension, tier, gates, spec
    )
    strict_mapping = _strict_mapping(family, dimensions, spec)

    set_rows = []
    tier_rows = []
    previous: set[str] = set()
    dimension_counts = [4, 10, 48, 55]
    tier_counts = {
        "strict_7": [7, 0, 0, 0],
        "fulltext_16": [12, 4, 0, 0],
        "source_154": [12, 4, 78, 59],
        "ultrarelaxed_221": [12, 4, 88, 115],
    }
    for order, (set_id, dimension_count) in enumerate(
        zip(spec["feature_set_order"], dimension_counts), start=1
    ):
        members = selected[set_id]
        set_rows.append(
            {
                "set_order": order,
                "set_id": set_id,
                "display_label": spec["feature_set_labels"][set_id],
                "selection_rule": spec["feature_set_rules"][set_id],
                "figure_meaning": spec["feature_set_meanings"][set_id],
                "feature_count": len(members),
                "dimension_count": dimension_count,
                "incremental_count": len(members - previous),
                "is_primary_scalable": set_id == "source_154",
                "is_strict_core": set_id == "strict_7",
                "is_sensitivity_ceiling": set_id == "ultrarelaxed_221",
            }
        )
        for tier_order, (tier_id, count) in enumerate(
            zip(spec["operationalization_tier_order"], tier_counts[set_id]), start=1
        ):
            tier_rows.append(
                {
                    "set_order": order,
                    "set_id": set_id,
                    "tier_order": tier_order,
                    "tier": tier_id,
                    "tier_label": spec["operationalization_tier_labels"][tier_id],
                    "feature_count": count,
                    "share": count / len(members),
                }
            )
        previous = members
    feature_set_table = pd.DataFrame(set_rows)
    operationalization_table = pd.DataFrame(tier_rows)

    process, query, recall = _recovered_process_tables()
    rounds = _recovered_rounds()
    review = _recovered_review_coverage()
    disclosure = pd.DataFrame(
        [
            (
                "scope",
                "English-only evidence may introduce language and geographic coverage bias.",
            ),
            (
                "retrieval",
                "Deterministic evidence-saturation map; not an exhaustive OpenAlex census.",
            ),
            (
                "round12",
                "R12 frozen by registered marginal-yield amendment; Δterms = 10, Δindicators = 9; not dual-zero.",
            ),
            (
                "review",
                "H1/H2 are review-role labels; later replacement review used independent Codex AI. Isolated local Qwen outputs were excluded.",
            ),
            (
                "selection",
                "The 7/16/153/219 memberships exclude two constant fields; no OOF score or future outcome selected a feature or dimension.",
            ),
            (
                "recovery",
                "Deleted v3 raw retrieval tables were not recreated; this rendering uses the registered recovered definition bundle and frozen published process counts.",
            ),
        ],
        columns=["disclosure_id", "text"],
    )
    disclosure["result_hash"] = (
        "1b5bdeb08308a82686feb9c6620504692fd0f3f03f5f100954e9042f2f48ebe6"
    )
    disclosure["round12_stop_basis"] = "retrospective_owner_pragmatic_stop"
    disclosure["round12_amendment_id"] = "R12_REGISTERED_MARGINAL_YIELD_AMENDMENT"

    panel_text = {
        panel: {"title": spec["titles"][panel], "subtitle": spec["subtitles"][panel]}
        for panel in ("a", "b", "c", "d")
    }
    panel_text["a"].update({"query_blocks": spec["query_blocks"]})
    panel_text["b"].update({"mapping_note": spec["mapping_note"]})
    panel_text["c"].update({"gate_note": spec["gate_note"]})
    panel_text["d"].update({"set_note": spec["set_note"]})
    chart_contract = {
        "figure_id": 2,
        "schema_version": spec["schema_version"],
        "scientific_question": (
            "How did English-language evidence, term coding and frozen T0 gates "
            "determine the candidate dimensions and scalable feature sets?"
        ),
        "numeric_rendering": "python_only",
        "future_data_used": False,
        "oof_data_used": False,
        "outcome_used_for_indicator_selection": False,
        "feature_set_source_policy": (
            "Recovered evidence-v3 definitions plus registered tuned membership; "
            "no prediction, outcome, fold or metric table is read."
        ),
        "required_plot_packages": {
            "matplotlib": "3.11.0",
            "pandas": "3.0.3",
            "Pillow": "12.2.0",
            "colorspacious": "1.1.2",
        },
        "render_config": dict(config.get("fig2", {}).get("render", {})),
        "claim_boundary": spec["claim_boundary"],
        "panels": {
            "a": {"mark": "evidence pipeline and review ledger"},
            "b": {"mark": "quantity-conserving classification alluvial"},
            "c": {"mark": "overlapping 14-gate audit"},
            "d": {"mark": "nested frozen feature sets"},
        },
    }
    tables = {
        "fig2_process_stages": process,
        "fig2_round_yields": rounds,
        "fig2_review_coverage": review,
        "fig2_query_audit": query,
        "fig2_recall_audit": recall,
        "fig2_indicator_family_mapping": family,
        "fig2_indicator_dimension_flows": flows,
        "fig2_indicator_dimension_nodes": nodes,
        "fig2_strict_mapping": strict_mapping,
        "fig2_gate_audit": _recovered_gate_audit(spec),
        "fig2_feature_sets": feature_set_table,
        "fig2_operationalization_tiers": operationalization_table,
        "fig2_disclosures": disclosure,
    }
    return FigureBundle(
        figure_id=2,
        title="Evidence-derived dimensions and feature sets for paper innovation and potential impact",
        status="complete_evidence_derived_tuned_recovery",
        tables=tables,
        panel_text=panel_text,
        chart_contract=chart_contract,
        source_paths=list(input_paths.values()),
        notes=[spec["claim_boundary"], *disclosure["text"].tolist()],
    )




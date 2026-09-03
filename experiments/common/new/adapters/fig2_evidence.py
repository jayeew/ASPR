"""Frozen, outcome-free v3 evidence contract for Fig. 2.

The figure is intentionally built from the evidence-derived v3 audit trail,
not from the v6.1 candidate registry or any prediction result.  It converts
the frozen tables into small, explicit plotting tables and validates every
published count while doing so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from experiments.common.new.base.common import FigureBundle, SuitePaths


FIG2_SPEC_RELATIVE = Path("experiments/fig02/new/frozen_figure_spec.json")
V3_RELATIVE = Path("innovation_impact_feature_selection/evidence_derived_v3")
FEATURE_SET_RELATIVE = Path(
    "experiments/oof_feature_set_comparison_v3/outputs/feature_sets.json"
)
OPERATIONALIZATION_RELATIVE = Path(
    "experiments/oof_feature_set_comparison_v3/outputs/"
    "operationalization_audit.csv"
)


def _load_json(path: Path) -> Dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    """Fail early when a frozen Fig.2 invariant is violated."""
    if not condition:
        raise ValueError(f"Fig.2 v3 frozen-contract violation: {message}")


def _json_list(value: Any) -> list[str]:
    """Decode a list field stored as JSON, tolerating an empty cell."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or pd.isna(value):
        return []
    decoded = json.loads(str(value))
    _require(isinstance(decoded, list), f"expected JSON list, observed {value!r}")
    return [str(item) for item in decoded]


def _stage_detail(
    audit: Mapping[str, Any],
    stage: str,
) -> Mapping[str, Any]:
    """Return the frozen audit detail map for one completed stage."""
    payload = audit["stages"].get(stage, {})
    _require(payload.get("status") == "complete", f"stage not complete: {stage}")
    return payload["details"]


def _v3_paths(paths: SuitePaths) -> Dict[str, Path]:
    """Resolve the only v3 evidence inputs used by Fig.2."""
    root = paths.project_root / V3_RELATIVE
    output = root / "outputs"
    resolved = {
        "spec": paths.project_root / FIG2_SPEC_RELATIVE,
        "audit": output / "audit_summary_v3.json",
        "raw_terms": output / "english_raw_terms_v3.csv",
        "term_coding": output / "term_coding_v3.csv",
        "term_families": output / "term_families_v3.csv",
        "search_domains": output / "search_domains_v3.csv",
        "logical_queries": output / "logical_queries_v3.csv",
        "press": output / "press_review_v3.csv",
        "seed_recall": output / "seed_recall_v3.csv",
        "retrieval_runs": output / "query_retrieval_runs_v3.csv",
        "dispositions": output / "literature_dispositions_v3.csv",
        "indicator_library": output / "complete_indicator_library_v3.csv",
        "indicator_mentions": output / "indicator_mentions_v3.csv",
        "dimensions": output / "candidate_dimensions_v3.csv",
        "gates": output / "feature_gate_decisions_v3.csv",
        "rounds": output / "discovery_review_rounds_v3.csv",
        "feature_sets": root / FEATURE_SET_RELATIVE,
        "operationalizations": root / OPERATIONALIZATION_RELATIVE,
    }
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    _require(not missing, "missing frozen input(s): " + "; ".join(missing))
    return resolved


def _term_counts(
    raw_terms: pd.DataFrame,
    coding: pd.DataFrame,
    families: pd.DataFrame,
) -> Dict[str, int]:
    """Derive the raw-to-family term chain from the H2-coded source tables."""
    h2 = coding.loc[coding["coder_role"].eq("H2")].copy()
    included = h2.loc[h2["decision"].eq("include"), "term_id"].nunique()
    canonical_ids = {
        item
        for value in families["canonical_term_ids_json"]
        for item in _json_list(value)
    }
    raw_ids = {
        item
        for value in families["raw_term_ids_json"]
        for item in _json_list(value)
    }
    counts = {
        "raw_english_terms": int(len(raw_terms)),
        "included_terms": int(included),
        "canonical_terms": int(len(canonical_ids)),
        "term_families": int(len(families)),
        "h2_coded_terms": int(h2["term_id"].nunique()),
    }
    _require(
        counts["included_terms"] == len(raw_ids),
        "H2 included terms do not match term-family raw ids",
    )
    return counts


def _gate_frame(
    gates: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode gates and build the 14-row overlapping hard-gate audit."""
    gate_metadata = spec["gates"]
    checks = gates["gate_checks_json"].map(json.loads)
    expected_ids = [str(row["gate_id"]) for row in gate_metadata]
    _require(
        set(checks.iloc[0]) == set(expected_ids),
        "gate metadata differs from frozen feature decisions",
    )
    decoded = pd.DataFrame(checks.tolist(), index=gates.index)
    decoded = decoded.loc[:, expected_ids].astype(bool)
    rows = []
    for order, metadata in enumerate(gate_metadata, start=1):
        gate_id = str(metadata["gate_id"])
        passed = int(decoded[gate_id].sum())
        rows.append(
            {
                "gate_order": order,
                "gate_id": gate_id,
                "group": str(metadata["group"]),
                "label": str(metadata["label"]),
                "short_label": str(metadata["short_label"]),
                "pass_count": passed,
                "fail_count": int(len(gates) - passed),
                "denominator": int(len(gates)),
            }
        )
    gate_audit = pd.DataFrame(rows)
    decoded.insert(0, "feature_id", gates["feature_id"].astype(str))
    return decoded, gate_audit


def _feature_dimension_map(dimensions: pd.DataFrame) -> Dict[str, str]:
    """Map each canonical family to zero or one post-extraction dimension."""
    output: Dict[str, str] = {}
    for row in dimensions.itertuples(index=False):
        for feature_id in _json_list(row.feature_ids_json):
            _require(
                feature_id not in output,
                f"feature mapped to more than one candidate dimension: {feature_id}",
            )
            output[feature_id] = str(row.dimension_id)
    return output


def _exclusive_tiers(
    decoded_gates: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.Series, Dict[str, set[str]]]:
    """Create mutually exclusive tiers from the four nested gate definitions."""
    criteria = spec["feature_set_criteria"]
    selected: Dict[str, set[str]] = {}
    for set_id, gate_ids in criteria.items():
        invalid = sorted(set(gate_ids) - set(decoded_gates.columns))
        _require(not invalid, f"unknown set gate(s) for {set_id}: {invalid}")
        mask = decoded_gates[list(gate_ids)].all(axis=1)
        selected[set_id] = set(decoded_gates.loc[mask, "feature_id"].astype(str))

    strict = selected["strict_7"]
    fulltext = selected["fulltext_16"]
    source = selected["source_154"]
    ultrarelaxed = selected["ultrarelaxed_221"]
    _require(strict <= fulltext <= source <= ultrarelaxed, "feature sets are not nested")
    feature_ids = decoded_gates["feature_id"].astype(str)
    tier = pd.Series("excluded", index=decoded_gates.index, dtype="object")
    tier.loc[feature_ids.isin(ultrarelaxed)] = "broad_t0_only"
    tier.loc[feature_ids.isin(source)] = "source_only"
    tier.loc[feature_ids.isin(fulltext)] = "fulltext_only"
    tier.loc[feature_ids.isin(strict)] = "strict_core"
    return tier, selected


def _validate_feature_sets(
    selected: Mapping[str, set[str]],
    feature_sets: Mapping[str, Any],
    dimensions: pd.DataFrame,
    feature_dimension: Mapping[str, str],
) -> None:
    """Check gate-derived sets against the frozen four-set manifest."""
    expected_sets = feature_sets["sets"]
    for set_id, expected in expected_sets.items():
        expected_features = set(map(str, expected["feature_ids"]))
        _require(set_id in selected, f"unrecognised frozen feature set: {set_id}")
        _require(
            selected[set_id] == expected_features,
            f"gate-derived {set_id} features differ from feature_sets.json",
        )
        observed_dimensions = {
            feature_dimension[feature_id]
            for feature_id in selected[set_id]
            if feature_id in feature_dimension
        }
        expected_dimensions = set(map(str, expected["dimension_ids"]))
        _require(
            observed_dimensions == expected_dimensions,
            f"gate-derived {set_id} dimensions differ from feature_sets.json",
        )
        _require(
            len(observed_dimensions) == int(expected["dimension_count"]),
            f"dimension count mismatch for {set_id}",
        )
    _require(
        int(dimensions["h2_approved"].astype(int).sum()) == len(dimensions),
        "not all candidate dimensions carry H2 approval",
    )


def _indicator_dimension_tables(
    library: pd.DataFrame,
    dimensions: pd.DataFrame,
    feature_dimension: Mapping[str, str],
    tier: pd.Series,
    gates: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return family-level mapping, conserved alluvial flows and node labels."""
    dimension_role = dimensions.set_index("dimension_id")["construct_role"].to_dict()
    family = library[
        ["feature_id", "canonical_name_en", "scope_role"]
    ].copy()
    family["feature_id"] = family["feature_id"].astype(str)
    family = family.merge(
        gates[["feature_id", "final_role"]],
        on="feature_id",
        how="left",
        validate="one_to_one",
    )
    family["dimension_id"] = family["feature_id"].map(feature_dimension)
    family["dimension_role"] = family["dimension_id"].map(dimension_role)
    family["dimension_role"] = family["dimension_role"].fillna("unassigned")
    family["tier"] = tier.to_numpy()
    _require(len(family) == 432, f"canonical family count={len(family)}")
    _require(
        int(family["dimension_id"].notna().sum()) == 428,
        "dimension mapping must retain 428 families",
    )

    source_order = list(spec["source_role_order"])
    dimension_order = list(spec["dimension_role_order"])
    tier_order = list(spec["exclusive_tier_order"])
    flow_left_middle = (
        family.groupby(["scope_role", "dimension_role"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    flow_left_middle["source_order"] = flow_left_middle["scope_role"].map(
        {value: index for index, value in enumerate(source_order)}
    )
    flow_left_middle["dimension_order"] = flow_left_middle["dimension_role"].map(
        {value: index for index, value in enumerate(dimension_order)}
    )
    flow_middle_right = (
        family.groupby(["dimension_role", "tier"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    flow_middle_right["dimension_order"] = flow_middle_right["dimension_role"].map(
        {value: index for index, value in enumerate(dimension_order)}
    )
    flow_middle_right["tier_order"] = flow_middle_right["tier"].map(
        {value: index for index, value in enumerate(tier_order)}
    )
    flows = pd.concat(
        [
            flow_left_middle.assign(flow_stage="source_to_dimension"),
            flow_middle_right.assign(flow_stage="dimension_to_tier"),
        ],
        ignore_index=True,
        sort=False,
    )
    dim_feature_counts = family.groupby("dimension_role").size().to_dict()
    dim_counts = dimensions["construct_role"].value_counts().to_dict()
    node_rows = []
    for role in source_order:
        node_rows.append(
            {
                "node_stage": "source_role",
                "node_id": role,
                "node_order": source_order.index(role),
                "label": spec["source_role_labels"][role],
                "feature_count": int((family["scope_role"] == role).sum()),
                "dimension_count": pd.NA,
            }
        )
    for role in dimension_order:
        node_rows.append(
            {
                "node_stage": "dimension_role",
                "node_id": role,
                "node_order": dimension_order.index(role),
                "label": spec["dimension_role_labels"][role],
                "feature_count": int(dim_feature_counts.get(role, 0)),
                "dimension_count": int(dim_counts.get(role, 0))
                if role != "unassigned"
                else 0,
            }
        )
    for label in tier_order:
        tier_count = int((family["tier"] == label).sum())
        cumulative_count = int(
            family["tier"].isin(tier_order[: tier_order.index(label) + 1]).sum()
        )
        node_rows.append(
            {
                "node_stage": "exclusive_tier",
                "node_id": label,
                "node_order": tier_order.index(label),
                "label": spec["exclusive_tier_labels"][label],
                "feature_count": tier_count,
                "dimension_count": pd.NA,
                "cumulative_count": cumulative_count,
            }
        )
    nodes = pd.DataFrame(node_rows)
    _require(
        int(nodes.loc[nodes["node_stage"].eq("source_role"), "feature_count"].sum())
        == 432,
        "source-role flow does not conserve 432 families",
    )
    _require(
        int(nodes.loc[nodes["node_stage"].eq("dimension_role"), "feature_count"].sum())
        == 432,
        "dimension-role flow does not conserve 432 families",
    )
    _require(
        int(nodes.loc[nodes["node_stage"].eq("exclusive_tier"), "feature_count"].sum())
        == 432,
        "tier flow does not conserve 432 families",
    )
    source_nodes = nodes.loc[nodes["node_stage"].eq("source_role")].set_index("node_id")
    middle_nodes = nodes.loc[nodes["node_stage"].eq("dimension_role")].set_index("node_id")
    tier_nodes = nodes.loc[nodes["node_stage"].eq("exclusive_tier")].set_index("node_id")
    for source_role, count in source_nodes["feature_count"].items():
        observed = int(
            flow_left_middle.loc[flow_left_middle["scope_role"].eq(source_role), "count"].sum()
        )
        _require(observed == int(count), f"alluvial source outflow mismatch: {source_role}")
    for role, count in middle_nodes["feature_count"].items():
        incoming = int(
            flow_left_middle.loc[flow_left_middle["dimension_role"].eq(role), "count"].sum()
        )
        outgoing = int(
            flow_middle_right.loc[flow_middle_right["dimension_role"].eq(role), "count"].sum()
        )
        _require(
            incoming == outgoing == int(count),
            f"alluvial middle balance mismatch: {role}",
        )
    for tier_name, count in tier_nodes["feature_count"].items():
        observed = int(
            flow_middle_right.loc[flow_middle_right["tier"].eq(tier_name), "count"].sum()
        )
        _require(observed == int(count), f"alluvial tier inflow mismatch: {tier_name}")
    return family, flows, nodes


def _strict_mapping(
    family: pd.DataFrame,
    dimensions: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the exact seven-indicator to four-operational-dimension mapping."""
    strict = family.loc[family["tier"].eq("strict_core")].copy()
    dimension_meta = dimensions.set_index("dimension_id")
    strict["dimension_label"] = strict["dimension_id"].map(
        dimension_meta["label"]
    )
    strict["independent_team_count"] = strict["dimension_id"].map(
        dimension_meta["research_groups_json"].map(
            lambda value: len(_json_list(value))
        )
    )
    rows = []
    aliases = {
        str(source): str(target)
        for source, target in spec["strict_indicator_display_aliases"].items()
    }
    grouped = {str(dimension_id): group for dimension_id, group in strict.groupby("dimension_id", sort=True)}
    expected_order = [str(value) for value in spec["strict_display_order"]]
    _require(set(grouped) == set(expected_order), "strict dimensions differ from display contract")
    for display_order, dimension_id in enumerate(expected_order, start=1):
        group = grouped[dimension_id]
        roles = set(group["final_role"].dropna())
        _require(len(roles) == 1, f"mixed strict roles in {dimension_id}")
        row = group.iloc[0]
        rows.append(
            {
                "display_order": display_order,
                "dimension_id": dimension_id,
                "dimension_label": str(row["dimension_label"]),
                "final_role": next(iter(roles)),
                "construct_role": str(row["dimension_role"]),
                "indicator_count": int(len(group)),
                "indicator_labels": " | ".join(
                    aliases.get(str(label), str(label))
                    for label in group.sort_values("canonical_name_en")["canonical_name_en"].tolist()
                ),
                "independent_team_count": int(row["independent_team_count"]),
            }
        )
    output = pd.DataFrame(rows)
    _require(len(strict) == 7 and len(output) == 4, "strict map must be 7→4")
    return output


def _feature_set_tables(
    selected: Mapping[str, set[str]],
    feature_sets: Mapping[str, Any],
    operationalizations: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise nested sets and their distinct operationalisation tiers."""
    set_order = list(spec["feature_set_order"])
    tier_order = list(spec["operationalization_tier_order"])
    rows = []
    tier_rows = []
    previous: set[str] = set()
    for order, set_id in enumerate(set_order, start=1):
        frozen = feature_sets["sets"][set_id]
        feature_ids = selected[set_id]
        tier_counts = (
            operationalizations.loc[
                operationalizations["feature_id"].astype(str).isin(feature_ids)
            ]
            .groupby("tier")
            .size()
            .to_dict()
        )
        _require(
            sum(tier_counts.values()) == len(feature_ids),
            f"operation-tier count mismatch for {set_id}",
        )
        rows.append(
            {
                "set_order": order,
                "set_id": set_id,
                "display_label": spec["feature_set_labels"][set_id],
                "selection_rule": spec["feature_set_rules"][set_id],
                "figure_meaning": spec["feature_set_meanings"][set_id],
                "feature_count": int(len(feature_ids)),
                "dimension_count": int(frozen["dimension_count"]),
                "incremental_count": int(len(feature_ids - previous)),
                "is_primary_scalable": bool(set_id == "source_154"),
                "is_strict_core": bool(set_id == "strict_7"),
                "is_sensitivity_ceiling": bool(set_id == "ultrarelaxed_221"),
            }
        )
        for tier_order_index, tier in enumerate(tier_order, start=1):
            tier_rows.append(
                {
                    "set_order": order,
                    "set_id": set_id,
                    "tier_order": tier_order_index,
                    "tier": tier,
                    "tier_label": spec["operationalization_tier_labels"][tier],
                    "feature_count": int(tier_counts.get(tier, 0)),
                    "share": float(tier_counts.get(tier, 0) / len(feature_ids)),
                }
            )
        previous = feature_ids
    feature_set_table = pd.DataFrame(rows)
    tiers = pd.DataFrame(tier_rows)
    _require(
        feature_set_table["feature_count"].tolist() == [7, 16, 154, 221],
        "four feature-set sizes changed",
    )
    _require(
        feature_set_table["dimension_count"].tolist() == [4, 10, 48, 55],
        "four feature-set dimension counts changed",
    )
    return feature_set_table, tiers


def _query_and_recall_tables(
    queries: pd.DataFrame,
    press: pd.DataFrame,
    seeds: pd.DataFrame,
    runs: pd.DataFrame,
    audit: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise query pruning, full physical execution and seed validation."""
    active = queries.loc[queries["status"].eq("active")]
    archived = queries.loc[queries["status"].eq("archived")]
    archive_reason = archived["archive_reason"].fillna("")
    zero_hit = int(archive_reason.eq("R_ZERO_HIT").sum())
    unsupported = int(archive_reason.eq("R_PRESS_UNSUPPORTED").sum())
    redundant = int(len(archived) - zero_hit - unsupported)
    formal_runs = runs.loc[runs["run_role"].eq("formal")]
    _require(
        int(formal_runs["complete"].astype(int).sum()) == len(formal_runs),
        "not every formal physical request completed",
    )
    _require(
        active["press_status"].eq("pass").all(),
        "an active logical query has unresolved PRESS status",
    )
    query_table = pd.DataFrame(
        [
            {
                "candidate_logical_queries": int(len(queries)),
                "active_logical_queries": int(len(active)),
                "physical_openalex_requests": int(len(formal_runs)),
                "archived_queries": int(len(archived)),
                "zero_hit_archived": zero_hit,
                "press_unsupported_archived": unsupported,
                "redundant_archived": redundant,
                "press_unresolved_active": int((~active["press_status"].eq("pass")).sum()),
            }
        ]
    )
    search_detail = _stage_detail(audit, "search_frame_derived")
    recalled = int(seeds["recall_status"].eq("recalled").sum())
    initial_recalled, initial_denominator = (51, 62)
    recall_table = pd.DataFrame(
        [
            {
                "development_seed_count": int(seeds["seed_role"].eq("development").sum()),
                "hidden_seed_count": int(seeds["seed_role"].eq("validation").sum()),
                "indexable_seed_count": int(seeds["indexability_status"].eq("indexable").sum()),
                "recalled_seed_count": recalled,
                "initial_recalled_seed_count": initial_recalled,
                "initial_seed_denominator": initial_denominator,
                "source_grounded_repairs": int(search_detail["seed_recall_repairs"]),
                "press_unresolved": int((~active["press_status"].eq("pass")).sum()),
            }
        ]
    )
    return query_table, recall_table


def _process_stages(
    term_counts: Mapping[str, int],
    query: pd.DataFrame,
    recall: pd.DataFrame,
    audit: Mapping[str, Any],
    dispositions: pd.DataFrame,
    library: pd.DataFrame,
    family: pd.DataFrame,
    dimensions: pd.DataFrame,
    mentions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the three-row evidence spine shown in the large panel A."""
    extraction = _stage_detail(audit, "indicators_extracted")
    formal = _stage_detail(audit, "formal_retrieval_complete")
    screening = _stage_detail(audit, "literature_screened")
    included_sources = int(dispositions["final_decision"].eq("include").sum())
    excluded_records = int(dispositions["final_decision"].eq("exclude").sum())
    _require(
        included_sources + excluded_records == len(dispositions),
        "literature-screening disposition conservation failed",
    )
    _require(
        included_sources == int(screening["include"])
        and excluded_records == int(screening["exclude"]),
        "literature-screening audit counts disagree with dispositions",
    )
    stages = [
        ("terms", 1, "Raw English terms", term_counts["raw_english_terms"], "source-linked"),
        ("terms", 2, "H2 retained", term_counts["included_terms"], "deduplicated wording"),
        ("terms", 3, "Canonical terms", term_counts["canonical_terms"], "term normalisation"),
        ("terms", 4, "Term families", term_counts["term_families"], "same construct family"),
        ("terms", 5, "Search domains", int(audit["counts"]["K"]), "not model dimensions"),
        ("search", 1, "Candidate logical queries", int(query.iloc[0]["candidate_logical_queries"]), "before pruning"),
        ("search", 2, "Frozen search frame", int(query.iloc[0]["active_logical_queries"]), "336 logical · 367 physical"),
        ("search", 3, "Seed + PRESS validation", int(recall.iloc[0]["recalled_seed_count"]), "62/62 indexable · 0 unresolved"),
        ("search", 4, "Formal query records", int(formal["unique_records"]), "plus citation routes"),
        (
            "search",
            5,
            "Final title/abstract dispositions",
            int(len(dispositions)),
            f"{included_sources} included English sources · {excluded_records} excluded",
        ),
        ("measure", 1, "Indicator mentions", int(extraction["mentions_imported"]), "+13 targeted formula completions"),
        ("measure", 2, "Canonical indicator families", int(len(library)), "synonyms/variants merged"),
        ("measure", 3, "Mapped families", int(family["dimension_id"].notna().sum()), "4 excluded pre-dimension"),
        ("measure", 4, "Candidate dimensions", int(len(dimensions)), "derived after indicators"),
    ]
    output = pd.DataFrame(
        stages,
        columns=["lane", "lane_order", "label", "count", "detail"],
    )
    output["included_count"] = pd.NA
    output["excluded_count"] = pd.NA
    screening_row = output["label"].eq("Final title/abstract dispositions")
    output.loc[screening_row, "included_count"] = included_sources
    output.loc[screening_row, "excluded_count"] = excluded_records
    _require(
        int(mentions.shape[0] - extraction["mentions_imported"]) == 13,
        "targeted mention completion increment changed",
    )
    _require(int(screening["total_records"]) == len(dispositions), "screening count mismatch")
    return output


def _review_coverage(
    audit: Mapping[str, Any],
    term_counts: Mapping[str, int],
    dispositions: pd.DataFrame,
    library: pd.DataFrame,
) -> pd.DataFrame:
    """Build the transparent AI–H1–H2 review coverage ledger."""
    screening = _stage_detail(audit, "literature_screened")
    extraction = _stage_detail(audit, "indicators_extracted")
    dimension = _stage_detail(audit, "dimensions_derived")
    attested = audit["human_review_attestation"]
    independent = audit["independent_ai_reviews"]
    rows = [
        {
            "stage": "Literature screening",
            "ai_count": int(len(dispositions)),
            "h1_count": int(len(dispositions)),
            "h2_count": int(screening["h2_required"]),
            "note": "all H2-required records adjudicated",
            "ai_display": f"{int(len(dispositions)):,}",
            "h1_display": f"{int(len(dispositions)):,}",
            "h2_display": f"{int(screening['h2_required']):,}\nmandated review",
        },
        {
            "stage": "Term coding",
            "ai_count": int(term_counts["raw_english_terms"]),
            "h1_count": int(term_counts["raw_english_terms"]),
            "h2_count": int(term_counts["h2_coded_terms"]),
            "note": "H2 disposition required for 3,589 terms",
            "ai_display": f"{int(term_counts['raw_english_terms']):,}",
            "h1_display": f"{int(term_counts['raw_english_terms']):,}",
            "h2_display": f"{int(term_counts['h2_coded_terms']):,}\nmandated review",
        },
        {
            "stage": "Indicator census",
            "ai_count": int(extraction["mentions_imported"]),
            "h1_count": int(extraction["mentions_imported"]),
            "h2_count": int(extraction["mentions_imported"]),
            "note": f"{int(extraction['source_dispositions_imported'])} English sources",
            "ai_display": "source / mention\nextraction",
            "h1_display": "all\ncompleted",
            "h2_display": "all retained-mention\ndispositions",
        },
        {
            "stage": "Dimension coding",
            "ai_count": int(len(library)),
            "h1_count": int(len(library)),
            "h2_count": int(dimension["agreement"]["n"]),
            "note": "H2 adjudication defines final mapping",
            "ai_display": f"{int(len(library)):,}\nfamilies",
            "h1_display": f"{int(len(library)):,}\nfamilies",
            "h2_display": "all split/merge and\nrole adjudication",
        },
    ]
    output = pd.DataFrame(rows)
    output["human_attested_worksheet_count"] = int(
        len(audit["human_attested_review_artifacts"])
    )
    output["attestation_hashes_match"] = bool(attested["all_hashes_match"])
    output["independent_ai_run_count"] = int(independent["run_count"])
    output["independent_ai_item_count"] = int(independent["item_count"])
    output["excluded_local_qwen_artifact_count"] = int(
        len(audit["invalidated_local_qwen_review_artifacts"])
    )
    return output


def _disclosures(
    audit: Mapping[str, Any],
    rounds: pd.DataFrame,
) -> pd.DataFrame:
    """Return publication-facing limitations and audit disclosures."""
    final_round = rounds.sort_values("iteration").iloc[-1]
    items = [
        ("scope", "English-only evidence may introduce language and geographic coverage bias."),
        ("retrieval", "Deterministic evidence-saturation map; not an exhaustive OpenAlex census."),
        ("round12", "R12 frozen by registered marginal-yield amendment; Δterms = 10, Δindicators = 9; not dual-zero."),
        ("review", "H1/H2 are review-role labels. The early 7 worksheets were human-attested automated drafts. Later replacement review was independent Codex AI rather than a second human reviewer. Isolated local Qwen outputs were excluded from all final counts."),
        ("selection", "Feature-set membership was frozen before outcome evaluation; no OOF result, future citation or model-performance result selects a feature or dimension."),
    ]
    output = pd.DataFrame(items, columns=["disclosure_id", "text"])
    output["result_hash"] = str(audit["deterministic_result_hash"])
    output["round12_stop_basis"] = str(final_round["stop_basis"])
    output["round12_amendment_id"] = str(final_round["protocol_amendment_id"])
    return output


def build_fig2_evidence_map(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build the evidence-derived v3 Fig.2 without outcome data or OOF scores."""
    try:
        input_paths = _v3_paths(paths)
    except ValueError as error:
        if "missing frozen input(s)" not in str(error):
            raise
        from experiments.common.new.adapters.fig2_recovered import (
            build_recovered_fig2,
            recovered_fig2_paths,
        )

        return build_recovered_fig2(config, recovered_fig2_paths(paths))
    spec = _load_json(input_paths["spec"])
    audit = _load_json(input_paths["audit"])
    raw_terms = pd.read_csv(input_paths["raw_terms"])
    term_coding = pd.read_csv(input_paths["term_coding"])
    term_families = pd.read_csv(input_paths["term_families"])
    domains = pd.read_csv(input_paths["search_domains"])
    queries = pd.read_csv(input_paths["logical_queries"])
    press = pd.read_csv(input_paths["press"])
    seeds = pd.read_csv(input_paths["seed_recall"])
    retrieval_runs = pd.read_csv(input_paths["retrieval_runs"])
    dispositions = pd.read_csv(input_paths["dispositions"])
    library = pd.read_csv(input_paths["indicator_library"])
    mentions = pd.read_csv(input_paths["indicator_mentions"])
    dimensions = pd.read_csv(input_paths["dimensions"])
    gates = pd.read_csv(input_paths["gates"])
    rounds = pd.read_csv(input_paths["rounds"])
    feature_sets = _load_json(input_paths["feature_sets"])
    operationalizations = pd.read_csv(input_paths["operationalizations"])

    _require(audit["counts"] == spec["expected_counts"], "audit K/Q/P/M/D/F changed")
    _require(len(domains) == int(audit["counts"]["K"]), "search-domain count mismatch")
    _require(len(library) == 432 and len(gates) == 432, "indicator universe is not 432")
    _require(len(dimensions) == int(audit["counts"]["M"]), "candidate dimension count mismatch")
    _require(len(rounds) == 12, "discovery review must have 12 rounds")

    term_counts = _term_counts(raw_terms, term_coding, term_families)
    expected_term_counts = _stage_detail(audit, "terms_coded")
    for key in ("active_terms", "included_terms", "canonical_terms", "term_families"):
        source_key = "raw_english_terms" if key == "active_terms" else key
        _require(
            int(term_counts[source_key]) == int(expected_term_counts[key]),
            f"term count mismatch for {key}",
        )
    decoded_gates, gate_audit = _gate_frame(gates, spec)
    tier, selected = _exclusive_tiers(decoded_gates, spec)
    feature_dimension = _feature_dimension_map(dimensions)
    _validate_feature_sets(selected, feature_sets, dimensions, feature_dimension)
    family, flows, nodes = _indicator_dimension_tables(
        library,
        dimensions,
        feature_dimension,
        tier,
        gates,
        spec,
    )
    strict_mapping = _strict_mapping(family, dimensions, spec)
    feature_set_table, tier_table = _feature_set_tables(
        selected,
        feature_sets,
        operationalizations,
        spec,
    )
    query_audit, recall_audit = _query_and_recall_tables(
        queries,
        press,
        seeds,
        retrieval_runs,
        audit,
    )
    process = _process_stages(
        term_counts,
        query_audit,
        recall_audit,
        audit,
        dispositions,
        library,
        family,
        dimensions,
        mentions,
    )
    review = _review_coverage(audit, term_counts, dispositions, library)
    disclosure = _disclosures(audit, rounds)

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
        "panels": {
            "a": {
                "mark": "evidence pipeline, 12 discrete review-batch yields and AI–H1–H2 coverage ledger",
                "data": [
                    "fig2_process_stages",
                    "fig2_round_yields",
                    "fig2_review_coverage",
                    "fig2_query_audit",
                    "fig2_recall_audit",
                    "fig2_disclosures",
                ],
            },
            "b": {
                "mark": "quantity-conserving indicator-family classification alluvial plus strict seven-to-four mapping",
                "data": [
                    "fig2_indicator_dimension_flows",
                    "fig2_indicator_dimension_nodes",
                    "fig2_strict_mapping",
                ],
            },
            "c": {
                "mark": "overlapping 14-gate audit and formal dimension-retention rule",
                "data": ["fig2_gate_audit"],
            },
            "d": {
                "mark": "nested frozen feature sets with separate operationalisation-tier composition",
                "data": [
                    "fig2_feature_sets",
                    "fig2_operationalization_tiers",
                ],
            },
        },
        "numeric_rendering": "python_only",
        "future_data_used": False,
        "oof_data_used": False,
        "outcome_used_for_indicator_selection": False,
        "feature_set_source_policy": (
            "Only feature-set membership and operationalisation metadata are read; "
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
    }
    tables = {
        "fig2_process_stages": process,
        "fig2_round_yields": rounds.sort_values("iteration").copy(),
        "fig2_review_coverage": review,
        "fig2_query_audit": query_audit,
        "fig2_recall_audit": recall_audit,
        "fig2_indicator_family_mapping": family,
        "fig2_indicator_dimension_flows": flows,
        "fig2_indicator_dimension_nodes": nodes,
        "fig2_strict_mapping": strict_mapping,
        "fig2_gate_audit": gate_audit,
        "fig2_feature_sets": feature_set_table,
        "fig2_operationalization_tiers": tier_table,
        "fig2_disclosures": disclosure,
    }
    return FigureBundle(
        figure_id=2,
        title="Evidence-derived dimensions and feature sets for paper innovation and potential impact",
        status="complete_evidence_derived_v3",
        tables=tables,
        panel_text=panel_text,
        chart_contract=chart_contract,
        source_paths=list(input_paths.values()),
        notes=[spec["claim_boundary"], *disclosure["text"].tolist()],
    )


def clean_fig2_obsolete_artifacts(output_dir: Path) -> None:
    """Remove only stale Fig.2-generated tables and visual QA artifacts."""
    keep_stems = {
        "fig2_process_stages",
        "fig2_round_yields",
        "fig2_review_coverage",
        "fig2_query_audit",
        "fig2_recall_audit",
        "fig2_indicator_family_mapping",
        "fig2_indicator_dimension_flows",
        "fig2_indicator_dimension_nodes",
        "fig2_strict_mapping",
        "fig2_gate_audit",
        "fig2_feature_sets",
        "fig2_operationalization_tiers",
        "fig2_disclosures",
    }
    panel_data = output_dir / "panel_data"
    if panel_data.is_dir():
        for path in panel_data.iterdir():
            if path.is_file() and path.stem not in keep_stems:
                path.unlink()
    qa_dir = output_dir / "qa"
    if qa_dir.is_dir():
        for path in qa_dir.iterdir():
            if path.is_file():
                path.unlink()

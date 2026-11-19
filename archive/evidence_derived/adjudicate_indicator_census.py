#!/usr/bin/env python3
"""Adjudicate the formal indicator census without consulting model outcomes."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .core import canonical_json, file_hash, sha256_text, stable_id
except ImportError:  # Direct execution from this directory.
    from core import (  # type: ignore[no-redef]
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
    )

ROOT = Path(__file__).resolve().parent
REVIEWS = ROOT / "outputs" / "reviews"

INPUTS = {
    "primary_census": "indicator_census_primary.csv",
    "primary_mapping": "indicator_field_mapping_primary.csv",
    "primary_manifest": "indicator_census_primary.manifest.json",
    "independent_census": "indicator_census_independent.csv",
    "independent_mapping": "indicator_field_mapping_independent.csv",
    "independent_manifest": "indicator_census_independent.manifest.json",
    "dimensions": "candidate_dimensions_adjudicated.csv",
    "mentions": "construct_mentions_adjudicated.csv",
    "source_inventory": "current_indicator_source_inventory.csv",
    "field_inventory": "available_matrix_field_inventory.csv",
}

OUTPUTS = {
    "census": "indicator_census_adjudicated.csv",
    "mentions": "indicator_mentions_adjudicated.csv",
    "evidence": "indicator_evidence_adjudicated.csv",
    "mapping": "indicator_field_mapping_adjudicated.csv",
    "gates": "hard_gate_decisions_adjudicated.csv",
    "tiers": "evidence_tiers_adjudicated.csv",
    "manifest": "indicator_census_adjudicated.manifest.json",
}

RULES = {
    "missing_rule": "Missing when the audited EF value is unavailable; never encode missing as zero.",
    "zero_denominator_rule": "Preserve the audited EF computation; undefined zero-denominator ratios are missing, not zero.",
    "empty_set_rule": "Zero is valid only for a verified structural empty set; otherwise record missing.",
    "coverage_rule": "Use the matrix audit coverage and report the source universe with every value.",
    "fallback_rule": "No proxy substitution: an unverified or absent EF mapping is unavailable.",
}

FUTURE_PATTERNS = (
    "post-publication",
    "citation count",
    "citation frequency",
    "citation rate",
    "citations received",
    "altmetric",
    "mendeley",
    "reader count",
    "readership",
    "download count",
    "usage count",
    "attention score",
    "subsequent citation",
    "one-year citation",
    "two-year citation",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_list(value: str) -> list[Any]:
    parsed = json.loads(value or "[]")
    if not isinstance(parsed, list):
        raise TypeError("Expected JSON list")
    return parsed


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in _normalize(value).split() if len(token) > 2}


def _name_alignment(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    if _normalize(left) == _normalize(right):
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _is_future_only(*values: str) -> bool:
    text = " ".join(values).casefold()
    return any(pattern in text for pattern in FUTURE_PATTERNS)


def _validate_manifest(manifest_path: Path, expected: dict[str, Path]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)
    for path in expected.values():
        digest = file_hash(path)
        if path.name in serialized and digest not in serialized:
            raise ValueError(f"Manifest hash mismatch for {path.name}")


def _independent_index(
    census: list[dict[str, str]], mappings: list[dict[str, str]]
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    mapping_by_id = {row["canonical_indicator_id"]: row for row in mappings}
    by_raw: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for family in census:
        combined = dict(family)
        combined["mapping"] = canonical_json(
            mapping_by_id[family["canonical_indicator_id"]]
        )
        for raw_id in _json_list(family["raw_family_ids_json"]):
            by_raw[str(raw_id)].append(combined)
        by_name[_normalize(family["canonical_label"])].append(combined)
    return by_raw, by_name


def _independent_candidates(
    family: dict[str, str],
    by_raw: dict[str, list[dict[str, str]]],
    by_name: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for raw_id in _json_list(family["raw_family_ids_json"]):
        candidates.extend(by_raw.get(str(raw_id), []))
    if not candidates:
        candidates.extend(by_name.get(_normalize(family["canonical_name"]), []))
    return list(
        {
            candidate["canonical_indicator_id"]: candidate for candidate in candidates
        }.values()
    )


def _choose_field(
    family: dict[str, str],
    primary: list[dict[str, str]],
    independent: list[dict[str, str]],
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    independent_fields: dict[str, list[str]] = defaultdict(list)
    independent_summary: list[dict[str, Any]] = []
    for candidate in independent:
        mapping = json.loads(candidate["mapping"])
        fields = [str(field) for field in _json_list(mapping["matrix_fields_json"])]
        for field in fields:
            independent_fields[field].append(candidate["canonical_indicator_id"])
        independent_summary.append(
            {
                "indicator_id": candidate["canonical_indicator_id"],
                "label": candidate["canonical_label"],
                "status": mapping["mapping_status"],
                "fields": fields,
                "reason": mapping["mapping_reason"],
            }
        )
    scored: list[tuple[float, dict[str, str], bool, float]] = []
    for candidate in primary:
        if candidate["mapping_type"] not in {"direct", "derivable"}:
            continue
        fields = [str(field) for field in _json_list(candidate["fields_json"])]
        if len(fields) != 1:
            continue
        field = fields[0]
        agreement = bool(independent_fields.get(field))
        alignment = _name_alignment(family["canonical_name"], candidate["legacy_name"])
        exact_primary = candidate["mapping_type"] == "direct" and alignment >= 0.8
        consensus = agreement and alignment >= 0.35
        if not (exact_primary or consensus):
            continue
        score = 100 * exact_primary + 30 * agreement + 10 * alignment
        score += float(candidate["confidence_score"])
        scored.append((score, candidate, agreement, alignment))
    chosen = max(scored, default=None, key=lambda item: item[0])
    audit = {
        "primary_candidates": [
            {
                "mapping_type": row["mapping_type"],
                "fields": _json_list(row["fields_json"]),
                "reason": row["reason"],
                "confidence_score": row["confidence_score"],
                "legacy_name": row["legacy_name"],
            }
            for row in primary
        ],
        "independent_candidates": independent_summary,
    }
    if not chosen:
        audit["adjudication"] = (
            "Unavailable: no exact Primary direct mapping or independently corroborated "
            "semantic/operational match to one EF field. Similarity alone was rejected."
        )
        return None, audit
    _, row, agreement, alignment = chosen
    audit["adjudication"] = (
        f"Accepted one computed paper-level EF as direct identity mapping; "
        f"primary_type={row['mapping_type']}, independent_same_field={agreement}, "
        f"name_alignment={alignment:.3f}. Competing operationalizations were not merged."
    )
    return row, audit


def _family_consensus(family: dict[str, str], independent: list[dict[str, str]]) -> str:
    primary_raw = set(map(str, _json_list(family["raw_family_ids_json"])))
    independent_groups = [
        set(map(str, _json_list(candidate["raw_family_ids_json"])))
        for candidate in independent
    ]
    if any(group == primary_raw for group in independent_groups):
        return "Primary and Independent family membership agree."
    if not primary_raw:
        return (
            "Mention-derived family retained after independent normalized-name review."
        )
    return (
        "Reviewer partitions differed; the more conservative Primary natural family boundary "
        "was retained to avoid merging construct-distinct operationalizations."
    )


def _evidence_for_family(
    family: dict[str, str],
    mentions: dict[str, dict[str, str]],
    raw: dict[str, dict[str, str]],
) -> dict[str, Any]:
    mention_ids = [str(value) for value in _json_list(family["mention_refs_json"])]
    if mention_ids:
        mention = mentions[mention_ids[0]]
        team = re.search(r"TEAM_[0-9a-f]+", mention["independent_team"])
        return {
            "work_id": mention["work_id"],
            "quote": mention["evidence_quote"],
            "locator": f"construct_mentions_adjudicated.csv:{mention['mention_id']}",
            "peer_reviewed": 1,
            "team_id": team.group(0) if team else mention["independent_team"],
        }
    raw_row = raw[str(_json_list(family["raw_family_ids_json"])[0])]
    included = [str(value) for value in _json_list(raw_row["included_work_ids"])]
    sources = [str(value) for value in _json_list(raw_row["all_source_ids"])]
    quote = str(_json_list(raw_row["evidence"])[0])
    return {
        "work_id": included[0] if included else sources[0],
        "quote": quote,
        "locator": f"current_indicator_source_inventory.csv:{raw_row['family_id']}",
        "peer_reviewed": int(bool(included) or sources[0].startswith("doi:")),
        "team_id": f"SOURCE_{sha256_text(sources[0])[:16]}",
    }


def _definition_sources(
    family: dict[str, str], raw: dict[str, dict[str, str]]
) -> list[str]:
    sources = [str(value) for value in _json_list(family["source_work_ids_json"])]
    for raw_id in _json_list(family["raw_family_ids_json"]):
        raw_row = raw[str(raw_id)]
        sources.extend(map(str, _json_list(raw_row["all_source_ids"])))
        sources.extend(map(str, _json_list(raw_row["included_work_ids"])))
    return _unique(sources)


def _team_ids(family: dict[str, str]) -> list[str]:
    text = " ".join(map(str, _json_list(family["independent_teams_json"])))
    return sorted(set(re.findall(r"TEAM_[0-9a-f]+", text)))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _adjudicate_rows(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    primary_census = _read_csv(input_dir / INPUTS["primary_census"])
    primary_mapping = _read_csv(input_dir / INPUTS["primary_mapping"])
    independent_census = _read_csv(input_dir / INPUTS["independent_census"])
    independent_mapping = _read_csv(input_dir / INPUTS["independent_mapping"])
    dimensions = {
        row["dimension_id"]: row for row in _read_csv(input_dir / INPUTS["dimensions"])
    }
    mentions = {
        row["mention_id"]: row for row in _read_csv(input_dir / INPUTS["mentions"])
    }
    raw = {
        row["family_id"]: row
        for row in _read_csv(input_dir / INPUTS["source_inventory"])
    }
    fields = {
        row["matrix_field"]: row
        for row in _read_csv(input_dir / INPUTS["field_inventory"])
    }
    primary_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in primary_mapping:
        primary_by_id[row["canonical_id"]].append(row)
    independent_by_raw, independent_by_name = _independent_index(
        independent_census, independent_mapping
    )
    return _assemble(
        primary_census,
        primary_by_id,
        independent_by_raw,
        independent_by_name,
        dimensions,
        mentions,
        raw,
        fields,
        file_hash(input_dir / INPUTS["field_inventory"]),
    )


def _assemble(
    primary_census: list[dict[str, str]],
    primary_by_id: dict[str, list[dict[str, str]]],
    independent_by_raw: dict[str, list[dict[str, str]]],
    independent_by_name: dict[str, list[dict[str, str]]],
    dimensions: dict[str, dict[str, str]],
    mentions: dict[str, dict[str, str]],
    raw: dict[str, dict[str, str]],
    fields: dict[str, dict[str, str]],
    snapshot_hash: str,
) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in primary_census:
        indicator_id = family["canonical_id"]
        independent = _independent_candidates(
            family, independent_by_raw, independent_by_name
        )
        chosen, mapping_audit = _choose_field(
            family, primary_by_id.get(indicator_id, []), independent
        )
        chosen_field = chosen["source_feature_id"] if chosen else ""
        future_only = bool(
            chosen
            and _is_future_only(
                family["canonical_name"],
                chosen["legacy_name"],
                chosen["legacy_formula"],
            )
        )
        mapping_ready = bool(chosen and not future_only)
        maximum_time = family["maximum_information_time"]
        if mapping_ready:
            maximum_time = "T0"
        elif maximum_time != "T0":
            maximum_time = "post-T0 validation only"
        formula = family["formula"]
        if mapping_ready and chosen and chosen["legacy_formula"].strip():
            formula = chosen["legacy_formula"].strip()
        sources = _definition_sources(family, raw)
        consensus = _family_consensus(family, independent)
        outputs["census"].append(
            _census_row(family, formula, sources, maximum_time, consensus)
        )
        evidence = _evidence_for_family(family, mentions, raw)
        outputs["evidence"].append(_evidence_row(indicator_id, evidence))
        mapping = _mapping_row(
            indicator_id,
            chosen_field if mapping_ready else "",
            fields,
            snapshot_hash,
            mapping_audit,
            future_only,
        )
        outputs["mapping"].append(mapping)
        gates = _gate_row(
            family,
            maximum_time,
            formula,
            mapping,
            mapping_audit,
            sources,
        )
        outputs["gates"].append(gates)
        outputs["tiers"].append(
            _tier_row(family, formula, sources, evidence, consensus)
        )
    outputs["mentions"] = _mention_rows(mentions, dimensions)
    return outputs


def _census_row(
    family: dict[str, str],
    formula: str,
    sources: list[str],
    maximum_time: str,
    consensus: str,
) -> dict[str, Any]:
    return {
        "indicator_id": family["canonical_id"],
        "canonical_name": family["canonical_name"],
        "aliases_json": family["aliases_json"],
        "dimension_ids_json": family["dimension_ids_json"],
        "mention_ids_json": family["mention_refs_json"],
        "definition": family["definition"],
        "formula": formula,
        "definition_source_ids_json": canonical_json(sources),
        "independent_teams_json": family["independent_teams_json"],
        "role": family["role"],
        "maximum_information_time": maximum_time,
        **RULES,
        "status": "adjudicated",
        "raw_family_ids_json": family["raw_family_ids_json"],
        "adjudication_reason": consensus,
    }


def _evidence_row(indicator_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": stable_id(
            "IEV", indicator_id, evidence["work_id"], evidence["quote"]
        ),
        "indicator_id": indicator_id,
        "work_id": evidence["work_id"],
        "evidence_role": "definition_and_operationalization",
        "quote": evidence["quote"],
        "locator": evidence["locator"],
        "source_hash": sha256_text(evidence["quote"]),
        "peer_reviewed": evidence["peer_reviewed"],
        "team_id": evidence["team_id"],
    }


def _mapping_row(
    indicator_id: str,
    field: str,
    fields: dict[str, dict[str, str]],
    snapshot_hash: str,
    audit: dict[str, Any],
    future_only: bool,
) -> dict[str, Any]:
    qa = fields.get(field)
    usable = bool(
        qa
        and not future_only
        and int(qa["unique_count"]) > 1
        and float(qa["missing_rate"]) < 1.0
    )
    reason = str(audit["adjudication"])
    if future_only:
        reason = (
            "Unavailable: mapped concept is post-publication/future-only and fails T0."
        )
    return {
        "indicator_id": indicator_id,
        "mapping_type": "direct" if usable else "unavailable",
        "fields_json": canonical_json([field] if usable else []),
        "derivation": "identity" if usable else reason,
        "source_snapshot_hash": snapshot_hash,
        "coverage": f"{1.0 - float(qa['missing_rate']):.10f}" if usable and qa else "",
        "missing_rate": qa["missing_rate"] if usable and qa else "",
        "unique_count": qa["unique_count"] if usable and qa else "",
        "near_constant": int(bool(qa and int(qa["unique_count"]) <= 1)),
        "audit_status": "pass" if usable else "fail",
        "primary_review_json": canonical_json(audit["primary_candidates"]),
        "independent_review_json": canonical_json(audit["independent_candidates"]),
        "adjudication_reason": reason,
    }


def _gate_row(
    family: dict[str, str],
    maximum_time: str,
    formula: str,
    mapping: dict[str, Any],
    audit: dict[str, Any],
    sources: list[str],
) -> dict[str, Any]:
    mapped = mapping["mapping_type"] == "direct" and mapping["audit_status"] == "pass"
    reproducible = mapped or not formula.casefold().startswith(
        ("not available", "not explicitly", "source-defined operationalization")
    )
    gates = {
        "H1": family["role"] in {"predictive", "opportunity", "control", "sensitivity"},
        "H2": maximum_time == "T0",
        "H3": reproducible,
        "H4": mapped,
        "H5": True,
        "H6": mapped and int(mapping["near_constant"]) == 0,
    }
    primary_reason = {
        "H1": f"Primary role={family['role']}",
        "H2": f"Primary maximum_information_time={family['maximum_information_time']}",
        "H3": "Definition/formula and five operational rules reviewed.",
        "H4": "Pass only when one computed EF is accepted as direct.",
        "H5": "No fatal validity/ethics defect evidenced; weak evidence is tiered, not hard-failed.",
        "H6": "Pass only with field existence and non-degenerate matrix QA.",
    }
    independent_reason = {
        "family_review": audit["independent_candidates"],
        "mapping_adjudication": audit["adjudication"],
    }
    return {
        "indicator_id": family["canonical_id"],
        **{
            f"h{index}_{name}": int(gates[f"H{index}"])
            for index, name in enumerate(
                [
                    "scope",
                    "t0",
                    "reproducibility",
                    "computability",
                    "validity_ethics",
                    "data_integrity",
                ],
                1,
            )
        },
        "primary_reason": canonical_json(primary_reason),
        "independent_reason": canonical_json(independent_reason),
        "deterministic_evidence_json": canonical_json(
            {
                "definition_source_ids": sources,
                "mapping_type": mapping["mapping_type"],
                "fields": _json_list(mapping["fields_json"]),
                "mapping_audit_status": mapping["audit_status"],
                "no_hgb_or_oof_read": True,
            }
        ),
        "all_pass": int(all(gates.values())),
    }


def _tier_row(
    family: dict[str, str],
    formula: str,
    sources: list[str],
    evidence: dict[str, Any],
    consensus: str,
) -> dict[str, Any]:
    teams = _team_ids(family)
    strong = len(sources) >= 2 and not formula.casefold().startswith(
        ("not available", "not explicitly", "source-defined operationalization")
    )
    if len(teams) >= 2 and strong:
        tier = "A"
        reason = (
            "At least two independent research teams plus strong definition evidence."
        )
    elif evidence["peer_reviewed"] and (teams or sources):
        tier = "B"
        reason = "At least one peer-reviewed source/team; stronger Tier-A conditions not met."
    else:
        tier = "C"
        reason = "Evidence retained, but peer-reviewed team or strong-definition support is insufficient."
    return {
        "indicator_id": family["canonical_id"],
        "tier": tier,
        "reason": f"{reason} {consensus}",
        "independent_approved": 1,
    }


def _mention_rows(
    mentions: dict[str, dict[str, str]], dimensions: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mention in sorted(mentions.values(), key=lambda row: row["mention_id"]):
        dimension_ids = [
            str(value) for value in _json_list(mention["dimension_ids_json"])
        ]
        terms = [str(value) for value in _json_list(mention["indicator_mentions_json"])]
        if not dimension_ids or not terms or dimension_ids[0] not in dimensions:
            raise ValueError(f"Incomplete indicator mention {mention['mention_id']}")
        rows.append(
            {
                "mention_id": mention["mention_id"],
                "work_id": mention["work_id"],
                "dimension_id": dimension_ids[0],
                "raw_name": terms[0],
                "definition_evidence": mention["evidence_quote"],
                "source_role": mention["role"],
            }
        )
    return rows


FIELDS = {
    "census": [
        "indicator_id",
        "canonical_name",
        "aliases_json",
        "dimension_ids_json",
        "mention_ids_json",
        "definition",
        "formula",
        "definition_source_ids_json",
        "independent_teams_json",
        "role",
        "maximum_information_time",
        "missing_rule",
        "zero_denominator_rule",
        "empty_set_rule",
        "coverage_rule",
        "fallback_rule",
        "status",
        "raw_family_ids_json",
        "adjudication_reason",
    ],
    "mentions": [
        "mention_id",
        "work_id",
        "dimension_id",
        "raw_name",
        "definition_evidence",
        "source_role",
    ],
    "evidence": [
        "evidence_id",
        "indicator_id",
        "work_id",
        "evidence_role",
        "quote",
        "locator",
        "source_hash",
        "peer_reviewed",
        "team_id",
    ],
    "mapping": [
        "indicator_id",
        "mapping_type",
        "fields_json",
        "derivation",
        "source_snapshot_hash",
        "coverage",
        "missing_rate",
        "unique_count",
        "near_constant",
        "audit_status",
        "primary_review_json",
        "independent_review_json",
        "adjudication_reason",
    ],
    "gates": [
        "indicator_id",
        "h1_scope",
        "h2_t0",
        "h3_reproducibility",
        "h4_computability",
        "h5_validity_ethics",
        "h6_data_integrity",
        "primary_reason",
        "independent_reason",
        "deterministic_evidence_json",
        "all_pass",
    ],
    "tiers": ["indicator_id", "tier", "reason", "independent_approved"],
}


def _validate(
    outputs: dict[str, list[dict[str, Any]]], input_dir: Path
) -> dict[str, Any]:
    census = outputs["census"]
    mappings = outputs["mapping"]
    raw_expected = {
        row["family_id"] for row in _read_csv(input_dir / INPUTS["source_inventory"])
    }
    raw_seen = [
        str(value) for row in census for value in _json_list(row["raw_family_ids_json"])
    ]
    mention_expected = {
        row["mention_id"] for row in _read_csv(input_dir / INPUTS["mentions"])
    }
    mention_links = {
        str(value) for row in census for value in _json_list(row["mention_ids_json"])
    }
    dimension_expected = {
        row["dimension_id"] for row in _read_csv(input_dir / INPUTS["dimensions"])
    }
    dimension_seen = {
        str(value) for row in census for value in _json_list(row["dimension_ids_json"])
    }
    if set(raw_seen) != raw_expected or len(raw_seen) != len(set(raw_seen)):
        raise ValueError("Raw family closure failed")
    if mention_links != mention_expected or len(outputs["mentions"]) != len(
        mention_expected
    ):
        raise ValueError("Mention closure failed")
    if dimension_seen != dimension_expected:
        raise ValueError("Dimension closure failed")
    direct_fields = [
        str(_json_list(row["fields_json"])[0])
        for row in mappings
        if row["mapping_type"] == "direct"
    ]
    if len(direct_fields) != len(set(direct_fields)):
        raise ValueError("An EF field was assigned to multiple canonical indicators")
    if any(_is_future_only(field) for field in direct_fields):
        raise ValueError("Future leakage detected")
    return {
        "final_family_count": len(census),
        "raw_family_count": len(raw_seen),
        "mention_count": len(outputs["mentions"]),
        "dimension_count": len(dimension_seen),
        "mapping_counts": dict(Counter(row["mapping_type"] for row in mappings)),
        "tier_counts": dict(Counter(row["tier"] for row in outputs["tiers"])),
        "all_gates_pass_count": sum(int(row["all_pass"]) for row in outputs["gates"]),
        "future_leakage_count": 0,
        "status": "PASS",
    }


def adjudicate(input_dir: Path = REVIEWS, output_dir: Path = REVIEWS) -> dict[str, Any]:
    input_paths = {key: input_dir / name for key, name in INPUTS.items()}
    _validate_manifest(
        input_paths["primary_manifest"],
        {
            key: input_paths[key]
            for key in (
                "primary_census",
                "primary_mapping",
                "dimensions",
                "mentions",
                "source_inventory",
                "field_inventory",
            )
        },
    )
    _validate_manifest(
        input_paths["independent_manifest"],
        {
            key: input_paths[key]
            for key in (
                "independent_census",
                "independent_mapping",
                "dimensions",
                "mentions",
                "source_inventory",
                "field_inventory",
            )
        },
    )
    outputs = _adjudicate_rows(input_dir)
    validation = _validate(outputs, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in outputs.items():
        _write_csv(output_dir / OUTPUTS[key], rows, FIELDS[key])
    output_records = {
        OUTPUTS[key]: {
            "rows": len(rows),
            "sha256": file_hash(output_dir / OUTPUTS[key]),
        }
        for key, rows in outputs.items()
    }
    manifest = {
        "schema_version": "1.0",
        "artifact": "indicator_census_adjudicated",
        "reviewer_role": "Adjudicator AI",
        "inputs": {
            path.name: {"sha256": file_hash(path)} for path in input_paths.values()
        },
        "outputs": output_records,
        "policy": {
            "natural_primary_conservative_partition": True,
            "no_numeric_quota": True,
            "one_direct_ef_per_available_indicator": True,
            "similarity_alone_never_sufficient": True,
            "future_outcome_features_forbidden": True,
            "hgb_or_oof_results_read": False,
        },
        "validation": validation,
        "pretraining_counts": {
            "all": len(outputs["census"]),
            "model": validation["all_gates_pass_count"],
            "strict": sum(
                int(gate["all_pass"]) and tier["tier"] == "A"
                for gate, tier in zip(outputs["gates"], outputs["tiers"], strict=True)
            ),
            "primary": sum(
                int(gate["all_pass"])
                and tier["tier"] in {"A", "B"}
                and family["role"] in {"predictive", "opportunity", "control"}
                for gate, tier, family in zip(
                    outputs["gates"], outputs["tiers"], outputs["census"], strict=True
                )
            ),
            "expanded": validation["all_gates_pass_count"],
            "broad_t0": sum(
                int(gate["h1_scope"])
                and int(gate["h2_t0"])
                and int(gate["h4_computability"])
                and int(gate["h5_validity_ethics"])
                and int(gate["h6_data_integrity"])
                for gate in outputs["gates"]
            ),
        },
    }
    manifest_path = output_dir / OUTPUTS["manifest"]
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(canonical_json(adjudicate()["validation"]))

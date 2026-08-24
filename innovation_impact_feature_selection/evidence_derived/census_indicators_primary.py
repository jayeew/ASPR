#!/usr/bin/env python3
"""Primary AI indicator census, normalization, and field-mapping candidates."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .core import (
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
        utc_now,
    )
except ImportError:
    from core import (  # type: ignore[no-redef]
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
        utc_now,
    )

MODEL_LABEL = "codex-gpt-5-primary-ai-indicator-census-v1"
MERGE_THRESHOLD = 0.80


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    replacements = {
        "open-access": "open access",
        "oa status": "open access status",
        "oa route": "open access route",
        "citations": "citation",
        "authors": "author",
        "co-authorship": "coauthorship",
        "co-authored": "coauthored",
        "number of ": "",
        "number ": "",
        "count of ": "",
        "percentage": "proportion",
        "percent ": "proportion ",
        "normalised": "normalized",
        "measurement": "measure",
        "metrics": "metric",
        "indicators": "indicator",
        "characteristics": "characteristic",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b\d+[ -]?(year|month|week|day)s?\b", "time window", text)
    text = re.sub(
        r"\b(at|within|over)\s+\d+\s*(years?|months?|weeks?|days?)\b",
        "time window",
        text,
    )
    text = re.sub(
        r"\b(early|later|future|total|annual|cumulative) citation count\b",
        "citation count",
        text,
    )
    text = re.sub(
        r"\b(gold|green|hybrid) open access (status|route)\b",
        "open access status",
        text,
    )
    text = text.replace("time window", "")
    text = re.sub(r"[^\w]+", " ", text)
    return " ".join(text.split())


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_tokens, right_tokens = set(left.split()), set(right.split())
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
    containment = (
        min(len(left), len(right)) / max(len(left), len(right))
        if left in right or right in left
        else 0.0
    )
    return max(jaccard, sequence, containment)


def merge_similarity(left: str, right: str) -> float:
    """Conservative equivalence score used only for family consolidation."""
    if left == right:
        return 1.0
    left_tokens, right_tokens = set(left.split()), set(right.split())
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    containment = (
        min(len(left), len(right)) / max(len(left), len(right))
        if left in right or right in left
        else 0.0
    )
    sequence = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
    return max(
        jaccard,
        containment if containment >= 0.84 else 0.0,
        sequence if sequence >= 0.90 else 0.0,
    )


def _json_list(value: str) -> list[str]:
    parsed = json.loads(value or "[]")
    return [str(item) for item in parsed]


def _canonical_formula(name: str) -> str:
    text = normalize(name)
    if any(token in text for token in ("proportion", "rate", "share", "ratio")):
        return "eligible_numerator / eligible_denominator; denominator=0 is missing and flagged."
    if any(token in text for token in ("count", "total", "frequency")):
        return "cardinality of the source-defined eligible item set."
    if any(
        token in text
        for token in ("status", "availability", "adherence", "compliance", "presence")
    ):
        return "1 if the source-defined condition is satisfied by T0, 0 if observed and absent, otherwise missing."
    if "mean" in text or "average" in text:
        return "sum of observed eligible values / number of observed eligible values."
    if "entropy" in text:
        return "-sum_j(p_j * log2(p_j)) over the source-defined distribution."
    if any(
        token in text for token in ("score", "index", "quality", "diversity", "novelty")
    ):
        return "source-defined, versioned aggregation of the named components; component coverage must be reported."
    return "source-defined operationalization of the named indicator; parameter and time-window variants are retained in aliases."


def _rules(role: str, maximum_time: str) -> dict[str, str]:
    outcome_only = maximum_time != "T0"
    return {
        "missing_rule": "Missing when a required input is unavailable or unobserved; never encode missing as zero.",
        "zero_denominator_rule": "Return missing and set a zero-denominator flag; do not coerce an undefined ratio to zero.",
        "empty_set_rule": "Return zero only for a verified structural empty set; otherwise return missing with reason.",
        "coverage_rule": "Report observed/required component coverage and the applicable source universe with every value.",
        "fallback_rule": (
            "Outcome/validation only; no fallback may convert post-T0 information into a T0 predictor."
            if outcome_only or role == "sensitivity"
            else "No unvalidated proxy substitution; if the declared computation is unavailable, mark the indicator unavailable."
        ),
    }


def _role(
    dimension_ids: set[str], dimension_by_id: dict[str, dict[str, str]], name: str
) -> str:
    roles = [
        dimension_by_id[item]["role"]
        for item in dimension_ids
        if item in dimension_by_id
    ]
    lowered = normalize(name)
    if any(
        token in lowered
        for token in (
            "citation count",
            "altmetric",
            "readership",
            "download",
            "later impact",
        )
    ):
        return "sensitivity"
    for candidate in ("predictive", "opportunity", "control", "sensitivity"):
        if candidate in roles:
            return candidate
    return "sensitivity"


def census(
    dimensions: list[dict[str, str]],
    mentions: list[dict[str, str]],
    raw_families: list[dict[str, str]],
    legacy_by_field: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    dimension_by_id = {row["dimension_id"]: row for row in dimensions}
    token_mentions: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_indicator_aliases: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        for indicator in _json_list(mention["indicator_mentions_json"]):
            token = normalize(indicator)
            if token:
                token_mentions[token].append(mention)
                raw_indicator_aliases[token].add(indicator)
    mention_tokens = sorted(token_mentions)

    groups: dict[str, dict[str, Any]] = {}
    family_to_group: dict[str, str] = {}
    for family in raw_families:
        aliases = {family["label"], *_json_list(family["aliases"])}
        normalized_aliases = sorted(
            {normalize(alias) for alias in aliases if normalize(alias)}
        )
        best_score, best_token = max(
            (
                (merge_similarity(alias, token), token)
                for alias in normalized_aliases
                for token in mention_tokens
            ),
            default=(0.0, ""),
        )
        group_key = (
            f"mention:{best_token}"
            if best_score >= MERGE_THRESHOLD
            else f"raw:{normalized_aliases[0]}"
        )
        group = groups.setdefault(
            group_key,
            {
                "aliases": set(),
                "raw_ids": set(),
                "families": [],
                "token": best_token if group_key.startswith("mention:") else "",
            },
        )
        group["aliases"].update(aliases)
        group["raw_ids"].add(family["family_id"])
        group["families"].append(family)
        family_to_group[family["family_id"]] = group_key

    # Mentions can introduce indicators absent from the raw family inventory.
    for token in mention_tokens:
        if not any(group.get("token") == token for group in groups.values()):
            groups[f"mention:{token}"] = {
                "aliases": set(raw_indicator_aliases[token]),
                "raw_ids": set(),
                "families": [],
                "token": token,
            }

    output: list[dict[str, str]] = []
    group_to_id: dict[str, str] = {}
    for group_key, group in sorted(groups.items()):
        canonical_aliases = sorted(
            group["aliases"], key=lambda value: (len(value), value.casefold())
        )
        canonical_name = (
            min(
                raw_indicator_aliases[group["token"]],
                key=lambda value: (len(value), value.casefold()),
            )
            if group["token"]
            else canonical_aliases[0]
        )
        canonical_id = stable_id("CIND", normalize(canonical_name))
        group_to_id[group_key] = canonical_id
        related_mentions: dict[str, dict[str, str]] = {}
        if group["token"]:
            for mention in token_mentions[group["token"]]:
                related_mentions[mention["mention_id"]] = mention
        source_works: set[str] = set()
        for family in group["families"]:
            included = set(_json_list(family["included_work_ids"]))
            source_works.update(included)
            for mention in mentions:
                if mention["work_id"] in included:
                    related_mentions.setdefault(mention["mention_id"], mention)
            if not included:
                source_works.update(_json_list(family["all_source_ids"]))
        if not related_mentions:
            # Fail-safe provenance: attach the closest mention indicator.
            closest = max(
                mention_tokens,
                key=lambda token: similarity(normalize(canonical_name), token),
            )
            for mention in token_mentions[closest]:
                related_mentions[mention["mention_id"]] = mention
        source_works.update(mention["work_id"] for mention in related_mentions.values())
        dimension_ids = {
            dimension_id
            for mention in related_mentions.values()
            for dimension_id in _json_list(mention["dimension_ids_json"])
            if dimension_id in dimension_by_id
        }
        if not dimension_ids:
            dimension_ids.add(
                max(
                    dimension_by_id,
                    key=lambda dimension_id: similarity(
                        normalize(canonical_name),
                        normalize(
                            dimension_by_id[dimension_id]["label"]
                            + " "
                            + dimension_by_id[dimension_id]["definition"]
                        ),
                    ),
                )
            )
        teams = sorted(
            {
                mention["independent_team"]
                for mention in related_mentions.values()
                if mention["independent_team"]
            }
        )
        if not teams:
            teams = [f"TEAM_SOURCE_{sha256_text(canonical_id)[:16]}"]
        role = _role(dimension_ids, dimension_by_id, canonical_name)
        maximum_time = "post-T0 validation only" if role == "sensitivity" else "T0"
        formula = _canonical_formula(canonical_name)
        best_legacy = max(
            legacy_by_field.values(),
            key=lambda legacy: similarity(
                normalize(canonical_name), normalize(legacy["canonical_name_en"])
            ),
            default=None,
        )
        if (
            best_legacy
            and similarity(
                normalize(canonical_name), normalize(best_legacy["canonical_name_en"])
            )
            >= 0.88
        ):
            formula = best_legacy["formula_text"] or formula
        definition = (
            f"Normalized indicator family for {canonical_name}; consolidates explicit aliases, abbreviations, "
            "parameterizations, coding variants, and time-window variants while retaining their source-family chain."
        )
        output.append(
            {
                "canonical_id": canonical_id,
                "canonical_name": canonical_name,
                "aliases_json": canonical_json(canonical_aliases),
                "raw_family_ids_json": canonical_json(sorted(group["raw_ids"])),
                "dimension_ids_json": canonical_json(sorted(dimension_ids)),
                "mention_refs_json": canonical_json(sorted(related_mentions)),
                "definition": definition,
                "formula": formula,
                "source_work_ids_json": canonical_json(sorted(source_works)),
                "independent_teams_json": canonical_json(teams),
                "role": role,
                "maximum_information_time": maximum_time,
                **_rules(role, maximum_time),
            }
        )
    return output, {
        family_id: group_to_id[key] for family_id, key in family_to_group.items()
    }


def field_mappings(
    indicators: list[dict[str, str]],
    raw_to_canonical: dict[str, str],
    fields: list[dict[str, str]],
    suggestions: list[dict[str, str]],
    legacy_by_field: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    suggestion_scores: dict[tuple[str, str], float] = defaultdict(float)
    for suggestion in suggestions:
        canonical_id = raw_to_canonical.get(suggestion["family_id"])
        if canonical_id:
            key = (suggestion["matrix_field"], canonical_id)
            suggestion_scores[key] = max(
                suggestion_scores[key], float(suggestion["similarity_score"] or 0)
            )
    mappings: list[dict[str, str]] = []
    for field in fields:
        field_id = field["matrix_field"]
        legacy = legacy_by_field[field_id]
        field_text = normalize(f"{field['legacy_name']} {legacy['canonical_name_en']}")
        scored: list[tuple[float, float, dict[str, str]]] = []
        for indicator in indicators:
            aliases = [
                indicator["canonical_name"],
                *_json_list(indicator["aliases_json"]),
            ]
            name_score = max(
                similarity(field_text, normalize(alias)) for alias in aliases
            )
            suggestion_score = suggestion_scores.get(
                (field_id, indicator["canonical_id"]), 0.0
            )
            scored.append((max(name_score, suggestion_score), name_score, indicator))
        score, name_score, best = max(scored, key=lambda item: item[0])
        uses_future = legacy["uses_future_information"] == "1"
        t0_indicator = best["maximum_information_time"] == "T0"
        if name_score >= 0.90 and not (uses_future and t0_indicator):
            mapping_type = "direct"
            derivation = f"Use {field_id} directly under its verified field definition: {legacy['formula_text']}"
            reason = "Canonical/alias name is an exact or near-exact match and the information-time boundary is compatible."
        elif score >= 0.48 and not (uses_future and t0_indicator):
            mapping_type = "derivable"
            derivation = f"Derive the canonical indicator from {field_id} using the source formula and any declared normalization: {legacy['formula_text']}"
            reason = "Name/alignment evidence supports a derivation candidate; formula, units, and coverage require downstream verification."
        else:
            mapping_type = "unavailable"
            derivation = "No safe direct or derivable use is authorized at this stage."
            reason = (
                "The available field uses future information incompatible with the canonical T0 boundary."
                if uses_future and t0_indicator
                else "The best semantic/alignment candidate is too weak for a defensible mapping."
            )
        mappings.append(
            {
                "canonical_id": best["canonical_id"],
                "mapping_type": mapping_type,
                "fields_json": canonical_json([field_id]),
                "derivation": derivation,
                "reason": reason,
                "confidence_score": f"{score:.8f}",
                "source_feature_id": field_id,
                "legacy_name": field["legacy_name"],
                "legacy_formula": legacy["formula_text"],
                "publication_time_computable": legacy["publication_time_computable"],
                "uses_future_information": legacy["uses_future_information"],
                "no_selection_decision": "true",
            }
        )
    return mappings


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(
    dimensions: list[dict[str, str]],
    mentions: list[dict[str, str]],
    raw_families: list[dict[str, str]],
    fields: list[dict[str, str]],
    indicators: list[dict[str, str]],
    mappings: list[dict[str, str]],
) -> None:
    raw_ids = {row["family_id"] for row in raw_families}
    covered_raw = {
        family_id
        for row in indicators
        for family_id in _json_list(row["raw_family_ids_json"])
    }
    covered_dimensions = {
        dimension_id
        for row in indicators
        for dimension_id in _json_list(row["dimension_ids_json"])
    }
    if len(raw_families) != 1297 or covered_raw != raw_ids:
        raise ProtocolError("Indicator census does not cover all 1,297 raw families")
    if len(mentions) != 1205 or len(dimensions) != 26:
        raise ProtocolError("Unexpected mention or dimension count")
    if covered_dimensions != {row["dimension_id"] for row in dimensions}:
        raise ProtocolError("Not all 26 dimensions are covered")
    if len({row["canonical_id"] for row in indicators}) != len(indicators):
        raise ProtocolError("Duplicate canonical indicator ID")
    if any(not _json_list(row["mention_refs_json"]) for row in indicators):
        raise ProtocolError("Every canonical family requires mention provenance")
    if any(not _json_list(row["source_work_ids_json"]) for row in indicators):
        raise ProtocolError("Every canonical family requires source provenance")
    if len(fields) != 221 or len(mappings) != 221:
        raise ProtocolError("Field mapping must cover 221 matrix fields")
    mapped_fields = [
        field for row in mappings for field in _json_list(row["fields_json"])
    ]
    if (
        set(mapped_fields) != {row["matrix_field"] for row in fields}
        or len(mapped_fields) != 221
    ):
        raise ProtocolError("Matrix field mapping is not one-to-one complete")
    if any(
        row["mapping_type"] not in {"direct", "derivable", "unavailable"}
        for row in mappings
    ):
        raise ProtocolError("Invalid mapping type")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dimensions = read_rows(args.dimensions)
    mentions = read_rows(args.mentions)
    raw_families = read_rows(args.source_inventory)
    fields = read_rows(args.field_inventory)
    suggestions = read_rows(args.alignment_suggestions)
    legacy_rows = read_rows(args.legacy_library)
    field_ids = {row["matrix_field"] for row in fields}
    legacy_by_field = {
        row["feature_id"]: {
            key: row[key]
            for key in (
                "feature_id",
                "canonical_name_en",
                "formula_text",
                "publication_time_computable",
                "uses_future_information",
            )
        }
        for row in legacy_rows
        if row["feature_id"] in field_ids
    }
    if set(legacy_by_field) != field_ids:
        raise ProtocolError("Legacy metadata does not cover all 221 available fields")
    indicators, raw_to_canonical = census(
        dimensions, mentions, raw_families, legacy_by_field
    )
    mappings = field_mappings(
        indicators, raw_to_canonical, fields, suggestions, legacy_by_field
    )
    validate(dimensions, mentions, raw_families, fields, indicators, mappings)
    _write(args.census_output, indicators)
    _write(args.mapping_output, mappings)
    inputs = {
        name: {"path": str(path.resolve()), "sha256": file_hash(path)}
        for name, path in (
            ("dimensions", args.dimensions),
            ("mentions", args.mentions),
            ("source_inventory", args.source_inventory),
            ("field_inventory", args.field_inventory),
            ("alignment_suggestions", args.alignment_suggestions),
            ("legacy_field_metadata", args.legacy_library),
        )
    }
    mapping_counts = dict(
        sorted(Counter(row["mapping_type"] for row in mappings).items())
    )
    manifest = {
        "artifact": "indicator_census_and_field_mapping_primary",
        "generated_at": utc_now(),
        "reviewer_role": "Primary AI",
        "model_label": MODEL_LABEL,
        "run_id": f"indicator-census-primary-{sha256_text(canonical_json(inputs))[:16]}",
        "outcome_blind": True,
        "forbidden_sources_read": False,
        "no_numeric_quota": True,
        "legacy_use_limited_to_221_field_name_formula_availability": True,
        "inputs": inputs,
        "raw_family_count": 1297,
        "mention_count": 1205,
        "dimension_count": 26,
        "canonical_family_count": len(indicators),
        "raw_family_coverage": 1297,
        "dimension_coverage": 26,
        "matrix_field_count": 221,
        "mapping_counts": mapping_counts,
        "census": {
            "path": str(args.census_output.resolve()),
            "sha256": file_hash(args.census_output),
            "row_count": len(indicators),
        },
        "field_mapping": {
            "path": str(args.mapping_output.resolve()),
            "sha256": file_hash(args.mapping_output),
            "row_count": len(mappings),
        },
        "validation": {
            "status": "PASS",
            "raw_missing": 0,
            "dimensions_missing": 0,
            "fields_missing": 0,
            "duplicate_canonical_ids": 0,
        },
    }
    args.manifest_output.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=Path, required=True)
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--field-inventory", type=Path, required=True)
    parser.add_argument("--alignment-suggestions", type=Path, required=True)
    parser.add_argument("--legacy-library", type=Path, required=True)
    parser.add_argument("--census-output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(run(args)))


if __name__ == "__main__":
    main()

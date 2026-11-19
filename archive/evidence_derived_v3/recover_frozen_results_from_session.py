"""Rebuild the evidence-v3 frozen definition bundle from a Codex session log.

The original ignored SQLite/output directory may be unrecoverable after an
ext4 discard.  This recovery path extracts the exact feature inventory, exact
nested frozen set membership, exact candidate-dimension labels, and exact
Broad-T0 feature-to-dimension mapping that were printed in the session log.
Mappings for features outside Broad T0 were not printed in full and are
therefore reconstructed deterministically and labelled as reconstructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping


FEATURE_LINE = re.compile(
    r"^(EF\d{4})\t([^\t]+)\t([^\t]+)\t"
    r"article=([01])\tt0=([01])\tfuture=([01])\tft=([01])$"
)
FORMULA_LINE = re.compile(
    r"^(EF\d{4})\t([^\t]+)\t([^\t]+)\t"
    r"(T0|unverified_without_fulltext)\t(.*)$"
)
DIMENSION_LINE = re.compile(
    r"^\| (CD\d{3}) \| ([^|]+?) \| ([^|]+?) \| "
    r"([^|]+?) \| ([^|]+?) \| (.*?) \|$"
)
COVERED_LINE = re.compile(
    r"^(CD\d{3})\|([^|]+)\|([^|]+)\|\d+\|\d+\|(EF\d{4}(?:,EF\d{4})*)$"
)
SELECTED_NAMES_LINE = re.compile(
    r"^(CD\d{3}) \| ([^|]+) \| ([^|]+) \| n=\d+ \| (.+)$"
)
SET_MARKER_START = "## feature_sets.json"
SET_MARKER_END = "## input_snapshot.json"
EXPECTED_SETS = {
    "strict_7": (7, 4),
    "fulltext_16": (16, 10),
    "source_154": (154, 48),
    "ultrarelaxed_221": (221, 55),
}
GATES = (
    "G01_IN_SCOPE_ROLE",
    "G02_ARTICLE_LEVEL",
    "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE",
    "G04_REPRODUCIBLE_DEFINITION_AND_OPERATIONALIZATION",
    "G05_PUBLICATION_TIME",
    "G06_NO_FUTURE_INFORMATION",
    "G07_CURRENT_DATA_READY",
    "G08_BIAS_GUARDRAIL",
    "G09_NO_FATAL_VALIDITY_CONCERN",
    "G10_OUTCOME_BLIND_SELECTION",
    "G11_DATA_QUALITY_PASS",
    "G12_NONCONSTANT",
    "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE",
    "G14_INDEPENDENT_SECOND_REVIEW_APPROVAL",
)


def _output_text(payload: Mapping[str, Any]) -> str:
    output = payload.get("output", "")
    if isinstance(output, list):
        return "".join(
            str(item.get("text", ""))
            for item in output
            if isinstance(item, Mapping)
        )
    return str(output)


def _session_outputs(path: Path) -> list[str]:
    outputs: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload", {})
            if payload.get("type") in {
                "function_call_output",
                "custom_tool_call_output",
            }:
                outputs.append(_output_text(payload))
    return outputs


def _extract_feature_sets(outputs: Iterable[str]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for output in outputs:
        if SET_MARKER_START not in output or SET_MARKER_END not in output:
            continue
        fragment = output.split(SET_MARKER_START, 1)[1].split(
            SET_MARKER_END, 1
        )[0].strip()
        try:
            value = json.loads(fragment)
        except json.JSONDecodeError:
            # Later diagnostic commands may quote the marker names while
            # printing source code; only a directly parseable payload counts.
            continue
        if value not in matches:
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(f"expected one feature-set payload, found {len(matches)}")
    payload = matches[0]
    for name, (feature_count, dimension_count) in EXPECTED_SETS.items():
        item = payload["sets"][name]
        if len(set(item["feature_ids"])) != feature_count:
            raise ValueError(f"wrong recovered feature count for {name}")
        if len(set(item["dimension_ids"])) != dimension_count:
            raise ValueError(f"wrong recovered dimension count for {name}")
    return payload


def _extract_features(outputs: Iterable[str]) -> dict[str, dict[str, Any]]:
    candidates: list[dict[str, dict[str, Any]]] = []
    for output in outputs:
        rows: dict[str, dict[str, Any]] = {}
        for line in output.splitlines():
            match = FEATURE_LINE.match(line)
            if not match:
                continue
            feature_id, name, role, article, t0, future, fulltext = match.groups()
            rows[feature_id] = {
                "feature_id": feature_id,
                "canonical_name_en": name,
                "scope_role": role,
                "article_level": article == "1",
                "publication_time": t0 == "1",
                "uses_future_information": future == "1",
                "english_fulltext_verified": fulltext == "1",
            }
        if len(rows) == 432:
            candidates.append(rows)
    if not candidates:
        raise ValueError("the complete 432-feature inventory was not found")
    first = candidates[0]
    if any(candidate != first for candidate in candidates[1:]):
        raise ValueError("conflicting complete feature inventories in session")
    return first


def _extract_formulas(outputs: Iterable[str]) -> dict[str, dict[str, str]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for output in outputs:
        for line in output.splitlines():
            match = FORMULA_LINE.match(line)
            if not match:
                continue
            feature_id, name, role, boundary, formula = match.groups()
            candidates[feature_id].append(
                {
                    "canonical_name_en": name,
                    "scope_role": role,
                    "maximum_information_time": boundary,
                    "formula_text": formula,
                }
            )
    if not candidates:
        return {}
    placeholders = {
        "",
        "Not explicitly reported for this mention in the verified evidence.",
        "Not available without verified full text.",
    }
    return {
        feature_id: max(
            rows,
            key=lambda row: (
                row["formula_text"] not in placeholders,
                len(row["formula_text"]),
            ),
        )
        for feature_id, rows in candidates.items()
    }


def _extract_dimensions(outputs: Iterable[str]) -> dict[str, dict[str, str]]:
    candidates: list[dict[str, dict[str, str]]] = []
    for output in outputs:
        rows: dict[str, dict[str, str]] = {}
        for line in output.splitlines():
            match = DIMENSION_LINE.match(line)
            if not match:
                continue
            dimension_id, label, role, status, selected, reason = (
                value.strip() for value in match.groups()
            )
            rows[dimension_id] = {
                "dimension_id": dimension_id,
                "label": label,
                "reported_role": role,
                "reported_status": status,
                "reported_selected": selected,
                "reported_reason": reason,
            }
        if len(rows) == 66:
            candidates.append(rows)
    if not candidates:
        raise ValueError("the complete 66-dimension table was not found")
    return candidates[-1]


def _extract_broad_mapping(
    outputs: Iterable[str],
) -> tuple[dict[str, str], dict[str, str]]:
    candidates: list[tuple[dict[str, str], dict[str, str]]] = []
    for output in outputs:
        mapping: dict[str, str] = {}
        roles: dict[str, str] = {}
        for line in output.splitlines():
            match = COVERED_LINE.match(line)
            if match:
                dimension_id, _label, role, feature_list = match.groups()
                feature_ids = feature_list.split(",")
            else:
                selected_match = SELECTED_NAMES_LINE.match(line)
                if not selected_match:
                    continue
                dimension_id, role, _label, selected_text = selected_match.groups()
                feature_ids = re.findall(r"EF\d{4}", selected_text)
            roles[dimension_id] = role
            for feature_id in feature_ids:
                mapping[feature_id] = dimension_id
        if len(mapping) == 154:
            candidates.append((mapping, roles))
        if len(mapping) == 221:
            candidates.append((mapping, roles))
    broad = [item for item in candidates if len(item[0]) == 221]
    if not broad:
        raise ValueError("the exact 221-feature Broad-T0 mapping was not found")
    return broad[-1]


def _tokens(value: str) -> set[str]:
    stop = {"and", "or", "of", "the", "context", "potential"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in stop
    }


def _similarity(name: str, label: str, examples: Iterable[str]) -> float:
    name_tokens = _tokens(name)
    label_tokens = _tokens(label)
    overlap = len(name_tokens & label_tokens) / max(1, len(name_tokens | label_tokens))
    sequence = SequenceMatcher(None, name.casefold(), label.casefold()).ratio()
    example_score = max(
        (SequenceMatcher(None, name.casefold(), item.casefold()).ratio() for item in examples),
        default=0.0,
    )
    return 3.0 * overlap + sequence + 1.5 * example_score


def _complete_mapping(
    features: Mapping[str, Mapping[str, Any]],
    dimensions: Mapping[str, Mapping[str, str]],
    exact_mapping: Mapping[str, str],
) -> tuple[dict[str, str], set[str]]:
    mapping = dict(exact_mapping)
    reconstructed: set[str] = set()
    examples: dict[str, list[str]] = defaultdict(list)
    for feature_id, dimension_id in mapping.items():
        examples[dimension_id].append(str(features[feature_id]["canonical_name_en"]))
    remaining = [feature_id for feature_id in sorted(features) if feature_id not in mapping]
    empty_dimensions = [dimension_id for dimension_id in dimensions if dimension_id not in examples]
    for dimension_id in empty_dimensions:
        if not remaining:
            raise ValueError("not enough excluded features to seed all dimensions")
        best = max(
            remaining,
            key=lambda feature_id: _similarity(
                str(features[feature_id]["canonical_name_en"]),
                str(dimensions[dimension_id]["label"]),
                (),
            ),
        )
        mapping[best] = dimension_id
        reconstructed.add(best)
        examples[dimension_id].append(str(features[best]["canonical_name_en"]))
        remaining.remove(best)
    for feature_id in remaining:
        name = str(features[feature_id]["canonical_name_en"])
        dimension_id = max(
            dimensions,
            key=lambda item: _similarity(
                name,
                str(dimensions[item]["label"]),
                examples[item],
            ),
        )
        mapping[feature_id] = dimension_id
        reconstructed.add(feature_id)
        examples[dimension_id].append(name)
    return mapping, reconstructed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _dimension_role(dimension_id: str, reported: str, exact: Mapping[str, str]) -> str:
    if dimension_id in exact:
        return exact[dimension_id]
    return {
        "predictive": "t0_potential",
        "opportunity": "opportunity",
        "control": "context_control",
        "sensitivity": "sensitivity",
    }.get(reported, reported)


def recover(session_logs: list[Path], output_dir: Path) -> dict[str, Any]:
    outputs = [
        output
        for session_log in session_logs
        for output in _session_outputs(session_log)
    ]
    feature_sets = _extract_feature_sets(outputs)
    features = _extract_features(outputs)
    formulas = _extract_formulas(outputs)
    dimensions = _extract_dimensions(outputs)
    exact_mapping, exact_dimension_roles = _extract_broad_mapping(outputs)
    broad_ids = set(feature_sets["sets"]["ultrarelaxed_221"]["feature_ids"])
    if set(exact_mapping) != broad_ids:
        raise ValueError("session Broad-T0 mapping disagrees with frozen membership")
    mapping, reconstructed_ids = _complete_mapping(features, dimensions, exact_mapping)

    strict = set(feature_sets["sets"]["strict_7"]["feature_ids"])
    fulltext = set(feature_sets["sets"]["fulltext_16"]["feature_ids"])
    primary = set(feature_sets["sets"]["source_154"]["feature_ids"])
    library_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for feature_id in sorted(features):
        feature = features[feature_id]
        formula = formulas.get(feature_id, {})
        library_rows.append(
            {
                "feature_id": feature_id,
                "canonical_name_en": feature["canonical_name_en"],
                "scope_role": feature["scope_role"],
                "maximum_information_time": formula.get(
                    "maximum_information_time",
                    "T0" if feature["publication_time"] else "unverified_without_fulltext",
                ),
                "formula_text": formula.get("formula_text", ""),
                "article_level": int(feature["article_level"]),
                "publication_time_computable": int(feature["publication_time"]),
                "uses_future_information": int(feature["uses_future_information"]),
                "english_fulltext_verified": int(feature_id in fulltext),
                "dimension_id": mapping[feature_id],
                "dimension_mapping_provenance": (
                    "session_exact_broad_t0"
                    if feature_id in exact_mapping
                    else "deterministic_reconstruction_excluded_from_broad_t0"
                ),
                "recovery_provenance": "codex_session_log_structured_output",
            }
        )
        checks = {
            "G01_IN_SCOPE_ROLE": True,
            "G02_ARTICLE_LEVEL": True,
            "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE": feature_id in primary,
            "G04_REPRODUCIBLE_DEFINITION_AND_OPERATIONALIZATION": feature_id in strict,
            # The later frozen Broad-T0 membership superseded the earlier
            # inventory flags for a few targeted operationalizations.
            "G05_PUBLICATION_TIME": feature_id in broad_ids,
            "G06_NO_FUTURE_INFORMATION": True,
            "G07_CURRENT_DATA_READY": feature_id in strict,
            "G08_BIAS_GUARDRAIL": True,
            "G09_NO_FATAL_VALIDITY_CONCERN": True,
            "G10_OUTCOME_BLIND_SELECTION": True,
            "G11_DATA_QUALITY_PASS": feature_id in strict,
            "G12_NONCONSTANT": feature_id in strict,
            "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE": feature_id in fulltext,
            "G14_INDEPENDENT_SECOND_REVIEW_APPROVAL": feature_id in fulltext,
        }
        failed = [gate for gate in GATES if not checks[gate]]
        gate_rows.append(
            {
                "feature_id": feature_id,
                "gate_checks_json": json.dumps(checks, sort_keys=True),
                "failed_gates_json": json.dumps(failed),
                "final_role": feature["scope_role"] if not failed else "excluded",
                "decision_reason": "PASS" if not failed else "RECOVERED_GATE_FAILURE",
                "gate_recovery_quality": (
                    "exact_for_frozen_set_membership_not_original_per_gate_bytes"
                ),
                "recovery_provenance": "codex_session_log_nested_membership",
            }
        )

    features_by_dimension: dict[str, list[str]] = defaultdict(list)
    for feature_id, dimension_id in mapping.items():
        features_by_dimension[dimension_id].append(feature_id)
    dimension_rows: list[dict[str, Any]] = []
    for dimension_id in sorted(dimensions):
        item = dimensions[dimension_id]
        feature_ids = sorted(features_by_dimension[dimension_id])
        dimension_rows.append(
            {
                "dimension_id": dimension_id,
                "label": item["label"],
                "definition": "Recovered label; original full definition was not printed.",
                "construct_role": _dimension_role(
                    dimension_id,
                    item["reported_role"],
                    exact_dimension_roles,
                ),
                "feature_ids_json": json.dumps(feature_ids),
                "reported_status": item["reported_status"],
                "reported_selected": item["reported_selected"],
                "reported_reason": item["reported_reason"],
                "exact_mapped_feature_count": sum(
                    feature_id not in reconstructed_ids for feature_id in feature_ids
                ),
                "reconstructed_mapped_feature_count": sum(
                    feature_id in reconstructed_ids for feature_id in feature_ids
                ),
                "recovery_provenance": "session_exact_label_mixed_mapping",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "complete_indicator_library_v3.csv": output_dir
        / "complete_indicator_library_v3.csv",
        "feature_gate_decisions_v3.csv": output_dir / "feature_gate_decisions_v3.csv",
        "candidate_dimensions_v3.csv": output_dir / "candidate_dimensions_v3.csv",
        "feature_sets_recovered_v3.json": output_dir / "feature_sets_recovered_v3.json",
    }
    _write_csv(paths["complete_indicator_library_v3.csv"], library_rows)
    _write_csv(paths["feature_gate_decisions_v3.csv"], gate_rows)
    _write_csv(paths["candidate_dimensions_v3.csv"], dimension_rows)
    paths["feature_sets_recovered_v3.json"].write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "RECOVERED_WITH_DISCLOSED_LIMITATION",
        "session_logs": [str(path.resolve()) for path in session_logs],
        "counts": {
            "indicator_families": len(library_rows),
            "candidate_dimensions": len(dimension_rows),
            "exact_broad_t0_dimension_mappings": len(exact_mapping),
            "reconstructed_excluded_dimension_mappings": len(reconstructed_ids),
            "strict_features": len(strict),
            "fulltext_features": len(fulltext),
            "primary_source_features": len(primary),
            "broad_t0_features": len(broad_ids),
        },
        "exactly_recovered": [
            "432 feature IDs, English labels, scope roles, article/T0/future flags",
            "66 candidate dimension IDs and labels",
            "nested 7/16/154/221 feature memberships",
            "nested 4/10/48/55 dimension memberships",
            "221 Broad-T0 feature-to-dimension mappings",
        ],
        "not_original_bytes": [
            "the original SQLite database and source-location evidence fields",
            "the original full 14-gate row values outside membership-defining gates",
            "211 feature-to-dimension mappings excluded from Broad T0",
            "candidate dimension full definitions",
        ],
        "files": {
            name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    report_path = output_dir / "recovery_manifest_v3.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README_RECOVERY_V3.md").write_text(
        "# Evidence-v3 session recovery\n\n"
        "Status: **RECOVERED_WITH_DISCLOSED_LIMITATION**\n\n"
        "This directory is a usable replacement definition bundle, but it is "
        "not a byte-for-byte copy of the deleted ignored SQLite/output tree.\n\n"
        "## Exact session-log recovery\n\n"
        "- 432 feature IDs, English labels, roles, and inventory flags.\n"
        "- 66 candidate-dimension IDs and labels.\n"
        "- Exact nested feature memberships: 7 / 16 / 154 / 221.\n"
        "- Exact nested dimension memberships: 4 / 10 / 48 / 55.\n"
        "- Exact feature-to-dimension mapping for all 221 Broad-T0 features.\n\n"
        "## Deterministically reconstructed fields\n\n"
        "- The 211 features excluded from Broad T0 were assigned to candidate "
        "dimensions from their English names and recovered dimension labels.\n"
        "- Per-gate rows reproduce the four frozen-set memberships, but do not "
        "claim the original deleted CSV bytes or every original diagnostic "
        "field.\n"
        "- Original full source-location evidence and the SQLite retrieval "
        "tables are not restored by this bundle.\n\n"
        "Use `recovery_preflight_v3.json` and `recovery_manifest_v3.json` for "
        "machine-readable verification and provenance.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-log", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = recover(args.session_log, args.output_dir)
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

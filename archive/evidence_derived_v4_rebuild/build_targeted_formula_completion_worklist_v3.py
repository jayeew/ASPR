from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from common import (
    DATABASE_PATH,
    OUTPUT_DIR,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)
from database import initialize


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGETS = ROOT / "targeted_formula_completion_targets_v3.json"
DEFAULT_INVENTORY = OUTPUT_DIR / "local_t0_input_inventory_v3.json"
DEFAULT_SUPPLEMENT_REPORT = (
    OUTPUT_DIR / "targeted_formula_supplement_acquisition_v3.json"
)
DEFAULT_OUTPUT = (
    OUTPUT_DIR
    / "human_tasks"
    / "formal_terminal_targeted_formula_H1_BLIND_v3.csv"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".manifest.json")

PROTECTED_FIELDS = (
    "target_id",
    "feature_id",
    "canonical_name_en",
    "record_key",
    "doi",
    "source_title",
    "source_scope",
    "fulltext_source_url",
    "fulltext_local_path",
    "fulltext_sha256",
    "fulltext_license",
    "original_h1_source_disposition",
    "original_h1_fulltext_status",
    "original_h1_source_notes",
    "local_source_ids_json",
    "local_columns_json",
    "verification_question",
    "target_definition_sha256",
    "local_inventory_sha256",
)
REVIEW_FIELDS = (
    "reviewer_role",
    "decision",
    "raw_name_en",
    "label_zh",
    "research_group",
    "research_group_id",
    "research_group_evidence",
    "source_role",
    "formula_location",
    "evidence_span",
    "formula",
    "units",
    "parameters",
    "direction",
    "source_reported_missing_rule",
    "required_data_json",
    "maximum_information_time",
    "scope_role",
    "validation_summary",
    "evidence_direction",
    "negative_evidence",
    "article_level",
    "primary_or_foundational_evidence",
    "formula_reproducible",
    "t0_computable",
    "requires_future",
    "fatal_validity_concern",
    "evidence_strength",
    "stability_score",
    "stability_basis",
    "selection_priority",
    "redundancy_family",
    "review_reason",
    "reviewer_provenance",
)
FIELDS = (*PROTECTED_FIELDS, *REVIEW_FIELDS)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _inventory_column_set(payload: Mapping[str, Any]) -> set[str]:
    columns: set[str] = set()
    for source_id, source in payload["sources"].items():
        for column in source["columns"]:
            columns.add(f"{source_id}:{column['name']}")
    return columns


def _formal_fulltext(
    connection: sqlite3.Connection,
    record_key: str,
) -> Dict[str, str]:
    row = connection.execute(
        """
        SELECT final_url, candidate_url, local_path, sha256,
               access_statement
        FROM fulltext_acquisitions
        WHERE record_key = ? AND status = 'downloaded'
        """,
        (record_key,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Formal target lacks frozen full text: {record_key}")
    path = Path(str(row["local_path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(row["sha256"]):
        raise ValueError(f"Formal target full-text hash mismatch: {record_key}")
    return {
        "fulltext_source_url": str(
            row["final_url"] or row["candidate_url"]
        ),
        "fulltext_local_path": str(path),
        "fulltext_sha256": str(row["sha256"]),
        "fulltext_license": str(row["access_statement"]),
    }


def _supplement_lookup(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(source["record_key"]): dict(source)
        for source in report["sources"]
    }


def _supplement_fulltext(
    supplement: Mapping[str, Any],
) -> Dict[str, str]:
    path = Path(str(supplement["local_path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(supplement["sha256"]):
        raise ValueError(
            "Formula-supplement full-text hash mismatch: "
            f"{supplement['record_key']}"
        )
    return {
        "fulltext_source_url": str(supplement["candidate_url"]),
        "fulltext_local_path": str(path),
        "fulltext_sha256": str(supplement["sha256"]),
        "fulltext_license": str(supplement["access_statement"]),
    }


def _h1_source_payload(
    connection: sqlite3.Connection,
    record_key: str,
    source_scope: str,
) -> Dict[str, str]:
    if source_scope == "cited_formula_and_validation_supplement":
        return {
            "original_h1_source_disposition": (
                "not_applicable_targeted_formula_supplement"
            ),
            "original_h1_fulltext_status": (
                "english_fulltext_pending_independent_formula_review"
            ),
            "original_h1_source_notes": (
                "The source is outside the formal terminal cohort and is "
                "used only as a cited formula/validation supplement."
            ),
        }
    row = connection.execute(
        """
        SELECT disposition, english_fulltext_status, notes
        FROM indicator_source_reviews
        WHERE record_key = ? AND reviewer_role = 'H1'
        """,
        (record_key,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Formal target lacks H1 source review: {record_key}")
    return {
        "original_h1_source_disposition": str(row["disposition"]),
        "original_h1_fulltext_status": str(
            row["english_fulltext_status"]
        ),
        "original_h1_source_notes": str(row["notes"]),
    }


def build_rows(
    connection: sqlite3.Connection,
    targets: Mapping[str, Any],
    inventory: Mapping[str, Any],
    supplement_report: Mapping[str, Any],
    target_definition_sha256: str,
    inventory_sha256: str,
) -> List[Dict[str, str]]:
    """Build a formula-free H1 worksheet with protected provenance."""
    known_columns = _inventory_column_set(inventory)
    supplements = _supplement_lookup(supplement_report)
    rows: List[Dict[str, str]] = []
    seen_targets: set[str] = set()
    for target in targets["targets"]:
        target_id = str(target["target_id"])
        if target_id in seen_targets:
            raise ValueError(f"Duplicate formula target: {target_id}")
        seen_targets.add(target_id)
        feature = connection.execute(
            """
            SELECT feature_id, canonical_name_en
            FROM indicator_families WHERE feature_id = ?
            """,
            (str(target["feature_id"]),),
        ).fetchone()
        if feature is None:
            raise ValueError(f"Unknown feature target: {target['feature_id']}")
        record = connection.execute(
            """
            SELECT record_key, doi, title, language
            FROM records WHERE record_key = ?
            """,
            (str(target["record_key"]),),
        ).fetchone()
        if record is None or str(record["language"]).casefold() != "en":
            raise ValueError(f"Target source is not English: {target_id}")
        source_scope = str(target["source_scope"])
        if source_scope == "formal_included_source":
            included = connection.execute(
                """
                SELECT 1 FROM screening_final
                WHERE record_key = ? AND final_decision = 'include'
                  AND final_language = 'en'
                """,
                (record["record_key"],),
            ).fetchone()
            if included is None:
                raise ValueError(f"Formal target is not included: {target_id}")
            fulltext = _formal_fulltext(
                connection,
                str(record["record_key"]),
            )
        elif source_scope == "cited_formula_and_validation_supplement":
            supplement = supplements.get(str(record["record_key"]))
            if supplement is None:
                raise ValueError(
                    f"Missing acquired formula supplement: {target_id}"
                )
            fulltext = _supplement_fulltext(supplement)
        else:
            raise ValueError(f"Unknown target source scope: {source_scope}")
        local_columns = [str(value) for value in target["local_columns"]]
        unknown = sorted(set(local_columns) - known_columns)
        if unknown:
            raise ValueError(
                f"Formula target references unknown local columns: {unknown}"
            )
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "target_id": target_id,
                "feature_id": str(feature["feature_id"]),
                "canonical_name_en": str(feature["canonical_name_en"]),
                "record_key": str(record["record_key"]),
                "doi": str(record["doi"]),
                "source_title": str(record["title"]),
                "source_scope": source_scope,
                **fulltext,
                **_h1_source_payload(
                    connection,
                    str(record["record_key"]),
                    source_scope,
                ),
                "local_source_ids_json": json.dumps(
                    target["local_source_ids"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "local_columns_json": json.dumps(
                    local_columns,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "verification_question": str(
                    target["verification_question"]
                ),
                "target_definition_sha256": target_definition_sha256,
                "local_inventory_sha256": inventory_sha256,
                "reviewer_role": "H1",
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    """Export the independent H1 targeted-formula worklist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--supplement-report",
        type=Path,
        default=DEFAULT_SUPPLEMENT_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    target_path = args.targets.resolve()
    inventory_path = args.inventory.resolve()
    supplement_path = args.supplement_report.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    targets = _read_json(target_path)
    inventory = _read_json(inventory_path)
    supplements = _read_json(supplement_path)
    connection = initialize(args.database.resolve())
    try:
        rows = build_rows(
            connection,
            targets,
            inventory,
            supplements,
            sha256_file(target_path),
            sha256_file(inventory_path),
        )
    finally:
        connection.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rows, FIELDS)
    manifest = {
        "schema_version": "targeted_formula_h1_worklist_manifest_v3",
        "created_at": utc_now(),
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "item_count": len(rows),
        "protected_fields": list(PROTECTED_FIELDS),
        "review_fields": list(REVIEW_FIELDS),
        "targets_path": str(target_path),
        "targets_sha256": sha256_file(target_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "supplement_report_path": str(supplement_path),
        "supplement_report_sha256": sha256_file(supplement_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "target_order_sha256": sha256_bytes(
            "\n".join(row["target_id"] for row in rows).encode("utf-8")
        ),
        "target_count_is_not_a_selection_quota": True,
        "creates_new_indicator_families": False,
        "round_13": False,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

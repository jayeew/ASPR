"""Validate the v3-comparator coverage floor before a v4 final freeze.

The old v3 tables are a discovery benchmark only.  This validator deliberately
does not authorise formulas or features: it makes a premature smaller v4 census
impossible to freeze and leaves a machine-readable explanation of every gap.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, ROOT, sha256_file, write_json

POLICY_PATH = ROOT / "protocol_amendment_v3_coverage_floor_v4.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v3_coverage_floor_status_v4.json"
MATERIAL_DISPOSITIONS = {
    "recovered_v4_family",
    "documented_merge",
    "h2_scope_exclusion",
}


def _anchor_count(policy: dict[str, Any]) -> int:
    """Return the declared historical discovery-label count."""
    return int(policy["historical_comparator"]["indicator_family_count"])


def _dimensions_count(policy: dict[str, Any]) -> int:
    """Return the declared historical dimension-label count."""
    return int(policy["historical_comparator"]["candidate_dimension_count"])


def _evidence_backed_family_count(connection: sqlite3.Connection) -> int:
    """Count v4 families with a source-level formula/application record."""
    return int(connection.execute("""
            SELECT COUNT(DISTINCT family.feature_id)
            FROM indicator_families AS family
            JOIN indicator_mentions AS mention
              ON mention.canonical_name_en = family.canonical_name_en
            JOIN indicator_source_reviews AS source_review
              ON source_review.record_key = mention.record_key
            WHERE source_review.disposition = 'extracted'
            """).fetchone()[0])


def validate(connection: sqlite3.Connection, policy: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic coverage-floor status without mutating the DB."""
    comparator_families = _anchor_count(policy)
    comparator_dimensions = _dimensions_count(policy)
    reconciliation = Counter(
        str(row[0])
        for row in connection.execute(
            "SELECT coverage_disposition FROM v3_coverage_reconciliation"
        )
    )
    total_reconciliation = sum(reconciliation.values())
    material = sum(reconciliation[name] for name in MATERIAL_DISPOSITIONS)
    unresolved = total_reconciliation - material
    family_total = int(
        connection.execute("SELECT COUNT(*) FROM indicator_families").fetchone()[0]
    )
    dimension_total = int(
        connection.execute("SELECT COUNT(*) FROM candidate_dimensions").fetchone()[0]
    )
    result = {
        "schema_version": "v3_coverage_floor_status_v4",
        "policy_path": str(POLICY_PATH.resolve()),
        "policy_sha256": sha256_file(POLICY_PATH),
        "comparator": {
            "historical_discovery_label_count": comparator_families,
            "historical_dimension_label_count": comparator_dimensions,
        },
        "current_v4": {
            "all_indicator_families": family_total,
            "evidence_backed_formula_or_application_families": _evidence_backed_family_count(
                connection
            ),
            "candidate_dimensions": dimension_total,
            "reconciliation_rows": total_reconciliation,
            "reconciliation_by_disposition": dict(sorted(reconciliation.items())),
            "unresolved_reconciliation_rows": unresolved,
            "material_reconciliation_rows": material,
        },
        "checks": {
            "all_comparator_rows_reconciled": total_reconciliation
            == comparator_families
            and material == comparator_families,
            "canonical_counts_are_not_quotas": True,
        },
    }
    result["passed"] = bool(result["checks"]["all_comparator_rows_reconciled"])
    result["freeze_allowed"] = result["passed"]
    return result


def main() -> None:
    """Run the read-only coverage check and write its JSON status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    connection = sqlite3.connect(args.database.resolve())
    try:
        result = validate(connection, policy)
    finally:
        connection.close()
    write_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

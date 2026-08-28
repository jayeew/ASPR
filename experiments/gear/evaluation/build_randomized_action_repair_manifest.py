"""Build a no-outcome repair manifest for invalid randomized action executions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_EXCLUSION_PREFIXES = (
    "assigned_action_not_executed:",
    "semantic_verification_unavailable",
    "incomplete_run_artifacts",
    "resource_caps_exceeded:",
    "randomized_action_not_executed",
)


def build_repair_manifest(
    manifest_path: Path,
    audit_path: Path,
    output_path: Path,
    *,
    include_incomplete: bool = False,
) -> dict[str, Any]:
    """Select reruns solely from protocol-validity exclusions, never outcomes."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not cases or manifest.get("randomization_precedes_outcomes") is not True:
        raise ValueError("invalid randomized action source manifest")
    audit = pd.read_parquet(audit_path)
    required = {"case_id", "included", "exclusion_reason"}
    if missing := required - set(audit):
        raise ValueError(f"action audit missing columns: {sorted(missing)}")
    if len(audit) != len(cases) or audit["case_id"].astype(str).duplicated().any():
        raise ValueError("action audit does not match source manifest cardinality")
    excluded = audit[~audit["included"].astype(bool)].copy()
    excluded["exclusion_reason"] = excluded["exclusion_reason"].fillna("").astype(str)
    if not include_incomplete:
        excluded = excluded[
            excluded["exclusion_reason"].ne("incomplete_run_artifacts")
        ]
    invalid_reasons = sorted(
        {
            reason
            for reason in excluded["exclusion_reason"]
            if not reason.startswith(ALLOWED_EXCLUSION_PREFIXES)
        }
    )
    if invalid_reasons:
        raise ValueError(f"outcome-derived or unknown exclusions: {invalid_reasons}")
    selected_ids = set(excluded["case_id"].astype(str))
    selected = [case for case in cases if str(case["case_id"]) in selected_ids]
    if len(selected) != len(selected_ids):
        raise ValueError("action audit contains cases outside source manifest")
    result = {
        "contract": "gear_randomized_graph_action_repair_manifest_v1",
        "randomization_precedes_outcomes": True,
        "randomization_seed": manifest.get("randomization_seed"),
        "propensity": manifest.get("propensity"),
        "matched_budget": manifest.get("matched_budget"),
        "repair_selection_uses_outcomes": False,
        "repair_selection_rule": "protocol_execution_validity_only",
        "source_manifest_sha256": _sha256(manifest_path),
        "source_action_audit_sha256": _sha256(audit_path),
        "excluded_incomplete_runs": not include_incomplete,
        "cases": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()
    result = build_repair_manifest(
        args.manifest,
        args.audit,
        args.output,
        include_incomplete=args.include_incomplete,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_repair_manifest"]

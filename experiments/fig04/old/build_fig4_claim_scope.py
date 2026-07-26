#!/usr/bin/env python3
"""Build Fig. 4 claim-scope demotion artifacts from current quality gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import write_json, write_run_manifest


DEFAULT_FIG4_DIR = PROJECT_ROOT / "outputs" / "fig04/old"
BLOCKING_EXTERNAL_VALIDATION_CHECKS = {
    "fig3_reference_tier_range_present",
    "fig3_peer_novelty_positive",
    "fig3_peer_significance_positive",
}


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def quality_checks(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Return checks from common Fig. 4 quality-report shapes."""
    gates = report.get("quality_gates") if isinstance(report.get("quality_gates"), Mapping) else report
    checks = gates.get("checks") if isinstance(gates.get("checks"), Mapping) else gates
    return dict(checks or {})


def failed_checks(checks: Mapping[str, Any]) -> Sequence[str]:
    """Return failed quality-check names."""
    return [str(key) for key, value in checks.items() if value in (0, False, "fail", "failed")]


def build_claim_scope_decision(fig4_dir: Path) -> Dict[str, Any]:
    """Build a Fig. 4 claim-scope decision without changing the raw quality report."""
    report_path = fig4_dir / "figure_quality_report.json"
    if not report_path.exists():
        return {
            "figure": "Fig.4",
            "source_quality_report": str(report_path),
            "quality_report_exists": 0,
            "strong_validation_ready": 0,
            "claim_scope_gate_pass": 0,
            "main_or_extended_data": "blocked",
            "claim_scope_action": "missing_quality_report",
            "allowed_claim": "",
            "forbidden_claim": "Do not make Fig.4 peer-review validation claims until figure_quality_report.json exists.",
            "required_action": "Rebuild Fig.4 quality report.",
            "failed_checks": "quality_report_missing",
        }
    report = read_json(report_path)
    checks = quality_checks(report)
    failed = list(failed_checks(checks))
    strong_ready = bool(report.get("overall_pass"))
    external_blockers = [item for item in failed if item in BLOCKING_EXTERNAL_VALIDATION_CHECKS]
    can_demote = (not strong_ready) and bool(external_blockers)
    if strong_ready:
        scope = {
            "main_or_extended_data": "main_validation",
            "claim_scope_action": "retain_main_validation_claim",
            "allowed_claim": "Fig.4 supports peer-review alignment validation under the completed quality gates.",
            "forbidden_claim": "Do not claim reviewer replacement or causal proof beyond the validated peer-review alignment task.",
            "required_action": "Keep quality report, metrics summary, and fixed-sample manifest frozen.",
            "claim_scope_gate_pass": 1,
        }
    elif can_demote:
        scope = {
            "main_or_extended_data": "extended",
            "claim_scope_action": "demote_to_range_restricted_peer_review_audit",
            "allowed_claim": (
                "Fig.4 is a range-restricted peer-review alignment atlas among accepted high-tier Nature Portfolio papers; "
                "it can show semantic coverage, missing peer points, and audit failure modes."
            ),
            "forbidden_claim": (
                "Do not claim global external validation, Fig.3-score novelty/significance alignment, peer-review equivalence, "
                "or reviewer replacement while low/middle Fig.3 tiers and positive peer novelty/significance gates are absent."
            ),
            "required_action": (
                "Keep Fig.4 in Extended Data or diagnostic role; promote only after tier-balanced blinded labels and positive "
                "novelty/significance alignment gates pass."
            ),
            "claim_scope_gate_pass": 1,
        }
    else:
        scope = {
            "main_or_extended_data": "blocked",
            "claim_scope_action": "unresolved_quality_failure",
            "allowed_claim": "Fig.4 claim scope is unresolved.",
            "forbidden_claim": "Do not use Fig.4 as evidence until failed checks are fixed or explicitly demoted.",
            "required_action": "Inspect failed checks and decide whether to fix, demote, or move to supplement.",
            "claim_scope_gate_pass": 0,
        }
    return {
        "figure": "Fig.4",
        "source_quality_report": str(report_path),
        "quality_report_exists": 1,
        "quality_status_label": report.get("status_label", ""),
        "strong_validation_ready": int(strong_ready),
        "failed_checks": ";".join(failed),
        "external_validation_blockers": ";".join(external_blockers),
        **scope,
    }


def write_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    """Write one decision row to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_outputs(fig4_dir: Path, decision: Mapping[str, Any]) -> None:
    """Write Fig. 4 claim-scope outputs."""
    write_json(fig4_dir / "fig4_claim_scope_decision.json", decision)
    write_csv_row(fig4_dir / "fig4_claim_scope_decision.csv", decision)
    write_run_manifest(
        fig4_dir,
        figure="fig4_claim_scope",
        argv=sys.argv,
        inputs={"quality_report": str(fig4_dir / "figure_quality_report.json")},
        quality_gates={
            "overall_pass": bool(decision.get("claim_scope_gate_pass")),
            "status_label": decision.get("claim_scope_action"),
            "checks": {"claim_scope_gate_pass": int(decision.get("claim_scope_gate_pass", 0))},
        },
        extra=dict(decision),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build Fig. 4 claim-scope decision artifacts.")
    parser.add_argument("--fig4-dir", type=Path, default=DEFAULT_FIG4_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    decision = build_claim_scope_decision(args.fig4_dir)
    write_outputs(args.fig4_dir, decision)
    print(f"[fig4-claim-scope] wrote {args.fig4_dir / 'fig4_claim_scope_decision.json'}")
    print(f"[fig4-claim-scope] action={decision['claim_scope_action']}")


if __name__ == "__main__":
    main()

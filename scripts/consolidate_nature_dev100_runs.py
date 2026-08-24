#!/usr/bin/env python3
"""Select valid dev100 runs and emit evaluation/retry manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gear.evaluation.artifacts import load_review_bundle
from experiments.gear.evaluation.contracts import EvaluationManifestV1
from gear.trace import EvidenceStore


def _valid_complete(run_dir: Path, paper_id: str) -> tuple[bool, list[str]]:
    try:
        bundle = load_review_bundle(run_dir)
    except (OSError, ValueError) as exc:
        return False, [f"bundle_invalid:{type(exc).__name__}"]
    reasons: list[str] = []
    if bundle.paper_ir.paper_id != paper_id:
        reasons.append("paper_id_mismatch")
    if bundle.status.value != "complete":
        reasons.append(f"status_{bundle.status.value}")
    if not bundle.verification.passed:
        reasons.append("verification_not_passed")
    reasons.extend(EvidenceStore(run_dir).validate_manifest())
    return not reasons, reasons


def _select_runs(
    manifest: EvaluationManifestV1, roots: list[Path]
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    selected: dict[str, Path] = {}
    rejected: dict[str, list[str]] = {}
    for case in manifest.cases:
        for root in roots:
            candidate = root / case.case_id
            if not (candidate / "review_bundle.json").is_file():
                continue
            valid, reasons = _valid_complete(candidate, case.paper_id)
            if valid:
                selected[case.case_id] = candidate.resolve()
                break
            rejected[f"{case.case_id}@{root.name}"] = reasons
    return selected, rejected


def _evaluation_payload(
    source: EvaluationManifestV1, selected: dict[str, Path]
) -> dict[str, Any]:
    payload = source.model_dump(mode="json")
    for case in payload["cases"]:
        case["clean_run_dir"] = str(selected[case["case_id"]])
    return payload


def _retry_payload(
    source: EvaluationManifestV1, selected: dict[str, Path]
) -> dict[str, Any]:
    cases = []
    for case in source.cases:
        if case.case_id in selected:
            continue
        cases.append(
            {
                "case_id": case.case_id,
                "paper_path": str(case.manuscript_path),
                "paper_id": case.paper_id,
                "metadata": json.loads(case.metadata_path.read_text()),
                "cutoff": case.cutoff_date.isoformat(),
            }
        )
    return {"cases": cases}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--retry-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    source = EvaluationManifestV1.model_validate_json(args.manifest.read_text())
    selected, rejected = _select_runs(source, args.run_root)
    if len(selected) == len(source.cases):
        _write_json(args.evaluation_output, _evaluation_payload(source, selected))
    else:
        args.evaluation_output.unlink(missing_ok=True)
    _write_json(args.retry_output, _retry_payload(source, selected))
    _write_json(
        args.summary_output,
        {
            "case_count": len(source.cases),
            "valid_complete_count": len(selected),
            "remaining_count": len(source.cases) - len(selected),
            "selected_run_dirs": {key: str(value) for key, value in selected.items()},
            "rejected": rejected,
        },
    )
    return 0 if len(selected) == len(source.cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.gear.evaluation.graph_action_randomized_runner import ACTIONS
from experiments.gear.evaluation.rescue_postprocess_audit import (
    EXPECTED_GATE1_SPLITS,
    audit_claim_and_gate_coverage,
    audit_evidence_coverage,
    audit_randomized_outcomes,
)


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_evidence_audit_requires_every_manifest_paper(tmp_path: Path) -> None:
    manifest = _json(
        tmp_path / "manifest.json",
        {
            "selection_uses_future_outcomes": False,
            "cases": [{"paper_id": "W1"}, {"paper_id": "W2"}],
        },
    )
    summary = _json(
        tmp_path / "summary.json",
        {"failed_runs": [], "manifest_filtered": True, "target_papers": 2},
    )
    claims = _parquet(tmp_path / "claims.parquet", [{"paper_id": "W1"}])
    papers = _parquet(
        tmp_path / "papers.parquet", [{"paper_id": "W1"}, {"paper_id": "W2"}]
    )

    with pytest.raises(ValueError, match="claim evidence coverage mismatch"):
        audit_evidence_coverage(summary, claims, papers, manifest)


def test_claim_gate_audit_requires_complete_exact_frozen_coverage(
    tmp_path: Path,
) -> None:
    split_rows: list[dict[str, str]] = []
    context_rows: list[dict[str, object]] = []
    gate1_rows: list[dict[str, str]] = []
    for split, count in EXPECTED_GATE1_SPLITS.items():
        for index in range(count):
            paper_id = f"{split}-{index}"
            split_rows.append({"paper_id": paper_id, "integration_split": split})
            context_rows.append(
                {"paper_id": paper_id, "fetch_status": "resolved", "context_rows": 1}
            )
            gate1_rows.append({"paper_id": paper_id, "integration_split": split})
    context_path = tmp_path / "context_papers.jsonl"
    context_path.write_text(
        "".join(json.dumps(row) + "\n" for row in context_rows), encoding="utf-8"
    )
    split_path = _json(
        tmp_path / "splits.json",
        {"selection_uses_future_outcomes": False, "cases": split_rows},
    )
    summary = _json(
        tmp_path / "summary.json",
        {"failed_papers": [], "papers_resolved": 241, "papers_labeled": 241},
    )
    labels = _parquet(
        tmp_path / "labels.parquet",
        [{"paper_id": row["paper_id"]} for row in split_rows],
    )
    temporal_gate1 = _parquet(tmp_path / "temporal_gate1.parquet", gate1_rows)
    domain_gate1 = _parquet(tmp_path / "domain_gate1.parquet", gate1_rows)
    temporal_ids = {
        row["paper_id"]
        for row in split_rows
        if row["integration_split"] in {"temporal_holdout", "joint_time_domain_holdout"}
    }
    domain_ids = {
        row["paper_id"]
        for row in split_rows
        if row["integration_split"] in {"domain_holdout", "joint_time_domain_holdout"}
    }
    temporal_gate2 = _parquet(
        tmp_path / "temporal_gate2.parquet",
        [{"paper_id": value} for value in temporal_ids],
    )
    domain_gate2 = _parquet(
        tmp_path / "domain_gate2.parquet", [{"paper_id": value} for value in domain_ids]
    )

    report = audit_claim_and_gate_coverage(
        summary,
        labels,
        context_path,
        split_path,
        temporal_gate1,
        domain_gate1,
        temporal_gate2,
        domain_gate2,
    )
    assert report["passed"] is True
    assert report["gate2_papers"] == {"temporal": 49, "domain": 68}

    _json(
        summary,
        {
            "failed_papers": [{"paper_id": "W", "reason": "ValueError"}],
            "papers_resolved": 241,
            "papers_labeled": 240,
        },
    )
    with pytest.raises(ValueError, match="failed papers"):
        audit_claim_and_gate_coverage(
            summary,
            labels,
            context_path,
            split_path,
            temporal_gate1,
            domain_gate1,
            temporal_gate2,
            domain_gate2,
        )


def test_randomized_audit_matches_every_frozen_assignment(tmp_path: Path) -> None:
    cases: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index in range(150):
        split = "development" if index < 90 else "confirmatory_holdout"
        row = {
            "case_id": f"W{index}",
            "paper_id": f"W{index}",
            "context_id": f"CTX-{index}",
            "assigned_action": ACTIONS[index % len(ACTIONS)],
            "experiment_split": split,
            "propensity": 1 / len(ACTIONS),
            "matched_budget": 20,
        }
        cases.append(row)
        rows.append(
            {
                key: row[key]
                for key in (
                    "paper_id",
                    "context_id",
                    "assigned_action",
                    "experiment_split",
                    "propensity",
                    "matched_budget",
                )
            }
        )
    manifest = _json(
        tmp_path / "manifest.json",
        {"randomization_precedes_outcomes": True, "cases": cases},
    )
    report = _json(
        tmp_path / "report.json",
        {"randomization_precedes_outcomes": True, "identifiable": True},
    )
    log = _parquet(tmp_path / "log.parquet", rows)

    result = audit_randomized_outcomes(report, log, manifest)
    assert result["passed"] is True

    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["identifiable"] = False
    _json(report, payload)
    with pytest.raises(ValueError, match="not identifiable"):
        audit_randomized_outcomes(report, log, manifest)

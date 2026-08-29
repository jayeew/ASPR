from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from experiments.gear.evaluation.expert_annotation_pack import (
    AnnotationPackNotReady,
    generate_annotation_pack,
    main,
    validate_annotation_pack,
)

ACTIONS = [
    "baseline",
    "antecedent_falsification",
    "cross_field_pathway",
    "remote_mechanism_analogue",
    "opportunity_attribution_audit",
    "targeted_structural_probe",
]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    runs = tmp_path / "runs"
    claim_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for paper_index, paper_id in enumerate(("paper-1", "paper-2"), start=1):
        run = runs / paper_id
        run.mkdir(parents=True)
        evidence_key = f"P:S-{paper_index}"
        relation_key = f"R:{paper_index}"
        trace = [
            {
                "evidence_id": evidence_key,
                "kind": "paper_span",
                "payload": {"text": f"Manuscript support {paper_index}", "page": 1},
            },
            {
                "evidence_id": relation_key,
                "kind": "prior_relation",
                "payload": {
                    "title": f"Prior work {paper_index}",
                    "candidate_excerpt": "Comparable mechanism in an earlier setting.",
                    "relation": "DIRECT_ANTECEDENT",
                    "confidence": 0.99,
                },
            },
        ]
        (run / "evidence_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8"
        )
        _write_json(
            run / "review_bundle.json",
            {
                "state": {
                    "claim_inventory": [
                        {
                            "claim_id": "C1",
                            "manuscript_evidence_keys": [evidence_key],
                        }
                    ],
                    "canonical_points": {
                        "point": {"relation_evidence_keys": [relation_key]}
                    },
                }
            },
        )
        claim_rows.append(
            {
                "paper_id": paper_id,
                "claim_id": "C1",
                "claim_text": f"Claim {paper_index}",
                "gear_run_path": str(run),
                "verification_passed": True,
                "manuscript_evidence_keys": [evidence_key],
            }
        )
        gate_rows.append(
            {
                "paper_id": paper_id,
                "claim_id": "C1",
                "domain12": f"domain-{paper_index}",
                "publication_year": 2010 + paper_index,
                "integration_split": "development",
                "graph_percentile": 20.0 + 40 * paper_index,
                "structural_innovation_score": 0.2 + 0.1 * paper_index,
                "structural_score_at_zero": 0.1 + 0.05 * paper_index,
            }
        )
    claims = tmp_path / "claims.parquet"
    gate1 = tmp_path / "gate1.parquet"
    pd.DataFrame(claim_rows).to_parquet(claims, index=False)
    pd.DataFrame(gate_rows).to_parquet(gate1, index=False)
    stage_c_rows: list[dict[str, Any]] = []
    stage_c_cases: list[dict[str, Any]] = []
    for split_index, split in enumerate(("development", "confirmatory_holdout")):
        for action_index, action in enumerate(ACTIONS):
            paper_id = (
                "paper-1"
                if split == "confirmatory_holdout" and action_index == 0
                else f"stage-c-{split_index}-{action_index}"
            )
            row = {
                "paper_id": paper_id,
                "assigned_action": action,
                "propensity": 1.0 / 6.0,
                "matched_budget": 20,
                "experiment_split": split,
                "context_id": f"ctx-{split_index}-{action_index}",
            }
            stage_c_rows.append(row)
            stage_c_cases.append(dict(row))
    stage_c = tmp_path / "stage_c.parquet"
    pd.DataFrame(stage_c_rows).to_parquet(stage_c, index=False)
    stage_b_manifest = tmp_path / "stage_b_manifest.json"
    stage_c_manifest = tmp_path / "stage_c_manifest.json"
    _write_json(
        stage_b_manifest,
        {
            "selection_uses_future_outcomes": False,
            "cases": [{"paper_id": "paper-1"}, {"paper_id": "paper-2"}],
        },
    )
    _write_json(
        stage_c_manifest,
        {
            "randomization_precedes_outcomes": True,
            "cases": stage_c_cases,
        },
    )
    return {
        "claims": claims,
        "gate1": gate1,
        "stage_c": stage_c,
        "stage_b_manifest": stage_b_manifest,
        "stage_c_manifest": stage_c_manifest,
        "runs": runs,
    }


def _generate(paths: dict[str, Path], output: Path) -> None:
    generate_annotation_pack(
        paths["claims"],
        paths["gate1"],
        paths["stage_c"],
        paths["stage_b_manifest"],
        paths["stage_c_manifest"],
        paths["runs"],
        output,
        claim_b_papers=2,
        claim_c_pairs=1,
        expected_stage_b_papers=2,
        expected_stage_c_cases=12,
    )


def test_partial_stage_inputs_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    stage_c = pd.read_parquet(paths["stage_c"]).iloc[:-1]
    stage_c.to_parquet(paths["stage_c"], index=False)
    with pytest.raises(AnnotationPackNotReady, match="stage_c_log_incomplete"):
        _generate(paths, tmp_path / "pack")
    assert not (tmp_path / "pack" / "manifest.json").exists()


def test_builds_label_free_blinded_pack_without_hash_bindings(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pack = tmp_path / "pack"
    _generate(paths, pack)
    report = validate_annotation_pack(pack)
    assert report["valid"] is True
    manifest = json.loads((pack / "manifest.json").read_text())
    assert manifest["labels_included"] is False
    assert "file_sha256" not in manifest
    assert "source_sha256" not in manifest
    assert "sealed_key_sha256" not in manifest["blinding"]
    assert manifest["review_design"]["independent_experts_per_task"] == 2
    task = json.loads((pack / "claim_c_tasks.jsonl").read_text().splitlines()[0])
    serialized = json.dumps(task).casefold()
    assert "paper-1" not in serialized
    assert "evidence_gated_fusion" not in serialized
    assert "gear_evidence_only" not in serialized
    assert "graph_percentile" not in serialized
    relation_excerpt = task["left"]["claims"][0]["relation_evidence"][0]["excerpt"]
    assert "direct_antecedent" not in relation_excerpt.casefold()
    template = json.loads(
        (pack / "claim_b_annotation_template.jsonl").read_text().splitlines()[0]
    )
    assert template["annotator_id"] is None
    assert template["assessments"][0]["confidence"] is None


def test_validator_does_not_enforce_file_hashes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pack = tmp_path / "pack"
    _generate(paths, pack)
    with (pack / "CODEBOOK.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    assert validate_annotation_pack(pack)["valid"] is True


def test_validate_cli_writes_machine_readable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    pack = tmp_path / "pack"
    output = tmp_path / "ready_validation.json"
    _generate(paths, pack)
    monkeypatch.setattr(
        "sys.argv",
        [
            "expert_annotation_pack",
            "validate",
            "--pack-dir",
            str(pack),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["completed_annotations_validated"] is False
    assert "annotation_sha256" not in report


def test_completed_validation_requires_two_independent_experts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pack = tmp_path / "pack"
    _generate(paths, pack)
    b_task = json.loads((pack / "claim_b_tasks.jsonl").read_text().splitlines()[0])
    c_task = json.loads((pack / "claim_c_tasks.jsonl").read_text().splitlines()[0])
    evidence_key = b_task["claims"][0]["manuscript_evidence"][0]["evidence_key"]
    b_row = {
        "contract": "gear_claim_b_expert_annotation_v1",
        "task_id": b_task["task_id"],
        "annotator_id": "expert-1",
        "inventory_complete": "YES",
        "inventory_rationale": "The proposed inventory covers the supplied material.",
        "assessments": [
            {
                "claim_alias": claim["claim_alias"],
                "inventory_valid": "YES",
                "relation": "UNVERIFIABLE",
                "residual_novelty": "UNVERIFIABLE",
                "manuscript_support": "YES",
                "trace_complete": "PARTIAL",
                "confidence": 0.7,
                "rationale": "The manuscript span supports the claim, but relation evidence is insufficient.",
                "evidence_keys": [claim["manuscript_evidence"][0]["evidence_key"]],
            }
            for claim in b_task["claims"]
        ],
    }
    c_evidence = c_task["left"]["claims"][0]["manuscript_evidence"][0]["evidence_key"]
    c_row = {
        "contract": "gear_claim_c_expert_annotation_v1",
        "task_id": c_task["task_id"],
        "annotator_id": "expert-1",
        "preference": "TIE",
        "confidence": 0.6,
        "rationale": "Both sides are equally supported by the supplied evidence.",
        "evidence_keys": [c_evidence],
    }
    (pack / "claim_b_annotations.jsonl").write_text(
        json.dumps(b_row) + "\n", encoding="utf-8"
    )
    (pack / "claim_c_annotations.jsonl").write_text(
        json.dumps(c_row) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly two distinct"):
        validate_annotation_pack(pack, require_completed=True)
    assert evidence_key

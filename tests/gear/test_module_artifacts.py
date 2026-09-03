from __future__ import annotations

import json

from artifact_store import ArtifactReference, ArtifactStore
from experiments.gear.evaluation.artifacts import load_reference_release
from gear.module_cli import main


def _review_payload(paper_id: str, evidence: str) -> dict[str, object]:
    return {
        "schema_version": "aspr_gear",
        "schema_revision": "evidence_state_delta_v2",
        "paper_id": paper_id,
        "summary": {
            "schema_version": "aspr_gear",
            "schema_revision": "evidence_state_delta_v2",
            "text": "A grounded summary.",
            "evidence_keys": [evidence],
        },
        "novelty": {
            "schema_version": "aspr_gear",
            "schema_revision": "evidence_state_delta_v2",
            "judgment": "not_discussed",
            "supporting_points": [],
            "limiting_points": [],
            "uncertain_points": [],
        },
        "strengths": [],
        "weaknesses": [],
        "questions": [],
    }


def test_module_publish_and_resolve(monkeypatch, paper_ir, tmp_path):
    store = tmp_path / "store"
    refs = tmp_path / "refs"
    source = tmp_path / "human"
    source.mkdir()
    evidence = f"P:{paper_ir.spans[0].span_id}"
    (source / "human_structured_reviews.jsonl").write_text(
        json.dumps(_review_payload(paper_ir.paper_id, evidence)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ASPR_GEAR_ARTIFACT_ROOT", str(store))
    monkeypatch.setenv("ASPR_GEAR_REFERENCE_ROOT", str(refs))

    assert (
        main(
            [
                "publish",
                "--module",
                "review_reconstruction",
                "--release",
                "r1",
                "--source",
                str(source),
            ]
        )
        == 0
    )
    reference_path = refs / "review_reconstruction/r1.json"
    reference = ArtifactReference.model_validate_json(
        reference_path.read_text(encoding="utf-8")
    )
    resolved = ArtifactStore(store).resolve(reference)
    assert (resolved / "human_structured_reviews.jsonl").is_file()


def test_agent_export_is_exact_structured_review(paper_ir, tmp_path):
    evidence = f"P:{paper_ir.spans[0].span_id}"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "review.json").write_text(
        json.dumps(_review_payload(paper_ir.paper_id, evidence)), encoding="utf-8"
    )
    output = tmp_path / "export"
    assert (
        main(["export-agent", "--run-dir", str(run_dir), "--output-dir", str(output)])
        == 0
    )
    assert (
        len((output / "agent_structured_reviews.jsonl").read_text().splitlines()) == 1
    )


def test_reference_loader_accepts_ai_session_release(paper_ir, tmp_path):
    evidence = f"P:{paper_ir.spans[0].span_id}"
    (tmp_path / "reference_structured_reviews.jsonl").write_text(
        json.dumps(_review_payload(paper_ir.paper_id, evidence)) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "release_manifest.json").write_text(
        json.dumps({"record_count": 1}), encoding="utf-8"
    )

    reviews, revision_labels = load_reference_release(tmp_path)

    assert set(reviews) == {paper_ir.paper_id}
    assert revision_labels == {}

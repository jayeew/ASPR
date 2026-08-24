from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from gear.contracts import ReviewRequest
from gear.review_contracts import PointSeverity, ReviewAspect, ReviewPoint
from gear.trace import EvidenceStore


def test_contracts_forbid_unknown_fields(sample_pdf):
    with pytest.raises(ValidationError):
        ReviewRequest(
            paper_path=sample_pdf,
            unexpected="silent drift",
        )

    with pytest.raises(ValidationError):
        ReviewRequest(
            schema_version="legacy",
            paper_path=sample_pdf,
        )


def test_major_point_requires_evidence_key():
    with pytest.raises(ValidationError):
        ReviewPoint(
            point_id="RP-x",
            aspect=ReviewAspect.METHOD,
            text="A major unsupported assertion.",
            evidence_keys=[],
            severity=PointSeverity.MAJOR,
        )


def test_evidence_date_is_derived_from_metadata(sample_pdf):
    request = ReviewRequest(
        paper_path=sample_pdf,
        metadata={"publication_date": "2020-01-02", "submission_date": "2019-12-01"},
        evaluation_date=date(2026, 8, 10),
    )
    assert request.evidence_date == date(2019, 12, 1)
    assert request.evidence_date_source == "submission_date"


def test_evidence_store_rejects_semantic_overwrite(tmp_path):
    store = EvidenceStore(tmp_path)
    store.add_evidence("P:stable", "paper_span", {"text": "first"})
    store.add_evidence("P:stable", "paper_span", {"text": "first"})
    with pytest.raises(ValueError, match="overwrite rejected"):
        store.add_evidence("P:stable", "paper_span", {"text": "changed"})
    with pytest.raises(ValueError, match="overwrite rejected"):
        store.add_evidence("P:stable", "other_kind", {"text": "first"})


def test_evidence_store_rejects_tampered_serialized_payload(tmp_path):
    store = EvidenceStore(tmp_path)
    store.add_evidence("P:stable", "paper_span", {"text": "first"})
    path = tmp_path / "evidence_trace.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["text"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stored evidence hash mismatch"):
        EvidenceStore(tmp_path)

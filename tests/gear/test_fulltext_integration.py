from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gear.config import GearConfig, load_config
from gear.contracts import EvidenceLevel, OpenAccessLocation, RetrievedWork
from gear.evidence_supervisor import EvidenceSupervisor
from gear.innovation.analysis import evidence_payloads
from gear.prior_art import PriorArtService
from gear.trace import EvidenceStore


def _work(work_id: str) -> RetrievedWork:
    return RetrievedWork(
        work_id=work_id,
        target_claim_id="c1",
        title="A useful scientific study title",
        doi="10.1234/test",
        abstract="Abstract evidence.",
        retrieval_query_id="q1",
        retrieval_source="openalex",
        open_access_locations=[
            OpenAccessLocation(
                is_oa=True,
                pdf_url="https://example.org/paper.pdf",
            )
        ],
    )


def test_external_fulltext_keeps_raw_source_and_limits_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gear import fulltext

    cfg = GearConfig(
        retrieval={"external_fulltext_enabled": True, "external_fulltext_max_works": 2}
    )
    supervisor = EvidenceSupervisor(cfg, EvidenceStore(tmp_path))
    works = [_work(f"W{i}") for i in range(5)]
    supervisor._works = {w.work_id: w for w in works}
    calls: list[str] = []

    def fetch(work_id: str, *args: object, **kwargs: object) -> dict:
        calls.append(work_id)
        return {
            "status": "success",
            "text": "This detailed experiment measured scientific effects. " * 40,
            "source_url": "https://example.org/paper.pdf",
            "provider": "publisher",
            "format": "pdf",
            "identity_verified": True,
            "cache_hit": False,
        }

    monkeypatch.setattr(fulltext, "fetch_external_fulltext", fetch)
    claim = SimpleNamespace(claim_id="c1", normalized_claim_text="scientific effects")
    result = supervisor._prepare_fulltext(claim, works)
    assert len(calls) == 2
    assert all(w.fulltext_provenance for w in result[:2])
    assert all(w.spans[0].source == EvidenceLevel.FULLTEXT for w in result[:2])
    supervisor._prepare_fulltext(claim, works[2:])
    assert len(calls) == 2
    raw = EvidenceStore(tmp_path)._evidence["FULLTEXT_FETCH:c1:W0"].payload
    assert raw["text"]
    assert "text" not in evidence_payloads(tmp_path)["FULLTEXT_FETCH:c1:W0"]


def test_external_failure_keeps_abstract_and_never_uses_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear import fulltext

    cfg = GearConfig(
        retrieval={"external_fulltext_enabled": True, "openalex_pdf_enabled": True}
    )
    service = PriorArtService(cfg)
    monkeypatch.setattr(
        fulltext,
        "fetch_external_fulltext",
        lambda *args, **kwargs: {
            "status": "unavailable",
            "text": "",
            "identity_verified": False,
        },
    )
    monkeypatch.setattr(
        service,
        "_openalex_pdf_text",
        lambda *args: pytest.fail("External mode must not fall back to paid content"),
    )
    work = _work("W1")
    assert service.upgrade_fulltext(work, "scientific effects") is work


def test_external_switch_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEAR_EXTERNAL_FULLTEXT_ENABLED", "true")
    monkeypatch.setenv("GEAR_EXTERNAL_FULLTEXT_MAX_WORKS", "2")
    assert load_config().retrieval.external_fulltext_enabled
    monkeypatch.setenv("GEAR_EXTERNAL_FULLTEXT_MAX_WORKS", "6")
    with pytest.raises(ValueError):
        load_config()

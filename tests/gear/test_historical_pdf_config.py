from __future__ import annotations

from pathlib import Path

import pytest

from gear import env
from gear.config import GearConfig, load_config
from gear.prior_art import PriorArtService


def test_historical_pdf_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "_LOADED", True)
    monkeypatch.delenv("GEAR_HISTORICAL_PDF_ENABLED", raising=False)
    assert not GearConfig().retrieval.openalex_pdf_enabled
    assert not load_config().retrieval.openalex_pdf_enabled


@pytest.mark.parametrize(
    "value,enabled",
    [("true", True), ("1", True), ("false", False), ("0", False), ("OFF", False)],
)
def test_environment_switch_overrides_json(
    monkeypatch: pytest.MonkeyPatch, value: str, enabled: bool
) -> None:
    monkeypatch.setattr(env, "_LOADED", True)
    monkeypatch.setenv("GEAR_HISTORICAL_PDF_ENABLED", value)
    config = load_config(overrides={"retrieval": {"openalex_pdf_enabled": not enabled}})
    assert config.retrieval.openalex_pdf_enabled is enabled


def test_dotenv_switch_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEAR_HISTORICAL_PDF_ENABLED=true\n", encoding="utf-8")
    monkeypatch.setattr(env, "DEFAULT_ENV_FILES", (dotenv,))
    monkeypatch.setattr(env, "_LOADED", False)
    monkeypatch.delenv("GEAR_HISTORICAL_PDF_ENABLED", raising=False)
    assert load_config().retrieval.openalex_pdf_enabled
    monkeypatch.setenv("GEAR_HISTORICAL_PDF_ENABLED", "false")
    assert not load_config().retrieval.openalex_pdf_enabled


def test_invalid_pdf_switch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEAR_HISTORICAL_PDF_ENABLED", "tru")
    with pytest.raises(ValueError, match="GEAR_HISTORICAL_PDF_ENABLED"):
        load_config()


def test_disabled_pdf_switch_ignores_previous_pdf_text_cache(
    gear_config: GearConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = gear_config.retrieval.model_copy(update={"openalex_pdf_enabled": False})
    service = PriorArtService(gear_config.model_copy(update={"retrieval": retrieval}))
    work_id = "https://openalex.org/W123"
    service._pdf_text_cache[work_id] = "Cached historical PDF text"
    monkeypatch.setattr(
        service,
        "_client",
        lambda: pytest.fail("PDF client must remain lazy when disabled"),
    )
    assert service._openalex_pdf_text(work_id) == ""
    assert service._pdf_downloads_used == 0


def test_supervisor_disabled_pdf_still_classifies_abstracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date
    from types import SimpleNamespace

    from gear.evidence_supervisor import EvidenceSupervisor
    from gear.trace import EvidenceStore

    supervisor = EvidenceSupervisor(GearConfig(), EvidenceStore(tmp_path))
    monkeypatch.setattr(
        supervisor.prior_art,
        "upgrade_fulltext",
        lambda *args: pytest.fail("Disabled PDF mode must not attempt an upgrade"),
    )
    seen: list[object] = []

    def classify(span: object, works: list[object], **kwargs: object) -> list:
        seen.extend(works)
        return []

    monkeypatch.setattr(supervisor.classifier, "classify_many", classify)
    work = SimpleNamespace(work_id="W123", abstract="An existing historical abstract")
    claim = SimpleNamespace(claim_id="c1", normalized_claim_text="A target claim")
    supervisor._classify(None, claim, [work], date(2026, 1, 1), {}, [])
    assert seen == [work]

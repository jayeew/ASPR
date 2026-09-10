from __future__ import annotations

import threading
from typing import Any

import pytest

from gear.config import GearConfig
from gear.prior_art import PriorArtService


def test_candidate_batches_overlap_and_preserve_all_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PriorArtService(GearConfig(), rerank_generator=lambda *args: {})
    gate = threading.Barrier(2)
    batches: list[list[int]] = []

    def batch(
        frame: object, works: list[int], **kwargs: object
    ) -> dict[str, dict[str, Any]]:
        batches.append(list(works))
        gate.wait(timeout=3)
        return {str(item): {"verdict": "partial"} for item in works}

    monkeypatch.setenv("GEAR_CANDIDATE_GATE_WORKERS", "2")
    monkeypatch.setattr(service, "_rerank_batch", batch)
    result = service._rerank(None, list(range(16)))
    assert sorted(batches) == [list(range(8)), list(range(8, 16))]
    assert list(result) == [str(item) for item in range(16)]

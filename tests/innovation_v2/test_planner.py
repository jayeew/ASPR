from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gear.config import GearConfig
from gear.evidence_supervisor import EvidenceSupervisor
from gear.review_contracts import GearClaim
from gear.trace import EvidenceStore


@pytest.fixture
def claim() -> GearClaim:
    return GearClaim(
        claim_id="c",
        claim_type="FINDING",
        author_claim_text="x",
        normalized_claim_text="x",
        source_span_ids=["s"],
        support_span_ids=["s"],
        internal_support="supported",
        narrowing_reason="",
    )


@pytest.mark.parametrize(
    "action",
    [
        "normal_search",
        "verify_relation",
        "contrastive_search",
        "stability_check",
        "citation_expand",
        "finalize",
    ],
)
def test_single_action_skips_planner(
    tmp_path: Path, claim: GearClaim, action: str
) -> None:
    supervisor = EvidenceSupervisor(GearConfig(), EvidenceStore(tmp_path))
    with patch.object(supervisor.planner, "generate_json") as generate:
        selected, reason = supervisor._choose_action(claim, [action], {}, {})
    generate.assert_not_called()
    assert selected == action
    assert reason
    assert supervisor.planner._client is None


def test_multiple_actions_keep_model_choice(tmp_path: Path, claim: GearClaim) -> None:
    supervisor = EvidenceSupervisor(GearConfig(), EvidenceStore(tmp_path))
    legal = ["citation_expand", "finalize"]
    with patch.object(
        supervisor.planner,
        "generate_json",
        return_value={"action": "finalize", "reason": "Evidence budget exhausted."},
    ) as generate:
        result = supervisor._choose_action(claim, legal, {}, {})
    generate.assert_called_once()
    assert json.loads(generate.call_args.kwargs["user"])["legal_actions"] == legal
    assert result == ("finalize", "Evidence budget exhausted.")

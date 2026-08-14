from __future__ import annotations

import pytest

from gear.reviewers.base import assert_graph_blind_payload


def test_reviewer_payload_rejects_graph_fields() -> None:
    with pytest.raises(ValueError, match="Graph fields"):
        assert_graph_blind_payload(
            {"paper_id": "paper", "graph_prior": {"score_0_100": 91.0}}
        )

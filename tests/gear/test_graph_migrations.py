from __future__ import annotations

from gear.graph_prior_contracts import GraphPriorResult
from gear.migrations import migrate_review_state_v2
from gear.paper_extraction import PaperRubricBuilder
from gear.review_state import initialize_review_state_v2


def _legacy_state(paper_ir, paper_request):
    return initialize_review_state_v2(
        paper_ir,
        PaperRubricBuilder().build(paper_ir),
        GraphPriorResult(
            paper_id=paper_ir.paper_id,
            status="exact_lookup",
            score_0_100=82.5,
            feature_coverage=0.75,
        ),
        paper_request.evidence_date,
    )


def test_v2_state_migrates_only_complete_graph_components(
    paper_ir, paper_request
) -> None:
    migrated = migrate_review_state_v2(
        _legacy_state(paper_ir, paper_request),
        {
            "forecast": {
                "p_uptake": 0.5,
                "conditional_diffusion": 0.4,
            }
        },
    )

    assert migrated.contract == "aspr_evidence_state_v3"
    assert migrated.graph_result is not None
    assert migrated.graph_result.score_0_100 == 82.5
    assert migrated.graph_result.feature_coverage == 1.0


def test_v2_state_does_not_invent_missing_graph_components(
    paper_ir, paper_request
) -> None:
    migrated = migrate_review_state_v2(_legacy_state(paper_ir, paper_request))

    assert migrated.graph_result is None
    assert migrated.process_features.graph_score_available is False

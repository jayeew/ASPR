from experiments.gear.evaluation.human_audit import (
    AuditEligibleCase,
    HumanCandidateJudgment,
    HumanVariantJudgment,
    TraceableIssue,
    quick_gate_passes,
    rank_metrics,
    select_quick_gate,
)


def case(index: int, status: str = "persists") -> AuditEligibleCase:
    return AuditEligibleCase(
        paper_id=f"paper-{index}",
        percentile=float(index),
        cutoff_safe=True,
        trace_complete=True,
        issues=[
            TraceableIssue(
                issue_id=f"issue-{index}",
                aspect="novelty_prior_art",
                status=status,
                reviewer_quote_keys=["reviewer:1"],
                reviewer_ids=["reviewer-a"],
                round_ids=["round-1"],
                final_paper_evidence_keys=["P:S-1"],
            )
        ],
    )


def judgment(
    paper_id: str, variant: str, gain: bool, *, review: str = "tie"
) -> HumanVariantJudgment:
    return HumanVariantJudgment(
        paper_id=paper_id,
        variant=variant,
        judgments=[
            HumanCandidateJudgment(
                candidate_id="candidate",
                relation="DIRECT" if gain else "DISTANT",
                claim_relevant=gain,
                material_review_change=gain,
            )
        ],
        issue_recall=1.0 if gain else 0.0,
        major_evidence_precision=1.0,
        hgb_analog_valid_relation_count=int(gain and variant == "full_calibrated"),
        final_review_comparison_to_topology=review,
    )


def test_quick_selection_is_low_median_high_and_prefers_active_issues() -> None:
    cases = [
        case(index, "resolved" if index in {0, 5, 10} else "persists")
        for index in range(11)
    ]
    selected = select_quick_gate(cases)
    assert len(selected) == 3
    assert selected[0].percentile <= selected[1].percentile <= selected[2].percentile
    assert all(row.preferred_status for row in selected)


def test_human_metrics_and_quick_gate_require_two_of_three_real_wins() -> None:
    rows = []
    for index in range(3):
        paper_id = f"paper-{index}"
        rows.extend(
            [
                rank_metrics(judgment(paper_id, "neutral", False)),
                rank_metrics(judgment(paper_id, "topology_only", False)),
                rank_metrics(judgment(paper_id, "scalar_score", index < 2)),
                rank_metrics(judgment(paper_id, "hgb_analog", index < 2)),
                rank_metrics(
                    judgment(
                        paper_id,
                        "full_calibrated",
                        index < 2,
                        review="win" if index == 0 else "tie",
                    )
                ),
                rank_metrics(judgment(paper_id, "shuffled_hgb", False)),
            ]
        )
    assert quick_gate_passes(rows)

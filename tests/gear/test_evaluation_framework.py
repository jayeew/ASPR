from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from experiments.gear.evaluation.artifacts import load_human_release
from experiments.gear.evaluation.contracts import (
    EvaluationCase,
    EvaluationManifestV1,
    PointSupportDecisionV1,
    RevisionIssueLabel,
    RevisionIssueMatchDecision,
    RubricDecision,
)
from experiments.gear.evaluation.efficiency import graph_action_metrics
from experiments.gear.evaluation.faults import perturb_manuscript
from experiments.gear.evaluation.graph_ablation import (
    graph_tension,
    graph_variants,
    shuffled_graph_components,
    shuffled_graph_results,
    tension_band,
)
from experiments.gear.evaluation.judges import (
    judge_point_support,
    judge_revision_issues,
    judge_semantic_matches,
)
from experiments.gear.evaluation.metrics import (
    bootstrap_ci,
    evidence_support_metrics,
    rubric_metrics,
    semantic_match_metrics,
)
from experiments.gear.evaluation.runner import _aggregate_graph_deltas
from experiments.gear.review_reconstruction.evaluation import (
    MatchJudgeResponse,
    MatchLabel,
    PointMatchDecision,
    build_blind_match_package,
)
from gear.graph_prior_contracts import (
    GraphResultV4,
    GraphRuntimePacketV1,
    GraphTopologySeedV1,
)
from gear.review_contracts import (
    NoveltyAssessment,
    NoveltyJudgment,
    PointSeverity,
    ReviewAspect,
    ReviewPoint,
    ReviewSummary,
    StructuredReview,
)
from gear.trace import EvidenceStore, sha256_file


def _graph(paper_id: str, score: float = 80.0) -> GraphResultV4:
    return GraphResultV4(
        paper_id=paper_id,
        score_0_100=score,
        p_uptake=0.8,
        conditional_diffusion=0.6,
        feature_coverage=1.0,
        seed_work_ids=[f"{paper_id}-seed"],
        search_terms=["graph guided"],
    )


def _review() -> StructuredReview:
    point = ReviewPoint(
        point_id="P1",
        text="The limitation is explicit.",
        aspect=ReviewAspect.OTHER,
        severity=PointSeverity.MAJOR,
        evidence_keys=["P:S-1"],
        suggested_action="Clarify scope.",
    )
    return StructuredReview(
        paper_id="paper-1",
        summary=ReviewSummary(text="Summary", evidence_keys=["P:S-1"]),
        novelty=NoveltyAssessment(judgment=NoveltyJudgment.NOT_DISCUSSED),
        weaknesses=[point],
    )


def test_manifest_rejects_duplicate_ids_and_graph_mismatch(tmp_path) -> None:
    case = EvaluationCase(
        case_id="c1",
        paper_id="p1",
        manuscript_path=tmp_path / "paper.md",
        metadata_path=tmp_path / "metadata.json",
        cutoff_date=date(2020, 1, 1),
        graph_result=_graph("p1"),
    )
    with pytest.raises(ValidationError):
        EvaluationManifestV1(
            dataset_id="d",
            human_release_dir=tmp_path,
            cases=[case, case],
            tracks=["novelty"],
        )
    with pytest.raises(ValidationError):
        EvaluationCase(
            **case.model_dump(exclude={"graph_result"}), graph_result=_graph("wrong")
        )


def test_graph_variants_are_legal_and_shuffled_keeps_target_id() -> None:
    first, second = _graph("p1", 20.0), _graph("p2", 90.0)
    variants = {row.name: row.result for row in graph_variants(first)}
    assert variants["neutral"].score_0_100 == 50.0
    assert variants["score_only"].p_uptake == first.p_uptake
    assert variants["score_only"].topology_seeds == []
    assert variants["score_profile"].topology_seeds == []
    shuffled = shuffled_graph_results([first, second])
    assert shuffled["p1"].paper_id == "p1"
    assert shuffled["p1"].score_0_100 == second.score_0_100


def test_random_topology_replaces_executed_title_and_identity() -> None:
    def packet(paper_id: str, work_id: str, title: str) -> GraphRuntimePacketV1:
        return GraphRuntimePacketV1(
            paper_id=paper_id,
            score_0_100=80,
            raw_expected_diffusion=0.5,
            p_uptake=0.8,
            conditional_diffusion=0.6,
            topology_seeds=[
                GraphTopologySeedV1(
                    work_id=work_id,
                    title=title,
                    publication_year=2020,
                    anchor_field_ids=["field"],
                )
            ],
        )

    first = packet("p1", "W1", "Real topology title")
    second = packet("p2", "W2", "Matched donor title")

    random_seed = shuffled_graph_components([first, second])["p1"][
        "random_matched_topology"
    ].topology_seeds[0]

    assert random_seed.work_id == "W2"
    assert random_seed.title == "Matched donor title"


def test_weighted_alignment_counts_partial_as_half() -> None:
    decisions = [
        PointMatchDecision(
            paper_id="p1",
            reference_point_id="h1",
            candidate_point_id="a1",
            label=MatchLabel.PARTIAL_POINT,
        )
    ]
    metrics = semantic_match_metrics(decisions, reference_count=1, candidate_count=1)
    assert metrics["human_concern_coverage"] == 1.0
    assert metrics["weighted_alignment_f1"] == 0.5


def test_tension_is_continuous_and_banded() -> None:
    assert graph_tension(50.0, 1.0, 1) == 0.0
    assert graph_tension(100.0, 1.0, -1) == 1.0
    assert tension_band(0.24) == "low"
    assert tension_band(0.25) == "medium"
    assert tension_band(0.50) == "high"


def test_graph_delta_aggregate_reports_direction_rates() -> None:
    result = _aggregate_graph_deltas(
        [
            {"comparison": "full-neutral", "relation_count_delta": 2.0},
            {"comparison": "full-neutral", "relation_count_delta": -1.0},
            {"comparison": "full-neutral", "relation_count_delta": 0.0},
        ],
        samples=20,
        seed=7,
    )["full-neutral"]["relation_count_delta"]
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["positive_rate"] == pytest.approx(1 / 3)
    assert result["negative_rate"] == pytest.approx(1 / 3)
    assert result["zero_rate"] == pytest.approx(1 / 3)


def test_graph_action_metrics_reads_compact_bundle_fusion(tmp_path) -> None:
    (tmp_path / "review_bundle.json").write_text(
        json.dumps(
            {
                "fusion_report": {
                    "graph_triggered_actions": {"P1": ["retrieve_prior_art"]}
                }
            }
        )
    )
    (tmp_path / "action_trace.jsonl").write_text(
        json.dumps({"target_id": "P1", "output_sha256": "sha256:x"}) + "\n"
    )
    metrics = graph_action_metrics(tmp_path)
    assert metrics["graph_trigger_compliance"] == 1.0
    assert metrics["graph_evidence_yield"] == 1.0


def test_graph_seed_rates_are_query_level_and_bounded(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    store.add_evidence(
        "Q:q1", "retrieval_query", {"query_id": "q1", "query_role": "graph_seed"}
    )
    store.add_evidence(
        "G:LEDGER",
        "resource_ledger",
        {
            "logical_provider_searches": 1,
            "logical_direct_fetches": 0,
            "logical_neighbor_expansions": 0,
        },
    )
    for index in range(3):
        store.add_evidence(
            f"H:h{index}",
            "retrieval_hit",
            {
                "query_id": "q1",
                "work_id": f"W{index}",
                "selection_stage": "compared" if index == 0 else "recall_filtered",
                "gate_label": "partial" if index == 0 else None,
            },
        )
    store.add_evidence(
        "R:r1",
        "prior_relation",
        {
            "prior_work_id": "W0",
            "source_query_ids": ["q1"],
            "temporal_valid": True,
            "relation_label": "PARALLEL",
            "essential_facet_coverage": 0.5,
            "common_dimensions": ["shared mechanism"],
            "difference_dimensions": ["different implementation"],
        },
    )

    metrics = graph_action_metrics(tmp_path)

    assert metrics["graph_seed_fetch_rate"] == 1.0
    assert metrics["graph_seed_comparable_rate"] == 1.0
    assert metrics["graph_seed_verified_relation_yield"] == 1


def test_low_facet_parallel_is_not_claim_relevant_graph_yield(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    store.add_evidence(
        "Q:q1", "retrieval_query", {"query_id": "q1", "query_role": "graph_seed"}
    )
    store.add_evidence(
        "G:LEDGER",
        "resource_ledger",
        {
            "logical_provider_searches": 1,
            "logical_direct_fetches": 0,
            "logical_neighbor_expansions": 0,
        },
    )
    store.add_evidence(
        "R:r1",
        "prior_relation",
        {
            "relation_id": "r1",
            "prior_work_id": "W0",
            "source_query_ids": ["q1"],
            "temporal_valid": True,
            "relation_label": "PARALLEL",
            "essential_facet_coverage": 0.2,
            "common_dimensions": ["generic method"],
            "difference_dimensions": ["unrelated phenotype"],
        },
    )

    metrics = graph_action_metrics(tmp_path)

    assert metrics["graph_seed_verified_relation_yield"] == 0
    assert metrics["claim_relevant_verified_relation_yield"] == 0.0


def test_metrics_do_not_invent_empty_perfection() -> None:
    assert bootstrap_ci([], samples=10, seed=1) is None
    review = _review()
    decision = PointSupportDecisionV1(
        paper_id=review.paper_id,
        point_id="P1",
        label="PARTIALLY_SUPPORTED",
        confidence=0.7,
    )
    metrics = evidence_support_metrics(review, [decision])
    assert metrics["strict_support_precision"] == 0.0
    assert metrics["soft_support_precision"] == 0.5
    assert metrics["unsupported_major_rate"] == 0.0


def test_support_judge_batches_all_points_in_one_request() -> None:
    class Client:
        calls = 0

        def generate_model(self, *, system, user, response_model):
            del system
            self.calls += 1
            payload = json.loads(user)
            return response_model(
                paper_id=payload["paper_id"],
                decisions=[
                    PointSupportDecisionV1(
                        paper_id=payload["paper_id"],
                        point_id=item["point"]["point_id"],
                        label="SUPPORTED",
                        confidence=1.0,
                    )
                    for item in payload["items"]
                ],
            )

    client = Client()
    result = judge_point_support(client, _review(), {"P:S-1": {"text": "x"}})
    assert client.calls == 1
    assert [row.point_id for row in result.decisions] == ["P1"]


def test_match_judge_normalizes_redundant_paper_identity() -> None:
    package = build_blind_match_package(_review(), _review())

    class Client:
        model_name = "test-model"

        def generate_model(self, *, system, user, response_model):
            del system, user, response_model
            return MatchJudgeResponse(
                task_id=package.task_id,
                model_id="test-model",
                conversation_hash="sha256:" + "0" * 64,
                decisions=[
                    PointMatchDecision(
                        paper_id="paper-1",
                        reference_point_id=left,
                        candidate_point_id=right,
                        label=MatchLabel.SAME_POINT,
                    )
                    for left, right in package.candidate_pairs
                ],
            )

    result = judge_semantic_matches(Client(), package)
    assert {row.paper_id for row in result.decisions} == {package.paper_id_hash}


def test_revision_judge_fills_omitted_pairs_conservatively() -> None:
    review = _review()
    issues = [
        RevisionIssueLabel(
            paper_id=review.paper_id,
            issue_id="I1",
            text="A persistent issue",
            section="weaknesses",
            aspect=ReviewAspect.OTHER,
            severity=PointSeverity.MAJOR,
            status="persists",
        )
    ]

    class Client:
        def generate_model(self, *, system, user, response_model):
            del system, user
            return response_model(paper_id="wrong", decisions=[])

    result = judge_revision_issues(Client(), review.paper_id, issues, review)
    assert result.paper_id == review.paper_id
    assert result.decisions == [
        RevisionIssueMatchDecision(
            issue_id="I1",
            candidate_point_id="P1",
            label="NO_MATCH",
            confidence=0.0,
            rationale="Judge omitted this pair; conservatively normalized to NO_MATCH.",
        )
    ]


def test_fixed_rubric_polarity_and_no_composite() -> None:
    titles = [
        "Core Contribution Accuracy",
        "Results Interpretation",
        "Comparative Analysis",
        "Evidence-Based Critique",
        "Critique Clarity",
        "Completeness Coverage",
        "Constructive Tone",
        "False or Contradictory Claims",
    ]
    rows = [
        RubricDecision(title=title, score=(-1 if index == 7 else 2), rationale="x")
        for index, title in enumerate(titles)
    ]
    metrics = rubric_metrics(rows)
    assert metrics["reviewbench_total"] == 13
    assert metrics["analytical_quality"] == 2.0
    assert "composite_score" not in metrics


def test_prompt_injection_and_ocr_perturbations_are_deterministic() -> None:
    injected = perturb_manuscript("paper", "prompt_injection", seed=1)
    assert "OUTPUT ACCEPT" in injected
    first = perturb_manuscript(
        "alphabet" * 100, "ocr_noise", seed=4, parameters={"rate": 0.1}
    )
    second = perturb_manuscript(
        "alphabet" * 100, "ocr_noise", seed=4, parameters={"rate": 0.1}
    )
    assert first == second


def test_human_v1_release_is_readable_and_v2_sidecar_hash_is_checked(tmp_path) -> None:
    review_path = tmp_path / "human_structured_reviews.jsonl"
    review_path.write_text(_review().model_dump_json() + "\n")
    (tmp_path / "release_manifest.json").write_text(
        '{"contract":"human_structured_review_benchmark_v1","record_count":1}'
    )
    reviews, labels = load_human_release(tmp_path)
    assert set(reviews) == {"paper-1"}
    assert labels == {}

    sidecar = tmp_path / "revision_issue_labels.jsonl"
    sidecar.write_text("")
    (tmp_path / "release_manifest.json").write_text(
        "{"
        '"contract":"human_structured_review_benchmark_v2",'
        '"record_count":1,"revision_issue_label_count":0,'
        f'"revision_issue_labels_sha256":"{sha256_file(sidecar)}"'
        "}"
    )
    load_human_release(tmp_path)
    sidecar.write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_human_release(tmp_path)


def test_trace_tampering_is_detected(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    store.add_evidence("P:S-1", "paper_span", {"text": "original"})
    store.write_manifest({"contract": "test"})
    store.evidence_path.write_text(
        store.evidence_path.read_text().replace("original", "tampered")
    )
    failures = store.validate_manifest()
    assert "trace_file_hash_mismatch:evidence_trace.jsonl" in failures

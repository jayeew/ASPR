from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from experiments.innovation_200.ablation_protocol import SUPPORTED_REPORT_SYSTEMS
from experiments.innovation_200.common import (
    experiment_config,
    run_stage,
    setup_stage_logging,
)
from experiments.innovation_200.compare_reports import comparison_tasks
from experiments.innovation_200.contracts import (
    SYSTEMS,
    HumanEvaluation,
    HumanMatch,
    HumanReference,
    HumanReferenceSet,
)
from experiments.innovation_200.evaluate_human import reviewer_consistency, score
from experiments.innovation_200.generate_reports import generate
from experiments.innovation_200.reporting import (
    _text_only,
    _without_metrics,
    _without_paths,
    build_system_context,
    report_markdown,
)
from experiments.innovation_200.sample_papers import JOURNAL_QUOTAS, allocate
from gear.artifacts import write_model
from gear.claim_graph.contracts import InnovationClaimType
from gear.contracts import PaperMetadata, ReviewRequest
from gear.innovation.contracts import ClaimSet
from gear.innovation.usage import log_progress, progress_logging, progress_scope
from gear.model_client import ROLE_MODELS, resolve_role_model
from gear.paper_compiler import PaperCompiler
from gear.review_contracts import (
    GearClaim,
    GraphClaim,
    GraphFactCard,
    GraphNeighbor,
    InternalSupportStatus,
    MetricFact,
    ReviewerStance,
)
from gear.trace import EvidenceStore


def test_all_experiment_roles_use_luna_with_planned_effort() -> None:
    config = experiment_config()
    resolved = {role: resolve_role_model(config, role) for role in ROLE_MODELS}
    assert {model for model, _ in resolved.values()} == {"gpt-5.6-luna"}
    low = {
        "field_classifier",
        "reference_extract",
        "reference_check",
        "graph_claim",
        "claim_miner",
        "claim_consolidator",
        "supervisor_planner",
    }
    assert all(resolved[role][1] == "low" for role in low)
    assert all(resolved[role][1] == "high" for role in set(ROLE_MODELS) - low)
    assert not config.model_cache_enabled
    assert not config.relation_stability_check_enabled
    assert not config.resume_fingerprint_checks_enabled
    assert config.retrieval.fulltext_max == 10
    assert config.retrieval.relation_cards_max == 10
    assert config.retrieval.retained_candidates_per_claim == 5


def test_stage_logging_writes_terminal_progress_and_log_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = setup_stage_logging(tmp_path, "synthetic", verbose=True)
    records = run_stage(
        [{"paper_id": "p1"}, {"paper_id": "p2"}],
        lambda row: {"skipped": row["paper_id"] == "p2"},
        workers=2,
        status_path=tmp_path / "status/synthetic.json",
        logger=logger,
    )
    with progress_logging(logger, "paper=p1"), progress_scope("claim=c1"):
        log_progress("[检索开始] family=normal")
    terminal = capsys.readouterr().out
    log_text = (tmp_path / "logs/synthetic.log").read_text(encoding="utf-8")
    assert "[开始] 任务=2" in terminal
    assert "[进度 2/2]" in terminal
    assert "跳过=1" in terminal
    assert "[任务详情]" in terminal
    assert "paper=p1 claim=c1 [检索开始] family=normal" in terminal
    assert "[完成] 总任务=2" in log_text
    assert len(records) == 2


def test_sampling_allocation_obeys_journal_quotas() -> None:
    rows = []
    for journal, quota in JOURNAL_QUOTAS.items():
        for field in ("A", "B"):
            rows.extend(
                {"journal_id": journal, "field_name": field} for _ in range(quota)
            )
    result = allocate(rows, ["A", "B"], {"A": 100, "B": 100})
    for journal, quota in JOURNAL_QUOTAS.items():
        assert sum(result[(journal, field)] for field in ("A", "B")) == quota
    assert sum(result.values()) == 200


def test_graph_ablation_views_remove_forbidden_information() -> None:
    cards = [
        {
            "claim": {"claim_id": "new", "claim_text": "new claim"},
            "neighbors": [
                {
                    "claim_id": "old",
                    "parent_paper_id": "paper-old",
                    "claim_type": "FINDING",
                    "claim_text": "old claim",
                    "publication_date": "2020-01-01",
                    "cosine_similarity": 0.8,
                    "community_id": 4,
                    "direct_citation": True,
                    "two_hop_path_count": 2,
                    "shared_reference_count": 3,
                    "shared_reference_salton": 0.5,
                    "parent_id_cache_hit": True,
                    "edge_type": "semantic_and_paper_path",
                }
            ],
            "neighbor_edges": [["old", "older"]],
            "metrics": [
                {"name": "component_merge_count", "value": 1},
                {"name": "citation_path_count", "value": 2},
            ],
            "community_ids": [4],
        }
    ]
    assert "metrics" not in _without_metrics(cards)[0]
    without_paths = json.dumps(_without_paths(cards), ensure_ascii=False)
    assert "direct_citation" not in without_paths
    assert "citation_path_count" not in without_paths
    text_only = json.dumps(_text_only(cards), ensure_ascii=False)
    for forbidden in ("cosine", "community", "neighbor_edges", "direct_citation"):
        assert forbidden not in text_only
    assert "old claim" in text_only


def test_all_seven_structured_report_conditions_have_separate_inputs(
    tmp_path: Path,
) -> None:
    claims = ClaimSet(paper_id="p", input_fingerprint="x", claims=[])
    for path, payload in (
        ("gear/analysis.json", {"lane": "gear"}),
        ("graph/analysis.json", {"lane": "graph"}),
        ("graph/joint/analysis.json", {"lane": "joint"}),
        ("graph/joint/facts.json", {"input_claims": [], "citation_path": 2}),
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
    contexts = {
        system: build_system_context(system, tmp_path, claims, inspect_legacy_variant=True)
        for system in SYSTEMS
        if system != "direct_llm"
    }
    assert set(contexts["gear"]) == {"claims", "gear"}
    assert set(contexts["graph"]) == {"claims", "graph", "joint_graph"}
    assert "joint_graph" in contexts["fusion"]
    assert "joint_graph" not in contexts["fusion_no_joint"]
    assert "graph" not in contexts["fusion_no_metrics"]
    assert "citation_path" not in json.dumps(contexts["fusion_no_citation_paths"])
    assert set(contexts["fusion_text_only"]) == {
        "claims",
        "gear",
        "historical_text",
    }


def _reference(
    reference_id: str,
    reviewer: str,
    round_number: int,
    stance: ReviewerStance,
) -> HumanReference:
    return HumanReference(
        reference_id=reference_id,
        paper_id="p",
        reviewer_id=reviewer,
        round_number=round_number,
        contribution_id="c",
        contribution_text="contribution",
        dimension="increment",
        stance=stance,
        reasons=["because"],
        source_quote="quote",
        source_context="context",
        tier="A",
    )


def test_human_metrics_keep_partial_separate_and_null_empty() -> None:
    refs = HumanReferenceSet(
        paper_id="p",
        references=[_reference("r1", "1", 1, ReviewerStance.RECOGNIZED)],
    )
    evaluation = HumanEvaluation(
        paper_id="p",
        system="fusion",
        matches=[
            HumanMatch(
                reference_id="r1",
                scope="partial",
                predicted_stance=ReviewerStance.RECOGNIZED,
                reason_coverage="partial",
                contradiction=False,
                rationale="overlap",
            )
        ],
    )
    metrics = score(evaluation, refs)
    assert metrics["coverage_full_rate"] == 0
    assert metrics["coverage_partial"] == 1
    assert metrics["stance_agreement_rate"] == 1
    empty = score(
        HumanEvaluation(paper_id="p", system="fusion", matches=[]),
        HumanReferenceSet(paper_id="p", references=[]),
    )
    assert empty["coverage_full_rate"] is None


def test_reviewer_conflict_and_cross_round_change_are_separate() -> None:
    refs = HumanReferenceSet(
        paper_id="p",
        references=[
            _reference("r1", "one", 1, ReviewerStance.RECOGNIZED),
            _reference("r2", "two", 1, ReviewerStance.CHALLENGED),
            _reference("r3", "one", 2, ReviewerStance.INCREMENTAL_OR_LIMITED),
        ],
    )
    result = reviewer_consistency(refs)
    assert result["comparisons"][0]["relation"] == "opposite"
    assert len(result["cross_round_changes"]) == 1


def test_pairwise_positions_are_exactly_balanced_for_200_papers() -> None:
    tasks = comparison_tasks([{"paper_id": str(index)} for index in range(200)])
    assert len(tasks) == 1400
    assert Counter(task["fusion_position"] for task in tasks) == {"A": 700, "B": 700}
    assert {task["baseline_system"] for task in tasks} == set(SYSTEMS) - {"fusion"}


def test_citation_appendix_contains_only_selected_sources() -> None:
    from experiments.innovation_200.contracts import ReportBundle, ReportSource

    report = ReportBundle(
        paper_id="p",
        system="fusion",
        body="分析正文。" * 200,
        references=[
            ReportSource(
                source_id="W:1:P01",
                passage_id="P01",
                source_type="fulltext",
                title="Prior work",
                year=2020,
                doi="10.1/test",
                passage="verbatim passage",
            )
        ],
    )
    text = report_markdown(report)
    assert "Prior work" in text
    assert "verbatim passage" in text
    assert "fulltext" in text


def test_mock_report_stage_handoff_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manuscript = tmp_path / "paper.md"
    manuscript.write_text(
        "# Abstract\nWe introduce method X. It improves result Y.\n\n# Results\nMethod X improves Y in the reported experiment.",
        encoding="utf-8",
    )
    paper = PaperCompiler().compile(
        ReviewRequest(
            paper_path=manuscript,
            metadata=PaperMetadata(
                title="Test paper",
                openalex_id="p",
                publication_date=date(2024, 1, 1),
            ),
        )
    )
    span_id = paper.spans[0].span_id
    claim = GearClaim(
        claim_id="p::CLAIM::01",
        claim_type=InnovationClaimType.METHOD,
        author_claim_text="Method X improves Y.",
        normalized_claim_text="Method X improves Y.",
        source_span_ids=[span_id],
        support_span_ids=[span_id],
        internal_support=InternalSupportStatus.SUPPORTED,
        narrowing_reason="",
    )
    claims = ClaimSet(paper_id="p", input_fingerprint="synthetic", claims=[claim])
    root = tmp_path / "papers/p"
    write_model(root / "shared/paper_ir.json", paper)
    write_model(root / "shared/claims.json", claims)
    for relative, payload in (
        ("gear/analysis.json", {"paper_id": "p", "lane": "gear"}),
        ("graph/analysis.json", {"paper_id": "p", "lane": "graph"}),
        ("graph/joint/analysis.json", {"paper_id": "p", "lane": "joint"}),
        (
            "graph/joint/facts.json",
            {"input_claims": [{"claim_id": claim.claim_id}], "historical_edges": []},
        ),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
    EvidenceStore(root / "gear/01").add_evidence(
        "WORK:old",
        "retrieved_work",
        {
            "work_id": "old",
            "title": "Historical work",
            "publication_year": 2020,
            "doi": "10.1/old",
            "abstract": "Historical abstract passage.",
            "spans": [],
        },
    )
    fulltext_key = "FULLTEXT:p::CLAIM::01:old"
    EvidenceStore(root / "gear/01").add_evidence(
        fulltext_key,
        "retrieved_work_fulltext",
        {
            "work_id": "old",
            "title": "Historical work",
            "publication_year": 2020,
            "doi": "10.1/old",
            "url": "https://doi.org/10.1/old",
            "spans": [
                {
                    "span_id": "full-p1",
                    "text": "Historical fulltext methods passage.",
                    "source": "fulltext_evidence",
                }
            ],
            "fulltext_provenance": {
                "source_url": "https://repository.example/paper.pdf"
            },
        },
    )
    EvidenceStore(root / "gear/01").add_evidence(
        "FULLTEXT_FETCH:p::CLAIM::01:old",
        "fulltext_acquisition",
        {
            "status": "success",
            "text": "Raw whole document must not become a report passage.",
        },
    )
    card = GraphFactCard(
        insertion_policy="top10_strict_gt_0.5_parent_path_v1",
        claim=GraphClaim(
            claim_id=claim.claim_id,
            paper_id="p",
            claim_type=InnovationClaimType.METHOD,
            claim_text=claim.normalized_claim_text,
            source_sentence_ids=[span_id],
            source_sentence_texts=[paper.spans[0].text],
        ),
        neighbors=[
            GraphNeighbor(
                claim_id="old-claim",
                parent_paper_id="old-paper",
                claim_type=InnovationClaimType.METHOD,
                claim_text="Historical graph claim.",
                publication_date=date(2020, 1, 1),
                cosine_similarity=0.8,
                semantic_rank=1,
            )
        ],
        metrics=[MetricFact(name="component_merge_count", value=0)],
    )
    EvidenceStore(root / "graph/01").add_evidence(
        f"GRAPH:{claim.claim_id}", "graph_fact", card
    )

    seen_catalogs: list[list[dict]] = []

    def fake_generate(
        _self: object,
        *,
        system: str,
        user: str,
        response_schema: dict | None = None,
    ) -> dict:
        del system, response_schema
        cited = []
        catalog = []
        if user.startswith("{"):
            catalog = json.loads(user).get("source_catalog", [])
            external = [
                source["source_id"]
                for source in catalog
                if source["source_type"] != "manuscript"
            ]
            fulltext = [
                source_id for source_id in external if source_id.startswith("FULLTEXT:")
            ]
            cited = (fulltext or external)[:1]
        seen_catalogs.append(catalog)
        return {"body": "合成分析。" * 400, "cited_source_ids": cited}

    monkeypatch.setattr(
        "experiments.innovation_200.reporting.LazyRoleClient.generate_json",
        fake_generate,
    )
    assert generate({"paper_id": "p"}, tmp_path, False)["generated"] == len(SUPPORTED_REPORT_SYSTEMS)
    assert generate({"paper_id": "p"}, tmp_path, False)["generated"] == 0
    assert len(list((tmp_path / "reports").glob("*/*.json"))) == len(SUPPORTED_REPORT_SYSTEMS)
    direct = json.loads((tmp_path / "reports/direct_llm/p.json").read_text())
    fusion = json.loads((tmp_path / "reports/fusion/p.json").read_text())
    assert direct["references"] == []
    assert fusion["references"][0]["passage"] == "Historical fulltext methods passage."
    assert fusion["references"][0]["source_type"] == "fulltext"
    assert fusion["references"][0]["url"] == "https://repository.example/paper.pdf"
    catalogs = dict(zip(SUPPORTED_REPORT_SYSTEMS, seen_catalogs))
    assert catalogs["direct_llm"] == []
    for system, catalog in catalogs.items():
        ids = [source["source_id"] for source in catalog]
        assert not any(key.startswith("FULLTEXT_FETCH:") for key in ids)
        if system in ("graph", "direct_llm"):
            assert not any(key.startswith(("FULLTEXT:", "WORK:")) for key in ids)
        else:
            assert f"{fulltext_key}:P01" in ids
            assert any(source["source_type"] == "abstract" for source in catalog)


def test_fulltext_source_uses_acquisition_url_without_relabeling_abstract() -> None:
    from experiments.innovation_200.reporting import _source_from_work

    sources = _source_from_work(
        "FULLTEXT:old",
        {
            "title": "Historical work",
            "url": "https://doi.org/10.1/old",
            "fulltext_provenance": {
                "source_url": "https://repository.example/paper.xml"
            },
            "spans": [
                {
                    "span_id": "a",
                    "text": "Abstract evidence",
                    "source": "abstract_evidence",
                },
                {
                    "span_id": "f",
                    "text": "Methods evidence",
                    "source": "fulltext_evidence",
                },
            ],
        },
        [],
    )
    assert [source.source_type for source in sources] == ["abstract", "fulltext"]
    assert [source.url for source in sources] == [
        "https://doi.org/10.1/old",
        "https://repository.example/paper.xml",
    ]

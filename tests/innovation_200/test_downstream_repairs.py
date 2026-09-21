from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.innovation_200 import ablation_interpretation as module
from experiments.innovation_200.ablation_protocol import MASKED_GRAPH_VARIANTS
from experiments.innovation_200.blinding import evaluation_root
from experiments.innovation_200.common import write_json
from experiments.innovation_200.summarize_results import human_rows, output_completeness
from gear.innovation.contracts import Assessment, ClaimSet, Finding
from gear.innovation.joint_graph import JointAnalysis, JointFinding
from gear.trace import EvidenceStore
from tests.innovation_200.test_gear_recovery import _claim


def test_summary_selects_one_current_condition_without_legacy_fill(
    tmp_path: Path,
) -> None:
    papers = [dict(paper_id=p, field_name="A", journal_name="J") for p in ("a", "b")]
    for paper in papers:
        write_json(
            tmp_path / "human_evaluation/graph" / f"{paper['paper_id']}.json",
            {"metrics": {"coverage_full_rate": 0.2, "stance_by_dimension": {}}},
        )
    assert len(human_rows(tmp_path, papers)) == 2
    write_json(
        evaluation_root(tmp_path) / "human/graph/a.json",
        {"metrics": {"coverage_full_rate": 0.9, "stance_by_dimension": {}}},
    )
    rows = human_rows(tmp_path, papers)
    assert len(rows) == 1 and rows[0]["coverage_full_rate"] == 0.9
    complete = output_completeness(tmp_path, papers)
    assert complete[0]["human_eval_graph"]
    assert not complete[1]["human_eval_graph"]


@pytest.mark.parametrize("system", sorted(MASKED_GRAPH_VARIANTS))
def test_ablation_masks_then_interprets_and_resumes_per_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
) -> None:
    claim = _claim("paper", "01")
    empty = _claim("paper", "02")
    claims = ClaimSet(paper_id="paper", input_fingerprint="", claims=[claim, empty])
    card = {
        "claim": {"claim_id": claim.claim_id, "claim_text": "test"},
        "neighbors": [
            {
                "claim_id": "old",
                "claim_text": "earlier result",
                "cosine_similarity": 0.7,
                "direct_citation": True,
            }
        ],
        "metrics": [
            {"name": "citation_path_count", "value": 3},
            {"name": "local_degree", "value": 1},
        ],
        "notes": ["secret path and metric prose"],
        "neighbor_edges": [],
        "community_ids": [5],
    }
    store = EvidenceStore(tmp_path / "graph/joint")
    store.add_evidence(f"GRAPH:{claim.claim_id}", "graph_fact", card)
    store.add_evidence(
        f"GRAPH:{empty.claim_id}",
        "graph_fact",
        {"claim": {"claim_id": empty.claim_id}, "neighbors": []},
    )
    store.add_evidence(
        "JOINT_GRAPH:paper",
        "joint_graph_fact",
        {
            "input_claims": [],
            "historical_edges": [],
            "insertion_edges": [],
            "joint_component_merge_count": 3,
            "citation_path": 2,
            "notes": ["secret path and metric prose"],
        },
    )
    write_json(tmp_path / "gear/analysis.json", {"assessments": []})
    write_json(tmp_path / "graph/analysis.json", {"leak": "FULL INTERPRETATION"})
    calls = []

    def assess(config, cid, text, sources, mode):
        calls.append(("claim", sources))
        assert "secret path" not in json.dumps(sources)
        return Assessment(
            claim_id=cid,
            claim_text=text,
            supported_scope=text,
            findings=[
                Finding(
                    dimension="knowledge_relation",
                    text="bounded",
                    evidence_keys=list(sources),
                )
            ],
            overall_stance="unresolved",
            overall_reason="bounded",
            limitations=[],
        )

    def joint(config, pid, sources, ids):
        calls.append(("joint", sources))
        if sum(c[0] == "joint" for c in calls) == 1:
            raise RuntimeError("interrupted joint")
        return JointAnalysis(
            paper_id=pid,
            knowledge_summary="bounded",
            limitations=[],
            findings=[
                JointFinding(
                    question="basis",
                    claim_ids=[claim.claim_id],
                    observation="observed",
                    interpretation="bounded",
                    evidence_keys=list(sources),
                    limitations=[],
                )
            ],
        )

    monkeypatch.setattr(module, "assess", assess)
    monkeypatch.setattr(module, "analyze_joint", joint)
    monkeypatch.setattr(module, "experiment_config", lambda: None)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.interpreted_context(tmp_path, claims, system)
    result = module.interpreted_context(tmp_path, claims, system)
    assert sum(c[0] == "claim" for c in calls) == 1
    assert sum(c[0] == "joint" for c in calls) == 2
    assert "FULL INTERPRETATION" not in json.dumps(result)
    assert len(result["graph"]["assessments"]) == 1
    assert empty.claim_id in result["graph"]["limitations"][0]
    assert module.interpreted_context(tmp_path, claims, system) == result
    assert len(calls) == 3
    seen = json.dumps(calls[-1][1])
    if system == "fusion_no_metrics":
        assert '"metrics"' not in seen and "joint_component_merge_count" not in seen
    if system == "fusion_no_citation_paths":
        assert "direct_citation" not in seen and "citation_path" not in seen
    if system == "fusion_text_only":
        for token in (
            "cosine",
            "community",
            "neighbor_edges",
            "insertion_edges",
            "direct_citation",
        ):
            assert token not in seen


def test_summary_ignores_auxiliary_judge_requests(tmp_path: Path) -> None:
    from experiments.innovation_200.blinding import summary_evaluation_root
    from experiments.innovation_200.summarize_results import pairwise_rows

    current = evaluation_root(tmp_path)
    write_json(current / "pairwise/p__fusion_vs_graph.request.json", {"payload": {}})
    write_json(current / "pairwise/p__fusion_vs_graph.mapping.json", {"reports": {}})
    assert summary_evaluation_root(tmp_path) == tmp_path
    write_json(
        current / "pairwise/p__fusion_vs_graph.json",
        {"paper_id": "p", "baseline_system": "graph", "winners": {"overall": "tie"}},
    )
    assert len(pairwise_rows(tmp_path)) == 1
    rows = output_completeness(tmp_path, [{"paper_id": "p", "field_name": "A"}])
    assert rows[0]["pairwise_complete"] == 1

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from aspr.graph_innovation_scorer import GraphInnovationScorer
from aspr.review_committee import run_reviewer_committee


def test_empty_related_papers_returns_conservative_tone() -> None:
    graph_result = GraphInnovationScorer().score("Test", "We propose a new method.", []).to_dict()
    report = run_reviewer_committee(
        paper_title="Test",
        paper_abstract="We propose a new method.",
        related_papers=[],
        graph_metric_result=graph_result,
    )
    assert report.recommended_tone == "conservative"
    assert report.disagreement_score > 0.4
    assert report.claim_cards


def test_graph_analyst_mentions_boundary_and_atypicality() -> None:
    graph_result = {
        "metrics": {
            "DeltaQ0": 0.82,
            "Uzzi": 0.74,
            "RS": 0.25,
            "PDE": 0.10,
            "B": 0.05,
            "RTD": 0.05,
            "BurtIP": 0.05,
        },
        "weighted_score": 0.72,
        "confidence": 0.82,
    }
    report = run_reviewer_committee(
        paper_title="Graph perturbation",
        paper_abstract="We propose a graph learning method for molecular prediction.",
        related_papers=[
            {
                "title": "Graph learning method for molecular prediction",
                "abstract": "Graph learning for molecular property prediction.",
                "fieldsOfStudy": ["Computer Science"],
            }
        ],
        graph_metric_result=graph_result,
    )
    graph_text = report.claim_cards[0].graph_support
    assert "DeltaQ0" in graph_text
    assert "Uzzi" in graph_text
    assert "边界扰动" in graph_text
    assert "非典型组合" in graph_text


def test_skeptic_flags_overclaim_without_references() -> None:
    report = run_reviewer_committee(
        paper_title="Breakthrough method",
        paper_abstract="We propose a breakthrough method that changes the paradigm.",
        related_papers=[],
        graph_metric_result={"metrics": {}, "weighted_score": 0.0, "confidence": 0.0},
    )
    counterarguments = " ".join(report.claim_cards[0].counterarguments)
    assert "强创新措辞" in counterarguments or "缺少直接相关参考文献" in counterarguments
    assert report.recommended_tone == "conservative"


def test_high_disagreement_forces_conservative_tone() -> None:
    report = run_reviewer_committee(
        paper_title="Sparse evidence",
        paper_abstract="We introduce a new framework.",
        related_papers=[],
        graph_metric_result={"metrics": {"DeltaQ0": 0.9}, "weighted_score": 0.8, "confidence": 0.1},
    )
    assert report.disagreement_score >= 0.45
    assert report.recommended_tone == "conservative"


if __name__ == "__main__":
    test_empty_related_papers_returns_conservative_tone()
    test_graph_analyst_mentions_boundary_and_atypicality()
    test_skeptic_flags_overclaim_without_references()
    test_high_disagreement_forces_conservative_tone()
    print("review_committee tests passed")

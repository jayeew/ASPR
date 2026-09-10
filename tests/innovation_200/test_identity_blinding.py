from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.innovation_200 import compare_reports, evaluate_human
from experiments.innovation_200.ablation_protocol import (
    MIXED_PATH_VARIANTS,
    UncontrolledAblationError,
    ablation_plan,
)
from experiments.innovation_200.blinding import blind_reports, evaluation_root
from experiments.innovation_200.common import write_json
from experiments.innovation_200.contracts import (
    HumanReference,
    HumanReferenceSet,
    ReportBundle,
    ReportSource,
)
from experiments.innovation_200.reporting import build_system_context, generate_report
from gear.innovation.contracts import ClaimSet


def _report(system: str, source_id: str = "G:historic::CLAIM::01") -> ReportBundle:
    return ReportBundle(
        paper_id="paper",
        system=system,
        body=f"Analysis [{source_id}]. " + "x" * 1100,
        references=[
            ReportSource(
                source_id=source_id,
                passage_id="historic::CLAIM::01",
                source_type="historical_claim",
                title="Real title",
                doi="10.123/real",
                passage="Verbatim original Graph study passage.",
            )
        ],
    )


def test_neutral_sources_share_alias_without_losing_evidence() -> None:
    report_a, report_b = _report("graph"), _report("fusion", "WORK:external:P01")
    visible, external = blind_reports({"report_A": report_a, "report_B": report_b})
    assert visible["report_A"] == visible["report_B"]
    for report in visible.values():
        assert set(report) == {"body", "references"}
        assert "G:historic" not in json.dumps(report)
        assert "WORK:external" not in json.dumps(report)
        assert report["references"][0]["passage"] == report_a.references[0].passage
        assert report["references"][0]["doi"] == "10.123/real"
    assert external["reports"]["report_A"]["system"] == "graph"
    assert external["reports"]["report_A"]["source_mapping"][0]["source_id"].startswith(
        "G:"
    )


def test_source_id_replacement_does_not_rewrite_larger_unknown_id() -> None:
    report = _report("graph", "G:1")
    report.body = "[G:1] [G:10] " + "x" * 1100
    visible, _ = blind_reports({"report": report})
    assert visible["report"]["body"].startswith("[S0001] [G:10]")


def test_human_judge_has_no_producer_metadata_or_schema_and_preserves_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = HumanReference(
        reference_id="HR1",
        paper_id="paper",
        reviewer_id="R1",
        round_number=1,
        contribution_text="Specific contribution",
        dimension="identification",
        source_quote="Reviewer original",
        source_context="Context",
        tier="B",
    )
    write_json(
        tmp_path / "human_refs/paper.json",
        HumanReferenceSet(paper_id="paper", references=[reference]),
    )
    write_json(tmp_path / "reports/graph/paper.json", _report("graph"))
    legacy = tmp_path / "human_evaluation/graph/paper.json"
    write_json(legacy, {"legacy": "keep"})
    captured = {}

    class Judge:
        def __init__(self, *args: object) -> None:
            pass

        def generate_json(self, **kwargs: object) -> dict:
            captured.update(kwargs)
            return {
                "matches": [
                    {
                        "reference_id": "HR1",
                        "scope": "partial",
                        "predicted_stance": None,
                        "reason_coverage": "missing",
                        "contradiction": True,
                        "rationale": "Partial overlap",
                    }
                ]
            }

    monkeypatch.setattr(evaluate_human, "LazyRoleClient", Judge)
    monkeypatch.setattr(evaluate_human, "experiment_config", lambda: None)
    monkeypatch.setattr(evaluate_human, "wait_for_memory", lambda logger: None)
    evaluate_human.evaluate({"paper_id": "paper", "system": "graph"}, tmp_path, False)
    payload = json.loads(captured["user"])
    assert set(payload["report"]) == {"body", "references"}
    assert set(captured["response_schema"]["properties"]) == {"matches"}
    assert '"graph"' not in json.dumps(captured["response_schema"])
    assert json.loads(legacy.read_text()) == {"legacy": "keep"}
    output = evaluation_root(tmp_path) / "human/graph/paper.json"
    saved = json.loads(output.read_text())
    assert saved["system"] == "graph"
    assert saved["metrics"]["contradiction_rate"] == 0
    assert output.with_suffix(".mapping.json").exists()
    assert output.with_suffix(".request.json").exists()


def test_pairwise_judge_receives_neutral_reports_and_keeps_external_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_json(tmp_path / "reports/fusion/paper.json", _report("fusion"))
    write_json(tmp_path / "reports/graph/paper.json", _report("graph", "WORK:a:P1"))
    write_json(
        tmp_path / "papers/paper/shared/paper_ir.json",
        {"markdown": "Actual manuscript"},
    )
    captured = {}

    class Judge:
        def __init__(self, *args: object) -> None:
            pass

        def generate_json(self, **kwargs: object) -> dict:
            captured.update(kwargs)
            return {
                **{
                    key: "A"
                    for key in (
                        "overall",
                        "contribution_accuracy",
                        "historical_comparison",
                        "knowledge_explanation",
                        "evidence_uncertainty",
                        "practical_value",
                    )
                },
                "rationale": "Comparison",
            }

    monkeypatch.setattr(compare_reports, "LazyRoleClient", Judge)
    monkeypatch.setattr(compare_reports, "experiment_config", lambda: None)
    compare_reports.compare(
        {"paper_id": "paper", "baseline_system": "graph", "fusion_position": "B"},
        tmp_path,
        False,
    )
    payload = json.loads(captured["user"])
    assert payload["report_A"] == payload["report_B"]
    assert all(
        "system" not in report
        for key, report in payload.items()
        if key.startswith("report")
    )
    saved = evaluation_root(tmp_path) / "pairwise/paper__fusion_vs_graph.json"
    assert json.loads(saved.read_text())["winners"]["overall"] == "graph"
    assert not (tmp_path / "pairwise").exists()


@pytest.mark.parametrize("system", sorted(MIXED_PATH_VARIANTS))
def test_confounded_ablation_is_blocked_before_reading_or_model_calls(
    tmp_path: Path, system: str
) -> None:
    claims = ClaimSet(paper_id="paper", input_fingerprint="x", claims=[])
    with pytest.raises(UncontrolledAblationError, match="mixes interpreted"):
        build_system_context(system, tmp_path, claims)
    with pytest.raises(UncontrolledAblationError):
        generate_report("paper", system, tmp_path)
    assert ablation_plan(system)["generation_blocked"]
    assert not ablation_plan(system)["causal_effect_ready"]

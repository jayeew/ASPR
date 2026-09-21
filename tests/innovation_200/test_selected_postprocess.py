from __future__ import annotations

from pathlib import Path

import pytest

from experiments.innovation_200 import evaluate_human, generate_reports, resource_guard
from experiments.innovation_200.common import write_json
from experiments.innovation_200.contracts import HumanReferenceSet, ReportBundle


@pytest.mark.parametrize("system", ["direct_llm", "graph"])
def test_selected_report_does_not_wait_for_gear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    waited: list[Path] = []
    monkeypatch.setattr(
        generate_reports,
        "wait_for_inputs",
        lambda paths, **kwargs: waited.extend(paths),
    )
    monkeypatch.setattr(generate_reports, "wait_for_memory", lambda logger: None)
    monkeypatch.setattr(
        generate_reports,
        "generate_report",
        lambda paper_id, name, root: ReportBundle(
            paper_id=paper_id, system=name, body="文" * 1200
        ),
    )
    result = generate_reports.generate(
        {"paper_id": "p", "system": system}, tmp_path, False, True
    )
    assert result["generated"] == 1
    assert all("gear" not in p.parts for p in waited)
    assert len(waited) == (2 if system == "direct_llm" else 4)
    assert list((tmp_path / "reports").glob("*/*.json")) == [
        tmp_path / "reports" / system / "p.json"
    ]


def test_evaluation_waits_only_for_selected_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_json(
        tmp_path / "human_refs/p.json", HumanReferenceSet(paper_id="p", references=[])
    )
    write_json(
        tmp_path / "reports/graph/p.json",
        ReportBundle(paper_id="p", system="graph", body="文" * 1200),
    )
    waited: list[Path] = []
    monkeypatch.setattr(
        evaluate_human, "wait_for_inputs", lambda paths, **kwargs: waited.extend(paths)
    )
    result = evaluate_human.evaluate(
        {"paper_id": "p", "system": "graph", "task_id": "p__graph"},
        tmp_path,
        False,
        True,
    )
    assert result["evaluated"] == 1
    assert waited == [tmp_path / "human_refs/p.json", tmp_path / "reports/graph/p.json"]
    assert not (tmp_path / "human_evaluation/gear").exists()


def test_memory_guard_waits_until_reserve_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(["MemAvailable: 1048576 kB", "MemAvailable: 6291456 kB"])
    sleeps = []
    monkeypatch.setenv("GEAR_POSTPROCESS_MIN_AVAILABLE_GIB", "5")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: next(values))
    monkeypatch.setattr(
        resource_guard.time, "sleep", lambda seconds: sleeps.append(seconds)
    )
    resource_guard.wait_for_memory()
    assert sleeps == [2]


def test_report_capacity_retry_preserves_other_errors(monkeypatch, tmp_path) -> None:
    from experiments.innovation_200 import generate_reports
    from gear.codex_cli import CodexCLIUnavailableError
    import pytest

    calls = []
    delays = []
    sentinel = object()

    def generate(*args):
        calls.append(args)
        if len(calls) == 1:
            raise CodexCLIUnavailableError('ERROR: Selected model is at capacity.')
        return sentinel

    monkeypatch.setattr(generate_reports, 'generate_report', generate)
    monkeypatch.setattr(generate_reports.time, 'sleep', delays.append)
    assert generate_reports.generate_with_capacity_retry('p', 'gear', tmp_path, None) is sentinel
    assert len(calls) == 2 and delays == [15]

    def blocked(*args):
        raise CodexCLIUnavailableError('ERROR: This content was flagged for possible biological risk.')

    monkeypatch.setattr(generate_reports, 'generate_report', blocked)
    with pytest.raises(CodexCLIUnavailableError, match='biological'):
        generate_reports.generate_with_capacity_retry('p', 'gear', tmp_path, None)
    assert delays == [15]

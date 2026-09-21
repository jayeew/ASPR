from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.innovation_200 import (
    common,
    compare_reports,
    evaluate_human,
    generate_reports,
)


def test_report_failure_is_detected_by_exact_missing_dependency(tmp_path: Path) -> None:
    status = tmp_path / "status/generate_reports.json"
    common.write_json(status, [{"paper_id": "p__fusion", "status": "failed"}])
    with pytest.raises(RuntimeError, match="Upstream"):
        common.wait_for_inputs(
            [tmp_path / "reports/fusion/p.json", tmp_path / "reports/graph/p.json"],
            paper_id="p",
            producer_statuses=[status],
        )


def test_unrelated_report_failure_does_not_abort_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = tmp_path / "status/generate_reports.json"
    common.write_json(status, [{"paper_id": "p__gear", "status": "failed"}])
    report = tmp_path / "reports/graph/p.json"
    monkeypatch.setattr(common.time, "sleep", lambda _: common.write_json(report, {}))
    common.wait_for_inputs([report], paper_id="p__graph", producer_statuses=[status])


def test_atomic_write_preserves_previous_result_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.json"
    common.write_json(target, {"old": True})

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        common.write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [target]


def test_stage_replaces_old_failure_before_workers_start(tmp_path: Path) -> None:
    status = tmp_path / "stage.json"
    common.write_json(status, [{"paper_id": "p", "status": "failed"}])

    def worker(row):
        assert json.loads(status.read_text())[0]["status"] == "pending"
        return {"ok": True}

    records = common.run_stage(
        [{"paper_id": "p"}], worker, workers=1, status_path=status
    )
    assert records[0]["status"] == "complete"


@pytest.mark.parametrize("module", [generate_reports, evaluate_human, compare_reports])
def test_postprocess_cli_exits_nonzero_on_task_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    monkeypatch.setattr("sys.argv", [module.__name__, "--study", str(tmp_path)])
    monkeypatch.setattr(module, "read_jsonl", lambda _: [])
    monkeypatch.setattr(module, "configure_limits", lambda _: None)
    monkeypatch.setattr(
        module, "run_stage", lambda *args, **kwargs: [{"status": "failed"}]
    )
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1

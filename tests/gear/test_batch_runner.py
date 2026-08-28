from __future__ import annotations

import json
import threading
import time

from gear.cli import _case_lock, _hang_diagnostic, main


def test_batch_records_case_failures_and_continues(tmp_path) -> None:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "missing-a", "paper_path": str(tmp_path / "a.md")},
                    {"case_id": "missing-b", "paper_path": str(tmp_path / "b.md")},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    status = main(
        [
            "benchmark",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ]
    )

    rows = [
        json.loads(line)
        for line in (output / "batch_results.jsonl").read_text().splitlines()
    ]
    assert status == 0
    assert [row["case_id"] for row in rows] == ["missing-a", "missing-b"]
    assert all(row["status"] == "failed" for row in rows)


def test_batch_strict_exit_reports_failed_cases(tmp_path) -> None:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {"cases": [{"case_id": "missing", "paper_path": str(tmp_path / "x.md")}]}
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "benchmark",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "output"),
                "--strict-exit",
            ]
        )
        == 1
    )


def test_batch_report_dir_isolated_from_shared_case_outputs(tmp_path) -> None:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {"cases": [{"case_id": "missing", "paper_path": str(tmp_path / "x.md")}]}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = tmp_path / "reports" / "shard-00"

    assert (
        main(
            [
                "benchmark",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--batch-report-dir",
                str(report),
            ]
        )
        == 0
    )

    assert (report / "batch_results.jsonl").is_file()
    assert (report / "batch_summary.json").is_file()
    assert not (output / "batch_results.jsonl").exists()


def test_case_lock_serializes_same_case_across_workers(tmp_path) -> None:
    acquired_first = threading.Event()
    release_first = threading.Event()
    acquired_second = threading.Event()

    def first() -> None:
        with _case_lock(tmp_path, "case-a"):
            acquired_first.set()
            release_first.wait(timeout=2)

    def second() -> None:
        acquired_first.wait(timeout=2)
        with _case_lock(tmp_path, "case-a"):
            acquired_second.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert acquired_first.wait(timeout=2)
    time.sleep(0.05)
    assert not acquired_second.is_set()
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert acquired_second.is_set()


def test_hang_diagnostic_marks_a_slow_normal_completion(monkeypatch, tmp_path) -> None:
    def dump_traceback_later(_timeout, *, repeat, file) -> None:
        assert repeat is True
        file.write("Timeout diagnostic\n")
        file.flush()

    monkeypatch.setattr(
        "gear.cli.faulthandler.dump_traceback_later", dump_traceback_later
    )
    monkeypatch.setattr(
        "gear.cli.faulthandler.cancel_dump_traceback_later", lambda: None
    )

    with _hang_diagnostic(tmp_path):
        pass

    diagnostic = (tmp_path / "hang_diagnostic.log").read_text(encoding="utf-8")
    assert "Timeout diagnostic" in diagnostic
    assert "GEAR_DIAGNOSTIC_RESOLUTION" in diagnostic

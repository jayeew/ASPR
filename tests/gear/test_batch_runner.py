from __future__ import annotations

import json

from gear.cli import main


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

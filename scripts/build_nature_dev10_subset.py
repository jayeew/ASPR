#!/usr/bin/env python3
"""Build a deterministic 10-paper Nature development subset.

Selection uses only ASPR score strata and the human novelty direction.  It never
reads GEAR outputs or evaluation scores, which keeps the subset useful for fast
engineering tests without selecting cases for favorable system performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

TARGET_DIRECTIONS = (
    "positive",
    "mixed",
    "positive",
    "positive",
    "mixed",
    "positive",
    "mixed",
    "positive",
    "mixed",
    "positive",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(text)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _select_cases(
    cases: list[dict[str, Any]], human_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    ordered = sorted(
        cases, key=lambda row: (row["graph_result"]["score_0_100"], row["paper_id"])
    )
    if len(ordered) < 10:
        raise ValueError("at least 10 cases are required")
    strata = [
        ordered[(i * len(ordered)) // 10 : ((i + 1) * len(ordered)) // 10]
        for i in range(10)
    ]
    selected: list[dict[str, Any]] = []
    for index, (stratum, target) in enumerate(
        zip(strata, TARGET_DIRECTIONS, strict=True)
    ):
        midpoint = sum(row["graph_result"]["score_0_100"] for row in stratum) / len(
            stratum
        )
        matching = [
            row
            for row in stratum
            if human_by_id[row["paper_id"]]["novelty"]["judgment"] == target
        ]
        pool = matching or stratum
        choice = min(
            pool,
            key=lambda row: (
                abs(row["graph_result"]["score_0_100"] - midpoint),
                row["paper_id"],
            ),
        )
        choice = dict(choice)
        choice["selection_stratum"] = index + 1
        choice["selection_target_direction"] = target
        selected.append(choice)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-setup-dir", type=Path, required=True)
    parser.add_argument("--human-release-dir", type=Path, required=True)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_setup_dir.resolve()
    human = args.human_release_dir.resolve()
    scoring = args.scoring_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    evaluation = json.loads((source / "evaluation_manifest.json").read_text())
    batch = json.loads((source / "batch_manifest.json").read_text())
    human_rows = _read_jsonl(human / "human_structured_reviews.jsonl")
    human_by_id = {row["paper_id"]: row for row in human_rows}
    scoring_packets = {
        row["paper_id"]: row
        for row in _read_jsonl(scoring / "graph_runtime_packets.jsonl")
    }
    selected = _select_cases(evaluation["cases"], human_by_id)
    selected_ids = {row["paper_id"] for row in selected}
    selected_case_ids = {row["case_id"] for row in selected}

    metadata_dir = output / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    eval_cases: list[dict[str, Any]] = []
    for row in selected:
        clean = {
            key: value for key, value in row.items() if not key.startswith("selection_")
        }
        if clean["paper_id"] not in scoring_packets:
            raise ValueError(f"missing Graph packet for {clean['paper_id']}")
        clean["graph_result"] = scoring_packets[clean["paper_id"]]
        metadata_source = Path(clean["metadata_path"])
        metadata_target = metadata_dir / metadata_source.name
        shutil.copyfile(metadata_source, metadata_target)
        clean["metadata_path"] = str(metadata_target)
        clean["clean_run_dir"] = None
        eval_cases.append(clean)

    evaluation.update(
        dataset_id="nature_dev10_graph_guidance_v2",
        cases=eval_cases,
        bootstrap_samples=5000,
        seed=20260824,
    )
    filtered_batch = [
        row for row in batch["cases"] if row["case_id"] in selected_case_ids
    ]
    runtime = json.loads((source / "runtime_config.json").read_text())
    runtime.setdefault("retrieval", {})["citation_expansion_max"] = 1
    packet_rows = [row["graph_result"] for row in eval_cases]
    packet_path = output / "graph_runtime_packets.jsonl"
    _write_jsonl(packet_path, packet_rows)
    runtime["graph_results_path"] = str(packet_path)

    selected_human = [row for row in human_rows if row["paper_id"] in selected_ids]
    revision = [
        row
        for row in _read_jsonl(human / "revision_issue_labels.jsonl")
        if row["paper_id"] in selected_ids
    ]
    scoring_sources = [
        row
        for row in json.loads((scoring / "source_manifest.json").read_text())
        if row["openalex_id"] in selected_ids
    ]
    human_sources = [
        row
        for row in json.loads((human / "source_manifest.json").read_text())
        if row["paper_id"] in selected_ids
    ]

    _write_json(output / "evaluation_manifest.json", evaluation)
    _write_json(output / "batch_manifest.json", {"cases": filtered_batch})
    _write_json(output / "runtime_config.json", runtime)
    _write_jsonl(output / "human_structured_reviews.jsonl", selected_human)
    _write_jsonl(output / "revision_issue_labels.jsonl", revision)
    _write_json(output / "graph_source_manifest.json", scoring_sources)
    _write_json(output / "human_source_manifest.json", human_sources)

    selection_rows = []
    for row in selected:
        packet = row["graph_result"]
        selection_rows.append(
            {
                "case_id": row["case_id"],
                "paper_id": row["paper_id"],
                "aspr_score": packet["score_0_100"],
                "feature_coverage": max(
                    0.0,
                    1.0 - len(set(packet.get("missing_feature_ids", []))) / 16.0,
                ),
                "missing_feature_ids": packet.get("missing_feature_ids", []),
                "human_novelty_direction": human_by_id[row["paper_id"]]["novelty"][
                    "judgment"
                ],
                "score_stratum": row["selection_stratum"],
                "target_direction": row["selection_target_direction"],
            }
        )
    selection_manifest = {
        "contract": "nature_dev10_subset_v1",
        "source_dataset_id": "nature_dev100_v1",
        "dataset_id": "nature_dev10_graph_guidance_v2",
        "development_non_confirmatory": True,
        "selection_policy": "one paper per rank-based ASPR decile; target 6 positive and 4 mixed human novelty judgments; closest to stratum mean; no GEAR outputs used",
        "records": selection_rows,
        "artifacts": {
            name: _sha256(output / name)
            for name in (
                "evaluation_manifest.json",
                "batch_manifest.json",
                "graph_runtime_packets.jsonl",
                "human_structured_reviews.jsonl",
                "revision_issue_labels.jsonl",
                "graph_source_manifest.json",
                "human_source_manifest.json",
            )
        },
    }
    _write_json(output / "selection_manifest.json", selection_manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

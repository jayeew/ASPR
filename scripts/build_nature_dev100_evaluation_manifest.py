#!/usr/bin/env python3
"""Build frozen runtime and evaluation manifests for Nature dev100."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gear.evaluation.contracts import EvaluationManifestV1
from gear.contracts import PaperMetadata
from gear.graph_prior_contracts import GraphResultV4

RECEIVED = re.compile(r"\bReceived:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b")


def _submission_date(text: str) -> str:
    match = RECEIVED.search(text[:20_000])
    if match is None:
        raise ValueError("manuscript lacks a Received date")
    parsed = time.strptime(match.group(1), "%d %B %Y")
    return date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday).isoformat()


def _title(text: str) -> str:
    headings = re.findall(r"^##\s+(.+?)\s*$", text[:20_000], flags=re.MULTILINE)
    for heading in headings:
        clean = heading.strip()
        if clean.casefold() != "article":
            return clean
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--human-release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scoring = args.scoring_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    sources = json.loads((scoring / "source_manifest.json").read_text())
    graphs = {
        row.paper_id: row
        for row in (
            GraphResultV4.model_validate_json(line)
            for line in (scoring / "graph_results.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    evaluation_cases = []
    benchmark_cases = []
    for source in sources:
        raw_id = str(source["source_paper_id"])
        paper_path = PROJECT_ROOT / "data/nature_markdown/paper" / f"{raw_id}.md"
        text = paper_path.read_text(encoding="utf-8", errors="replace")
        cutoff = _submission_date(text)
        metadata = PaperMetadata(
            title=_title(text),
            doi=source["doi"],
            openalex_id=source["openalex_id"],
            submission_date=cutoff,
            venue="Nature Communications",
        )
        metadata_path = metadata_dir / f"{raw_id}.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n")
        case_id = f"10.1038_{raw_id}"
        graph = graphs.get(str(source["openalex_id"]))
        if graph is None:
            raise ValueError(f"missing GraphResultV4 for {raw_id}")
        evaluation_cases.append(
            {
                "case_id": case_id,
                "paper_id": graph.paper_id,
                "manuscript_path": str(paper_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "cutoff_date": cutoff,
                "graph_result": graph.model_dump(mode="json"),
            }
        )
        benchmark_cases.append(
            {
                "case_id": case_id,
                "paper_id": graph.paper_id,
                "paper_path": str(paper_path.resolve()),
                "metadata": metadata.model_dump(mode="json"),
                "cutoff": cutoff,
            }
        )
    manifest = {
        "contract": "gear_evaluation_manifest_v1",
        "dataset_id": "nature_dev100_v1",
        "development_non_confirmatory": True,
        "human_release_dir": str(args.human_release_dir.resolve()),
        "tracks": [
            "review_quality",
            "novelty",
            "evidence_support",
            "revision",
            "reliability",
            "graph_ablation",
            "efficiency",
        ],
        "bootstrap_samples": 5000,
        "seed": 20260821,
        "cases": evaluation_cases,
    }
    EvaluationManifestV1.model_validate(manifest)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "batch_manifest.json").write_text(
        json.dumps({"cases": benchmark_cases}, ensure_ascii=False, indent=2) + "\n"
    )
    runtime_config = {
        "graph_results_path": str((scoring / "graph_results.jsonl").resolve()),
        "max_claims": 8,
        "retrieval": {
            "normal_max": 2,
            "contrastive_max": 1,
            "citation_expansion_max": 0,
            "fulltext_max": 6,
            "provider_limit": 30,
            "relation_cards_max": 8,
            "total_actions_max": 16,
            "lexical_candidate_limit": 12,
            "semantic_candidate_limit": 20,
            "candidate_union_limit": 60,
            "embedding_candidate_limit": 40,
            "rerank_candidate_limit": 12,
            "dual_rerank_top_k": 8,
            "retained_candidates_per_claim": 6,
            "per_family_retained_max": 2,
            "minimum_comparable_candidates": 6,
            "minimum_unique_candidates": 12,
        },
    }
    (output / "runtime_config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

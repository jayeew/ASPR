#!/usr/bin/env python3
"""Select 200 papers by fixed journal quotas and historical field proportions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import (
    configure_limits,
    experiment_config,
    generate_with_retries,
    read_jsonl,
    setup_stage_logging,
    write_json,
    write_jsonl,
)
from experiments.innovation_200.contracts import PaperRow
from gear.contracts import StrictModel
from gear.model_client import LazyRoleClient
from gear.paper_extraction import fetch_openalex, recover_abstract

SEED = 20260907
JOURNAL_QUOTAS = {
    "41467": 150,
    "42003": 10,
    "42004": 10,
    "42005": 10,
    "43246": 10,
    "43247": 10,
}


class FieldChoice(StrictModel):
    field_name: str


def excerpt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lower = text.casefold()
    start = lower.find("abstract")
    return text[max(start, 0) : max(start, 0) + 8_000]


def historical_fields(path: Path) -> tuple[list[str], Counter[str], dict[str, dict]]:
    rows = pq.read_table(
        path,
        columns=["doi", "field_name", "title", "work_id", "primary_topic_name"],
    ).to_pylist()
    counts = Counter(str(row["field_name"]) for row in rows if row.get("field_name"))
    by_doi = {
        str(row["doi"]).casefold(): row for row in rows if str(row.get("doi") or "")
    }
    return sorted(counts), counts, by_doi


def enrich(row: dict, fields: list[str], by_doi: dict[str, dict], output: Path) -> dict:
    paper_id = str(row["article_id"])
    target = output / "metadata" / f"{paper_id}.json"
    if target.exists():
        return json.loads(target.read_text())
    known = by_doi.get(str(row["doi"]).casefold())
    work: dict = {}
    field = str(known.get("field_name")) if known else ""
    source = "historical_metadata" if field else ""
    try:
        work = fetch_openalex(str(row["doi"]))
        topic = work.get("primary_topic") or {}
        candidate = (topic.get("field") or {}).get("display_name")
        if not field and candidate in fields:
            field, source = str(candidate), "openalex"
    except (OSError, ValueError, KeyError, TypeError):
        work = {}
    abstract = recover_abstract(work.get("abstract_inverted_index"))
    paper_excerpt = excerpt(Path(str(row["paper_markdown_path"])))
    if not abstract:
        abstract = paper_excerpt
    if not field:
        schema = FieldChoice.model_json_schema()
        schema["properties"]["field_name"]["enum"] = fields
        result = generate_with_retries(
            lambda: LazyRoleClient(
                experiment_config(), "field_classifier"
            ).generate_json(
                system="Choose exactly one supplied historical Nature field for this paper from title and abstract. Do not add a field.",
                user=json.dumps(
                    {
                        "fields": fields,
                        "title": row["title"],
                        "abstract": abstract[:8000],
                    },
                    ensure_ascii=False,
                ),
                response_schema=schema,
            )
        )
        field, source = FieldChoice.model_validate(result).field_name, "model"
    result = {
        **row,
        "field_name": field,
        "field_source": source,
        "abstract_text": abstract,
        "openalex_work_id": str(work.get("id") or "") or None,
        "reference_work_ids": [str(x) for x in work.get("referenced_works", [])],
        "authors": [
            str(x["author"].get("display_name", ""))
            for x in work.get("authorships", [])
            if isinstance(x, dict) and isinstance(x.get("author"), dict)
        ],
    }
    write_json(target, result)
    return result


def target_counts(counts: Counter[str], total: int) -> dict[str, int]:
    denominator = sum(counts.values())
    raw = {field: total * count / denominator for field, count in counts.items()}
    output = {field: int(value) for field, value in raw.items()}
    rng = random.Random(SEED)
    tie_break = {field: rng.random() for field in sorted(raw)}
    for field in sorted(
        raw, key=lambda x: (raw[x] - output[x], tie_break[x]), reverse=True
    )[: total - sum(output.values())]:
        output[field] += 1
    return output


def allocate(
    rows: list[dict], fields: list[str], targets: dict[str, int]
) -> dict[tuple[str, str], int]:
    journals = sorted(JOURNAL_QUOTAS)
    pairs = [(journal, field) for journal in journals for field in fields]
    available = Counter(
        (str(row["journal_id"]), str(row["field_name"])) for row in rows
    )
    n = len(pairs)
    tie_rng = random.Random(SEED)
    tie_cost = np.asarray([tie_rng.random() / (201 * n) for _ in pairs])
    objective = np.r_[tie_cost, np.ones(2 * len(fields))]
    lower = np.zeros(len(objective))
    upper = np.r_[[available[pair] for pair in pairs], np.full(2 * len(fields), np.inf)]
    constraints = []
    bounds = []
    for journal in journals:
        line = np.zeros(len(objective))
        for i, pair in enumerate(pairs):
            line[i] = pair[0] == journal
        constraints.append(line)
        bounds.append(float(JOURNAL_QUOTAS[journal]))
    for field_index, field in enumerate(fields):
        line = np.zeros(len(objective))
        for i, pair in enumerate(pairs):
            line[i] = pair[1] == field
        line[n + field_index] = -1
        line[n + len(fields) + field_index] = 1
        constraints.append(line)
        bounds.append(float(targets[field]))
    result = milp(
        objective,
        integrality=np.r_[np.ones(n), np.zeros(2 * len(fields))],
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.asarray(constraints), bounds, bounds),
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Field allocation failed: {result.message}")
    return {pair: round(result.x[i]) for i, pair in enumerate(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/nature_2026_testset/manifest.jsonl"
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / "data/claim_graph/canonical_target_works.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cli-limit", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.output, "sample_papers", args.verbose)
    started = time.monotonic()
    logger.info(
        "[配置] source=%s，history=%s，output=%s，workers=%d，cli_limit=%d，seed=%d",
        args.source,
        args.history,
        args.output,
        args.workers,
        args.cli_limit,
        SEED,
    )
    configure_limits(args.cli_limit)
    logger.info("[步骤 1/4] 读取固定候选集和历史领域分布")
    source = read_jsonl(args.source)
    if len(source) != 1_000:
        logger.error("[步骤 1/4] 候选数量错误：实际=%d，预期=1000", len(source))
        raise ValueError(f"Expected the fixed 1,000 candidates, found {len(source)}")
    fields, history, by_doi = historical_fields(args.history)
    from gear.innovation.usage import usage_log

    cached_before = sum(
        (args.output / "metadata" / f"{row['article_id']}.json").exists()
        for row in source
    )
    logger.info(
        "[步骤 2/4] 补全领域元数据：候选=%d，已有缓存=%d，待处理=%d，领域=%d，并发=%d",
        len(source),
        cached_before,
        len(source) - cached_before,
        len(fields),
        args.workers,
    )

    def enrich_logged(row: dict) -> dict:
        paper_id = str(row["article_id"])
        with usage_log(
            args.output / "status/usage/sample_papers" / f"{paper_id}.jsonl"
        ):
            return enrich(row, fields, by_doi, args.output)

    enriched = []
    enrich_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(enrich_logged, row): str(row["article_id"]) for row in source
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                enriched.append(future.result())
            except (OSError, RuntimeError, ValueError, TypeError, KeyError):
                logger.exception(
                    "[步骤 2/4] 元数据补全失败：paper_id=%s", futures[future]
                )
                raise
            if completed == 1 or completed % 25 == 0 or completed == len(source):
                elapsed = max(time.monotonic() - enrich_started, 0.001)
                logger.info(
                    "[步骤 2/4] 进度=%d/%d，当前=%s，速度=%.2f篇/秒",
                    completed,
                    len(source),
                    futures[future],
                    completed / elapsed,
                )
    logger.info(
        "[步骤 2/4] 元数据完成：historical=%d，openalex=%d，model=%d",
        sum(row["field_source"] == "historical_metadata" for row in enriched),
        sum(row["field_source"] == "openalex" for row in enriched),
        sum(row["field_source"] == "model" for row in enriched),
    )
    logger.info("[步骤 3/4] 求解期刊×领域整数配额并固定种子抽样")
    targets = target_counts(history, 200)
    allocation = allocate(enriched, fields, targets)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in enriched:
        grouped[(str(row["journal_id"]), str(row["field_name"]))].append(row)
    rng = random.Random(SEED)
    selected = []
    for key in sorted(allocation):
        candidates = sorted(grouped[key], key=lambda row: str(row["article_id"]))
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[key]])
    if len(selected) != 200:
        raise ValueError(f"Selected {len(selected)}, expected 200")
    papers = [
        PaperRow(
            paper_id=str(row["article_id"]),
            title=str(row["title"]),
            doi=str(row["doi"]),
            journal_id=str(row["journal_id"]),
            journal_name=str(row["journal_name"]),
            publication_date=str(row["publication_date"])[:10],
            paper_path=str(Path(str(row["paper_markdown_path"])).resolve()),
            review_path=str(Path(str(row["peer_review_markdown_path"])).resolve()),
            field_name=str(row["field_name"]),
            field_source=str(row["field_source"]),
            abstract_text=str(row["abstract_text"]),
            openalex_work_id=row.get("openalex_work_id"),
            reference_work_ids=row.get("reference_work_ids", []),
            authors=row.get("authors", []),
        ).model_dump(mode="json")
        for row in sorted(selected, key=lambda row: str(row["article_id"]))
    ]
    write_jsonl(args.output / "papers.jsonl", papers)
    actual = Counter(row["field_name"] for row in papers)
    write_json(
        args.output / "field_distribution.json",
        {
            "historical_total": sum(history.values()),
            "target": targets,
            "actual": dict(actual),
            "journal_actual": dict(Counter(row["journal_name"] for row in papers)),
            "field_source": dict(Counter(row["field_source"] for row in papers)),
        },
    )
    logger.info(
        "[步骤 4/4] 完成：抽样=%d；期刊=%s；输出=%s；耗时=%.1f秒",
        len(papers),
        dict(Counter(row["journal_name"] for row in papers)),
        args.output / "papers.jsonl",
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main()

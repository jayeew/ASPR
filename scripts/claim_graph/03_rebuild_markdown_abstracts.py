#!/usr/bin/env python3
"""仅用本地 Nature Markdown 修复指定论文的 Phase 3 摘要与编号句。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABSTRACTS = PROJECT_ROOT / "data" / "claim_graph" / "abstracts.parquet"
DEFAULT_SENTENCES = PROJECT_ROOT / "data" / "claim_graph" / "abstract_sentences.parquet"
PHASE3_SCRIPT = PROJECT_ROOT / "scripts" / "claim_graph" / "03_extract_abstracts.py"


def parse_article_ids(value: str) -> list[str]:
    """Parse a comma-separated article-ID list without accepting an empty target."""
    article_ids: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        article_id = raw.strip()
        if article_id and article_id not in seen:
            article_ids.append(article_id)
            seen.add(article_id)
    if not article_ids:
        raise ValueError("--article-ids 未包含有效 article ID")
    return article_ids


def load_phase3_helpers() -> Any:
    """Load the shared Markdown abstract locator without invoking Phase 3."""
    spec = importlib.util.spec_from_file_location("claim_graph_phase3_helpers", PHASE3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Phase 3 工具函数：{PHASE3_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rebuild_records(
    articles: list[dict[str, Any]], helpers: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Extract complete Markdown abstracts and numbered sentences for selected papers."""
    abstract_replacements: dict[str, dict[str, Any]] = {}
    sentence_replacements: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        article_id = str(article["article_id"])
        markdown_path = Path(str(article["paper_markdown_path"]))
        raw = markdown_path.read_text(encoding="utf-8", errors="replace")
        located = helpers.locate_abstract(raw, str(article["title"]))
        if located is None:
            raise ValueError(f"{article_id} 未在本地 Markdown 找到可信摘要：{markdown_path}")
        start, end, method = located
        masked = helpers.cleaned_preserving_offsets(raw[start:end])
        abstract = helpers.normalized(masked)
        spans = helpers.sentence_spans(masked, start)
        if len(spans) < 2:
            raise ValueError(f"{article_id} Markdown 摘要切句少于 2 句")
        abstract_replacements[article_id] = {
            **article,
            "abstract_text": abstract,
            "abstract_start_char": spans[0][1],
            "abstract_end_char": spans[-1][2],
            "extraction_method": method,
        }
        sentence_replacements[article_id] = [
            {
                "sentence_id": f"{article_id}::S{number:02d}",
                "article_id": article_id,
                "sentence_index": number,
                "sentence_text": text,
                "markdown_start_char": source_start,
                "markdown_end_char": source_end,
            }
            for number, (text, source_start, source_end) in enumerate(spans, 1)
        ]
        print(f"已重建 {article_id}：摘要={len(abstract)} 字符，句子={len(spans)}，来源={method}")
    return abstract_replacements, sentence_replacements


def replace_sentence_rows(
    rows: list[dict[str, Any]], replacements: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Replace only selected papers while retaining every other sentence row and its order."""
    updated: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for row in rows:
        article_id = str(row["article_id"])
        if article_id not in replacements:
            updated.append(row)
        elif article_id not in emitted:
            updated.extend(replacements[article_id])
            emitted.add(article_id)
    missing = set(replacements) - emitted
    if missing:
        raise ValueError(f"指定论文未出现在原摘要句表中：{sorted(missing)}")
    return updated


def write_table_atomic(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    """Write one replacement table atomically in the existing Parquet schema."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    """Repair only requested Phase 3 records from the already-mounted Markdown corpus."""
    article_ids = parse_article_ids(args.article_ids)
    if not args.abstracts.is_file() or not args.sentences.is_file():
        raise FileNotFoundError("未找到 Phase 3 的 abstracts.parquet 或 abstract_sentences.parquet")
    abstracts_table = pq.read_table(args.abstracts)
    sentences_table = pq.read_table(args.sentences)
    abstracts = abstracts_table.to_pylist()
    targets = {str(row["article_id"]): row for row in abstracts}
    missing = set(article_ids) - set(targets)
    if missing:
        raise ValueError(f"指定论文不在 abstracts.parquet：{sorted(missing)}")
    print(f"开始 Markdown 定点修复：论文={len(article_ids)}；不会请求 OpenAlex，也不会修改其他论文。")
    helpers = load_phase3_helpers()
    replacements, sentence_replacements = rebuild_records([targets[item] for item in article_ids], helpers)
    updated_abstracts = [replacements.get(str(row["article_id"]), row) for row in abstracts]
    updated_sentences = replace_sentence_rows(sentences_table.to_pylist(), sentence_replacements)
    print("所有目标摘要均已在内存中完成重建，开始覆盖两个 Phase 3 输出表。")
    write_table_atomic(updated_abstracts, abstracts_table.schema, args.abstracts)
    write_table_atomic(updated_sentences, sentences_table.schema, args.sentences)
    print(f"修复完成：摘要表={args.abstracts}；句子表={args.sentences}")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-ids", required=True, help="逗号分隔的 Nature article ID")
    parser.add_argument("--abstracts", type=Path, default=DEFAULT_ABSTRACTS)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    return parser


def main() -> int:
    """Run the repair and present concise Chinese errors."""
    try:
        run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, pa.ArrowException) as error:
        print(f"Markdown 摘要定点修复失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

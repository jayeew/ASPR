"""Command-line interface for the current innovation-only GEAR runtime."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .artifacts import read_model, write_model
from .config import load_config
from .paper_extraction import prepare_input
from .review_contracts import InnovationPaperInput
from .review_pipeline import review_paper


def _input_from_args(args: argparse.Namespace) -> InnovationPaperInput:
    if args.input_contract is not None:
        return read_model(args.input_contract, InnovationPaperInput)
    required = {
        "paper_id": args.paper_id,
        "title": args.title,
        "doi": args.doi,
        "publication_date": args.publication_date,
        "cutoff": args.cutoff,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("--paper 模式缺少参数：" + ", ".join(missing))
    return prepare_input(
        paper_path=args.paper,
        paper_id=args.paper_id,
        title=args.title,
        doi=args.doi,
        publication_date=date.fromisoformat(args.publication_date),
        cutoff_date=date.fromisoformat(args.cutoff),
        venue=args.venue,
        abstract_text=(
            args.abstract_file.read_text(encoding="utf-8")
            if args.abstract_file is not None
            else None
        ),
    )


def _review(args: argparse.Namespace) -> int:
    item = _input_from_args(args)
    outputs = review_paper(
        item,
        output_dir=args.output_dir,
        config=load_config(args.config),
        stage=args.stage,
        fusion_mode=args.fusion_mode,
        graph_root=args.claim_graph_root,
        embedding_model=args.claim_embedding_model,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    item = _input_from_args(args)
    write_model(args.output, item)
    print(args.output)
    return 0


def _show_run(args: argparse.Namespace) -> int:
    paths = (
        args.run_dir / "graph" / "graph_branch_result.json",
        args.run_dir / "gear" / "gear_branch_result.json",
        args.run_dir / "fusion" / "fusion_result.json",
    )
    payload = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.exists()]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _paper_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-contract", type=Path)
    source.add_argument("--paper", "--pdf", dest="paper", type=Path)
    parser.add_argument("--paper-id")
    parser.add_argument("--title")
    parser.add_argument("--doi")
    parser.add_argument("--publication-date")
    parser.add_argument("--cutoff")
    parser.add_argument("--venue")
    parser.add_argument("--abstract-file", type=Path)


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Claim Graph + GEAR 创新性评价")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-input", help="准备统一测试论文输入")
    _paper_arguments(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(handler=_prepare)
    review = commands.add_parser("review", help="运行 Graph、GEAR 或融合分支")
    _paper_arguments(review)
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--stage", choices=("all", "graph", "gear", "fusion"), default="all")
    review.add_argument("--fusion-mode", choices=("passive", "active"), default="passive")
    review.add_argument("--claim-graph-root", type=Path, default=Path("data/claim_graph"))
    review.add_argument("--claim-embedding-model", type=Path, default=Path("data/models/Qwen3-Embedding-4B"))
    review.add_argument("--config", type=Path)
    review.set_defaults(handler=_review)
    show = commands.add_parser("validate-run", help="显示各分支落盘结果，不执行一致性校验")
    show.add_argument("run_dir", type=Path)
    show.set_defaults(handler=_show_run)
    return root


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


__all__ = ["build_parser", "main"]

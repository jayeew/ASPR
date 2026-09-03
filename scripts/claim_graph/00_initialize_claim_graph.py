#!/usr/bin/env python3
"""初始化 Claim Graph Phase 0：目录、配置与共享契约。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.claim_graph.contracts import (
    ClaimCandidateEdge,
    ClaimCommunity,
    ClaimInsertionProfile,
    InnovationClaim,
    InnovationClaimInventory,
    InnovationClaimType,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "claim_graph"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gear" / "claim_graph.yaml"
DIRECTORIES = ("logs", "chunks", "evaluation")


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    """Create concise Chinese console and file logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase0")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def load_config(path: Path) -> dict[str, Any]:
    """Load the small hand-maintained configuration required by later phases."""
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是 YAML 对象：{path}")
    allowed_types = payload.get("claim_extraction", {}).get("allowed_types")
    expected_types = [item.value for item in InnovationClaimType]
    if allowed_types != expected_types:
        raise ValueError(
            "claim_extraction.allowed_types 必须按固定顺序包含："
            + ", ".join(expected_types)
        )
    return payload


def contract_schemas() -> dict[str, Any]:
    """Return the public schemas consumed by later independent modules."""
    models = (
        InnovationClaim,
        InnovationClaimInventory,
        ClaimCandidateEdge,
        ClaimCommunity,
        ClaimInsertionProfile,
    )
    return {model.__name__: model.model_json_schema() for model in models}


def initialize(output_root: Path, config_path: Path, verbose: bool) -> dict[str, Any]:
    """Create Phase 0 directories and write a compact contract snapshot."""
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_root / "logs" / "phase0_initialize.log", verbose)
    logger.info("[步骤 1/4] 开始校验配置文件：%s", config_path)
    config = load_config(config_path)
    logger.info("[步骤 1/4] 配置校验通过")
    logger.info("[步骤 2/4] 创建约定目录：%s", output_root)
    for name in DIRECTORIES:
        (output_root / name).mkdir(parents=True, exist_ok=True)
        logger.info("[步骤 2/4] 目录就绪：%s", output_root / name)
    logger.info("[步骤 3/4] 载入共享 Claim Graph 契约")
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "output_root": str(output_root.resolve()),
        "config_path": str(config_path.resolve()),
        "claim_types": [item.value for item in InnovationClaimType],
        "directories": list(DIRECTORIES),
        "config": config,
        "contracts": contract_schemas(),
    }
    summary_path = output_root / "phase0_contracts.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("[步骤 3/4] 固定 Claim 类型：%s", ", ".join(summary["claim_types"]))
    logger.info("[步骤 4/4] 写入共享契约摘要：%s", summary_path)
    logger.info("Phase 0 完成：后续模块可开始读取 data/claim_graph 中的约定路径")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        initialize(args.output_root, args.config, args.verbose)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Phase 0 初始化失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

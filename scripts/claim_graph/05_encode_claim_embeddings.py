#!/usr/bin/env python3
"""将 Claim 编码为统一语义空间向量（Claim Graph Phase 5）。

默认使用本地 Qwen/Qwen3-Embedding-4B。脚本只读取 Phase 4 的
claim_nodes.parquet，按批写出独立 Parquet chunk；中断后再次以 --resume
运行时，会跳过已经落盘的 claim_id，最后合并为 claim_embeddings.parquet。

一个 4B 模型应只驻留一张 GPU 一次。并发由单次 GPU batch 提供，而不是启动
多个模型进程；后者会重复占用显存，通常更慢且容易 OOM。
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAIMS = PROJECT_ROOT / "data" / "claim_graph" / "claim_nodes.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "models" / "Qwen3-Embedding-4B"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "claim_graph"
EMBEDDING_DIMENSION = 2560
EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("claim_id", pa.string(), False),
        pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIMENSION), False),
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse the standalone Phase 5 command line."""
    parser = argparse.ArgumentParser(description="Phase 5：用本地 Qwen3-Embedding-4B 编码 Claim")
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS, help="Phase 4 Claim 节点 Parquet")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="本地 Qwen3-Embedding-4B 目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="claim_graph 输出目录")
    parser.add_argument("--device", default="cuda", help="编码设备；默认 cuda，可设为 cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="每次 GPU 前向批量；0=从 512 自动试起，OOM 时自动减半",
    )
    parser.add_argument("--max-seq-length", type=int, default=512, help="Claim 最大 token 数")
    parser.add_argument("--resume", action="store_true", help="保留已有 chunk，跳过已经完成的 Claim")
    parser.add_argument("--restart", action="store_true", help="删除旧 Phase 5 输出后从头编码")
    parser.add_argument("--limit", type=int, default=0, help="仅编码排序后的前 N 条 Claim；0=全量")
    parser.add_argument("--verbose", action="store_true", help="输出每批详细日志")
    return parser.parse_args()


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    """Create Chinese console and file logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase5")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def require_input(path: Path, label: str) -> None:
    """Fail early with a human-readable missing-input error."""
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")


def reset_outputs(chunk_dir: Path, embedding_path: Path, failure_path: Path) -> None:
    """Remove only Phase 5 artifacts when the user explicitly requests restart."""
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    for path in (embedding_path, failure_path):
        if path.exists():
            path.unlink()


def load_claims(path: Path, limit: int) -> pa.Table:
    """Read only the stable Claim identifier and text required for encoding."""
    table = pq.read_table(path, columns=["claim_id", "claim_text"])
    if table.num_rows == 0:
        raise ValueError("Claim 节点表为空，无法进行 Phase 5")
    if table["claim_id"].null_count or table["claim_text"].null_count:
        raise ValueError("Claim 节点表含空 claim_id 或 claim_text")
    table = table.sort_by([("claim_id", "ascending")])
    if pc.count_distinct(table["claim_id"]).as_py() != table.num_rows:
        raise ValueError("Claim 节点表存在重复 claim_id，无法安全断点续跑")
    if limit > 0:
        table = table.slice(0, min(limit, table.num_rows))
    return table


def completed_claim_ids(chunk_dir: Path) -> set[str]:
    """Collect completed IDs from independent chunks for resumability."""
    ids: set[str] = set()
    for path in sorted(chunk_dir.glob("chunk_*.parquet")):
        try:
            table = pq.read_table(path, columns=["claim_id"])
        except (OSError, pa.ArrowException) as error:
            raise RuntimeError(f"无法读取既有 chunk：{path}；请删除该文件后用 --resume 重跑") from error
        ids.update(value.as_py() for value in table["claim_id"])
    return ids


def choose_initial_batch_size(requested: int, device: str) -> int:
    """Choose a high-throughput starting batch without duplicating the model."""
    if requested < 0:
        raise ValueError("--batch-size 必须为非负整数")
    if requested:
        return requested
    return 512 if device.startswith("cuda") else 32


def load_encoder(model_path: Path, device: str, max_seq_length: int):
    """Load Qwen exactly once, in BF16 on CUDA and with its native pooling config."""
    if max_seq_length <= 0:
        raise ValueError("--max-seq-length 必须为正整数")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("缺少 torch 或 sentence-transformers；请安装当前项目环境依赖") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 PyTorch 未发现可用 GPU；可显式传 --device cpu")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        kwargs: dict[str, object] = {"torch_dtype": torch.bfloat16}
    else:
        kwargs = {}
    encoder = SentenceTransformer(str(model_path), device=device, model_kwargs=kwargs)
    encoder.max_seq_length = max_seq_length
    return encoder, torch


def encode_with_oom_backoff(encoder, torch_module, texts: list[str], batch_size: int) -> tuple[np.ndarray, int]:
    """Encode one pending segment, halving only the offending batch after CUDA OOM."""
    current_size = batch_size
    while True:
        try:
            vectors = encoder.encode(
                texts,
                batch_size=current_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32), current_size
        except torch_module.OutOfMemoryError:
            if not torch_module.cuda.is_available() or current_size == 1:
                raise
            torch_module.cuda.empty_cache()
            current_size = max(1, current_size // 2)


def write_chunk(chunk_path: Path, claim_ids: list[str], vectors: np.ndarray) -> None:
    """Persist one completed batch atomically enough for normal interruption recovery."""
    if vectors.ndim != 2 or vectors.shape != (len(claim_ids), EMBEDDING_DIMENSION):
        raise ValueError(
            f"模型向量形状异常：得到 {vectors.shape}，预期 ({len(claim_ids)}, {EMBEDDING_DIMENSION})"
        )
    flattened = pa.array(vectors.reshape(-1), type=pa.float32())
    embeddings = pa.FixedSizeListArray.from_arrays(flattened, EMBEDDING_DIMENSION)
    table = pa.Table.from_arrays([pa.array(claim_ids, type=pa.string()), embeddings], schema=EMBEDDING_SCHEMA)
    temporary = chunk_path.with_suffix(".tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False)
    temporary.replace(chunk_path)


def merge_chunks(chunk_dir: Path, output_path: Path, expected_ids: set[str]) -> None:
    """Merge chunks into the single Phase 5 contract consumed by later phases."""
    chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
    if not chunks:
        raise RuntimeError("没有可合并的 Phase 5 chunk")
    tables = [pq.read_table(path, schema=EMBEDDING_SCHEMA) for path in chunks]
    table = pa.concat_tables(tables).sort_by([("claim_id", "ascending")])
    ids = [value.as_py() for value in table["claim_id"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Phase 5 chunk 存在重复 claim_id；请使用 --restart 重跑")
    if set(ids) != expected_ids:
        raise RuntimeError(f"合并后 Claim 覆盖不完整：得到 {len(ids)}，预期 {len(expected_ids)}")
    temporary = output_path.with_suffix(".tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False)
    temporary.replace(output_path)


def write_failure_csv(path: Path, failures: list[tuple[str, str]]) -> None:
    """Write the small explicit failure receipt only when an unrecoverable error occurs."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "reason"])
        writer.writerows(failures)


def batches(values: list[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    """Yield deterministic chunks without holding another copy of the whole table."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> int:
    """Run the resumable local GPU Claim embedding job."""
    args = parse_args()
    if args.resume and args.restart:
        raise ValueError("--resume 与 --restart 不能同时使用")
    require_input(args.claims, "Claim 节点表")
    require_input(args.model / "config.json", "Qwen 模型 config.json")
    output_root = args.output_root.resolve()
    chunk_dir = output_root / "phase5_embedding_chunks"
    embedding_path = output_root / "claim_embeddings.parquet"
    failure_path = output_root / "claim_embedding_failures.csv"
    logger = setup_logging(output_root / "phase5_encode_claim_embeddings.log", args.verbose)
    if args.restart:
        logger.info("[步骤 0/5] 清空旧 Phase 5 chunk、向量表和失败表")
        reset_outputs(chunk_dir, embedding_path, failure_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    claims = load_claims(args.claims, args.limit)
    pairs = list(zip(claims["claim_id"].to_pylist(), claims["claim_text"].to_pylist(), strict=True))
    expected_ids = {claim_id for claim_id, _ in pairs}
    done_ids = completed_claim_ids(chunk_dir) if args.resume else set()
    unknown_ids = done_ids - expected_ids
    if unknown_ids:
        raise RuntimeError("已有 chunk 不属于当前输入 Claim；请使用 --restart 清空后重跑")
    pending = [(claim_id, text) for claim_id, text in pairs if claim_id not in done_ids]
    batch_size = choose_initial_batch_size(args.batch_size, args.device)
    logger.info(
        "[步骤 1/5] Claim=%d，已完成=%d，待处理=%d，设备=%s，初始批量=%d",
        len(pairs), len(done_ids), len(pending), args.device, batch_size,
    )
    if pending:
        logger.info("[步骤 2/5] 加载 Qwen3-Embedding-4B；单模型单 GPU 批量编码")
        encoder, torch_module = load_encoder(args.model, args.device, args.max_seq_length)
        started = time.monotonic()
        completed = len(done_ids)
        chunk_index = len(list(chunk_dir.glob("chunk_*.parquet")))
        failures: list[tuple[str, str]] = []
        for group in batches(pending, batch_size):
            claim_ids, texts = zip(*group, strict=True)
            try:
                vectors, effective_batch = encode_with_oom_backoff(encoder, torch_module, list(texts), batch_size)
                chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.parquet"
                write_chunk(chunk_path, list(claim_ids), vectors)
                chunk_index += 1
                completed += len(group)
                elapsed = max(time.monotonic() - started, 0.001)
                logger.info(
                    "[步骤 2/5] 进度：完成=%d/%d，当前批=%d，速度=%.2f Claim/秒",
                    completed, len(pairs), effective_batch, (completed - len(done_ids)) / elapsed,
                )
                batch_size = effective_batch
            except (RuntimeError, ValueError, OSError, pa.ArrowException) as error:
                failures.extend((claim_id, f"{type(error).__name__}: {error}") for claim_id in claim_ids)
                logger.error("Claim batch 编码失败，已记录 %d 条：%s", len(claim_ids), error)
                break
        if failures:
            write_failure_csv(failure_path, failures)
            logger.error("Phase 5 未完成；失败记录：%s", failure_path)
            return 1
    logger.info("[步骤 3/5] 合并全部 Phase 5 chunk")
    merge_chunks(chunk_dir, embedding_path, expected_ids)
    logger.info("[步骤 4/5] 写入空失败记录")
    write_failure_csv(failure_path, [])
    size_mb = embedding_path.stat().st_size / 1024 / 1024
    logger.info("[步骤 5/5] Phase 5 完成：Claim=%d，向量维度=%d，文件=%.1f MiB", len(pairs), EMBEDDING_DIMENSION, size_mb)
    logger.info("Claim 向量表：%s", embedding_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, pa.ArrowException) as error:
        print(f"Phase 5 构建失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error

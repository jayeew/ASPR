#!/usr/bin/env python3
"""从编号摘要句并行抽取 1--3 条原子贡献 Claim（Claim Graph Phase 4）。

模型调用被隔离在 ClaimProvider 中。主流程、Prompt、输出 Parquet、断点和
结果校验对 Codex CLI 与 DeepSeek API 完全共用；切换提供方不会改变落盘格式。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABSTRACTS = PROJECT_ROOT / "data" / "claim_graph" / "abstracts.parquet"
DEFAULT_SENTENCES = PROJECT_ROOT / "data" / "claim_graph" / "abstract_sentences.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "claim_graph"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
CLAIM_TYPES = {"METHOD", "FINDING", "MECHANISM", "RESOURCE", "THEORY"}
TYPE_ALIASES = {
    "method_or_system": "METHOD",
    "intervention_or_design": "METHOD",
    "finding": "FINDING",
    "mechanism": "MECHANISM",
    "resource_or_synthesis": "RESOURCE",
    "resource": "RESOURCE",
    "theory": "THEORY",
}
CLAIM_SCHEMA = pa.schema(
    [
        pa.field("claim_id", pa.string(), False),
        pa.field("parent_paper_id", pa.string(), False),
        pa.field("claim_type", pa.string(), False),
        pa.field("claim_text", pa.string(), False),
        pa.field("source_sentence_ids", pa.list_(pa.string()), False),
        pa.field("source_sentence_texts", pa.list_(pa.string()), False),
        pa.field("source_fragments", pa.list_(pa.string()), False),
        pa.field("title", pa.string(), False),
        pa.field("abstract_text", pa.string(), False),
        pa.field("publication_date", pa.date32(), False),
    ]
)
FAILURE_COLUMNS = ("article_id", "reason", "detail")

SYSTEM_PROMPT = """You extract atomic contribution claims from scientific abstracts.
Return JSON only. Do not use Markdown, explanations, or code fences."""

PROMPT_TEMPLATE = """For ARTICLE_ID {article_id}, extract 1--3 central atomic contribution Claims from the numbered abstract sentences below.

Return JSON only in exactly this shape:
{{"claims":[{{"claim_text":"string","claim_type":"method_or_system|finding|mechanism|resource_or_synthesis|theory","evidence_sentence_ids":["exact sentence IDs"]}}]}}

Use claims=[] only when no sentence supports a qualifying central contribution.

Include paper-specific findings, mechanisms, concrete methods/systems/interventions, or resources/syntheses. Exclude background, aims, routine design, sample size alone, future applications, and generic significance.

Claim type definitions (choose exactly one):
- method_or_system: the reusable procedure, algorithm, model, system, material/catalyst design, or intervention design itself; not an outcome observed after applying it.
- finding: an observed, measured, estimated, comparative, structural, performance, association, or intervention outcome.
- mechanism: an evidence-supported causal account of how or why a result occurs.
- resource_or_synthesis: a dataset, atlas, database, catalog, benchmark, genome assembly, or curated synthesis delivered for reuse; not a protocol or an ordinary result.
- theory: a newly proposed formal principle, law, or conceptual/mathematical framework; not merely theoretical/computational analysis or an inferred historical explanation.

Rules:
1. One Claim has one subject, one result axis, and one experimental setting. Split behavioural versus neural results, quantitative findings versus classifications/resources, and animal versus human results. Do not compress distinct contributions to reduce Claim count; prefer separate central Claims up to the limit of three.
2. One named dataset, atlas, platform, or system is one Claim containing its essential scale, coverage, or components.
3. Evidence is the smallest self-contained sufficient set. Every entity, population, intervention, comparison, mechanism, and number in claim_text must occur in the cited sentences.
4. If a cited sentence contains an unresolved phrase such as this shift, this effect, these results, it, or they, it cannot stand alone: also cite the nearest sentence that explicitly defines the referent. Claim text itself must contain no unresolved references.
5. Preserve source strength. Do not use introduced, established, proved, produced, caused, novel, new, first, or superior unless explicit. Prefer reported, observed, showed, or was associated with for non-randomized results.
6. Preserve valid discriminative entities, direction, mechanism, and numbers. Never repair malformed data. Use 15--45 English words per Claim.
7. Use mechanism only when the evidence explicitly explains how or why an effect occurs. Correlations, clustering patterns, predictive associations, and inferred connectivity are findings, not mechanisms.
8. An observed effect of an intervention or perturbation is a finding, not a method, unless the central contribution is the reusable intervention design itself.
9. Genetic associations, inheritance patterns, and epistatic variance explanations are findings unless the evidence directly establishes a causal molecular or biological pathway.
10. Do not output intervention_or_design. A reusable intervention design is method_or_system; its observed effects are findings.

NUMBERED ABSTRACT SENTENCES:
{sentences}
"""


class ProviderError(RuntimeError):
    """A model provider did not return a usable completion."""


class ResponseValidationError(ValueError):
    """The provider response violates the shared extraction contract."""


@dataclass(frozen=True)
class ClaimDraft:
    """Provider-neutral Claim payload before local source binding."""

    claim_text: str
    claim_type: str
    evidence_sentence_ids: list[str]


class ClaimProvider(Protocol):
    """Only this protocol differs between local Codex and remote APIs."""

    def extract(self, prompt: str) -> dict[str, Any]:
        """Return decoded JSON produced for one paper."""


@dataclass(frozen=True)
class CodexCliProvider:
    """Use an authenticated local Codex CLI without embedding credentials."""

    model: str
    reasoning_effort: str
    project_root: Path
    timeout_seconds: int
    retries: int

    def extract(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                with tempfile.TemporaryDirectory(prefix="claim_graph_codex_") as directory:
                    output_path = Path(directory) / "response.json"
                    command = [
                        "codex", "exec", "--ephemeral", "-m", self.model,
                        "-s", "read-only", "-C", str(self.project_root),
                        "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                        "--output-last-message", str(output_path), prompt,
                    ]
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                    if completed.returncode != 0:
                        stderr = completed.stderr.strip()[-1_000:]
                        raise ProviderError(f"Codex CLI 退出码 {completed.returncode}: {stderr}")
                    if not output_path.is_file():
                        raise ProviderError("Codex CLI 未写入最终输出文件")
                    return decode_json(output_path.read_text(encoding="utf-8"))
            except (OSError, subprocess.SubprocessError, ProviderError, ValueError) as error:
                last_error = error
                if attempt <= self.retries:
                    time.sleep(min(2**attempt, 10))
        raise ProviderError(f"Codex CLI 重试后仍失败：{last_error}")


@dataclass(frozen=True)
class DeepSeekProvider:
    """Call DeepSeek's OpenAI-compatible Chat Completions endpoint via stdlib."""

    api_key: str
    base_url: str
    model: str
    reasoning_effort: str
    timeout_seconds: int
    retries: int

    def extract(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": 1_200,
        }
        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                request = Request(
                    self.base_url.rstrip("/") + "/chat/completions",
                    data=encoded,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ProviderError("DeepSeek 返回了空 content")
                return decode_json(content)
            except (HTTPError, URLError, KeyError, IndexError, OSError, ValueError) as error:
                last_error = error
                retryable = not isinstance(error, HTTPError) or error.code in {408, 409, 429, 500, 502, 503, 504}
                if attempt <= self.retries and retryable:
                    time.sleep(min(2**attempt, 20))
                    continue
                break
        raise ProviderError(f"DeepSeek API 重试后仍失败：{last_error}")


def setup_logging(path: Path, verbose: bool) -> logging.Logger:
    """Create detailed Chinese console and file logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase4")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler, level in (
        (logging.StreamHandler(sys.stdout), logging.DEBUG if verbose else logging.INFO),
        (logging.FileHandler(path, encoding="utf-8", mode="a"), logging.DEBUG),
    ):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def decode_json(text: str) -> dict[str, Any]:
    """Parse a JSON object, accepting accidental Markdown fences from a CLI model."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ResponseValidationError("模型输出根节点必须是 JSON object")
    return value


def normalize_type(value: Any) -> str:
    """Map prompt labels to the stable Claim Graph enum labels."""
    normalized = str(value or "").strip()
    upper = normalized.upper()
    if upper in CLAIM_TYPES:
        return upper
    mapped = TYPE_ALIASES.get(normalized.lower())
    if mapped is None:
        raise ResponseValidationError(f"未知 claim_type：{normalized!r}")
    return mapped


def parse_drafts(response: dict[str, Any], sentence_ids: set[str]) -> list[ClaimDraft]:
    """Enforce only the shared structural contract; do not add heuristic scoring."""
    claims = response.get("claims")
    if not isinstance(claims, list) or len(claims) > 3:
        raise ResponseValidationError("claims 必须是最多 3 条的数组")
    drafts: list[ClaimDraft] = []
    for position, item in enumerate(claims, 1):
        if not isinstance(item, dict):
            raise ResponseValidationError(f"第 {position} 条 Claim 不是 object")
        text = item.get("claim_text")
        evidence = item.get("evidence_sentence_ids")
        if not isinstance(text, str) or not text.strip():
            raise ResponseValidationError(f"第 {position} 条 Claim 的 claim_text 为空")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(value, str) for value in evidence):
            raise ResponseValidationError(f"第 {position} 条 Claim 未绑定句子 ID")
        if len(set(evidence)) != len(evidence):
            raise ResponseValidationError(f"第 {position} 条 Claim 含重复句子 ID")
        unknown = set(evidence) - sentence_ids
        if unknown:
            raise ResponseValidationError(f"第 {position} 条 Claim 绑定了不存在的句子：{sorted(unknown)}")
        drafts.append(ClaimDraft(text.strip(), normalize_type(item.get("claim_type")), evidence))
    return drafts


def make_prompt(article_id: str, sentences: list[dict[str, Any]]) -> str:
    """Format one complete numbered abstract; selection is entirely delegated to the model."""
    numbered = "\n".join(f'[{row["sentence_id"]}] {row["sentence_text"]}' for row in sentences)
    return PROMPT_TEMPLATE.format(article_id=article_id, sentences=numbered)


def connect_state(path: Path) -> sqlite3.Connection:
    """Open the dedicated Phase 4 checkpoint table inside the shared local DB."""
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS claim_state "
        "(article_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    return connection


def load_inputs(
    abstracts_path: Path,
    sentences_path: Path,
    limit: int | None,
    article_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Join Phase 3 abstracts and source sentences in deterministic article order."""
    abstracts = pq.read_table(abstracts_path).to_pylist()
    if article_ids is not None:
        abstracts = [row for row in abstracts if str(row["article_id"]) in article_ids]
    elif limit is not None:
        abstracts = abstracts[:limit]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sentence in pq.read_table(sentences_path).to_pylist():
        grouped.setdefault(str(sentence["article_id"]), []).append(sentence)
    papers: list[dict[str, Any]] = []
    for abstract in abstracts:
        article_id = str(abstract["article_id"])
        rows = sorted(grouped.get(article_id, []), key=lambda row: int(row["sentence_index"]))
        if rows:
            papers.append({"abstract": abstract, "sentences": rows})
    return papers


def parse_article_ids(value: str | None) -> list[str]:
    """Parse a comma-separated retry list while preserving its declared order."""
    if value is None:
        return []
    article_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in value.split(","):
        article_id = raw_value.strip()
        if article_id and article_id not in seen:
            article_ids.append(article_id)
            seen.add(article_id)
    if not article_ids:
        raise ValueError("--retry-article-ids 未包含有效的 article ID")
    return article_ids


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a completed paper chunk atomically, making resume safe after interruption."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_default(value: Any) -> str:
    """Serialize the date carried by a Claim chunk without adding a custom format."""
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"无法写入 JSON chunk 的类型：{type(value).__name__}")


def extract_one(
    paper: dict[str, Any], provider: ClaimProvider, chunk_dir: Path
) -> dict[str, Any]:
    """Call a provider for one paper and locally bind its Claims to original sentences."""
    abstract = paper["abstract"]
    article_id = str(abstract["article_id"])
    try:
        sentences = paper["sentences"]
        sentence_map = {str(row["sentence_id"]): str(row["sentence_text"]) for row in sentences}
        response = provider.extract(make_prompt(article_id, sentences))
        drafts = parse_drafts(response, set(sentence_map))
        rows: list[dict[str, Any]] = []
        for position, draft in enumerate(drafts, 1):
            source_texts = [sentence_map[sentence_id] for sentence_id in draft.evidence_sentence_ids]
            rows.append(
                {
                    "claim_id": f"{article_id}::C{position:02d}",
                    "parent_paper_id": article_id,
                    "claim_type": draft.claim_type,
                    "claim_text": draft.claim_text,
                    "source_sentence_ids": draft.evidence_sentence_ids,
                    "source_sentence_texts": source_texts,
                    "source_fragments": source_texts,
                    "title": str(abstract["title"]),
                    "abstract_text": str(abstract["abstract_text"]),
                    "publication_date": date(int(abstract["year"]), 1, 1),
                }
            )
        result: dict[str, Any] = {"article_id": article_id, "status": "complete", "claims": rows}
    except (ProviderError, ResponseValidationError, ValueError, KeyError, TypeError) as error:
        result = {"article_id": article_id, "status": "failed", "reason": type(error).__name__, "detail": str(error)}
    write_json_atomic(chunk_dir / f"{article_id}.json", result)
    return result


def iter_chunks(chunk_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield per-paper completed chunks in deterministic article-ID order."""
    for path in sorted(chunk_dir.glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def write_parquet(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """Write a single stable Claim node table without keeping all Claims in memory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, CLAIM_SCHEMA, compression="zstd")
    batch: list[dict[str, Any]] = []
    count = 0
    try:
        for row in rows:
            normalized = dict(row)
            publication_date = normalized["publication_date"]
            if isinstance(publication_date, str):
                normalized["publication_date"] = date.fromisoformat(publication_date)
            batch.append(normalized)
            if len(batch) >= 10_000:
                writer.write_table(pa.Table.from_pylist(batch, schema=CLAIM_SCHEMA))
                count += len(batch)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CLAIM_SCHEMA))
            count += len(batch)
    finally:
        writer.close()
    temporary.replace(path)
    return count


def make_provider(args: argparse.Namespace) -> ClaimProvider:
    """Construct exactly one provider; credential handling remains outside the core flow."""
    if args.provider == "codex":
        return CodexCliProvider(
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning_effort,
            project_root=PROJECT_ROOT,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
    api_key = os.environ.get(args.deepseek_api_key_env, "").strip()
    if not api_key:
        api_key = read_dotenv_value(args.env_file, args.deepseek_api_key_env)
    if not api_key:
        raise ValueError(
            f"未找到 DeepSeek 密钥：请设置环境变量 {args.deepseek_api_key_env} "
            f"或写入 {args.env_file}"
        )
    return DeepSeekProvider(
        api_key=api_key,
        base_url=args.deepseek_base_url,
        model=args.deepseek_model,
        reasoning_effort=args.deepseek_reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )


def read_dotenv_value(path: Path, key: str) -> str:
    """Read one KEY=VALUE entry without printing or modifying the .env file."""
    if not path.is_file():
        return ""
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() == key:
                return value.strip().strip("\"'")
    except OSError as error:
        raise ValueError(f"无法读取环境变量文件：{path} ({error})") from error
    return ""


def run(args: argparse.Namespace) -> None:
    """Execute Phase 4 with per-paper durable chunks and a lightweight SQLite checkpoint."""
    if not args.abstracts.is_file() or not args.sentences.is_file():
        raise FileNotFoundError("Phase 3 输入不存在：请先生成 abstracts.parquet 与 abstract_sentences.parquet")
    if args.workers < 0 or args.retries < 0 or args.timeout_seconds <= 0:
        raise ValueError("--workers/--retries/--timeout-seconds 参数值无效")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "logs" / "phase4_extract_claims.log", args.verbose)
    state_path = root / "build_state.sqlite"
    chunk_dir = root / "chunks" / "claims"
    output_path = root / "claim_nodes.parquet"
    failure_path = root / "claim_failures.csv"
    retry_article_ids = parse_article_ids(args.retry_article_ids)
    if retry_article_ids and args.restart:
        raise ValueError("--retry-article-ids 不能与 --restart 同时使用")
    if retry_article_ids and not chunk_dir.is_dir():
        raise FileNotFoundError("定点重跑需要既有 Phase 4 chunk；当前未找到 data/claim_graph/chunks/claims")
    if args.restart:
        logger.info("[步骤 0/5] 清空旧的 Phase 4 chunk、Claim 表和失败表")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        output_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        if state_path.exists():
            reset = connect_state(state_path)
            reset.execute("DELETE FROM claim_state")
            reset.commit()
            reset.close()
    provider = make_provider(args)
    requested_ids = set(retry_article_ids) if retry_article_ids else None
    papers = load_inputs(args.abstracts, args.sentences, args.limit, requested_ids)
    if not papers:
        raise ValueError("没有可抽取的摘要-句子对")
    loaded_ids = {str(paper["abstract"]["article_id"]) for paper in papers}
    missing_ids = requested_ids - loaded_ids if requested_ids is not None else set()
    if missing_ids:
        raise ValueError(f"指定论文在 Phase 3 输入中不存在或没有摘要句子：{sorted(missing_ids)}")
    state = connect_state(state_path)
    if retry_article_ids:
        logger.info("[步骤 0/5] 定点重跑 %d 篇论文；仅清除其旧 chunk 和断点", len(retry_article_ids))
        for article_id in retry_article_ids:
            (chunk_dir / f"{article_id}.json").unlink(missing_ok=True)
        state.executemany(
            "DELETE FROM claim_state WHERE article_id=?",
            ((article_id,) for article_id in retry_article_ids),
        )
        state.commit()
    completed = {
        row[0]
        for row in state.execute("SELECT article_id FROM claim_state WHERE status='complete'")
        if row[0] in loaded_ids
    } if args.resume and not retry_article_ids else set()
    if not args.resume and not args.restart and not retry_article_ids and completed:
        state.close()
        raise FileExistsError("已有 Claim 断点；继续请传 --resume，重建请传 --restart")
    pending = [paper for paper in papers if str(paper["abstract"]["article_id"]) not in completed]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or min(64, max(16, (os.cpu_count() or 1) * 2))
    logger.info("[步骤 1/5] 提供方=%s，摘要=%d，已完成=%d，待处理=%d，并发=%d", args.provider, len(papers), len(completed), len(pending), workers)
    logger.info("[步骤 2/5] 开始并行调用模型；每篇成功后立即写入独立 chunk，随时可断点续跑")
    started = time.monotonic()
    success_count = len(completed)
    failure_count = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="claim-extract") as pool:
        futures = {pool.submit(extract_one, paper, provider, chunk_dir): paper["abstract"]["article_id"] for paper in pending}
        for future in as_completed(futures):
            result = future.result()
            status = str(result["status"])
            state.execute(
                "INSERT OR REPLACE INTO claim_state VALUES (?, ?, ?)",
                (result["article_id"], status, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            state.commit()
            if status == "complete":
                success_count += 1
            else:
                failure_count += 1
                logger.warning("论文 %s 失败：%s | %s", result["article_id"], result["reason"], result["detail"])
            total_done = success_count + failure_count
            if total_done % args.progress_every == 0 or total_done == len(papers):
                elapsed = max(time.monotonic() - started, 0.001)
                logger.info("[步骤 2/5] 进度：完成=%d/%d，成功=%d，失败=%d，速度=%.2f 篇/秒", total_done, len(papers), success_count, failure_count, total_done / elapsed)
    logger.info("[步骤 3/5] 合并所有已完成 Claim chunk 为 claim_nodes.parquet")
    chunks = list(iter_chunks(chunk_dir))
    claim_count = write_parquet((claim for chunk in chunks if chunk["status"] == "complete" for claim in chunk["claims"]), output_path)
    logger.info("[步骤 4/5] 写入失败记录 CSV")
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        for chunk in chunks:
            if chunk["status"] != "complete":
                writer.writerow({key: chunk.get(key, "") for key in FAILURE_COLUMNS})
    state.close()
    logger.info("[步骤 5/5] Phase 4 完成：成功论文=%d，Claim=%d，失败=%d", sum(chunk["status"] == "complete" for chunk in chunks), claim_count, sum(chunk["status"] != "complete" for chunk in chunks))
    logger.info("Claim 节点表：%s", output_path)
    logger.info("失败记录：%s", failure_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abstracts", type=Path, default=DEFAULT_ABSTRACTS)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--provider", choices=("codex", "deepseek"), default="codex")
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument("--codex-reasoning-effort", default="medium")
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--deepseek-reasoning-effort", choices=("low", "high", "max"), default="high")
    parser.add_argument(
        "--deepseek-base-url",
        default="https://tokens.tjjtsz.com/v1",
        help="OpenAI-compatible Chat Completions 网关根地址",
    )
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--workers", type=int, default=0, help="0=按 CPU 自动选择 16--64 线程")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--retry-article-ids",
        help="逗号分隔的 article ID；仅重跑这些论文并覆盖其 Claim，其他 chunk 保持不变",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run and present concise Chinese errors for manual execution."""
    args = build_parser().parse_args(argv)
    if args.progress_every <= 0:
        print("Phase 4 构建失败：--progress-every 必须为正数", file=sys.stderr)
        return 1
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, pa.ArrowException) as error:
        print(f"Phase 4 构建失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

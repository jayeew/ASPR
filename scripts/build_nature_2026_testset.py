#!/usr/bin/env python3
"""Build a validated 2026 Nature paper/peer-review Markdown test set."""

from __future__ import annotations

import argparse
import errno
import hashlib
import html
import json
import logging
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader
from pypdf.errors import PyPdfError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = Path("/mnt/d/dataset/nature_2026_testset")
DEFAULT_MARKDOWN_ROOT = PROJECT_ROOT / "data" / "nature_2026_testset"
NATURE_ARTICLE_ROOT = "https://www.nature.com/articles"
CROSSREF_ROOT = "https://api.crossref.org/journals"
ARTICLE_PATTERN = re.compile(
    r"^s(?P<journal>\d{5})-(?P<doi_year>\d{3})-(?P<number>\d{5})-[a-z0-9]{1,2}$"
)
REVIEW_LINK_PATTERN = re.compile(
    r'<a[^>]+data-track-label=["\']transparent peer review file["\'][^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
FALLBACK_REVIEW_LINK_PATTERN = re.compile(
    r'<a[^>]+href=["\']([^"\']+\.pdf)["\'][^>]*>[^<]*transparent peer review',
    re.IGNORECASE,
)
MOESM_PATTERN = re.compile(r"_MOESM(?P<number>\d+)_ESM\.pdf", re.IGNORECASE)
LOGGER = logging.getLogger("nature_2026_testset")
THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Journal:
    name: str
    journal_id: str
    issn: str
    weight: int


JOURNALS = (
    Journal("Nature Communications", "41467", "2041-1723", 150),
    Journal("Communications Biology", "42003", "2399-3642", 10),
    Journal("Communications Chemistry", "42004", "2399-3669", 10),
    Journal("Communications Physics", "42005", "2399-3650", 10),
    Journal("Communications Materials", "43246", "2662-4443", 10),
    Journal("Communications Earth & Environment", "43247", "2662-4435", 10),
)


@dataclass(frozen=True)
class Candidate:
    article_id: str
    article_no: str
    doi: str
    title: str
    publication_date: str
    journal_name: str
    journal_id: str
    issn: str
    doi_year: int


@dataclass(frozen=True)
class PdfAudit:
    page_count: int
    text_chars: int
    sha256: str
    byte_count: int


class CandidateRejected(RuntimeError):
    """Raised when either member of a paper/review pair fails validation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def session() -> requests.Session:
    current = getattr(THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        current.headers.update(
            {
                "User-Agent": "ASPR-GEAR-Nature-Testset/1.0 (research dataset builder)",
                "Accept-Language": "en-US,en;q=0.8",
            }
        )
        THREAD_LOCAL.session = current
    return current


def request(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    stream: bool = False,
    timeout: int = 60,
    retries: int = 4,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session().get(url, params=params, stream=stream, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except (requests.RequestException, TimeoutError) as error:
            last_error = error
            current = getattr(THREAD_LOCAL, "session", None)
            if current is not None:
                current.close()
                THREAD_LOCAL.session = None
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise CandidateRejected(f"request_failed: {url}: {last_error}")


def crossref_date(item: Mapping[str, Any]) -> str:
    for field in ("published-online", "published-print", "published", "created"):
        parts = item.get(field, {}).get("date-parts", [])
        if not parts or not parts[0]:
            continue
        values = [int(value) for value in parts[0][:3]]
        values.extend([1] * (3 - len(values)))
        try:
            return date(*values).isoformat()
        except ValueError:
            continue
    return ""


def parse_candidate(item: Mapping[str, Any], journal: Journal) -> Candidate | None:
    doi = str(item.get("DOI") or "").strip().lower()
    article_id = doi.removeprefix("10.1038/")
    match = ARTICLE_PATTERN.fullmatch(article_id)
    publication_date = crossref_date(item)
    if not match or match.group("journal") != journal.journal_id:
        return None
    if int(match.group("doi_year")) != 26:
        return None
    if not publication_date.startswith("2026-") or publication_date > today_iso():
        return None
    titles = item.get("title") or []
    title = html.unescape(str(titles[0] if titles else "")).strip()
    return Candidate(
        article_id=article_id,
        article_no=match.group("number"),
        doi=doi,
        title=title,
        publication_date=publication_date,
        journal_name=journal.name,
        journal_id=journal.journal_id,
        issn=journal.issn,
        doi_year=2000 + int(match.group("doi_year")),
    )


def discover_candidates(journal: Journal, maximum: int) -> list[Candidate]:
    response = request(
        f"{CROSSREF_ROOT}/{journal.issn}/works",
        params={
            "filter": "from-pub-date:2026-01-01,until-pub-date:" + today_iso(),
            "rows": min(maximum, 1000),
            "sort": "published",
            "order": "asc",
            "select": "DOI,title,published-online,published-print,published,created",
        },
        timeout=90,
    )
    items = response.json().get("message", {}).get("items", [])
    candidates = [
        candidate for item in items if (candidate := parse_candidate(item, journal))
    ]
    unique = {candidate.article_id: candidate for candidate in candidates}
    return sorted(
        unique.values(), key=lambda value: (value.publication_date, value.article_id)
    )


def extract_review_url(article_html: str) -> str:
    for pattern in (REVIEW_LINK_PATTERN, FALLBACK_REVIEW_LINK_PATTERN):
        match = pattern.search(article_html)
        if match:
            return html.unescape(match.group(1))
    raise CandidateRejected("transparent_peer_review_link_missing")


def download_pdf(url: str, target: Path, maximum_bytes: int) -> None:
    response = request(url, stream=True, timeout=90)
    content_type = response.headers.get("content-type", "").lower()
    temporary = target.with_suffix(target.suffix + ".part")
    byte_count = 0
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise CandidateRejected("pdf_exceeds_maximum_bytes")
            handle.write(chunk)
    signature = temporary.read_bytes()[:5]
    if byte_count < 10_000 or signature != b"%PDF-" or "html" in content_type:
        temporary.unlink(missing_ok=True)
        raise CandidateRejected(
            f"invalid_pdf_response: bytes={byte_count}, content_type={content_type}"
        )
    temporary.replace(target)


def normalize_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def extract_pdf(path: Path) -> tuple[list[str], PdfAudit]:
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise CandidateRejected("encrypted_pdf")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PyPdfError, OSError, ValueError) as error:
        raise CandidateRejected(f"pdf_parse_failed: {error}") from error
    text_chars = sum(len(page) for page in pages)
    return pages, PdfAudit(
        page_count=len(pages),
        text_chars=text_chars,
        sha256=sha256_file(path),
        byte_count=path.stat().st_size,
    )


def audit_paper(candidate: Candidate, path: Path) -> tuple[list[str], PdfAudit]:
    pages, audit = extract_pdf(path)
    text = "\n".join(pages[:3])
    if audit.page_count < 2 or audit.text_chars < 5_000:
        raise CandidateRejected(f"paper_text_too_short: {audit}")
    if normalize_identity(candidate.article_id) not in normalize_identity(text):
        raise CandidateRejected("paper_article_id_mismatch")
    return pages, audit


def review_markers(text: str) -> tuple[list[str], list[str], int]:
    positive_patterns = {
        "peer_review_file": (r"peer review file", 6),
        "reviewer_numbered": (r"reviewer\s*(?:#|no\.?|number)?\s*[1-9]", 6),
        "reviewer_report": (r"reviewer(?:'s)?\s+(?:comments?|reports?)", 4),
        "author_response": (
            r"(?:author|response|rebuttal).{0,30}(?:response|rebuttal|reviewer)",
            3,
        ),
    }
    negative_patterns = {
        "reporting_summary": r"reporting summary",
        "supplementary_information_only": r"supplementary information",
    }
    positive = [
        name
        for name, (pattern, _) in positive_patterns.items()
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    ]
    negative = [
        name
        for name, pattern in negative_patterns.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]
    score = sum(
        weight for name, (_, weight) in positive_patterns.items() if name in positive
    )
    return positive, negative, score


def audit_review(path: Path) -> tuple[list[str], PdfAudit, list[str], list[str], int]:
    pages, audit = extract_pdf(path)
    text = "\n".join(pages)
    positive, negative, score = review_markers(text)
    if audit.page_count < 1 or audit.text_chars < 1_000:
        raise CandidateRejected(f"review_text_too_short: {audit}")
    if "peer_review_file" not in positive or not {
        "reviewer_numbered",
        "reviewer_report",
    }.intersection(positive):
        raise CandidateRejected(f"review_markers_invalid: positive={positive}")
    return pages, audit, positive, negative, score


def markdown_document(candidate: Candidate, kind: str, pages: Sequence[str]) -> str:
    heading = candidate.title or candidate.article_id
    metadata = (
        f"# {heading}\n\n"
        f"- Document type: {kind}\n"
        f"- Journal: {candidate.journal_name}\n"
        f"- DOI: https://doi.org/{candidate.doi}\n"
        f"- Published: {candidate.publication_date}\n"
        f"- Article ID: {candidate.article_id}\n"
    )
    body = "\n\n".join(
        f"<!-- GEAR_PAGE: {index} -->\n\n{text}"
        for index, text in enumerate(pages, start=1)
    )
    return metadata + "\n" + body.strip() + "\n"


def stage_candidate(
    candidate: Candidate, stage_root: Path, maximum_bytes: int
) -> dict[str, Any]:
    candidate_root = Path(
        tempfile.mkdtemp(prefix=f"{candidate.article_id}-", dir=stage_root)
    )
    paper_pdf = candidate_root / f"{candidate.article_id}.pdf"
    review_pdf = candidate_root / f"{candidate.article_id}_r.pdf"
    try:
        page_html = request(
            f"{NATURE_ARTICLE_ROOT}/{candidate.article_id}", timeout=60
        ).text
        review_url = extract_review_url(page_html)
        paper_url = f"{NATURE_ARTICLE_ROOT}/{candidate.article_id}.pdf"
        download_pdf(paper_url, paper_pdf, maximum_bytes)
        download_pdf(review_url, review_pdf, maximum_bytes)
        paper_pages, paper_audit = audit_paper(candidate, paper_pdf)
        review_pages, review_audit, positive, negative, score = audit_review(review_pdf)
        atomic_text(
            candidate_root / "paper.md",
            markdown_document(candidate, "paper", paper_pages),
        )
        atomic_text(
            candidate_root / "peer_review.md",
            markdown_document(candidate, "peer_review", review_pages),
        )
        moesm = MOESM_PATTERN.search(review_url)
        return {
            "candidate": asdict(candidate),
            "stage_root": str(candidate_root),
            "paper_url": paper_url,
            "peer_review_url": review_url,
            "paper_audit": asdict(paper_audit),
            "peer_review_audit": asdict(review_audit),
            "peer_review_positive_matches": positive,
            "peer_review_negative_matches": negative,
            "peer_review_score": score,
            "peer_review_moesm_number": int(moesm.group("number")) if moesm else None,
        }
    except (CandidateRejected, requests.RequestException, OSError) as error:
        shutil.rmtree(candidate_root, ignore_errors=True)
        raise CandidateRejected(str(error)) from error


def final_paths(
    candidate: Candidate, raw_root: Path, markdown_root: Path
) -> dict[str, Path]:
    return {
        "paper_pdf": raw_root / "paper" / f"{candidate.article_id}.pdf",
        "review_pdf": raw_root / "peer_review" / f"{candidate.article_id}_r.pdf",
        "paper_md": markdown_root / "paper" / f"{candidate.article_id}.md",
        "review_md": markdown_root / "peer_review" / f"{candidate.article_id}_r.md",
    }


def publish_file(source: Path, target: Path) -> None:
    try:
        source.replace(target)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    temporary = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    source.unlink()


def commit_staged(
    result: Mapping[str, Any], raw_root: Path, markdown_root: Path
) -> dict[str, Any]:
    candidate = Candidate(**result["candidate"])
    source_root = Path(str(result["stage_root"]))
    paths = final_paths(candidate, raw_root, markdown_root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    publish_file(source_root / f"{candidate.article_id}.pdf", paths["paper_pdf"])
    publish_file(source_root / f"{candidate.article_id}_r.pdf", paths["review_pdf"])
    publish_file(source_root / "paper.md", paths["paper_md"])
    publish_file(source_root / "peer_review.md", paths["review_md"])
    shutil.rmtree(source_root, ignore_errors=True)
    project_paper = paths["paper_md"].resolve()
    project_review = paths["review_md"].resolve()
    return {
        **asdict(candidate),
        "year": 2026,
        "pair_key": f"{candidate.journal_id}_2026_{candidate.article_no}",
        "article_pdf_path": str(paths["paper_pdf"].resolve()),
        "peer_review_pdf_path": str(paths["review_pdf"].resolve()),
        "paper_markdown_path": str(paths["paper_md"].resolve()),
        "peer_review_markdown_path": str(paths["review_md"].resolve()),
        "project_paper_markdown_link": str(project_paper.resolve()),
        "project_peer_review_markdown_link": str(project_review.resolve()),
        "paper_write_status": "created",
        "peer_review_write_status": "created",
        "paper_url": result["paper_url"],
        "peer_review_url": result["peer_review_url"],
        "paper_audit": result["paper_audit"],
        "peer_review_audit": result["peer_review_audit"],
        "peer_review_positive_matches": result["peer_review_positive_matches"],
        "peer_review_negative_matches": result["peer_review_negative_matches"],
        "peer_review_score": result["peer_review_score"],
        "peer_review_moesm_number": result["peer_review_moesm_number"],
        "peer_review_alternatives": [],
        "accepted_at": utc_now(),
    }


def discard_staged(result: Mapping[str, Any]) -> None:
    shutil.rmtree(Path(str(result["stage_root"])), ignore_errors=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def valid_existing(row: Mapping[str, Any]) -> bool:
    required = (
        "article_pdf_path",
        "peer_review_pdf_path",
        "paper_markdown_path",
        "peer_review_markdown_path",
    )
    return all(Path(str(row.get(field) or "")).is_file() for field in required)


def quota_for(journal: Journal, target: int) -> int:
    base = max(
        1, round(target * journal.weight / sum(item.weight for item in JOURNALS))
    )
    return base


def normalized_quotas(target: int) -> dict[str, int]:
    if target < len(JOURNALS):
        raise ValueError(f"target must be at least {len(JOURNALS)}")
    quotas = {journal.journal_id: quota_for(journal, target) for journal in JOURNALS}
    difference = target - sum(quotas.values())
    quotas[JOURNALS[0].journal_id] += difference
    return quotas


def process_journal(
    journal: Journal,
    candidates: Sequence[Candidate],
    quota: int,
    accepted: list[dict[str, Any]],
    rejected_ids: set[str],
    raw_root: Path,
    markdown_root: Path,
    rejected_path: Path,
    workers: int,
    maximum_bytes: int,
) -> None:
    existing_ids = {str(row["article_id"]) for row in accepted}
    existing_count = sum(
        row.get("journal_id") == journal.journal_id for row in accepted
    )
    remaining = quota - existing_count
    queue = [
        candidate
        for candidate in candidates
        if candidate.article_id not in existing_ids | rejected_ids
    ]
    LOGGER.info(
        "%s: existing=%d target=%d candidates=%d",
        journal.name,
        existing_count,
        quota,
        len(queue),
    )
    cursor = 0
    while remaining > 0 and cursor < len(queue):
        batch = queue[
            cursor : cursor + min(max(workers * 2, remaining), remaining + workers)
        ]
        cursor += len(batch)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Future[dict[str, Any]], Candidate] = {
                executor.submit(
                    stage_candidate, candidate, raw_root / ".staging", maximum_bytes
                ): candidate
                for candidate in batch
            }
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    result = future.result()
                except CandidateRejected as error:
                    rejected_ids.add(candidate.article_id)
                    append_jsonl(
                        rejected_path,
                        {
                            **asdict(candidate),
                            "reason": str(error),
                            "rejected_at": utc_now(),
                        },
                    )
                    LOGGER.info("rejected %s: %s", candidate.article_id, error)
                    continue
                if remaining <= 0:
                    discard_staged(result)
                    continue
                row = commit_staged(result, raw_root, markdown_root)
                append_jsonl(markdown_root / "manifest.jsonl", row)
                accepted.append(row)
                remaining -= 1
                LOGGER.info(
                    "accepted %s (%s): %d/%d",
                    candidate.article_id,
                    journal.name,
                    quota - remaining,
                    quota,
                )
    if remaining > 0:
        raise RuntimeError(
            f"{journal.name} exhausted candidates with {remaining} pairs still needed"
        )


def write_summary(
    raw_root: Path,
    markdown_root: Path,
    accepted: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
) -> None:
    counts = {
        journal.name: sum(
            row.get("journal_id") == journal.journal_id for row in accepted
        )
        for journal in JOURNALS
    }
    summary = {
        "schema_version": "nature_2026_paired_testset_v1",
        "year": 2026,
        "pair_count": len(accepted),
        "journal_count": sum(value > 0 for value in counts.values()),
        "journal_counts": counts,
        "journal_quotas": dict(quotas),
        "validation": {
            "paper": "PDF signature, parseable pages, >=5000 text chars, article ID match",
            "peer_review": "publisher transparent-review link, parseable PDF, >=1000 text chars, review markers",
            "pair_policy": "both members must pass; otherwise the candidate is rejected",
        },
        "raw_pdf_root": str(raw_root.resolve()),
        "markdown_root": str(markdown_root.resolve()),
        "manifest_sha256": sha256_file(markdown_root / "manifest.jsonl"),
        "completed_at": utc_now(),
    }
    atomic_text(
        markdown_root / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )


def build(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.resolve()
    markdown_root = args.markdown_root.resolve()
    if args.fresh:
        for directory in (
            raw_root / "paper",
            raw_root / "peer_review",
            raw_root / ".staging",
            markdown_root / "paper",
            markdown_root / "peer_review",
        ):
            if directory.exists():
                shutil.rmtree(directory)
        for path in (
            markdown_root / "manifest.jsonl",
            markdown_root / "rejected.jsonl",
            markdown_root / "summary.json",
        ):
            path.unlink(missing_ok=True)
    for root in (raw_root, markdown_root):
        root.mkdir(parents=True, exist_ok=True)
    staging_root = raw_root / ".staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest_path = markdown_root / "manifest.jsonl"
    rejected_path = markdown_root / "rejected.jsonl"
    accepted = [row for row in read_jsonl(manifest_path) if valid_existing(row)]
    if len(accepted) != len(read_jsonl(manifest_path)):
        raise RuntimeError(
            "manifest contains rows with missing files; repair them before resuming"
        )
    if len(accepted) > args.target:
        raise RuntimeError("existing manifest already exceeds requested target")
    rejected_ids = {str(row["article_id"]) for row in read_jsonl(rejected_path)}
    quotas = normalized_quotas(args.target)
    for journal in JOURNALS:
        maximum = min(
            1000, max(args.candidates_per_journal, quotas[journal.journal_id] * 5)
        )
        candidates = discover_candidates(journal, maximum)
        process_journal(
            journal,
            candidates,
            quotas[journal.journal_id],
            accepted,
            rejected_ids,
            raw_root,
            markdown_root,
            rejected_path,
            args.workers,
            args.maximum_bytes,
        )
    if len(accepted) != args.target:
        raise RuntimeError(
            f"expected {args.target} accepted pairs, found {len(accepted)}"
        )
    write_summary(raw_root, markdown_root, accepted, quotas)
    LOGGER.info("complete: %d validated pairs", len(accepted))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidates-per-journal", type=int, default=100)
    parser.add_argument("--maximum-bytes", type=int, default=200_000_000)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    build(args)


if __name__ == "__main__":
    main()

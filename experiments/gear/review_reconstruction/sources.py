"""Deterministic source-role parsing for transparent peer-review files."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from gear.review_contracts import ReviewModel


class SourceKind(str, Enum):
    REVIEWER_REPORT = "reviewer_report"
    AUTHOR_RESPONSE = "author_response"
    EDITORIAL = "editorial"


class ReviewSourceSpan(ReviewModel):
    source_span_id: str
    paper_id: str
    source_kind: SourceKind
    source_path: Path
    reviewer_id_hash: str | None = None
    round_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    text_sha256: str

    @model_validator(mode="after")
    def validate_span(self) -> "ReviewSourceSpan":
        if self.char_end < self.char_start:
            raise ValueError("source span end precedes its start")
        expected = _sha256(self.text)
        if self.text_sha256 != expected:
            raise ValueError("source span text hash mismatch")
        if self.source_kind == SourceKind.REVIEWER_REPORT and not self.reviewer_id_hash:
            raise ValueError("reviewer reports require a reviewer identifier")
        return self


_PARAGRAPH_RE = re.compile(r"\S.*?(?=\n[ \t]*\n|\Z)", re.S)
_REVIEWER_RE = re.compile(r"\b(?:reviewer|referee)\s*(?:#|no\.?\s*)?(\d+|[IVX]+)\b", re.I)
_RESPONSE_RE = re.compile(
    r"\b(?:author\s+)?(?:response|reply|rebuttal)\b|\bresponses?\s+to\s+reviewers?\b",
    re.I,
)
_ROUND_RE = re.compile(r"\b(?:round|version|revision)\s*(\d+)\b", re.I)
_AUTHOR_CUE_RE = re.compile(
    r"^\s*(?:response|reply)\s*:|^\s*we\s+(?:thank|agree|have|added|revised|"
    r"clarified|changed|removed|corrected|addressed|performed|included)\b",
    re.I,
)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _paragraphs(text: str):
    for match in _PARAGRAPH_RE.finditer(text):
        paragraph = match.group(0).rstrip()
        if paragraph.strip():
            yield match.start(), match.start() + len(paragraph), paragraph


class NatureTransparentReviewParser:
    """Separate reviewer opinions from author responses without model inference."""

    def parse(self, paper_id: str, path: Path) -> list[ReviewSourceSpan]:
        source_path = Path(path).resolve()
        text = source_path.read_text(encoding="utf-8", errors="replace")
        spans: list[ReviewSourceSpan] = []
        kind = SourceKind.EDITORIAL
        reviewer_id: str | None = None
        round_number = 1
        seen: set[tuple[str, str, int, str]] = set()
        for index, (start, end, paragraph) in enumerate(_paragraphs(text)):
            heading = paragraph.strip().strip("#*_ ")
            if len(heading) <= 180:
                round_match = _ROUND_RE.search(heading)
                if round_match:
                    round_number = int(round_match.group(1))
                if _RESPONSE_RE.search(heading):
                    kind = SourceKind.AUTHOR_RESPONSE
                else:
                    reviewer_match = _REVIEWER_RE.search(heading)
                    if reviewer_match:
                        reviewer_id = _stable_id(
                            "REV-", paper_id, reviewer_match.group(1).casefold()
                        )
                        kind = SourceKind.REVIEWER_REPORT
                    elif re.search(r"\b(?:editor|decision)\b", heading, re.I):
                        kind = SourceKind.EDITORIAL
                        reviewer_id = None
            if kind == SourceKind.REVIEWER_REPORT and _AUTHOR_CUE_RE.search(paragraph):
                kind = SourceKind.AUTHOR_RESPONSE
            digest = _sha256(paragraph)
            identity = (kind.value, reviewer_id or "", round_number, digest)
            if identity in seen:
                continue
            seen.add(identity)
            spans.append(
                ReviewSourceSpan(
                    source_span_id=_stable_id(
                        "SRC-", paper_id, index, start, kind.value, digest
                    ),
                    paper_id=paper_id,
                    source_kind=kind,
                    source_path=source_path,
                    reviewer_id_hash=(
                        reviewer_id
                        if kind in {SourceKind.REVIEWER_REPORT, SourceKind.AUTHOR_RESPONSE}
                        else None
                    ),
                    round_id=f"round_{round_number}",
                    char_start=start,
                    char_end=end,
                    text=paragraph,
                    text_sha256=digest,
                )
            )
        return spans


__all__ = ["NatureTransparentReviewParser", "ReviewSourceSpan", "SourceKind"]

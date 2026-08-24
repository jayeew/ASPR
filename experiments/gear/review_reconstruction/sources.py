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
_REVIEWER_RE = re.compile(
    r"\b(?:reviewer|referee)\s*(?:#|no\.?\s*)?(\d+|[IVX]+)\b", re.I
)
_RESPONSE_RE = re.compile(
    r"\b(?:author\s+)?(?:response|reply|rebuttal)\b|\bresponses?\s+to\s+reviewers?\b",
    re.I,
)
_ROUND_RE = re.compile(r"\b(?:round|version|revision)\s*(\d+)\b", re.I)
_AUTHOR_CUE_RE = re.compile(
    r"^\s*(?:response|reply)\s*:|^\s*(?:a\d+(?:\.\d+)*)\s*[:.)-]|"
    r"^\s*we\s+(?:thank|would\s+like\s+to\s+thank|agree|have|added|revised|"
    r"clarified|changed|removed|corrected|addressed|performed|included)\b",
    re.I,
)
_REVIEWER_QUESTION_RE = re.compile(
    r"^\s*(?:q|question)\s*\d+(?:\.\d+)*\s*[:.)-]",
    re.I,
)


class SpeakerRoleAudit(ReviewModel):
    """Deterministic role-separation signals for reconstruction eligibility."""

    reviewer_span_count: int = Field(default=0, ge=0)
    author_span_count: int = Field(default=0, ge=0)
    reviewer_character_count: int = Field(default=0, ge=0)
    author_character_count: int = Field(default=0, ge=0)
    reviewer_question_count: int = Field(default=0, ge=0)
    author_label_count: int = Field(default=0, ge=0)
    generic_approval_count: int = Field(default=0, ge=0)
    status: str = "automated_valid"
    reasons: list[str] = Field(default_factory=list)


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
            if _REVIEWER_QUESTION_RE.search(paragraph):
                kind = SourceKind.REVIEWER_REPORT
                if reviewer_id is None:
                    reviewer_id = _stable_id("REV-", paper_id, "unattributed")
            elif _AUTHOR_CUE_RE.search(paragraph):
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
                        if kind
                        in {SourceKind.REVIEWER_REPORT, SourceKind.AUTHOR_RESPONSE}
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


def audit_source_roles(spans: list[ReviewSourceSpan]) -> SpeakerRoleAudit:
    """Produce conservative, deterministic eligibility metadata for one source."""
    reviewer = [row for row in spans if row.source_kind == SourceKind.REVIEWER_REPORT]
    author = [row for row in spans if row.source_kind == SourceKind.AUTHOR_RESPONSE]
    generic = sum(
        bool(
            re.search(
                r"\b(?:thank you for all the corrections|nice job of responding|"
                r"addressed my questions|no other concerns)\b",
                row.text,
                re.I,
            )
        )
        for row in reviewer
    )
    reasons: list[str] = []
    if not reviewer:
        reasons.append("no_reviewer_spans")
    if sum(len(row.text) for row in reviewer) < 1000:
        reasons.append("reviewer_channel_too_short")
    status = "automated_valid" if not reasons else "automated_limited"
    return SpeakerRoleAudit(
        reviewer_span_count=len(reviewer),
        author_span_count=len(author),
        reviewer_character_count=sum(len(row.text) for row in reviewer),
        author_character_count=sum(len(row.text) for row in author),
        reviewer_question_count=sum(
            bool(_REVIEWER_QUESTION_RE.search(row.text)) for row in reviewer
        ),
        author_label_count=sum(bool(_AUTHOR_CUE_RE.search(row.text)) for row in author),
        generic_approval_count=generic,
        status=status,
        reasons=reasons,
    )


__all__ = [
    "NatureTransparentReviewParser",
    "ReviewSourceSpan",
    "SourceKind",
    "SpeakerRoleAudit",
    "audit_source_roles",
]

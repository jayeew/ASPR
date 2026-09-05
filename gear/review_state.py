"""Build innovation-only human reference state from peer-review Markdown."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gear.config import GearConfig

from .review_contracts import (
    DiscussionResolvedReference,
    ReviewerClaim,
    ReviewerStance,
    ReviewerView,
)
from gear.artifacts import write_jsonl, write_model
from gear.model_client import LazyRoleClient


ROUND_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:version|round)\s*(\d+)\s*:?.*$"
)
REVIEWER_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?reviewer\s*#?\s*(?!comments?\b)([\w.-]+)"
    r"(?:\s*\([^\n]*\))?\s*:?\s*$"
)
AUTHOR_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?:response\s+to\s+reviewers|response\s+letter|author\s+response|rebuttal)"
    r"\s*:?.*$"
)


def split_review_document(text: str) -> list[dict[str, object]]:
    author_match = AUTHOR_PATTERN.search(text)
    report_end = author_match.start() if author_match else len(text)
    report_text = text[:report_end]
    markers: list[tuple[int, str, str]] = [
        (match.start(), "round", match.group(1))
        for match in ROUND_PATTERN.finditer(report_text)
    ]
    markers.extend(
        (match.start(), "reviewer", match.group(1))
        for match in REVIEWER_PATTERN.finditer(report_text)
    )
    markers.sort(key=lambda item: item[0])
    blocks: list[dict[str, object]] = []
    round_number, reviewer_id, role = 1, "unknown", "reviewer"
    for index, (start, kind, value) in enumerate(markers):
        if kind == "round":
            round_number = int(value)
            continue
        reviewer_id, role = value, "reviewer"
        end = markers[index + 1][0] if index + 1 < len(markers) else report_end
        body = text[start:end].strip()
        if body:
            blocks.append({"block_id": f"B{len(blocks)+1:04d}", "round_number": round_number, "reviewer_id": reviewer_id, "role": role, "text": body})
    if not blocks and report_text.strip():
        blocks.append({"block_id": "B0001", "round_number": 1, "reviewer_id": "unknown", "role": "reviewer", "text": report_text.strip()})
    if author_match:
        blocks.append({
            "block_id": f"B{len(blocks)+1:04d}",
            "round_number": round_number,
            "reviewer_id": "authors",
            "role": "author",
            "text": text[author_match.start():].strip(),
        })
    return blocks


def _bind_quote(source: str, quote: str) -> str | None:
    """Return the exact source slice while tolerating PDF-induced whitespace."""
    if quote in source:
        return quote
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source)
    return match.group(0) if match else None


class ReviewerReferenceBuilder:
    """Reviewer views exclude author text; discussion resolution uses it conservatively."""

    def __init__(self, config: GearConfig) -> None:
        self.client = LazyRoleClient(config, "relation_fusion")

    def build(self, paper_id: str, review_path: Path, output_dir: Path) -> tuple[list[ReviewerView], DiscussionResolvedReference]:
        blocks = split_review_document(review_path.read_text(encoding="utf-8"))
        reviewer_blocks = [x for x in blocks if x["role"] == "reviewer"]
        claims = self._extract(paper_id, reviewer_blocks)
        views = self._views(paper_id, claims)
        resolved_claims = [claim for view in views for claim in view.claims]
        resolved = self._resolve(paper_id, resolved_claims, [x for x in blocks if x["role"] == "author"])
        write_jsonl(output_dir / "reviewer_views.jsonl", views)
        write_model(output_dir / "discussion_resolved_reference.json", resolved)
        write_jsonl(output_dir / "review_blocks.jsonl", blocks)
        return views, resolved

    def _extract(self, paper_id: str, blocks: list[dict[str, object]]) -> list[ReviewerClaim]:
        raw = self.client.generate_json(
            system="""Extract innovation judgments from every reviewer block and round independently.
For each judgment, separate the neutral scientific contribution being evaluated from the reviewer's stance.
target_claim_text must be an atomic, concrete, author-side contribution proposition about the target manuscript, recoverable from that reviewer's surrounding text. A prior-work fact is evidence for the stance, never the target: for example, if a reviewer says an effect was already observed in atoms, write the target manuscript's first-observation claim and mark it challenged or incremental_or_limited, rather than using the atom study as target_claim_text. The target must not contain evaluative phrases such as novel, important, sufficient novelty, fills a gap, the paper, the manuscript, or the work. Include a record only when the reviewer explicitly discusses firstness, novelty, originality, conceptual contribution, a gap, duplication of prior work, or a specific new increment over prior work. Questions about correctness, causal support, controls, effect size, interpretation, or requested experiments are not innovation judgments and must be excluded even when they concern a contribution claim. Use recognized, incremental_or_limited, challenged, or unresolved for stance. Preserve conflicting reviewers. If a judgment is completely generic and no scientific target can be recovered, omit it. source_quote must quote the exact judgment-bearing source text from source_block_id. Return JSON.""",
            user=json.dumps({"paper_id": paper_id, "reviewer_blocks": blocks}, ensure_ascii=False),
            response_schema={"type": "object", "properties": {"claims": {"type": "array", "items": {
                "type": "object", "properties": {
                    "target_claim_text": {"type": "string"},
                    "stance": {"type": "string", "enum": [x.value for x in ReviewerStance]},
                    "source_block_id": {"type": "string"}, "source_quote": {"type": "string"},
                }, "required": ["target_claim_text", "stance", "source_block_id", "source_quote"], "additionalProperties": False,
            }}}, "required": ["claims"], "additionalProperties": False},
        )
        block_map = {str(x["block_id"]): x for x in blocks}
        output = []
        for index, row in enumerate(raw.get("claims", []), 1):
            block = block_map.get(str(row.get("source_block_id")))
            quote = str(row.get("source_quote", "")).strip()
            target = str(row.get("target_claim_text", "")).strip()
            bound_quote = _bind_quote(str(block["text"]), quote) if block and quote else None
            if block is None or bound_quote is None or not target:
                continue
            output.append(ReviewerClaim(
                reviewer_claim_id=f"{paper_id}::HUMAN::{index:03d}", paper_id=paper_id,
                reviewer_id=str(block["reviewer_id"]), round_number=int(block["round_number"]),
                target_claim_text=target,
                stance=ReviewerStance(str(row.get("stance", "unresolved"))),
                source_block_id=str(block["block_id"]), source_quote=bound_quote,
            ))
        return output

    def _views(self, paper_id: str, claims: list[ReviewerClaim]) -> list[ReviewerView]:
        reviewers = sorted({x.reviewer_id for x in claims})
        output = []
        for reviewer in reviewers:
            rows = sorted((x for x in claims if x.reviewer_id == reviewer), key=lambda x: x.round_number)
            raw = self.client.generate_json(
                system="Group statements by the same underlying innovation point across rounds. For each group retain the reviewer_claim_id of the last explicit stance. Silence in a later round is not an update. Return retained IDs only.",
                user=json.dumps({"claims": [x.model_dump(mode="json") for x in rows]}, ensure_ascii=False),
                response_schema={"type": "object", "properties": {"retained_claim_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["retained_claim_ids"], "additionalProperties": False},
            )
            row_map = {x.reviewer_claim_id: x for x in rows}
            retained = [row_map[str(x)] for x in raw["retained_claim_ids"] if str(x) in row_map]
            output.append(ReviewerView(paper_id=paper_id, reviewer_id=reviewer, claims=retained or rows))
        return output

    def _resolve(self, paper_id: str, claims: list[ReviewerClaim], author_blocks: list[dict[str, object]]) -> DiscussionResolvedReference:
        if not author_blocks:
            return DiscussionResolvedReference(paper_id=paper_id, claims=claims)
        raw = self.client.generate_json(
            system="For each existing reviewer_claim_id, summarize whether author replies address it. An author reply cannot create a positive novelty judgment or change stance without a later explicit reviewer statement. Return notes only.",
            user=json.dumps({"reviewer_claims": [x.model_dump(mode="json") for x in claims], "author_blocks": author_blocks}, ensure_ascii=False),
            response_schema={"type": "object", "properties": {"resolutions": {
                "type": "array", "items": {
                    "type": "object", "properties": {
                        "reviewer_claim_id": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["reviewer_claim_id", "note"],
                    "additionalProperties": False,
                },
            }}, "required": ["resolutions"], "additionalProperties": False},
        )
        known = {x.reviewer_claim_id for x in claims}
        notes = {
            str(row["reviewer_claim_id"]): str(row["note"])
            for row in raw["resolutions"]
            if str(row["reviewer_claim_id"]) in known
        }
        return DiscussionResolvedReference(paper_id=paper_id, claims=claims, resolution_notes=notes)

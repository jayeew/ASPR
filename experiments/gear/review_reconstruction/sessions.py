"""Build and validate isolated Codex/ChatGPT reconstruction handoffs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from gear.config import GearConfig, load_config
from gear.contracts import PaperMetadata, ReviewRequest
from gear.paper_compiler import PaperCompiler
from gear.review_contracts import ContextClaim, ContextSpan
from gear.trace import sha256_file, sha256_value

from .contracts import (
    ReconstructionPaperContext,
    ReconstructionSessionPackage,
    ReconstructionSessionResponse,
    ReviewSourceExcerpt,
    ReviewSourceRole,
)
from .sources import NatureTransparentReviewParser, SourceKind, audit_source_roles

RECONSTRUCTION_INSTRUCTIONS = """Reconstruct the final-state peer review for one paper.
Use reviewer_report spans as the only source of review opinions. Author responses may
only determine whether an issue was resolved; they cannot create or rewrite a review
point. Remove greetings, publication/acceptance recommendations, editorial decisions,
and generic praise. Merge duplicates, preserve real reviewer disagreement, and produce
at most 24 atomic points in StructuredReview. Include contribution summary, novelty
support/limits, strengths, weaknesses, and questions. Bind every retained point to at
least one reviewer quote key and one final PaperIR P:S-* span. Put resolved issues only
in ReferenceTrace/revision_ledger. Keep partially_resolved or persists issues only
when the residual is independently supported by the final paper. Unverifiable items are
audit-only. Do not use graph signals, GEAR output, old reconstructions, ratings,
decisions, recommendations, or accept/reject language. Return only the completed
ReconstructionSessionResponse JSON file. Set external_verification_required=true
for every novelty/prior-art supporting or limiting point.
"""

GENERIC_PUBLICATION_RE = re.compile(
    r"\b(?:recommend(?:ed|ation)?\s+(?:this\s+)?(?:paper\s+)?(?:for\s+)?"
    r"publication|suitable for publication|accept(?:ed|ance)?|reject(?:ed)?|"
    r"recommend(?:ed|ation)?\s+(?:this\s+)?(?:paper\s+)?(?:for\s+)?rejection)\b",
    re.IGNORECASE,
)
GENERIC_APPROVAL_RE = re.compile(
    r"\b(?:thank you for all the corrections|nice job of responding|"
    r"addressed my questions|no other concerns|suitable for publication)\b",
    re.IGNORECASE,
)


def build_reconstruction_package(
    case: Mapping[str, Any],
    *,
    config: GearConfig | None = None,
) -> tuple[ReconstructionSessionPackage, Any]:
    """Compile one Nature pair into a graph-blind reconstruction package."""
    paper_path, review_path = _case_paths(case)
    metadata = PaperMetadata(
        title=str(case.get("title") or ""),
        doi=str(case.get("doi") or "") or _extract_doi(paper_path),
        publication_date=_optional_date(case.get("publication_date")),
        submission_date=_optional_date(case.get("submission_date")),
        venue=str(case.get("domain") or "") or None,
    )
    paper_ir = PaperCompiler(config or load_config()).compile(
        ReviewRequest(paper_path=paper_path, metadata=metadata)
    )
    parsed = NatureTransparentReviewParser().parse(paper_ir.paper_id, review_path)
    speaker_role_audit = audit_source_roles(parsed).model_dump(mode="json")
    reviewer_spans = [
        _source_excerpt(span)
        for span in parsed
        if span.source_kind == SourceKind.REVIEWER_REPORT
    ]
    author_spans = [
        _source_excerpt(span)
        for span in parsed
        if span.source_kind == SourceKind.AUTHOR_RESPONSE
    ]
    if not reviewer_spans:
        raise ValueError(
            "Nature pair has no deterministically separated reviewer spans"
        )
    context = ReconstructionPaperContext(
        paper_id=paper_ir.paper_id,
        paper_sha256=paper_ir.paper_sha256,
        claims=[
            ContextClaim(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type.value,
                evidence_key=f"P:{claim.span_id}",
                text=claim.text,
            )
            for claim in paper_ir.claims
        ],
        spans=[
            ContextSpan(
                evidence_key=f"P:{span.span_id}",
                span_id=span.span_id,
                page=span.page,
                section_path=list(span.section_path),
                text=span.text,
                text_sha256=span.text_sha256,
            )
            for span in paper_ir.spans
        ],
    )
    package = _build_package(
        paper_context=context,
        review_source_sha256=sha256_file(review_path),
        reviewer_spans=reviewer_spans,
        author_spans=author_spans,
        speaker_role_audit=speaker_role_audit,
        instructions=RECONSTRUCTION_INSTRUCTIONS,
    )
    return package, paper_ir


def validate_session_response(
    package: ReconstructionSessionPackage,
    response: ReconstructionSessionResponse,
) -> None:
    """Fail on drift, cross-session reuse, unknown IDs, or target leakage."""
    validate_session_package(package)
    expected = (
        response.package_id == package.package_id
        and response.session_kind == package.session_kind
        and response.paper_id == package.paper_id
        and response.prompt_sha256 == package.prompt_sha256
        and response.schema_sha256 == package.schema_sha256
        and response.input_sha256 == package.input_sha256
    )
    if not expected:
        raise ValueError("session response identity/hash drift")
    if response.output_sha256 != sha256_value(response.hash_payload()):
        raise ValueError("session response output hash mismatch")
    reviewer_map = {span.source_key: span for span in package.reviewer_spans}
    author_map = {span.source_key: span for span in package.author_response_spans}
    paper_keys = {span.evidence_key for span in package.paper_context.spans}
    used_paper_keys = {
        *response.review.summary.evidence_keys,
        *(key for point in response.review.all_points() for key in point.evidence_keys),
    }
    if not used_paper_keys.issubset(paper_keys):
        raise ValueError("reconstructed review contains unknown final-paper evidence")
    if any(
        not point.evidence_keys
        or any(not key.startswith("P:") for key in point.evidence_keys)
        for point in response.review.all_points()
    ):
        raise ValueError("every reconstructed point requires final-paper evidence")
    point_ids = {point.point_id for point in response.review.all_points()}
    traced_ids: set[str] = set()
    for trace in response.reference_traces:
        if trace.paper_id != package.paper_id:
            raise ValueError("reference trace paper_id mismatch")
        if not set(trace.reviewer_quote_keys).issubset(reviewer_map):
            raise ValueError("reference trace contains unknown/non-reviewer quote key")
        if not set(trace.author_response_keys).issubset(author_map):
            raise ValueError("reference trace contains unknown author-response key")
        if not set(trace.final_paper_evidence_keys).issubset(paper_keys):
            raise ValueError("reference trace contains unknown final-paper evidence")
        quote_rows = [reviewer_map[key] for key in trace.reviewer_quote_keys]
        if not set(trace.round_ids).issubset({row.round_id for row in quote_rows}):
            raise ValueError("trace round IDs do not follow reviewer quotes")
        if not set(trace.reviewer_id_hashes).issubset(
            {str(row.reviewer_id_hash) for row in quote_rows}
        ):
            raise ValueError("trace reviewer IDs do not follow reviewer quotes")
        if trace.point_id:
            if all(GENERIC_APPROVAL_RE.search(row.text) for row in quote_rows):
                raise ValueError(
                    "generic reviewer approval cannot entail a reconstructed point"
                )
            traced_ids.add(trace.point_id)
    if traced_ids != point_ids:
        raise ValueError("every retained review point requires exactly trace coverage")
    for entry in response.revision_ledger:
        if entry.paper_id != package.paper_id:
            raise ValueError("revision ledger paper_id mismatch")
        if not set(entry.reviewer_quote_keys).issubset(reviewer_map):
            raise ValueError("revision ledger contains unknown reviewer quote key")
        if not set(entry.author_response_keys).issubset(author_map):
            raise ValueError("revision ledger contains unknown author-response key")
        if not set(entry.final_paper_evidence_keys).issubset(paper_keys):
            raise ValueError("revision ledger contains unknown final-paper evidence")
    texts = [
        response.review.summary.text,
        *(point.text for point in response.review.all_points()),
    ]
    if any(GENERIC_PUBLICATION_RE.search(text) for text in texts):
        raise ValueError(
            "publication/decision language leaked into reconstructed target"
        )


def write_session_handoff(
    package: ReconstructionSessionPackage,
    paper_ir: Any | None,
    output_dir: Path,
) -> Path:
    """Write one self-contained handoff; existing conflicting files are rejected."""
    validate_session_package(package)
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    _write_once(target / "package.json", package.model_dump_json(indent=2) + "\n")
    if paper_ir is not None:
        _write_once(target / "paper_ir.json", paper_ir.model_dump_json(indent=2) + "\n")
    _write_once(target / "PROMPT.md", package.instructions.strip() + "\n")
    _write_once(
        target / "response.schema.json",
        json.dumps(
            ReconstructionSessionResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    template = _response_template(package)
    _write_once(
        target / "response.template.json",
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
    )
    return target


def import_session_response(
    package_path: Path,
    response_path: Path,
    output_dir: Path,
) -> Path:
    package = ReconstructionSessionPackage.model_validate_json(
        Path(package_path).read_text(encoding="utf-8")
    )
    response = ReconstructionSessionResponse.model_validate_json(
        Path(response_path).read_text(encoding="utf-8")
    )
    validate_session_response(package, response)
    target = (
        Path(output_dir).resolve() / _safe_id(package.paper_id) / package.session_kind
    )
    target.mkdir(parents=True, exist_ok=True)
    frozen = response.model_dump_json(indent=2) + "\n"
    _write_once(target / "package.json", package.model_dump_json(indent=2) + "\n")
    _write_once(target / "response.json", frozen)
    manifest = {
        "contract": "reconstruction_import_manifest",
        "schema_version": "aspr_gear",
        "package_sha256": sha256_file(Path(package_path)),
        "response_file_sha256": sha256_file(Path(response_path)),
        "output_sha256": response.output_sha256,
        "paper_id": response.paper_id,
        "model_id": response.model_id,
        "conversation_hash": response.conversation_hash,
        "session_kind": response.session_kind,
        "graph_blind": True,
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    _write_once(
        target / "import_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return target


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _build_package(
    *,
    paper_context: ReconstructionPaperContext,
    review_source_sha256: str,
    reviewer_spans: list[ReviewSourceExcerpt],
    author_spans: list[ReviewSourceExcerpt],
    speaker_role_audit: dict[str, object],
    instructions: str,
) -> ReconstructionSessionPackage:
    prompt_sha256 = sha256_value(instructions.strip())
    schema_sha256 = sha256_value(ReconstructionSessionResponse.model_json_schema())
    core = _package_core(
        paper_context=paper_context,
        review_source_sha256=review_source_sha256,
        reviewer_spans=reviewer_spans,
        author_spans=author_spans,
        speaker_role_audit=speaker_role_audit,
        instructions=instructions,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
    )
    input_sha256 = sha256_value(core)
    package_id = _package_id(paper_context.paper_id, input_sha256)
    return ReconstructionSessionPackage(
        package_id=package_id,
        input_sha256=input_sha256,
        **core,
    )


def validate_session_package(package: ReconstructionSessionPackage) -> None:
    """Recompute every package identity hash and fail closed on artifact drift."""
    prompt_sha256 = sha256_value(package.instructions.strip())
    schema_sha256 = sha256_value(ReconstructionSessionResponse.model_json_schema())
    core = _package_core(
        paper_context=package.paper_context,
        review_source_sha256=package.review_source_sha256,
        reviewer_spans=package.reviewer_spans,
        author_spans=package.author_response_spans,
        speaker_role_audit=package.speaker_role_audit,
        instructions=package.instructions,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
    )
    input_sha256 = sha256_value(core)
    expected_id = _package_id(package.paper_id, input_sha256)
    if package.prompt_sha256 != prompt_sha256:
        raise ValueError("session package prompt hash drift")
    if package.schema_sha256 != schema_sha256:
        raise ValueError("session package schema hash drift")
    if package.input_sha256 != input_sha256:
        raise ValueError("session package input hash drift")
    if package.package_id != expected_id:
        raise ValueError("session package ID drift")


def _package_core(
    *,
    paper_context: ReconstructionPaperContext,
    review_source_sha256: str,
    reviewer_spans: list[ReviewSourceExcerpt],
    author_spans: list[ReviewSourceExcerpt],
    speaker_role_audit: dict[str, object],
    instructions: str,
    prompt_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    return {
        "session_kind": "reconstruction",
        "paper_id": paper_context.paper_id,
        "paper_sha256": paper_context.paper_sha256,
        "review_source_sha256": review_source_sha256,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "paper_context": paper_context,
        "reviewer_spans": reviewer_spans,
        "author_response_spans": author_spans,
        "speaker_role_audit": speaker_role_audit,
        "instructions": instructions.strip(),
    }


def _package_id(paper_id: str, input_sha256: str) -> str:
    identity = f"{paper_id}|reconstruction|{input_sha256}"
    return "RSP-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _source_excerpt(span: Any) -> ReviewSourceExcerpt:
    role = (
        ReviewSourceRole.REVIEWER_REPORT
        if span.source_kind == SourceKind.REVIEWER_REPORT
        else ReviewSourceRole.AUTHOR_RESPONSE
    )
    return ReviewSourceExcerpt(
        source_key=f"RR:{span.source_span_id}",
        source_role=role,
        reviewer_id_hash=span.reviewer_id_hash,
        round_id=span.round_id,
        char_start=span.char_start,
        char_end=span.char_end,
        text=span.text,
        text_sha256=span.text_sha256,
    )


def _case_paths(case: Mapping[str, Any]) -> tuple[Path, Path]:
    paper = Path(str(case.get("paper_path") or case.get("paper_markdown_path") or ""))
    review = Path(
        str(case.get("review_path") or case.get("peer_review_markdown_path") or "")
    )
    if not paper.is_file() or not review.is_file():
        raise FileNotFoundError(f"Nature pair is incomplete: {paper} / {review}")
    return paper.resolve(), review.resolve()


def _extract_doi(path: Path) -> str | None:
    match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        path.read_text(encoding="utf-8", errors="replace")[:40_000],
        re.IGNORECASE,
    )
    return match.group(0).rstrip(".,;)").casefold() if match else None


def _optional_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text[:10]) if text else None
    except ValueError:
        return None


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180]


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _response_template(package: ReconstructionSessionPackage) -> dict[str, Any]:
    return {
        "schema_version": "aspr_gear",
        "contract": "reconstruction_session_response",
        "package_id": package.package_id,
        "session_kind": package.session_kind,
        "paper_id": package.paper_id,
        "model_id": "FILL_MODEL_ID",
        "conversation_hash": "sha256:" + "0" * 64,
        "prompt_sha256": package.prompt_sha256,
        "schema_sha256": package.schema_sha256,
        "input_sha256": package.input_sha256,
        "review": {
            "schema_version": "aspr_gear",
            "paper_id": package.paper_id,
            "summary": {
                "schema_version": "aspr_gear",
                "text": "FILL",
                "evidence_keys": ["P:S-FILL"],
            },
            "novelty": {
                "schema_version": "aspr_gear",
                "judgment": "not_discussed",
                "supporting_points": [],
                "limiting_points": [],
            },
            "strengths": [],
            "weaknesses": [],
            "questions": [],
        },
        "reference_traces": [
            {
                "schema_version": "aspr_gear",
                "trace_id": "FILL_UNIQUE_TRACE_ID",
                "paper_id": package.paper_id,
                "point_id": "FILL_RETAINED_POINT_ID_OR_NULL",
                "reviewer_quote_keys": ["RR:FILL_REVIEWER_QUOTE_KEY"],
                "author_response_keys": [],
                "round_ids": ["FILL_ROUND_ID"],
                "reviewer_id_hashes": ["FILL_REVIEWER_ID_HASH"],
                "final_paper_evidence_keys": ["P:S-FILL"],
                "resolution_status": "persists",
                "rationale": "FILL",
            }
        ],
        "revision_ledger": [
            {
                "schema_version": "aspr_gear",
                "ledger_id": "FILL_UNIQUE_LEDGER_ID",
                "paper_id": package.paper_id,
                "reviewer_quote_keys": ["RR:FILL_REVIEWER_QUOTE_KEY"],
                "author_response_keys": [],
                "resolution_status": "resolved",
                "final_paper_evidence_keys": ["P:S-FILL"],
                "residual_summary": "FILL",
            }
        ],
        "output_sha256": "FILL_AFTER_HASHING_hash_payload",
    }


__all__ = [
    "RECONSTRUCTION_INSTRUCTIONS",
    "build_reconstruction_package",
    "import_session_response",
    "load_jsonl",
    "validate_session_package",
    "validate_session_response",
    "write_session_handoff",
]

"""Revision-aware, provenance-first human/AI review audit utilities.

This module deliberately keeps evaluation metadata outside ``StructuredReview`` so
that frozen human reconstructions and GEAR runtime outputs remain compatible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, model_validator
from scipy.optimize import linear_sum_assignment

from experiments.gear.review_reconstruction.contracts import (
    ReconstructionSessionPackage,
    ReconstructionSessionResponse,
    ResolutionStatus,
)
from gear.config import GearConfig
from gear.contracts import ReviewStatus, StrictModel
from gear.model_client import build_json_model_client
from gear.review_contracts import ReviewBundle, ReviewPoint, StructuredReview
from gear.trace import EvidenceStore, sha256_file, sha256_value

PILOT_IDS = (
    "10.1038/s41467-023-36025-x",
    "10.1038/s41467-023-43541-3",
    "10.1038/s41467-024-51536-x",
)
ABSOLUTE_NOVELTY_RE = re.compile(
    r"\b(first|only|unique|unprecedented|never before)\b", re.IGNORECASE
)
BLIND_JUDGE_PROMPT = (
    "You are a blinded scientific-review evaluator. Compare every listed L/R pair. "
    "SAME_POINT means the same atomic proposition and direction; PARTIAL_POINT means "
    "a material overlap; CONTRADICTORY means opposite direction; otherwise NO_MATCH. "
    "The pairs array is exhaustive: return every listed pair exactly once and no other "
    "pair. Do not infer source identity or use writing style."
)


class RevisionAuditCase(StrictModel):
    paper_id: str
    manuscript_path: Path
    metadata_path: Path
    reconstruction_dir: Path
    agent_run_dir: Path
    cutoff_date: str


class RevisionAuditManifest(StrictModel):
    contract: Literal["revision_audit_manifest"] = "revision_audit_manifest"
    task: Literal["revision_aware_audit"] = "revision_aware_audit"
    model: Literal["gpt-5.6-terra"] = "gpt-5.6-terra"
    reasoning_effort: Literal["medium"] = "medium"
    cases: list[RevisionAuditCase] = Field(min_length=1)

    @model_validator(mode="after")
    def pilot_only(self) -> RevisionAuditManifest:
        ids = tuple(case.paper_id for case in self.cases)
        if ids != PILOT_IDS:
            raise ValueError("revision-aware pilot must contain the fixed three papers")
        return self


class BlindPoint(StrictModel):
    blind_id: str
    section: Literal["novelty", "strengths", "weaknesses", "questions", "resolved"]
    aspect: str
    severity: str
    text: str
    suggested_action: str | None = None


class BlindPairDecision(StrictModel):
    left_id: str
    right_id: str
    label: Literal["SAME_POINT", "PARTIAL_POINT", "CONTRADICTORY", "NO_MATCH"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class BlindJudgeResponse(StrictModel):
    task_id: str
    model_id: str
    decisions: list[BlindPairDecision]


class AuditRubric(StrictModel):
    title: str
    description: str
    polarity: Literal["positive", "risk"]


class RubricSet(StrictModel):
    paper_id: str
    model_id: str
    rubrics: list[AuditRubric] = Field(min_length=8, max_length=8)


class RubricScore(StrictModel):
    title: str
    score: int = Field(ge=-2, le=2)
    rationale: str
    paper_excerpt: str = ""


class RubricScoreResponse(StrictModel):
    paper_id: str
    model_id: str
    scores: list[RubricScore] = Field(min_length=8, max_length=8)


@dataclass(frozen=True)
class AuditPoint:
    point_id: str
    section: str
    aspect: str
    severity: str
    text: str
    suggested_action: str | None
    status: str | None = None
    reviewer_support_count: int = 0
    semantic_verified: bool | None = None
    validation_status: str | None = None
    evidence_keys: tuple[str, ...] = ()


def load_manifest(path: Path) -> RevisionAuditManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    base = Path(path).resolve().parent
    for item in raw.get("cases", []):
        for field in (
            "manuscript_path",
            "metadata_path",
            "reconstruction_dir",
            "agent_run_dir",
        ):
            value = Path(item[field])
            item[field] = str(
                value if value.is_absolute() else (base / value).resolve()
            )
    return RevisionAuditManifest.model_validate(raw)


def preflight(manifest: RevisionAuditManifest) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in manifest.cases:
        required = {
            "manuscript": case.manuscript_path,
            "metadata": case.metadata_path,
            "human_package": case.reconstruction_dir / "package.json",
            "human_response": case.reconstruction_dir / "response.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        metadata = (
            json.loads(case.metadata_path.read_text(encoding="utf-8"))
            if not missing
            else {}
        )
        rows.append(
            {
                "paper_id": case.paper_id,
                "cutoff_date": case.cutoff_date,
                "metadata_submission_date": metadata.get("submission_date"),
                "missing": missing,
                "input_sha256": {
                    name: sha256_file(path)
                    for name, path in required.items()
                    if path.is_file()
                },
            }
        )
    return {
        "contract": "revision_audit_input_audit",
        "cases": rows,
        "passed": all(
            not row["missing"] and row["cutoff_date"] == row["metadata_submission_date"]
            for row in rows
        ),
    }


def agent_availability(case: RevisionAuditCase) -> dict[str, Any]:
    run = case.agent_run_dir
    required = (
        "review.json",
        "review_bundle.json",
        "review_state.json",
        "evidence_trace.jsonl",
        "run_manifest.json",
    )
    missing = [name for name in required if not (run / name).is_file()]
    result: dict[str, Any] = {
        "paper_id": case.paper_id,
        "available": False,
        "missing": missing,
    }
    if missing:
        return result
    try:
        raw_bundle = json.loads((run / "review_bundle.json").read_text(encoding="utf-8"))
        if raw_bundle.get("schema_revision") != "evidence_state_delta_v2":
            result["schema_revision"] = raw_bundle.get("schema_revision")
            result["rejection_reason"] = "unsupported_schema_revision"
            return result
        review = StructuredReview.model_validate_json(
            (run / "review.json").read_text(encoding="utf-8")
        )
        bundle = ReviewBundle.model_validate_json(
            (run / "review_bundle.json").read_text(encoding="utf-8")
        )
        store = EvidenceStore(run)
        key_set = _store_keys(run)
        review_keys = _review_keys(review)
        manifest_failures = store.validate_manifest()
        identical = bundle.structured_review == review
        result.update(
            {
                "status": bundle.status.value,
                "verification_passed": bundle.verification.passed,
                "semantic_verification_available": bundle.verification.semantic_verification_available,
                "failure_count": (
                    len(bundle.state_v2.failures) if bundle.state_v2 else -1
                ),
                "review_matches_bundle": identical,
                "unresolved_evidence_keys": sorted(review_keys - key_set),
                "manifest_failures": manifest_failures,
                "available": (
                    bundle.status == ReviewStatus.COMPLETE
                    and bundle.verification.passed
                    and bundle.verification.semantic_verification_available
                    and (not bundle.state_v2 or not bundle.state_v2.failures)
                    and identical
                    and not (review_keys - key_set)
                    and not manifest_failures
                ),
            }
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def human_points(
    case: RevisionAuditCase,
) -> tuple[list[AuditPoint], list[AuditPoint], int]:
    package = _load_human_package(case.reconstruction_dir / "package.json")
    response = _load_human_response(case.reconstruction_dir / "response.json")
    source_keys = {
        span.source_key
        for span in [*package.reviewer_spans, *package.author_response_spans]
    }
    traces = {
        trace.point_id: trace for trace in response.reference_traces if trace.point_id
    }
    points: list[AuditPoint] = []
    for section, point in _review_sections(response.review):
        trace = traces[point.point_id]
        if not set(trace.reviewer_quote_keys).issubset(source_keys):
            raise ValueError(f"human trace has unknown reviewer span: {point.point_id}")
        points.append(
            _point_from_review(
                point,
                section,
                status=trace.resolution_status.value,
                reviewer_support_count=len(trace.reviewer_id_hashes),
            )
        )
    reviewer_map = {row.source_key: row for row in package.reviewer_spans}
    resolved: list[AuditPoint] = []
    for entry in response.revision_ledger:
        if entry.resolution_status != ResolutionStatus.RESOLVED:
            continue
        quotes = " ".join(reviewer_map[key].text for key in entry.reviewer_quote_keys)
        resolved.append(
            AuditPoint(
                entry.ledger_id,
                "resolved",
                "revision_resolved",
                "minor",
                quotes[:2500],
                None,
                "resolved",
            )
        )
    unverifiable = sum(
        trace.resolution_status == ResolutionStatus.UNVERIFIABLE
        for trace in response.reference_traces
    )
    return points, resolved, unverifiable


def _load_human_package(path: Path) -> ReconstructionSessionPackage:
    """Accept historic reconstruction packages with audit-only extra fields."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    permitted = set(ReconstructionSessionPackage.model_fields)
    return ReconstructionSessionPackage.model_validate(
        {name: value for name, value in raw.items() if name in permitted}
    )


def _load_human_response(path: Path) -> ReconstructionSessionResponse:
    raw = json.loads(path.read_text(encoding="utf-8"))
    permitted = set(ReconstructionSessionResponse.model_fields)
    return ReconstructionSessionResponse.model_validate(
        {name: value for name, value in raw.items() if name in permitted}
    )


def agent_points(case: RevisionAuditCase) -> list[AuditPoint]:
    review = StructuredReview.model_validate_json(
        (case.agent_run_dir / "review.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (case.agent_run_dir / "review_state.json").read_text(encoding="utf-8")
    )
    canonical = state.get("canonical_points", {})
    return [
        _point_from_review(
            point,
            section,
            semantic_verified=canonical.get(point.point_id, {}).get(
                "semantic_verified"
            ),
            validation_status=canonical.get(point.point_id, {}).get(
                "validation_status"
            ),
        )
        for section, point in _review_sections(review)
    ]


def build_blind_package(
    case: RevisionAuditCase,
    left: Sequence[AuditPoint],
    right: Sequence[AuditPoint],
    *,
    kind: str,
    chunk: int,
) -> dict[str, Any]:
    if not right:
        raise ValueError("blind package requires at least one right-side point")
    left_per_chunk = max(1, 48 // len(right))
    start = chunk * left_per_chunk
    selected_left = list(left[start : start + left_per_chunk])
    if not selected_left:
        raise ValueError("blind package chunk has no pairs")
    selected = [
        (row.point_id, column.point_id) for row in selected_left for column in right
    ]
    left_ids = {point.point_id for point in selected_left}
    right_ids = {point.point_id for point in right}
    identity = sha256_value(
        {
            "paper": case.paper_id,
            "kind": kind,
            "pairs": selected,
            "left": [_audit_payload(x) for x in left],
            "right": [_audit_payload(x) for x in right],
        }
    )
    left_mapping = {
        _opaque_blind_id("L", identity, point_id): point_id for point_id in left_ids
    }
    right_mapping = {
        _opaque_blind_id("R", identity, point_id): point_id for point_id in right_ids
    }
    return {
        "task_id": f"RA-{identity.split(':')[-1][:20]}",
        "paper_id_hash": "sha256:" + hashlib.sha256(case.paper_id.encode()).hexdigest(),
        "kind": kind,
        "chunk": chunk,
        "left": [
            _blind_point(point, _lookup_blind_id(left_mapping, point.point_id))
            for point in left
            if point.point_id in left_ids
        ],
        "right": [
            _blind_point(point, _lookup_blind_id(right_mapping, point.point_id))
            for point in right
            if point.point_id in right_ids
        ],
        "pairs": [
            (
                _lookup_blind_id(left_mapping, left_id),
                _lookup_blind_id(right_mapping, right_id),
            )
            for left_id, right_id in selected
        ],
        "mapping": left_mapping | right_mapping,
        "input_sha256": identity,
    }


def blind_package_count(left_count: int, right_count: int) -> int:
    """Return complete rectangular blinded-comparison package count."""
    if left_count <= 0 or right_count <= 0:
        return 0
    left_per_chunk = max(1, 48 // right_count)
    return (left_count + left_per_chunk - 1) // left_per_chunk


def judge_package(package: dict[str, Any], config: GearConfig) -> dict[str, Any]:
    client = build_json_model_client(config)
    schema = BlindJudgeResponse.model_json_schema()
    public = {
        key: package[key]
        for key in ("task_id", "paper_id_hash", "kind", "left", "right", "pairs")
    }
    response: BlindJudgeResponse | None = None
    last_error: ValidationError | ValueError | TypeError | None = None
    for attempt in range(2):
        system = BLIND_JUDGE_PROMPT
        if attempt:
            system += (
                " The prior response was invalid. Return the requested decisions "
                "object, not a JSON Schema or explanation."
            )
        try:
            payload = client.generate_json(
                system=system,
                user=json.dumps(public, ensure_ascii=False),
                response_schema=schema,
            )
            payload["task_id"] = package["task_id"]
            payload["model_id"] = client.model_name
            response = BlindJudgeResponse.model_validate(payload)
            break
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = exc
    if response is None:
        raise ValueError(
            f"blind judge returned no valid decision response: {last_error}"
        )
    observed = {(row.left_id, row.right_id) for row in response.decisions}
    expected = {tuple(pair) for pair in package["pairs"]}
    if observed != expected or len(observed) != len(response.decisions):
        raise ValueError("judge response must cover every blinded pair exactly once")
    return {
        "package": package,
        "response": response.model_dump(mode="json"),
        "prompt_sha256": sha256_value(BLIND_JUDGE_PROMPT),
        "model_id": client.model_name,
    }


def score_pairs(
    left: Sequence[AuditPoint],
    right: Sequence[AuditPoint],
    decisions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    left_index = {item.point_id: index for index, item in enumerate(left)}
    right_index = {item.point_id: index for index, item in enumerate(right)}
    strict = _assignment(left, right, decisions, {"SAME_POINT": 1.0})
    soft = _assignment(
        left, right, decisions, {"SAME_POINT": 1.0, "PARTIAL_POINT": 0.5}
    )
    strict_weight = sum(value for _, _, value in strict)
    soft_weight = sum(value for _, _, value in soft)
    section_correct = sum(
        left[left_index[a]].section == right[right_index[b]].section
        for a, b, _ in strict
    )
    labels = Counter(row["label"] for row in decisions)
    return {
        "strict": _prf(strict_weight, len(right), len(left)),
        "soft": _prf(soft_weight, len(right), len(left)),
        "strict_match_count": len(strict),
        "soft_match_weight": soft_weight,
        "section_correct_count": section_correct,
        "section_correct_rate": _ratio(section_correct, len(strict)),
        "contradictory_pair_count": labels["CONTRADICTORY"],
        "partial_pair_count": labels["PARTIAL_POINT"],
        "strict_matches": [{"left_id": a, "right_id": b} for a, b, _ in strict],
        "soft_matches": [
            {"left_id": a, "right_id": b, "weight": weight} for a, b, weight in soft
        ],
    }


def rubric_set(case: RevisionAuditCase, config: GearConfig) -> dict[str, Any]:
    client = build_json_model_client(config)
    response = _load_human_response(case.reconstruction_dir / "response.json")
    prompt = """Create exactly eight paper-specific review rubrics from a final manuscript and a human reference review. The rubrics must be titled: Core Contribution Accuracy, Results Interpretation, Comparative Analysis, Evidence-Based Critique, Critique Clarity, Completeness Coverage, Constructive Tone, False or Contradictory Claims. First seven have positive polarity; the final has risk polarity. Do not mention any AI review."""
    user = json.dumps(
        {
            "paper_id": case.paper_id,
            "manuscript": _read_limited(case.manuscript_path),
            "human_reference": _review_public(response.review),
        },
        ensure_ascii=False,
    )
    payload = client.generate_json(
        system=prompt, user=user, response_schema=RubricSet.model_json_schema()
    )
    payload.update({"paper_id": case.paper_id, "model_id": client.model_name})
    result = RubricSet.model_validate(payload)
    return {
        "rubrics": result.model_dump(mode="json"),
        "prompt_sha256": sha256_value(prompt),
        "input_sha256": sha256_value(user),
    }


def score_rubrics(
    case: RevisionAuditCase, rubric: RubricSet, config: GearConfig
) -> dict[str, Any]:
    client = build_json_model_client(config)
    review = StructuredReview.model_validate_json(
        (case.agent_run_dir / "review.json").read_text(encoding="utf-8")
    )
    prompt = """Score this blinded scientific review against each supplied rubric. Positive rubrics use 0, 1, or 2. Risk rubrics use 0, -1, or -2. Give a concise rationale and a supporting manuscript excerpt when possible. Do not identify the review system and do not compute a composite score."""
    user = json.dumps(
        {
            "paper_id": case.paper_id,
            "manuscript": _read_limited(case.manuscript_path),
            "rubrics": rubric.model_dump(mode="json")["rubrics"],
            "review": _review_public(review),
        },
        ensure_ascii=False,
    )
    payload = client.generate_json(
        system=prompt,
        user=user,
        response_schema=RubricScoreResponse.model_json_schema(),
    )
    payload.update({"paper_id": case.paper_id, "model_id": client.model_name})
    result = RubricScoreResponse.model_validate(payload)
    expected = {item.title for item in rubric.rubrics}
    if {item.title for item in result.scores} != expected:
        raise ValueError("rubric scorer must return exactly the supplied rubrics")
    return {
        "scores": result.model_dump(mode="json"),
        "prompt_sha256": sha256_value(prompt),
        "input_sha256": sha256_value(user),
    }


def _assignment(
    left: Sequence[AuditPoint],
    right: Sequence[AuditPoint],
    decisions: Sequence[dict[str, Any]],
    weights: dict[str, float],
) -> list[tuple[str, str, float]]:
    if not left or not right:
        return []
    matrix = [[0.0 for _ in right] for _ in left]
    base_matrix = [[0.0 for _ in right] for _ in left]
    for row in decisions:
        left_id = str(row["left_id"]).removeprefix("L-")
        right_id = str(row["right_id"]).removeprefix("R-")
        for i, point in enumerate(left):
            if point.point_id != left_id:
                continue
            for j, candidate in enumerate(right):
                if candidate.point_id == right_id:
                    base_weight = weights.get(str(row["label"]), 0.0)
                    base_matrix[i][j] = base_weight
                    matrix[i][j] = base_weight + (
                        float(row.get("confidence", 1.0)) * 1e-6
                    )
    rows, cols = linear_sum_assignment([[-value for value in row] for row in matrix])
    return [
        (left[i].point_id, right[j].point_id, base_matrix[i][j])
        for i, j in zip(rows, cols)
        if base_matrix[i][j] > 0
    ]


def _review_sections(review: StructuredReview) -> list[tuple[str, ReviewPoint]]:
    return [
        ("novelty", point)
        for point in [
            *review.novelty.supporting_points,
            *review.novelty.limiting_points,
        ]
    ] + [
        (section, point)
        for section, points in (
            ("strengths", review.strengths),
            ("weaknesses", review.weaknesses),
            ("questions", review.questions),
        )
        for point in points
    ]


def _point_from_review(point: ReviewPoint, section: str, **kwargs: Any) -> AuditPoint:
    return AuditPoint(
        point.point_id,
        section,
        point.aspect.value,
        point.severity.value,
        point.text,
        point.suggested_action,
        evidence_keys=tuple(point.evidence_keys),
        **kwargs,
    )


def _blind_point(point: AuditPoint, blind_id: str) -> dict[str, Any]:
    return BlindPoint(
        blind_id=blind_id,
        section=cast(
            Literal["novelty", "strengths", "weaknesses", "questions", "resolved"],
            point.section,
        ),
        aspect=point.aspect,
        severity=point.severity,
        text=point.text,
        suggested_action=point.suggested_action,
    ).model_dump(mode="json")


def _opaque_blind_id(prefix: str, identity: str, point_id: str) -> str:
    digest = hashlib.sha256(f"{identity}:{prefix}:{point_id}".encode()).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _lookup_blind_id(mapping: dict[str, str], point_id: str) -> str:
    return next(key for key, value in mapping.items() if value == point_id)


def _audit_payload(point: AuditPoint) -> dict[str, Any]:
    return {
        "section": point.section,
        "aspect": point.aspect,
        "severity": point.severity,
        "text": point.text,
        "suggested_action": point.suggested_action,
    }


def _review_public(review: StructuredReview) -> dict[str, Any]:
    return {
        "summary": review.summary.text,
        "novelty": review.novelty.judgment.value,
        "points": [
            _audit_payload(_point_from_review(point, section))
            for section, point in _review_sections(review)
        ],
    }


def _review_keys(review: StructuredReview) -> set[str]:
    return set(review.summary.evidence_keys).union(
        *(set(point.evidence_keys) for point in review.all_points())
    )


def _store_keys(run_dir: Path) -> set[str]:
    return {
        json.loads(line)["evidence_id"]
        for line in (run_dir / "evidence_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }


def _read_limited(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")[:30_000]


def _ratio(value: float, total: float) -> float | None:
    return value / total if total else None


def _prf(matched: float, candidate: int, reference: int) -> dict[str, float | None]:
    if not candidate and not reference:
        return {"precision": None, "recall": None, "f1": None, "both_empty": True}
    precision = matched / candidate if candidate else 0.0
    recall = matched / reference if reference else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else (0.0 if recall is not None else None)
    )
    return {"precision": precision, "recall": recall, "f1": f1, "both_empty": False}


__all__ = [
    "BLIND_JUDGE_PROMPT",
    "PILOT_IDS",
    "RevisionAuditCase",
    "RevisionAuditManifest",
    "agent_availability",
    "agent_points",
    "build_blind_package",
    "human_points",
    "judge_package",
    "load_manifest",
    "preflight",
    "rubric_set",
    "score_pairs",
    "score_rubrics",
]

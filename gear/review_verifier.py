"""Fail-closed verifier for the five-part StructuredReview contract."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, List, Mapping, Optional

from .codex_critic import FORBIDDEN_DECISION_TEXT
from .config import GearConfig, load_config
from .model_client import build_json_model_client
from .contracts import PaperIR
from .review_contracts import (
    PointSeverity,
    ReviewState,
    StructuredReview,
    VerificationIssue,
    VerificationReport,
)
from .trace import EvidenceStore

GRAPH_SEMANTIC_TERMS = re.compile(
    r"\b(?:ASPR(?: score)?|p[_ -]?uptake|conditional[_ -]?diffusion|"
    r"D5 percentile|OOF(?: spearman)?|EF\d{4}|opportunity field|"
    r"context[- ]control field)\b",
    re.I,
)


class ReviewVerifier:
    def __init__(
        self,
        config: Optional[GearConfig] = None,
        *,
        semantic_checker: Optional[Callable[[str, str], Mapping[str, Any]]] = None,
    ) -> None:
        self.config = config or load_config()
        client = build_json_model_client(self.config)
        self.semantic_checker = semantic_checker or (
            lambda system, user: client.generate_json(system=system, user=user)
        )
        self.semantic_calls = 0

    def verify(
        self,
        review: StructuredReview,
        state: ReviewState,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> VerificationReport:
        issues = self._deterministic_issues(review, state, paper_ir, evidence_store)
        unsupported_major = sum(issue.code == "unsupported_major" for issue in issues)
        graph_violations = sum(
            issue.code == "graph_semantic_violation" for issue in issues
        )
        semantic_available = self.semantic_checker is not None
        limited = False
        if semantic_available:
            semantic_issues = self._semantic_issues(review, paper_ir)
            issues.extend(semantic_issues)
            unsupported_major += sum(
                issue.code == "unsupported_major" for issue in semantic_issues
            )
        else:
            limited = True
            issues.append(
                self._issue(
                    "semantic_verifier_unavailable",
                    "Semantic paper-span support verification was not available.",
                )
            )
        hard_issues = [
            issue for issue in issues if issue.code != "semantic_verifier_unavailable"
        ]
        return VerificationReport(
            passed=not hard_issues,
            limited=limited,
            issues=issues,
            semantic_verification_available=semantic_available,
            graph_semantic_violation_count=graph_violations,
            unsupported_major_count=unsupported_major,
        )

    def _deterministic_issues(
        self,
        review: StructuredReview,
        state: ReviewState,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> List[VerificationIssue]:
        issues: List[VerificationIssue] = []
        if review.paper_id != paper_ir.paper_id or review.paper_id != state.paper_id:
            issues.append(
                self._issue(
                    "paper_identity_mismatch",
                    "Review, state, and PaperIR paper IDs must match.",
                )
            )
        if state.paper_sha256 != paper_ir.paper_sha256:
            issues.append(
                self._issue(
                    "paper_hash_mismatch",
                    "ReviewState paper hash differs from PaperIR.",
                )
            )
        for key in review.summary.evidence_keys:
            if not self._valid_evidence_key(key, paper_ir, evidence_store):
                issues.append(
                    self._issue(
                        "invalid_summary_evidence",
                        f"Summary evidence key is invalid: {key}",
                    )
                )
        for point in review.all_points():
            invalid = [
                key
                for key in point.evidence_keys
                if not self._valid_evidence_key(key, paper_ir, evidence_store)
            ]
            if invalid:
                issues.append(
                    self._issue(
                        "invalid_evidence_key",
                        f"Point contains invalid evidence keys: {invalid}",
                        point.point_id,
                    )
                )
            if point.severity == PointSeverity.MAJOR and (
                not point.evidence_keys or invalid
            ):
                issues.append(
                    self._issue(
                        "unsupported_major",
                        "Major point lacks fully valid evidence.",
                        point.point_id,
                    )
                )
            if GRAPH_SEMANTIC_TERMS.search(point.text):
                issues.append(
                    self._issue(
                        "graph_semantic_violation",
                        "Visible review text uses graph fields as a scientific judgment.",
                        point.point_id,
                    )
                )
            if FORBIDDEN_DECISION_TEXT.search(f"{point.text} {point.suggested_action}"):
                issues.append(
                    self._issue(
                        "decision_language_forbidden",
                        "GEAR review points cannot contain decision language.",
                        point.point_id,
                    )
                )
        if FORBIDDEN_DECISION_TEXT.search(review.summary.text):
            issues.append(
                self._issue(
                    "decision_language_forbidden",
                    "GEAR summary cannot contain decision language.",
                )
            )
        if GRAPH_SEMANTIC_TERMS.search(review.summary.text):
            issues.append(
                self._issue(
                    "graph_semantic_violation",
                    "Visible review summary uses graph fields as a scientific judgment.",
                )
            )
        return issues

    def _semantic_issues(
        self,
        review: StructuredReview,
        paper_ir: PaperIR,
    ) -> List[VerificationIssue]:
        if self.semantic_checker is None:
            return []
        self.semantic_calls += 1
        evidence = {
            f"P:{span.span_id}": span.text
            for span in paper_ir.spans
            if f"P:{span.span_id}" in _review_evidence_keys(review)
        }
        user = json.dumps(
            {
                "review": review.model_dump(mode="json"),
                "paper_evidence": evidence,
                "output": {
                    "unsupported_point_ids": [],
                    "summary_supported": True,
                },
            },
            ensure_ascii=False,
        )
        try:
            payload = self.semantic_checker(
                "Judge only whether each review proposition is supported by its cited paper spans.",
                user,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return [
                self._issue(
                    "semantic_verifier_failed",
                    f"Semantic verifier failed closed: {exc}",
                )
            ]
        raw_ids = payload.get("unsupported_point_ids") or []
        if not isinstance(raw_ids, list):
            return [
                self._issue(
                    "semantic_verifier_invalid",
                    "Semantic verifier returned an invalid point list.",
                )
            ]
        point_map = {point.point_id: point for point in review.all_points()}
        issues: List[VerificationIssue] = []
        for point_id in sorted({str(item) for item in raw_ids} & set(point_map)):
            code = (
                "unsupported_major"
                if point_map[point_id].severity == PointSeverity.MAJOR
                else "paper_span_semantic_support_failed"
            )
            issues.append(
                self._issue(
                    code,
                    "Point is not semantically supported by its cited paper spans.",
                    point_id,
                )
            )
        if payload.get("summary_supported") is not True:
            issues.append(
                self._issue(
                    "summary_semantic_support_failed",
                    "Summary is not supported by its cited paper spans.",
                )
            )
        return issues

    @staticmethod
    def _valid_evidence_key(
        key: str,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> bool:
        if not evidence_store.has(key):
            return False
        if key.startswith("P:"):
            return key in {f"P:{span.span_id}" for span in paper_ir.spans}
        if key.startswith("R:"):
            record = evidence_store.get(key)
            return bool(record and record.payload.get("temporal_valid") is True)
        return False

    @staticmethod
    def _issue(
        code: str,
        message: str,
        point_id: Optional[str] = None,
    ) -> VerificationIssue:
        identity = f"{code}|{point_id}|{message}"
        return VerificationIssue(
            issue_id="VI2-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
            code=code,
            message=message,
            point_id=point_id,
        )


def _review_evidence_keys(review: StructuredReview) -> set[str]:
    keys = set(review.summary.evidence_keys)
    for point in review.all_points():
        keys.update(point.evidence_keys)
    return keys


__all__ = ["GRAPH_SEMANTIC_TERMS", "ReviewVerifier"]

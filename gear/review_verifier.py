"""Fail-closed verifier for the five-part StructuredReview contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from .codex_critic import FORBIDDEN_DECISION_TEXT
from .config import GearConfig, load_config
from .contracts import PaperIR
from .model_client import build_json_model_client
from .review_contracts import (
    PointSeverity,
    PointValidationStatus,
    ReviewPhase,
    ReviewSource,
    ReviewState,
    ReviewStateV3,
    StructuredReview,
    VerificationIssue,
    VerificationReport,
)
from .trace import EvidenceStore, sha256_value

GRAPH_SEMANTIC_TERMS = re.compile(
    r"\b(?:ASPR score|p[_ -]?uptake|conditional[_ -]?diffusion|"
    r"D5 percentile|OOF(?: spearman)?|EF\d{4}|opportunity field|"
    r"context[- ]control field)\b",
    re.IGNORECASE,
)

ABSOLUTE_PRIORITY_TERMS = re.compile(
    r"\b(?:first|first-ever|unprecedented|unique|world-first)\b|首次|首个|前所未有|唯一",
    re.IGNORECASE,
)


class ReviewVerifier:
    def __init__(
        self,
        config: GearConfig | None = None,
        *,
        semantic_checker: Callable[[str, str], Mapping[str, Any]] | None = None,
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

    def verify_state(
        self,
        review: StructuredReview,
        state: ReviewStateV3,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> VerificationReport:
        issues = self._deterministic_v2_issues(review, state, paper_ir, evidence_store)
        semantic_available = self.semantic_checker is not None
        semantic_success = False
        if semantic_available:
            semantic_issues = self._semantic_issues(review, paper_ir)
            issues.extend(semantic_issues)
            semantic_success = not any(
                issue.code in {"semantic_verifier_failed", "semantic_verifier_invalid"}
                for issue in semantic_issues
            )
        else:
            issues.append(
                self._issue(
                    "semantic_verifier_unavailable",
                    "Semantic paper-span support verification was not available.",
                )
            )
        non_blocking_semantic_codes = {
            "semantic_verifier_unavailable",
            "semantic_verifier_failed",
            "semantic_verifier_invalid",
        }
        hard = [
            issue for issue in issues if issue.code not in non_blocking_semantic_codes
        ]
        report = VerificationReport(
            passed=not hard,
            limited=not semantic_success,
            issues=issues,
            semantic_verification_available=semantic_success,
            graph_semantic_violation_count=sum(
                issue.code == "graph_semantic_violation" for issue in issues
            ),
            unsupported_major_count=sum(
                issue.code == "unsupported_major" for issue in issues
            ),
        )
        if report.passed:
            state.phase = ReviewPhase.VERIFIED
            state.process_features.semantic_verifier_passed = semantic_success
        return report

    def _deterministic_v2_issues(
        self,
        review: StructuredReview,
        state: ReviewStateV3,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        if review.paper_id != state.paper_id or review.paper_id != paper_ir.paper_id:
            issues.append(
                self._issue("paper_identity_mismatch", "V2 paper IDs differ.")
            )
        if state.paper_sha256 != paper_ir.paper_sha256:
            issues.append(self._issue("paper_hash_mismatch", "V2 paper hashes differ."))
        issues.extend(self._branch_independence_issues(state, evidence_store))
        issues.extend(self._relation_issues(state, evidence_store, paper_ir))
        issues.extend(self._correction_issues(state, evidence_store))
        visible = [review.summary.text]
        for point in review.all_points():
            visible.extend([point.text, point.suggested_action])
            canonical = state.canonical_points.get(point.point_id)
            if canonical is None:
                issues.append(
                    self._issue(
                        "compiler_created_unknown_point",
                        "Final review point is absent from canonical state.",
                        point.point_id,
                    )
                )
                continue
            if (
                not canonical.retained
                or canonical.validation_status != PointValidationStatus.VALIDATED
            ):
                issues.append(
                    self._issue(
                        "unverified_point_compiled",
                        "Final review contains a point that was not retained and "
                        "validated.",
                        point.point_id,
                    )
                )
            qwen_only = (
                not canonical.agent_support
                and ReviewSource.ASPR_QWEN in canonical.source_point_ids
            )
            if qwen_only and point.severity == PointSeverity.MAJOR:
                issues.append(
                    self._issue(
                        "qwen_only_major_unverified",
                        "Qwen-only major point cannot bypass semantic verification.",
                        point.point_id,
                    )
                )
            if canonical.qwen_conflict and not canonical.validation_notes:
                issues.append(
                    self._issue(
                        "fusion_conflict_unresolved",
                        "Contradictory branch point has no resolution note.",
                        point.point_id,
                    )
                )
            if canonical.graph_tension and canonical.stability_status == "pending":
                issues.append(
                    self._issue(
                        "graph_tension_unprocessed",
                        "Graph-text tension lacks a completed stability action.",
                        point.point_id,
                    )
                )
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
            coverage_only_external = bool(
                canonical.coverage_evidence_keys
            ) and not bool(canonical.relation_evidence_keys)
            if coverage_only_external and ABSOLUTE_PRIORITY_TERMS.search(point.text):
                issues.append(
                    self._issue(
                        "coverage_overclaim",
                        "Search coverage cannot support an absolute priority claim.",
                        point.point_id,
                    )
                )
            if point.severity == PointSeverity.MAJOR and invalid:
                issues.append(
                    self._issue(
                        "unsupported_major",
                        "Major point lacks fully valid evidence.",
                        point.point_id,
                    )
                )
        for text in visible:
            if GRAPH_SEMANTIC_TERMS.search(text):
                issues.append(
                    self._issue(
                        "graph_semantic_violation",
                        "Visible review leaks a prohibited Graph-specific field.",
                    )
                )
                break
            if FORBIDDEN_DECISION_TEXT.search(text):
                issues.append(
                    self._issue(
                        "decision_language_forbidden",
                        "Final review contains decision language.",
                    )
                )
                break
        deleted_text = {
            point.proposition.casefold()
            for point in state.canonical_points.values()
            if not point.retained
        }
        if any(
            text and text in review.summary.text.casefold() for text in deleted_text
        ):
            issues.append(
                self._issue(
                    "deleted_point_in_summary",
                    "Summary retained the proposition of a deleted point.",
                )
            )
        return issues

    def _correction_issues(
        self, state: ReviewStateV3, evidence_store: EvidenceStore
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        corrected_points: set[str] = set()
        direction_events: list[dict[str, Any]] = []
        for key in state.correction_event_evidence_keys:
            record = evidence_store.get(key)
            if record is None or record.kind != "review_correction_event":
                issues.append(
                    self._issue(
                        "correction_event_missing", f"Missing correction event: {key}"
                    )
                )
                continue
            payload = record.payload
            if payload.get("before_direction") != payload.get("after_direction"):
                direction_events.append(payload)
            corrected_points.add(str(payload.get("point_id") or ""))
            relation_ids = payload.get("trigger_relation_ids") or []
            if not relation_ids:
                issues.append(
                    self._issue(
                        "graph_only_correction",
                        "Substantive correction lacks a verified relation.",
                        payload.get("point_id"),
                    )
                )
                continue
            for relation_id in relation_ids:
                relation = evidence_store.get(f"R:{relation_id}")
                relation_payload = relation.payload if relation is not None else {}
                if relation_payload.get(
                    "temporal_valid"
                ) is not True or relation_payload.get("relation_label") in {
                    "DISTANT",
                    "UNRESOLVED",
                    None,
                }:
                    issues.append(
                        self._issue(
                            "correction_relation_invalid",
                            f"Correction relation is not verified: {relation_id}",
                            payload.get("point_id"),
                        )
                    )
        for point in state.canonical_points.values():
            changed = (
                point.initial_section is not None
                and point.section != point.initial_section
            ) or point.novelty_resolution in {
                "antecedent_found",
                "incremental_or_parallel",
            }
            if changed and point.point_id not in corrected_points:
                issues.append(
                    self._issue(
                        "correction_event_missing",
                        "Substantive point change lacks an auditable correction event.",
                        point.point_id,
                    )
                )
        agent = state.branch_reviews.get(ReviewSource.AGENT)
        if (
            agent is not None
            and state.novelty_direction is not None
            and state.novelty_direction != agent.novelty.judgment
            and not any(
                event.get("after_direction") == state.novelty_direction.value
                for event in direction_events
            )
        ):
            issues.append(
                self._issue(
                    "direction_correction_missing",
                    "Final novelty direction differs from the Graph-blind Reviewer "
                    "without a relation-mediated correction event.",
                )
            )
        return issues

    def _branch_independence_issues(
        self,
        state: ReviewStateV3,
        evidence_store: EvidenceStore,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        forbidden = (
            "graph_context",
            "graph_prior",
            "graph_result",
            "aspr_score_0_100",
            "score_0_100",
            "p_uptake",
            "conditional_diffusion",
            "d5_percentile",
        )
        for source, key in (
            (ReviewSource.AGENT, "A:BRANCH"),
            (ReviewSource.ASPR_QWEN, "Q:BRANCH"),
        ):
            branch = state.branch_reviews.get(source)
            if branch is None:
                continue
            if not branch.graph_blind:
                issues.append(
                    self._issue(
                        "branch_not_graph_blind", f"{source.value} is not graph blind."
                    )
                )
            record = evidence_store.get(key)
            if record is None:
                issues.append(
                    self._issue(
                        "branch_input_unverifiable",
                        f"{source.value} input payload was not retained for hash "
                        "verification.",
                    )
                )
                continue
            payload = record.payload.get("input_payload")
            if sha256_value(payload) != branch.input_sha256:
                issues.append(
                    self._issue(
                        "branch_input_hash_mismatch",
                        f"{source.value} input hash does not match retained payload.",
                    )
                )
            serialized = json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            ).casefold()
            if any(field in serialized for field in forbidden):
                issues.append(
                    self._issue(
                        "branch_graph_leakage",
                        f"{source.value} input contains prohibited Graph fields.",
                    )
                )
        return issues

    def _relation_issues(
        self,
        state: ReviewStateV3,
        evidence_store: EvidenceStore,
        paper_ir: PaperIR,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        relevant_keys = {
            key: point.point_id
            for point in state.canonical_points.values()
            if point.retained
            for key in point.relation_evidence_keys
        }
        for key, point_id in relevant_keys.items():
            record = evidence_store.get(key)
            payload = record.payload if record is not None else {}
            missing = [
                field
                for field in (
                    "target_span_id",
                    "prior_span_id",
                    "difference_dimensions",
                )
                if not payload.get(field)
            ]
            if record is None or missing:
                issues.append(
                    self._issue(
                        "relation_missing_paired_spans",
                        f"Relation {key} lacks paired spans or difference "
                        f"dimensions: {missing}",
                        point_id,
                    )
                )
            elif payload.get("target_span_id") not in paper_ir.span_map():
                issues.append(
                    self._issue(
                        "relation_target_span_mismatch",
                        f"Relation {key} targets a span outside the current PaperIR.",
                        point_id,
                    )
                )
            elif payload.get("temporal_valid") is not True:
                issues.append(
                    self._issue(
                        "relation_temporal_invalid",
                        f"Relation {key} is not valid before the cutoff.",
                        point_id,
                    )
                )
        return issues

    @staticmethod
    def reject_failed_points(
        state: ReviewStateV3,
        report: VerificationReport,
    ) -> int:
        """Apply the only allowed repair: delete points named by verifier issues."""
        rejected = 0
        for point_id in {
            issue.point_id for issue in report.issues if issue.point_id is not None
        }:
            point = state.canonical_points.get(point_id)
            if point is None or not point.retained:
                continue
            point.retained = False
            point.validation_status = PointValidationStatus.REJECTED
            point.validation_notes.append("delete_only_verifier_repair")
            rejected += 1
        return rejected

    def _deterministic_issues(
        self,
        review: StructuredReview,
        state: ReviewState,
        paper_ir: PaperIR,
        evidence_store: EvidenceStore,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
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
                        "Visible review text uses graph fields as a scientific "
                        "judgment.",
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
                    "Visible review summary uses graph fields as a scientific "
                    "judgment.",
                )
            )
        return issues

    def _semantic_issues(
        self,
        review: StructuredReview,
        paper_ir: PaperIR,
    ) -> list[VerificationIssue]:
        if self.semantic_checker is None:
            return []
        self.semantic_calls += 1
        evidence = {
            f"P:{span.span_id}": span.text
            for span in paper_ir.spans
            if f"P:{span.span_id}" in _review_evidence_keys(review)
        }
        external_evidence: dict[str, Any] = {}
        for point in review.all_points():
            for key in point.evidence_keys:
                if not key.startswith(("R:", "COV:")):
                    continue
                # The caller already validates these immutable evidence records.
                external_evidence[key] = True
        user = json.dumps(
            {
                "review": review.model_dump(mode="json"),
                "paper_evidence": evidence,
                "external_evidence_keys": external_evidence,
                "output": {
                    "unsupported_point_ids": [],
                    "summary_supported": True,
                },
            },
            ensure_ascii=False,
        )
        try:
            payload = self.semantic_checker(
                "Judge manuscript assertions against cited paper spans. Treat clauses "
                "explicitly describing audited prior-art search scope as externally "
                "supported when they cite validated R: or COV: keys; do not require "
                "those search-process clauses to appear in the manuscript.",
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
        issues: list[VerificationIssue] = []
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
        if key.startswith("COV:"):
            record = evidence_store.get(key)
            return bool(
                record
                and record.kind == "retrieval_coverage"
                and record.payload.get("service_failed") is False
            )
        return False

    @staticmethod
    def _issue(
        code: str,
        message: str,
        point_id: str | None = None,
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

#!/usr/bin/env python3
"""Score the Nature dev100 GEAR-agent baseline against audited human reviews."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.gear.review_reconstruction.contracts import ReconstructionSessionResponse
from experiments.gear.review_reconstruction.evaluation import (
    MatchJudgePackage,
    MatchJudgeResponse,
    MatchLabel,
    PointMatchDecision,
    evaluate_corpus,
    evaluate_review_pair,
    validate_match_judge_response,
)
from gear.review_contracts import ReviewBundle, StructuredReview

DEFAULT_RECONSTRUCTION_ROOT = Path(
    "outputs/gear/reconstruction/nature_dev100"
)
DEFAULT_AGENT_ROOT = Path(
    "outputs/gear/consistency/nature_dev100/agent_reviews"
)
DEFAULT_JUDGE_ROOT = Path(
    "outputs/gear/consistency/nature_dev100/match_judging"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/gear/consistency/nature_dev100/results"
)


def _paper_dirs(root: Path) -> list[Path]:
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    return [Path(row["path"]).resolve().parent for row in manifest["sessions"]]


def _load_reconstruction(session_dir: Path) -> ReconstructionSessionResponse:
    response = ReconstructionSessionResponse.model_validate_json(
        (session_dir / "response.json").read_text(encoding="utf-8")
    )
    return response


def _valid_evidence_keys(session_dir: Path) -> set[str]:
    package = json.loads((session_dir / "package.json").read_text(encoding="utf-8"))
    spans = package["paper_context"]["spans"]
    return {str(span["evidence_key"]) for span in spans}


def _final_human(paper_dir: Path) -> ReconstructionSessionResponse:
    return _load_reconstruction(paper_dir / "reconstruction")


def _agent(agent_root: Path, slug: str, paper_id: str) -> tuple[StructuredReview, ReviewBundle]:
    run_dir = agent_root / slug
    review = StructuredReview.model_validate_json(
        (run_dir / "review.json").read_text(encoding="utf-8")
    ).model_copy(update={"paper_id": paper_id})
    bundle = ReviewBundle.model_validate_json(
        (run_dir / "review_bundle.json").read_text(encoding="utf-8")
    )
    return review, bundle


def _decisions(
    judge_root: Path, slug: str, paper_id: str
) -> tuple[list[PointMatchDecision], str]:
    session_dir = judge_root / slug
    package = MatchJudgePackage.model_validate_json(
        (session_dir / "package.json").read_text(encoding="utf-8")
    )
    response = MatchJudgeResponse.model_validate_json(
        (session_dir / "response.json").read_text(encoding="utf-8")
    )
    validate_match_judge_response(package, response)
    decisions = [
        row.model_copy(update={"paper_id": paper_id}) for row in response.decisions
    ]
    return decisions, response.model_id


def _matched_ids(decisions: list[PointMatchDecision]) -> tuple[set[str], set[str]]:
    same = sorted(
        (row for row in decisions if row.label == MatchLabel.SAME_POINT),
        key=lambda row: (-row.confidence, row.reference_point_id, row.candidate_point_id),
    )
    references: set[str] = set()
    candidates: set[str] = set()
    for row in same:
        if row.reference_point_id in references or row.candidate_point_id in candidates:
            continue
        references.add(row.reference_point_id)
        candidates.add(row.candidate_point_id)
    return references, candidates


def _publication_year(slug: str) -> str:
    match = re.search(r"-(\d{3})-", slug)
    return str(2000 + int(match.group(1)[-2:])) if match else "unknown"


def _journal_code(slug: str) -> str:
    match = re.match(r"10\.1038_([^-/]+)-", slug)
    return match.group(1) if match else "unknown"


def _point_band(count: int) -> str:
    if count <= 5:
        return "low_0_5"
    if count <= 9:
        return "medium_6_9"
    return "high_10_plus"


def _strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = (
        "publication_year",
        "journal",
        "agent_status",
        "reference_novelty",
        "reference_point_band",
    )
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        output[dimension] = {}
        for value, group in sorted(groups.items()):
            metrics = [row["metrics_model"] for row in group]
            novelty = [row["novelty_pair"] for row in group]
            output[dimension][value] = evaluate_corpus(
                metrics,
                novelty,
                development_non_confirmatory=True,
                bootstrap_samples=1000,
            ).model_dump(mode="json")
    return output


def _audit_row(
    row: dict[str, Any], reference: StructuredReview, candidate: StructuredReview
) -> dict[str, Any]:
    matched_reference, matched_candidate = _matched_ids(row["decisions"])
    return {
        "paper_id": reference.paper_id,
        "slug": row["slug"],
        "atomic_f1": row["metrics_model"].atomic_f1,
        "agent_status": row["agent_status"],
        "reference_point_count": len(reference.all_points()),
        "candidate_point_count": len(candidate.all_points()),
        "unmatched_reference_points": [
            point.model_dump(mode="json")
            for point in reference.all_points()
            if point.point_id not in matched_reference
        ],
        "unmatched_candidate_points": [
            point.model_dump(mode="json")
            for point in candidate.all_points()
            if point.point_id not in matched_candidate
        ],
        "partial_count": row["metrics_model"].partial_count,
        "contradictory_count": row["metrics_model"].contradictory_count,
    }


def _render_report(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    macro = corpus["paper_macro"]
    intervals = corpus["bootstrap_95_ci"]
    metric_rows = []
    for key in (
        "atomic_precision",
        "atomic_recall",
        "atomic_f1",
        "major_weakness_question_recall",
        "novelty_point_f1",
        "valid_evidence_key_ratio",
        "unsupported_major_rate",
    ):
        lower, upper = intervals[key]
        metric_rows.append(
            f"| {key} | {macro[key]:.4f} | [{lower:.4f}, {upper:.4f}] |"
        )
    lines = [
        "# Nature dev100 GEAR agent–human consistency",
        "",
        "This is a development-set, non-confirmatory evaluation over 100 papers.",
        "",
        "## Availability result",
        "",
        f"- Structurally available agent outputs: {summary['agent_structured_output_available_count']}/100",
        f"- Agent status counts: `{json.dumps(summary['agent_status_counts'], sort_keys=True)}`",
        f"- Critic source counts: `{json.dumps(summary['critic_source_counts'], sort_keys=True)}`",
        f"- Verification-passed runs: {summary['agent_verification_passed_count']}/100",
        f"- Semantic verification available: {summary['semantic_verification_available_count']}/100",
        f"- Graph-semantic validation violations: {summary['graph_semantic_violation_count']}",
        f"- Candidate atomic points: {summary['candidate_point_count']}",
        f"- Human-reference atomic points: {summary['reference_point_count']}",
        "",
        "Availability indicators above describe the executed agent-review backend.",
        "",
        "## System-level consistency",
        "",
        "| Metric | Paper macro | Bootstrap 95% CI |",
        "|---|---:|---:|",
        *metric_rows,
        "",
        f"Novelty judgment accuracy: {corpus['novelty_judgment_accuracy']:.4f}",
        "",
        f"Novelty judgment macro-F1: {corpus['novelty_judgment_macro_f1']:.4f}",
        "",
        "Conditional metrics are reported only when the agent produced usable structured reviews.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `sample_metrics.jsonl`: one row per paper",
        "- `corpus_metrics.json`: aggregate metrics and bootstrap intervals",
        "- `stratified_metrics.json`: year, journal, status, novelty, and point-volume strata",
        "- `disagreement_audit.json`: 20 lowest-agreement cases",
        "- `summary.json`: machine-readable experiment summary",
        "",
    ]
    return "\n".join(lines)


def evaluate(
    reconstruction_root: Path,
    agent_root: Path,
    judge_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    audits: list[tuple[dict[str, Any], StructuredReview, StructuredReview]] = []
    for paper_dir in _paper_dirs(reconstruction_root):
        slug = paper_dir.name
        try:
            session_dir = paper_dir / "reconstruction"
            human_response = _final_human(paper_dir)
            reference = human_response.review
            candidate, bundle = _agent(agent_root, slug, reference.paper_id)
            decisions, judge_model_id = _decisions(
                judge_root, slug, reference.paper_id
            )
            semantic_ids = (
                {point.point_id for point in candidate.all_points()}
                if bundle.verification.passed
                and bundle.verification.semantic_verification_available
                else None
            )
            valid_keys = _valid_evidence_keys(session_dir)
            metrics = evaluate_review_pair(
                reference,
                candidate,
                decisions,
                valid_evidence_keys=valid_keys,
                semantically_supported_point_ids=semantic_ids,
                development_non_confirmatory=True,
            )
            point_count = len(reference.all_points())
            row = {
                "slug": slug,
                "paper_id": reference.paper_id,
                "publication_year": _publication_year(slug),
                "journal": _journal_code(slug),
                "agent_status": bundle.status.value,
                "critic_source": bundle.critic.critic_source.value,
                "critic_model_id": bundle.critic.model_id,
                "verification_passed": bundle.verification.passed,
                "semantic_verification_available": (
                    bundle.verification.semantic_verification_available
                ),
                "judge_model_id": judge_model_id,
                "reference_novelty": reference.novelty.judgment.value,
                "candidate_novelty": candidate.novelty.judgment.value,
                "reference_point_count": point_count,
                "candidate_point_count": len(candidate.all_points()),
                "reference_point_band": _point_band(point_count),
                "metrics_model": metrics,
                "novelty_pair": (reference.novelty.judgment, candidate.novelty.judgment),
                "decisions": decisions,
            }
            rows.append(row)
            audits.append((row, reference, candidate))
        except (FileNotFoundError, OSError, ValueError) as exc:
            failures.append(f"{slug}:{type(exc).__name__}:{exc}")
    if len(rows) != 100 or failures:
        raise ValueError(f"consistency inputs incomplete: rows={len(rows)} failures={failures[:5]}")
    corpus = evaluate_corpus(
        [row["metrics_model"] for row in rows],
        [row["novelty_pair"] for row in rows],
        development_non_confirmatory=True,
        bootstrap_samples=5000,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"metrics_model", "novelty_pair", "decisions"}}
        item["metrics"] = row["metrics_model"].model_dump(mode="json")
        serializable.append(item)
    (output_root / "sample_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in serializable),
        encoding="utf-8",
    )
    (output_root / "corpus_metrics.json").write_text(
        corpus.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    strata = _strata(rows)
    (output_root / "stratified_metrics.json").write_text(
        json.dumps(strata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    disagreement = [
        _audit_row(row, reference, candidate)
        for row, reference, candidate in sorted(
            audits, key=lambda value: value[0]["metrics_model"].atomic_f1
        )[:20]
    ]
    (output_root / "disagreement_audit.json").write_text(
        json.dumps(disagreement, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    available = [
        row
        for row in rows
        if row["critic_source"] != "unavailable"
        and row["candidate_point_count"] > 0
    ]
    conditional = (
        evaluate_corpus(
            [row["metrics_model"] for row in available],
            [row["novelty_pair"] for row in available],
            development_non_confirmatory=True,
            bootstrap_samples=5000,
        ).model_dump(mode="json")
        if available
        else None
    )
    summary = {
        "paper_count": len(rows),
        "agent_structured_output_available_count": len(available),
        "agent_structured_output_availability_rate": len(available) / len(rows),
        "reference_point_count": sum(row["reference_point_count"] for row in rows),
        "candidate_point_count": sum(row["candidate_point_count"] for row in rows),
        "reference_novelty_counts": dict(
            Counter(row["reference_novelty"] for row in rows)
        ),
        "candidate_novelty_counts": dict(
            Counter(row["candidate_novelty"] for row in rows)
        ),
        "agent_status_counts": dict(Counter(row["agent_status"] for row in rows)),
        "critic_source_counts": dict(Counter(row["critic_source"] for row in rows)),
        "critic_model_counts": dict(Counter(row["critic_model_id"] for row in rows)),
        "agent_verification_passed_count": sum(
            row["verification_passed"] for row in rows
        ),
        "semantic_verification_available_count": sum(
            row["semantic_verification_available"] for row in rows
        ),
        "graph_semantic_violation_count": sum(
            row["metrics_model"].graph_semantic_violation_count for row in rows
        ),
        "judge_model_counts": dict(Counter(row["judge_model_id"] for row in rows)),
        "judge_reasoning_effort": "high",
        "judge_candidate_pair_count": sum(len(row["decisions"]) for row in rows),
        "codex_judge_session_count": sum(
            row["judge_model_id"] == "gpt-5.6-terra" for row in rows
        ),
        "corpus": corpus.model_dump(mode="json"),
        "conditional_on_available_agent_outputs": conditional,
        "failures": failures,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "RESULTS.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-root", type=Path, default=DEFAULT_RECONSTRUCTION_ROOT)
    parser.add_argument("--agent-root", type=Path, default=DEFAULT_AGENT_ROOT)
    parser.add_argument("--judge-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = evaluate(
        args.reconstruction_root.resolve(),
        args.agent_root.resolve(),
        args.judge_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

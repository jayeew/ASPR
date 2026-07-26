"""Score Fig.10 generic LLM baseline with the Fig.4 semantic matcher."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig04.old.main_fig4 import (  # noqa: E402
    INNOVATION_ASPECTS,
    candidate_aspect_for_point,
    candidate_records_for_peer_aspect,
    group_jsonl_by_kind,
    normalize_semantic_relation,
    normalize_whitespace,
    read_csv_records,
    semantic_match_one_point,
    semantic_relation_score,
)
from experiments.fig10.old.build_fig10_generic_baseline import (  # noqa: E402
    REQUIRED_REVIEW_SECTIONS,
    parse_json_response,
    review_points,
    score_1_5,
    write_jsonl,
)


DEFAULT_FIG4_DIR = PROJECT_ROOT / "outputs" / "fig04/old"
DEFAULT_FIG10_DIR = PROJECT_ROOT / "outputs" / "fig10/old"
DEFAULT_BASELINE_OUTPUTS = DEFAULT_FIG10_DIR / "fig10_generic_llm_baseline_outputs.jsonl"
SCORING_PROTOCOL = "same_fig4_semantic_matcher"
ASPECT_MAP = {
    "novelty": "novelty",
    "significance": "significance",
    "prior_art_comparison": "prior_art",
    "evidence_rigor": "evidence_rigor",
    "limitations": "limitations",
    "future_work": "future_work",
}
METRIC_COLUMNS = [
    "semantic_agreement",
    "novelty_coverage",
    "prior_art_accuracy",
    "factuality",
    "readability",
    "unsupported_claim_rate",
    "evidence_trace_completeness",
    "review_structure_coverage",
]
EXCLUSION_COLUMNS = [
    "case_id",
    "paper_id",
    "title",
    "exclusion_reason",
    "peer_point_count",
    "baseline_run_status",
    "exclusion_policy",
    "included_in_evaluable_sample",
]


def raw_response_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return parsed baseline response from a saved raw output row."""
    parsed = raw.get("parsed_response")
    if isinstance(parsed, Mapping):
        return dict(parsed)
    response_text = raw.get("raw_response")
    if isinstance(response_text, str) and response_text.strip():
        return parse_json_response(response_text)
    return {}


def generic_response_to_fig4_label(paper_id: str, response: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a generic LLM response into the Fig.4 innovation-label schema."""
    scores = response.get("scores_1_5") if isinstance(response.get("scores_1_5"), Mapping) else {}
    aspects: Dict[str, Any] = {}
    for fig4_aspect, response_aspect in ASPECT_MAP.items():
        points = review_points(response, response_aspect)
        aspects[fig4_aspect] = {
            "score_1_5": score_1_5(scores.get(response_aspect), default=3.0),
            "points": points,
            "quotes": [],
            "confidence": response.get("confidence", 0.5),
            "point_records": [
                {
                    "point": point,
                    "quote": "",
                    "evidence_type": "generic_llm_review_point",
                }
                for point in points
            ],
        }
    return {
        "paper_id": paper_id,
        "kind": "generic_llm_baseline",
        "overall_innovation_stance": {
            "score_1_5": score_1_5(scores.get("overall"), default=3.0),
            "label": str(response.get("recommendation", "")),
            "quote": "",
            "confidence": response.get("confidence", 0.5),
        },
        "aspects": aspects,
    }


def peer_points_for_aspect(peer_label: Mapping[str, Any], aspect: str) -> tuple[List[str], List[str]]:
    """Return Fig.4 peer points and quotes for an aspect."""
    peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
    peer_item = peer_aspects.get(aspect) if isinstance(peer_aspects.get(aspect), Mapping) else {}
    points = [
        normalize_whitespace(str(value))
        for value in (peer_item.get("points") if isinstance(peer_item.get("points"), list) else [])
        if normalize_whitespace(str(value))
    ]
    quotes = [
        normalize_whitespace(str(value))
        for value in (peer_item.get("quotes") if isinstance(peer_item.get("quotes"), list) else [])
        if normalize_whitespace(str(value))
    ]
    if not points:
        points = quotes[:]
    return points, quotes


def peer_point_count(peer_label: Mapping[str, Any], max_points_per_aspect: int = 4) -> int:
    """Count the peer-review points that define the same-rubric denominator."""
    total = 0
    for aspect in INNOVATION_ASPECTS:
        points, _ = peer_points_for_aspect(peer_label, aspect)
        total += len(points[:max_points_per_aspect])
    return total


def classify_evaluable_cases(
    *,
    manifest: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], Mapping[str, Any]],
    raw_by_paper: Mapping[str, Mapping[str, Any]],
    max_points_per_aspect: int = 4,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Return zero-peer-point exclusions and case ids eligible for scoring."""
    exclusions: List[Dict[str, Any]] = []
    evaluable_ids: List[str] = []
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        peer_label = labels.get((paper_id, "peer_review"), {})
        count = peer_point_count(peer_label, max_points_per_aspect=max_points_per_aspect)
        if count > 0:
            evaluable_ids.append(paper_id)
            continue
        raw = raw_by_paper.get(paper_id, {})
        exclusions.append(
            {
                "case_id": paper_id,
                "paper_id": paper_id,
                "title": str(row.get("title") or ""),
                "exclusion_reason": "zero_peer_review_points",
                "peer_point_count": 0,
                "baseline_run_status": str(raw.get("run_status") or "missing_baseline_output"),
                "exclusion_policy": "pre_specified_evaluable_sample_exclusion",
                "included_in_evaluable_sample": 0,
            }
        )
    return exclusions, evaluable_ids


def same_rubric_status_from_counts(*, case_count: int, expected_case_count: int, excluded_case_count: int) -> str:
    """Classify whether the same-rubric baseline covers the full or evaluable sample."""
    if expected_case_count > 0 and case_count >= expected_case_count:
        return "observed_generic_llm_run_same_rubric"
    evaluable_count = expected_case_count - excluded_case_count
    if excluded_case_count > 0 and evaluable_count > 0 and case_count >= evaluable_count:
        return "observed_generic_llm_run_same_rubric_evaluable_complete"
    return "observed_generic_llm_run_same_rubric_partial"


def match_paper_same_rubric(
    *,
    paper_id: str,
    title: str,
    peer_label: Mapping[str, Any],
    generic_label: Mapping[str, Any],
    max_points_per_aspect: int = 4,
) -> List[Dict[str, Any]]:
    """Run Fig.4 point-level semantic matching for one generic baseline output."""
    rows: List[Dict[str, Any]] = []
    for aspect in INNOVATION_ASPECTS:
        peer_points, peer_quotes = peer_points_for_aspect(peer_label, aspect)
        for point_index, point in enumerate(peer_points[:max_points_per_aspect]):
            peer_quote = peer_quotes[min(point_index, len(peer_quotes) - 1)] if peer_quotes else ""
            candidate_records = candidate_records_for_peer_aspect(generic_label, aspect, point)
            agent_candidates = [record["point"] for record in candidate_records]
            match = semantic_match_one_point(
                title=title,
                aspect=aspect,
                peer_point=point,
                peer_quote=peer_quote,
                agent_candidates=agent_candidates,
                client=None,
            )
            relation = normalize_semantic_relation(match.get("relation"))
            candidate_aspect = candidate_aspect_for_point(match.get("best_agent_point", ""), candidate_records, aspect)
            rows.append(
                {
                    "paper_id": paper_id,
                    "case_id": paper_id,
                    "row_id": f"{aspect}:{point_index}",
                    "aspect": aspect,
                    "peer_point": point,
                    "peer_quote": peer_quote,
                    "agent_candidates": agent_candidates[:6],
                    "candidate_aspect": candidate_aspect,
                    "cross_aspect_match": bool(relation != "no_match" and candidate_aspect and candidate_aspect != aspect),
                    "bge_only_relation": relation,
                    "refined_relation": relation,
                    "relation": relation,
                    "score": semantic_relation_score(relation),
                    "match_backend": match.get("match_backend", ""),
                    "similarity": match.get("similarity"),
                    "rationale": match.get("rationale", ""),
                    "relation_source": "fig4_semantic_matcher_no_llm_refine",
                    "scoring_protocol": SCORING_PROTOCOL,
                    "source": "observed_generic_llm_run",
                }
            )
    return rows


def safe_mean(values: Iterable[float]) -> float:
    nums = [float(value) for value in values if pd.notna(value)]
    return sum(nums) / len(nums) if nums else float("nan")


def relation_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"entailed": 0, "related": 0, "contradicted": 0, "no_match": 0}
    for row in rows:
        relation = normalize_semantic_relation(row.get("relation"))
        counts[relation] = counts.get(relation, 0) + 1
    return counts


def aspect_score(rows: Sequence[Mapping[str, Any]], aspect: str) -> float:
    aspect_rows = [row for row in rows if str(row.get("aspect")) == aspect]
    return safe_mean(float(row.get("score", 0.0)) for row in aspect_rows) if aspect_rows else 0.0


def summarize_same_rubric_results(matches: Sequence[Mapping[str, Any]], expected_case_count: int) -> pd.DataFrame:
    """Summarize point-level matches into Fig.10 metric columns."""
    by_case: Dict[str, List[Mapping[str, Any]]] = {}
    for row in matches:
        by_case.setdefault(str(row.get("case_id") or row.get("paper_id")), []).append(row)
    out: List[Dict[str, Any]] = []
    for case_id, rows in sorted(by_case.items()):
        counts = relation_counts(rows)
        total = sum(counts.values())
        semantic = safe_mean(float(row.get("score", 0.0)) for row in rows)
        soft = (counts.get("entailed", 0) + counts.get("related", 0)) / total if total else 0.0
        contradiction = counts.get("contradicted", 0) / total if total else 0.0
        missing = counts.get("no_match", 0) / total if total else 1.0
        covered_aspects = sum(1 for aspect in INNOVATION_ASPECTS if any(str(row.get("aspect")) == aspect and float(row.get("score", 0.0)) > 0 for row in rows))
        out.append(
            {
                "case_id": case_id,
                "paper_id": case_id,
                "source": "observed_generic_llm_run",
                "run_status": "ok",
                "scoring_protocol": SCORING_PROTOCOL,
                "semantic_agreement": max(0.0, min(1.0, semantic)),
                "novelty_coverage": max(0.0, min(1.0, aspect_score(rows, "novelty"))),
                "prior_art_accuracy": max(0.0, min(1.0, aspect_score(rows, "prior_art_comparison"))),
                "factuality": max(0.0, min(1.0, 1.0 - contradiction)),
                "readability": 1.0,
                "unsupported_claim_rate": max(0.0, min(1.0, missing * 0.55 + contradiction * 0.45)),
                "evidence_trace_completeness": max(0.0, min(1.0, soft)),
                "review_structure_coverage": covered_aspects / len(INNOVATION_ASPECTS) if INNOVATION_ASPECTS else 0.0,
                "entailed_points": counts.get("entailed", 0),
                "related_points": counts.get("related", 0),
                "contradicted_points": counts.get("contradicted", 0),
                "no_match_points": counts.get("no_match", 0),
                "total_points": total,
                "expected_case_count": expected_case_count,
            }
        )
    return pd.DataFrame(out)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_same_rubric_baseline(
    *,
    fig4_dir: Path,
    baseline_outputs: Path,
    out_dir: Path,
    max_points_per_aspect: int = 4,
) -> Dict[str, Any]:
    """Build same-rubric generic LLM baseline tables from saved qwen outputs."""
    labels = group_jsonl_by_kind(fig4_dir / "fig4_innovation_label_judgements.jsonl")
    manifest = read_csv_records(fig4_dir / "fig4_manifest.csv")
    title_by_paper = {str(row.get("paper_id") or ""): str(row.get("title") or "") for row in manifest}
    raw_by_paper = {str(row.get("paper_id") or ""): row for row in read_jsonl(baseline_outputs)}
    exclusions, evaluable_case_ids = classify_evaluable_cases(
        manifest=manifest,
        labels=labels,
        raw_by_paper=raw_by_paper,
        max_points_per_aspect=max_points_per_aspect,
    )
    evaluable_case_set = set(evaluable_case_ids)
    matches: List[Dict[str, Any]] = []
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in evaluable_case_set:
            continue
        raw = raw_by_paper.get(paper_id, {})
        if not raw or raw.get("run_status") != "ok":
            continue
        response = raw_response_payload(raw)
        generic_label = generic_response_to_fig4_label(paper_id, response)
        peer_label = labels.get((paper_id, "peer_review"), {})
        matches.extend(
            match_paper_same_rubric(
                paper_id=paper_id,
                title=title_by_paper.get(paper_id, ""),
                peer_label=peer_label,
                generic_label=generic_label,
                max_points_per_aspect=max_points_per_aspect,
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "fig10_generic_llm_same_rubric_claim_matches.jsonl", matches)
    pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS).to_csv(out_dir / "fig10_generic_llm_same_rubric_exclusions.csv", index=False)
    results = summarize_same_rubric_results(matches, expected_case_count=len(manifest))
    if len(results):
        results["evaluable_case_count"] = int(len(evaluable_case_ids))
        results["excluded_case_count"] = int(len(exclusions))
    results.to_csv(out_dir / "fig10_generic_llm_same_rubric_results.csv", index=False)
    summary = results[METRIC_COLUMNS].agg(["mean", "std", "count"]).transpose().reset_index().rename(columns={"index": "metric"}) if len(results) else pd.DataFrame(columns=["metric", "mean", "std", "count"])
    summary.to_csv(out_dir / "fig10_generic_llm_same_rubric_summary.csv", index=False)
    status = same_rubric_status_from_counts(
        case_count=int(len(results)),
        expected_case_count=int(len(manifest)),
        excluded_case_count=int(len(exclusions)),
    )
    manifest_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scoring_protocol": SCORING_PROTOCOL,
        "case_count": int(len(results)),
        "expected_case_count": int(len(manifest)),
        "evaluable_case_count": int(len(evaluable_case_ids)),
        "excluded_case_count": int(len(exclusions)),
        "excluded_case_ids": [str(row["case_id"]) for row in exclusions],
        "exclusion_policy": "pre_specified_evaluable_sample_exclusion_for_zero_peer_review_points",
        "match_count": int(len(matches)),
        "baseline_outputs": str(baseline_outputs),
        "results_csv": str(out_dir / "fig10_generic_llm_same_rubric_results.csv"),
        "matches_jsonl": str(out_dir / "fig10_generic_llm_same_rubric_claim_matches.jsonl"),
        "exclusions_csv": str(out_dir / "fig10_generic_llm_same_rubric_exclusions.csv"),
    }
    (out_dir / "fig10_generic_llm_same_rubric_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig4-dir", type=Path, default=DEFAULT_FIG4_DIR)
    parser.add_argument("--baseline-outputs", type=Path, default=DEFAULT_BASELINE_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG10_DIR)
    parser.add_argument("--max-points-per-aspect", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = build_same_rubric_baseline(
        fig4_dir=args.fig4_dir,
        baseline_outputs=args.baseline_outputs,
        out_dir=args.out_dir,
        max_points_per_aspect=args.max_points_per_aspect,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

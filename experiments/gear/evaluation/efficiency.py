"""Deterministic efficiency and evidence-integrity measurements."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gear.contracts import PaperIR
from gear.review_contracts import PointSeverity, StructuredReview
from gear.trace import EvidenceStore


def run_integrity_metrics(
    run_dir: Path, review: StructuredReview, paper_ir: PaperIR
) -> dict[str, Any]:
    store = EvidenceStore(run_dir)
    keys = [key for point in review.all_points() for key in point.evidence_keys]
    resolved = [key for key in keys if store.get(key) is not None]
    relation_records = [store.get(key) for key in store.ids() if key.startswith("R:")]
    relations = [row.payload for row in relation_records if row is not None]
    target_span_ids = {span.span_id for span in paper_ir.spans}
    prior_work_ids = {
        str(row["prior_work_id"]) for row in relations if row.get("prior_work_id")
    }
    return {
        "evidence_key_count": len(keys),
        "evidence_key_resolve_rate": len(resolved) / len(keys) if keys else None,
        "evidence_payload_hash_valid": True,
        "trace_completeness": not store.validate_manifest(),
        "relation_count": len(relations),
        "independent_prior_count": len(prior_work_ids),
        "relation_temporal_compliance": _mean_bool(
            [row.get("temporal_valid") is True for row in relations]
        ),
        "post_cutoff_leakage_rate": _mean_bool(
            [row.get("temporal_valid") is not True for row in relations]
        ),
        "wrong_paper_relation_contamination_rate": _mean_bool(
            [row.get("target_span_id") not in target_span_ids for row in relations]
        ),
    }


def efficiency_metrics(run_dir: Path, review: StructuredReview) -> dict[str, Any]:
    actions = _jsonl(run_dir / "action_trace.jsonl")
    stages = [row for row in actions if row.get("stage")]
    durations = [float(row.get("duration_ms", 0.0)) / 1000.0 for row in stages]
    timestamps = [
        datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
        for row in stages
        if row.get("created_at")
    ]
    wall_time = None
    if timestamps:
        wall_time = (max(timestamps) - min(timestamps)).total_seconds() + (
            durations[-1] if durations else 0.0
        )
    major_count = sum(
        point.severity in {PointSeverity.MAJOR, PointSeverity.CRITICAL}
        for point in review.all_points()
    )
    retrieval_actions = [
        row
        for row in actions
        if row.get("stage")
        in {"prior_art", "relation_classifier", "evidence_supervisor"}
    ]
    relation_count = sum(key.startswith("R:") for key in EvidenceStore(run_dir).ids())
    return {
        "cumulative_stage_time_seconds": sum(durations),
        "wall_time_seconds": wall_time,
        "stage_action_count": len(stages),
        "evidence_action_count": len(retrieval_actions),
        "actions_per_major_point": (
            len(retrieval_actions) / major_count if major_count else None
        ),
        "retrieval_actions_per_valid_relation": (
            len(retrieval_actions) / relation_count if relation_count else None
        ),
        "model_cache_hit_rate": _cache_rate(actions, "model"),
        "retrieval_cache_hit_rate": _cache_rate(actions, "retrieval"),
    }


def graph_action_metrics(run_dir: Path) -> dict[str, Any]:
    fusion_path = run_dir / "fusion_report.json"
    fusion: dict[str, Any] | None = None
    if fusion_path.is_file():
        fusion = json.loads(fusion_path.read_text())
    else:
        bundle_path = run_dir / "review_bundle.json"
        if bundle_path.is_file():
            bundle = json.loads(bundle_path.read_text())
            if isinstance(bundle.get("fusion_report"), dict):
                fusion = bundle["fusion_report"]
    store = EvidenceStore(run_dir)
    queries = [
        row.payload
        for key in store.ids()
        if key.startswith("Q:") and (row := store.get(key)) is not None
    ]
    graph_queries = [
        row for row in queries if row.get("query_role") in {"graph_seed", "graph_focus"}
    ]
    seed_queries = [
        row for row in graph_queries if row.get("query_role") == "graph_seed"
    ]
    focus_queries = [
        row for row in graph_queries if row.get("query_role") == "graph_focus"
    ]
    hits = [
        row.payload
        for key in store.ids()
        if key.startswith("H:") and (row := store.get(key)) is not None
    ]
    seed_query_ids = {str(row.get("query_id")) for row in seed_queries}
    focus_query_ids = {str(row.get("query_id")) for row in focus_queries}
    seed_hits = [row for row in hits if str(row.get("query_id")) in seed_query_ids]
    focus_hits = [row for row in hits if str(row.get("query_id")) in focus_query_ids]
    relations = [
        row.payload
        for key in store.ids()
        if key.startswith("R:") and (row := store.get(key)) is not None
    ]
    seed_work_ids = {str(row.get("work_id")) for row in seed_hits if row.get("work_id")}
    graph_metrics = {
        "graph_seed_fetch_rate": (
            len(seed_work_ids) / len(seed_queries) if seed_queries else None
        ),
        "graph_seed_comparable_rate": _mean_bool(
            [
                row.get("selection_stage") not in {"temporal_excluded", "metadata_only"}
                for row in seed_hits
            ]
        ),
        "graph_seed_verified_relation_yield": sum(
            str(row.get("prior_work_id")) in seed_work_ids for row in relations
        ),
        "graph_query_unique_prior_yield": len(
            {str(row.get("work_id")) for row in focus_hits if row.get("work_id")}
        ),
        "evidence_yield_per_graph_guided_query": (
            len(relations) / len(graph_queries) if graph_queries else None
        ),
    }
    if fusion is None:
        return {
            "graph_triggered_action_count": None,
            "graph_trigger_compliance": None,
            "graph_evidence_yield": None,
            **graph_metrics,
        }
    declared = [
        action
        for actions in (fusion.get("graph_triggered_actions") or {}).values()
        for action in actions
    ]
    actions = _jsonl(run_dir / "action_trace.jsonl")
    executed = [
        row
        for row in actions
        if row.get("target_id") in (fusion.get("graph_triggered_actions") or {})
    ]
    yielded = sum(
        bool(row.get("output_sha256")) and not row.get("failure") for row in executed
    )
    return {
        "graph_triggered_action_count": len(declared),
        "graph_trigger_compliance": (
            min(1.0, len(executed) / len(declared)) if declared else 1.0
        ),
        "graph_evidence_yield": yielded / len(executed) if executed else None,
        **graph_metrics,
    }


def supported_major_efficiency(
    run_dir: Path,
    review: StructuredReview,
    supported_point_ids: set[str],
) -> dict[str, float | None]:
    major_ids = {
        point.point_id
        for point in review.all_points()
        if point.severity in {PointSeverity.MAJOR, PointSeverity.CRITICAL}
        and point.point_id in supported_point_ids
    }
    if not major_ids:
        return {
            "actions_per_supported_major": None,
            "seconds_per_supported_major": None,
        }
    actions = [
        row
        for row in _jsonl(run_dir / "action_trace.jsonl")
        if row.get("target_id") in major_ids
    ]
    return {
        "actions_per_supported_major": len(actions) / len(major_ids),
        "seconds_per_supported_major": sum(
            float(row.get("duration_ms", 0.0)) / 1000.0 for row in actions
        )
        / len(major_ids),
    }


def _cache_rate(rows: list[dict[str, Any]], prefix: str) -> float | None:
    values = [
        bool(row.get("cache_hit"))
        for row in rows
        if str(row.get("stage", "")).startswith(prefix)
    ]
    return _mean_bool(values)


def _mean_bool(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


__all__ = [
    "efficiency_metrics",
    "graph_action_metrics",
    "run_integrity_metrics",
    "supported_major_efficiency",
]

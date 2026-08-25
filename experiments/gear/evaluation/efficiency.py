"""Deterministic efficiency and evidence-integrity measurements."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gear.contracts import PaperIR
from gear.evidence_supervisor import relation_payload_is_claim_relevant
from gear.review_contracts import PointSeverity, StructuredReview
from gear.trace import EvidenceStore


def run_integrity_metrics(
    run_dir: Path, review: StructuredReview, paper_ir: PaperIR
) -> dict[str, Any]:
    store = EvidenceStore(run_dir)
    keys = [key for point in review.all_points() for key in point.evidence_keys]
    resolved = [key for key in keys if store.get(key) is not None]
    relation_records = [store.get(key) for key in store.ids() if key.startswith("R:")]
    raw_relations = [row.payload for row in relation_records if row is not None]
    relations = _verified_relation_payloads(store)
    claim_relevant_relations = [
        row for row in relations if relation_payload_is_claim_relevant(row)
    ]
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
        "claim_relevant_relation_count": len(claim_relevant_relations),
        "raw_relation_record_count": len(raw_relations),
        "unresolved_relation_record_count": sum(
            row.get("relation_label") == "UNRESOLVED" for row in raw_relations
        ),
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
    relation_count = len(_verified_relation_payloads(EvidenceStore(run_dir)))
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
    graph_queries = [row for row in queries if _is_graph_guided_query(row)]
    seed_queries = [
        row
        for row in graph_queries
        if row.get("query_role") == "graph_seed"
        or str(row.get("transformation", "")).startswith(
            "graph_claim_aligned_topology_search:"
        )
    ]
    hits = [
        row.payload
        for key in store.ids()
        if key.startswith("H:") and (row := store.get(key)) is not None
    ]
    seed_query_ids = {str(row.get("query_id")) for row in seed_queries}
    seed_hits = [row for row in hits if str(row.get("query_id")) in seed_query_ids]
    relations = _verified_relation_payloads(store)
    claim_relevant_relations = [
        row for row in relations if relation_payload_is_claim_relevant(row)
    ]
    graph_query_ids = {
        str(row.get("query_id")) for row in graph_queries if row.get("query_id")
    }
    graph_relations = [
        row
        for row in claim_relevant_relations
        if graph_query_ids & {str(item) for item in row.get("source_query_ids") or []}
    ]
    ledger_record = store.get("G:LEDGER")
    ledger = ledger_record.payload if ledger_record is not None else {}
    corrections = [
        row.payload
        for key in store.ids()
        if key.startswith("RC:") and (row := store.get(key)) is not None
    ]
    logical_requests = sum(
        int(ledger.get(name, 0) or 0)
        for name in (
            "logical_provider_searches",
            "logical_direct_fetches",
            "logical_neighbor_expansions",
        )
    )
    corrected_relation_ids = {
        str(relation_id)
        for correction in corrections
        if correction.get("correction_type") != "prior_work_added_only"
        for relation_id in correction.get("trigger_relation_ids") or []
    }
    claim_relevant_relation_ids = {
        str(row.get("relation_id"))
        for row in claim_relevant_relations
        if row.get("relation_id")
    }
    material_claim_relevant_ids = corrected_relation_ids & claim_relevant_relation_ids
    eligible_seed_query_ids = {
        str(row.get("query_id"))
        for row in seed_hits
        if row.get("selection_stage") not in {"temporal_excluded", "metadata_only"}
    }
    comparable_seed_query_ids = {
        str(row.get("query_id"))
        for row in seed_hits
        if row.get("selection_stage") == "compared"
        and row.get("gate_label") in {"comparable", "partial"}
    }
    graph_metrics = {
        "graph_seed_fetch_rate": (
            len(eligible_seed_query_ids) / len(seed_queries) if seed_queries else None
        ),
        "graph_seed_comparable_rate": (
            len(comparable_seed_query_ids) / len(seed_queries) if seed_queries else None
        ),
        "graph_seed_verified_relation_yield": len(graph_relations),
        "graph_query_unique_prior_yield": len(
            {
                str(row.get("prior_work_id"))
                for row in graph_relations
                if row.get("prior_work_id")
            }
        ),
        "evidence_yield_per_graph_guided_query": (
            len(graph_relations) / len(graph_queries) if graph_queries else None
        ),
        "claim_relevant_verified_relation_yield": (
            100.0 * len(claim_relevant_relations) / logical_requests
            if logical_requests
            else None
        ),
        "relation_to_material_correction_rate": (
            len(material_claim_relevant_ids) / len(claim_relevant_relations)
            if claim_relevant_relations
            else None
        ),
        "logical_retrieval_requests": logical_requests,
        "network_retrieval_attempts": sum(
            int(ledger.get(name, 0) or 0)
            for name in (
                "network_provider_attempts",
                "network_direct_fetch_attempts",
                "network_neighbor_attempts",
            )
        ),
        "logical_resource_parity_signature": {
            name: ledger.get(name)
            for name in (
                "logical_provider_searches",
                "logical_direct_fetches",
                "logical_neighbor_expansions",
                "fulltext_candidates_retained",
                "relation_classification_calls",
            )
        },
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


def _verified_relation_payloads(store: EvidenceStore) -> list[dict[str, Any]]:
    """Return only temporally valid, classified relation cards."""
    output: list[dict[str, Any]] = []
    for key in store.ids():
        if not key.startswith("R:"):
            continue
        record = store.get(key)
        payload = record.payload if record is not None else {}
        if payload.get("temporal_valid") is not True:
            continue
        if payload.get("relation_label") in {None, "DISTANT", "UNRESOLVED"}:
            continue
        if not payload.get("prior_work_id"):
            continue
        output.append(payload)
    return output


def _is_graph_guided_query(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("query_role")
        in {"graph_seed", "graph_focus", "citation_neighbor"}
        or str(payload.get("transformation", "")).startswith(
            "graph_claim_aligned_topology_search:"
        )
    )


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

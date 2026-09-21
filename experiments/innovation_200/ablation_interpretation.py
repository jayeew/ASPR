"""Interpret masked Graph evidence before writing ablation reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.innovation_200.common import experiment_config, write_json
from experiments.innovation_200.resource_guard import wait_for_memory
from gear.innovation.analysis import assess, evidence_payloads
from gear.innovation.contracts import ClaimSet
from gear.innovation.joint_graph import analyze_joint
from gear.innovation.locking import stage_lock

PATH_FIELDS = frozenset(
    {
        "direct_citation",
        "two_hop_path_count",
        "shared_reference_count",
        "shared_reference_salton",
        "parent_id_cache_hit",
    }
)
TEXT_FIELDS = frozenset(
    {
        "claim_id",
        "parent_paper_id",
        "claim_type",
        "claim_text",
        "publication_date",
    }
)


def mask_graph(card: dict[str, Any], system: str) -> dict[str, Any]:
    """Project raw facts only; never reuse full-condition interpreted prose."""
    row = json.loads(json.dumps(card))
    if system == "fusion_text_only":
        return {
            "claim": row["claim"],
            "historical_neighbors": [
                {k: v for k, v in n.items() if k in TEXT_FIELDS}
                for n in row.get("neighbors", [])
            ],
        }
    # Notes may contain summaries of the masked statistics or citation paths.
    row.pop("notes", None)
    if system == "fusion_no_metrics":
        row.pop("metrics", None)
    elif system == "fusion_no_citation_paths":
        for neighbor in row.get("neighbors", []):
            for key in PATH_FIELDS:
                neighbor.pop(key, None)
            neighbor["edge_type"] = "semantic_only"
        row["metrics"] = [
            metric
            for metric in row.get("metrics", [])
            if not any(
                term in str(metric.get("name", "")).lower()
                for term in ("citation", "path", "reference")
            )
        ]
    else:
        raise ValueError(f"Unknown masked condition: {system}")
    return row


def mask_joint(fact: dict[str, Any], system: str) -> dict[str, Any]:
    if system == "fusion_text_only":
        keys = {"input_claims", "missing_fact_claim_ids", "claims_without_neighbors"}
    elif system == "fusion_no_metrics":
        keys = {
            "input_claims",
            "historical_edges",
            "insertion_edges",
            "missing_fact_claim_ids",
            "claims_without_neighbors",
        }
    elif system == "fusion_no_citation_paths":
        keys = {
            key
            for key in fact
            if key != "notes"
            and not any(
                term in key.lower() for term in ("citation", "path", "reference")
            )
        }
    else:
        raise ValueError(f"Unknown masked condition: {system}")
    return {key: value for key, value in fact.items() if key in keys}


def masked_sources(root: Path, claims: ClaimSet, system: str) -> dict[str, Any]:
    raw = evidence_payloads(root / "graph/joint")
    result = {}
    for claim in claims.claims:
        key = f"GRAPH:{claim.claim_id}"
        # Missing raw facts are technical failures, unlike a valid empty neighborhood.
        result[key] = mask_graph(raw[key], system)
    key = f"JOINT_GRAPH:{claims.paper_id}"
    result[key] = mask_joint(raw[key], system)
    return result


def interpreted_context(root: Path, claims: ClaimSet, system: str) -> dict[str, Any]:
    """Reuse completed interpretations only for the exact same masked inputs."""
    sources = masked_sources(root, claims, system)
    target = root / "report_inputs" / system
    with stage_lock(root / ".locks" / system):
        target.mkdir(parents=True, exist_ok=True)
        return _interpret(root, target, claims, sources)


def _cached(path: Path, sources: dict[str, Any], generate: Callable[[], Any]) -> dict:
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("sources") == sources and saved.get("analysis") is not None:
        return saved["analysis"]
    wait_for_memory()
    analysis = generate().model_dump(mode="json")
    temporary = path.with_suffix(".tmp")
    write_json(temporary, {"sources": sources, "analysis": analysis})
    temporary.replace(path)
    return analysis


def _interpret(
    root: Path,
    target: Path,
    claims: ClaimSet,
    sources: dict[str, Any],
) -> dict[str, Any]:
    config = experiment_config()
    assessments, missing = [], []
    for claim in claims.claims:
        key = f"GRAPH:{claim.claim_id}"
        card = sources[key]
        neighbors = card.get("neighbors", card.get("historical_neighbors", []))
        if not neighbors:
            missing.append(claim.claim_id)
            continue
        path = target / f"{claim.claim_id.rsplit('::', 1)[-1]}.json"
        result = _cached(
            path,
            {key: card},
            lambda: assess(
                config,
                claim.claim_id,
                claim.normalized_claim_text,
                {key: card},
                "graph",
            ),
        )
        assessments.append(result)
    path = target / "joint.json"
    joint = _cached(
        path,
        sources,
        lambda: analyze_joint(
            config, claims.paper_id, sources, {c.claim_id for c in claims.claims}
        ),
    )
    limitations = [f"no_eligible_neighbors:{cid}" for cid in missing]
    branch = {
        "paper_id": claims.paper_id,
        "system": "graph",
        "claim_fingerprint": "",
        "schema_version": "innovation_v2",
        "status": (
            "limited"
            if limitations or any(a["limitations"] for a in assessments)
            else "complete"
        ),
        "assessments": assessments,
        "limitations": limitations,
    }
    return {
        "claims": claims.model_dump(mode="json")["claims"],
        "gear": json.loads((root / "gear/analysis.json").read_text()),
        "graph": branch,
        "joint_graph": joint,
    }

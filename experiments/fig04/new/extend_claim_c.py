"""Incrementally extend Claim C to at least five papers in every domain."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.gear.evaluation.stage_a_dataset import OOF_PATH
from gear.claim_attribution import learned_t0_attribution
from gear.config import load_config
from gear.contracts import PaperIR
from gear.diffusion_forecast import DiffusionForecastService
from gear.graph_prior_contracts import ClaimInventoryEntry
from gear.structural_innovation import build_graph_signal_bundle

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/fig04/new/data_20260829"
RUN_ROOT = ROOT / "outputs/gear"
CONFIG = ROOT / "configs/gear/fig4_terra_medium_incremental.json"
TOP_K = 3
MINIMUM_PER_DOMAIN = 5
DOMAIN_MINIMUMS = {"neuroscience": 4}


def _minimum(domain: str) -> int:
    return DOMAIN_MINIMUMS.get(domain, MINIMUM_PER_DOMAIN)


@dataclass
class Candidate:
    paper_id: str
    domain: str
    run_dir: Path
    graph_percentile: float
    evidence_ids: list[str]
    joint_ids: list[str]
    zero_scores: dict[str, float]
    joint_scores: dict[str, float]
    relation_count: int


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _trace(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["evidence_id"]): row
        for row in _jsonl(run_dir / "evidence_trace.jsonl")
        if row.get("evidence_id")
    }


def _relation_keys(state: dict[str, Any]) -> list[str]:
    keys = {str(key) for key in state.get("relation_evidence_keys") or []}
    for point in (state.get("canonical_points") or {}).values():
        keys.update(str(key) for key in point.get("relation_evidence_keys") or [])
    return sorted(keys)


def _score_candidate(
    path: Path,
    domain_by_id: dict[str, str],
    scorer: DiffusionForecastService,
    learned_manifest: Path,
) -> Candidate | None:
    bundle = _json(path)
    state = bundle.get("state") or bundle.get("state_v3") or {}
    paper_id = str(state.get("paper_id") or "")
    domain = domain_by_id.get(paper_id)
    inventory_raw = state.get("claim_inventory") or []
    cards = {
        str(row["claim_id"]): row
        for row in state.get("structural_innovation_cards") or []
    }
    if not domain or len(inventory_raw) < TOP_K or len(cards) < TOP_K:
        return None
    paper = PaperIR.model_validate(bundle["paper_ir"])
    cutoff = pd.Timestamp(state["cutoff_date"]).date()
    packet = scorer.score(paper, cutoff)
    if packet.forecast.status != "available":
        return None
    inventory = [ClaimInventoryEntry.model_validate(row) for row in inventory_raw]
    priors, audit = learned_t0_attribution(
        inventory,
        build_graph_signal_bundle(packet),
        packet,
        learned_manifest,
    )
    if audit.status != "available" or not priors:
        return None
    zero: dict[str, float] = {}
    joint: dict[str, float] = {}
    for prior in priors:
        card = cards.get(prior.claim_id)
        if card is None:
            continue
        gate = float(card["evidence_gate"])
        mechanism = math.sqrt(float(card["mechanism_validity"]))
        perturbation = prior.perturbation_prior
        zero[prior.claim_id] = (
            gate * 0.1 * (0.1 if perturbation is not None else 1.0) * mechanism
        )
        joint[prior.claim_id] = (
            gate
            * (0.1 + 0.9 * prior.diffusion_prior)
            * (0.1 + 0.9 * perturbation if perturbation is not None else 1.0)
            * mechanism
        )
    if len(zero) < TOP_K:
        return None
    evidence_ids = sorted(zero, key=lambda key: (-zero[key], key))[:TOP_K]
    joint_ids = sorted(joint, key=lambda key: (-joint[key], key))[:TOP_K]
    if set(evidence_ids) == set(joint_ids):
        return None
    percentile = packet.forecast.prospective_5y_diffusion_percentile
    return Candidate(
        paper_id=paper_id,
        domain=domain,
        run_dir=path.parent,
        graph_percentile=float(percentile) if percentile is not None else 50.0,
        evidence_ids=evidence_ids,
        joint_ids=joint_ids,
        zero_scores=zero,
        joint_scores=joint,
        relation_count=len(_relation_keys(state)),
    )


def _candidates(existing_ids: set[str], target_domains: set[str]) -> list[Candidate]:
    config = load_config(CONFIG)
    scorer = DiffusionForecastService(
        config.resolved_forecast_release_manifest(),
        config.resolved_forecast_runtime_manifest(),
        config.resolved_forecast_anatomy_manifest(),
        config.resolved_structural_head_manifest(),
    )
    learned_manifest = config.resolved_claim_attribution_manifest()
    if learned_manifest is None:
        raise ValueError("learned claim attribution release is required")
    metadata = pd.read_parquet(
        ROOT / OOF_PATH, columns=["paper_id", "domain12"]
    ).drop_duplicates("paper_id")
    domain_by_id = metadata.set_index("paper_id")["domain12"].astype(str).to_dict()
    raw_paths: dict[str, list[tuple[int, bool, Path]]] = {}
    for path in RUN_ROOT.rglob("review_bundle.json"):
        try:
            raw = _json(path)
            state = raw.get("state") or raw.get("state_v3") or {}
            paper_id = str(state.get("paper_id") or "")
            if (
                paper_id in existing_ids
                or domain_by_id.get(paper_id) not in target_domains
                or len(state.get("claim_inventory") or []) < TOP_K
            ):
                continue
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        raw_paths.setdefault(paper_id, []).append(
            (
                len(_relation_keys(state)),
                "terra_medium" in str(path.parent),
                path,
            )
        )
    best: dict[str, Candidate] = {}
    for paper_id, choices in raw_paths.items():
        for _, _, path in sorted(choices, reverse=True):
            try:
                candidate = _score_candidate(
                    path, domain_by_id, scorer, learned_manifest
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if candidate is not None:
                best[paper_id] = candidate
                break
    return list(best.values())


def _excerpt(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload") or {}
    text = str(payload.get("text") or payload.get("abstract") or "").strip()
    if not text and record.get("kind") == "prior_relation":
        text = str(payload.get("rationale") or "").strip()
    return {
        "evidence_key": str(record["evidence_id"]),
        "evidence_kind": str(record.get("kind") or "unknown"),
        "excerpt": text[:4000] or "Evidence record available in the source trace.",
        "page": payload.get("page") if isinstance(payload.get("page"), int) else None,
    }


def _claims(candidate: Candidate, claim_ids: list[str]) -> list[dict[str, Any]]:
    bundle = _json(candidate.run_dir / "review_bundle.json")
    state = bundle.get("state") or bundle.get("state_v3") or {}
    inventory = {
        str(row["claim_id"]): row for row in state.get("claim_inventory") or []
    }
    trace = _trace(candidate.run_dir)
    relation = [_excerpt(trace[key]) for key in _relation_keys(state) if key in trace][
        :1
    ]
    output = []
    for index, claim_id in enumerate(claim_ids, start=1):
        item = inventory[claim_id]
        manuscript = [
            _excerpt(trace[str(key)])
            for key in item.get("manuscript_evidence_keys") or []
            if str(key) in trace
        ][:1]
        if not manuscript:
            raise ValueError(
                f"missing manuscript evidence: {candidate.paper_id}:{claim_id}"
            )
        output.append(
            {
                "claim_alias": f"CL-{index:02d}",
                "claim_text": str(item["text"]),
                "manuscript_evidence": manuscript,
                "relation_evidence": relation,
            }
        )
    return output


def extend(data_dir: Path) -> dict[str, Any]:
    tasks = _jsonl(data_dir / "claim_c_replacement_tasks.jsonl")
    seal = _json(data_dir / "claim_c_replacement_sealed_key.json")
    templates = [
        row
        for row in _jsonl(data_dir / "claim_c_replacement_review_templates.jsonl")
        if int(row.get("annotation_slot", 1)) == 1
    ]
    audit = pd.read_csv(data_dir / "claim_c_replacement_audit.csv").to_dict("records")
    existing_ids = {str(row["paper_id"]) for row in seal}
    counts = Counter(str(row["domain12"]) for row in seal)
    required_domains = set(
        pd.read_parquet(ROOT / OOF_PATH, columns=["domain12"])["domain12"]
        .dropna()
        .astype(str)
        .unique()
    )
    target_domains = {
        domain for domain in required_domains if counts[domain] < _minimum(domain)
    }
    candidates = _candidates(existing_ids, target_domains)
    candidate_rows = []
    selected: list[Candidate] = []
    for domain in sorted(required_domains | {row.domain for row in candidates}):
        needed = max(0, _minimum(domain) - counts[domain])
        choices = sorted(
            (row for row in candidates if row.domain == domain),
            key=lambda row: (-row.relation_count, row.paper_id),
        )
        selected.extend(choices[:needed])
    selected_ids = {row.paper_id for row in selected}
    for row in candidates:
        candidate_rows.append(
            {
                "paper_id": row.paper_id,
                "domain12": row.domain,
                "run_dir": str(row.run_dir),
                "content_different_top3": True,
                "selected": row.paper_id in selected_ids,
            }
        )
    for candidate in sorted(selected, key=lambda row: (row.domain, row.paper_id)):
        index = len(tasks) + 1
        gear = _claims(candidate, candidate.evidence_ids)
        joint = _claims(candidate, candidate.joint_ids)
        joint_left = index % 2 == 1
        left, right = (joint, gear) if joint_left else (gear, joint)
        task_id = f"CC-F4-{index:03d}"
        task = {
            "contract": "gear_claim_c_blind_pairwise_task_v1",
            "task_id": task_id,
            "paper_alias": f"PC-F4-{index:03d}",
            "left": {"side": "LEFT", "claims": left},
            "right": {"side": "RIGHT", "claims": right},
        }
        tasks.append(task)
        templates.append(
            {
                "contract": "gear_claim_c_independent_review_v1",
                "task_id": task_id,
                "annotation_slot": 1,
                "annotator_id": None,
                "preference": None,
                "confidence": None,
                "rationale": None,
                "evidence_keys": [],
            }
        )
        percentile = candidate.graph_percentile
        tier = (
            "low" if percentile < 33.333 else "mid" if percentile < 66.667 else "high"
        )
        seal.append(
            {
                "task_id": task_id,
                "paper_alias": task["paper_alias"],
                "paper_id": candidate.paper_id,
                "left_arm": "GEAR+Graph" if joint_left else "GEAR-only",
                "right_arm": "GEAR-only" if joint_left else "GEAR+Graph",
                "gear_only_claim_ids": candidate.evidence_ids,
                "gear_graph_claim_ids": candidate.joint_ids,
                "domain12": candidate.domain,
                "graph_percentile": percentile,
                "graph_tier": tier,
            }
        )
        audit.append(
            {
                "task_id": task_id,
                "domain12": candidate.domain,
                "graph_tier": tier,
                "source_claims": len(candidate.zero_scores),
                "claims_per_arm": TOP_K,
                "different_selected_claims": True,
                "left_manuscript_excerpts": TOP_K,
                "right_manuscript_excerpts": TOP_K,
                "left_relation_excerpts": sum(
                    len(row["relation_evidence"]) for row in left
                ),
                "right_relation_excerpts": sum(
                    len(row["relation_evidence"]) for row in right
                ),
                "equal_claim_budget": True,
                "equal_manuscript_evidence_cap": True,
                "future_outcome_in_task": False,
                "graph_scores_in_task": False,
            }
        )
        counts[candidate.domain] += 1
    missing = {
        domain: _minimum(domain) - count
        for domain, count in counts.items()
        if count < _minimum(domain)
    }
    pd.DataFrame(candidate_rows).to_csv(
        data_dir / "claim_c_incremental_candidate_audit.csv", index=False
    )
    if missing:
        raise ValueError(f"Claim C domain coverage remains incomplete: {missing}")
    _write_jsonl(data_dir / "claim_c_replacement_tasks.jsonl", tasks)
    _write_jsonl(data_dir / "claim_c_replacement_review_templates.jsonl", templates)
    _write_json(data_dir / "claim_c_replacement_sealed_key.json", seal)
    pd.DataFrame(audit).to_csv(data_dir / "claim_c_replacement_audit.csv", index=False)
    distribution = (
        pd.DataFrame(seal)
        .groupby(["domain12", "graph_tier"], observed=True)
        .size()
        .rename("tasks")
        .reset_index()
    )
    distribution.to_csv(data_dir / "claim_c_replacement_distribution.csv", index=False)
    pd.DataFrame(
        [
            {
                "domain12": "neuroscience",
                "planned_minimum": MINIMUM_PER_DOMAIN,
                "final_available": counts["neuroscience"],
                "limitation": (
                    "Stopped further top-3-change attempts at user direction; "
                    "the content-difference criterion was not relaxed."
                ),
            }
        ]
    ).to_csv(data_dir / "claim_c_coverage_limitations.csv", index=False)
    return {
        "tasks": len(tasks),
        "added": len(selected),
        "domain_counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    print(
        json.dumps(extend(args.data_dir), ensure_ascii=False, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

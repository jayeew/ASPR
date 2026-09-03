"""Prepare only the missing Fig. 4new cases without rebuilding frozen outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.gear.evaluation.stage_a_dataset import OOF_PATH, SCORE_PATH

ROOT = Path(__file__).resolve().parents[3]
ACQUISITION = (
    ROOT
    / "outputs/gear/stage_b_targeted_expansion_20260828"
    / "acquisition_6014/acquisition_manifest.csv"
)
RUN_ROOT = ROOT / "outputs/gear"
TARGET_DOMAINS = ("clinical_health", "neuroscience")
EXPERT_PACK = (
    ROOT / "outputs/gear/graph_rescue_replication_20260828/expert_annotation_pack"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _existing_paper_ids() -> set[str]:
    values: set[str] = set()
    for path in RUN_ROOT.rglob("review_bundle.json"):
        try:
            bundle = _json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        state = bundle.get("state") or bundle.get("state_v3") or {}
        paper_id = state.get("paper_id")
        if paper_id:
            values.add(str(paper_id))
    return values


def build_claim_c_expansion_manifest(
    output_path: Path,
    per_domain: int | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Select all still-unreviewed available papers in the two remaining domains."""
    oof = pd.read_parquet(
        ROOT / OOF_PATH, columns=["paper_id", "domain12", "publication_year"]
    )
    score = pd.read_parquet(
        ROOT / SCORE_PATH,
        columns=["paper_id", "prospective_5y_diffusion_percentile"],
    )
    acquisition = pd.read_csv(ACQUISITION)[
        [
            "paper_id",
            "manuscript_path",
            "publication_date_resolved",
            "title",
            "doi",
        ]
    ]
    frame = oof.merge(score, on="paper_id", validate="one_to_one").merge(
        acquisition, on="paper_id", validate="one_to_one"
    )
    frame = frame[
        frame["domain12"].isin(TARGET_DOMAINS)
        & frame["manuscript_path"].notna()
        & ~frame["paper_id"].astype(str).isin(_existing_paper_ids())
    ].copy()
    if domain is not None:
        frame = frame[frame["domain12"].eq(domain)].copy()
    frame = frame.sort_values(
        ["domain12", "prospective_5y_diffusion_percentile", "paper_id"],
        ascending=[True, False, True],
    )
    if per_domain is not None:
        frame = frame.groupby("domain12", observed=True, group_keys=False).head(
            per_domain
        )
    cases = []
    for row in frame.itertuples(index=False):
        publication_date = str(row.publication_date_resolved)[:10]
        cases.append(
            {
                "case_id": str(row.paper_id).rsplit("/", 1)[-1],
                "paper_id": str(row.paper_id),
                "paper_path": str(Path(str(row.manuscript_path)).resolve()),
                "cutoff": publication_date,
                "domain12": str(row.domain12),
                "metadata": {
                    "title": str(row.title),
                    "doi": None if pd.isna(row.doi) else str(row.doi),
                    "openalex_id": str(row.paper_id),
                    "publication_date": publication_date,
                    "domain": str(row.domain12),
                },
            }
        )
    payload = {
        "selection_uses_future_outcomes": False,
        "purpose": "Claim C domain coverage only",
        "cases": cases,
    }
    _write_json(output_path, payload)
    return {
        "cases": len(cases),
        "domains": frame.groupby("domain12").size().astype(int).to_dict(),
        "output": str(output_path.resolve()),
    }


def _relation_records(run_dir: Path) -> list[dict[str, Any]]:
    """Return only complete, cutoff-safe relation traces from one local run."""
    trace = {
        str(row["evidence_id"]): row
        for row in _read_jsonl(run_dir / "evidence_trace.jsonl")
        if row.get("evidence_id")
    }
    bundle = _json(run_dir / "review_bundle.json")
    state = bundle.get("state") or bundle.get("state_v3") or {}
    cutoff = date.fromisoformat(str(state.get("cutoff_date") or "")[:10])
    keys = {str(key) for key in state.get("relation_evidence_keys") or []}
    for point in (state.get("canonical_points") or {}).values():
        keys.update(str(key) for key in point.get("relation_evidence_keys") or [])
    output = []
    for key in sorted(keys):
        relation = trace.get(key)
        payload = (relation or {}).get("payload") or {}
        prior_work_id = str(payload.get("prior_work_id") or "").strip()
        prior_date = str(payload.get("prior_work_date") or "").strip()
        target_span_id = str(payload.get("target_span_id") or "").strip()
        if (
            not payload.get("temporal_valid")
            or payload.get("temporal_order_unresolved")
            or not prior_work_id
            or not prior_date
            or not target_span_id
        ):
            continue
        try:
            cutoff_verified = date.fromisoformat(prior_date[:10]) < cutoff
        except ValueError:
            continue
        if not cutoff_verified:
            continue
        span_id = str(payload.get("prior_span_id") or "")
        work_id = str(payload.get("prior_work_id") or "")
        span = (trace.get(f"PS:{span_id}") or {}).get("payload") or {}
        work = (trace.get(f"W:{work_id}") or {}).get("payload") or {}
        text = str(span.get("text") or "").strip()
        stable_identifier = str(
            work.get("doi") or work.get("openalex_id") or prior_work_id
        ).strip()
        if not text or not stable_identifier:
            continue
        common = " ".join(
            str(value) for value in payload.get("common_dimensions") or []
        )
        differences = " ".join(
            str(value) for value in payload.get("difference_dimensions") or []
        )
        output.append(
            {
                "evidence_key": key,
                "evidence_kind": "prior_relation",
                "target_span_id": target_span_id,
                "target_claim_id": str(payload.get("target_claim_id") or ""),
                "prior_work_id": prior_work_id,
                "prior_work_title": str(work.get("title") or ""),
                "prior_work_date": prior_date,
                "prior_work_identifier": stable_identifier,
                "prior_work_location": str(
                    span.get("page")
                    or span.get("location")
                    or payload.get("evidence_level")
                    or ""
                ),
                "relation_label": str(payload.get("relation_label") or ""),
                "relation_rationale": str(payload.get("rationale") or ""),
                "cutoff_verified": cutoff_verified,
                "excerpt": (
                    f"Prior work: {work.get('title') or work_id} ({prior_date}). "
                    f"Quoted antecedent: {text} Shared dimensions: {common} "
                    f"Candidate residual dimensions: {differences}"
                )[:4000],
                "page": None,
            }
        )
    return output


def enrich_claim_b(
    output_path: Path, audit_path: Path, claim_table_path: Path
) -> dict[str, Any]:
    """Materialize only Claim-B claims with a direct, complete local relation trace."""
    tasks = _read_jsonl(EXPERT_PACK / "claim_b_tasks.jsonl")
    assignments = _json(EXPERT_PACK / "sealed_assignment_key.json")["claim_b"]
    by_task = {str(row["task_id"]): str(row["paper_id"]) for row in assignments}
    runs: dict[str, list[Path]] = defaultdict(list)
    target_ids = set(by_task.values())
    for bundle_path in RUN_ROOT.rglob("review_bundle.json"):
        try:
            bundle = _json(bundle_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        state = bundle.get("state") or bundle.get("state_v3") or {}
        paper_id = str(state.get("paper_id") or "")
        if paper_id in target_ids:
            runs[paper_id].append(bundle_path.parent)
    enriched: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for task in tasks:
        paper_id = by_task[str(task["task_id"])]
        relations_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
        source_runs: dict[str, str] = {}
        for run_dir in runs.get(paper_id, []):
            try:
                relations = _relation_records(run_dir)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for relation in relations:
                relations_by_span[relation["target_span_id"]].append(relation)
                source_runs[relation["evidence_key"]] = str(run_dir)
        eligible_claims = []
        for claim in task["claims"]:
            manuscript = list(claim.get("manuscript_evidence") or [])
            manuscript_span_ids = {
                str(evidence.get("evidence_key") or "")[2:]
                for evidence in manuscript
                if str(evidence.get("evidence_key") or "").startswith("P:")
            }
            matched = [
                relation
                for span_id in sorted(manuscript_span_ids)
                for relation in relations_by_span.get(span_id, [])
            ][:2]
            eligible = bool(manuscript) and bool(matched)
            for relation in matched or [{}]:
                claim_rows.append(
                    {
                        "claim_id": f"{task['task_id']}:{claim['claim_alias']}",
                        "task_id": task["task_id"],
                        "paper_alias": task["paper_alias"],
                        "claim_alias": claim["claim_alias"],
                        "manuscript_evidence_keys": ";".join(
                            str(item.get("evidence_key") or "") for item in manuscript
                        ),
                        "manuscript_excerpt": " ".join(
                            str(item.get("excerpt") or "") for item in manuscript
                        ),
                        "prior_work_id": relation.get("prior_work_id", ""),
                        "prior_work_identifier": relation.get(
                            "prior_work_identifier", ""
                        ),
                        "prior_work_title": relation.get("prior_work_title", ""),
                        "prior_work_date": relation.get("prior_work_date", ""),
                        "prior_work_excerpt": relation.get("excerpt", ""),
                        "prior_work_location": relation.get("prior_work_location", ""),
                        "cutoff_verified": bool(relation.get("cutoff_verified", False)),
                        "relation_evidence_key": relation.get("evidence_key", ""),
                        "relation_rationale": relation.get("relation_rationale", ""),
                        "relation_status": relation.get(
                            "relation_label", "UNVERIFIABLE"
                        ),
                        "residual_novelty_eligible": eligible,
                        "exclusion_reason": (
                            "" if eligible else "no_direct_complete_relation_trace"
                        ),
                        "source_run": source_runs.get(
                            relation.get("evidence_key", ""), ""
                        ),
                    }
                )
            if eligible:
                claim["relation_evidence"] = [
                    {
                        "evidence_key": relation["evidence_key"],
                        "evidence_kind": relation["evidence_kind"],
                        "excerpt": relation["excerpt"],
                        "page": relation["prior_work_location"] or None,
                    }
                    for relation in matched
                ]
                eligible_claims.append(claim)
        if eligible_claims:
            enriched.append({**task, "claims": eligible_claims})
        audit.append(
            {
                "task_id": task["task_id"],
                "paper_alias": task["paper_alias"],
                "included": bool(eligible_claims),
                "claims_total": len(task["claims"]),
                "claims_eligible": len(eligible_claims),
                "relation_excerpts": sum(
                    len(claim["relation_evidence"]) for claim in eligible_claims
                ),
                "reason": (
                    "direct_relation_traces_ready"
                    if eligible_claims
                    else "no_direct_complete_relation_trace"
                ),
            }
        )
    _write_jsonl(output_path, enriched)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    pd.DataFrame(claim_rows).to_csv(claim_table_path, index=False)
    return {
        "existing_tasks": len(tasks),
        "enriched_tasks": len(enriched),
        "eligible_claims": len(
            {row["claim_id"] for row in claim_rows if row["residual_novelty_eligible"]}
        ),
        "output": str(output_path.resolve()),
        "claim_table": str(claim_table_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/fig04/new/data_20260829/claim_c_incremental_benchmark.json",
    )
    parser.add_argument("--enrich-claim-b", action="store_true")
    parser.add_argument("--per-domain", type=int)
    parser.add_argument("--domain", choices=TARGET_DOMAINS)
    args = parser.parse_args()
    if args.enrich_claim_b:
        result = enrich_claim_b(
            ROOT / "outputs/fig04/new/data_20260829/claim_b_enriched_tasks.jsonl",
            ROOT / "outputs/fig04/new/data_20260829/claim_b_enrichment_audit.csv",
            ROOT / "outputs/fig04/new/data_20260829/claim_b_evidence_completion.csv",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(
        json.dumps(
            build_claim_c_expansion_manifest(args.output, args.per_domain, args.domain),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

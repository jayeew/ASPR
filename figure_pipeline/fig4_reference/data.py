"""Deterministic plotting tables; no generation or scientific relabelling."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_reference"
NEW = ROOT / "outputs/fig4_mechanisms"
OLD = ROOT / "outputs/fig3_reference/study"
EXT = ROOT / "outputs/fig3_reference/extensions"
ORDER = ["T", "E", "G", "F", "F_noJ", "F_noM", "F_noP"]
NAMES = dict(zip(ORDER, ["Text analysis", "GEAR + text", "Graph + text", "Full system",
                       "Full without Joint", "Full: metrics hidden", "Full: paths hidden"]))
COMPONENTS = ["GEAR", "Graph branch", "Joint", "Structural summaries", "Paper paths"]
ACTIONS = ["retain", "merge", "correct", "omit", "unresolved"]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def table(name: str, data: pd.DataFrame | list[dict[str, Any]]) -> None:
    pd.DataFrame(data).to_csv(OUT / "data" / (name + ".csv"), index=False, escapechar="\\")


def prepare() -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    conditions = read(NEW / "conditions.json")
    (OUT / "data/conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2))
    pm = pd.read_csv(NEW / "paper_metrics.csv")
    table("paper_metrics", pm)
    table("a_information_exposure", conditions["conditions"])
    endpoints = pm.groupby("condition").agg(n_papers=("paper_id", "nunique"), H=("H", "mean"),
        V=("V", "mean"), R=("R", "mean"), unknown_support_share=("unknown_support_share", "mean"),
        body_chars_median=("body_chars", "median")).reindex(ORDER).reset_index()
    endpoints["unknown_support_percent"] = endpoints.unknown_support_share * 100
    endpoints["name"] = endpoints.condition.map(NAMES)
    table("a_condition_endpoints", endpoints)
    effects = pd.read_csv(NEW / "component_effects_paper.csv")
    effects["display_delta"] = np.where(effects.metric.eq("R"), -effects.delta, effects.delta)
    table("b_paper_effects_improvement", effects)
    summary = pd.read_csv(NEW / "component_effects_summary.csv")
    for dst, positive, negative in [("display_mean", "mean_delta", "mean_delta"),
            ("display_ci_low", "ci_low", "ci_high"), ("display_ci_high", "ci_high", "ci_low"),
            ("display_q1", "q1", "q3"), ("display_median", "median", "median"), ("display_q3", "q3", "q1")]:
        summary[dst] = np.where(summary.metric.eq("R"), -summary[negative], summary[positive])
    table("b_effect_summary_improvement", summary)
    frequency = effects[effects.metric.eq("H")].groupby(["component", "display_delta"]).size()
    table("b_H_discrete_frequencies", frequency.rename("papers").reset_index())
    ip = pd.read_csv(NEW / "interaction_paper.csv")
    table("c_interaction_paper", ip)
    table("c_interaction_counts", ip.groupby("I").size().rename("papers").reset_index())
    (OUT / "data/c_interaction_summary.json").write_bytes((NEW / "interaction_summary.json").read_bytes())
    joint = pd.read_csv(NEW / "joint_effects_paper.csv").sort_values(["J_topo", "paper_id"], kind="stable")
    joint["structural_rank"] = np.arange(1, len(joint) + 1)
    joint["J_percent"] = joint.J_topo * 100
    table("d_structure_and_report_tracks", joint)
    table("d_nonzero_papers", joint[joint.delta_cross.ne(0)])
    cross_audit()
    fusion_tables()
    relation_tables()
    sources = ["outputs/fig4_mechanisms/paper_metrics.csv", "outputs/fig4_mechanisms/conditions.json",
        "outputs/fig4_mechanisms/component_effects_paper.csv", "outputs/fig4_mechanisms/component_effects_summary.csv",
        "outputs/fig4_mechanisms/interaction_paper.csv", "outputs/fig4_mechanisms/interaction_summary.json",
        "outputs/fig4_mechanisms/joint_effects_paper.csv", "outputs/fig4_mechanisms/joint_relations.jsonl",
        "outputs/fig4_mechanisms/information_clusters.jsonl", "outputs/fig3_reference/study/reports/full/papers/",
        "outputs/fig3_reference/study/derived/fusion_sets.jsonl", "outputs/fig3_reference/extensions/cd/first_stage/d_relation_candidates.jsonl",
        "outputs/fig3_reference/extensions/cd/d_verification/verification_results.jsonl",
        "outputs/fig3_reference/extensions/abcd/evaluation/report_trace_records.jsonl"]
    (OUT / "data/source_files.json").write_text(json.dumps(sources, indent=2))


def cross_audit() -> None:
    members = {(c["paper_id"], c["cluster_id"], key): member for c in rows(NEW / "information_clusters.jsonl")
               for key, member in c["members"].items()}
    audit = []
    for r in rows(NEW / "joint_relations.jsonl"):
        if r["condition"] not in {"F", "F_noJ"} or not r["valid"]:
            continue
        original = members[r["paper_id"], r["cluster_id"], r["unit_key"]]["support"]
        eligible = original["support"] == "supported" and original["scope_correct"] and original["nonparaphrase_insight"]
        audit.append({"paper_id": r["paper_id"], "condition": r["condition"], "cluster_id": r["cluster_id"],
            "unit_key": r["unit_key"], "relation_valid": r["valid"], "original_support": original["support"],
            "scope_correct": original["scope_correct"], "nonparaphrase_insight": original["nonparaphrase_insight"],
            "strict_eligible": eligible, "quote": r["original_quote"]})
    table("audit_cross_relation_vs_report_validity", audit)


def fusion_tables() -> None:
    decisions, outcomes = [], []
    normalize = dict(zip(["retained", "merged", "corrected", "omitted", "unresolved"], ACTIONS))
    for path in sorted((OLD / "reports/full/papers").glob("*.json")):
        report = read(path)
        for group in report["fusion_details"]:
            for r in group["decisions"]:
                decisions.append({"paper_id": path.stem, "claim_id": group["claim_id"], "finding_key": r["finding_key"],
                    "decision_action": r["action"], "decision_reason": r["reason"]})
        for group in report["finding_retention"]:
            for r in group["outcomes"]:
                outcomes.append({"paper_id": path.stem, "claim_id": group["claim_id"], "finding_key": r["finding_key"],
                    "report_action": normalize[r["action"]], "report_quote": r["report_quote"], "report_reason": r["reason"]})
    pairs = pd.DataFrame(decisions).merge(pd.DataFrame(outcomes), on=["paper_id", "claim_id", "finding_key"], validate="one_to_one")
    if len(pairs) != len(decisions) or len(pairs) != len(outcomes):
        raise ValueError("Fusion action records are not completely matched")
    table("e1_action_pairs", pairs)
    counts = pairs.groupby(["decision_action", "report_action"]).size()
    cells = []
    for a in ACTIONS:
        total = int(pairs.decision_action.eq(a).sum())
        for b in ACTIONS:
            n = int(counts.get((a, b), 0))
            cells.append({"decision_action": a, "report_action": b, "count": n, "row_total": total, "row_percent": 100 * n / total})
    table("e1_action_transition_cells", cells)
    information = []
    for r in rows(OLD / "derived/fusion_sets.jsonl"):
        groups = {k: set(r[k]) for k in ["full", "union", "retained", "added", "newly_supported", "not_retained", "mentioned_without_support"]}
        information.append({"paper_id": r["paper_id"], **{k: len(v) for k, v in groups.items()}})
    source = pd.DataFrame(information).sort_values("paper_id")
    table("e2_information_sources_paper", source)
    table("e2_information_sources_totals", [{"category": col, "count": int(source[col].sum())} for col in source if col != "paper_id"])
    table("supp_three_method_intersections", pd.read_csv(EXT / "abcd/B_intersection_statistics.csv"))


def relation_tables() -> None:
    candidates = rows(EXT / "cd/first_stage/d_relation_candidates.jsonl")
    verified = {r["id"]: r for r in rows(EXT / "cd/d_verification/verification_results.jsonl")}
    candidate_rows = [{"relation_id": r["id"], "paper_id": r["paper_id"], "cluster_id": r["cluster_id"],
        "verified_state": verified[r["id"]]["verified_state"] if r["id"] in verified else "not_submitted_source_gap"} for r in candidates]
    table("f1_all_candidates", candidate_rows)
    table("f1_evidence_status_counts", [{"verified_state": k, "count": v} for k, v in Counter(r["verified_state"] for r in candidate_rows).items()])
    traces = {(r["relation_id"], r["method"]): r for r in rows(EXT / "abcd/evaluation/report_trace_records.jsonl")}
    selected = sorted([r for r in candidate_rows if r["verified_state"] in {"supported", "supported_after_narrowing"}], key=lambda r: (r["paper_id"], r["relation_id"]))
    tiles, paired = [], Counter()
    correspondence = {"same_scope", "narrowed_or_corrected"}
    for rank, r in enumerate(selected, 1):
        present = []
        for method in ["gear", "fusion"]:
            trace = traces[r["relation_id"], method]
            state = trace["relation_treatment"]
            tiles.append({**r, "relation_rank": rank, "method": method, "treatment": state,
                "report_quotes": json.dumps(trace["report_quotes"], ensure_ascii=False)})
            present.append(state in correspondence)
        paired[(present[0], present[1])] += 1
    table("f2_relation_trace_tiles", tiles)
    table("f2_paired_treatment_counts", [{"gear_correspondence": k[0], "full_correspondence": k[1], "relations": v} for k, v in paired.items()])

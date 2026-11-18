"""Paper-level summaries derived from observed statements and paired observations."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .settings import DIMENSIONS, METHODS, OUTPUT, missing_path, read, roster, write

TYPES = ("supported_scope", "historical_increment", "knowledge_relation", "appropriate_limitation")
INSIGHTS = ("historical_increment", "cross_work_relation", "scope_correction", "cross_contribution")
STATES = ("supported", "partly_supported", "not_verifiable", "contradicted")
ERRORS = ("unsupported_definitive", "false_antecedence", "semantic_causal", "omitted_scope")


def interval(values: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(20260917)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high), "n": len(values)}


def paper_metrics(ident: str, method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inp = read(OUTPUT / "data/inputs" / f"{ident}.json")
    if any(missing_path(stage,ident,method).exists() for stage in ("reports","extract","support")):
        row={"paper_id":ident,"method":method,"claims":len(inp["claims"]),"status":"not_assessed"}
        quality_path=OUTPUT/"data/quality"/method/f"{ident}.json"
        if quality_path.exists():
            row.update({d["dimension"]:d["score"] if d["applicable"] else None for d in read(quality_path)["dimensions"]})
        return row,[]
    extracted = read(OUTPUT / "data/units" / method / f"{ident}.json")
    checks = {u["unit_id"]: u for u in read(OUTPUT / "data/support" / method / f"{ident}.json")["units"]}
    quality_path = OUTPUT / "data/quality" / method / f"{ident}.json"
    quality = ([] if missing_path("quality",ident,method).exists() and not quality_path.exists()
               else read(quality_path)["dimensions"])
    mentioned = {m["claim_id"] for m in extracted["mentions"] if m["quotes"]}
    substantive, supported = set(), set()
    commentary, eligible, unit_rows = {}, [], []
    for u in extracted["units"]:
        # A claim-linked, anchored assertion itself establishes mention, independently
        # of the extractor's separate mention list; no rate is clipped or inflated.
        mentioned.update(u["claim_ids"])
        v = checks[u["unit_id"]]
        valid = v["support"] == "supported" and v["scope_correct"] and bool(v["source_ids"]) and bool(v["source_quotes"])
        is_substantive = u["substantive"] and not u["paraphrase"]
        if is_substantive:
            substantive.update(u["claim_ids"])
            if valid:
                supported.update(u["claim_ids"])
                commentary.setdefault(u["cluster_id"], u["primary_type"])
        if u["needs_verification"]:
            eligible.append(v)
        unit_rows.append({"paper_id": ident, "method": method, **u, **v})
    if not supported <= substantive <= mentioned:
        raise ValueError(f"Contribution coverage violates nesting for {ident}/{method}; repair extraction, do not clamp")
    count = len(inp["claims"])
    n = len(eligible)
    row = {"paper_id": ident, "method": method, "status":"assessed", "claims": count, "mentioned": len(mentioned)/count,
           "substantive": len(substantive)/count, "supported_commentary": len(supported)/count,
           "eligible_units": n, "commentary_clusters": len(commentary),
           "S": sum(v["support"] == "supported" and v["scope_correct"] for v in eligible)/n if n else None,
           "T": sum(v["originally_substantiated"] for v in eligible)/n if n else None}
    for state in STATES:
        row[f"status_{state}"] = sum(v["support"] == state for v in eligible)/n if n else None
    for kind in TYPES:
        row[f"composition_{kind}"] = sum(t == kind for t in commentary.values())/len(commentary) if commentary else None
    for error in ERRORS:
        hits = sum(error in v["errors"] for v in eligible)
        row[f"error_{error}"] = int(hits > 0) if n else None
        row[f"error_units_{error}"] = hits
        row[f"error_per100_units_{error}"] = 100 * hits / n if n else None
    row.update({dimension: None for dimension in DIMENSIONS})
    row.update({d["dimension"]: d["score"] if d["applicable"] else None for d in quality})
    return row, unit_rows


def insight_counts(ident: str) -> tuple[dict[str, Counter], list[dict[str, Any]]]:
    data = read(OUTPUT / "data/insight_clusters" / f"{ident}.json")
    counts = {m: Counter() for m in METHODS}
    rows = []
    for cluster in data["clusters"]:
        methods = sorted({data["unit_mapping"][key]["method"] for key in cluster["unit_keys"]})
        for method in methods:
            counts[method][cluster["primary_type"]] += 1
        rows.append({"paper_id": ident, **cluster, "methods": methods,"available_methods":data.get("available_methods",list(METHODS))})
    return counts, rows


def report_summaries(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [c for c in frame.columns if c not in ("paper_id", "method","status")]
    summaries = {m: {c: interval(frame.loc[frame.method == m, c]) for c in columns} for m in METHODS}
    delta = {}
    for dimension in DIMENSIONS:
        wide = frame.pivot(index="paper_id", columns="method", values=dimension)
        delta[dimension] = interval(wide.fusion - wide.direct_a)
    return {"methods": summaries, "quality_delta": delta}


def preferences(allow_partial: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries, all_rows = {}, []
    dimensions = ("overall", "clarity_increment", "evidence_traceability", "knowledge_usefulness", "appropriate_limitations")
    for method in METHODS[:-1]:
        paired_rows, method_rows, matrix = [], [], np.zeros((3,3), dtype=int)
        for paper in roster():
            ident=paper["paper_id"]
            paths=[OUTPUT / "data/preferences" / method / f"{ident}_{o}.json" for o in ("AB", "BA")]
            if allow_partial and not all(path.exists() for path in paths):
                continue
            if any(missing_path("reports",ident,m).exists() for m in ("fusion",method)) or any(missing_path("preference",ident,f"{method}_{o}").exists() for o in ("AB","BA")):
                all_rows.append({"paper_id":ident,"comparator":method,"status":"not_assessed"})
                continue
            orders = [read(OUTPUT / "data/preferences" / method / f"{ident}_{o}.json") for o in ("AB", "BA")]
            decoded = [{d: ("tie" if r[d] == "tie" else "full" if r[d] == r["full_position"] else "other" if r[d] in ("A","B") else "cannot_judge") for d in dimensions} for r in orders]
            row = {"paper_id": paper["paper_id"], "comparator": method}
            for d in dimensions:
                values = [v[d] for v in decoded]
                if "cannot_judge" in values:
                    row[d] = None
                else:
                    row[d] = sum({"full":1,"tie":0.5,"other":0}[v] for v in values)/2
            if all(v["overall"] != "cannot_judge" for v in decoded):
                states = ("full", "tie", "other")
                matrix[states.index(decoded[0]["overall"]), states.index(decoded[1]["overall"])] += 1
                for state in states:
                    row[state] = sum(v["overall"] == state for v in decoded)/2
                paired_rows.append(row)
            all_rows.append({**row,"orders":decoded})
            method_rows.append(row)
        summaries[method] = {"n":len(paired_rows),"matrix":matrix.tolist(),
                             **{d: interval([r[d] for r in method_rows if r.get(d) is not None]) for d in (*dimensions,"full","tie","other")}}
        if paired_rows:
            wl=np.array([[r["full"],r["other"]] for r in paired_rows])
            indices=np.random.default_rng(20260917).integers(0,len(wl),(10000,len(wl)))
            draws=wl[indices].mean(axis=1)
            summaries[method]["bootstrap_xy"] = np.column_stack((draws[:,1]+.5*draws[:,0],np.sqrt(3)/2*draws[:,0]))[::20].tolist()
    return summaries, all_rows


def save_table(name: str, rows: list[dict[str, Any]]) -> None:
    write(OUTPUT / "data/tables" / f"{name}.json", rows)
    pd.DataFrame(rows).to_csv(OUTPUT / "data/tables" / f"{name}.csv",index=False)


def summarize(allow_partial: bool = False) -> None:
    papers, units, clusters, overlap_papers = [], [], [], []
    for p in roster():
        ident=p["paper_id"]
        cluster_path=OUTPUT / "data/insight_clusters" / f"{ident}.json"
        if not cluster_path.exists() and (allow_partial or missing_path("clusters",ident).exists()):
            counts,cluster_rows,available=None,[],[]
        else:
            counts,cluster_rows=insight_counts(ident)
            available=read(cluster_path).get("available_methods",METHODS)
        if {"fusion","direct_a","graph"} <= set(available):
            overlap_papers.append(p["paper_id"])
        clusters.extend(cluster_rows)
        for m in METHODS:
            quality_path=OUTPUT / "data/quality" / m / f"{ident}.json"
            ready=((OUTPUT / "data/support" / m / f"{ident}.json").exists()
                   and (quality_path.exists() or missing_path("quality",ident,m).exists()))
            if allow_partial and not ready:
                row={"paper_id":ident,"method":m,"status":"not_assessed",**{d:None for d in DIMENSIONS}}
                if quality_path.exists():
                    row.update({d["dimension"]:d["score"] if d["applicable"] else None for d in read(quality_path)["dimensions"]})
                atomic=[]
            else:
                row,atomic=paper_metrics(ident,m)
            row.update({f"insights_{k}":counts[m][k] if counts is not None and m in available else None for k in INSIGHTS})
            papers.append(row)
            units.extend(atomic)
    frame = pd.DataFrame(papers)
    summary = report_summaries(frame)
    summary["interim"] = allow_partial
    summary["preferences"], preference_rows = preferences(allow_partial)
    summary["insight_overlap_n"] = len(overlap_papers)
    summary["micro_support"]={}
    for method in METHODS:
        eligible=[u for u in units if u["method"]==method and u["needs_verification"]]
        n=len(eligible)
        summary["micro_support"][method]={"n_units":n,"n_papers":len({u["paper_id"] for u in eligible}),
            "status_counts":dict(Counter(u["support"] for u in eligible)),
            "status_proportions":{s:sum(u["support"]==s for u in eligible)/n if n else None for s in STATES},
            "S":sum(u["support"]=="supported" and u["scope_correct"] for u in eligible)/n if n else None,
            "T":sum(u["originally_substantiated"] for u in eligible)/n if n else None,
            "errors_per100_units":{e:100*sum(e in u["errors"] for u in eligible)/n if n else None for e in ERRORS}}
    summary["insight_overlap"] = dict(Counter("|".join(m for m in ("fusion","direct_a","graph") if m in c["methods"]) for c in clusters if set(("fusion","direct_a","graph"))<=set(c["available_methods"]) and any(m in c["methods"] for m in ("fusion","direct_a","graph"))))
    for name, rows in (("paper_metrics",papers),("atomic_units",units),("insight_clusters",clusters),("preferences",preference_rows)):
        save_table(name,rows)
    write(OUTPUT / "data/aggregates.json",summary)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial",action="store_true",help="Summarize completed observations for an explicitly interim preview")
    args=parser.parse_args()
    summarize(args.partial)


if __name__ == "__main__":
    main()

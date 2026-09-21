"""Conditional published-review statistics; no inferred acceptance or persuasion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from .aggregate import interval, save_table
from .settings import OUTPUT, missing_path, read, roster, write

STANCES = ("recognized","limited","challenged","unresolved")
TONES = ("supportive","neutral","critical","mixed")
RESPONSE_TYPES = ("new_analyses","clarification","scope_narrowing","prior_art","mixed_unknown")


def reviewer_pairs(opinions: list[dict[str,Any]], ident: str, first_round_only: bool = True) -> list[dict[str,Any]]:
    rows=[]
    for dimension in ("novelty","evidence","scope"):
        rounds=sorted({o["round"] for o in opinions if o["dimension"]==dimension and o["identity_explicit"]})
        for round_number in rounds:
            reviewers=defaultdict(lambda:defaultdict(set))
            for o in opinions:
                if o["dimension"]==dimension and o["round"]==round_number and o["identity_explicit"]:
                    reviewers[o["reviewer_id"]][o["concern_cluster"]].add(o["stance"])
            pairs=[]
            for a,b in combinations(sorted(reviewers),2):
                aa,bb=reviewers[a],reviewers[b]
                common=set(aa)&set(bb); union=set(aa)|set(bb)
                definite=[c for c in common if len(aa[c])==len(bb[c])==1 and "unresolved" not in aa[c]|bb[c]]
                if not union or not definite:
                    continue
                pairs.append({"reviewer_a":a,"reviewer_b":b,"attention_overlap":len(common)/len(union),
                              "disagreement":sum(aa[c]!=bb[c] for c in definite)/len(definite),
                              "coassessed":len(definite),"unresolved_share":1-len(definite)/len(common)})
            if pairs:
                rows.append({"paper_id":ident,"dimension":dimension,"round":round_number,
                             "attention_overlap":float(np.mean([p["attention_overlap"] for p in pairs])),
                             "disagreement":float(np.mean([p["disagreement"] for p in pairs])),
                             "coassessed":sum(p["coassessed"] for p in pairs),"reviewer_pairs":pairs})
                if first_round_only:
                    break
    return rows


def ratio_interval(rows: list[dict[str,Any]]) -> dict[str,Any]:
    if not rows:
        return {"mean":None,"low":None,"high":None,"n":0,"n_papers":0}
    groups=defaultdict(list)
    for row in rows: groups[row["paper_id"]].append(row["resolution"]=="resolved")
    counts=np.array([[sum(v),len(v)] for v in groups.values()])
    if len(counts)<2:
        return {"mean":float(counts[:,0].sum()/counts[:,1].sum()),"low":None,"high":None,
                "n":len(rows),"n_papers":len(counts),"outcomes":dict(Counter(r["resolution"] for r in rows))}
    samples=np.random.default_rng(20260917).integers(0,len(counts),(10000,len(counts)))
    totals=counts[samples].sum(axis=1);ratios=totals[:,0]/totals[:,1]
    lo,hi=np.quantile(ratios,[.025,.975])
    return {"mean":float(counts[:,0].sum()/counts[:,1].sum()),"low":float(lo),"high":float(hi),
            "n":len(rows),"n_papers":len(counts),"outcomes":dict(Counter(r["resolution"] for r in rows))}


def summarize(allow_partial: bool = False) -> None:
    opinions_all,pairs_all,transitions_all,responses_all,unavailable=[],[],[],[],[]
    pairs_all_rounds=[]
    completed_papers=[]
    for paper in roster():
        ident=paper["paper_id"];folder=OUTPUT/"data/reviews"/ident
        if missing_path("reviews",ident).exists():
            unavailable.append(read(missing_path("reviews",ident)))
            continue
        if allow_partial and not (folder/"linked.json").exists():
            continue
        opinions=read(folder/"opinions.json");linked=read(folder/"linked.json")
        completed_papers.append(ident)
        assignments={a["opinion_id"]:a["concern_cluster"] for a in linked["assignments"]}
        for o in opinions: o.update(paper_id=ident,concern_cluster=assignments[o["opinion_id"]])
        opinions_all.extend(opinions);pairs_all.extend(reviewer_pairs(opinions,ident))
        pairs_all_rounds.extend(reviewer_pairs(opinions,ident,first_round_only=False))
        by_id={o["opinion_id"]:o for o in opinions}
        sections={s["section_id"]:s for s in read(folder/"sections.json")["sections"]}
        for trajectory in linked["trajectories"]:
            initial=by_id[trajectory["initial_opinion_id"]]
            row={"paper_id":ident,"initial":initial["stance"],"dimension":initial["dimension"],**trajectory}
            transitions_all.append(row)
            followup=sections.get(trajectory["comparable_later_review_section"])
            response=sections.get(trajectory["response_section"])
            eligible=bool(initial["explicit_concern"] and initial["identity_explicit"] and response and followup and trajectory["response_quote"]
                          and followup["identity_explicit"] and followup["reviewer_id"]==initial["reviewer_id"]
                          and followup["round_number"]>initial["round"] and followup["round_number"]>response["round_number"])
            responses_all.append({**row,"eligible":eligible})
    tone=np.zeros((4,4),int);seen=set();tone_by_paper=defaultdict(lambda:np.zeros((4,4),int))
    for o in opinions_all:
        key=(o["paper_id"],o["reviewer_id"],o["round"],o["concern_cluster"],o["dimension"])
        if o["dimension"]=="novelty" and key not in seen:
            cell=(TONES.index(o["tone"]),STANCES.index(o["stance"]))
            tone[cell]+=1;tone_by_paper[o["paper_id"]][cell]+=1;seen.add(key)
    transitions=Counter((t["initial"],t["last_outcome"]) for t in transitions_all)
    summary={"completed_papers":completed_papers,"interim":allow_partial,"unavailable":unavailable,"tone_matrix":tone.tolist(),"tone_n":int(tone.sum()),"n_pairs_papers":len({p["paper_id"] for p in pairs_all}),
             "transitions":[{"initial":a,"last":b,"count":n} for (a,b),n in transitions.items()],
             "transition_n":len(transitions_all),"response_observable":sum(r["eligible"] for r in responses_all),
             "responses":{kind:ratio_interval([r for r in responses_all if r["eligible"] and r["response_type"]==kind]) for kind in RESPONSE_TYPES}}
    summary["tone_macro_n_papers"]=len(tone_by_paper)
    summary["tone_paper_macro_share"]=(np.mean([m/m.sum() for m in tone_by_paper.values()],axis=0).tolist()
                                       if tone_by_paper else None)
    tone_rows=[{"paper_id":ident,"tone":t,"stance":s,"count":int(matrix[i,j]),"paper_share":float(matrix[i,j]/matrix.sum())}
               for ident,matrix in tone_by_paper.items() for i,t in enumerate(TONES) for j,s in enumerate(STANCES)]
    save_table("review_tone_paper_counts",tone_rows)
    for name,rows in (("review_opinions",opinions_all),("reviewer_pairs",pairs_all),("reviewer_pairs_all_rounds",pairs_all_rounds),("review_transitions",transitions_all),("response_resolution",responses_all)):
        save_table(name,rows)
    write(OUTPUT/"data/review_aggregates.json",summary)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial",action="store_true",help="Render-data preview from completed review histories only")
    args=parser.parse_args()
    summarize(args.partial)


if __name__=="__main__":
    main()

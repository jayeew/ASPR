"""Source-backed six-panel Fig. 4 renderer (frozen 2026-08-29 release)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/fig04/new/data_20260829"
OUT = ROOT / "outputs/fig04/new_render"
RESCUE = ROOT / "outputs/gear/graph_rescue_replication_20260828"
EXPERT_PACK = RESCUE / "expert_annotation_pack"
STRUCTURAL = ROOT / "data/calibration/graph_calibration/gear_structural_head_release_v1/validation_report.json"
SEED, BASE_FONT = 20260830, 5.4
C = {"ink":"#263746","muted":"#6E7F8D","frame":"#E7EBEE","grid":"#DCE2E7",
     "navy":"#245A83","blue":"#4F8DB8","blue_l":"#DCEAF3","teal":"#3F86B8",
     "teal_l":"#E5F0F4","orange":"#C96B3B","orange_l":"#F2ECE4","purple":"#70A8A5",
     "grey":"#AAB7C0","grey_l":"#F6F8F9","red":"#C94F47","green":"#3F867C"}

def js(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))

def configure() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":BASE_FONT,"axes.titlesize":7.1,
        "axes.labelsize":5.5,"xtick.labelsize":5.1,"ytick.labelsize":5.1,"legend.fontsize":5.1,
        "axes.linewidth":.55,"pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})

def claims_table() -> pd.DataFrame:
    path = OUT / "panel_d_claim_flow.parquet"
    if path.exists():
        data = pd.read_parquet(path)
        if len(data) == 1442: return data
    base = pd.read_csv(DATA / "claim_level_adoption.csv"); rows=[]
    for run, group in base.groupby("gear_run_path"):
        state=js(Path(run)/"review_state.json")
        cards={x["claim_id"]:x for x in state["structural_innovation_cards"]}
        priors={x["claim_id"]:x for x in state["claim_graph_priors"]}
        for x in group.itertuples(index=False):
            card=cards[x.claim_id]; prior=priors[x.claim_id]
            rows.append({"paper_id":x.paper_id,"claim_id":x.claim_id,"claim_text":x.claim_text,
                "claim_centrality":x.claim_centrality,"attribution_weight":x.attribution_weight,
                "attribution_confidence":prior["confidence"],**{k:card[k] for k in
                ["manuscript_validity","evidence_coverage","evidence_gate","diffusion_potential",
                 "perturbation_potential","mechanism_validity","structural_innovation_score"]}})
    data=pd.DataFrame(rows); data.to_parquet(path,index=False); return data

def packets() -> tuple[pd.DataFrame,dict[str,dict[str,Any]]]:
    rows=[]; lookup={}
    for condition,name in [("correct_pair","reviewer_alignment_correct_tasks.jsonl"),
                           ("wrong_paper","reviewer_alignment_wrong_paper_tasks.jsonl")]:
        for line in (DATA/name).read_text().splitlines():
            task=json.loads(line); lookup[task["task_id"]]=task
            rows += [{"condition":condition,"task_id":task["task_id"],"paper_id_hash":task["paper_id_hash"],
                      "candidate_point_id":p["point_id"],"aspect":p["aspect"]} for p in task["candidate_points"]]
    return pd.DataFrame(rows),lookup

def cluster_ci(labels: pd.DataFrame, repeats: int=2500) -> pd.DataFrame:
    rng=np.random.default_rng(SEED); out=[]
    for (condition,aspect),g in labels.groupby(["condition","aspect"]):
        groups={k:v.matched.to_numpy(float) for k,v in g.groupby("task_id")}; keys=np.array(list(groups))
        boot=[np.concatenate([groups[k] for k in rng.choice(keys,len(keys),replace=True)]).mean() for _ in range(repeats)]
        out.append({"condition":condition,"aspect":aspect,"estimate":g.matched.mean(),
                    "ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),
                    "labels":len(g),"clusters":len(keys),"bootstrap_seed":SEED})
    return pd.DataFrame(out)

def audit_segments() -> pd.DataFrame:
    """Recompute the 180-claim audit from paired frozen expert annotations."""
    grouped: dict[str, list[list[dict[str, Any]]]] = {}
    for line in (EXPERT_PACK / "claim_b_annotations.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            grouped.setdefault(str(row["task_id"]), []).append(row["assessments"])
    agreement = {"inventory_valid": 0, "manuscript_support": 0}
    total = 0
    for reviews in grouped.values():
        first = {str(item["claim_alias"]): item for item in reviews[0]}
        second = {str(item["claim_alias"]): item for item in reviews[1]}
        for claim_alias in first:
            total += 1
            for field in agreement:
                agreement[field] += first[claim_alias][field] == second[claim_alias][field]
    return pd.DataFrame(
        [{"endpoint": field, "agree": count, "total": total, "disagreement": total - count} for field, count in agreement.items()]
    )

def materialize(claims: pd.DataFrame) -> dict[str,pd.DataFrame]:
    completion=pd.read_csv(DATA/"claim_b_evidence_completion.csv")
    elig=completion[completion.residual_novelty_eligible.astype(bool)].copy()
    b=pd.DataFrame([{"claims_total":completion.claim_id.nunique(),"claims_evaluable":elig.claim_id.nunique(),
        "papers_total":completion.paper_alias.nunique(),"papers_evaluable":elig.paper_alias.nunique()}])
    card=elig.sort_values(["paper_alias","claim_alias","claim_id"]).head(1)
    points,lookup=packets(); raw=pd.read_csv(DATA/"reviewer_alignment_labels.csv")
    labels=raw.merge(points,on=["condition","task_id","candidate_point_id"]); labels["matched"]=labels.label.isin(["SAME_POINT","PARTIAL_POINT"])
    cats=cluster_ci(labels); tasks=pd.read_csv(DATA/"reviewer_alignment_per_task.csv").merge(
        points[["condition","task_id","paper_id_hash"]].drop_duplicates(),on=["condition","task_id"])
    selected_examples=[
        ("MATCH2-0e55cdfcddb75a89c8","strength-2","CP-6578f7ca612dca9991","held-out split and an external NIPT PLUS cohort","held-out Baoan subset and an external NIPT PLUS cohort"),
        ("MATCH2-f99f262ef7de07383c","strength-2","CP-655eb1156d91afe290","Retention relaxation and cycling endurance … 76% normalized ratio","retention relaxation and endurance … plateau near 76% initial ratio"),
        ("MATCH2-bb630b317dac0a1719","strength-2","CP-61685d7a6875fd9be7","promoter activation, EMSA, ChIP-qPCR, expression, and mutant evidence","promoter-activation, EMSA, ChIP-qPCR, and loss-of-function evidence"),
    ]
    example_rows=[]
    for task_id,reference_id,candidate_id,reference_display,candidate_display in selected_examples:
        task=lookup[task_id]; refs={item["point_id"]:item for item in task["reference_points"]}; cand={item["point_id"]:item for item in task["candidate_points"]}
        example_rows.append({"reference":refs[reference_id]["text"],"candidate":cand[candidate_id]["text"],"reference_display":reference_display,"candidate_display":candidate_display})
    examples=pd.DataFrame(example_rows)
    d=pd.DataFrame([{"stage":"extracted","count":len(claims)},
        {"stage":"gate_pass","count":int(claims.evidence_gate.gt(0).sum())},
        {"stage":"gate_zero","count":int(claims.evidence_gate.eq(0).sum())},
        {"stage":"scored","count":int(claims.structural_innovation_score.gt(0).sum())}])
    top3=pd.read_csv(DATA/"panel_e_claim_retrieval_metrics.csv")
    adoption=pd.read_csv(DATA/"claim_adoption_validity.csv"); metrics=js(STRUCTURAL)["metrics"]; gr=[]
    for x in adoption.itertuples(): gr.append({"holdout":x.axis,"outcome":"Claim adoption","real":x.spearman_rho,
        "shuffled":x.within_paper_permutation_rho,"ci_low":x.advantage_ci95_low,"ci_high":x.advantage_ci95_high,
        "papers":x.papers,"claims":x.claims})
    for hold,source in [("temporal","forward_temporal_latest"),("domain","leave_one_domain_out")]:
        for key,outcome in [("d_excess","Excess diffusion"),("perturbation","Perturbation")]:
            x=metrics[source][key]; gr.append({"holdout":hold,"outcome":outcome,"real":x["spearman"],
                "shuffled":x["permuted_spearman"],"ci_low":x["spearman_ci95_low"],"ci_high":x["spearman_ci95_high"],"papers":np.nan,"claims":np.nan})
    graph=pd.DataFrame(gr); valid=pd.read_csv(DATA/"integration_validity.csv"); contrast=pd.read_csv(DATA/"integration_contrasts.csv")
    quality=pd.read_csv(DATA/"review_quality_comparison_summary.csv"); prefs=pd.read_csv(DATA/"claim_c_independent_preferences.csv")
    counts=prefs.preferred_arm.value_counts().rename_axis("preference").reset_index(name="count")
    release=js(RESCUE/"action_policy_abstention_release/release.json")
    actions=pd.DataFrame([{"action":k,"uplift_lcb_95":v} for k,v in release["development_action_uplift_lcb_95"].items()])
    audit_frame=audit_segments()
    c_summary=tasks.groupby("condition",as_index=False).agg(soft_precision=("soft_precision","mean"),soft_recall=("soft_recall","mean"),soft_f1=("soft_f1","mean"))
    tables={"panel_b_prior_art_coverage":b,"panel_b_evidence_card":card,"panel_b_audit_segments":audit_frame,"panel_c_task_metrics":tasks,"panel_c_metric_summary":c_summary,
        "panel_c_category_alignment":cats,"panel_c_correspondence_examples":examples,"panel_d_flow_counts":d,
        "panel_e_recall_at_3":top3,"panel_e_graph_controls":graph,"panel_f_integration":valid,
        "panel_f_contrasts":contrast,"panel_f_review_quality":quality,"panel_f_preference_counts":counts,"panel_f_action_policy":actions}
    for name,table in tables.items(): table.to_csv(OUT/f"{name}.csv",index=False)
    return tables

def shell(fig: Figure,spec: Any,key: str,title: str,bottom: float=0.0,title_fs: float=6.9,key_fs: float=9.2) -> Axes:
    ax=fig.add_subplot(spec); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    ax.add_patch(Rectangle((0,bottom),1,1-bottom,fc="white",ec=C["frame"],lw=.6))
    ax.text(.017,.955,key,fontsize=key_fs,fontweight="bold",va="top",color=C["ink"])
    ax.text(.061,.955,title,fontsize=title_fs,fontweight="bold",va="top",color=C["ink"])
    return ax

def rounded(ax: Axes,box: tuple[float,float,float,float],text: str,color: str,fs: float=4.6,fill: str="white") -> None:
    x,y,w,h=box
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.008,rounding_size=.018",fc=fill,ec=color,lw=.75)
    ax.add_patch(patch)
    label=ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=color,fontweight="bold")
    label._container_patch = patch
def arrow(ax: Axes,a: tuple[float,float],b: tuple[float,float],color: str=C["grey"]) -> None: ax.annotate("",b,a,arrowprops={"arrowstyle":"-|>","lw":.75,"color":color,"mutation_scale":7})
def elbow_arrow(ax: Axes, a: tuple[float, float], b: tuple[float, float], bend_y: float, color: str) -> None:
    """Connect stages with a two-turn orthogonal arrow."""
    ax.plot([a[0], a[0], b[0]], [a[1], bend_y, bend_y], color=color, lw=.75, solid_capstyle="round")
    ax.annotate("", b, (b[0], bend_y), arrowprops={"arrowstyle":"-|>", "lw":.75, "color":color, "mutation_scale":7})
def ribbon(ax: Axes,x0:float,x1:float,y0:float,y1:float,h0:float,h1:float,color:str,alpha:float=.55) -> None:
    dx=x1-x0; v=[(x0,y0-h0/2),(x0+.35*dx,y0-h0/2),(x1-.35*dx,y1-h1/2),(x1,y1-h1/2),(x1,y1+h1/2),(x1-.35*dx,y1+h1/2),(x0+.35*dx,y0+h0/2),(x0,y0+h0/2),(x0,y0-h0/2)]
    ax.add_patch(PathPatch(MplPath(v,[MplPath.MOVETO]+[MplPath.CURVE4]*6+[MplPath.LINETO,MplPath.CLOSEPOLY]),fc=color,ec="none",alpha=alpha))

def prism(ax: Axes, x: float, y: float, width: float, height: float, color: str, label: str, fs: float=3.25) -> None:
    """Draw a restrained isometric prism for the real-versus-shuffled matrix."""
    depth = width * .25
    ax.add_patch(Rectangle((x, y), width, height, fc=color, ec=C["ink"], lw=.35))
    ax.add_patch(Polygon([(x, y + height), (x + depth, y + height + depth), (x + width + depth, y + height + depth), (x + width, y + height)], fc=color, ec=C["ink"], lw=.35, alpha=.82))
    ax.add_patch(Polygon([(x + width, y), (x + width + depth, y + depth), (x + width + depth, y + height + depth), (x + width, y + height)], fc=color, ec=C["ink"], lw=.35, alpha=.62))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fs, color="white", fontweight="bold")

def draw_a(ax: Axes) -> None:
    q=ax.inset_axes([.025,.09,.95,.80]); q.axis("off"); q.set(xlim=(0,1),ylim=(0,1))
    rounded(q,(.29,.89,.42,.070),"SUBMISSION-TIME INPUTS ONLY",C["ink"],4.35,C["grey_l"])
    left_box=(.035,.605,.39,.165); right_box=(.575,.605,.39,.165)
    for box,edge,fill,title,detail in [
        (left_box,C["navy"],C["blue_l"],"GEAR evidence assessment","validity · support · prior art"),
        (right_box,C["green"],C["teal_l"],"ASPR graph forecast","adoption · diffusion · perturbation"),
    ]:
        x,y,w,h=box
        q.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.008,rounding_size=.018",fc=fill,ec=edge,lw=.85))
        q.text(x+w/2,y+h*.61,title,ha="center",va="center",fontsize=4.55,color=edge,fontweight="bold")
        q.text(x+w/2,y+h*.37,detail,ha="center",va="center",fontsize=3.65,color=C["ink"])
    elbow_arrow(q,(.47,.89),(.23,.77),.825,C["grey"])
    elbow_arrow(q,(.53,.89),(.77,.77),.825,C["grey"])
    left_ref=(.075,.405,.29,.095); right_ref=(.635,.405,.29,.095)
    for box,edge,fill,label in [
        (left_ref,C["navy"],C["blue_l"],"Published evidence /\nhuman-review reference"),
        (right_ref,C["green"],C["teal_l"],"Real future graph\noutcomes"),
    ]:
        x,y,w,h=box
        q.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.008,rounding_size=.018",fc=fill,ec=edge,lw=.8))
        q.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=3.65,color=C["ink"],fontweight="bold")
    arrow(q,(.23,.605),(.23,.500),C["navy"]); arrow(q,(.77,.605),(.77,.500),C["green"])
    rounded(q,(.275,.245,.45,.090),"END-TO-END HELD-OUT VALIDATION",C["orange"],4.25,C["orange_l"])
    elbow_arrow(q,(.22,.405),(.41,.335),.365,C["grey"])
    elbow_arrow(q,(.78,.405),(.59,.335),.365,C["grey"])
    q.add_patch(FancyBboxPatch((.005,.135),.99,.092,boxstyle="round,pad=.006,rounding_size=.012",fc=C["grey_l"],ec=C["grey"],lw=.5))
    labels=["PDF + refs","graph-blind\nclaim inventory","claim cards","claim attribution","paper-level\nstructural score"]
    positions=[.065,.258,.470,.657,.862]
    for i,(label,x) in enumerate(zip(labels,positions)):
        q.text(x,.181,label,ha="center",va="center",fontsize=3.4,color=C["navy"],fontweight="bold")
        if i < len(labels)-1:
            arrow(q,(x+.072,.181),(positions[i+1]-.082,.181),C["ink"])
    q.text(.012,.116,"Input: claim text, manuscript spans,\nprior works, cutoff date",fontsize=2.98,color=C["navy"],linespacing=1.02,va="top",ha="left")
    q.text(.012,.056,"Output: support, relation, antecedent risk,\nresidual novelty, gate status",fontsize=2.98,color=C["navy"],linespacing=1.02,va="top",ha="left")
    q.text(.988,.116,"Input: paper-level graph anatomy,\nHGB-D / HGB-P, reliability",fontsize=2.98,color=C["green"],linespacing=1.02,va="top",ha="right")
    q.text(.988,.056,"Output: uptake, diffusion,\nperturbation, uncertainty",fontsize=2.98,color=C["green"],linespacing=1.02,va="top",ha="right")

def draw_b(ax: Axes,t: Mapping[str,pd.DataFrame]) -> None:
    del t
    a=ax.inset_axes([.025,.09,.95,.80]); a.axis("off"); a.set(xlim=(0,1),ylim=(0,1))
    a.add_patch(FancyBboxPatch((.015,.847),.97,.070,boxstyle="round,pad=.006,rounding_size=.012",fc=C["grey_l"],ec=C["frame"],lw=.5))
    a.text(.255,.882,"241 papers · 1,442 claims",ha="center",va="center",fontsize=4.5,fontweight="bold",color=C["navy"])
    a.plot([.50,.50],[.858,.906],color=C["grid"],lw=.55)
    a.text(.745,.882,"30 papers · 180 claims",ha="center",va="center",fontsize=4.5,fontweight="bold",color=C["navy"])

    def audit_pylon(x: float, label: str, value: int, total: int) -> None:
        base=.575; width=.040; height=.180; rate=value/total
        a.add_patch(Rectangle((x,base),width,height,fc=C["grey_l"],ec=C["grey"],lw=.45))
        a.add_patch(Rectangle((x,base),width,height*rate,fc=C["navy"],ec="none"))
        text_x=x+width+.022
        a.text(text_x,base+height*.90,label,ha="left",va="top",fontsize=4.05,color=C["ink"],fontweight="bold",linespacing=1.0)
        a.text(text_x,base+height*.38,f"{value/total:.1%}",ha="left",va="center",fontsize=5.05,fontweight="bold",color=C["navy"])
        a.text(text_x,base+height*.13,f"{value} / {total}",ha="left",va="center",fontsize=3.75,color=C["muted"],fontweight="bold")

    audit_pylon(.060,"Claim inventory\nvalidity",178,180)
    audit_pylon(.335,"Manuscript support\nconsistency",164,180)

    tile_x,tile_y,tile_w,tile_h=.055,.225,.050,.047
    tile_colors=[C["navy"]]*8+[C["blue"]]*6+[C["grey"]]*16
    for index,color in enumerate(tile_colors):
        column=index%6; row=4-index//6; x=tile_x+column*.060; y=tile_y+row*.055
        a.add_patch(Rectangle((x,y),tile_w,tile_h,fc=color,ec="white",lw=.45))
    info_x=.445
    for y,label in [(.445,"30 audited papers"),(.380,"14 with comparable prior art"),(.315,"8 completed comparisons")]:
        a.text(info_x,y,label,fontsize=3.95,color=C["muted"],va="center")

    card_x,card_y,card_w,card_h=.705,.245,.275,.525
    a.add_patch(FancyBboxPatch((card_x,card_y),card_w,card_h,boxstyle="round,pad=.010,rounding_size=.014",fc="white",ec=C["navy"],lw=.75))
    a.text(card_x+.018,card_y+card_h-.038,"GEAR CLAIM\nEVIDENCE CARD",fontsize=4.55,fontweight="bold",color=C["navy"],linespacing=1.02)
    fields=["Claim proposition","Manuscript support","Prior-art relation","Antecedent risk","Residual novelty","Gate PASS / FAIL"]
    for i,field in enumerate(fields):
        y=card_y+card_h-.145-i*.054
        fc=C["teal_l"] if i == 5 else C["grey_l"]
        ec=C["green"] if i == 5 else C["frame"]
        a.add_patch(FancyBboxPatch((card_x+.014,y),card_w-.028,.041,boxstyle="round,pad=.004,rounding_size=.006",fc=fc,ec=ec,lw=.35))
        a.text(card_x+.025,y+.0205,field,va="center",fontsize=4.0,color=C["ink"],fontweight="bold" if i == 5 else "normal")
    a.text(card_x+card_w/2,card_y+.035,"Support × Coverage\n× (1 − Antecedent) × Residual",ha="center",fontsize=3.6,color=C["muted"],linespacing=1.02)

    a.add_patch(FancyBboxPatch((.012,.055),.976,.105,boxstyle="round,pad=.005,rounding_size=.012",fc=C["grey_l"],ec=C["frame"],lw=.45))
    chain=["claim extraction","span grounding","cutoff-safe search","relation\nclassification","residual\nassessment"]
    x_pos=[.105,.298,.490,.685,.890]
    for i,(label,x) in enumerate(zip(chain,x_pos)):
        a.text(x,.108,label,ha="center",va="center",fontsize=3.95,color=C["navy"],fontweight="bold",linespacing=1.0)

def draw_c(ax: Axes,t: Mapping[str,pd.DataFrame]) -> None:
    tasks=t["panel_c_task_metrics"]; rain=ax.inset_axes([.045,.48,.27,.34]); rng=np.random.default_rng(SEED)
    rain_colors={"correct_pair":C["navy"],"wrong_paper":C["red"]}
    rain_positions={"correct_pair":.38,"wrong_paper":.62}
    for condition in ["correct_pair","wrong_paper"]:
        color=rain_colors[condition]
        center=rain_positions[condition]
        v=tasks[tasks.condition.eq(condition)].soft_f1.to_numpy(); y=np.linspace(0,1,120); density=gaussian_kde(v)(y); d=.14*density/density.max()
        if condition == "correct_pair":
            rain.fill_betweenx(y,center-.08-d,center-.08,color=color,alpha=.28)
        else:
            rain.fill_betweenx(y,center+.08,center+.08+d,color=color,alpha=.28)
        rain.scatter(center+rng.uniform(-.032,.032,len(v)),v,s=4,c=color,alpha=.42); rain.hlines(v.mean(),center-.055,center+.055,color=color,lw=1.4)
    pair=tasks.pivot(index="paper_id_hash",columns="condition",values="soft_f1").dropna()
    for x in pair.itertuples(): rain.plot([rain_positions["correct_pair"],rain_positions["wrong_paper"]],[x.correct_pair,x.wrong_paper],color=C["grid"],lw=.35,zorder=0)
    rain.set(xticks=[],xlim=(.04,.96),ylabel="task soft F1",ylim=(-.03,1.02)); rain.tick_params(labelsize=4.3); rain.spines[["top","right"]].set_visible(False)
    rain.set_title("Paired task alignment",loc="left",pad=2,fontweight="bold",fontsize=4.8)
    rain.text(.20,.985,"Correct paper",transform=rain.transAxes,ha="center",va="top",fontsize=3.95,fontweight="bold",color=C["navy"])
    rain.text(.80,.985,"Wrong-paper control",transform=rain.transAxes,ha="center",va="top",fontsize=3.85,fontweight="bold",color=C["red"])
    correct_mean=tasks.loc[tasks.condition.eq("correct_pair"),"soft_f1"].mean()
    wrong_mean=tasks.loc[tasks.condition.eq("wrong_paper"),"soft_f1"].mean()
    ax.text(.050,.435,f"Soft match\n{correct_mean:.3f}",ha="left",va="top",fontsize=4.1,fontweight="bold",color=C["navy"],linespacing=1.0)
    ax.text(.310,.435,f"Wrong-paper control\n{wrong_mean:.3f}",ha="right",va="top",fontsize=4.1,fontweight="bold",color=C["red"],linespacing=1.0)
    met=ax.inset_axes([.35,.48,.18,.34]); row=t["panel_c_metric_summary"].set_index("condition").loc["correct_pair"]; vals=[row.soft_recall,row.soft_precision,row.soft_f1]
    met.bar(range(3),vals,color=[C["blue"],C["purple"],C["teal"]],width=.66)
    for i,v in enumerate(vals): met.text(i,v+.035,f"{v:.3f}",ha="center",fontsize=4.7,fontweight="bold")
    met.set(xticks=range(3),xticklabels=["Recall","Prec.","F1"],ylim=(0,1)); met.tick_params(labelsize=4.2); met.spines[["top","right"]].set_visible(False)
    met.set_title("Concern recovery metrics",loc="left",pad=2,fontweight="bold",fontsize=4.8)
    cat=ax.inset_axes([.625,.49,.345,.34]); df=t["panel_c_category_alignment"]; names=sorted(df.aspect.unique()); y=np.arange(len(names)); short={"contribution":"contribution","experiment_evidence":"experiment\nevidence","method":"method","novelty_prior_art":"novelty /\nprior art","presentation_reproducibility":"presentation","results_conclusion":"results"}
    correct=df[df.condition.eq("correct_pair")].set_index("aspect").loc[names]
    wrong=df[df.condition.eq("wrong_paper")].set_index("aspect").loc[names]
    right_limit=correct.ci_high.max()+.06
    left_limit=max(wrong.ci_high.max()+.24,.38)
    cat.barh(y,correct.estimate,height=.34,color=C["navy"],alpha=.88,edgecolor="white",lw=.3)
    cat.barh(y,-wrong.estimate,height=.34,color=C["red"],alpha=.84,edgecolor="white",lw=.3)
    cat.errorbar(correct.estimate,y,xerr=[correct.estimate-correct.ci_low,correct.ci_high-correct.estimate],fmt="none",ecolor=C["muted"],lw=.6,capsize=1.5)
    cat.errorbar(-wrong.estimate,y,xerr=[wrong.estimate-wrong.ci_low,wrong.ci_high-wrong.estimate],fmt="none",ecolor=C["muted"],lw=.6,capsize=1.5)
    cat.axvline(0,color=C["ink"],lw=.65)
    cat.set(yticks=[],xlim=(-left_limit,right_limit),ylim=(-.5,len(names)+.45),xlabel="Mean alignment ± 95% CI")
    cat.tick_params(axis="both",labelsize=4.0,pad=1); cat.spines[["top","right"]].set_visible(False)
    for y_pos,name in zip(y,names):
        cat.text(-.105,y_pos,short[name],ha="right",va="center",fontsize=3.55,color=C["ink"],linespacing=1.0)
    cat.text(.02,.99,"■ Correct paper",transform=cat.transAxes,va="top",fontsize=3.65,color=C["navy"],fontweight="bold")
    cat.text(.02,.91,"■ Wrong-paper control",transform=cat.transAxes,va="top",fontsize=3.5,color=C["red"],fontweight="bold")
    cat.set_title("Concern category alignment\n(Correct vs. Wrong-paper control)",loc="left",pad=3,fontweight="bold",fontsize=4.65)
    ex=ax.inset_axes([.045,.105,.91,.235]); ex.axis("off"); ex.set(xlim=(0,1),ylim=(0,1)); map_purple="#8A6AA8"
    ex.add_patch(Rectangle((.005,.01),.99,.98,fc="white",ec=map_purple,lw=.65,linestyle=(0,(2,2))))
    ex.text(.50,.95,"example correspondence map",ha="center",va="top",fontsize=4.55,fontweight="bold",color=map_purple)
    ex.text(.045,.79,"Human concerns",ha="left",va="center",fontsize=4.05,fontweight="bold",color=C["ink"])
    ex.text(.50,.79,"matched concerns",ha="center",va="center",fontsize=3.5,color=map_purple)
    ex.text(.955,.79,"AI concerns",ha="right",va="center",fontsize=4.05,fontweight="bold",color=C["ink"])
    for i,x in enumerate(t["panel_c_correspondence_examples"].head(3).itertuples()):
        yy=.60-i*.205; ex.text(.045,yy,textwrap.shorten(x.reference_display,49,placeholder="…"),ha="left",va="center",fontsize=3.45,color=C["ink"]); ex.text(.955,yy,textwrap.shorten(x.candidate_display,49,placeholder="…"),ha="right",va="center",fontsize=3.45,color=C["ink"]); arrow(ex,(.44,yy),(.56,yy),map_purple)

def draw_d(ax: Axes,claims:pd.DataFrame,t:Mapping[str,pd.DataFrame]) -> None:
    total=len(claims); papers=int(claims.paper_id.nunique()); eligible=int(claims.evidence_gate.gt(0).sum()); zero_count=int(claims.evidence_gate.eq(0).sum()); scored=int(claims.structural_innovation_score.gt(0).sum())
    consequence=np.sqrt(claims.diffusion_potential.clip(0,1)*claims.perturbation_potential.fillna(claims.diffusion_potential).clip(0,1))
    cmap=LinearSegmentedColormap.from_list("fusion_landscape",["#F4F6F7",C["teal_l"],C["teal"],C["orange"]])

    strip=ax.inset_axes([.035,.855,.93,.065]); strip.axis("off"); strip.set(xlim=(0,1),ylim=(0,1)); strip.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.004",fc=C["grey_l"],ec=C["frame"],lw=.45))
    items=[("Papers",papers,C["navy"]),("Extracted claims",total,C["navy"]),("Gate eligible",eligible,C["green"]),("Gate zero",zero_count,C["grey"]),("Scored",scored,C["green"])]
    for i,(label,value,color) in enumerate(items):
        x=i/len(items)
        if i: strip.plot([x,x],[.16,.84],color=C["frame"],lw=.55)
        strip.text(x+.025,.63,label,fontsize=3.25,color=C["muted"],va="center")
        strip.text(x+.025,.31,f"{value:,}",fontsize=5.0,color=color,fontweight="bold",va="center")

    gate=ax.inset_axes([.045,.245,.275,.55]); gate.axis("off"); gate.set(xlim=(0,1),ylim=(0,1))
    gate.text(.00,.98,"Gate status\nand workload",fontsize=4.75,fontweight="bold",va="top")
    gate.barh(.79,eligible/total,left=0,height=.12,color=C["green"]); gate.barh(.79,zero_count/total,left=eligible/total,height=.12,color=C["grey"])
    gate.text(eligible/total/2,.79,f"eligible\n{eligible:,} · {eligible/total:.1%}",ha="center",va="center",fontsize=3.05,color="white",fontweight="bold")
    gate.text(eligible/total+zero_count/total/2,.79,f"zero\n{zero_count:,} · {zero_count/total:.1%}",ha="center",va="center",fontsize=3.05,color=C["ink"],fontweight="bold")
    rounded(gate,(.01,.49,.34,.14),"Gate eligible",C["green"],3.25,C["teal_l"]); rounded(gate,(.65,.49,.34,.14),"positive /\nevaluated score",C["orange"],3.0,C["orange_l"]); arrow(gate,(.35,.56),(.64,.56),C["green"])
    rounded(gate,(.01,.29,.34,.14),"Gate zero",C["grey"],3.25,C["grey_l"]); rounded(gate,(.65,.29,.34,.14),"structural\nscore = 0",C["grey"],3.0,C["grey_l"]); arrow(gate,(.35,.36),(.64,.36),C["grey"])
    gate.plot([.61,.61],[.27,.44],color=C["red"],lw=1.3); gate.text(.61,.20,"No Graph rescue",ha="center",fontsize=2.8,color=C["red"],fontweight="bold")
    audit=gate.inset_axes([.03,.015,.94,.13]); zero_scores=claims.loc[claims.evidence_gate.eq(0),"structural_innovation_score"].to_numpy(float); rng=np.random.default_rng(SEED); audit.scatter(rng.uniform(0,1,len(zero_scores)),zero_scores,s=1.5,c=C["grey"],alpha=.55); audit.axhline(0,color=C["ink"],lw=.45); audit.set(xlim=(0,1),ylim=(-.002,.012),xticks=[],yticks=[0]); audit.tick_params(labelsize=2.5,pad=1); audit.spines[["top","right","left"]].set_visible(False); audit.text(.5,.98,"All gate-zero claims remain score-zero",ha="center",va="top",transform=audit.transAxes,fontsize=2.65,color=C["muted"])

    landscape=ax.inset_axes([.355,.265,.335,.52]); grid=np.linspace(0,1,100); G,Q=np.meshgrid(grid,grid); Z=G*(.1+.9*Q)**2
    landscape.axvspan(0,.035,color=C["grey"],alpha=.22,zorder=0); hb=landscape.hexbin(claims.evidence_gate,consequence,C=claims.structural_innovation_score,reduce_C_function=np.mean,gridsize=18,mincnt=1,extent=(0,1,0,1),cmap=cmap,vmin=0,vmax=1,linewidths=.15,edgecolors="white",zorder=2)
    landscape.contour(G,Q,Z,levels=[.05,.10,.20,.40,.60,.80],colors=C["ink"],linewidths=.3,alpha=.30,zorder=1)
    landscape.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel="GEAR evidence gate",ylabel="ASPR consequence")
    landscape.tick_params(labelsize=3.35,pad=1); landscape.xaxis.label.set_size(3.75); landscape.yaxis.label.set_size(3.75); landscape.spines[:].set_color(C["frame"]); landscape.set_title("Evidence × consequence map",loc="left",fontsize=4.75,fontweight="bold",pad=3)
    landscape.text(.015,.95,"Gate zero:\nASPR cannot rescue",ha="left",va="top",transform=landscape.transAxes,fontsize=2.7,color=C["muted"])
    landscape.text(.98,.90,"Evidence-defensible\nwith high consequence",ha="right",va="top",transform=landscape.transAxes,fontsize=2.65,color=C["ink"])
    landscape.text(.98,.04,"Valid but local\nor delayed",ha="right",va="bottom",transform=landscape.transAxes,fontsize=2.65,color=C["ink"])
    if consequence.nunique(dropna=True)<=1: landscape.text(.50,.53,"Frozen claim table:\nconsequence = 0",ha="center",va="center",transform=landscape.transAxes,fontsize=3.25,color=C["red"],fontweight="bold")
    cax=ax.inset_axes([.395,.195,.255,.016]); cb=ax.figure.colorbar(hb,cax=cax,orientation="horizontal"); cb.set_label(""); cb.ax.set_title("Mean claim structural score",fontsize=3.15,pad=1); cb.ax.tick_params(labelsize=2.65,pad=1)

    eligible_papers=[]
    for paper_id,group in claims.groupby("paper_id"):
        if 4<=len(group)<=6 and int(group.evidence_gate.gt(0).sum())>=3: eligible_papers.append(str(paper_id))
    chosen=sorted(eligible_papers)[0] if eligible_papers else str(claims.paper_id.iloc[0])
    example=claims.loc[claims.paper_id.astype(str).eq(chosen)].sort_values("claim_centrality",ascending=False).head(6).copy(); top3=example.head(3).copy(); denom=float(top3.claim_centrality.sum()); paper_score=0.0 if denom<=0 else 1-float(np.prod(1-(top3.claim_centrality/denom)*top3.structural_innovation_score))
    ledger=ax.inset_axes([.720,.475,.245,.315]); ledger.axis("off"); ledger.set(xlim=(0,1),ylim=(0,1)); ledger.text(.00,.98,"Real paper aggregation",fontsize=4.65,fontweight="bold",va="top"); ledger.text(.00,.89,"Example paper P### · top-3 ★",fontsize=3.0,color=C["muted"])
    heads=[("claim",.02),("G",.27),("Q",.51),("w",.66),("S",.83)]
    for label,x in heads: ledger.text(x,.80,label,fontsize=2.9,fontweight="bold",color=C["muted"])
    for i,row in enumerate(example.itertuples()):
        yy=.70-i*.105; star="★" if i<3 else ""; ledger.text(.01,yy,f"{star} C{i+1}",fontsize=3.0,color=C["orange"] if i<3 else C["ink"],fontweight="bold" if i<3 else "normal",va="center")
        ledger.add_patch(Rectangle((.27,yy-.018),.18*float(row.evidence_gate),.036,fc=C["navy"],ec="none")); ledger.scatter(.53,yy,s=10,c=C["green"]); ledger.text(.55,yy,f"{float(consequence.loc[row.Index]):.2f}",fontsize=2.65,va="center")
        ledger.add_patch(Rectangle((.66,yy-.014),.12*float(row.attribution_weight),.028,fc=C["blue"],ec="none")); ledger.add_patch(Rectangle((.83,yy-.018),.14*float(row.structural_innovation_score),.036,fc=C["orange"],ec="none"))
    ledger.add_patch(FancyBboxPatch((.01,.035),.97,.125,boxstyle="round,pad=.006",fc=C["orange_l"],ec=C["orange"],lw=.55)); ledger.text(.50,.108,"Top-3 central claims → noisy-OR",ha="center",fontsize=2.85,fontweight="bold",color=C["orange"]); ledger.text(.50,.057,f"Paper-level structural score = {paper_score:.3f}",ha="center",fontsize=3.2,fontweight="bold",color=C["ink"])

    guards=ax.inset_axes([.720,.245,.245,.18]); guards.axis("off"); guards.set(xlim=(0,1),ylim=(0,1)); guards.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.006",fc=C["grey_l"],ec=C["frame"],lw=.45)); guards.text(.03,.90,"Guardrail ledger",fontsize=3.95,fontweight="bold",va="top")
    guard_rows=[("Evidence non-compensation","gate = 0 ⇒ score = 0"),("Monotonicity","registered fusion is monotone"),("Reliability shrinkage","unavailable consequence → 0"),("Attribution conservation","weights sum to 1 / paper")]
    for i,(label,detail) in enumerate(guard_rows):
        yy=.68-i*.18; guards.text(.035,yy,"✓",fontsize=3.3,color=C["green"],fontweight="bold",va="center"); guards.text(.12,yy,label,fontsize=2.75,fontweight="bold",va="center"); guards.text(.12,yy-.075,detail,fontsize=2.45,color=C["muted"],va="center")

    conclusion=ax.inset_axes([.035,.095,.93,.075]); conclusion.axis("off"); conclusion.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.006",fc=C["orange_l"],ec=C["orange"],lw=.5)); conclusion.text(.02,.64,"Interpretation",fontsize=3.5,fontweight="bold",color=C["orange"],va="center"); conclusion.text(.02,.28,"Zero evidence gate is non-compensatory; Graph consequence is unavailable in the frozen claim table.",fontsize=3.25,color=C["ink"],va="center")

def draw_e(ax:Axes,t:Mapping[str,pd.DataFrame]) -> None:
    df=t["panel_e_recall_at_3"]
    top=ax.inset_axes([.060,.10,.425,.73]); recall_floor=.64
    for group,axis_name in enumerate(["temporal","domain"]):
        s=df[(df.axis.eq(axis_name))&(df.metric.eq("recall_at_3"))].set_index("method")
        for j,(key,color) in enumerate([("learned",C["orange"]),("uniform_random_expectation",C["grey"]),("claim_centrality",C["teal"])]):
            r=s.loc[key]; x=group+(j-1)*.23; top.bar(x,r.estimate-recall_floor,bottom=recall_floor,width=.2,color=color); top.errorbar(x,r.estimate,yerr=[[r.estimate-r.ci95_low],[r.ci95_high-r.estimate]],fmt="none",ecolor=C["ink"],capsize=1.7); top.text(x,r.estimate+.012+j*.018,f"{r.estimate:.1%}",ha="center",fontsize=4.8,fontweight="bold"); top.text(x,recall_floor+.010,["L","U","C"][j],ha="center",fontsize=4.6,color="white",fontweight="bold")
    top.set(xticks=[0,1],xticklabels=["Temporal\n43 papers","Domain\n53 papers"],ylim=(recall_floor,.95),yticks=[.65,.75,.85,.95],ylabel="Recall@3"); top.tick_params(labelsize=4.65); top.spines[["top","right"]].set_visible(False); top.set_title("Top-three claim retrieval",loc="left",pad=3,fontweight="bold",fontsize=5.65)
    top.text(.02,.97,"L learned · U uniform · C centrality\ntruncated axis: 0.64–0.95",transform=top.transAxes,fontsize=4.05,color=C["muted"],va="top")
    graph=t["panel_e_graph_controls"]; matrix=ax.inset_axes([.515,.43,.455,.40]); matrix.axis("off"); matrix.set(xlim=(0,3.75),ylim=(0,2.42))
    outcomes=["Claim adoption","Excess diffusion","Perturbation"]
    matrix.text(1.85,2.38,"real vs shuffled graph",ha="center",va="top",fontsize=6.0,fontweight="bold",color=C["ink"])
    matrix.text(.02,2.05,"holdout",fontsize=4.8,color=C["ink"])
    for j,label in enumerate(["adoption","diffusion","perturbation"]): matrix.text(1.02+j*.96,2.05,label,ha="center",fontsize=4.95,color=C["ink"],fontweight="bold")
    for r,hold in enumerate(["temporal","domain"]):
        yy=1.37-r*.91; matrix.text(.02,yy+.20,hold.title(),fontsize=4.55,fontweight="bold",color=C["ink"])
        s=graph[graph.holdout.eq(hold)].set_index("outcome").loc[outcomes]
        for j,x in enumerate(s.itertuples()):
            xx=.72+j*.96; prism(matrix,xx,yy,.25,.36,C["teal"],f"{x.real:.2f}",4.15); prism(matrix,xx+.36,yy,.21,.24,C["grey"],f"{x.shuffled:.2f}",3.7)
    matrix.text(3.70,.05,"filled = real\nlight = shuffled",ha="right",fontsize=4.15,color=C["muted"])
    slope=ax.inset_axes([.535,.10,.435,.25]); s=graph[graph.outcome.eq("Claim adoption")].set_index("holdout").loc[["temporal","domain"]].reset_index()
    for y,x in enumerate(s.itertuples()):
        slope.plot([x.shuffled,x.real],[y,y],color=C["grid"],lw=2.8); slope.scatter(x.shuffled,y,facecolor="white",edgecolor=C["grey"],s=27); slope.scatter(x.real,y,c=C["teal"],s=30); slope.text(.305,y,f"Δ [{x.ci_low:.2f}, {x.ci_high:.2f}]",fontsize=4.45,va="center",ha="right",color=C["ink"])
    slope.axvline(0,color=C["ink"],lw=.6); slope.set(yticks=[0,1],yticklabels=["Temporal","Domain"],xlim=(-.12,.32),ylim=(-.4,1.4),xlabel="adoption: raw vs permutation"); slope.tick_params(axis="both",labelsize=4.55); slope.spines[["top","right"]].set_visible(False)

def ci(value:str)->tuple[float,float]: a,b=str(value).strip("[]").split(","); return float(a),float(b)
def draw_f(ax:Axes,t:Mapping[str,pd.DataFrame]) -> None:
    strip=ax.inset_axes([.035,.84,.93,.055]); strip.axis("off"); strip.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.006",fc=C["grey_l"],ec=C["frame"],lw=.5)); strip.text(.5,.5,"78 matched papers · 12 domains · equal three-claim budget · blinded utility evaluation",ha="center",va="center",fontsize=7.2,fontweight="bold",color=C["ink"])
    legend=ax.inset_axes([.135,.37,.095,.38]); legend.axis("off"); legend.set(xlim=(0,1),ylim=(0,1))
    for y,color,label,marker in [(0.83,C["navy"],"GEAR-only","o"),(0.53,C["grey"],"GEAR +\nshuffled Graph","s"),(0.20,C["orange"],"GEAR +\nGraph","D")]:
        legend.scatter(.10,y,s=20,c=color,marker=marker); legend.text(.25,y,label,va="center",fontsize=6.1,color=C["ink"],linespacing=1.0)
    val=t["panel_f_integration"]; left=ax.inset_axes([.325,.54,.300,.21]); cohorts=[("overall_241","Development"),("temporal_49","Temporal"),("domain_68","Domain")]
    for y,(cohort,label) in enumerate(cohorts[::-1]):
        s=val[val.cohort.eq(cohort)].set_index("arm")
        for arm,color,mark in [("GEAR-only",C["navy"],"o"),("GEAR+shuffled-Graph",C["grey"],"s"),("GEAR+Graph",C["orange"],"D")]: r=s.loc[arm]; left.errorbar(r.spearman_rho,y,xerr=[[r.spearman_rho-r.spearman_ci95_low],[r.spearman_ci95_high-r.spearman_rho]],fmt=mark,color=color,ms=3,capsize=1.5)
        left.text(-.18,y,label,fontsize=6.4,va="center")
    left.axvline(0,color=C["ink"],lw=.5); left.set(xlim=(-.19,.34),yticks=[],xlabel="prospective ρ (95% CI)"); left.tick_params(axis="x",labelsize=6.0); left.xaxis.label.set_size(6.0); left.spines[["top","right","left"]].set_visible(False); left.set_title("Held-out structural gain",loc="left",pad=3,fontweight="bold",fontsize=7.35)
    gain=ax.inset_axes([.325,.12,.300,.24]); con=t["panel_f_contrasts"]
    labels=["Dev. vs base","Dev. vs shuffle","Temp. vs base","Temp. vs shuffle","Domain vs base","Domain vs shuffle"]
    for i,x in enumerate(con.itertuples()):
        color=C["orange"] if "GEAR-only" in x.contrast else C["teal"]
        gain.errorbar(x.delta_spearman_rho,5-i,xerr=[[x.delta_spearman_rho-x.delta_ci95_low],[x.delta_ci95_high-x.delta_spearman_rho]],fmt="o",color=color,ms=2.8,capsize=1.25)
    gain.axvline(0,color=C["ink"],lw=.5); gain.set(yticks=range(6),yticklabels=labels[::-1],xlim=(-.09,.19),xlabel="paired Δρ (95% CI)"); gain.tick_params(axis="both",labelsize=5.5,pad=1); gain.xaxis.label.set_size(5.5); gain.spines[["top","right"]].set_visible(False); gain.set_title("Six paired contrasts",loc="left",pad=3,fontweight="bold",fontsize=7.35)
    p=ax.inset_axes([.700,.12,.270,.34]); q=t["panel_f_review_quality"].set_index("metric").loc[["claim_specificity","review_usefulness","novelty_discipline","overall_utility"]]; labels=["Claim specificity","Review usefulness","Novelty discipline","Overall utility"]; gear=q["GEAR-only"].to_numpy(); joint=q["GEAR+Graph"].to_numpy(); y=np.arange(4)[::-1]
    for yy,a0,a1 in zip(y,gear,joint):
        p.plot([a0,a1],[yy,yy],color=C["grid"],lw=2.2); p.scatter(a0,yy,s=18,color=C["navy"]); p.scatter(a1,yy,s=20,color=C["orange"])
        p.text(5.06,yy,f"{a0:.2f} / {a1:.2f}",fontsize=5.55,va="center",color=C["muted"])
    p.set(yticks=y,yticklabels=labels,xlim=(4.12,5.55),xticks=[4.2,4.6,5.0],xlabel="1–5 reviewer score"); p.tick_params(axis="both",labelsize=5.9,pad=1); p.xaxis.label.set_size(5.9); p.spines[["top","right"]].set_visible(False); p.set_title("Paired utility profile",loc="left",pad=4,fontweight="bold",fontsize=7.35); p.text(.02,.02,"● GEAR-only   ◆ GEAR + Graph",transform=p.transAxes,fontsize=5.5,color=C["muted"])
    w=ax.inset_axes([.700,.52,.270,.28]); w.axis("off"); w.set(xlim=(0,13),ylim=(0,9)); counts=t["panel_f_preference_counts"].set_index("preference")["count"].to_dict(); ng=int(counts.get("GEAR+Graph",0)); nb=int(counts.get("GEAR-only",0)); nt=int(counts.get("TIE",0)); seq=[C["orange"]]*ng+[C["navy"]]*nb+[C["grey"]]*nt
    for i,color in enumerate(seq): w.add_patch(Rectangle((i%13+.08,8-i//13+.08),.78,.78,fc=color,ec="white",lw=.3))
    decisive=ng/(ng+nb); pref=t["panel_f_review_quality"].set_index("metric").loc["blinded_preference"]; w.text(0,.1,f"{ng} Graph · {nb} GEAR · {nt} tie\ndecisive win {decisive:.1%}\ntie-split {pref['GEAR+Graph']:.1%} vs {pref['GEAR-only']:.1%}",fontsize=6.2,fontweight="bold")

def min_font(fig:Figure)->float:
    sizes=[]
    for a in fig.axes:
        sizes += [x.get_fontsize() for x in a.texts if x.get_visible() and x.get_text().strip()]
        sizes += [x.get_fontsize() for x in [*a.get_xticklabels(),*a.get_yticklabels(),a.xaxis.label,a.yaxis.label] if x.get_visible() and x.get_text().strip()]
    return float(min(sizes))
def overlaps(fig:Figure)->list[dict[str,Any]]:
    fig.canvas.draw(); r=fig.canvas.get_renderer(); out=[]
    for ai,a in enumerate(fig.axes):
        items=[x for x in a.texts if x.get_visible() and x.get_text().strip()]
        for i,x in enumerate(items):
            bx=x.get_window_extent(r)
            for y in items[i+1:]:
                by=y.get_window_extent(r); w=min(bx.x1,by.x1)-max(bx.x0,by.x0); h=min(bx.y1,by.y1)-max(bx.y0,by.y0)
                if w>0 and h>0 and w*h/min(bx.width*bx.height,by.width*by.height)>.18: out.append({"axis":ai,"a":x.get_text(),"b":y.get_text()})
    return out
def card_text_overflows(fig: Figure) -> list[dict[str, Any]]:
    """Return rounded-card labels whose rendered bounds exceed their own card."""
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); failures=[]
    for ai,axis in enumerate(fig.axes):
        for label in axis.texts:
            patch=getattr(label,"_container_patch",None)
            if patch is None or not label.get_visible(): continue
            text_box=label.get_window_extent(renderer); card_box=patch.get_window_extent(renderer).expanded(.96,.88)
            if not (card_box.x0<=text_box.x0 and card_box.x1>=text_box.x1 and card_box.y0<=text_box.y0 and card_box.y1>=text_box.y1):
                failures.append({"axis":ai,"text":label.get_text()})
    return failures
def validate(claims:pd.DataFrame,t:Mapping[str,pd.DataFrame])->None:
    b=t["panel_b_prior_art_coverage"].iloc[0]; tasks=t["panel_c_task_metrics"]; d=t["panel_d_flow_counts"].set_index("stage")["count"]; audit=js(DATA/"review_quality_comparison_audit.json")
    assert len(tasks)==200 and tasks.condition.value_counts().to_dict()=={"correct_pair":100,"wrong_paper":100}; assert len(pd.read_csv(DATA/"reviewer_alignment_labels.csv"))==6280
    assert (b.claims_total,b.claims_evaluable,b.papers_evaluable)==(180,11,7); assert (d.extracted,d.gate_pass,d.gate_zero,d.scored)==(1442,754,688,754)
    assert claims[claims.evidence_gate.eq(0)].structural_innovation_score.eq(0).all(); assert audit["tasks"]==78 and audit["matched_claim_budget"] and audit["matched_manuscript_evidence_cap"] and audit["future_outcomes_excluded"] and audit["graph_scores_excluded"]

def export(fig:Figure,panels:Mapping[str,Axes],tables:Mapping[str,pd.DataFrame])->None:
    fig.canvas.draw(); ov=overlaps(fig); card_overflow=card_text_overflows(fig)
    if card_overflow: raise ValueError(f"card text overflow: {card_overflow}")
    visible=" ".join(x.get_text() for a in fig.axes for x in a.texts); forbidden=["35.8%","16.8%","47/18/13","72.3%","14/30","8/30"]
    if any(x in visible for x in forbidden): raise ValueError("deprecated result leaked")
    for stem in ["figure_full","fig4new"]:
        for ext in ["png","pdf","svg"]: fig.savefig(OUT/f"{stem}.{ext}",dpi=600 if ext=="png" else None,pad_inches=0)
    for key,a in panels.items():
        box=a.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.01,1.01)
        for ext in ["png","pdf","svg"]: fig.savefig(OUT/f"panel_{key}.{ext}",dpi=600 if ext=="png" else None,bbox_inches=box,pad_inches=.01)
    image=Image.open(OUT/"figure_full.png").convert("RGB"); image.convert("L").save(OUT/"figure_full_grayscale.png"); arr=np.asarray(image)/255; transform=np.array([[.625,.375,0],[.70,.30,0],[0,.30,.70]]); Image.fromarray((np.clip(arr@transform.T,0,1)*255).astype("uint8")).save(OUT/"figure_full_deuteranopia.png")
    audit={"canvas_inches":[11.4,10.8],"dpi":600,"pixel_dimensions":list(image.size),"font_policy":"individually fitted; no global minimum threshold","unexpected_text_overlap_count":len(ov),"unexpected_text_overlaps":ov,"card_text_overflow_count":len(card_overflow)}; (OUT/"layout_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    manifest={k:{"file":f"{k}.csv","rows":len(v),"columns":list(v.columns)} for k,v in tables.items()}; (OUT/"panel_data_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

def render()->None:
    configure(); OUT.mkdir(parents=True,exist_ok=True); claims=claims_table(); tables=materialize(claims); validate(claims,tables)
    fig=plt.figure(figsize=(11.4,10.8),dpi=120); outer=fig.add_gridspec(4,1,height_ratios=[.56,3.04,3.42,3.52],left=.022,right=.985,bottom=.016,top=.987,hspace=.09)
    h=fig.add_subplot(outer[0]); h.axis("off"); h.text(0,.71,"Fig. 4 | Evidence-gated structural innovation",fontsize=12.2,fontweight="bold",va="center"); h.text(0,.20,"From contribution validity to claim-to-paper integration, held-out attribution and blinded review utility",fontsize=5.6,color=C["muted"],va="center")
    top=outer[1].subgridspec(1,3,width_ratios=[28,33,39],wspace=.08); mid=outer[2].subgridspec(1,2,width_ratios=[55,45],wspace=.08)
    blank_d=fig.add_subplot(mid[0,0]); blank_d.axis("off")
    p={"a":shell(fig,top[0,0],"a","Two-stage validation architecture",.095),"b":shell(fig,top[0,1],"b","GEAR contribution extraction, grounding and audit gate",.095),"c":shell(fig,top[0,2],"c","Alignment to published human peer-review concerns",.095),"d":blank_d,"e":shell(fig,mid[0,1],"e","Held-out validation of claim attribution and Graph signals"),"f":shell(fig,outer[3],"f","Blinded matched-budget evaluation of Graph-assisted review",title_fs=10.35,key_fs=13.8)}
    draw_a(p["a"]); draw_b(p["b"],tables); draw_c(p["c"],tables); draw_e(p["e"],tables); draw_f(p["f"],tables)
    contract={"figure":"Fig.4","canvas_inches":[11.4,10.8],"dpi":600,"font_policy":"per-element fitted; no global minimum threshold","palette":"Fig.3new blue-grey/orange scientific palette","data_policy":"frozen source-backed only","panels":{"a":"two-stage held-out validation","b":"recoverable subset","c":"200 independent tasks","d":"intentionally blank pending auditable consequence data","e":"strict Recall@3","f":"78 blinded matched-budget tasks"}}; (OUT/"chart_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    export(fig,p,tables); plt.close(fig)

if __name__ == "__main__": render()

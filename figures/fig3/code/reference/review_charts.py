"""Four observed-review-history views, with separate denominators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .charts import plot
from .drawing import INK, NAVY, component, save_panel, text, wrap
from .review_stats import RESPONSE_TYPES, STANCES, TONES

COLORS=("#329aaa","#edb363","#e98075","#9198a4","#bcc4ce","#dce1e8")


def attention(frame:pd.DataFrame) -> Any:
    def draw(fig:Any) -> None:
        ax=fig.add_axes([.18,.22,.76,.65])
        for dimension,color,marker in (("novelty","#8b2be2","o"),("evidence","#2379ed","s"),("scope","#ed8b39","^")):
            part=frame[frame.dimension==dimension] if not frame.empty else frame
            if not part.empty:
                ax.scatter(part.attention_overlap,part.disagreement,s=12+5*np.sqrt(part.coassessed),color=color,marker=marker,alpha=.45,label=dimension.title(),clip_on=False)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Attention overlap",ylabel="Definite-appraisal disagreement")
        ax.xaxis.label.set_fontsize(11);ax.yaxis.label.set_fontsize(11);ax.grid(alpha=.4)
        ax.legend(frameon=False,fontsize=9,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.04),handletextpad=.3,columnspacing=.7)
        ax.text(.98,-.25,f"n={frame.paper_id.nunique() if not frame.empty else 0} papers",transform=ax.transAxes,ha="right",fontsize=10)
    return plot("e1_attention_disagreement",500,394,"Attention overlap vs disagreement",draw)


def tone(summary:dict[str,Any]) -> Any:
    def draw(fig:Any) -> None:
        ax=fig.add_axes([.24,.23,.71,.63]);matrix=np.array(summary["tone_matrix"])
        maximum=max(matrix.max(),1)
        for i in range(4):
            for j in range(4):
                if matrix[i,j]:
                    ax.scatter(j,i,s=matrix[i,j]/maximum*370,color=("#77a3dd","#aeb6c6","#e4907d","#bf91be")[i],alpha=.8)
                    ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=9)
        ax.set(xticks=range(4),yticks=range(4),xticklabels=["Recognized","Limited","Challenged","Unresolved"],yticklabels=[t.title() for t in TONES],xlim=(-.5,3.5),ylim=(3.5,-.5))
        ax.tick_params(axis="x",labelsize=9,rotation=25);ax.tick_params(axis="y",labelsize=10);ax.grid(alpha=.35)
        ax.text(.5,1.10,f"n={summary['tone_n']} reviewer–concern–round records",transform=ax.transAxes,ha="center",fontsize=10)
    return plot("e2_tone_stance",500,394,"Novelty stance × textual tone",draw)


def evolution(summary:dict[str,Any]) -> Any:
    s=component("e3_opinion_evolution",500,394,"Within-reviewer opinion evolution")
    endpoints=(*STANCES,"not_reassessed","no_comparable_followup")
    rows=summary["transitions"];total=sum(r["count"] for r in rows)
    left={k:sum(r["count"] for r in rows if r["initial"]==k) for k in STANCES}
    right={k:sum(r["count"] for r in rows if r["last"]==k) for k in endpoints}
    scale=240/max(total,1);starts_l={};starts_r={};y=87
    for key in STANCES:
        starts_l[key]=y;y+=left[key]*scale+8
    y=87
    for key in endpoints:
        starts_r[key]=y;y+=right[key]*scale+5
    offset_l={k:0 for k in STANCES};offset_r={k:0 for k in endpoints}
    for row in rows:
        a,b,n=row["initial"],row["last"],row["count"]
        yl=starts_l[a]+offset_l[a];yr=starts_r[b]+offset_r[b];h=n*scale
        s.path(f"M123,{yl} C195,{yl} 235,{yr} 309,{yr} L309,{yr+h} C235,{yr+h} 195,{yl+h} 123,{yl+h} Z",COLORS[STANCES.index(a)],opacity=.34)
        offset_l[a]+=h;offset_r[b]+=h
    for i,key in enumerate(STANCES):
        y=starts_l[key];h=left[key]*scale;s.rect(110,y,13,h,COLORS[i]);label_y=100+i*73
        s.line(102,label_y-6,109,y+h/2,"#98a3b3",.7);text(s,98,label_y,key.title(),18,INK,anchor="end")
    for i,key in enumerate(endpoints):
        y=starts_r[key];h=right[key]*scale;s.rect(309,y,13,h,COLORS[i]);label_y=94+i*48
        s.line(324,y+h/2,333,label_y-5,"#98a3b3",.7)
        wrap(s,337,label_y,{"not_reassessed":"Not reassessed","no_comparable_followup":"No comparable follow-up"}.get(key,key.title()),150,18)
    text(s,20,65,"Initial",18,NAVY,True);text(s,305,65,"Last observed outcome",18,NAVY,True)
    text(s,13,380,f"n={total} initial concern trajectories",18)
    return s


def resolution(summary:dict[str,Any]) -> Any:
    def draw(fig:Any) -> None:
        ax=fig.add_axes([.38,.23,.57,.65])
        labels=("New analyses","Clarification","Scope narrowing","Prior-art comparison","Mixed / unknown")
        for i,key in enumerate(RESPONSE_TYPES):
            row=summary["responses"][key]
            if row["mean"] is not None:
                x=row["mean"]
                error=None if row["low"] is None else [[x-row["low"]],[row["high"]-x]]
                ax.errorbar(x,i,xerr=error,fmt="o",color="#269c9b",capsize=3,clip_on=False)
                ax.text(.98,i+.22,f"{row['n']} concerns / {row['n_papers']} papers",fontsize=8,ha="right",color=INK)
                outcomes=row.get("outcomes",{})
                counts=" / ".join(str(outcomes.get(k,0)) for k in ("partial","still_concerned","not_reassessed"))
                ax.text(.98,i+.42,f"Partial / persistent / unmentioned: {counts}",fontsize=7,ha="right",color=INK)
            else:
                ax.text(.05,i,"No observable eligible trajectory",fontsize=9,va="center")
        ax.set(xlim=(0,1),ylim=(4.7,-.5),yticks=range(5),yticklabels=labels,xlabel="Explicit resolution rate")
        ax.tick_params(axis="y",labelsize=10);ax.xaxis.label.set_fontsize(11);ax.grid(axis="x",alpha=.4)
        if any(row["n_papers"]==1 for row in summary["responses"].values()):
            fig.text(.98,.015,"Single-paper groups: interval not estimable",ha="right",fontsize=8)
    return plot("e4_response_resolution",500,394,"Concern resolution after response",draw)


def render_e(summary:dict[str,Any],frame:pd.DataFrame) -> None:
    save_panel("e","Reviewer agreement, tone and opinion evolution",[(attention(frame),12,56),(tone(summary),520,56),
               (evolution(summary),1028,56),(resolution(summary),1536,56)])

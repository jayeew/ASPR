"""Reference-layout scientific plots, always drawn from computed source tables."""

from __future__ import annotations

from collections import Counter
import io
import xml.etree.ElementTree as ET
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from figure_pipeline.fig1_reference.svg import Scene, el, font_path
from .drawing import INK, NAVY, component, save_panel, text, wrap
from .settings import COLORS, DIMENSIONS, LABELS, METHODS, OUTPUT, read

Q_LABELS = ("Contribution\nfidelity", "Historical\nincrement", "Knowledge\nrelation", "Scope &\nuncertainty", "Whole-paper\nsynthesis")
MARKERS = ("o", "s", "^", "o", "o", "o")


def style() -> None:
    for bold in (False, True):
        fontManager.addfont(str(font_path(bold)))
    plt.rcParams.update({"font.family":"Times New Roman", "font.size":14, "text.color":INK,
        "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":"#78839a",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,
        "svg.fonttype":"none","svg.hashsalt":"fig3","savefig.transparent":True,
        "grid.color":"#dce3ed","grid.linewidth":0.6,"axes.titleweight":"bold"})


def plot(ident: str, width: float, height: float, title: str, draw: Callable[[Any], None]) -> Scene:
    style()
    s = component(ident,width,height,title)
    fig = plt.figure(figsize=(width/100,(height-40)/100),dpi=100)
    draw(fig)
    stream = io.StringIO()
    fig.savefig(stream,format="svg",transparent=True)
    plt.close(fig)
    svg = ET.fromstring(stream.getvalue())
    svg.set("x","0");svg.set("y","40");svg.set("width",str(width));svg.set("height",str(height-40))
    id_map = {n.get("id"):f"{ident}__{n.get('id')}" for n in svg.iter() if n.get("id")}
    for node in svg.iter():
        for key,value in list(node.attrib.items()):
            if key == "id":
                node.set(key,id_map[value])
            else:
                for old,new in id_map.items():
                    value = value.replace(f"url(#{old})",f"url(#{new})")
                    if value == f"#{old}": value = f"#{new}"
                node.set(key,value)
    s.add(svg)
    return s


def legend(ax: Any, size: int = 11) -> None:
    ax.legend(frameon=False,fontsize=size,loc="upper left",bbox_to_anchor=(1.0,1.04),handlelength=1.2,labelspacing=.55)


def quality_profile(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.10,.23,.65,.70])
        for i,m in enumerate(METHODS):
            vals=[summary["methods"][m][d] for d in DIMENSIONS]
            y=np.array([r["mean"] for r in vals]); lo=np.array([r["low"] for r in vals]);hi=np.array([r["high"] for r in vals])
            ax.errorbar(range(5),y,yerr=[y-lo,hi-y],fmt=MARKERS[i]+"-",lw=1.2 if m!="fusion" else 1.8,
                        color=COLORS[m],capsize=2,markersize=4,label=f"{LABELS[i]} (n={vals[0]['n']})")
        ax.set(ylim=(0,3),yticks=(0,1,2,3),xticks=range(5),xticklabels=Q_LABELS,ylabel="Mean quality score (0–3)")
        ax.tick_params(axis="x",labelsize=11);ax.grid(alpha=.6);legend(ax)
    return plot("b1_quality_profile",720,365,"Five-dimensional quality profile",draw)


def cloud(ident: str, frame: pd.DataFrame, columns: list[str], labels: list[str], width: int, height: int, title: str, quality: bool = False) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.03,.11,.59 if width>=600 else .90,.86],projection="3d")
        for m in ("direct_a","fusion"):
            counts=Counter(tuple(r) for r in frame.loc[frame.method==m,columns].dropna().to_numpy())
            if counts:
                xyz=np.array(list(counts));sizes=np.array(list(counts.values()))
                ax.scatter(*xyz.T,s=8*sizes,c=COLORS[m],alpha=.30,depthshade=False)
        for i,m in enumerate(METHODS):
            center=frame.loc[frame.method==m,columns].mean().to_numpy()
            ax.scatter(*center,s=70,color=COLORS[m],marker=MARKERS[i],edgecolor="white",linewidth=.6,label=LABELS[i],depthshade=False)
        ax.set(xlabel=labels[0],ylabel=labels[1],zlabel=labels[2]);ax.view_init(elev=23,azim=-55)
        if width<600:
            ax.set_zlabel("")
            fig.text(.05,.94,"Vertical: "+labels[2].replace("\n"," "),fontsize=9)
            n=frame.dropna(subset=columns).paper_id.nunique()
            fig.text(.05,.90,f"n={n} papers; large markers: means",fontsize=9)
        ax.tick_params(labelsize=10,pad=0)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis): axis.label.set_fontsize(10)
        if quality: ax.set(xlim=(0,3),ylim=(0,3),zlim=(0,3),xticks=(0,1.5,3),yticks=(0,1.5,3),zticks=(0,1.5,3))
        else:
            for setter,column in ((ax.set_xlim,columns[0]),(ax.set_ylim,columns[1]),(ax.set_zlim,columns[2])):
                setter(0,max(1,float(frame[column].max())))
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.set_major_locator(MaxNLocator(nbins=5,integer=True))
        if width>=600:
            ax.legend(frameon=False,fontsize=10,loc="upper left",bbox_to_anchor=(1.23,.99),handletextpad=.3)
    return plot(ident,width,height,title,draw)


def forest(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.38,.22,.53,.70]);vals=[summary["quality_delta"][d] for d in DIMENSIONS]
        y=np.array([r["mean"] for r in vals]);lo=np.array([r["low"] for r in vals]);hi=np.array([r["high"] for r in vals])
        ax.errorbar(y,np.arange(5),xerr=[y-lo,hi-y],fmt="o",color=COLORS["fusion"],capsize=3)
        ax.axvline(0,color="#687386",lw=.8);ax.set(yticks=range(5),yticklabels=Q_LABELS,xlabel="Full − Direct-A (score units)")
        ax.invert_yaxis();ax.grid(axis="x",alpha=.6)
        margin=max(.15,(max(hi.max(),0)-min(lo.min(),0))*.15)
        ax.set_xlim(min(lo.min(),0)-margin,max(hi.max(),0)+margin)
        ns=[r["n"] for r in vals]
        ax.text(1,-.27,f"Paired papers: n={min(ns)}"+(f"–{max(ns)}" if min(ns)!=max(ns) else ""),transform=ax.transAxes,ha="right",fontsize=11)
    return plot("b3_quality_delta",620,365,"Paired difference: Full − Direct-A",draw)


def render_b(summary: dict[str, Any], frame: pd.DataFrame) -> None:
    save_panel("b","Source-grounded report quality across models",[(quality_profile(summary),12,56),
        (cloud("b2_quality_space",frame,list(DIMENSIONS[:3]),["Contribution\nfidelity","Historical\nincrement","Knowledge\nrelation"],660,365,"Quality space (observed scores)",True),740,56),
        (forest(summary),1408,56)])


def progression(summary: dict[str, Any]) -> Scene:
    s=component("c1_progression",625,365,"From mention to supported analysis")
    for i,label in enumerate(("Mentioned","Substantive","Supported")):
        text(s,215+i*147,63,label,20,NAVY,True,anchor="middle")
    for i,m in enumerate(METHODS):
        y=110+i*39
        if m=="fusion": s.rect(8,y-26,609,36,"#f0e8ff",radius=4)
        text(s,15,y,LABELS[i],21,INK if i<3 else COLORS[m],True)
        for j,key in enumerate(("mentioned","substantive","supported_commentary")):
            v=summary["methods"][m][key]
            text(s,215+j*147,y,f"{100*v['mean']:.1f}%",23,INK,anchor="middle")
    text(s,15,352,"Paper means; denominator: shared claims per paper",18)
    return s


def composition(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.20,.20,.76,.65]);left=np.zeros(6)
        cols=("#42ad96","#428cec","#f2b16d","#df747f")
        kinds=("supported_scope","historical_increment","knowledge_relation","appropriate_limitation")
        for j,k in enumerate(kinds):
            vals=np.array([summary["methods"][m]["composition_"+k]["mean"] or 0 for m in METHODS])*100
            ax.barh(range(6),vals,left=left,height=.72,color=cols[j],label=("Scope","Increment","Relation","Limitation")[j]);left+=vals
        ax.set(xlim=(0,100),yticks=range(6),yticklabels=LABELS,xlabel="Composition of supported commentary (%)")
        ax.invert_yaxis();ax.tick_params(axis="y",labelsize=11)
        ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=4,frameon=False,fontsize=9,handlelength=1,columnspacing=.6)
        for i,m in enumerate(METHODS):
            n=summary["methods"][m]["composition_supported_scope"]["n"]
            total=summary["methods"][m]["supported_commentary"]["n"]
            ax.text(99,i,f"n={n}/{total}",ha="right",va="center",fontsize=8,color=INK)
    return plot("c2_composition",805,365,"Composition of source-supported commentary",draw)


def support_scatter(frame: pd.DataFrame) -> Scene:
    def draw(fig: Any) -> None:
        from scipy.stats import gaussian_kde
        ax=fig.add_axes([.14,.25,.64,.68]);ax.fill_between([0,1],[0,0],[0,1],color="#f0f1f5")
        ax.plot([0,1],[0,1],"--",color="#9aa3b4",lw=.8)
        contour_groups=[]
        for m in ("direct_a","fusion"):
            points=frame.loc[frame.method==m,["T","S"]].dropna().to_numpy()
            counts=Counter(tuple(r) for r in points)
            if counts:
                xy=np.array(list(counts));ax.scatter(*xy.T,s=8*np.array(list(counts.values())),color=COLORS[m],alpha=.3,clip_on=False)
            inside=points[(points[:,0]>0)&(points[:,1]<1)&(points[:,0]<points[:,1])]
            if len(inside)>=20 and np.linalg.matrix_rank(np.cov(inside.T))==2:
                xx,yy=np.meshgrid(np.linspace(0,1,81),np.linspace(0,1,81))
                density=gaussian_kde(inside.T,bw_method="scott")(np.vstack((xx.ravel(),yy.ravel()))).reshape(xx.shape)
                mask=xx<=yy;density[~mask]=0
                ordered=np.sort(density[mask])[::-1];mass=np.cumsum(ordered)/ordered.sum()
                levels=sorted(set(ordered[min(np.searchsorted(mass,q),len(ordered)-1)] for q in (.5,.8)))
                ax.contour(xx,yy,np.ma.masked_where(~mask,density),levels=levels,colors=[COLORS[m]],linewidths=.7,alpha=.6)
                contour_groups.append(f"{'Full' if m=='fusion' else 'Direct-A'}: {len(inside)}/{len(points)} interior")
        for i,m in enumerate(METHODS):
            vals=frame.loc[frame.method==m,["T","S"]].mean()
            n=len(frame.loc[frame.method==m,["T","S"]].dropna())
            ax.scatter(vals["T"],vals["S"],s=55,color=COLORS[m],edgecolor="white",label=f"{LABELS[i]} (n={n})")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Original substantiation (T)",ylabel="Independent support (S)")
        ax.grid(alpha=.35);legend(ax,10)
        fig.text(.03,.015,"50/80% interior-mass contours; boundary points retained" if contour_groups else "Observed count bubbles; insufficient interior variation for contours",fontsize=10)
    return plot("d1_support_traceability",730,365,"Independent support vs original substantiation",draw)


def support_composition(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.13,.25,.82,.60]);bottom=np.zeros(6)
        for state,color,label in (("supported","#56b986","Supported"),("partly_supported","#79b8e9","Partial"),("not_verifiable","#f6b971","Not verifiable"),("contradicted","#ec6d70","Contradicted")):
            vals=np.array([summary["methods"][m]["status_"+state]["mean"] for m in METHODS])
            ax.bar(range(6),vals,bottom=bottom,color=color,width=.70,label=label);bottom+=vals
        labels=[f"{label}\nn={summary['methods'][m]['status_supported']['n']}" for label,m in zip(("Direct-A","Direct-B","Direct-C","GEAR","Graph","Full"),METHODS)]
        ax.set(ylim=(0,1),xticks=range(6),xticklabels=labels,ylabel="Paper-mean proportion")
        ax.tick_params(axis="x",labelsize=10);ax.legend(loc="lower center",bbox_to_anchor=(.5,1.03),ncol=2,frameon=False,fontsize=10)
    return plot("d2_support_composition",690,365,"Support status (fully assessed reports)",draw)


def reliability(summary: dict[str, Any]) -> Scene:
    s=component("d3_reliability_issues",590,365,"Material reliability issues")
    text(s,18,67,"% of fully assessed papers with ≥1 instance",19)
    text(s,350,107,"Direct-A",20,NAVY,True);text(s,470,107,"Full",20,NAVY,True)
    text(s,350,127,f"n={summary['methods']['direct_a']['error_unsupported_definitive']['n']}",16)
    text(s,470,127,f"n={summary['methods']['fusion']['error_unsupported_definitive']['n']}",16)
    labels=("Unsupported definitive claims","False antecedence / firstness","Semantic-to-causal overreach","Omitted material scope limits")
    keys=("unsupported_definitive","false_antecedence","semantic_causal","omitted_scope")
    for i,(label,key) in enumerate(zip(labels,keys)):
        y=170+i*47;s.line(13,y-30,577,y-30,"#d6e2ee",.8);wrap(s,18,y,label,310,20)
        for j,m in enumerate(("direct_a","fusion")):
            v=summary["methods"][m]["error_"+key]
            text(s,390+j*120,y,f"{v['mean']*100:.1f}%",23,INK,anchor="middle")
    return s


def render_d(summary: dict[str, Any],frame: pd.DataFrame) -> None:
    save_panel("d","Evidence reliability beyond reviewer agreement",[(support_scatter(frame),12,56),(support_composition(summary),750,56),(reliability(summary),1448,56)])


def ternary(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.11,.30,.81,.62]);h=np.sqrt(3)/2
        ax.plot([0,.5,1,0],[0,h,0,0],color=NAVY,lw=1)
        for q in (.25,.5,.75):
            ax.plot([.5*q,1-.5*q],[h*q,h*q],color="#d4dfea",lw=.6)
        ax.plot([0,.75],[0,h/2],"--",color="#9cacc0",lw=.9)
        for i,m in enumerate(METHODS[:-1]):
            v=summary["preferences"][m];w=v["full"]["mean"];l=v["other"]["mean"]
            draws=np.asarray(v.get("bootstrap_xy",[]))
            if len(draws):
                ax.scatter(draws[:,0],draws[:,1],s=2,color=COLORS[m],alpha=.08,edgecolors="none")
            ax.scatter(l+.5*w,h*w,s=65,color=COLORS[m],marker=MARKERS[i],label=f"{LABELS[i]} (n={v['n']})",edgecolor="white",clip_on=False)
        ax.text(.5,h+.05,"Full wins",ha="center",fontsize=12);ax.text(0,-.055,"Tie",ha="center",fontsize=12)
        ax.text(1,-.055,"Comparator\nwins",ha="center",va="top",fontsize=11)
        ax.set(xlim=(-.12,1.12),ylim=(-.15,1.05));ax.axis("off")
        ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.03),ncol=1,fontsize=10)
        fig.text(.5,.025,"Faint points: paper bootstrap resamples",ha="center",fontsize=10)
    return plot("g1_ternary",365,653,"Full vs five comparators",draw)


def order_matrix(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.27,.27,.65,.62]);matrix=np.array(summary["preferences"]["direct_a"]["matrix"])
        ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,matrix/max(matrix.sum(),1),cmap="Blues",vmin=0,vmax=1,shading="flat")
        ax.set_xlim(-.5,2.5);ax.set_ylim(2.5,-.5);ax.set_aspect("equal")
        for i in range(3):
            for j in range(3): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=12,color="white" if matrix[i,j]/max(matrix.sum(),1)>.6 else INK)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=["Full","Tie","Other"],yticklabels=["Full","Tie","Other"],xlabel="BA outcome",ylabel="AB outcome")
        ax.tick_params(labelsize=10)
    return plot("g2_order_matrix",365,308,"Order agreement: Direct-A",draw)


def dimension_preference(summary: dict[str, Any]) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.49,.22,.46,.67]);stats=summary["preferences"]["direct_a"]
        keys=("clarity_increment","evidence_traceability","knowledge_usefulness","appropriate_limitations")
        y=np.array([stats[k]["mean"] for k in keys]);lo=np.array([stats[k]["low"] for k in keys]);hi=np.array([stats[k]["high"] for k in keys])
        ax.errorbar(y,range(4),xerr=[y-lo,hi-y],fmt="o",color=COLORS["fusion"],capsize=2,clip_on=False)
        ax.axvline(.5,color="#8996aa",ls="--",lw=.8)
        ax.set(xlim=(0,1),xticks=(0,.5,1),yticks=range(4),yticklabels=["Increment\nclarity","Evidence\ntraceability","Knowledge\nusefulness","Appropriate\nlimitations"],xlabel="Win + ½ tie")
        ax.tick_params(labelsize=10);ax.invert_yaxis()
    return plot("g3_preference_dimensions",365,338,"Preference vs Direct-A",draw)


def render_g(summary: dict[str, Any]) -> None:
    save_panel("g","Independent LLM preference",[(ternary(summary),12,56),(order_matrix(summary),385,56),(dimension_preference(summary),385,373)])


def projection(frame: pd.DataFrame) -> Scene:
    def draw(fig: Any) -> None:
        ax=fig.add_axes([.25,.24,.69,.66])
        columns=["insights_historical_increment","insights_cross_work_relation"]
        for m in ("direct_a","fusion"):
            counts=Counter(tuple(r) for r in frame.loc[frame.method==m,columns].dropna().to_numpy())
            if counts:
                xy=np.array(list(counts));ax.scatter(*xy.T,s=8*np.array(list(counts.values())),color=COLORS[m],alpha=.4,clip_on=False)
        ax.set(xlabel="Historical\nincrements",ylabel="Cross-work relations")
        ax.set_xlim(0,max(1,frame[columns[0]].max()));ax.set_ylim(0,max(1,frame[columns[1]].max()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
        ax.tick_params(labelsize=10);ax.xaxis.label.set_fontsize(11);ax.yaxis.label.set_fontsize(11);ax.grid(alpha=.4)
    return plot("f2_projection",242,427,"Same-data x–y view",draw)


def overlap(summary: dict[str,Any]) -> Scene:
    s=component("f3_insight_overlap",250,427,"Unique insight overlap")
    vals=summary["insight_overlap"]
    for x,y,color in ((125,155,COLORS["fusion"]),(85,222,COLORS["direct_a"]),(165,222,COLORS["graph"])):
        s.add(el("circle",cx=x,cy=y,r=72,fill=color,opacity=.22,stroke=color,stroke_width=1))
    mapping=(("fusion",125,117),("direct_a",62,240),("graph",190,240),("fusion|direct_a",89,172),
             ("fusion|graph",161,172),("direct_a|graph",125,264),("fusion|direct_a|graph",125,213))
    for key,x,y in mapping:
        text(s,x,y,str(vals.get(key,0)),23,INK,True,"middle")
    text(s,125,65,"Full",23,COLORS["fusion"],True,"middle")
    text(s,10,320,"Direct-A",20,INK);text(s,150,320,"Graph-only",20,COLORS["graph"])
    wrap(s,14,350,"Schematic areas; counts are within-paper insight clusters",220,18)
    if "insight_overlap_n" in summary:
        text(s,14,413,f"n={summary['insight_overlap_n']} shared papers",17)
    return s


def insight_types(summary: dict[str,Any]) -> Scene:
    def draw(fig:Any) -> None:
        ax=fig.add_axes([.49,.23,.46,.66])
        nf=summary["methods"]["fusion"]["insights_historical_increment"]["n"]
        nd=summary["methods"]["direct_a"]["insights_historical_increment"]["n"]
        fig.text(.97,.96,f"Full: n={nf}; Direct-A: n={nd}",ha="right",fontsize=9)
        kinds=("historical_increment","cross_work_relation","scope_correction","cross_contribution")
        for method,offset in (("direct_a",.12),("fusion",-.12)):
            vals=[summary["methods"][method]["insights_"+k] for k in kinds]
            x=np.array([v["mean"] for v in vals]);lo=np.array([v["low"] for v in vals]);hi=np.array([v["high"] for v in vals])
            ax.errorbar(x,np.arange(4)+offset,xerr=[x-lo,hi-x],fmt="o",color=COLORS[method],capsize=2,markersize=4)
        ax.set(yticks=range(4),yticklabels=["Historical\nincrements","Cross-work\nrelations","Scope\ncorrections","Cross-contribution\ninterpretations"],xlabel="Mean count\nper paper",xlim=(0,None))
        ax.invert_yaxis();ax.tick_params(labelsize=10);ax.xaxis.label.set_fontsize(11);ax.grid(axis="x",alpha=.5)
    return plot("f4_insight_types",357,427,"Supported insights by type",draw)


def commentary_example(example:dict[str,Any]) -> Scene:
    s=component("c3_commentary_example",580,365,"Analysis levels: an anchored example")
    for i,(label,key) in enumerate((("Mentioned contribution","mention"),("Substantive commentary","substantive"),("Source-supported commentary","supported"))):
        y=46+i*94
        s.rect(12,y,556,87,"#fcfdff","#cadfec",5,sw=.8)
        wrap(s,22,y+27,label,165,19,NAVY,True)
        wrap(s,211,y+25,example[key],342,19)
    text(s,16,342,example["alias"]+" · English rendering",17)
    text(s,16,360,"Original excerpts and source offsets accompany the data.",16)
    return s


def insight_chain(example:dict[str,Any]) -> Scene:
    s=component("f5_insight_chain",1236,215,"Example: source-supported analytical insight")
    widths=(275,280,350,245);x=14
    values=(("Target contribution",example["target"]),("Historical work",example["prior"]),
            ("Interpreted relation",example["relation"]),("Source anchors",example["sources"]))
    for i,(width,(title,value)) in enumerate(zip(widths,values)):
        s.rect(x,48,width,141,"#f3f7fd" if i!=2 else "#fff8ef",radius=5)
        text(s,x+10,77,title,21,NAVY,True);wrap(s,x+10,106,value,width-20,19)
        if i<3: s.arrow(x+width+2,120,x+width+23,120,INK,1.6)
        x+=width+22
    return s


def render_c(summary:dict[str,Any],examples:dict[str,Any]) -> None:
    save_panel("c","From extracted contributions to substantive commentary",[(progression(summary),12,56),(composition(summary),645,56),
                 (commentary_example(examples["commentary"]),1458,56)])


def render_f(summary:dict[str,Any],frame:pd.DataFrame,examples:dict[str,Any]) -> None:
    xyz=["insights_historical_increment","insights_cross_work_relation","insights_cross_contribution"]
    save_panel("f","Source-supported complementary knowledge insights",[
        (cloud("f1_insight_space",frame,xyz,["Historical\nincrements","Cross-work\nrelations","Cross-contribution\ninterpretations"],360,427,"Insight count space"),12,56),
        (projection(frame),380,56),(overlap(summary),630,56),(insight_types(summary),888,56),(insight_chain(examples["insight"]),12,494)])


def supplementary_projections(frame:pd.DataFrame) -> None:
    from figure_pipeline.fig1_reference.export import convert
    sets={"quality":list(DIMENSIONS[:3]),"insights":["insights_historical_increment","insights_cross_work_relation","insights_cross_contribution"]}
    for family,columns in sets.items():
        for a,b in ((0,1),(0,2),(1,2)):
            def draw(fig:Any) -> None:
                for i,method in enumerate(METHODS):
                    ax=fig.add_subplot(2,3,i+1)
                    values=frame.loc[frame.method==method,[columns[a],columns[b]]].dropna().to_numpy()
                    counts=Counter(tuple(row) for row in values)
                    if counts:
                        xy=np.array(list(counts));ax.scatter(*xy.T,s=8*np.array(list(counts.values())),color=COLORS[method],alpha=.5,marker=MARKERS[i])
                    highx=3 if family=="quality" else max(1,frame[columns[a]].max())
                    highy=3 if family=="quality" else max(1,frame[columns[b]].max())
                    ax.set(xlim=(0,highx),ylim=(0,highy),xlabel=columns[a].replace("_"," "),ylabel=columns[b].replace("_"," "),title=f"{LABELS[i]} · n={len(values)}")
                    ax.grid(alpha=.4)
                fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.43,hspace=.5)
            ident=f"{family}_projection_{'xyz'[a]}{'xyz'[b]}_all_methods"
            s=plot(ident,1500,950,"Observed paper-level values · count-proportional marker areas",draw)
            path=OUTPUT/"components"/f"{ident}.svg";s.save(path);convert(path,1.5)

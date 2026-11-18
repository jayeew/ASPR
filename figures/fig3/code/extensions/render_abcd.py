"""Independent ABCD figures, reading existing and append-only supplementary data."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from . import render_ab as ab
from .abcd_followup import DEST, PREP
from .cd_sources import OUTPUT, SOURCE, read, stage_path, write_json
from .run_cd_first import rows

COLORS=ab.COLORS
METHOD_NAMES=ab.NAMES


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ('png','pdf','svg'):
        path=DEST/f'{name}.{suffix}'
        fig.savefig(path,dpi=220,facecolor='white')
    plt.close(fig)


def paired_samples() -> list[dict]:
    source=pd.read_csv(SOURCE/'derived/paper_metrics.csv')
    source=source[source.metric.isin(['recall','precision'])]
    wide=source.pivot(index='paper_id',columns=['method','metric'],values='value')
    result=[]
    for method in ('gear','graph'):
        columns=[(m,t) for m in ('fusion',method) for t in ('recall','precision')]
        selected=wide[columns].dropna()
        delta=np.column_stack([selected['fusion',t]-selected[method,t] for t in ('recall','precision')])*100
        rng=np.random.default_rng(20260922)
        bootstrap=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
        result.append({'comparator':method,'n':len(delta),'mean':delta.mean(axis=0),'bootstrap':bootstrap,
                       'ci':np.quantile(bootstrap,[.025,.975],axis=0)})
    return result


def render_a() -> dict:
    data=paired_samples()
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.1),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.78,wspace=.13)
    fig.text(.035,.95,'A',fontsize=23,fontweight='bold')
    fig.text(.075,.95,'Paired gains in contribution identification',fontsize=18,fontweight='bold')
    fig.text(.075,.895,'10,000 joint paper bootstrap samples  |  Seed 20260922  |  Both metrics observed in both methods',fontsize=10.5)
    for ax,r in zip(axes,data):
        ab.clean(ax);ax.grid(color='#E9ECF0',lw=.65)
        ax.axhline(0,color='#637080',lw=1);ax.axvline(0,color='#637080',lw=1)
        ab.density_cloud(ax,r)
        ax.set(xlim=(-5,42),ylim=(-16,24),xticks=[0,10,20,30,40],yticks=[-10,0,10,20],xlabel='Recall gain (percentage points)')
        ax.set_title(f"Full − {METHOD_NAMES[r['comparator']]}  ·  n = {r['n']} papers",loc='left',fontsize=13,
                     color=COLORS[r['comparator']],pad=12,fontweight='bold')
        ax.text(.97,.95,f"Mean gain\nRecall {r['mean'][0]:+.1f} pp\nPrecision {r['mean'][1]:+.1f} pp",
            transform=ax.transAxes,ha='right',va='top',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':3})
    axes[0].set_ylabel('Precision gain (percentage points)')
    handles=[Line2D([],[],marker='D',color='none',markerfacecolor=COLORS['fusion'],markeredgecolor='#222733',label='Observed paired mean'),
             Line2D([],[],color=COLORS['fusion'],label='50% joint density region'),
             Line2D([],[],color=COLORS['fusion'],ls='--',label='95% joint density region')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.075),ncol=3,frameon=False,fontsize=10)
    fig.text(.08,.032,'Complete-pair subsets differ from single-metric summaries; missing precision is never filled with zero.',fontsize=10)
    save(fig,'A_paired_gains')
    stat={'seed':20260922,'bootstrap_repeats':10000,'comparisons':[]}
    for r in data:stat['comparisons'].append({'comparator':r['comparator'],'n_papers':r['n'],'mean_gain_pp':r['mean'].tolist(),'marginal_ci95_pp':r['ci'].tolist()})
    return stat


def render_b() -> dict:
    previous=ab.OUT
    ab.OUT=DEST
    try:return ab.render_b()
    finally:ab.OUT=previous


R_ORDER=['supported_difference','bounded_increment','substantially_covered']
P_ORDER=['positive_increment','limited_increment','substantially_known','explicit_abstention','not_addressed']
V_ORDER=['recognized','limited','challenged','mixed','explicitly_unresolved','no_same_scope_explicit_review','uncertain_correspondence_or_source','pending']
LABELS={'supported_difference':'Supported\ndifference','bounded_increment':'Bounded\nincrement','substantially_covered':'Already\ncovered',
 'positive_increment':'Positive\nincrement','limited_increment':'Limited\nincrement','substantially_known':'Substantially\nknown',
 'explicit_abstention':'Explicit\nabstention','not_addressed':'Not\naddressed','recognized':'Recognized','limited':'Limited',
 'challenged':'Challenged','mixed':'Mixed','explicitly_unresolved':'Explicitly\nunresolved','no_same_scope_explicit_review':'No explicit\nsame-scope opinion',
 'uncertain_correspondence_or_source':'Uncertain\nalignment/source','pending':'Pending\nsource/alignment'}


def ribbon(ax: plt.Axes,x0: float,x1: float,y0: float,y1: float,z0: float,z1: float,color: str) -> None:
    dx=(x1-x0)*.48
    vertices=[(x0,y0),(x0+dx,y0),(x1-dx,z0),(x1,z0),(x1,z1),(x1-dx,z1),(x0+dx,y1),(x0,y1),(x0,y0)]
    codes=[MplPath.MOVETO]+[MplPath.CURVE4]*3+[MplPath.LINETO]+[MplPath.CURVE4]*3+[MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=color,edgecolor='none',alpha=.25,zorder=1))


def alluvial(ax: plt.Axes,records: list[dict],method: str) -> None:
    stages=[('reference_state',R_ORDER),('prediction_state',P_ORDER),('review_state',V_ORDER)]
    positions=[]
    gap=.022;scale=(1-gap*(max(len(o) for _,o in stages)-1))/200
    xs=[.08,.50,.92];width=.026
    for index,(field,order) in enumerate(stages):
        count=Counter(r[field] for r in records);active=[v for v in order if count[v]]
        total=sum(count[v]*scale for v in active)+gap*(len(active)-1)
        cursor=(1+total)/2;pos={}
        for v in active:
            top=cursor;bottom=top-count[v]*scale;pos[v]=(bottom,top);cursor=bottom-gap
            ax.add_patch(plt.Rectangle((xs[index]-width/2,bottom),width,top-bottom,
                color=COLORS[method] if index==1 else '#7B8798',alpha=.9,zorder=3))
        label_y=[sum(pos[v])/2 for v in active]
        spacing=.085 if index==1 else .075
        for j in range(1,len(label_y)):
            label_y[j]=min(label_y[j],label_y[j-1]-spacing)
        if label_y and label_y[-1]<.025:
            label_y[-1]=.025
            for j in range(len(label_y)-2,-1,-1):
                label_y[j]=max(label_y[j],label_y[j+1]+spacing)
        for v,ly in zip(active,label_y):
            bottom,top=pos[v];center=(bottom+top)/2
            if abs(ly-center)>.01:
                ax.plot([xs[index],xs[index]-.015 if index==0 else xs[index]+.015],
                        [center,ly],color='#A8B1BF',lw=.6,zorder=4)
            if index==0:
                ax.text(xs[index]-.023,ly,LABELS[v]+f'  {count[v]}',ha='right',va='center',fontsize=8)
            elif index==2:
                ax.text(xs[index]+.022,ly,LABELS[v]+f'  {count[v]}',ha='left',va='center',fontsize=7.8)
            else:
                ax.text(xs[index],ly,LABELS[v]+f'\n{count[v]}',ha='center',va='center',fontsize=7.6,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.84,'pad':1.2},zorder=5)
        positions.append(pos)
    for index in range(2):
        left,right=stages[index][0],stages[index+1][0]
        count=Counter((r[left],r[right]) for r in records)
        lc={v:positions[index][v][0] for v in positions[index]};rc={v:positions[index+1][v][0] for v in positions[index+1]}
        for a in stages[index][1]:
            for b in stages[index+1][1]:
                n=count[a,b]
                if not n:continue
                h=n*scale;ribbon(ax,xs[index]+width/2,xs[index+1]-width/2,lc[a],lc[a]+h,rc[b],rc[b]+h,COLORS[method]);lc[a]+=h;rc[b]+=h
    ax.set(xlim=(-.20,1.22),ylim=(-.04,1.08));ax.axis('off')
    ax.set_title(METHOD_NAMES[method]+'  ·  200 contributions',color=COLORS[method],fontweight='bold',fontsize=13,pad=26)
    for x,label in zip(xs,['Historical reference','System report','Reviewer alignment']):ax.text(x,1.065,label,ha='center',fontsize=8.3)


def render_c() -> dict:
    joint=rows(OUTPUT/'first_stage/c_method_review_joint.jsonl')
    figure=plt.figure(figsize=(16.8,10.8))
    figure.text(.025,.96,'C',fontsize=23,fontweight='bold')
    figure.text(.065,.96,'Contribution-level evidence and reviewer alignment',fontsize=18,fontweight='bold')
    figure.text(.065,.918,'Same 200 core contributions in every method  |  Source-bound automatic matching  |  Silence is not a negative label',fontsize=10.5)
    gs=figure.add_gridspec(2,3,left=.045,right=.965,top=.84,bottom=.15,height_ratios=[2.05,1.0],hspace=.32,wspace=.27)
    for i,method in enumerate(('gear','graph','fusion')):
        values=[{**r,'review_state':r['review_summary']['state']} for r in joint if r['method']==method]
        alluvial(figure.add_subplot(gs[0,i]),values,method)
    summary=read(OUTPUT/'first_stage/summary.json')['C']['supported_recognized_identification']
    ax=figure.add_subplot(gs[1,0]);ab.clean(ax);ax.grid(axis='x',color='#E9ECF0')
    for y,method in enumerate(('gear','graph','fusion')):
        r=summary['methods'][method];value=(r['fraction'] or 0)*100
        ax.scatter(value,y,color=COLORS[method],s=110,marker='D',edgecolor='#263142',lw=.7,zorder=3)
        ax.text(value+3 if value<90 else value-3,y,f"{r['numerator']}/{r['denominator']}",ha='left' if value<90 else 'right',va='center',fontsize=10)
    ax.set(xlim=(-3,106),ylim=(2.6,-.6),yticks=range(3),yticklabels=['GEAR','Graph','Full'],xlabel='Effective identification (%)')
    ax.set_title(f"Supported + explicitly endorsed subset\nn = {summary['common_quote_ready_cores']} contributions; exploratory",loc='left',fontsize=11)
    coverage=read(DEST/'evaluation/coverage_statistics.json')
    ax=figure.add_subplot(gs[1,1:]);ax.set_title('Contribution-level coverage of matched reviewer reasons',loc='left',fontsize=11)
    states=['complete','partial','none','not_applicable','uncertain'];palette=['#3F738D','#91AFBF','#E1E6EC','#C4CCD5','#E9DFBE']
    for y,method in enumerate(('gear','graph','fusion')):
        values=coverage['contribution_states'][method];total=sum(values.values());left=0
        for state,color in zip(states,palette):
            n=values.get(state,0);width=n/total*100 if total else 0
            ax.barh(y,width,left=left,color=color,height=.52,edgecolor='white',lw=.8)
            if n:ax.text(left+width/2,y,str(n),ha='center',va='center',fontsize=10)
            left+=width
        ax.text(102,y,f'n={total}',va='center',fontsize=10)
    ax.set(xlim=(0,115),ylim=(2.6,-.6),yticks=range(3),yticklabels=['GEAR','Graph','Full'],xlabel='Share of contributions (%)',xticks=[0,25,50,75,100]);ab.clean(ax)
    ax.legend(handles=[Line2D([],[],marker='s',color='none',markerfacecolor=c,label=l) for c,l in zip(palette,['Complete','Partial','None','Reason applicability unresolved','Uncertain'])],
        loc='upper center',bbox_to_anchor=(.48,-.22),ncol=5,frameon=False,fontsize=8.5)
    figure.text(.05,.060,'Flows count each contribution once per method; mixed, unresolved, no explicit opinion and pending correspondence remain distinct.',fontsize=10)
    figure.text(.05,.027,'The 9-contribution subset is small and selective. Reviewer correspondence and reason coverage are automatic judgments, not new human annotations.',fontsize=10)
    save(figure,'C_contribution_alignment')
    return {'core_contributions':200,'methods':3,'identification':summary,'reason_coverage':coverage}


def case_data() -> dict:
    verified=rows(OUTPUT/'d_verification/verification_results.jsonl')
    selected=sorted([r for r in verified if r['verified_state'] in {'supported','supported_after_narrowing'}],key=lambda r:r['id'])[0]
    candidate=next(r for r in rows(OUTPUT/'tables/d_candidates.jsonl') if r['id']==selected['paper_id']+'/'+selected['cluster_id'])
    claim_ids=sorted({cid for u in candidate['units'] if u['method']=='graph' for cid in u['unit']['claim_ids']})
    report=read(stage_path(selected['paper_id'],'graph'))
    source=next(s for s in report['sources'] if s['source_id']=='GRAPH:'+claim_ids[0])
    facts=json.loads(source['passage'])
    return {'case':selected,'candidate':candidate,'facts':facts,'selection_rule':'First supported or narrowed case in lexical relation-ID order; no performance selection.'}


def arrow(ax: plt.Axes,p0: tuple,p1: tuple,color: str,style: str='-',label: str='') -> None:
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle='-|>',mutation_scale=13,color=color,lw=1.5,linestyle=style,connectionstyle='arc3,rad=0.0',zorder=2))
    if label:ax.text((p0[0]+p1[0])/2,(p0[1]+p1[1])/2+.02,label,ha='center',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','pad':1})


def box(ax: plt.Axes,x: float,y: float,text: str,color: str,width: float=.32,height: float=.16) -> None:
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=.015',facecolor='white',edgecolor=color,lw=1.4,zorder=3))
    ax.text(x,y,text,ha='center',va='center',fontsize=10,zorder=4)


def render_d() -> dict:
    summary=read(OUTPUT/'d_verification/summary.json');case=case_data();trace=read(DEST/'evaluation/trace_statistics.json')
    fig=plt.figure(figsize=(14.5,10.4));fig.text(.035,.955,'D',fontsize=23,fontweight='bold')
    fig.text(.075,.955,'Graph relation candidates: original evidence and report treatment',fontsize=17,fontweight='bold')
    fig.text(.075,.91,'261 Graph-valid / GEAR-ineligible clusters screened  →  76 relation candidates  →  52 assessed against existing original text',fontsize=10.5)
    ax=fig.add_axes([.035,.58,.27,.25]);states=['supported','supported_after_narrowing','insufficient_material'];counts=[summary['usable_verdicts'].get(s,0) for s in states]+[24]
    labels=['Supported','Supported after narrowing','Insufficient excerpts','No original-text packet'];colors=['#2C8293','#8DAABA','#C6CED8','#E4E7EC']
    ax.pie(counts,colors=colors,startangle=90,counterclock=False,wedgeprops={'width':.3,'edgecolor':'white','linewidth':2})
    ax.text(0,.10,'76',ha='center',fontsize=25,fontweight='bold');ax.text(0,-.13,'relation candidates',ha='center',fontsize=10)
    ax.legend(handles=[Line2D([],[],marker='s',color='none',markerfacecolor=c,label=f'{l}: {n}') for c,l,n in zip(colors,labels,counts)],loc='upper left',bbox_to_anchor=(.03,-.07),frameon=False,fontsize=9.5)
    ax.set_title('Evidence outcome\n0 explicitly contradicted',loc='left',fontsize=12)
    ax=fig.add_axes([.38,.55,.57,.25]);methods=['gear','fusion'];treatment=['same_scope','merged_same_scope','narrowed_or_corrected','explicitly_unresolved','partial','different_or_conflicting','not_located','uncertain']
    shades=['#365F7B','#7894AA','#8BAC99','#C0C6CD','#C9B98E','#AD8A83','#E5E8ED','#EADFBF']
    for y,m in enumerate(methods):
        values=trace['supported_relation_treatment'][m];total=sum(values.values());left=0
        for state,c in zip(treatment,shades):
            n=values.get(state,0);span=n/total*100 if total else 0
            ax.barh(y,span,left=left,height=.44,color=c,edgecolor='white')
            if n:ax.text(left+span/2,y,str(n),ha='center',va='center',fontsize=10)
            left+=span
    ax.set(xlim=(0,100),ylim=(1.6,-.6),yticks=[0,1],yticklabels=['GEAR','Full'],xlabel='Share of 44 supported or narrowed relations (%)');ab.clean(ax)
    ax.set_title('Same-relation treatment in existing reports',loc='left',fontsize=12)
    ax.legend(handles=[Line2D([],[],marker='s',color='none',markerfacecolor=c,label=l) for c,l in zip(shades,['Same scope','Merged','Corrected/narrowed','Explicitly unresolved','Partial','Conflicting','Not located','Uncertain'])],
        loc='upper left',bbox_to_anchor=(-.01,-.17),ncol=4,frameon=False,fontsize=8.5)
    ax=fig.add_axes([.035,.12,.92,.32]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(0,1.05,'Fixed-ID case  ·  '+case['case']['id']+'  ·  supported after narrowing',fontsize=11,fontweight='bold')
    case_states=trace['case_treatments']
    display={'same_scope':'same scope','merged_same_scope':'merged, same scope','narrowed_or_corrected':'narrowed/corrected',
             'explicitly_unresolved':'explicitly unresolved','partial':'partial','different_or_conflicting':'different/conflicting',
             'not_located':'not located','uncertain':'uncertain'}
    treatment_text='  |  '.join(METHOD_NAMES[m]+': '+display.get(case_states[case['case']['id']+'/'+m],'uncertain') for m in ('gear','fusion'))
    ax.text(0,.95,treatment_text,fontsize=9,color='#536174')
    box(ax,.14,.68,'X-ray crystallographic\nanalysis','#C18C3B',.23,.18)
    box(ax,.14,.23,'Semiempirical + DFT\ncalculations','#C18C3B',.23,.18)
    box(ax,.47,.68,'Targeted\ncomponent sequences','#C18C3B',.20,.18)
    box(ax,.47,.23,'Thermodynamic preference\nover alternative isomers','#C18C3B',.27,.18)
    arrow(ax,(.265,.68),(.36,.68),'#C18C3B',label='confirms')
    arrow(ax,(.265,.23),(.325,.23),'#C18C3B',label='reveals')
    ax.text(.02,.01,'HRMS is excluded from this supported relation: the supplied source excerpt does not establish its role.',fontsize=9)
    facts=case['facts'];neighbors=sorted(facts['neighbors'],key=lambda r:r['semantic_rank'])[:3]
    target=(.75,.48);ax.scatter(*target,s=180,c=COLORS['graph'],edgecolor='white',zorder=3);ax.text(.75,.36,'Target claim',ha='center',fontsize=9)
    positions=[(.95,.82),(.95,.48),(.95,.14)]
    for neighbor,pos in zip(neighbors,positions):
        ax.plot([target[0],pos[0]],[target[1],pos[1]],color=COLORS['graph'],ls=':',lw=1.5,zorder=1)
        ax.scatter(*pos,s=100,c='#9DADBF',edgecolor='white',zorder=3)
        ax.text(pos[0],pos[1]+.06,neighbor['claim_id'].split('::')[0].replace('s41467-','')+'\n'+neighbor['claim_id'].split('::')[-1],ha='center',fontsize=7.5)
        if neighbor.get('direct_citation'):arrow(ax,target,pos,'#657080',style='--')
    ax.text(.73,1.0,'Observed semantic neighborhood',fontsize=10)
    direct=sum(n.get('direct_citation',False) for n in neighbors)
    fig.text(.05,.094,'Technical-relation source: target manuscript, '+', '.join(case['case']['assessment']['evidence_ids'])+'. Full excerpts and provenance: D_case_evidence.json.',fontsize=8.8)
    fig.text(.05,.065,'Solid gold: original-text-supported technical relation. Dotted blue: semantic edge. Dashed gray: direct parent-paper citation.',fontsize=9.5)
    fig.text(.05,.033,f'{direct} direct citation edges in the three displayed neighbors; shared references are not drawn as direct citations. Relations do not automatically establish novelty.',fontsize=9.5)
    save(fig,'D_graph_relation_evidence')
    evidence_ids=set(case['case']['assessment']['evidence_ids'])
    write_json(DEST/'D_case_evidence.json',{'verification':case['case'],
        'original_evidence':[r for r in rows(OUTPUT/'tables/evidence_index.jsonl') if r['id'] in evidence_ids],
        'graph_structure':case['facts'],'selection_rule':case['selection_rule'],
        'report_traces':[r for r in rows(DEST/'evaluation/report_trace_records.jsonl') if r['relation_id']==case['case']['id']]})
    return {'evidence':summary,'case_id':case['case']['id'],'selection_rule':case['selection_rule'],'trace':trace}


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--groups',default='ABCD');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);ab.configure();statistics={}
    for group,function,name in [('A',render_a,'A_paired_gains'),('B',render_b,'B_information_intersections'),
                                ('C',render_c,'C_contribution_alignment'),('D',render_d,'D_graph_relation_evidence')]:
        if group not in args.groups:continue
        if (DEST/(name+'.png')).exists():continue
        statistics[group]=function()
    for group,values in statistics.items():
        path=DEST/f'{group}_statistics.json'
        if not path.exists():
            with path.open('x') as stream:json.dump(values,stream,ensure_ascii=False,indent=2)
    print('Rendered',list(statistics))


if __name__=='__main__':main()

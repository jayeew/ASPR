"""Render the reduced, source-backed six-panel Fig5 as static publication files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/fig5_six_panel'
OUT = DATA / 'figure'
BLUE, GOLD, RED, GRAY = '#286A8C', '#C19132', '#B8595C', '#A7AFB7'
COLORS = {'F': BLUE, 'E50': GOLD, 'NONCRITICAL': '#6B8894', 'CRITICAL': RED}
MARKERS = {'F': 'o', 'E50': 's', 'NONCRITICAL': '^', 'CRITICAL': 'D'}
LABELS = {'F': 'Full', 'E50': 'E50', 'NONCRITICAL': 'Noncritical', 'CRITICAL': 'Critical'}
ASPECTS = ['historical_verification', 'knowledge_position', 'joint_contribution', 'structural_resolution', 'citation_contact']


def read(panel: str, name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f'panel_{panel}' / f'{name}.csv')


def style(ax: plt.Axes, grid: str = 'x') -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#B6BDC3')
    ax.tick_params(length=3, color='#A0A7AD')
    ax.grid(axis=grid, color='#E8ECEF', linewidth=.65)
    ax.set_axisbelow(True)


def heading(fig: plt.Figure, rect: list[float], letter: str, title: str, subtitle: str) -> None:
    x, y, w, h = rect
    fig.text(x-.028, y+h+.042, letter, size=18, weight='bold', va='top')
    fig.text(x, y+h+.042, title, size=13, weight='bold', va='top')
    fig.text(x, y+h+.013, subtitle, size=9, color='#58616B', va='top')


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'text.color': '#263440', 'axes.labelcolor': '#263440', 'xtick.color': '#46535D', 'ytick.color': '#46535D'})
    fig = plt.figure(figsize=(18, 11.5), facecolor='white')
    rectangles = [[.062, .61, .255, .265], [.386, .61, .255, .265], [.713, .61, .255, .265],
                  [.062, .16, .255, .255], [.386, .16, .255, .255], [.713, .16, .255, .255]]
    fig.text(.034, .965, 'Fig. 5 | Reliability under evidence and configuration changes', size=20, weight='bold')
    fig.text(.034, .937, 'Reduced-scope initial study  •  existing reports and model-based evaluation', size=11, color='#65717B')

    # a: existing paper-bootstrap intervals, kept on their original metric scale.
    r = rectangles[0]
    heading(fig, r, 'a', 'Observed-domain differences', '95 papers • F − GEAR only • original Fig. 4 metric')
    ax = fig.add_axes(r)
    a = read('a', 'domain_summary')
    domains = [('life', 'Life sciences', BLUE, 'o'), ('physical_engineering', 'Physical / engineering', GOLD, 's'),
               ('medicine', 'Medicine', RED, '^'), ('earth_environment', 'Earth / environment', '#697C68', 'D')]
    for j, (domain, label, color, marker) in enumerate(domains):
        for i, aspect in enumerate(ASPECTS):
            row = a[(a.domain == domain) & (a.aspect == aspect)].iloc[0]
            y = i + (j-1.5)*.16
            ax.errorbar(row['mean']*100, y, xerr=np.array([[row['mean']-row.low], [row.high-row['mean']]])*100,
                        fmt=marker, ms=4, color=color, capsize=2, lw=1, label=label if i == 0 else None)
            ax.text(106, y, str(int(row['n'])), va='center', ha='center', fontsize=8, color=color)
    ax.set(yticks=range(5), yticklabels=['H', 'N', 'J', 'S', 'C'], ylim=(4.55, -.65), xlim=(-5, 111), xticks=[0,25,50,75,100])
    ax.axvline(0, color='#69727B', lw=.8)
    ax.text(106, -.52, 'n', ha='center', fontsize=9, style='italic')
    ax.set_xlabel('Δ grounded coverage (percentage points)')
    style(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(-.02,-.20), ncol=2, frameon=False, fontsize=8, handletextpad=.4, columnspacing=1)
    ax.text(0, -.39, 'Bars: paper-bootstrap 95% CI; n = applicable pairs.', transform=ax.transAxes, size=8, color='#65717B')

    # b: true paired endpoints; no interpolated degradation curve.
    r = rectangles[1]
    heading(fig, r, 'b', 'Original-text loss sensitivity', '5 papers • full versus 46–50% of independent works')
    b = read('b','paper_metrics')
    paired = read('b','paired_answerable')
    for i, (aspect, label) in enumerate(zip(ASPECTS[:2], ['H · History', 'N · Knowledge position'])):
        ax = fig.add_axes([r[0]+i*.14,r[1],.112,r[3]-.035])
        subset = b[b.aspect == aspect]
        for j,paper in enumerate(sorted(subset.paper_id.unique())):
            vals = subset[subset.paper_id == paper].set_index('condition').loc[['F','E50']]
            offset = (j-2)*.034
            ax.plot(np.array([0,1])+offset, vals.q_fixed*100, '-o', color=BLUE, alpha=.4, lw=1, ms=4)
        means = subset.groupby('condition').q_fixed.mean().reindex(['F','E50'])*100
        available = (subset.assign(ratio=subset.answerable_denominator/subset.fixed_denominator).groupby('condition').ratio.mean().reindex(['F','E50']))*100
        ax.plot([0,1],means,'-o',color=BLUE,lw=2.4,ms=6,zorder=5)
        ax.plot([0,1],available,'--D',color=GOLD,lw=1.5,ms=5,mfc='white',zorder=6)
        ax.set(xlim=(-.3,1.35),ylim=(-5,108),xticks=[0,1],xticklabels=['Full','E50'],yticks=[0,25,50,75,100])
        ax.set_title(label, size=10, pad=8)
        ax.set_ylabel('Fixed-task coverage (%)' if i == 0 else '')
        if i: ax.set_yticklabels([])
        style(ax,'y')
        val = paired[(paired.aspect==aspect)&paired.delta_q_answerable.notna()]
        ax.text(.5,-.24,f'Common-answerable Δ: {val.delta_q_answerable.mean()*100:+.0f} pp\n{len(val)}/5 papers; {int(val.common_answerable_denominator.sum())} parts',ha='center',transform=ax.transAxes,fontsize=8,color='#58616B')
    fig.legend(handles=[Line2D([0],[0],color=BLUE,marker='o',label='Correct coverage'),Line2D([0],[0],color=GOLD,ls='--',marker='D',mfc='white',label='Answerable fraction')],loc='center',bbox_to_anchor=(.513,.515),ncol=2,frameon=False,fontsize=8)

    # c: state matrix; preserve the incomplete response instead of forcing abstention.
    r = rectangles[2]
    heading(fig,r,'c','Critical-evidence dependence','5 prespecified targets • matched work-count deletion')
    ax = fig.add_axes([r[0]+.015,r[1]+.025,r[2]-.015,r[3]-.025])
    c = read('c','target_transitions')
    states = {'grounded_correct': (BLUE, 'Correct'), 'target_abstention': (GOLD, 'Withhold'), 'substantive_incomplete_or_ungrounded': (GRAY, 'Partial')}
    for i,row in c.iterrows():
        for j,condition in enumerate(['F','NONCRITICAL','CRITICAL']):
            color,label=states[row[condition+'_state']]
            ax.add_patch(plt.Rectangle((j-.46,i-.42),.92,.84,facecolor=color,alpha=.16,edgecolor=color,lw=1))
            ax.text(j,i,label,ha='center',va='center',color='#58616B' if color==GRAY else color,weight='bold',fontsize=10)
    ax.set(xlim=(-.5,2.5),ylim=(4.55,-.55),yticks=range(5),yticklabels=[x.replace('s41467-026-','') for x in c.paper_id],xticks=range(3),xticklabels=['Full','Noncritical','Critical'])
    ax.tick_params(length=0,pad=9)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(r[0],.566,'Noncritical: 0/5 harmful flips   |   Critical: 4/5 withhold',size=9,color='#46535D')
    fig.text(r[0],.543,'Partial = some facts answered; target comparison incomplete.',size=8,color='#65717B')

    # d: pooled fixed parts, with explicit denominators and every condition.
    r = rectangles[3]
    heading(fig,r,'d','Answering and withholding boundaries','H + N pooled • 104 fixed parts • counts shown within bars')
    d = read('d','decision_matrix')
    categories=[('grounded_correct','Grounded correct',BLUE),('target_abstention','Target withheld',GOLD),('substantive_incomplete_or_ungrounded','Partial / ungrounded',GRAY),('incorrect_or_unsupported','Error / unsupported',RED)]
    for i,answerability in enumerate(['answerable','unanswerable']):
        ax=fig.add_axes([r[0]+i*.14,r[1]+.025,.112,r[3]-.045])
        for y,condition in enumerate(['F','E50','NONCRITICAL','CRITICAL']):
            subset=d[(d.condition==condition)&(d.answerability==answerability)]
            n=int(subset.parts.sum());left=0
            for category,_,color in categories:
                count=int(subset[subset.response_state==category].parts.sum())
                if not count:continue
                width=100*count/n
                ax.barh(y,width,left=left,color=color,height=.60,edgecolor='white',lw=.7)
                ax.text(left+width/2,y,str(count),ha='center',va='center',fontsize=8,color='white' if color in (BLUE,RED) else '#263440')
                left+=width
            ax.text(103,y,f'n={n}',va='center',size=8,color='#65717B')
            if not n:ax.text(35,y,'No parts',size=8,va='center',color='#84909A')
        ax.set(xlim=(0,128),ylim=(3.6,-.6),xticks=[0,50,100],yticks=range(4),yticklabels=['Full','E50','Noncrit.','Critical'] if i==0 else [])
        ax.set_title('Answerable' if i==0 else 'Unanswerable',fontsize=10,pad=8)
        ax.set_xlabel('Within-stratum share (%)',fontsize=8)
        style(ax)
    fig.legend(handles=[Patch(facecolor=color,label=label) for _,label,color in categories],loc='upper left',bbox_to_anchor=(.054,.10),ncol=2,frameon=False,fontsize=8,columnspacing=1)

    # e: individual paper deltas; n=3, no invented uncertainty estimates.
    r=rectangles[4]
    heading(fig,r,'e','Parameter and model sensitivity','3 exploration papers • paired changes from full system')
    ax=fig.add_axes(r)
    e=read('e','paired_fixed')
    for y,(aspect,condition) in enumerate([(ASPECTS[0],'K5'),(ASPECTS[0],'LUNA'),(ASPECTS[2],'K5'),(ASPECTS[2],'LUNA')]):
        subset=e[(e.aspect==aspect)&(e.condition==condition)].sort_values('paper_id')
        color=BLUE if condition=='K5' else RED
        for j,(_,row) in enumerate(subset.iterrows()):
            yy=y+(j-1)*.17
            ax.plot([0,row.delta_q_fixed*100],[yy,yy],color=color,alpha=.3,lw=1)
            ax.scatter(row.delta_q_fixed*100,yy,color=color,marker=['o','s','^'][j],s=40,zorder=3,edgecolors='white',linewidth=.5)
    ax.axvline(0,color='#68747D',lw=.9)
    ax.set(xlim=(-110,65),ylim=(3.6,-.65),xticks=[-100,-50,0,50],yticks=range(4),yticklabels=['H · K5','H · LUNA','J · K5','J · LUNA'])
    ax.set_xlabel('Δ fixed-task correct coverage (percentage points)')
    style(ax)
    ax.legend(handles=[Line2D([0],[0],marker=m,ls='',color='#58616B',label=p) for m,p in zip(['o','s','^'],['68332-4','68412-5','68427-y'])],loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=8,columnspacing=.7,handletextpad=.3)

    # f: genuine observed inference-cost points; no line/frontier fit.
    r=rectangles[5]
    heading(fig,r,'f','Inference cost on fixed materials','H coverage • 5 papers × 4 conditions • successful stages only')
    ax=fig.add_axes(r)
    f=read('f','cost_quality');f=f[f.aspect==ASPECTS[0]].copy()
    assert len(f)==20 and f.cost_complete.all()
    f['tokens_k']=(f.input_tokens+f.output_tokens)/1000
    for condition in COLORS:
        vals=f[f.condition==condition]
        ax.scatter(vals.tokens_k,vals.q_fixed*100,s=48,marker=MARKERS[condition],facecolor='none' if condition in ('F','NONCRITICAL') else COLORS[condition],edgecolor=COLORS[condition],linewidth=1.2,label=LABELS[condition],alpha=.9)
    ax.set(ylim=(-5,108),yticks=[0,25,50,75,100])
    ax.set_xlabel('Input + output tokens per report (thousands)')
    ax.set_ylabel('H fixed-task correct coverage (%)')
    style(ax,'both')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=4,frameon=False,fontsize=8,columnspacing=.8,handletextpad=.3)
    fig.text(r[0],.081,'GEAR + Graph + writing; excludes evaluation and failed attempts.',size=8,color='#65717B')
    fig.text(.034,.023,'H: historical verification   N: knowledge position   J: joint contribution   S: structural interpretation   C: citation contact\nPanel a uses the original Fig. 4 evaluation; b–f use revised judgments. Small-sample results are descriptive; no claim of a budget frontier.',size=9,color='#58616B',linespacing=1.6)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'Fig5_initial.{ext}',dpi=300 if ext=='png' else None,facecolor='white')
    fig.savefig(OUT/'Fig5_preview.png',dpi=140,facecolor='white')
    plt.close(fig)
    f.to_csv(OUT/'panel_f_plotted_points.csv',index=False)
    (OUT/'render_manifest.json').write_text(json.dumps({'source':str(DATA),'script':str(Path(__file__).resolve()),'panels':6,'new_model_calls':0,'new_reports':0,'a_interval':'existing paper-bootstrap 95% CI','b_to_f_uncertainty':'individual observations; no inferential interval added','f_cost':'input plus output tokens; cached input and reasoning output not double-counted'},indent=2))


if __name__ == '__main__':
    main()

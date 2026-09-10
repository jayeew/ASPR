"""Paired benchmark and operational availability from the frozen 200-paper cohort."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/FROM_WEB/data'
OUT = ROOT / 'outputs/FROM_WEB_v2'
from figure_pipeline.renderers.theme import PALETTE
from figure_pipeline.analysis.benchmark import METRICS, interval, paired, transitions
from figure_pipeline.data.status import operational

COLORS = PALETTE


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))










def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=240, facecolor='white')
    plt.close(fig)


def render(summary: list[dict], points: list[dict], transitions_: list[dict]) -> None:
    plt.rcParams.update({'font.size':8, 'axes.titlesize':10, 'axes.labelsize':8, 'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    fig = plt.figure(figsize=(180/25.4, 220/25.4), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.15, 1])
    ax = fig.add_subplot(gs[0])
    for i, row in enumerate(summary):
        for offset, key, color, label in [(-.13,'direct',COLORS[2],'Direct LLM'),(.13,'graph',COLORS[4],'Graph + joint')]:
            mean, low, high = row[key]
            ax.errorbar(mean,i+offset,xerr=[[mean-low],[high-mean]],fmt='o',color=color,capsize=2,ms=4,label=label if i == 0 else None)
        ax.text(101,i,f'n={row["n"]}',va='center',fontsize=8)
    ax.set(yticks=range(4),yticklabels=[r['label'] for r in summary],xlabel='Mean paper rate (%)',xlim=(-3,116),xticks=[0,25,50,75,100],title='a  Graph + joint versus Direct LLM')
    ax.invert_yaxis(); ax.legend(loc='lower center',bbox_to_anchor=(.58,.02),frameon=False); ax.grid(axis='x',alpha=.15)
    ax = fig.add_subplot(gs[1])
    for i,row in enumerate(summary):
        vals=[p['delta'] for p in points if p['metric']==row['metric']]
        jitter=np.random.default_rng(i).uniform(-.16,.16,len(vals))
        ax.scatter(vals,i+jitter,s=8,alpha=.22,color=COLORS[2],rasterized=False)
        mean,lo,hi=row['delta']; ax.errorbar(mean,i+.24,xerr=[[mean-lo],[hi-mean]],fmt='D',ms=4,capsize=3,color=COLORS[0])
        ax.text(102,i,f'{mean:+.2f}\n[{lo:+.2f}, {hi:+.2f}]',fontsize=7.5,va='center')
    ax.axvline(0,color='.45',lw=.7); ax.set(yticks=range(4),yticklabels=['Coverage ↑','Agreement ↑','Reasons ↑','Contradiction ↓'],xlim=(-103,155),xticks=[-100,-50,0,50,100],xlabel='Graph + joint − Direct LLM (percentage points)',title='b  Paired papers and 95% bootstrap intervals'); ax.invert_yaxis()
    ax=fig.add_subplot(gs[2]); states=['same','partial','none']; matrix=np.zeros((3,3),int)
    for row in transitions_:
        if row['direct_scope'] in states and row['graph_scope'] in states:
            matrix[states.index(row['direct_scope']),states.index(row['graph_scope'])]+=1
    ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,matrix,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('seq',['#ffffff',COLORS[2],COLORS[0]]),shading='flat'); ax.set_ylim(2.5,-.5)
    for i in range(3):
        for j in range(3): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.5 else COLORS[0],fontsize=10)
    ax.set(xticks=range(3),xticklabels=states,yticks=range(3),yticklabels=states,xlabel='Graph + joint: matched scope',ylabel='Direct LLM: matched scope',title='c  Reference-aligned scope transitions (counts)')
    fig.suptitle('Saved AI matches to reviewer references\nLegacy unblinded evaluation · paper-macro estimates',fontsize=10)
    save(fig,'fig04_benchmark')


def readiness(rows: list[dict]) -> None:
    fig, axes=plt.subplots(2,1,figsize=(180/25.4,150/25.4),layout='constrained')
    journals=list(dict.fromkeys(r['journal'] for r in rows))
    labels=[j.replace('Communications ','Comm. ').replace('Nature Communications','Nature Comm.') for j in journals]
    for i,j in enumerate(journals):
        group=[r for r in rows if r['journal']==j]
        unknown=sum(r['analyzed_claims']==0 for r in group)
        partial=sum(0<r['analyzed_claims']<r['requested_claims'] for r in group)
        complete=len(group)-unknown-partial
        start=0
        for n,color,label in [(complete,COLORS[4],'All cards saved'),(partial,COLORS[2],'Some cards saved'),(unknown,'#dddddd','No saved card; execution unknown')]:
            axes[0].barh(i,n,left=start,color=color,label=label if i==0 else None); start+=n
    axes[0].set(yticks=range(len(labels)),yticklabels=labels,xlabel='Papers',title='S1a  Artifact availability by journal'); axes[0].legend(fontsize=7.5)
    observed=[r for r in rows if r['analyzed_claims']]
    groups=Counter((r['execution_fraction'],r['evidence_coverage_among_analyzed']) for r in observed)
    for (x,y),count in sorted(groups.items()):
        axes[1].scatter([x],[y],s=16*count,alpha=.5,color=COLORS[2],edgecolors='white',linewidths=.5)
        if count > 1:
            axes[1].annotate(str(count),(x,y),xytext=(-5,-10) if x>.9 and y>.9 else (5,4),textcoords='offset points',fontsize=7.5)
    axes[1].text(.03,.08,'Point area = papers; unlabelled = 1',transform=axes[1].transAxes,fontsize=7.5)
    axes[1].set(xlim=(-.03,1.09),ylim=(-.03,1.12),xlabel='Saved GEAR cards / requested claims',ylabel='Usable comparisons / saved cards',title=f'S1b  Separate denominators · {len(rows)-len(observed)} papers unknown')
    save(fig,'supplement_S1_readiness')


def main() -> None:
    snapshot=json.loads((SOURCE/'fig04_fig06_fig07_snapshot.json').read_text())
    summary,points=paired(snapshot); changes=transitions(snapshot); statuses=operational(snapshot)
    for name,value in [('benchmark_summary',summary),('benchmark_paired',points),('reference_transitions',changes),('paper_status',statuses)]: dump(OUT/'data'/f'{name}.json',value)
    render(summary,points,changes); readiness(statuses)
    dump(OUT/'audit/benchmark.json',dict(bootstrap_unit='paper',replicates=10000,seed=6082026,summary=summary,source='fig04_fig06_fig07_snapshot.json',independent_correctness=False,identity_blinded=False))


if __name__=='__main__': main()

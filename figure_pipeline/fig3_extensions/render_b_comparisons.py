"""Fit PR, joint gains and four historical facets in the existing b2+b4 area."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import render_ad_replacements as ad
from .cd_sources import SOURCE, read
from .render_abcd import paired_samples

NAMES = ['GPT-5.6', 'GEAR', 'Graph', 'Full', 'EACL', 'ReviewG']
METHODS = ['direct_a', 'gear', 'graph', 'fusion', 'eacl', 'reviewgrounder']
COLORS = ['#84909E', '#E16C35', '#1269DF', '#713BCB', '#219D91', '#92944F']
MARKERS = ['o', 's', '^', 'D', 'h', 'v']


def heading(fig: plt.Figure, x: float, text: str, size: float = 12.5) -> None:
    fig.text(x/228, .982, text, va='top', fontfamily='Times New Roman',
             fontweight='bold', fontsize=size)


def axes(fig: plt.Figure, x: float, width: float, bottom: float = .235,
         height: float = .55) -> plt.Axes:
    ax = fig.add_axes([x/228, bottom, width/228, height])
    ax.set_facecolor('none'); ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8, length=2, pad=1)
    return ax


def main() -> None:
    ad.configure()
    fig = plt.figure(figsize=(228/25.4, 43/25.4))
    heading(fig, 0, 'b2  Precision–recall')
    ax = axes(fig, 12, 37)
    ax.set(xlim=(.25,.8), ylim=(.5,1), xticks=[.3,.5,.7], yticks=[.5,.75,1])
    ax.grid(color=ad.GRID, lw=.45)
    data = read(SOURCE/'derived/components/b2.json')['rows']
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        r = next(r for r in data if r['method']==method)
        ax.errorbar(r['x'], r['y'], xerr=[[r['x']-r['x_low']],[r['x_high']-r['x']]],
            yerr=[[r['y']-r['y_low']],[r['y_high']-r['y']]], fmt=marker,
            color=color, capsize=2, ms=3.5, lw=.8, alpha=.9)
    ax.set_xlabel('Recall', fontsize=9, labelpad=1.5)
    ax.set_ylabel('Precision', fontsize=9, labelpad=1.5)

    heading(fig, 123, 'b4  Paired contribution gains')
    for index, r in enumerate(paired_samples()):
        start = 123+index*52.5
        ax = axes(fig, start+8, 43, bottom=.32, height=.41)
        ad.density(ax, r)
        ax.axvline(0,color='#647487',lw=.65);ax.axhline(0,color='#647487',lw=.65)
        ax.set(xlim=(-5,42),ylim=(-16,24),xticks=[0,20,40],yticks=[-10,0,10,20])
        ax.grid(color=ad.GRID,lw=.4);ax.set_axisbelow(True)
        label='GEAR' if r['comparator']=='gear' else 'Graph'
        fig.text((start+8)/228,.78,f"Full − {label} · n={r['n']} papers",
                 fontsize=8,color=ad.COLORS[r['comparator']],fontweight='bold')
        ax.text(.98,.97,f"ΔR {r['mean'][0]:+.1f}\nΔP {r['mean'][1]:+.1f}",
            transform=ax.transAxes,ha='right',va='top',fontsize=7,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.4})
        ax.set_xlabel('Recall gain (pp)',fontsize=8,labelpad=1)
        if index==0:ax.set_ylabel('Precision gain (pp)',fontsize=8,labelpad=1)
    handles=[Line2D([],[],color='none',marker='D',markersize=3,markerfacecolor=ad.COLORS['fusion'],markeredgecolor=ad.INK,label='Mean'),
             Line2D([],[],color=ad.COLORS['fusion'],lw=1,label='50%'),
             Line2D([],[],color=ad.COLORS['fusion'],ls='--',lw=.7,label='95% joint region')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(175.5/228,-.008),ncol=3,
               frameon=False,fontsize=7.3,handlelength=1.3,handletextpad=.5,columnspacing=.8)

    heading(fig,55,'b3  Full − comparator (pp)',11.5)
    rows = read(SOURCE/'derived/components/b4.json')['rows']
    for index,(metric,label) in enumerate(zip(
            ['recall','precision','historical_comparison','scope_coverage'],
            ['Recall','Precision','Hist. comp.','Scope coverage'])):
        ax = axes(fig,56+(index%2)*32.5,30,bottom=.57 if index<2 else .13,height=.20)
        ax.axvline(0,color='#8C9DAF',lw=.65);ax.axhspan(.55,1.45,color='#FFF0E7',zorder=-1)
        for y,method in enumerate(m for m in METHODS if m!='fusion'):
            r=next(r for r in rows if r['method']==method and r['metric']==metric)
            value,low,high=[r[k]*100 for k in ('estimate','low','high')]
            mi=METHODS.index(method)
            ax.errorbar(value,y,xerr=[[max(0,value-low)],[max(0,high-value)]],
                fmt=MARKERS[mi],color=COLORS[mi],ms=2.3,capsize=1,elinewidth=.65)
        ax.set(xlim=(-10,50),ylim=(4.5,-.5),xticks=[0,20,40],yticks=[])
        ax.tick_params(labelsize=7,length=1.5,pad=.5)
        ax.set_title(label,fontsize=8,pad=2)
    ad.save(fig,'B_precision_joint_and_historical')
    print('Rendered combined b2/b4 area with restored four facets.')


if __name__=='__main__':
    main()

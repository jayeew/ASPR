"""Draw the complete 100-paper study from prepared observations, without model calls."""
from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MPath
from figure_pipeline.fig1_reference.svg import measure
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig3_reference'
DATA = ROOT / 'outputs/fig3_reference/study/derived'
METHODS = ['direct_a', 'gear', 'graph', 'fusion', 'eacl', 'reviewgrounder']
OTHERS = [m for m in METHODS if m != 'fusion']
NAMES = dict(zip(METHODS, ['GPT-5.6', 'GEAR', 'Graph', 'Full', 'EACL', 'ReviewG']))
COLORS = dict(zip(METHODS, ['#84909E','#E16C35','#1269DF','#713BCB','#219D91','#92944F']))
MARKERS = dict(zip(METHODS, ['o','s','^','D','h','v']))
INK, MUTED, GRID = '#10213B', '#647487', '#E8EEF3'
STATES = ['supported','partly_supported','not_verifiable','contradicted']
STATE_COLORS = ['#48AD83','#F0BA58','#BAC2CD','#EB596E']
Q = ['contribution_fidelity','historical_increment','knowledge_relations','scope_uncertainty','whole_paper_synthesis']
QL = ['CF','HI','KR','SU','WS']


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


class Figure:
    def __init__(self) -> None:
        for name in ['arial.ttf','arialbd.ttf','times.ttf','timesbd.ttf','msyh.ttc']:
            font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/'+name)
        plt.rcParams.update({'font.family':['Arial','Microsoft YaHei'],'font.size':12,'text.color':INK,
            'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
            'axes.edgecolor':'#8295A7','axes.linewidth':.7,'svg.fonttype':'path',
            'pdf.fonttype':42,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
            'lines.linewidth':1.1,'savefig.facecolor':'white'})
        self.layout = read(OUT/'layouts/style.json')
        self.height = self.layout['canvas']['height_mm']
        self.fig = plt.figure(figsize=(420/25.4,self.height/25.4),facecolor='white')
        self.comp = {p.stem:read(p) for p in (DATA/'components').glob('*.json')}
        self.summary = pd.read_csv(DATA/'summary.csv')
        self.pm = pd.read_csv(DATA/'paper_metrics.csv')
        self.wide = self.pm.pivot(index=['paper_id','method'],columns='metric',values='value')
        self.header()

    def words(self, x: float, y: float, value: str, size: float = 12,
              color: str = INK, bold: bool = False, **kw: Any) -> None:
        self.fig.text(x/420,1-y/self.height,value,fontsize=size,color=color,
                      weight='bold' if bold else 'normal',va='top',**kw)

    def header(self) -> None:
        self.words(8,6,'Fig. 3 | Evidence-grounded evaluation of innovation analysis',30,bold=True,fontfamily='Times New Roman')
        counts={r['label']:r['value'] for r in self.rows('a1')}
        labels=['●  '+label for label in ['GPT-5.6','GEAR','Graph','Full','EACL','ReviewGrounder']]
        widths=[measure(label,13,sans=True)*25.4/72 for label in labels]
        gap=12.0
        x=8.0
        for m,label,width in zip(METHODS,labels,widths):
            self.words(x,28,label,13,COLORS[m]); x+=width+gap
        for key,p in self.layout['panels'].items():
            x,y,w,h = [p[k] for k in ['x','y','w','h']]
            patch = FancyBboxPatch((x/420,1-(y+h)/self.height),w/420,h/self.height,
                boxstyle='round,pad=0,rounding_size=0.003',transform=self.fig.transFigure,
                facecolor='#FBFDFF',edgecolor='#BDD8EB',linewidth=.8,zorder=-10)
            self.fig.add_artist(patch)
            self.fig.add_artist(Rectangle(((x+.3)/420,1-(y+10)/self.height),(w-.6)/420,9.7/self.height,
                transform=self.fig.transFigure,facecolor='#EEF7FC',edgecolor='none',zorder=-9))
            self.words(x+3,y+1.7,key,25,bold=True)
            self.words(x+12,y+2.2,p['title'],22 if key not in 'gh' else 19,bold=True,fontfamily='Times New Roman')

    def box(self, key: str) -> tuple[float,float,float,float]:
        return tuple(self.layout['components'][key]['page_box_mm'])

    def title(self, key: str, title: str | None = None) -> None:
        x,y,w,h = self.box(key)
        value = key+'  '+(title or self.layout['components'][key]['title'])
        size = min(15, 15*(w*72/25.4-2)/measure(value,15,True))
        self.words(x,y,value,size,bold=True,fontfamily='Times New Roman')

    def ax(self, key: str, margins: tuple[float,float,float,float] = (13,8,3,8),
           projection: str | None = None) -> Axes:
        x,y,w,h = self.box(key); l,t,r,b=margins
        ax=self.fig.add_axes([(x+l)/420,1-(y+h-b)/self.height,(w-l-r)/420,(h-t-b)/self.height],projection=projection)
        ax.set_facecolor('none')
        if projection is None:
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(length=2.5,pad=2)
        return ax

    def subax(self, key: str, rel: tuple[float,float,float,float], projection: str | None = None) -> Axes:
        x,y,w,h=self.box(key);a,b,c,d=rel
        ax=self.fig.add_axes([(x+a*w)/420,1-(y+(b+d)*h)/self.height,c*w/420,d*h/self.height],projection=projection)
        ax.set_facecolor('none')
        if projection is None:
            ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2,pad=2)
        return ax

    def note(self,key: str,value: str,size: float=10.5) -> None:
        x,y,w,h=self.box(key)
        size=min(size,size*(w*72/25.4)/measure(value,size,sans=True))
        self.words(x,y+h-3.5,value,size,MUTED)

    def rows(self,key: str) -> list[dict[str,Any]]:
        return self.comp[key]['rows']

    def row(self,key: str,method: str,metric: str | None=None) -> dict[str,Any]:
        return next(r for r in self.rows(key) if r.get('method')==method and
                    (metric is None or r.get('metric',r.get('category'))==metric))

    def summary_row(self,method: str,metric: str) -> pd.Series:
        return self.summary[(self.summary.method==method)&(self.summary.metric==metric)].iloc[0]


def method_axis(ax: Axes, methods: list[str]=METHODS) -> None:
    ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods]);ax.set_ylim(len(methods)-.5,-.5)
    for t,m in zip(ax.get_yticklabels(),methods):t.set_color(COLORS[m])


def interval(ax: Axes, row: Any, y: float, color: str, marker: str='o', scale: float=1) -> None:
    v,lo,hi=[float(row[k])*scale for k in ['estimate','low','high']]
    ax.errorbar(v,y,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt=marker,color=color,
                capsize=2,markersize=4,elinewidth=.9)


def card(ax: Axes,x: float,y: float,w: float,h: float,title: str,body: str,
         color: str='#1269DF',size: float=12) -> None:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007',lw=.65,
                              ec=color,fc=matplotlib.colors.to_rgba(color,.055)))
    height_pt=ax.get_position().height*ax.figure.get_figheight()*72*h
    width_pt=ax.get_position().width*ax.figure.get_figwidth()*72*w
    balanced=(height_pt-3)/(1.15*(len(body.splitlines())+1))
    title_size=min(size,balanced,size*(width_pt-5)/measure(title,size,True,sans=True))
    body_size=min(size-1,balanced)
    body_size=min(body_size,min(body_size*(width_pt-6)/max(1,measure(line,body_size,sans=True)) for line in body.splitlines()))
    pt_to_y=h/height_pt
    ax.text(x+w/2,y+h-1*pt_to_y,title,ha='center',va='top',weight='bold',fontsize=title_size,color=color)
    ax.text(x+w/2,y+h-(title_size*1.15+2)*pt_to_y,body,ha='center',va='top',fontsize=body_size,linespacing=1.15)


def panel_a(f: Figure) -> None:
    for k in ['a1','a2','a3','a4']:f.title(k)
    ax=f.ax('a1',(0,7,0,0));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    counts={r['label']:r['value'] for r in f.rows('a1')}
    for i,(v,l) in enumerate([(counts['Roster papers'],'papers'),(counts['Shared claims'],'shared\nclaims'),(counts['Selected core contributions'],'core\nobjects'),(len(f.wide),'reports')]):
        ax.text(.12+i*.25,.91,str(v),ha='center',size=21,weight='bold',fontfamily='Times New Roman')
        ax.text(.12+i*.25,.80,l,ha='center',va='top',size=10)
    fields=f.rows('a1')[4:]; groups=sorted(fields,key=lambda r:-r['value'])
    top=groups[:3];other=sum(r['value'] for r in groups[3:]);colors=['#366896','#6E98BB','#A0BDD3','#D5E0E9']
    labels=['Biochem.','Medicine','Engineering','Other']
    vals=[r['value'] for r in top]+[other];idx=0
    for j,n in enumerate(vals):
        for _ in range(n):
            ax.plot(.035+(idx%10)*.047,.46-(idx//10)*.047,'o',ms=3.7,color=colors[j]);idx+=1
        ax.text(.55,.47-j*.115,f'{labels[j]}  {n}',size=9.5,color='#536F88',va='center')
    ax.text(0,.57,'Fields · one dot = one paper',size=10,weight='bold')
    ax=f.ax('a2',(24,16,2,4))
    # Independent extraction includes the baseline's own report construction.
    inputs=np.array([[1,0,0,0,0,0,0],[1,0,1,1,0,0,0],[1,0,1,0,1,1,0],
                     [1,0,1,1,1,1,1],[1,1,0,1,0,0,0],[1,1,0,1,0,0,0]])
    for i in range(6):
        for j in range(7):
            ax.text(j,i,'●' if inputs[i,j] else '–',ha='center',va='center',size=13,color='#53789A' if inputs[i,j] else '#BCC9D4')
        ax.axhline(i+.5,color=GRID,lw=.5)
    ax.set_xlim(-.5,6.5);method_axis(ax)
    ax.set_xticks(range(7),['Manu-\nscript','Own\nextract.','Shared\nclaims','Prior\nretrieval','Claim\nGraph','Joint\nGraph','Branch\nfusion'])
    ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,labelsize=10);ax.spines[:].set_visible(False)
    ax=f.ax('a3',(0,7,0,1));ax.axis('off')
    card(ax,.02,.61,.45,.32,'Historical R','Core manuscript\n+ dated literature')
    card(ax,.54,.61,.43,.32,'Report P','Method report\nclaims','#713BCB')
    card(ax,.23,.22,.54,.25,'Evidence judgment V','R + P + source material')
    for x in [.24,.76]:ax.annotate('',(.5,.48),(x,.60),arrowprops={'arrowstyle':'->','color':'#53789A'})
    ax.text(.5,.08,'Quality · Support · Reviewer concerns\nInformation · Preference',ha='center',va='center',size=11)
    ax=f.ax('a4',(0,7,0,0));ax.axis('off')
    for y,t,b,c in [(.70,'Generation','Manuscript + method inputs','#1269DF'),(.41,'Historical reference','Manuscript + prior sources','#92944F'),(.12,'Post-hoc assessment','Reports + common source pool','#713BCB')]:
        card(ax,.02,y,.96,.24,t,b,c,11.5)


def panel_b(f: Figure) -> None:
    f.title('b1','Reference availability')
    ax=f.ax('b1',(0,8,1,1));ax.axis('off')
    rows={r['category']:r for r in f.rows('b1')};total=sum(r['count'] for r in rows.values());ins=rows.get('insufficient_material',{}).get('count',0)
    ax.barh(.78,ins/total,left=0,height=.22,color='#BAC2CD');ax.barh(.78,(total-ins)/total,left=ins/total,height=.22,color='#1269DF')
    if ins:
        ax.text(ins/total/2,.78,f'{ins}',ha='center',va='center',color='white',weight='bold',size=19)
    ax.text(.95,.78,str(total-ins),ha='center',va='center',color='white',size=12,weight='bold')
    ax.text(0,.58,f'Insufficient: {ins} / determinate: {total-ins}',size=10.7)
    ax.text(0,.42,f'{total-ins} determinate cores',size=11,weight='bold')
    left=0
    for cat,c in [('supported_difference','#48AD83'),('bounded_increment','#F0BA58'),('substantially_covered','#8DA8C0')]:
        n=rows[cat]['count'];ax.barh(.27,n/(total-ins),left=left,height=.17,color=c)
        ax.text(left+n/(total-ins)/2,.27,str(n),ha='center',va='center',size=11);left+=n/(total-ins)
    ax.text(0,.06,'Difference · Bounded · Covered',size=10)
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    # b2/b3/b4 are drawn together by the inserted comparison SVG.
    f.title('b5','Historical comparison and overclaims')
    ax=f.ax('b5',(0,7,0,2));ax.axis('off')
    cols=['historical_comparison','false_firstness','false_antecedence','material_overclaim']
    headers=['Correct ↑','Firstness ↓','Anteced. ↓','Overclaim ↓']
    for j,t in enumerate(headers):ax.text(.36+j*.18,.88,t,ha='center',size=10)
    for i,m in enumerate(METHODS):
        y=.75-i*.115
        if m=='fusion':
            ax.add_patch(Rectangle((-.015,y-.03),1.02,.115,fc=matplotlib.colors.to_rgba(COLORS[m],.16),ec='none',zorder=-1))
        ax.text(0,y,NAMES[m],size=11,color=COLORS[m])
        for j,met in enumerate(cols):ax.text(.36+j*.18,y,f"{f.row('b5',m,met)['estimate']*100:.1f}",ha='center',size=11)
        ax.axhline(y-.03,color=GRID,lw=.5)
    f.title('b6','Historical reference × report judgment · paper macro joint %')
    refs=['supported_difference','bounded_increment','substantially_covered','insufficient_material']
    pred=['positive_increment','limited_increment','substantially_known','explicit_abstention','not_addressed']
    for j,m in enumerate(METHODS):
        ax=f.subax('b6',(.11+j*.148,.24,.125,.53));a=np.array([[next((r['estimate'] for r in f.rows('b3') if r['method']==m and r['x']==r0 and r['y']==p), 0.0) for p in pred] for r0 in refs])
        cmap=LinearSegmentedColormap.from_list('b3_'+m,['#FFFFFF',COLORS[m]])
        ax.imshow(a,cmap=cmap,vmin=0,vmax=.85,aspect='auto')
        for i in range(4):
            for k in range(5):ax.text(k,i,f'{a[i,k]*100:.0f}',ha='center',va='center',size=9,color='white' if a[i,k]>.45 else INK)
        ax.axhline(2.5,color=INK,lw=.7);ax.set_xticks(range(5),['Pos','Lim','Known','Abst','NA'])
        ax.set_yticks(range(4),['Difference','Bounded','Covered','Insufficient'] if j==0 else [])
        ax.tick_params(length=0,labelsize=10);ax.spines[:].set_visible(False)


def panel_c(f: Figure) -> None:
    f.title('c1','Five-dimensional score distributions')
    for j,(metric,label) in enumerate(zip(Q,QL)):
        ax=f.subax('c1',(.10+j*.179,.22,.16,.60))
        ax.set_axisbelow(True);ax.grid(axis='y',color=GRID,lw=.45)
        ax.axhspan(2.55,3.45,color=COLORS['fusion'],alpha=.06,zorder=0)
        for i,method in enumerate(METHODS):
            values=f.pm.loc[(f.pm.method==method)&(f.pm.metric=='quality_'+metric),'value'].dropna()
            for score,count in values.value_counts().items():
                proportion=count/len(values)
                ax.scatter(score,i,s=proportion*120,color=COLORS[method],alpha=.9,
                           edgecolors='white',linewidths=.3,zorder=3)
        ax.set_xlim(-.55,3.55);ax.set_ylim(5.55,-.55)
        ax.set_xticks([0,1,2,3]);ax.set_yticks(range(6),[NAMES[m] for m in METHODS] if j==0 else [])
        ax.tick_params(length=0,labelsize=8,pad=3);ax.spines[:].set_visible(False)
        for tick,method in zip(ax.get_yticklabels(),METHODS):tick.set_color(COLORS[method])
        ax.set_title(label,fontsize=11,pad=6)
    x,y,w,h=f.box('c1')
    legend=f.subax('c1',(.52,.91,.45,.065));legend.axis('off');legend.set_xlim(0,1);legend.set_ylim(0,1)
    for i,prop in enumerate([.1,.5,1.0]):
        legend.scatter(.07+i*.32,.5,s=prop*120,color=MUTED)
        legend.text(.13+i*.32,.5,f'{prop:.0%}',va='center',fontsize=7.5,color=MUTED)
    f.title('c3','Paired quality differences · Full − comparator')
    ax=f.ax('c3',(10,11,4,7));rows=f.rows('c3');limit=max(abs(r[v]) for r in rows for v in ['low','high'])
    a=np.array([[f.row('c3',m,'quality_'+q)['estimate'] for m in OTHERS] for q in Q])
    cmap=LinearSegmentedColormap.from_list('effect',['#EAA678','#FFFFFF','#9671D0'])
    ax.imshow(a,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
    for i,q in enumerate(Q):
        for j,m in enumerate(OTHERS):
            r=f.row('c3',m,'quality_'+q);ax.text(j,i-.06,f"{r['estimate']:+.2f}",ha='center',va='center',size=11)
            ax.plot([j+r['low']/limit*.38,j+r['high']/limit*.38],[i+.30]*2,color=INK,lw=.7)
            ax.plot([j,j],[i+.23,i+.37],color='#AAB4BE',lw=.6)
    ax.add_patch(Rectangle((.5,-.5),1,5,fill=False,ec=COLORS['gear'],lw=1.1))
    ax.set_xticks(range(5),[NAMES[m] for m in OTHERS]);ax.xaxis.tick_top();ax.set_yticks(range(5),QL);ax.tick_params(length=0,labelsize=11);ax.spines[:].set_visible(False)
    ns=[r['n'] for r in f.rows('c3')]
    f.title('c4','Contribution coverage · mention vs supported')
    ax=f.ax('c4',(19,7,9,8));method_axis(ax)
    ax.set_xlim(.4,1.075);ax.set_xticks([.4,.6,.8,1],['40','60','80','100%'])
    ax.grid(axis='x',color=GRID,lw=.5);ax.set_axisbelow(True)
    for i,m in enumerate(METHODS):
        mention=f.row('c4',m,'Mentioned')['estimate']
        supported=f.row('c4',m,'Source_supported')['estimate']
        ax.plot([supported,mention],[i,i],color=COLORS[m],alpha=.6,lw=1.8,zorder=2)
        ax.plot(supported,i,'D',ms=4,mfc=COLORS[m],mec=COLORS[m],zorder=3)
        ax.plot(mention,i,'o',ms=4,mfc='white',mec=COLORS[m],zorder=4)
        ax.annotate(f'{supported:.1%}',(supported,i),xytext=((5 if supported<.52 else -5),0),textcoords='offset points',
                    ha='left' if supported<.52 else 'right',va='center',fontsize=8,color=COLORS[m],
                    bbox={'facecolor':'#FBFDFF','edgecolor':'none','pad':.2})
        ax.annotate(f'{mention:.1%}',(mention,i),xytext=(5,0),textcoords='offset points',
                    ha='left',va='center',fontsize=8,color=COLORS[m])
    f.title('c5','Assertion composition')
    cats=['manuscript_fact','historical_comparison','increment','structure','scope'];colors=['#83B5D9','#E9AE80','#B19BCC','#62AAA2','#CCD6DE']
    for i,m in enumerate(METHODS):
        ax=f.subax('c5',(.01+(i%3)*.33,.22+(i//3)*.33,.28,.31));ax.set_aspect('equal',adjustable='box');ax.axis('off')
        vals=[f.row('c5',m,c)['estimate'] for c in cats]
        ax.pie(vals,colors=colors,wedgeprops={'width':.47,'edgecolor':'white','linewidth':.35},startangle=90,radius=1.22)
        ax.text(0,0,NAMES[m],ha='center',va='center',size=8.5,color=COLORS[m])
    x,y,w,h=f.box('c5')
    for j,(t,c) in enumerate(zip(['Fact','History','Increment','Structure','Scope'],colors)):
        f.words(x+(j%3)*31,y+h-6+(j//3)*3.1,'■ '+t,8.5,c)
    f.title('c6','One core, three report descriptions')
    ax=f.ax('c6',(0,7,0,0));ax.axis('off')
    # English translations of selected passages from the current c6 source excerpts.
    translations={
        'gear': 'The manuscript supports 64 × 64 reflection imaging above 30,000 pixels/s, and reports 14.2 FPS / 14,540 PPS at a sampling ratio of 0.4.',
        'graph': 'The material supports throughput under the stated conditions. It does not establish that 20 kHz operation and 20 averages per image used the same conditions.',
        'fusion': 'The manuscript reports throughput over five times prior technology; however, the supplied material does not independently verify the full scope of that baseline comparison.',
    }
    quotes=list(translations.items())
    for i,(m,q) in enumerate(quotes):
        r=f.row('c6',m);x=i/3
        ax.add_patch(Rectangle((x+.005,.16),.319,.68,fc=matplotlib.colors.to_rgba(COLORS[m],.06),ec=matplotlib.colors.to_rgba(COLORS[m],.3),lw=.6))
        ax.text(x+.015,.89,NAMES[m]+' · '+str(r['P']).replace('_',' '),color=COLORS[m],weight='bold',size=9)
        ax.text(x+.015,.77,textwrap.fill(q,38),size=9,va='top',linespacing=1.08)


def panel_d(f: Figure) -> None:
    f.title('d1','Support versus original traceability')
    ax=f.ax('d1',(14,8,23,15));df=f.wide[['S','T']].dropna()
    ax.fill([0,.6,.6],[0,0,.6],color='#EDEFF2',zorder=-1);ax.plot([0,.6],[0,.6],'--',color='#66829F',lw=.8)
    cmap=LinearSegmentedColormap.from_list('d1_blues',plt.get_cmap('Blues')(np.linspace(.38,1,256)))
    coll=ax.hexbin(df['T'],df['S'],gridsize=42,extent=(0,.6,0,.6),mincnt=1,cmap=cmap,linewidths=.12,edgecolors='white')
    for m in METHODS:
        g=df.xs(m,level='method');ax.plot(g['T'].mean(),g['S'].mean(),MARKERS[m],color=COLORS[m],mec='white',mew=.6,ms=7)
    ax.set(xlim=(0,.6),ylim=(0,.6),xlabel='Original traceability T',ylabel='Source support S');ax.set_xticks([0,.2,.4,.6]);ax.set_yticks([0,.2,.4,.6])
    cb=f.fig.colorbar(coll,ax=ax,fraction=.045,pad=.03);cb.ax.tick_params(labelsize=9);cb.set_label('Observations / bin',size=10)
    ax.text(.43,.13,'T > S\n(infeasible)',color='#8995A2',ha='center',size=11)
    f.title('d2','Support-state composition')
    ax=f.ax('d2',(20,10,10,20));method_axis(ax);ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('Paper macro proportion')
    for i,m in enumerate(METHODS):
        left=0
        for state,c in zip(STATES,STATE_COLORS):
            r=f.summary_row(m,'support_'+state);v=r['estimate'];ax.barh(i,v,left=left,color=c,height=.60);left+=v
        ax.text(1.03,i,str(int(r['n'])),size=10,va='center')
    x,y,w,h=f.box('d2')
    for i,(lab,c) in enumerate(zip(['Supported','Partly supported','Not verifiable','Contradicted'],STATE_COLORS)):
        f.words(x+(i%2)*48,y+h-8+(i//2)*4,'● '+lab,10,c)
    f.title('d3','Reliability issue prevalence')
    ax=f.ax('d3',(35,14,4,13));metrics=['unsupported_definitive','false_antecedence','false_firstness','semantic_causal','omitted_scope']
    for i,met in enumerate(metrics):
        for j,m in enumerate(METHODS):
            r=f.row('d3',m,'error_prevalence_'+met);v=r['estimate']
            ax.scatter(j,i,s=v*300,color='#BA8194',alpha=.82,edgecolors='white',lw=.5)
            if v==0:ax.text(j,i,'–',ha='center',va='center',size=10)
    ax.set_xticks(range(6),[NAMES[m] for m in METHODS],rotation=30,ha='left');ax.xaxis.tick_top()
    ax.set_yticks(range(5),['Unsupported\ndefinitive','False\nantecedence','False\nfirstness','Semantic →\ncausal','Omitted\nscope'])
    ax.set_ylim(4.5,-.5);ax.set_xlim(-.5,5.5);ax.spines[:].set_visible(False);ax.tick_params(length=0,labelsize=10)
    for j,v in enumerate([.1,.5,1.]):ax.scatter(.1+j*1.7,5.25,s=v*300,color='#BA8194',clip_on=False);ax.text(.4+j*1.7,5.25,f'{v:.0%}',size=10,va='center')
    ns=[r['n'] for r in f.rows('d3')]
    f.title('d4','S − T gap')
    ax=f.ax('d4',(10,8,3,29));ax.set(xlim=(0,.30),ylim=(0,1),xlabel='S − T',ylabel='ECDF');ax.grid(color=GRID,lw=.5)
    ax.set_xticks([0,.15,.3]);ax.set_yticks([0,.5,1])
    x,y,w,h=f.box('d4')
    for i,m in enumerate(METHODS):
        vals=np.sort(f.wide.xs(m,level='method')['S_minus_T'].dropna().to_numpy())
        ax.step(np.r_[0,vals,.30],np.r_[0,np.arange(1,len(vals)+1)/len(vals),1],where='post',color=COLORS[m],lw=1)
        f.words(x,y+h-20+i*3.1,f'{NAMES[m]}  zero: {np.mean(vals==0):.0%}',9.2,COLORS[m])


def panel_e(f: Figure) -> None:
    f.title('e1','Issue and reason coverage')
    for j,(metric,label) in enumerate([('issue_coverage','Issue coverage'),('reason_coverage','Reason coverage')]):
        ax=f.subax('e1',(.17+j*.43,.27,.36,.59))
        for i,m in enumerate(METHODS):
            r=f.row('e1',m,metric);interval(ax,r,i,COLORS[m],'o',100)
            ax.annotate(f"{r['estimate']:.1%}",(r['high']*100,i),xytext=(3,0),
                        textcoords='offset points',fontsize=8,color=COLORS[m],va='center')
        ax.set_xlim(0,40);ax.set_xticks([0,20,40],['0','20','40%']);ax.set_ylim(5.55,-.55)
        ax.set_yticks(range(6),[NAMES[m] for m in METHODS] if j==0 else [])
        ax.tick_params(length=0,labelsize=9);ax.grid(axis='x',color=GRID,lw=.5)
        ax.set_title(label,fontsize=11,pad=6)
        for tick,m in zip(ax.get_yticklabels(),METHODS):tick.set_color(COLORS[m])
    f.title('e2','Scientific disagreement · shading scaled within columns')
    cats=['none','system_supported','system_error','review_unsubstantiated','unresolved']
    a=np.array([[f.row('e2',m,c)['estimate'] for c in cats] for m in METHODS])
    lo=a.min(axis=0);span=a.max(axis=0)-lo
    shade=np.divide(a-lo,span,out=np.zeros_like(a),where=span!=0)
    ax=f.subax('e2',(.17,.32,.81,.57))
    cmap=LinearSegmentedColormap.from_list('e2_columns',['#F5F8FC','#80AED4'])
    ax.imshow(shade,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    for i in range(6):
        for j in range(5):ax.text(j,i,f'{a[i,j]:.1%}',ha='center',va='center',size=9.5)
    ax.set_xticks(range(5),['No\ndisagreement','System\nsupported','System\nerror','Review lacks\nsupport','Unresolved'])
    ax.xaxis.tick_top();ax.set_yticks(range(6),[NAMES[m] for m in METHODS]);ax.tick_params(length=0,labelsize=9,pad=4)
    for tick,m in zip(ax.get_yticklabels(),METHODS):tick.set_color(COLORS[m])
    ax.spines[:].set_visible(False)
    ax.add_patch(Rectangle((-.5,2.5),5,1,fill=False,ec=COLORS['fusion'],lw=.8))
    f.title('e3','Reviewer attention')
    ax=f.ax('e3',(0,8,0,6));ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.5,1.5);ax.set_ylim(-1.2,1.3)
    for i,(cat,c) in enumerate([('novelty','#79AEE0'),('evidence','#65B4A6'),('scope','#A1B6C9')]):
        r=f.row('e3','all',cat);rad=np.sqrt(r['estimate'])*1.5;start=90+i*120
        ax.add_patch(Wedge((0,0),rad,start,start+120,fc=c,alpha=.7,ec='white',lw=1))
        theta=np.deg2rad(start+60);ax.text(np.cos(theta)*1.05,np.sin(theta)*1.02,
            cat.title()+'\n'+f"{r['estimate']:.1%}",ha='center',va='center',size=11)
    f.title('e4','Observed tone × scientific stance')
    ax=f.ax('e4',(22,9,3,10));tones=['positive','neutral','negative','mixed'];stances=['recognized','limited','challenged','unresolved']
    for i,stance in enumerate(stances):
        for j,tone in enumerate(tones):
            r=next(r for r in f.rows('e4') if r['x']==tone and r['y']==stance)
            ax.scatter(j,i,s=r['count']*.20,color='#69AEC2',alpha=.8,edgecolors='white',lw=.5)
            ax.text(j,i,str(r['count']),ha='center',va='center',size=9)
    ax.set_xticks(range(4),['Positive','Neutral','Negative','Mixed']);ax.tick_params(length=0,labelsize=10)
    ax.set_yticks(range(4),['Recognized','Limited','Challenged','Unresolved']);ax.set_ylim(3.5,-.5);ax.set_xlim(-.5,3.5);ax.spines[:].set_visible(False)
    f.title('e5','Documented within-reviewer follow-up')
    data=pd.read_csv(DATA/'review_dynamics.csv');same=data.same_reviewer_explicit.eq(True);paired=data[same&data.later_round.gt(data.earlier_round)&data.earlier_round.gt(0)]
    ax=f.subax('e5',(.02,.22,.47,.69));ax.axis('off')
    card(ax,.03,.72,.94,.28,f'{len(paired)} linked records',f'{paired.paper_id.nunique()} papers · explicit same reviewer',size=13)
    categories=['explicitly_resolved','partly_resolved','unresolved','no_explicit_followup','not_applicable']
    for i,(cat,c) in enumerate(zip(categories,['#48AD83','#F0BA58','#D47A89','#BAC2CD','#DDE4EA'])):
        n=int((paired.resolution==cat).sum());y=.58-i*.115
        ax.add_patch(Rectangle((.03,y-.035),n/max(1,len(paired))*.83,.08,fc=c,alpha=.15))
        ax.text(.88,y,str(n),va='center',ha='right',size=11,weight='bold')
        ax.text(.04,y,cat.replace('_',' '),size=10.5,va='center')
    ax=f.subax('e5',(.53,.20,.46,.72));ax.axis('off')
    ax.text(0,.99,'Observed record example',size=12,weight='bold',va='top')
    if len(paired):
        example=paired.sort_values(['paper_id','concern_id']).iloc[0]
        ax.text(0,.75,f"{example.paper_id}\n{example.reviewer_id} · R{int(example.earlier_round)} → R{int(example.later_round)}",size=9,color=MUTED,va='top')
        excerpt=str(example.later_reviewer_quote)
        ax.text(0,.50,textwrap.fill(excerpt[:95]+('…' if len(excerpt)>95 else ''),35),size=10,va='top',linespacing=1.15)
    else:
        ax.text(0,.65,'No eligible later-round linkage.',size=11,va='top')
    
    f.title('e6','Follow-up proportions · all records and explicit outcomes')
    counts={key:0 for key in categories};counts.update({r['category']:r['count'] for r in f.rows('e6')})
    explicit=sum(counts[k] for k in ['explicitly_resolved','partly_resolved','unresolved']);total=sum(counts.values())
    groups=[([counts['no_explicit_followup'],counts['not_applicable'],explicit],
             ['No explicit follow-up','Not applicable','Explicit outcome'],['#BECAD8','#EDF0F4','#648AB8'],total),
            ([counts['explicitly_resolved'],counts['partly_resolved'],counts['unresolved']],
             ['Resolved','Partly resolved','Unresolved'],['#48AD83','#F0BA58','#CE6F85'],explicit)]
    for j,(vals,labels,colors,n) in enumerate(groups):
        ax=f.subax('e6',(.02+j*.51,.19,.46,.49));ax.set_aspect('equal');ax.axis('off')
        ax.pie(vals,colors=colors,startangle=90,counterclock=False,
               wedgeprops={'width':.35,'edgecolor':'white','linewidth':.65})
        ax.text(0,.20,f'{n:,}',ha='center',va='center',fontsize=14,fontweight='bold')
        ax.text(0,-.40,'All records' if j==0 else 'Explicit outcomes',ha='center',fontsize=7.5)
        legend=f.subax('e6',(.02+j*.51,.70,.47,.27));legend.axis('off')
        for i,(v,label,c) in enumerate(zip(vals,labels,colors)):
            y=.85-i*.34;legend.scatter(.02,y,s=18,color=c,marker='s',edgecolors='#CBD3DD',linewidths=.3)
            legend.text(.065,y,label,va='center',fontsize=8)
            legend.text(.99,y,f'{v/n:.1%}',ha='right',va='center',fontsize=8,fontweight='bold')
        legend.set_xlim(0,1);legend.set_ylim(0,1)
    ax=f.subax('e6',(.475,.28,.055,.16));ax.axis('off')
    ax.annotate('',(.98,.5),(.02,.5),arrowprops={'arrowstyle':'->','lw':1,'color':'#648AB8'})


def panel_f(f: Figure) -> None:
    f.title('f1','Valid information yield')
    ax=f.ax('f1',(19,9,10,7));maximum=int(f.wide.information_count.max())
    xx=np.linspace(0,maximum+3,600);bandwidth=1.0
    values={m:f.wide.xs(m,level='method').information_count.dropna().to_numpy() for m in METHODS}
    # Common bandwidth and amplitude scale; reflection avoids density below zero.
    densities={m:np.mean(np.exp(-.5*((xx[:,None]-v)/bandwidth)**2)+
                         np.exp(-.5*((xx[:,None]+v)/bandwidth)**2),axis=1)
               /(bandwidth*np.sqrt(2*np.pi)) for m,v in values.items()}
    peak=max(d.max() for d in densities.values())
    for i,m in enumerate(METHODS):
        vals=values[m];height=densities[m]/peak*.80
        ax.fill_between(xx,i-height,i,color=COLORS[m],alpha=.50,linewidth=0)
        ax.plot(xx,i-height,color=COLORS[m],lw=.85)
        mean_height=np.interp(vals.mean(),xx,height)
        ax.plot([vals.mean()]*2,[i-mean_height,i],color=COLORS[m],lw=1)
        ax.text(maximum+3.15,i-.08,f'{np.mean(vals==0):.0%}',size=9,color=COLORS[m])
    ax.set_xlim(0,maximum+3);ax.set_ylim(5.3,-1)
    ax.set_yticks(range(6),[NAMES[m] for m in METHODS]);ax.tick_params(axis='y',length=0,labelsize=10)
    ax.set_xticks(np.arange(0,maximum+1,8));ax.set_xlabel('Information clusters / report',fontsize=9)
    ax.text(1.01,1.02,'Zero',transform=ax.transAxes,fontsize=8,color=INK)
    f.title('f2','Per-paper Full − branch union yield')
    ax=f.ax('f2',(12,8,3,10))
    d=f.wide.xs('fusion',level='method')
    delta=(d.fusion_full-d.fusion_union).dropna().sort_values(kind='stable')
    ax.bar(np.arange(len(delta)),delta,color=np.where(delta>=0,COLORS['fusion'],'#93A8BB'),width=.85)
    ax.axhline(0,color=INK,lw=.6)
    ax.set_xlim(-1,len(delta));ax.set_xticks([0,49,99],['1','50','100'])
    ax.set_ylabel('Δ information clusters',size=9)
    ax.set_xlabel('Papers sorted by net gain',size=9,labelpad=2)
    ax.tick_params(labelsize=8)
    f.title('f3','Six-method information intersections')
    rows=sorted(f.rows('f3'),key=lambda r:(-r['count'],r['category']));shown=rows[:12];remainder=rows[12:]
    vals=[r['count'] for r in shown]+[sum(r['count'] for r in remainder)];xs=np.r_[np.arange(12),12.7]
    ax=f.subax('f3',(.12,.19,.86,.34));ax.bar(xs,vals,color=['#6A9FC7']*12+['#B4C2CE'],width=.7)
    ax.set_xlim(-.6,13.2);ax.set_ylim(0,max(vals)*1.23);ax.set_xticks([]);ax.tick_params(labelsize=9);ax.set_ylabel('Clusters',size=10)
    for x,v in zip(xs,vals):ax.text(x,v+3,str(v),ha='center',size=10)
    ax=f.subax('f3',(.12,.55,.86,.40));ax.set_xlim(-.6,13.2);method_axis(ax);ax.spines[:].set_visible(False);ax.set_xticks([]);ax.tick_params(length=0,labelsize=9)
    for j,r in enumerate(shown):
        members=r['category'].split('+');ids=[i for i,m in enumerate(METHODS) if m in members]
        ax.plot([j,j],[min(ids),max(ids)],color='#69859A',lw=.8)
        for i,m in enumerate(METHODS):ax.plot(j,i,'o',ms=4,color=COLORS[m] if m in members else '#E2E7EC')
    ax.text(12.7,2.5,'Other\n'+str(len(remainder))+' sets',ha='center',va='center',size=9,color=MUTED)
    f.title('f4','Information types · per-paper distributions')
    ax=f.ax('f4',(8,7,2,8));mets=['historical_increment','cross_work_relation','scope_correction','cross_contribution']
    top=0
    for j,met in enumerate(mets):
        for i,m in enumerate(METHODS):
            vals=f.wide.xs(m,level='method')['information_'+met].dropna().to_numpy()
            x=j+(i-2.5)*.13;top=max(top,float(vals.max()))
            if len(np.unique(vals))>1:
                parts=ax.violinplot([vals],positions=[x],widths=.12,showextrema=False,bw_method=.35)
                for body in parts['bodies']:
                    body.set_facecolor(COLORS[m]);body.set_edgecolor(COLORS[m]);body.set_alpha(.55);body.set_linewidth(.4)
            else:
                ax.plot([x-.045,x+.045],[vals[0],vals[0]],color=COLORS[m],lw=1.5)
            ax.plot(x,np.median(vals),'o',ms=1.8,mfc='white',mec=COLORS[m],mew=.5,zorder=4)
    ax.set_ylim(-.2,max(3,top*1.1));ax.set_xlim(-.5,3.5)
    ax.set_xticks(range(4),['Historical\nincrement','Cross-work\nrelation','Scope\ncorrection','Cross-claim\nsynthesis'])
    ax.tick_params(labelsize=8,length=2);ax.set_ylabel('Clusters',fontsize=8)
    ax.grid(axis='y',color=GRID,lw=.5);ax.set_axisbelow(True)


def ribbon(ax: Axes,x0: float,y0: float,x1: float,y1: float,height: float,color: str) -> None:
    dx=(x1-x0)*.45
    verts=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+height),
           (x1-dx,y1+height),(x0+dx,y0+height),(x0,y0+height),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,
           MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts,codes),fc=color,alpha=.38,ec='none'))


def panel_g(f: Figure) -> None:
    records=[r for r in f.rows('h3') if r['count']>0]
    total=sum(r['count'] for r in records)
    f.title('g1',f'AB/BA judgment flows · {total} report pairs')
    ax=f.ax('g1',(1,9,1,1));ax.axis('off');ax.set_xlim(0,1.20);ax.set_ylim(1.04,-.06)
    states=['Full','tie','Opponent','cannot_judge']
    scale=.70/total;xs=[.28,.65,1.0];positions=[]
    for stage,(field,groups) in enumerate([('method',OTHERS),('x',states),('y',states)]):
        slots={};cursor=.07
        for group in groups:
            rows=[(i,r) for i,r in enumerate(records) if r[field]==group]
            n=sum(r['count'] for _,r in rows)
            if not n:continue
            height=n*scale
            color=COLORS[group] if stage==0 else {'Full':'#713BCB','tie':'#91A0AD','Opponent':'#E16C35','cannot_judge':'#BAC2CD'}[group]
            ax.add_patch(Rectangle((xs[stage]-.007,cursor),.014,height,fc=color,zorder=4))
            if stage==0:
                win=f.row('h1',group,'stable_full_win')['estimate']
                ax.text(xs[stage]-.022,cursor+height/2,NAMES[group]+f' · {n}'+chr(10)+f'{win:.0%} stable Full win',
                        ha='right',va='center',size=7.5,color=COLORS[group])
            else:
                label={'Full':'Full','tie':'Tie','Opponent':'Opponent','cannot_judge':'Unclear'}[group]
                ax.text(xs[stage]+.018,cursor+height/2,label+f' · {n}',ha='left',va='center',size=9,
                        bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1},zorder=5)
            offset=cursor
            for i,r in rows:
                slots[i]=offset;offset+=r['count']*scale
            cursor+=height+(.032 if stage==0 else .095)
        positions.append(slots)
    for i,r in enumerate(records):
        for stage in range(2):
            ribbon(ax,xs[stage]+.007,positions[stage][i],xs[stage+1]-.007,
                   positions[stage+1][i],r['count']*scale,COLORS[r['method']])
    for x,label in zip(xs,['Comparator','AB outcome','BA outcome']):
        ax.text(x,0,label,ha='center',va='bottom',size=10,weight='bold')

    f.title('g2','Source support · original vs altered')
    ax=f.ax('g2',(21,14,3,5))
    controls=[('source_mismatch','Source mismatch'),('firstness_overreach','Firstness overreach'),('scope_deletion','Scope deletion')]
    positions=[];labels=[]
    for i,(kind,label) in enumerate(controls):
        base=i*3.1
        n=f.row('h5',kind+'/original','supported')['denominator']
        ax.text(0,base-.57,f'{label} · {n} matched pairs',size=9,weight='bold')
        for j,version in enumerate(['original','altered']):
            yy=base+j;positions.append(yy);labels.append('Original' if j==0 else 'Altered');left=0
            for state,color in zip(STATES,STATE_COLORS):
                v=f.row('h5',kind+'/'+version,state)['estimate']
                ax.barh(yy,v,left=left,height=.62,color=color,edgecolor='white',linewidth=.6)
                if v>0:ax.text(left+v/2,yy,f'{v:.0%}',ha='center',va='center',size=8,color=INK)
                left+=v
    ax.set_yticks(positions,labels);ax.set_ylim(8.0,-1.0);ax.set_xlim(0,1)
    ax.set_xticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);ax.tick_params(length=0,labelsize=10)
    ax.spines[['left','top','right']].set_visible(False)
    x,y,w,h=f.box('g2')
    for i,(label,color) in enumerate(zip(['Supported','Partly supported','Not verifiable','Contradicted'],STATE_COLORS)):
        f.words(x+(i%2)*77,y+6+(i//2)*3.5,'■ '+label,9,color if i!=2 else '#718094')


def render(only: str | None = None) -> None:
    import cairosvg

    os.environ['FONTCONFIG_FILE']=str(OUT/'layouts/fonts.conf')
    f=Figure()
    for panel in [panel_a,panel_b,panel_c,panel_d,panel_e,panel_f,panel_g]:
        panel(f)
    master=OUT/'final/Fig3.svg'
    master.parent.mkdir(parents=True,exist_ok=True)
    f.fig.savefig(master,pad_inches=0)
    root=ET.parse(master).getroot()
    from figure_pipeline.fig3_extensions.insert_ad_into_fig3 import insert_replacements
    insert_replacements(root, f.layout)
    ET.ElementTree(root).write(master,encoding='utf-8',xml_declaration=True)
    exports=[master] if only is None else []
    for key,p in f.layout['panels'].items():
        if only is not None and key!=only:
            continue
        x,y,w,h=[p[k]*72/25.4 for k in ['x','y','w','h']]
        node=copy.deepcopy(root)
        node.set('viewBox',f'{x} {y} {w} {h}')
        node.set('width',f'{w}pt');node.set('height',f'{h}pt')
        path=OUT/'panels'/f'{key}.svg'
        path.parent.mkdir(parents=True,exist_ok=True)
        ET.ElementTree(node).write(path,encoding='utf-8',xml_declaration=True)
        exports.append(path)
    for path in exports:
        cairosvg.svg2pdf(url=str(path),write_to=str(path.with_suffix('.pdf')))
        cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')),scale=1.875)
        print('Wrote',path,flush=True)
    plt.close(f.fig)


if __name__=='__main__':
    render()

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

from figure_pipeline.fig1_reference.export import fonts

from .data import ASPECTS, CONDITIONS, FIELDS, OUT, SHORT, read, write

W, H = 324., 252.
PT = 72 / 25.4
INK, GREY, GRID = '#24282D', '#737B85', '#E4ECF2'
EDGE, HEADER, BG = '#A9CDE4', '#EEF6FC', '#FCFDFF'
COLORS = {'F': '#713BCB', 'E50': '#E16C35', 'K5': '#1269DF', 'LUNA': '#40A7BE'}
COVER = LinearSegmentedColormap.from_list('coverage', ['#F3F6F9', '#A7CBDD', '#286DA6'])
ERROR = LinearSegmentedColormap.from_list('errors', ['#FFF9F6', '#E7AB93', '#B04C34'])
REASON, UNNEEDED, UNKNOWN = '#4A9DA8', '#E9A15F', '#ADB6C1'
PANELS = {'a': (5, 22, 150, 60), 'b': (159, 22, 160, 60), 'c': (5, 87, 150, 90),
              'd': (159, 87, 160, 90), 'e': (5, 182, 150, 63), 'f': (159, 182, 160, 63)}
TITLES = {'a': 'Performance across research domains', 'b': 'Historical coverage and performance',
              'c': 'Paired effects of controlled perturbations', 'd': 'Abstention and unsupported assertions',
              'e': 'Observed variation under repeated generation', 'f': 'Analysis and report-generation cost'}
BOXES = {'a1': (8, 32, 85, 47), 'a2': (96, 32, 56, 47),
             'b1': (162, 32, 56, 47), 'b2': (219, 32, 31, 47), 'b3': (251, 32, 31, 47), 'b4': (283, 32, 33, 47),
             'c1': (8, 97, 46, 76), 'c2': (57, 97, 46, 76), 'c3': (106, 97, 46, 76),
             'd1': (162, 97, 88, 76), 'd2': (254, 97, 62, 76),
             'e1': (8, 192, 87, 49), 'e2': (99, 192, 53, 49),
             'f1': (162, 192, 89, 49), 'f2': (255, 192, 61, 49)}


class Figure:
    def __init__(self) -> None:
        for name in ['arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf']:
            font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/' + name)
        mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
            'xtick.labelsize': 7, 'ytick.labelsize': 7.5, 'axes.linewidth': .55,
            'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': GREY,
            'ytick.color': GREY, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
        self.fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
        self.serial = 0
        self.text('heading', W / 2, 3.8, 'Fig. 5 | Robustness, evidence boundaries and computational cost', 17, bold=True, ha='center')
        self.text('subtitle', W / 2, 13, '100-paper observational cohort · 20-paper paired stress test · 5 repeated runs · 84/85 reports available', 9.5, color=GREY, ha='center')
        for name, box in PANELS.items():
            x, y, w, _h = box
            self.card(name, box)
            self.rect(name, (x + .3, y + .3, w - .6, 7), HEADER)
            self.text(name, x + 2, y + 1, name, 14.5, bold=True)
            self.text(name, x + 9, y + 1.8, TITLES[name], 10.5, bold=True)
        self.text('footer', 6, 248, 'H: History · N: Neighborhood · J: Joint structure · S: Structural values · C: Citation contact    |    Paper-weighted rates; technical gaps are not zero.', 7.5, color=GREY)

    def text(self, group: str, x: float, y: float, text: str, size: float = 8,
             color: str = INK, bold: bool = False, **kwargs: Any) -> None:
        self.serial += 1
        self.fig.text(x / W, 1 - y / H, text, fontsize=size, va='top', color=color,
                      fontfamily='Times New Roman' if bold else 'Arial', weight='bold' if bold else 'normal',
                      linespacing=1.15, gid=f'{group}_text{self.serial}', **kwargs)

    def rect(self, group: str, box: tuple[float, float, float, float], color: Any) -> None:
        x, y, w, h = box
        self.serial += 1
        self.fig.add_artist(Rectangle((x/W, 1-(y+h)/H), w/W, h/H, transform=self.fig.transFigure,
                                     fc=color, ec='white', lw=.5, gid=f'{group}_rect{self.serial}'))

    def card(self, group: str, box: tuple[float, float, float, float]) -> None:
        x, y, w, h = box
        self.serial += 1
        self.fig.add_artist(FancyBboxPatch((x/W, 1-(y+h)/H), w/W, h/H,
             boxstyle='round,pad=0,rounding_size=.004', transform=self.fig.transFigure,
             fc=BG, ec=EDGE, lw=.65, zorder=-5, gid=f'{group}_card{self.serial}'))

    def ax(self, group: str, box: tuple[float, float, float, float]) -> Axes:
        x, y, w, h = box
        ax = self.fig.add_axes([x/W, 1-(y+h)/H, w/W, h/H])
        ax.set_gid(group + '_axes')
        ax.set_facecolor('none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=2, width=.5)
        return ax

    def data(self, name: str) -> Any:
        return read(OUT / 'data' / (name + '.json'))

    def scale(self, group: str, x: float, y: float, w: float, cmap: Any, maximum: int) -> None:
        ax = self.ax(group, (x, y, w, 1.8))
        ax.imshow(np.linspace(0, 1, 100)[None, :], extent=(0, maximum, 0, 1), cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set(yticks=[], xticks=[0, maximum/2, maximum]); ax.tick_params(length=0, labelsize=6.5)
        for s in ax.spines.values(): s.set_visible(False)


def panel_a(f: Figure) -> None:
    data = {(r['group'], r['aspect'], r['metric']): r for r in f.data('a_field_coverage')}
    labels = ['Life sciences\nN=40', 'Physical / engineering\nN=38', 'Medicine\nN=14', 'Earth / environment\nN=8']
    for i, label in enumerate(labels): f.text('a1', 8, 44 + i*7, label, 7.5)
    for group, x, metric, title in [('a1', 39, 'content_coverage', 'Correct content (%)'), ('a2', 97, 'grounded_coverage', 'Evidence-grounded (%)')]:
        f.text(group, x+26.5, 32, title, 9, bold=True, ha='center')
        for j, aspect in enumerate(ASPECTS):
            f.text(group, x+j*10.7+5.35, 38.5, 'HNJSC'[j], 8, bold=True, ha='center')
            for i, field in enumerate(FIELDS):
                r = data[field, aspect, metric]; value = r['mean']*100
                f.rect(group, (x+j*10.7, 43+i*7, 10.7, 7), COVER(value/100))
                f.text(group, x+j*10.7+5.35, 43.5+i*7, f'{value:.1f}', 7.6, color='white' if value>73 else INK, ha='center')
                f.text(group, x+j*10.7+5.35, 47+i*7, f'n={r["n"]}', 6.3, color='white' if value>73 else GREY, ha='center')
    f.text('a1', 8, 74, 'N: candidates; n: valid papers per cell', 7, color=GREY)
    f.scale('a2', 112, 74, 35, COVER, 100)


def panel_b(f: Figure) -> None:
    rows = f.data('b_stratum_summary')
    variables = ['sparse_claim_fraction', 'mean_nearest_similarity', 'readable_sources', 'fulltext_fraction']
    titles = ['Sparse claims', 'Similarity', 'Readable works', 'Full-text share']
    for j, (variable, title) in enumerate(zip(variables, titles)):
        group = f'b{j+1}'; x = 190 + j*31.7
        subset = [r for r in rows if r['variable'] == variable]; threshold = subset[0]['threshold']
        f.text(group, x+12.5, 32, title, 8.4, bold=True, ha='center')
        labels = ['0', '>0'] if j==0 else ['<1', '=1'] if j==3 else [f'≤{threshold:.3f}', f'>{threshold:.3f}'] if j==1 else [f'≤{threshold:g}', f'>{threshold:g}']
        for k, color in enumerate(['#8896A8', '#1269DF']):
            n = next(r['planned'] for r in subset if r['stratum']==k)
            f.text(group, x+12.5, 36.5+k*3.5, f'{labels[k]} · N={n}', 6.8, color=color, ha='center')
        ax = f.ax(group, (x, 47, 25, 23))
        ax.set(xlim=(70, 102), ylim=(4.6, -.6), xticks=[70, 85, 100], yticks=[])
        ax.spines['left'].set_visible(False); ax.grid(axis='x', color=GRID, lw=.4)
        for i, aspect in enumerate(ASPECTS):
            pair = [next(r for r in subset if r['aspect']==aspect and r['stratum']==k) for k in (0, 1)]
            ax.plot([r['mean']*100 for r in pair], [i-.1, i+.1], color='#B9C7D2', lw=.7)
            for k, r in enumerate(pair):
                color = '#8896A8' if k==0 else '#1269DF'
                assert r['low']*100 >= 70, 'Expand shared coverage axes to include interval'
                ax.plot([r['low']*100, r['high']*100], [i+(k-.5)*.2]*2, color=color, lw=.8)
                ax.scatter(r['mean']*100, i+(k-.5)*.2, s=12, marker='o' if k==0 else 'D',
                           facecolors='white' if k==0 else color, edgecolors=color, lw=.6, zorder=3)
            if j==0: f.text(group, 162, 47.5+i*4.4, SHORT[i], 7)
    f.text('b1', 162, 75, 'Grounded coverage (%) · mean [95% CI] · descriptive strata', 7, color=GREY)


def panel_c(f: Figure) -> None:
    effects = f.data('paired_summary'); papers = f.data('paired_differences')
    for j, condition in enumerate(['E50', 'K5', 'LUNA']):
        group = f'c{j+1}'; x = 8+j*49
        f.card(group, (x, 97, 46, 76))
        f.text(group, x+23, 98, f'{condition} − F', 10, color=COLORS[condition], bold=True, ha='center')
        detail = ['Half historical sources', 'Neighbor cap: 10 → 5', 'gpt-5.6-luna / medium'][j]
        f.text(group, x+23, 103, detail, 7.2, color=GREY, ha='center')
        ax = f.ax(group, (x+3, 110, 40, 50))
        ax.set(xlim=(-105, 25), ylim=(5, 0), xticks=[-100, -50, 0, 25], yticks=[])
        ax.spines['left'].set_visible(False); ax.axvline(0, color=GREY, lw=.65, ls=(0, (2, 2)))
        ax.grid(axis='x', color=GRID, lw=.4)
        for i, aspect in enumerate(ASPECTS):
            r = next(r for r in effects if r['condition']==condition and r['aspect']==aspect and r['metric']=='grounded_correct')
            vals = [r['delta']*100 for r in papers if r['condition']==condition and r['aspect']==aspect and r['metric']=='grounded_correct' and r['delta'] is not None]
            if i%2==0: ax.axhspan(i, i+1, color='#EFF4F8', zorder=-2)
            ax.text(.015, i+.07, SHORT[i], transform=ax.get_yaxis_transform(), fontsize=7.4, va='top', color=COLORS[condition], weight='bold')
            ax.text(.985, i+.07, f'n={r["n"]}', transform=ax.get_yaxis_transform(), fontsize=6.6, va='top', ha='right', color=GREY)
            for k, value in enumerate(vals): ax.scatter(value, i+.37+(k%5-2)*.028, s=6, color=COLORS[condition], alpha=.4, lw=0)
            ax.plot([r['low']*100, r['high']*100], [i+.58]*2, color=COLORS[condition], lw=1.3)
            ax.scatter(r['mean']*100, i+.58, marker='D', s=20, color=COLORS[condition], zorder=4)
            ax.text(.985, i+.79, f'{r["mean"]*100:+.1f} [{r["low"]*100:.1f}, {r["high"]*100:.1f}]',
                    transform=ax.get_yaxis_transform(), ha='right', va='top', fontsize=6.9)
        f.text(group, x+23, 166, 'Δ grounded coverage (pp)', 7.5, ha='center')
        f.text(group, x+23, 170, 'Mean [95% paper-bootstrap CI]', 6.7, color=GREY, ha='center')


def panel_d(f: Figure) -> None:
    data = {(r['condition'], r['aspect']): r for r in f.data('d_judgment_rates')}
    f.text('d1', 162, 98, 'Explicit abstention (%)', 9.4, bold=True)
    for j, (label, color) in enumerate([('Reasonable', REASON), ('Unnecessary', UNNEEDED), ('Unresolved', UNKNOWN)]):
        x = 162 + j*29
        f.rect('d1', (x, 104, 2, 2), color); f.text('d1', x+3, 104, label, 6.5)
    ax = f.ax('d1', (198, 111, 45, 50))
    ax.set(xlim=(0, 100), ylim=(5, 0), xticks=[0, 50, 100], yticks=[])
    ax.spines['left'].set_visible(False); ax.grid(axis='x', color=GRID, lw=.4, zorder=0)
    for i, aspect in enumerate(ASPECTS):
        f.text('d1', 162, 111.5+i*10, SHORT[i].replace(' ', '\n') if len(SHORT[i])>12 else SHORT[i], 7.2)
        for j, condition in enumerate(CONDITIONS):
            r = data[condition, aspect]; y=i+.13+j*.22
            ax.text(-.04, y, condition, ha='right', va='center', fontsize=6.2, transform=ax.get_yaxis_transform(), color=COLORS[condition])
            left = 0.
            for key, color in [('abstention_reasonable', REASON), ('abstention_unnecessary', UNNEEDED), ('abstention_unresolved', UNKNOWN)]:
                v=r[key]*100; ax.barh(y, v, height=.155, left=left, color=color, zorder=3); left+=v
            if left: ax.text(left+1, y, f'{left:.1f}', fontsize=5.8, va='center')
    f.text('d1', 162, 167, 'Same fixed applicable parts; omission is not abstention.', 6.8, color=GREY)
    f.text('d2', 254, 98, 'Confirmed unsupported (%)', 9.1, bold=True)
    f.text('d2', 254, 104, 'Cell labels: confirmed / unresolved', 6.8, color=GREY)
    for j, condition in enumerate(CONDITIONS): f.text('d2', 254+j*15.5+7.75, 109, condition, 7.5, color=COLORS[condition], ha='center', bold=True)
    for i, aspect in enumerate(ASPECTS):
        for j, condition in enumerate(CONDITIONS):
            r=data[condition, aspect]; v=r['unsupported_definitive_numerator']*100; u=r['unsupported_definitive_unresolved']*100
            x,y=254+j*15.5,114+i*9
            f.rect('d2',(x,y,15.5,9),ERROR(v/30))
            f.text('d2', x+7.75,y+1,f'{v:.1f} / {u:.1f}',7.1,ha='center',color='white' if v>23 else INK)
            f.text('d2', x+7.75,y+5.1,f'n={r["n"]}',6.2,ha='center',color='white' if v>23 else GREY)
    f.scale('d2', 271, 163, 34, ERROR, 30)
    f.text('d2', 254, 171, 'Unresolved is not confirmed absence.', 6.8, color=GREY)


def panel_e(f: Figure) -> None:
    rows=f.data('e_repeat_paper_differences'); summary=f.data('repeat_variation')
    ids=f.data('cohort')['repeated']; markers=['o','s','^','D','v']
    f.text('e1', 8, 192, 'F_REPEAT − F · grounded coverage',9.3,bold=True)
    ax=f.ax('e1',(37,202,55,27))
    ax.set(xlim=(-30,5),ylim=(4.6,-.6),xticks=[-25,0],yticks=range(5),yticklabels=SHORT)
    ax.axvline(0,color=GREY,lw=.65,ls=(0,(2,2))); ax.spines['left'].set_visible(False)
    for k,paper in enumerate(ids):
        for i,aspect in enumerate(ASPECTS):
            r=next(r for r in rows if r['paper_id']==paper and r['aspect']==aspect and r['metric']=='grounded_correct')
            if r['delta'] is not None: ax.scatter(r['delta']*100,i+(k-2)*.12,marker=markers[k],s=16,facecolor='white',edgecolor=COLORS['F'],lw=.8)
    f.text('e1', 64, 234, 'Paired difference (pp)',7.5,ha='center')
    f.text('e1', 8, 239, 'Five papers; citation contact n=4. Symbols identify papers.',6.9,color=GREY)
    f.text('e2', 99, 192, 'Other changes · observed range (pp)',8.7,bold=True)
    f.text('e2', 116, 198, 'Abstention',7.3,ha='center'); f.text('e2', 140, 198, 'Unsupported',7.3,ha='center')
    for i,aspect in enumerate(ASPECTS):
        f.text('e2',100,204+i*5,'HNJSC'[i],7.2,bold=True)
        for j,metric in enumerate(['explicit_abstention','unsupported_definitive']):
            r=next(r for r in summary if r['aspect']==aspect and r['metric']==metric)
            label=f'{r["minimum"]*100:+.0f} to {r["maximum"]*100:+.0f}' if r['minimum']!=r['maximum'] else f'{r["minimum"]*100:.0f}'
            f.text('e2',116+j*24,204+i*5,label,7.5,ha='center')
    f.text('e2', 99, 234, 'Observed points and ranges only.\nNo CI or significance threshold.',7,color=GREY)


def panel_f(f: Figure) -> None:
    rows=f.data('f_cost_ratios'); summaries=f.data('f_cost_summary')
    for group, x, width, metrics, title in [('f1',180,66,['input_tokens','output_tokens'],'Token usage · ratio to F'),
                                          ('f2',265,49,['call_seconds_sum'],'Summed call time · ratio to F')]:
        f.text(group, x+width/2,192,title,9,bold=True,ha='center')
        f.text(group,x+width/2,198,'Input: open circle · Output: filled diamond' if group=='f1' else 'Median; line: interquartile range',7,ha='center',color=GREY)
        vals=[r['ratio'] for r in rows if r['metric'] in metrics and r['ratio'] is not None]
        upper=max(1.25,np.ceil(max(vals)*4)/4+.05)
        ax=f.ax(group,(x,207,width,22))
        ax.set(xlim=(0,upper),ylim=(2.6,-.6),xticks=[0,.5,1]+([1.5] if upper>1.5 else []),yticks=[])
        ax.axvline(1,color=GREY,lw=.65,ls=(0,(2,2)));ax.spines['left'].set_visible(False)
        for i,condition in enumerate(['E50','K5','LUNA']):
            if group=='f1': f.text(group,162,208+i*7,condition,8.5,color=COLORS[condition],bold=True)
            for j,metric in enumerate(metrics):
                yy=i+(j-.5)*.25 if len(metrics)==2 else i
                values=[r['ratio'] for r in rows if r['condition']==condition and r['metric']==metric and r['ratio'] is not None]
                r=next(r for r in summaries if r['condition']==condition and r['metric']==metric)
                ax.scatter(values,[yy+(k%5-2)*.025 for k in range(len(values))],s=6,color=COLORS[condition],alpha=.35,lw=0)
                ax.plot([r['q1'],r['q3']],[yy]*2,lw=1.2,color=COLORS[condition])
                ax.scatter(r['median'],yy,s=25,marker='o' if metric=='input_tokens' else 'D',
                           facecolor='white' if metric=='input_tokens' else COLORS[condition],edgecolor=COLORS[condition],zorder=5)
            medians = [next(r['median'] for r in summaries if r['condition']==condition and r['metric']==metric)
                       for metric in metrics]
            label = ' / '.join(f'{value:.2f}' for value in medians)
            ax.text(.99, i+.25, label, ha='right', va='top', transform=ax.get_yaxis_transform(), fontsize=6.5)
    f.text('f1',162,235,'Complete cost pairs: E50 n=19; K5 / LUNA n=20.',7,color=GREY)
    f.text('f2',255,235,'Fixed-input generation only.\nCall-time sum ≠ elapsed wall time.',7,color=GREY)


def crop_svg(source: Path, destination: Path, box: tuple[float,float,float,float], prefixes: list[str]) -> None:
    root=ET.parse(source).getroot(); figure=root.find('{http://www.w3.org/2000/svg}g[@id="figure_1"]')
    for child in list(figure):
        gid=child.get('id','')
        if gid!='patch_1' and not any(gid.startswith(p+'_') for p in prefixes): figure.remove(child)
    x,y,w,h=box; x-=.5;y-=.5;w+=1;h+=1
    root.set('viewBox',f'{x*PT} {y*PT} {w*PT} {h*PT}');root.set('width',f'{w}mm');root.set('height',f'{h}mm')
    destination.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)


def render() -> None:
    import cairosvg
    fonts(OUT); f=Figure()
    for draw in [panel_a,panel_b,panel_c,panel_d,panel_e,panel_f]: draw(f)
    final=OUT/'final';final.mkdir(parents=True,exist_ok=True)
    master=final/'Fig5.svg';outlined=final/'Fig5_outlined.svg'
    f.fig.savefig(master)
    with mpl.rc_context({'svg.fonttype':'path'}):f.fig.savefig(outlined)
    f.fig.savefig(final/'Fig5.pdf')
    for name,dpi in [('Fig5.png',300),('Fig5_600dpi.png',600),('Fig5_900dpi.png',900)]:
        cairosvg.svg2png(url=str(master),write_to=str(final/name),output_width=round(W/25.4*dpi),output_height=round(H/25.4*dpi))
    for directory,boxes in [('panels',PANELS),('components',BOXES)]:
        for name,box in boxes.items():
            path=OUT/directory/f'{name}.svg'
            prefixes=[name]+[k for k in BOXES if k.startswith(name)] if directory=='panels' else [name]
            crop_svg(master,path,box,prefixes);crop_svg(outlined,path.with_name(name+'_outlined.svg'),box,prefixes)
            cairosvg.svg2pdf(url=str(path),write_to=str(path.with_suffix('.pdf')))
            cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')),scale=3)
    write(OUT/'layouts/style.json',{'canvas_mm': [W,H],'panels': PANELS,'components': BOXES,'reference': 'fig4_reference',
          'coverage_scale': [0,100],'coverage_strata_axis': [70,102],'paired_scale': [-105,25],'colors': COLORS})
    plt.close(f.fig)
    print(f'Rendered {master}',flush=True)

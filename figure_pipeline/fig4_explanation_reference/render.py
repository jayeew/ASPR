"""Millimetre-positioned reference cards with editable vector exports."""
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
from .data import ASPECTS, NAMES, ORDER, OUT, read, write

W, H = 324., 181.
PT = 72 / 25.4
INK, GREY, GRID = '#24282D', '#737B85', '#E4ECF2'
EDGE, HEADER, BG = '#A9CDE4', '#EEF6FC', '#FCFDFF'
ORANGE, BLUE, PURPLE, SLATE, TEAL = '#E16C35', '#1269DF', '#713BCB', '#8896A8', '#40A7BE'
LOST, RETAINED, PALE = '#C56154', '#A7CADD', '#EDF0F3'
COVER = LinearSegmentedColormap.from_list('coverage', ['#F3F6F9', '#A7CBDD', '#286DA6'])
PANELS = {'a': (5, 22, 150, 74), 'b': (159, 22, 160, 74),
          'c': (5, 101, 99, 75), 'd': (108, 101, 104, 75),
          'e': (216, 101, 103, 32), 'f': (216, 138, 103, 38)}
TITLES = {'a': 'Configurations and capability profiles',
          'b': 'Paired branch and component effects',
          'c': 'Correct content retained, gained or lost',
          'd': 'Joint graph: cross-card structure',
          'e': 'Structural values', 'f': 'Citation paths'}
BOXES = {'a1': (8, 32, 49, 61), 'a2': (61, 32, 90, 61),
         'b1': (162, 32, 49, 61), 'b2': (215, 32, 49, 61), 'b3': (268, 32, 48, 61),
         'c1': (8, 111, 93, 62), 'd1': (111, 111, 98, 39), 'd2': (111, 151, 98, 23),
         'e2': (219, 111, 97, 20), 'f1': (219, 148, 97, 13), 'f2': (219, 163, 97, 11)}
SHORT = ['History', 'Neighborhood', 'Joint structure', 'Structural values', 'Citation contact']


class Figure:
    def __init__(self) -> None:
        for name in ['arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf']:
            font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/' + name)
        mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
            'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': .55,
            'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': GREY,
            'ytick.color': GREY, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
        self.fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
        self.serial = 0
        self.data = read(OUT / 'data/snapshot.json')
        self.abs = {(r['condition'], r['aspect']): r for r in self.data['absolute']
                    if r['metric'] == 'content_coverage'}
        self.eff = {(r['other'], r['aspect']): r for r in self.data['effects']
                    if r['metric'] == 'content_delta'}
        self.metrics = {(r['paper_id'], r['condition'], r['aspect']): r for r in self.data['metrics']}
        self.text('heading', W / 2, 4, 'Fig. 4 | Complementary branch contributions and graph component effects',
                  17, bold=True, ha='center')
        for name, box in PANELS.items():
            x, y, w, h = box
            self.card(name + '_card', box)
            self.rect(name + '_header', (x + .3, y + .3, w - .6, 7), HEADER)
            self.text(name + '_title', x + 2, y + .9, name, 15, bold=True)
            self.text(name + '_title', x + 10, y + 1.7, TITLES[name], 10 if name == 'c' else 11, bold=True)

    def text(self, group: str, x: float, y: float, text: str, size: float = 8,
             color: str = INK, bold: bool = False, **kwargs: Any) -> None:
        self.serial += 1
        self.fig.text(x / W, 1 - y / H, text, fontsize=size, va='top', color=color,
                      fontfamily='Times New Roman' if bold else 'Arial',
                      weight='bold' if bold else 'normal', linespacing=1.22,
                      gid=f'{group}_text{self.serial}', **kwargs)

    def rect(self, group: str, box: tuple[float, float, float, float], color: str,
             edge: str = 'none', **kwargs: Any) -> None:
        x, y, w, h = box
        self.serial += 1
        self.fig.add_artist(Rectangle((x / W, 1 - (y + h) / H), w / W, h / H,
                            transform=self.fig.transFigure, facecolor=color,
                            edgecolor=edge, linewidth=.6, gid=f'{group}_rect{self.serial}', **kwargs))

    def card(self, group: str, box: tuple[float, float, float, float], color: str = BG,
             edge: str = EDGE) -> None:
        x, y, w, h = box
        self.serial += 1
        self.fig.add_artist(FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H,
            boxstyle='round,pad=0,rounding_size=0.004', transform=self.fig.transFigure,
            facecolor=color, edgecolor=edge, linewidth=.65, zorder=-5, gid=f'{group}_card{self.serial}'))

    def ax(self, group: str, box: tuple[float, float, float, float]) -> Axes:
        x, y, w, h = box
        ax = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        ax.set_gid(group + '_axes')
        ax.set_facecolor('none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=2, width=.5)
        return ax

    def scale(self, group: str, x: float, y: float, w: float = 25) -> None:
        ax = self.ax(group, (x, y, w, 1.8))
        for i in range(100):
            ax.add_patch(Rectangle((i, 0), 1.01, 1, facecolor=COVER(i/99), edgecolor='none'))
        ax.set(xlim=(0, 100), ylim=(0, 1), yticks=[], xticks=[0, 50, 100]); ax.tick_params(labelsize=6.8, length=0)
        for spine in ax.spines.values(): spine.set_visible(False)


def panel_a(f: Figure) -> None:
    f.card('a1', BOXES['a1']); f.card('a2', BOXES['a2'])
    f.text('a1', 32.5, 33, 'Controlled inputs', 9.5, bold=True, ha='center')
    ax = f.ax('a1', (9, 39, 47, 44)); ax.axis('off'); ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlim(0, 1.06)
    nodes = {'T': (.22, .12), 'E': (.22, .85), 'G': (.76, .12), 'F': (.76, .85)}
    colors = {'T': GREY, 'E': ORANGE, 'G': BLUE, 'F': PURPLE}
    labels = {'T': 'Paper\nonly', 'E': 'GEAR\nonly', 'G': 'Graph\nonly', 'F': 'Full\nsystem'}
    for key, (x, y) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - .19, y - .105), .38, .21,
            boxstyle='round,pad=0,rounding_size=.018', ec=colors[key],
            fc=mpl.colors.to_rgba(colors[key], .09), lw=.8))
        ax.text(x, y, labels[key], ha='center', va='center', fontsize=8.6,
                color=colors[key], weight='bold', linespacing=1.05)
    for left, right, col in [('T', 'E', ORANGE), ('T', 'G', BLUE), ('E', 'F', BLUE)]:
        x1, y1 = nodes[left]; x2, y2 = nodes[right]
        dx, dy = (.205, 0) if y1 == y2 else (0, .12)
        ax.annotate('', (x2-dx, y2-dy), (x1+dx, y1+dy), arrowprops=dict(arrowstyle='-|>', lw=1, color=col))
    ax.text(.02, .49, '+ GEAR', rotation=90, fontsize=7.7, color=ORANGE, va='center')
    ax.text(.5, .99, '+ Graph', fontsize=7.7, color=BLUE, ha='center')
    ax.plot([.965, 1.035, 1.035], [.12, .12, .85], color=ORANGE, lw=.8)
    ax.annotate('', (.955, .85), (1.035, .85), arrowprops=dict(arrowstyle='-|>', lw=.8, color=ORANGE))
    for y, label in [(.64, 'Without joint graph'), (.48, 'Without structural\nvalues'), (.32, 'Without citation paths')]:
        ax.add_patch(FancyBboxPatch((.29, y-.065), .70, .13,
            boxstyle='round,pad=0,rounding_size=.012', fc='#F0F3F7', ec='#BCC9D4', lw=.6))
        ax.text(.64, y, label, ha='center', va='center', fontsize=7, linespacing=1.0)
    f.text('a2', 106, 33, 'Correct scientific-part coverage (%)', 9.3, bold=True, ha='center')
    ax = f.ax('a2', (62, 39, 88, 42)); ax.axis('off'); ax.set(xlim=(0, 8.2), ylim=(7.1, -1.2))
    cols = ['History\ncheck', 'Neighbor-\nhood', 'Joint\nstructure', 'Structural\nvalues', 'Citation\ncontact']
    for j, label in enumerate(cols):
        ax.text(3.25+j, -.75, label, fontsize=7.1, ha='center', va='center', linespacing=1.05)
    for i, condition in enumerate(ORDER):
        label = NAMES[condition].replace('Without structural values', 'Without structural\nvalues')
        label = label.replace('Without citation paths', 'Without citation\npaths')
        ax.add_patch(Rectangle((0, i), 2.75, 1, fc='#EDE7F6' if condition == 'F' else '#F0F3F6', ec='white', lw=.6))
        ax.text(.06, i+.5, label, fontsize=7.5, va='center', linespacing=1.0,
                weight='bold' if condition == 'F' else 'normal')
        for j, aspect in enumerate(ASPECTS):
            value = f.abs[(condition, aspect)]['estimate'] * 100
            ax.add_patch(Rectangle((2.75+j, i), 1, 1, fc=COVER(value/100), ec='white', lw=.6))
            ax.text(3.25+j, i+.5, f'{value:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if value > 73 else INK)
        if condition == 'F': ax.add_patch(Rectangle((0, i), 7.75, 1, fill=False, ec=PURPLE, lw=1.05))
    f.text('a2', 64, 82.5, 'Common n: 98 / 98 / 98 / 98 / 74', 7.8)
    f.scale('a2', 121, 87, 26)


def effect_strip(f: Figure, group: str, other: str, color: str,
                 aspects: list[str], box: tuple[float, float, float, float]) -> None:
    ax = f.ax(group, box)
    n = len(aspects)
    ax.set(xlim=(-55, 105), ylim=(n, 0), yticks=[], xticks=[-50, 0, 50, 100])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.axvline(0, color='#8A929B', lw=.7, ls=(0, (2, 2)), zorder=1)
    ax.grid(axis='x', color=GRID, lw=.45)
    for i, aspect in enumerate(aspects):
        comp = other if other else {'joint_contribution':'F_noJ','structural_resolution':'F_noM','citation_contact':'F_noP'}[aspect]
        hue = color if other else {'F_noJ':PURPLE, 'F_noM':SLATE, 'F_noP':TEAL}[comp]
        r = f.eff[(comp, aspect)]
        if i % 2 == 0: ax.axhspan(i, i+1, facecolor='#EDF4F9', alpha=.8, zorder=0)
        values = np.array([p['content_delta']*100 for p in f.data['pairs']
            if p['other'] == comp and p['aspect'] == aspect and p['complete_pair']])
        for value in np.unique(values):
            k = int(np.sum(values == value))
            ys = i + .38 + (np.arange(k) % 6) * .034
            xs = value + (np.arange(k) // 6 - (int((k-1)//6))/2) * .4
            ax.scatter(xs, ys, s=5, c=hue, alpha=.36, linewidths=0, zorder=2)
        ax.plot([r['low']*100, r['high']*100], [i+.67]*2, color=hue, lw=1.3, zorder=3)
        ax.scatter([r['estimate']*100], [i+.67], marker='D', s=18, color=hue, edgecolor='white', linewidth=.4, zorder=4)
        label = SHORT[ASPECTS.index(aspect)] if other else {'F_noJ':'Joint graph', 'F_noM':'Structural values', 'F_noP':'Citation paths'}[comp]
        ax.text(-52, i+.12, label, va='center', fontsize=7.8, color=hue, weight='bold')
        ax.text(103, i+.12, f'n={r["n"]}', ha='right', va='center', fontsize=6.8, color=GREY)
        ax.text(103, i+.89, f'{r["estimate"]*100:+.1f} [{r["low"]*100:.1f}, {r["high"]*100:.1f}]',
                ha='right', va='center', fontsize=7.1)
    ax.tick_params(axis='x', labelsize=7.3, pad=2)


def panel_b(f: Figure) -> None:
    for group in ['b1', 'b2', 'b3']: f.card(group, BOXES[group])
    for group, x, title, subtitle in [
        ('b1',186.5,'Add GEAR','Full system − Graph only'),
        ('b2',239.5,'Add Graph','Full system − GEAR only'),
        ('b3',292,'Restore graph component','Full system − corresponding deletion')]:
        f.text(group, x, 33, title, 9.3, bold=True, ha='center')
        f.text(group, x, 37.5, subtitle, 6.9, ha='center', color=GREY)
    effect_strip(f, 'b1', 'G', ORANGE, ASPECTS, (164, 42, 45, 43))
    effect_strip(f, 'b2', 'E', BLUE, ASPECTS, (217, 42, 45, 43))
    effect_strip(f, 'b3', '', PURPLE, ASPECTS[2:], (270, 42, 44, 43))


def panel_c(f: Figure) -> None:
    f.text('c1', 9, 111, 'Scientific parts in complete report pairs', 9.2, bold=True)
    ax = f.ax('c1', (9, 120, 91, 44)); ax.axis('off')
    ax.set(xlim=(-.70, 1.15), ylim=(5, 0))
    states = [('retained', RETAINED), ('gained', PURPLE), ('lost', LOST), ('neither_correct', PALE)]
    for i, aspect in enumerate(ASPECTS):
        row_label = ['History', 'Neighbor-\nhood', 'Joint\nstructure', 'Structural\nvalues', 'Citation\ncontact'][i]
        ax.text(-.69, i+.49, row_label, fontsize=7.1, va='center', linespacing=1.0)
        for j, other in enumerate(['G', 'E']):
            r = next(r for r in f.data['transitions'] if r['other'] == other and r['aspect'] == aspect)
            y = i+.28+j*.42
            ax.text(-.025, y, 'Graph' if other == 'G' else 'GEAR', color=BLUE if other == 'G' else ORANGE,
                    fontsize=7, ha='right', va='center')
            left = 0.
            for state, color in states:
                width = r[state] / r['total']
                ax.add_patch(Rectangle((left,y-.17),width,.34,fc=color,ec='white',lw=.25))
                if width > .115:
                    ax.text(left+width/2,y,str(r[state]),fontsize=6.8,ha='center',va='center',
                            color='white' if state in ['gained','lost'] else INK)
                left += width
            ax.text(1.02, y, str(r['total']), fontsize=6.6, va='center', color=GREY)
    f.text('c1', 98, 116, 'N', 7, color=GREY, ha='right')
    for x, label, color in [(9,'Retained',RETAINED),(31,'Gained',PURPLE),(52,'Lost',LOST),(69,'Neither correct',PALE)]:
        f.rect('c1',(x,169,2.2,2.2),color)
        f.text('c1',x+3,168.8,label,7)


def coverage_tiles(f: Figure, group: str, papers: list[str], conditions: list[str],
                   aspect: str, box: tuple[float, float, float, float]) -> None:
    ax = f.ax(group, box); ax.axis('off')
    ax.set(xlim=(0,len(papers)),ylim=(len(conditions),0))
    for y, condition in enumerate(conditions):
        for x, paper in enumerate(papers):
            r = f.metrics[(paper,condition,aspect)]
            v = r['content_coverage'] if r['state'] == 'completed' else None
            ax.add_patch(Rectangle((x,y),.96,.88,facecolor=COVER(v) if v is not None else 'white',
                edgecolor='white' if v is not None else GREY,lw=.3,hatch=None if v is not None else '////'))


def panel_d(f: Figure) -> None:
    facts = sorted(f.data['facts'], key=lambda r:(r['historical_edges_only_in_joint'],r['paper_id']))
    n_extra = sum(r['historical_edges_only_in_joint'] > 0 for r in facts)
    n_merge = sum(r['components_from_single_cards'] > r['components_with_joint'] for r in facts)
    f.text('d1',111,111,'Historical edges visible only in the joint graph',8.7,bold=True)
    f.text('d1',111,115,f'{n_extra}/100 with extra edges · {n_merge}/100 merge more components',7.1)
    ax=f.ax('d1',(133,123,75,22))
    values=np.array([r['historical_edges_only_in_joint'] for r in facts])
    ax.vlines(np.arange(100)+.48,0,values,color='#C7D9EA',lw=.55)
    ax.scatter(np.arange(100)+.48,values,s=10,c=values,
        cmap=LinearSegmentedColormap.from_list('edges',[BLUE,PURPLE,ORANGE]),linewidths=0)
    ax.set(xlim=(0,100),ylim=(-.5,21),xticks=[.48,49.48,99.48],xticklabels=['1','50','100'],yticks=[0,10,20])
    ax.grid(axis='y',color=GRID,lw=.45);ax.tick_params(labelsize=7,pad=1)
    papers=[r['paper_id'] for r in facts]
    f.text('d2',111,153,'Full system',7.2,color=PURPLE)
    f.text('d2',111,158,'Without joint',7.2)
    coverage_tiles(f,'d2',papers,['F','F_noJ'],ASPECTS[2],(133,152.5,75,9))
    r=f.eff[('F_noJ',ASPECTS[2])]
    full=f.abs[('F',ASPECTS[2])]['estimate']*100
    other=f.abs[('F_noJ',ASPECTS[2])]['estimate']*100
    f.text('d2',111,165,f'{full:.1f}% vs {other:.1f}% · Δ {r["estimate"]*100:+.1f} pp · n={r["n"]}',8.3,bold=True)
    f.scale('d2',175,170,23)
    f.rect('d2',(202,170,2,2),'white',edge=GREY,hatch='////')
    f.text('d2',205,170,'NA',6.8)


def panel_e(f: Figure) -> None:
    f.text('e2',219,111,'Quantitative-structure coverage (%) · n=98',8.7,bold=True)
    ax=f.ax('e2',(255,116,59,8))
    ax.set(xlim=(0,100),ylim=(1.5,-.5),yticks=[0,1],
           yticklabels=['Full system','Without structural values'],xticks=[])
    ax.spines[:].set_visible(False);ax.tick_params(length=0,labelsize=7.5)
    for y,c,col in [(0,'F',PURPLE),(1,'F_noM',SLATE)]:
        v=f.abs[(c,ASPECTS[3])]['estimate']*100
        ax.barh(y,v,height=.52,color=col)
        ax.text(v-2 if v>30 else v+3,y,f'{v:.2f}%',va='center',
                ha='right' if v>30 else 'left',color='white' if v>30 else INK,fontsize=8)
    r=f.eff[('F_noM',ASPECTS[3])]
    f.text('e2',219,128,f'Δ {r["estimate"]*100:+.2f} pp [{r["low"]*100:.2f}, {r["high"]*100:.2f}]',8.2,bold=True)


def panel_f(f: Figure) -> None:
    facts=f.data['facts'];n=sum(r['citation_opportunity'] for r in facts)
    ax=f.ax('f1',(219,148,97,4));ax.axis('off');ax.set(xlim=(0,100),ylim=(0,1))
    for left,width,color,label in [(0,n,TEAL,f'{n} applicable'),(n,100-n,PALE,f'{100-n} not applicable')]:
        ax.add_patch(Rectangle((left,0),width,1,facecolor=color,edgecolor='white'))
        ax.text(left+width/2,.5,label,ha='center',va='center',fontsize=7.5,color='white' if left==0 else INK)
    r=f.eff[('F_noP',ASPECTS[4])]
    full=f.abs[('F',ASPECTS[4])]['estimate']*100;other=f.abs[('F_noP',ASPECTS[4])]['estimate']*100
    f.text('f1',219,155,f'{full:.2f}% vs {other:.2f}% · Δ {r["estimate"]*100:+.2f} pp [{r["low"]*100:.2f}, {r["high"]*100:.2f}]',7.8,bold=True)
    papers=sorted(r['paper_id'] for r in facts if r['citation_opportunity'])
    f.text('f2',219,164,'Full system',7.3,color=PURPLE)
    f.text('f2',219,168.5,'Without paths',7.3)
    coverage_tiles(f,'f2',papers,['F','F_noP'],ASPECTS[4],(243,164,73,8))


def crop_svg(source: Path, destination: Path, box: tuple[float,float,float,float],
             prefixes: list[str]) -> None:
    root=ET.parse(source).getroot()
    ns={'s':'http://www.w3.org/2000/svg'}
    figure=root.find('s:g[@id="figure_1"]',ns)
    for child in list(figure):
        gid=child.get('id','')
        if gid!='patch_1' and not any(gid.startswith(p+'_') for p in prefixes):
            figure.remove(child)
    x,y,w,h=box;x-=.8;y-=.8;w+=1.6;h+=1.6
    root.set('viewBox',f'{x*PT} {y*PT} {w*PT} {h*PT}')
    root.set('width',f'{w}mm');root.set('height',f'{h}mm')
    destination.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)


def render() -> None:
    import cairosvg
    fonts(OUT)
    f=Figure()
    for draw in [panel_a,panel_b,panel_c,panel_d,panel_e,panel_f]: draw(f)
    final=OUT/'final';final.mkdir(parents=True,exist_ok=True)
    master=final/'Fig4.svg';outlined=final/'Fig4_outlined.svg'
    f.fig.savefig(master)
    with mpl.rc_context({'svg.fonttype':'path'}):f.fig.savefig(outlined)
    f.fig.savefig(final/'Fig4.pdf')
    # Explicitly sized 300/600/900-dpi PNGs; every scientific mark stays vector in SVG/PDF.
    for name,dpi in [('Fig4.png',300),('Fig4_600dpi.png',600),('Fig4_900dpi.png',900)]:
        cairosvg.svg2png(url=str(master),write_to=str(final/name),
                       output_width=round(W/25.4*dpi),output_height=round(H/25.4*dpi))
    for name,box in PANELS.items():
        path=OUT/'panels'/f'{name}.svg'
        prefixes=[name]+[k for k in BOXES if k.startswith(name)]
        crop_svg(master,path,box,prefixes)
        crop_svg(outlined,path.with_name(name+'_outlined.svg'),box,prefixes)
        cairosvg.svg2pdf(url=str(path),write_to=str(path.with_suffix('.pdf')))
        cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')),scale=4)
    for name,box in BOXES.items():
        path=OUT/'components'/f'{name}.svg'
        crop_svg(master,path,box,[name])
        crop_svg(outlined,path.with_name(name+'_outlined.svg'),box,[name])
        cairosvg.svg2pdf(url=str(path),write_to=str(path.with_suffix('.pdf')))
        cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')),scale=4)
    write(OUT/'layouts/style.json',dict(canvas_mm=[W,H],panels=PANELS,components=BOXES,
          reference='fig4_reference',coverage_scale=[0,100],paired_scale=[-50,100]))
    plt.close(f.fig)
    print(f'Rendered {master}',flush=True)

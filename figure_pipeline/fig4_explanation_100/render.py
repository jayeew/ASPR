from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from figure_pipeline.fig3_revision.storage import read

from .models import ASPECTS, CONDITIONS, Config

LABELS = ['Paper only', 'GEAR only', 'Graph only', 'Full system', 'Without joint graph', 'Without structural values', 'Without citation paths']
ASPECT_LABELS = ['History', 'Knowledge\nposition', 'Joint\nstructure', 'Structural\ncontrast', 'Citation\ncontact']
COLORS = ['#94A1AF', '#E7833B', '#367ACA', '#8051B6', '#709BAF', '#70AAA7', '#ABA0C7']


def style() -> None:
    for name in ('arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf'):
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#ABBFCF', 'text.color': '#243547',
                         'xtick.color': '#556A7A', 'ytick.color': '#556A7A', 'pdf.fonttype': 42})


def card(fig: Any, box: tuple[float, float, float, float], title: str) -> Axes:
    x, y, w, h = box
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.007', transform=fig.transFigure,
                                 facecolor='#FBFDFF', edgecolor='#B5D3E7', linewidth=1, zorder=-10))
    fig.text(x + .008, y + h - .032, title, fontsize=11, weight='bold', fontfamily='Times New Roman')
    return fig.add_axes([x + .12 * w, y + .18 * h, w * .82, h * .66])


def matrix(ax: Axes, values: np.ndarray, columns: list[str], rows: list[str], percentages: bool) -> None:
    ax.imshow(values, cmap='Purples' if percentages else 'Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(columns)), columns, fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    for label, color in zip(ax.get_yticklabels(), COLORS):
        label.set_color(color)
    ax.tick_params(length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text = 'NA' if np.isnan(val) else f'{val * 100:.0f}' if percentages else '●' if val else '—'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if val > .6 else '#344859')
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel(ax: Axes, panel: str, data: dict[str, Any]) -> None:
    summaries, effects, paper = data['summary'], data['effects'], data['paper']
    if panel == 'a':
        matrix(ax, np.array([[1,0,0,0,0,0], [1,1,0,0,0,0], [1,0,1,1,1,1], [1,1,1,1,1,1],
                            [1,1,1,0,1,1], [1,1,1,1,0,1], [1,1,1,1,1,0]], dtype=float),
               ['Paper', 'GEAR', 'Graph', 'Joint', 'Values', 'Paths'], LABELS, False)
    elif panel == 'b':
        index = {(r['condition'], r['aspect']): r for r in summaries if r['metric'] == 'content_coverage'}
        values = np.array([[index.get((c, a), {}).get('estimate') for a in ASPECTS] for c in CONDITIONS], dtype=float)
        columns = [name + f"\nn={index.get(('F', a), {}).get('n', 0)}" for a, name in zip(ASPECTS, ASPECT_LABELS)]
        matrix(ax, values, columns, LABELS, True)
    elif panel == 'c':
        for other, offset, color in [('G', -.13, COLORS[1]), ('E', .13, COLORS[2])]:
            for i, aspect in enumerate(ASPECTS):
                row = next((r for r in effects if r['other'] == other and r['aspect'] == aspect and r['metric'] == 'content_delta'), None)
                if row and row['estimate'] is not None:
                    v, lo, hi = [row[k] * 100 for k in ('estimate', 'low', 'high')]
                    ax.errorbar(v, i + offset, xerr=[[v-lo], [hi-v]], fmt='o', color=color, capsize=3, ms=5)
        ax.axvline(0, color='#9DAFBC', lw=.8)
        ax.set_yticks(range(5), [a.replace('\n', ' ') for a in ASPECT_LABELS], fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel('Full − single branch (percentage points)')
        ax.plot([], [], 'o', color=COLORS[1], label='Add GEAR'); ax.plot([], [], 'o', color=COLORS[2], label='Add Graph')
        ax.legend(frameon=False, fontsize=7, loc='lower right')
    else:
        aspect, other = {'d': ('joint_contribution', 'F_noJ'), 'e': ('structural_resolution', 'F_noM'), 'f': ('citation_contact', 'F_noP')}[panel]
        ids = data['ids']
        idx = {(r['paper_id'], r['condition'], r['aspect']): r for r in paper}
        for i, ident in enumerate(ids):
            values = [idx.get((ident, c, aspect), {}).get('content_coverage') for c in ('F', other)]
            if all(v is not None for v in values):
                ax.plot([i-.12, i+.12], [v*100 for v in values], color='#CCD5DF', lw=1.5)
            for j, (value, color) in enumerate(zip(values, [COLORS[3], COLORS[4 + 'def'.index(panel)]])):
                if value is not None:
                    ax.scatter(i + (-.12 if j == 0 else .12), value*100, color=color, s=35, zorder=3)
                else:
                    ax.text(i + (-.12 if j == 0 else .12), -12, 'NA', ha='center', fontsize=6)
        ax.set_xlim(-.5, len(ids) - .5)
        ax.set_ylim(-18, 108); ax.set_yticks([0, 50, 100]); ax.set_ylabel('Correct coverage (%)')
        ax.set_xticks(range(len(ids)), [data['short'][p] for p in ids], fontsize=7)
        if panel == 'd':
            ax.set_xticks(range(len(ids)), [data['short'][p] + '\n+' + str(data['joint_edges'].get(p, 0)) + ' edges' for p in ids], fontsize=7)
        ax.plot([], [], 'o', color=COLORS[3], label='Full system')
        ax.plot([], [], 'o', color=COLORS[4 + 'def'.index(panel)], label=LABELS[list(CONDITIONS).index(other)])
        ax.legend(frameon=False, fontsize=7, loc='lower right')
        ax.grid(axis='y', color='#E4EDF4', zorder=-1)


def export(fig: Any, path: Path) -> None:
    with mpl.rc_context({'svg.fonttype': 'none'}):
        fig.savefig(path.with_suffix('.svg'))
    with mpl.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(path.with_name(path.name + '_outlined').with_suffix('.svg'))
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=450)


def render(config: Config) -> None:
    style()
    root = config.output
    pilot = read(root / 'cohort.json')['pilot_ids']
    summary = read(root / 'run_summary.json')
    ids = sorted(summary['included_papers'])
    short = dict(zip(pilot, ['Cx46/50', 'TDP-43', 'Co catalyst', 'Silk', 'THz']))
    for i, ident in enumerate(ids):
        short.setdefault(ident, str(i+1))
    data = dict(summary=read(root / 'configuration_summary.json'), effects=read(root / 'paired_effects_summary.json'),
                paper=read(root / 'paper_aspect_metrics.json'), ids=ids, short=short)
    with (root / 'input_information.csv').open() as handle:
        data['joint_edges'] = {r['paper_id']: int(r['historical_edges_only_in_joint']) for r in csv.DictReader(handle)}
    titles = dict(a='a   Shared task, controlled inputs', b='b   Complementary scientific capabilities',
                  c='c   Adding the other branch', d='d   What the joint graph contributes',
                  e='e   What structural values explain', f='f   What citation contacts establish')
    out = root / 'figure'; out.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(13.8, 11), facecolor='white')
    fig.text(.5, .971, 'Fig. 4 | Distinct scientific contributions of GEAR and Graph', ha='center', fontsize=17, weight='bold', fontfamily='Times New Roman')
    fig.text(.5, .943, f"Development pilot · {len(ids)} scientifically retained papers · {summary['reports']} reports · model evaluation", ha='center', color='#687F90')
    boxes = [(x, y, .462, .266) for y in (.656, .350, .044) for x in (.027, .512)]
    for panel, box in zip(titles, boxes):
        ax = card(fig, box, titles[panel])
        if panel in 'abc':
            x, y, w, h = box
            ax.set_position([x + .27*w, y + .18*h, .67*w, .66*h])
        draw_panel(ax, panel, data)
    fig.text(.5, .015, 'Common complete sets for b; complete pairs for c; available reports in d–f. NA states are distinguished in source tables. Development preview.', ha='center', fontsize=8, color='#687F90')
    export(fig, out / 'Fig4')
    fig.savefig(out / 'Fig4_preview.png', dpi=150); plt.close(fig)
    for panel in titles:
        fig = plt.figure(figsize=(7.3, 4.8), facecolor='white')
        ax = card(fig, (.02,.03,.96,.91), titles[panel])
        if panel in 'abc':
            ax.set_position([.28,.20,.66,.60])
        draw_panel(ax, panel, data)
        export(fig, out / f'panel_{panel}'); plt.close(fig)
    (out / 'caption.md').write_text('Fig. 4. 原五篇开发试跑。a：实际输入。b：七配置共同完整样本上的五方面科学要点正确覆盖率。c：加入另一分支的配对差值，误差线为10000次论文bootstrap的95%区间。d–f：联合图、结构数值、引用路径对应方面的完整系统与删除配置。NA为技术缺失或科学不适用，详见源表。此预览尚未加入95篇，也不宣称独立专家评价；后续正式图须加入真实案例与图事实注释。\n', encoding='utf-8')

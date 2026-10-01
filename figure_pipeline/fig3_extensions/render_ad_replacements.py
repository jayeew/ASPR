"""Render exact-size b4/c6 replacement assets without changing Fig3."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from scipy.stats import gaussian_kde

from .render_abcd import paired_samples
from .cd_sources import read

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig3_extensions/abcd/fig3_replacements'
INK, GRID = '#10213B', '#E8EEF3'
COLORS = {'gear': '#E16C35', 'graph': '#1269DF', 'fusion': '#713BCB'}


def configure() -> None:
    for name in ('arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf'):
        font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/' + name)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': '#647487',
        'ytick.color': '#647487', 'axes.edgecolor': '#8295A7', 'axes.linewidth': .65,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def title(fig: plt.Figure, text: str, height: float) -> None:
    fig.text(0, 1 - .7/height, text, va='top', fontfamily='Times New Roman',
             fontsize=14, fontweight='bold')


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / (name + '.' + suffix), dpi=600, transparent=suffix!='png',
                    facecolor='#FBFDFF' if suffix=='png' else 'none')
    plt.close(fig)


def density(ax: plt.Axes, data: dict) -> None:
    xx, yy = np.meshgrid(np.linspace(-5, 42, 180), np.linspace(-16, 24, 160))
    samples = data['bootstrap']
    z = gaussian_kde(samples.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ordered = np.sort(z.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = [ordered[np.searchsorted(cumulative, mass)] for mass in (.95, .5)]
    ax.contourf(xx, yy, z, levels=[levels[0], levels[1], z.max()*1.01],
                colors=['#EEE7F9', '#D5C3EF'], zorder=1)
    ax.scatter(samples[:, 0], samples[:, 1], s=.65, color=COLORS['fusion'],
               alpha=.04, edgecolors='none', rasterized=True, zorder=2)
    ax.contour(xx, yy, z, levels=levels, colors=COLORS['fusion'],
               linewidths=[.7, 1], linestyles=['--', '-'], zorder=3)
    ax.scatter(*data['mean'], marker='D', s=20, facecolor=COLORS['fusion'],
               edgecolor=INK, linewidth=.6, zorder=5)


def render_a() -> None:
    width, height = 142., 43.
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    title(fig, 'b4  Paired contribution gains · Full − comparator', height)
    fig.text(.99, .975, 'pp', ha='right', va='top', fontsize=8.5)
    for index, data in enumerate(paired_samples()):
        left = .087 + index*.495
        ax = fig.add_axes([left, .32, .40, .41])
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=.45)
        density(ax, data)
        ax.axhline(0, color='#647487', lw=.7, zorder=4)
        ax.axvline(0, color='#647487', lw=.7, zorder=4)
        ax.set(xlim=(-5, 42), ylim=(-16, 24), xticks=[0, 20, 40], yticks=[-10, 0, 10, 20])
        ax.tick_params(labelsize=8, length=2, pad=1)
        label = 'GEAR' if data['comparator']=='gear' else 'Graph'
        fig.text(left, .775, f"Full − {label} · n = {data['n']} papers",
                 color=COLORS[data['comparator']], fontsize=9.5, fontweight='bold')
        ax.text(.975, .94, f"ΔR {data['mean'][0]:+.1f}\nΔP {data['mean'][1]:+.1f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=7.7,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.5})
        ax.set_xlabel('Recall gain (pp)', fontsize=8.5, labelpad=1.2)
        if index==0:
            ax.set_ylabel('Precision gain (pp)', fontsize=8.5, labelpad=1.5)
    handles = [Line2D([], [], color='none', marker='D', markerfacecolor=COLORS['fusion'],
                     markeredgecolor=INK, markersize=3.5, label='Mean'),
               Line2D([], [], color=COLORS['fusion'], lw=1, label='50% joint region'),
               Line2D([], [], color=COLORS['fusion'], lw=.7, ls='--', label='95% joint region')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.55, -.012),
               ncol=3, frameon=False, fontsize=7.6, handlelength=1.5, columnspacing=1.2)
    save(fig, 'A_b4_paired_gains')


def card(ax: plt.Axes, x: float, width: float, color: str, label: str) -> None:
    ax.add_patch(Rectangle((x,.03),width,.94, facecolor=matplotlib.colors.to_rgba(color,.045),
                           edgecolor=matplotlib.colors.to_rgba(color,.30), linewidth=.65))
    ax.text(x+.009,.91,label,ha='left',va='top',fontsize=9.5,color=color,fontweight='bold')


def render_d() -> None:
    width, height = 177., 34.
    case = read(ROOT / 'outputs/fig3_extensions/abcd/D_case_evidence.json')
    assessment = case['verification']['assessment']
    if case['verification']['id'] != 's41467-026-68348-w/I0035/R001':
        raise ValueError('Fixed-ID source case changed')
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    title(fig, 'c6  Graph relation and manuscript evidence', height)
    ax = fig.add_axes([0, .02, 1, .78]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    card(ax,.002,.312,'#657B8B','Manuscript evidence')
    card(ax,.342,.300,COLORS['graph'],'Graph · relation candidate')
    card(ax,.674,.324,'#397B65','Supported after narrowing')
    # Only excerpted source wording is quoted; ellipses visibly mark omitted text.
    text = ('“X-ray crystallographic analysis …\n'
            'conﬁrms the targeted sequences”\n'
            '“density functional theory calculations …\n'
            'thermodynamic preference\n'
            'over alternative isomers”')
    ax.text(.011,.73,text,fontsize=8.3,va='top',linespacing=1.02)
    ax.text(.011,.055,'s41467-026-68348-w · manuscript',fontsize=7.3,va='bottom')
    ax.text(.351,.72,'SCXRD + HRMS +\nsemiempirical / DFT calculations',fontsize=9,va='top',linespacing=1.15)
    ax.annotate('',xy=(.491,.37),xytext=(.491,.45),
                arrowprops={'arrowstyle':'-|>','lw':.9,'color':COLORS['graph']})
    ax.text(.491,.33,'Structure and\nisomer preference',fontsize=9,ha='center',va='top',linespacing=1.1)
    ax.text(.683,.72,'X-ray analysis → targeted sequences\n'
            'Semiempirical / DFT calculations\n→ preference over alternative isomers',
            fontsize=8.7,va='top',linespacing=1.12)
    ax.text(.683,.30,'HRMS role: unresolved in the excerpt.\nHistorical novelty: not established.',
            fontsize=8.3,va='top',linespacing=1.12)
    for left, right in ((.315,.339),(.644,.671)):
        ax.add_patch(FancyArrowPatch((left,.5),(right,.5),arrowstyle='-|>',
                                    mutation_scale=8,lw=.9,color='#708297'))
    save(fig, 'D_c6_graph_relation')


def main() -> None:
    configure(); render_a(); render_d()
    path=OUT/'README.md'
    if not path.exists():
        path.write_text('''# Fig3 replacement assets

No current Fig3 or existing study output was changed. PNG (600 dpi), PDF and SVG have exact slot dimensions; no tight bounding-box crop is applied. PNG uses the current panel background (#FBFDFF); PDF and SVG backgrounds are transparent.

| Asset | Slot | Width × height | Top-left in existing Fig3 |
|---|---|---|---|
| A_b4_paired_gains | b4 | 142 × 43 mm | x=161, y=120 mm |
| D_c6_graph_relation | c6 | 177 × 34 mm | x=231, y=269 mm |

A reads the same complete-pair populations (84 Full–GEAR and 76 Full–Graph papers), jointly resamples paper indices 10,000 times with seed 20260922, and expresses both axes in percentage points (pp). The contours are KDE joint-density regions; they are not rectangular combinations of independent intervals. Cohorts differ from the existing single-metric summaries. Missing precision is not filled with zero. The 95% precision intervals cross zero.

D uses the fixed lexical-first supported/narrowed case s41467-026-68348-w/I0035/R001, manuscript evidence s41467-026-68348-w/E00252. Ellipses mark omitted source wording. The middle card paraphrases the existing Graph relation, not a newly generated system answer. The right card reports the existing independent evidence assessment. Neither the manuscript relation nor Graph proximity establishes historical novelty. Full is not shown because its same-relation treatment remains unresolved. Full quotes, locations and source roles are in ../D_case_evidence.json.

Only these two illustration slots are supplied. Original b4 historical-comparison/scope-difference facets should be retained in supplementary material if these replacements are inserted.
''',encoding='utf-8')
    print('Rendered exact-size b4 and c6 assets:', OUT)


if __name__ == '__main__':
    main()

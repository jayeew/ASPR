"""Publish a static, source-backed Fig.8 for the complete local historical cohort."""
from __future__ import annotations

import argparse
import json
import gzip
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from matplotlib import font_manager
from matplotlib.transforms import Bbox

from .helpers import configure, label
from .helpers import effective_communities, neighborhood_overlap

from .data import JOURNALS, OUT, SHORT, TYPES, YEARS, prepare

INK, MUTED, GRID = '#24282D', '#737B85', '#E4ECF2'
EDGE, HEADER = '#A9CDE4', '#EEF6FC'
COLORS = ['#398BEA', '#FA993B', '#AB7BD2', '#54A76A', '#E985B6', '#E6B549', '#30B8C5', '#EB596E']
TYPE_COLORS = ['#1269DF', '#239D72', '#713BCB', '#E16C35', '#D847A4']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
BOUNDS = {'a': (.014, .525, .378, .918), 'b': (.387, .04, .987, .918),
          'c': (.014, .04, .378, .505)}


def card(fig: plt.Figure, letter: str, title: str) -> None:
    x0, y0, x1, y1 = BOUNDS[letter]
    fig.add_artist(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle='round,pad=0,rounding_size=0.004', transform=fig.transFigure,
        facecolor='#FCFDFF', edgecolor=EDGE, linewidth=.8, zorder=-5))
    fig.add_artist(Rectangle((x0+.001, y1-.049), x1-x0-.002, .047,
        transform=fig.transFigure, facecolor=HEADER, edgecolor='none', zorder=-4))
    label(fig, x0+.009, y1-.034, letter, 17, True, fontfamily='Times New Roman')
    label(fig, x0+.035, y1-.032, title, 13, True, fontfamily='Times New Roman')


def summary(data: dict, journal: str, year: int = 0) -> dict:
    return next(r for r in data['summaries'] if r['journal'] == journal and r['year'] == year)


def panel_a(fig: plt.Figure, data: dict) -> None:
    card(fig, 'a', 'Contribution profiles')
    label(fig, .028, .842, 'Abstract-derived claim types (%)', 11, True, fontfamily='Times New Roman')
    ax = fig.add_axes([.120, .606, .216, .222], facecolor='none')
    left = np.zeros(8)
    for role, color in zip(TYPES, TYPE_COLORS):
        values = np.array([summary(data, j)[f'type_{role}'] * 100 for j in JOURNALS])
        ax.barh(range(8), values, left=left, color=color, height=.70, edgecolor='white', linewidth=.4)
        left += values
    ax.set(yticks=range(8), yticklabels=SHORT, xlim=(0, 100), xticks=[0, 50, 100])
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0, pad=6, labelsize=8.5)
    ax.spines['left'].set_visible(False)
    ax.text(105, -1.1, 'n', color=MUTED, fontsize=8.5)
    for i, journal in enumerate(JOURNALS):
        ax.text(105, i, f'{summary(data, journal)["n_claimed_papers"]:,}', va='center', fontsize=8.5, color=MUTED)
    ax.legend([Patch(facecolor=c) for c in TYPE_COLORS],
              ['Method', 'Finding', 'Mechanism', 'Resource', 'Theory'], frameon=False,
              loc='upper left', bbox_to_anchor=(-.43, -.18), ncol=5, fontsize=8.5,
              handlelength=1, columnspacing=1.0)


def legend(fig: plt.Figure) -> None:
    handles = [Line2D([], [], marker=m, color=c, linestyle='', markersize=5,
                      label=name) for c, m, name in zip(COLORS, MARKERS, SHORT)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.412, .865),
               frameon=False, ncol=4, fontsize=8.5, columnspacing=1.5, handletextpad=.35)


def panel_b(fig: plt.Figure, data: dict) -> list[plt.Axes]:
    card(fig, 'b', 'Paper-level knowledge structure')
    legend(fig)
    axes = []
    for k, year in enumerate(YEARS):
        ax = fig.add_axes([.429 + k * .185, .120, .173, .638], facecolor='none')
        rows = [p for p in data['papers'] if p['year'] == year and p['scatter_eligible']]
        for j, journal in enumerate(JOURNALS):
            sub = [p for p in rows if p['journal'] == journal]
            ax.scatter([p['effective_communities'] for p in sub], [p['mean_jaccard'] for p in sub],
                       c=COLORS[j], marker=MARKERS[j], s=5 if j == 0 else 19,
                       alpha=.13 if j == 0 else .85, linewidths=0, zorder=2 + int(j > 0))
        ax.set(xlim=(0, 18), ylim=(-.025, 1.025), xticks=[0, 6, 12, 18], yticks=[0, .25, .5, .75, 1])
        ax.set_title(f'{year}  |  n = {len(rows):,}', fontsize=10.5, weight='bold', pad=9)
        ax.grid(color=GRID, lw=.5, zorder=0)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Mean claim-neighborhood overlap (Jaccard)', fontsize=9.5)
        axes.append(ax)
    label(fig, .702, .058, 'Effective communities among assigned prior neighbors', 10, ha='center')
    return axes


def panel_c(fig: plt.Figure, data: dict) -> None:
    card(fig, 'c', 'Within-journal heterogeneity')
    label(fig, .028, .429, 'Shared historical neighbors across claims', 11, True, fontfamily='Times New Roman')
    ax = fig.add_axes([.120, .115, .216, .295], facecolor='none')
    rng = np.random.default_rng(20261003)
    for j, journal in enumerate(JOURNALS):
        values = np.array([p['shared_neighbor_fraction'] * 100 for p in data['papers']
                           if p['journal'] == journal and p['shared_neighbor_fraction'] is not None])
        ax.scatter(values, j + rng.uniform(-.23, .23, len(values)), s=4 if j == 0 else 12,
                   marker=MARKERS[j], color=COLORS[j], alpha=.09 if j == 0 else .60,
                   edgecolors='none', zorder=2)
        q1, med, q3 = np.quantile(values, [.25, .5, .75])
        ax.plot([q1, q3], [j, j], color=INK, lw=2.2, zorder=4)
        ax.scatter(med, j, marker='|', c='white', s=65, linewidths=1.5, zorder=5)
        ax.text(104, j, f'{len(values):,}', fontsize=8.5, va='center', color=MUTED)
    ax.set(xlim=(-2, 102), ylim=(7.6, -.6), yticks=range(8), yticklabels=SHORT,
           xticks=[0, 25, 50, 75, 100], xlabel='Historical nodes shared by ≥2 claims (%)')
    ax.tick_params(axis='y', length=0, pad=6, labelsize=8.5)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=.5, zorder=0)
    ax.xaxis.label.set_size(8.5)
    ax.text(104, -1.0, 'n', fontsize=8.5, color=MUTED)


def save(fig: plt.Figure, stem: Path, dpi: int = 180) -> None:
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(stem.with_suffix('.' + extension), dpi=dpi)


def render(data: dict) -> None:
    configure()
    for name in ['times.ttf', 'timesbd.ttf']:
        font_manager.fontManager.addfont(str(Path('/mnt/c/Windows/Fonts') / name))
    plt.rcParams.update({'text.color': INK, 'axes.labelcolor': INK,
                         'axes.edgecolor': '#737B85', 'axes.linewidth': .55,
                         'xtick.color': MUTED, 'ytick.color': INK})
    fig = plt.figure(figsize=(18, 7), facecolor='white')
    label(fig, .5, .963, 'Fig. 8 | Contribution and knowledge-structure profiles across journals',
          20, True, ha='center', fontfamily='Times New Roman')
    panel_a(fig, data)
    panel_b(fig, data)
    panel_c(fig, data)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert all(fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) and
               fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1]) for t in fig.texts)
    save(fig, OUT / 'final/Fig8')
    fig.savefig(OUT / 'final/Fig8_600dpi.png', dpi=600)
    with plt.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(OUT / 'final/Fig8_outlined.svg')
    for name, bounds in BOUNDS.items():
        bbox = Bbox.from_extents(bounds[0] * 18, bounds[1] * 7, bounds[2] * 18, bounds[3] * 7)
        for extension in ['svg', 'pdf', 'png']:
            fig.savefig(OUT / 'panels' / f'Fig8{name}.{extension}', dpi=240, bbox_inches=bbox)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true', help='Reuse derived data from this figure directory.')
    args = parser.parse_args()
    assert effective_communities([None, 1]) == 1
    assert effective_communities([None]) is None
    assert abs(effective_communities([1, 2]) - 2) < 1e-12
    assert neighborhood_overlap([{'a', 'b'}, {'b', 'c'}, set()]) == (1 / 3, 1)
    assert neighborhood_overlap([{'a'}, {'b'}]) == (0, 1)
    if args.render_only:
        with gzip.open(OUT / 'data/plot_data.json.gz', 'rt', encoding='utf-8') as stream:
            data = json.load(stream)
    else:
        data = prepare()
    render(data)
    from .documents import deliver
    deliver(data)
    print(json.dumps({'output': str(OUT), 'papers': len(data['papers']), 'model_calls': 0}))

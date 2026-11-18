"""Reference-style Fig.5: profiles, paired trajectories, flows and cost portraits."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import ASPECTS

from .prepare import OUT

W, H = 420., 300.
INK, GRID = '#24282D', '#E4ECF2'
EDGE, HEADER, BG = '#A9CDE4', '#EEF6FC', '#FCFDFF'
ORANGE, BLUE, PURPLE, SLATE, TEAL = '#E16C35', '#1269DF', '#713BCB', '#8896A8', '#40A7BE'
COLORS = {'F': PURPLE, 'E50': ORANGE, 'K5': BLUE, 'LUNA': TEAL}
NAMES = {'F': 'Full system · GPT-6.1 Sol', 'E50': 'About half of historical sources',
         'K5': 'At most five semantic neighbors', 'LUNA': 'GPT-5.6 Luna generator'}
ASPECT_LABELS = ['Historical\nverification', 'Knowledge\nposition', 'Joint\nstructure',
                 'Structural\nresolution', 'Citation\ncontact']
BEHAVIORS = {'grounded': BLUE, 'abstention': ORANGE, 'unsupported': '#C56154',
             'incomplete': '#BDC8D2', 'unresolved': '#FFFFFF'}
STATE_LABELS = {'grounded': 'Grounded\nanswers', 'abstention': 'Explicit\nabstention',
                'unsupported': 'Unsupported\nassertions', 'incomplete': 'Incomplete\nor omitted',
                'unresolved': 'Evaluator\nunresolved'}
PANELS = {'a': (5, 22, 184, 126), 'b': (194, 22, 221, 126),
          'c': (5, 154, 223, 141), 'd': (233, 154, 182, 141)}
TITLES = {'a': 'Performance across observed settings', 'b': 'Paired parameter and model effects',
          'c': 'Answer transitions after evidence reduction', 'd': 'Answer quality and computational cost'}


class Figure:
    def __init__(self) -> None:
        for name in ('arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf'):
            font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/' + name)
        mpl.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.labelsize': 10,
                             'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': .65,
                             'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                             'ytick.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
        self.fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
        self.serial = 0
        self.text('heading', W/2, 5, 'Fig. 5 | Robustness, evidence boundaries and computational cost',
                  22, bold=True, ha='center')
        for key, (x, y, w, h) in PANELS.items():
            self.card(key, (x, y, w, h))
            self.rect(key, (x+.3, y+.3, w-.6, 7.7), HEADER)
            self.text(key, x+2.2, y+1.0, key, 18, bold=True)
            self.text(key, x+11, y+1.7, TITLES[key], 14, bold=True)

    def text(self, group: str, x: float, y: float, text: str, size: float = 10,
             color: str = INK, bold: bool = False, **kwargs: Any) -> None:
        self.serial += 1
        self.fig.text(x/W, 1-y/H, text, fontsize=size, va='top', color=color,
                      fontfamily='Times New Roman' if bold else 'Arial',
                      weight='bold' if bold else 'normal', linespacing=1.2,
                      gid=f'{group}_text_{self.serial}', **kwargs)

    def rect(self, group: str, box: tuple[float, float, float, float], color: str) -> None:
        x, y, w, h = box
        self.fig.add_artist(Rectangle((x/W, 1-(y+h)/H), w/W, h/H,
                           transform=self.fig.transFigure, facecolor=color, edgecolor='none', zorder=-4))

    def card(self, group: str, box: tuple[float, float, float, float], edge: str = EDGE) -> None:
        x, y, w, h = box
        self.fig.add_artist(FancyBboxPatch((x/W, 1-(y+h)/H), w/W, h/H,
            boxstyle='round,pad=0,rounding_size=0.004', transform=self.fig.transFigure,
            facecolor=BG, edgecolor=edge, linewidth=.75, zorder=-5, gid=group+'_card'))

    def ax(self, group: str, box: tuple[float, float, float, float]) -> Axes:
        x, y, w, h = box
        ax = self.fig.add_axes([x/W, 1-(y+h)/H, w/W, h/H])
        ax.set_gid(group+'_axes')
        ax.set_facecolor('none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5, pad=3, width=.6)
        ax.set_axisbelow(True)
        return ax


def profile(f: Figure, groups: list[str], colors: list[str], markers: list[str],
            box: tuple[float, float, float, float], dashed: bool = False) -> None:
    ax = f.ax('a', box)
    rows = read(OUT / 'panel_a_summary.json')
    for group, color, marker in zip(groups, colors, markers):
        values = [next(r for r in rows if r['group'] == group and r['aspect'] == aspect)['mean'] * 100 for aspect in ASPECTS]
        ax.plot(range(5), values, color=color, marker=marker, ms=5, lw=1.6,
                ls='--' if dashed else '-', markeredgecolor='white', markeredgewidth=.5)
    ax.set(xlim=(-.15, 4.15), ylim=(85, 102), yticks=[85, 90, 95, 100],
           xticks=range(5), xticklabels=ASPECT_LABELS, ylabel='Grounded correct\ncoverage (%)')
    ax.grid(axis='both', color=GRID, lw=.5)
    ax.tick_params(axis='x', labelsize=9)


def key(f: Figure, x: float, y: float, color: str, text: str, marker: str = 'o') -> None:
    f.fig.add_artist(plt.Line2D([(x+.5)/W, (x+4)/W], [1-(y+1.5)/H]*2,
        transform=f.fig.transFigure, color=color, lw=1.5, marker=marker, markersize=4, markevery=[0]))
    f.text('a', x+6, y, text, 8.8)


def panel_a(f: Figure) -> None:
    groups = ['life', 'physical_engineering', 'medicine', 'earth_environment']
    colors = [ORANGE, BLUE, PURPLE, TEAL]
    labels = ['Life sciences: 40 papers; 27 with citation tasks',
              'Physics / engineering: 38 papers; 29 with citation tasks',
              'Medicine: 13 papers; 11 with citation tasks',
              'Earth / environment: 8 papers; 7 with citation tasks']
    for i, (label, color, marker) in enumerate(zip(labels, colors, ['o', 's', 'D', '^'])):
        key(f, 10+(i%2)*89, 34+(i//2)*5.5, color, label, marker)
    profile(f, groups, colors, ['o', 's', 'D', '^'], (26, 48, 155, 38))
    f.text('a', 10, 98, 'Grouped by readable independent historical sources', 11, bold=True)
    key(f, 10, 104.5, ORANGE, 'At most 16 sources: 61 papers; 45 with citation tasks', 'o')
    key(f, 99, 104.5, BLUE, 'More than 16 sources: 38 papers; 29 with citation tasks', 's')
    profile(f, ['lower', 'higher'], [ORANGE, BLUE], ['o', 's'], (26, 112, 155, 25), dashed=True)


def panel_b(f: Figure) -> None:
    points = read(OUT / 'panel_b_paired.json')
    summary = read(OUT / 'panel_b_summary.json')
    f.text('b', 200, 33.5, 'Relative to full system · one path per paper · diamonds: means with 95% confidence intervals', 10)
    for i, aspect in enumerate(ASPECTS):
        x, width = 211+i*39.5, 32.5
        f.text('b', x+width/2, 41, ASPECT_LABELS[i], 11, bold=True, ha='center')
        pairs: dict[str, dict[str, float]] = defaultdict(dict)
        for r in points:
            if r['aspect'] == aspect and r['condition'] in ('K5', 'LUNA') and r['complete']:
                pairs[r['paper_id']][r['condition']] = r['delta_pp']
        assert all(set(p) == {'K5', 'LUNA'} for p in pairs.values())
        f.text('b', x+width/2, 52, f'{len(pairs)} paired papers', 9, ha='center')
        ax = f.ax('b', (x, 59, width, 69))
        for j, (_, p) in enumerate(sorted(pairs.items())):
            jitter = ((j%5)-2)*.018
            xs = [.16+jitter, .84+jitter]
            ax.plot(xs, [p['K5'], p['LUNA']], color='#ACC4D5', alpha=.48, lw=.6, zorder=1)
            ax.scatter(xs, [p['K5'], p['LUNA']], c=[BLUE, TEAL], s=13, alpha=.5, linewidths=0, zorder=2)
        for c, xpos in [('K5', .16), ('LUNA', .84)]:
            r = next(r for r in summary if r['condition'] == c and r['aspect'] == aspect)
            ax.errorbar(xpos, r['mean'], yerr=[[r['mean']-r['low']], [r['high']-r['mean']]],
                        fmt='D', markersize=5.7, color=COLORS[c], markeredgecolor='white',
                        markeredgewidth=.7, capsize=3.5, elinewidth=2, zorder=5)
            ax.text(xpos, r['mean']+10, f'{r["mean"]:+.1f}', color=COLORS[c], ha='center', va='bottom', fontsize=9.5, weight='bold')
        ax.axhline(0, color='#647B8F', lw=.75)
        ax.set(xlim=(-.12, 1.12), ylim=(-109, 55), xticks=[.16, .84],
               xticklabels=['Five-neighbor\nlimit', 'GPT-5.6\nLuna'], yticks=[-100, -50, 0, 50])
        ax.tick_params(axis='x', labelsize=8.8, pad=5)
        for tick, color in zip(ax.get_xticklabels(), [BLUE, TEAL]):
            tick.set_color(color)
        if i == 0:
            ax.set_ylabel('Grounded coverage change\n(percentage points)')
        else:
            ax.set_yticklabels([])
        ax.grid(axis='y', color=GRID, lw=.5)


def transition_data() -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    for r in read(OUT / 'panel_c_parts.json'):
        pairs[(r['group'], r['paper_id'], r['question_id'], r['part_id'])][r['condition']] = r['behavior']
    assert all(set(p) == {'F', 'E50'} for p in pairs.values())
    counts = Counter((k[0], v['F'], v['E50']) for k, v in pairs.items())
    rows = [{'group': g, 'full_system_behavior': a, 'reduced_evidence_behavior': b, 'answer_parts': n}
            for (g, a, b), n in sorted(counts.items())]
    expected = read(OUT / 'panel_c_summary.json')
    for row in expected:
        field = 'full_system_behavior' if row['condition'] == 'F' else 'reduced_evidence_behavior'
        observed = sum(r['answer_parts'] for r in rows if r['group'] == row['group'] and r[field] == row['behavior'])
        assert observed == row['count'], 'Transition marginals differ from validated source totals'
    write(OUT / 'panel_c_transitions.json', rows)
    write_csv(OUT / 'panel_c_transitions.csv', rows)
    return rows


def ribbon(ax: Axes, left: tuple[float, float], right: tuple[float, float], color: str) -> None:
    x0, x1 = .32, .68
    lo, hi = left
    rlo, rhi = right
    points = [(x0, lo), (.49, lo), (.51, rlo), (x1, rlo), (x1, rhi),
              (.51, rhi), (.49, hi), (x0, hi), (x0, lo)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(points, codes), facecolor=mpl.colors.to_rgba(color, .3),
                           edgecolor=mpl.colors.to_rgba(color, .55), lw=.5, zorder=1))


def flow(f: Figure, group: str, box: tuple[float, float, float, float], rows: list[dict[str, Any]]) -> None:
    ax = f.ax('c', box)
    ax.set(xlim=(0, 1), ylim=(0, 1.1))
    ax.axis('off')
    data = [r for r in rows if r['group'] == group]
    total = sum(r['answer_parts'] for r in data)
    order = list(BEHAVIORS)
    counts = [Counter(), Counter()]
    for r in data:
        counts[0][r['full_system_behavior']] += r['answer_parts']
        counts[1][r['reduced_evidence_behavior']] += r['answer_parts']
    bounds: list[dict[str, tuple[float, float]]] = []
    for side in range(2):
        top, mapping = .89, {}
        for state in order:
            if counts[side][state]:
                height = .70*counts[side][state]/total
                mapping[state] = (top-height, top)
                top -= height+.10
        bounds.append(mapping)
    cursor = [{k: v[1] for k, v in m.items()} for m in bounds]
    for r in sorted(data, key=lambda r: (order.index(r['full_system_behavior']), order.index(r['reduced_evidence_behavior']))):
        a, b, height = r['full_system_behavior'], r['reduced_evidence_behavior'], .70*r['answer_parts']/total
        ribbon(ax, (cursor[0][a]-height, cursor[0][a]), (cursor[1][b]-height, cursor[1][b]), BEHAVIORS[b])
        cursor[0][a] -= height
        cursor[1][b] -= height
    for side, x in [(0, .295), (1, .68)]:
        for state, (lo, hi) in bounds[side].items():
            ax.add_patch(Rectangle((x, lo), .025, hi-lo, fc=BEHAVIORS[state], ec='white', lw=.6, zorder=3))
            count = counts[side][state]
            text = f'{STATE_LABELS[state]}\n{count} / {total} ({100*count/total:.1f}%)'
            ax.text(x-.025 if side == 0 else x+.05, (lo+hi)/2, text,
                    ha='right' if side == 0 else 'left', va='center', fontsize=9.3, linespacing=1.3,
                    color=ORANGE if state == 'abstention' else INK)
    ax.text(.30, 1.04, 'All historical\nsources retained', ha='center', va='top', fontsize=10, color=PURPLE, weight='bold')
    ax.text(.70, 1.04, 'About half of\nsources retained', ha='center', va='top', fontsize=10, color=ORANGE, weight='bold')
    ax.text(.5, 1.015, '→', ha='center', va='top', fontsize=15)


def panel_c(f: Figure) -> None:
    rows = transition_data()
    summaries = read(OUT / 'panel_c_summary.json')
    for i, (group, label) in enumerate([
            ('retained', 'Evidence remains sufficient'), ('lost', 'Required evidence is lost')]):
        summary = next(r for r in summaries if r['group'] == group)
        papers, parts = summary['n'], summary['N']
        x = 9+i*109
        f.card('c_group', (x, 166, 105, 123))
        f.text('c', x+52.5, 168, label, 13, bold=True, ha='center')
        f.text('c', x+52.5, 175, f'{papers} papers · {parts} paired answer parts', 10, ha='center')
        flow(f, group, (x+1.5, 185, 102, 100), rows)


def configuration_card(f: Figure, y: float, row: dict[str, Any]) -> None:
    c = row['condition']
    f.card('d_config', (339, y, 71, 28), edge=COLORS[c])
    f.text('d', 342, y+1.5, NAMES[c], 10.2, color=COLORS[c], bold=True)
    fields = [('Grounded answers', 100*row['grounded_answer_rate'], '%'),
              ('Reasonable abstention', 100*row['reasonable_abstention_rate'], '%'),
              ('Unsupported assertions', 100*row['confirmed_unsupported_rate'], '%'),
              ('Relative call time', row['time_ratio_median'], '×')]
    for i, (label, value, unit) in enumerate(fields):
        f.text('d', 342, y+8+i*4.4, label, 9)
        f.text('d', 406, y+8+i*4.4, f'{value:.1f}{unit}' if unit == '%' else f'{value:.2f}{unit}',
               9, ha='right', color=COLORS[c])


def panel_d(f: Figure) -> None:
    rows = {r['condition']: r for r in read(OUT / 'panel_d_summary.json')}
    f.text('d', 239, 166.5, f'{rows["F"]["n"]} papers with all four configurations', 10.5)
    f.text('d', 239, 174, 'Full system: at most 10 semantic neighbors', 9)
    ax = f.ax('d', (251, 186, 79, 92))
    full = rows['F']
    for c in ('E50', 'K5', 'LUNA'):
        r = rows[c]
        ax.annotate('', (r['time_ratio_median'], 100*r['grounded_answer_rate']),
                    (full['time_ratio_median'], 100*full['grounded_answer_rate']),
                    arrowprops={'arrowstyle': '-|>', 'color': COLORS[c], 'lw': 1.1, 'alpha': .65,
                                'shrinkA': 6, 'shrinkB': 5}, zorder=2)
    labels = {'F': ('Full system', (-44, 14)), 'K5': ('Five-neighbor\nlimit', (-67, -3)),
              'E50': ('Half of historical\nsources', (-81, -23)), 'LUNA': ('GPT-5.6 Luna', (9, -5))}
    for i, c in enumerate(('F', 'K5', 'E50', 'LUNA')):
        r = rows[c]
        x, y = r['time_ratio_median'], 100*r['grounded_answer_rate']
        ax.scatter(x, y, s=75, marker={'F':'D','K5':'s','E50':'o','LUNA':'^'}[c],
                   color=COLORS[c], edgecolors='white', linewidths=.8, zorder=5)
        label, offset = labels[c]
        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', color=COLORS[c],
                    fontsize=9.2, va='center', annotation_clip=False,
                    arrowprops={'arrowstyle':'-', 'color':COLORS[c], 'lw':.55, 'shrinkA':3,'shrinkB':4})
        configuration_card(f, 171+i*30, r)
    ax.axvline(1, color='#9BACB9', ls=':', lw=.8)
    ax.set(xlim=(0, 1.15), ylim=(0, 110), xticks=[0, .25, .5, .75, 1],
           xticklabels=['0', '0.25', '0.50', '0.75', '1.00'], yticks=[0, 25, 50, 75, 100],
           xlabel='Cumulative call time\nrelative to full system', ylabel='Grounded answer yield (%)')
    ax.grid(axis='y', color=GRID, lw=.5)


def main() -> None:
    names = ['panel_a_summary.json', 'panel_b_paired.json', 'panel_b_summary.json',
             'panel_c_parts.json', 'panel_d_summary.json']
    hashes = {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}
    f = Figure()
    panel_a(f)
    panel_b(f)
    panel_c(f)
    panel_d(f)
    for ext in ('png', 'pdf', 'svg'):
        f.fig.savefig(OUT / f'Fig5.{ext}', dpi=300, facecolor='white')
    f.fig.savefig(OUT / 'Fig5_preview.png', dpi=120, facecolor='white')
    plt.close(f.fig)
    assert hashes == {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}
    write(OUT / 'render_manifest.json', {'style_reference': 'figure_pipeline/fig4_explanation_reference/render.py',
          'fonts': ['Times New Roman', 'Arial'], 'panel_colors': {'edge': EDGE, 'header': HEADER, 'background': BG},
          'chart_forms': ['grouped capability profiles', 'paired paper trajectories with mean confidence intervals',
                          'observed answer-part transition flows', 'quality-cost displacement and configuration portraits'],
          'source_hashes_unchanged_by_render': hashes, 'gray_footer_notes': 0,
          'abbreviated_configuration_and_metric_labels': False, 'new_model_calls': 0})


if __name__ == '__main__':
    main()

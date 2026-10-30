"""Render the current Fig.8 directly from the unified analysis tables."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from scipy.stats import gaussian_kde

from .helpers import configure, label

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig8_reference'
TABLES = ROOT / 'outputs/fig8_data/analysis/tables'
FIELDS = ['biochemistry', 'materials', 'physics']
FIELD_NAMES = ['Biochemistry', 'Materials', 'Physics']
COLORS = ['#4C78A8', '#E4AE4A', '#D87562']
INK, MUTED, GRID = '#24282D', '#687582', '#E3E8EE'
RED, PALE_RED = '#B43E54', '#E99AA2'
TYPE_COLORS = {'Education': '#4C78A8', 'Healthcare': '#42A99B',
               'Facility': '#9A79C5', 'Company': '#E5A24A',
               'Government-type': '#AD7950', 'Other': '#8D969D',
               'Unclassified': '#CBD0D5'}
BOUNDS = {'a': (.012, .656, .988, .951),
          'b': (.012, .343, .988, .649),
          'c': (.012, .025, .988, .334)}
CASES = ['10.1038/s41586-024-08053-0', '10.1038/s41586-024-07188-4']
SHORT_INSTITUTIONS = ['Chinese Acad. of Sciences', 'CNRS', 'Russian Acad. of Sciences',
                      'Univ. Chinese Acad. Sciences', 'Harvard Univ.',
                      'Shanghai Jiao Tong Univ.', 'Zhejiang Univ.', 'Fudan Univ.',
                      'Univ. of Tokyo', 'Peking Univ.', 'Inserm', 'Sun Yat-sen Univ.',
                      'Univ. Sci. & Tech. China', 'Université Paris-Saclay', 'Univ. of Oxford']
COUNTRY_NAMES = {'CN': 'China', 'US': 'USA', 'DE': 'Germany', 'IN': 'India',
                 'GB': 'UK', 'RU': 'Russia', 'JP': 'Japan', 'IT': 'Italy',
                 'FR': 'France', 'ES': 'Spain', 'KR': 'South Korea', 'CA': 'Canada'}


def load_data() -> dict[str, pd.DataFrame]:
    filenames = {'papers': 'panel_b_four_journal_papers', 'journals': 'panel_b_journals',
                 'countries': 'panel_c_countries', 'institutions': 'panel_c_institutions',
                 'country_top': 'panel_c_country_top12',
                 'institution_top': 'panel_c_institution_top15'}
    return {key: pd.read_csv(TABLES / f'{name}.csv', low_memory=False)
            for key, name in filenames.items()}


def text(fig: plt.Figure, x: float, y: float, value: str, size: float = 9,
         bold: bool = False, **kwargs: Any) -> None:
    label(fig, x, y, value, size, bold, **kwargs)


def card(fig: plt.Figure, letter: str, title: str) -> None:
    x0, y0, x1, y1 = BOUNDS[letter]
    fig.add_artist(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle='round,pad=0,rounding_size=0.003', transform=fig.transFigure,
        facecolor='white', edgecolor='#A9CDE4', linewidth=.8, zorder=-5))
    wash = fig.add_axes([x0+.001, y1-.045, x1-x0-.002, .044], zorder=-4)
    rgb = np.array([.91, .96, 1.0])
    gradient = 1-(1-rgb)[None, None, :]*np.linspace(0, 1, 128)[:, None, None]
    wash.imshow(gradient, origin='lower', aspect='auto', extent=(0, 1, 0, 1))
    wash.axis('off')
    text(fig, x0+.009, y1-.025, letter, 18, True, fontfamily='Arial')
    text(fig, x0+.033, y1-.024, title, 15, True, fontfamily='Times New Roman')


def subtitle(fig: plt.Figure, x: float, y: float, code: str, title: str) -> None:
    text(fig, x, y, code, 11, True, fontfamily='Arial')
    text(fig, x+.026, y, title, 11, True, fontfamily='Arial')


def style_axis(ax: plt.Axes, grid: str = 'both') -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=.55, linestyle='--')
    ax.tick_params(length=3, width=.6, pad=3, labelsize=8)


def panel_a_scatter(fig: plt.Figure, papers: pd.DataFrame) -> None:
    subtitle(fig, .024, .899, 'a1', 'Paper-level structural change')
    ax = fig.add_axes([.060, .708, .286, .172])
    valid = papers.dropna(subset=['neighbor_jaccard_mean', 'efficiency_change'])
    valid = valid.sample(frac=1, random_state=20261029)
    color = valid.field.map(dict(zip(FIELDS, COLORS)))
    ax.scatter(valid.neighbor_jaccard_mean, valid.efficiency_change,
               s=valid.historical_neighbor_count*1.05, c=color, alpha=.47,
               edgecolors='white', linewidths=.15, rasterized=True)
    for i, doi in enumerate(CASES):
        row = papers.loc[papers.doi.eq(doi)].iloc[0]
        x, y = row.neighbor_jaccard_mean, row.efficiency_change
        ax.scatter([x], [y], s=90, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=8)
        ax.annotate('AB'[i], (x, y), xytext=(15, -14 if i == 0 else 12),
                    textcoords='offset points', fontsize=11, weight='bold',
                    arrowprops={'arrowstyle': '-', 'lw': .8, 'color': INK}, zorder=9)
    ax.set(xlim=(0, .8), ylim=(0, .52), xticks=[0, .2, .4, .6, .8],
           yticks=[0, .1, .2, .3, .4, .5],
           xlabel='Mean overlap between contribution neighbourhoods',
           ylabel='Change in local efficiency, ΔE')
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    style_axis(ax)
    ax.text(.98, .96, f'n = {len(valid):,}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9)
    legend = fig.add_axes([.354, .724, .073, .158]); legend.axis('off')
    legend.set(xlim=(0, 1), ylim=(0, 1))
    for k, (name, col) in enumerate(zip(FIELD_NAMES, COLORS)):
        legend.scatter(.08, .94-k*.105, s=30, color=col)
        legend.text(.19, .94-k*.105, name, fontsize=8, va='center')
    legend.text(.03, .55, 'Reference\nneighbours', fontsize=8.5, linespacing=1.3)
    for k, h in enumerate([20, 40, 60]):
        legend.scatter(.17, .40-k*.145, s=h*1.05, color='#9AA5AF', alpha=.65)
        legend.text(.38, .40-k*.145, str(h), va='center', fontsize=8)


def panel_a_raincloud(fig: plt.Figure, papers: pd.DataFrame) -> None:
    subtitle(fig, .434, .899, 'a2', 'Newly connected pair ratio')
    ax = fig.add_axes([.487, .708, .160, .172])
    rng = np.random.default_rng(20261029)
    for k, (field, col, name) in enumerate(zip(FIELDS, COLORS, FIELD_NAMES)):
        sub = papers[papers.field.eq(field)].dropna(subset=['newly_connected_pair_ratio'])
        values = sub.newly_connected_pair_ratio.to_numpy()*100
        y = 2-k; grid = np.linspace(0, 100, 250)
        density = gaussian_kde(values)(grid)
        ax.fill_between(grid, y+.045, y+.045+.34*density/density.max(),
                        color=col, alpha=.28, lw=0)
        ax.plot(grid, y+.045+.34*density/density.max(), color=col, lw=.9)
        ax.scatter(values, y-rng.uniform(.07, .30, len(values)), s=2.4,
                   color=col, alpha=.38, edgecolors='none', rasterized=True)
        q1, med, q3 = np.quantile(values, [.25, .5, .75])
        ax.plot([q1, q3], [y, y], color=INK, lw=3, solid_capstyle='butt')
        ax.scatter([med], [y], s=24, facecolor='white', edgecolor=INK, lw=1, zorder=5)
        ax.text(1.02, y+.20, f'n = {len(sub):,}', transform=ax.get_yaxis_transform(),
                fontsize=7.7, color=col, ha='right')
    ax.set(xlim=(0, 100), ylim=(-.40, 2.48), yticks=[2, 1, 0], yticklabels=FIELD_NAMES,
           xticks=[0, 25, 50, 75, 100], xlabel='Newly connected pair ratio (%)')
    ax.xaxis.label.set_fontsize(9)
    ax.tick_params(axis='y', length=0, pad=6, labelsize=8)
    ax.spines['left'].set_visible(False)
    style_axis(ax, 'x')


def panel_a_cases(fig: plt.Figure, papers: pd.DataFrame) -> None:
    subtitle(fig, .674, .899, 'a3', 'Example paper comparisons')
    text(fig, .824, .876, '6 matched contributions · 36 reference neighbours · 60 insertion edges',
         8.2, ha='center', bbox={'facecolor': '#EDF0F3', 'edgecolor': 'none', 'pad': 5})
    for i, doi in enumerate(CASES):
        row = papers.loc[papers.doi.eq(doi)].iloc[0]
        x = .674+i*.155; w = .148
        accent, pale = RED, '#F8E9EC'
        fig.add_artist(FancyBboxPatch((x, .693), w, .166,
            boxstyle='round,pad=0,rounding_size=0.002', transform=fig.transFigure,
            facecolor='white', edgecolor='#D9A3AC', lw=.9, zorder=-1))
        fig.add_artist(Rectangle((x, .837), w, .022, transform=fig.transFigure,
                                facecolor=pale, edgecolor='none'))
        fig.add_artist(FancyBboxPatch((x+.004, .840), .017, .016,
            boxstyle='round,pad=0,rounding_size=0.006', transform=fig.transFigure,
            facecolor=accent, edgecolor='none'))
        text(fig, x+.0125, .843, 'AB'[i], 12, True, color='white', ha='center')
        text(fig, x+.024, .844, ['Denser reference neighbourhood',
                                'Sparser reference neighbourhood'][i], 7.6, True, color=INK)
        for j in range(6):
            xx = x+.006+(j%3)*.046; yy = .810-(j//3)*.025
            role = 'FINDING' if j < 4 else ('MECHANISM' if i == 0 else 'METHOD')
            fill = {'FINDING': '#DBEAF6', 'MECHANISM': '#DBEEE6', 'METHOD': '#E9E1F3'}[role]
            fig.add_artist(Rectangle((xx, yy), .043, .022, transform=fig.transFigure,
                                    facecolor=fill, edgecolor='white', lw=.6))
            code = f'F{j+1}' if j < 4 else f'{"M" if i == 0 else "T"}{j-3}'
            text(fig, xx+.0215, yy+.013, code, 7.5, True, ha='center')
            text(fig, xx+.0215, yy+.0035, role, 5.9, ha='center')
        metrics = [('Reference edges', f'{row.historical_edge_count:.0f}'),
                   ('Communities reached', f'{row.community_count:.0f}'),
                   ('Components merged', f'{row.component_merge_count:.0f}'),
                   ('Newly connected pairs', f'{row.newly_connected_pair_ratio:.1%}'),
                   ('Local efficiency', f'{row.efficiency_before:.3f} → {row.efficiency_after:.3f}'),
                   ('ΔE', f'+{row.efficiency_change:.3f}')]
        for k, (name, val) in enumerate(metrics):
            y = .773-k*.013
            if k % 2 == 0:
                fig.add_artist(Rectangle((x+.006, y-.003), w-.012, .013,
                    transform=fig.transFigure, facecolor='#F4F5F7', edgecolor='none', zorder=-.5))
            text(fig, x+.010, y, name, 7.4, bold=k == 5)
            text(fig, x+w-.010, y, val, 7.4, bold=k == 5, ha='right', color=accent if k == 5 else INK)
        text(fig, x+w/2, .679, doi, 6.8, ha='center', color=MUTED)


def panel_a(fig: plt.Figure, data: dict[str, pd.DataFrame]) -> None:
    papers = data['papers'].loc[data['papers'].journal.eq('Nature')]
    card(fig, 'a', 'Nature main journal: contribution insertion and local structure')
    panel_a_scatter(fig, papers)
    panel_a_raincloud(fig, papers)
    panel_a_cases(fig, papers)


def journal_size(n: Any) -> Any:
    return 2.0*np.sqrt(n)


def annotate_journals(ax: plt.Axes, sub: pd.DataFrame, field: str) -> None:
    positions = {'Nature': (.45, .88), 'Science': (.26, .58), 'Cell': (.57, .72),
                 'Proceedings of the National Academy of Sciences': (.26, .39),
                 'Nature Communications': (.20, .15), 'Scientific Reports': (.16, .05)}
    if field == 'materials':
        positions.update({'Nature': (.47, .84), 'Science': (.36, .28),
                          'Proceedings of the National Academy of Sciences': (.35, .55),
                          'Nature Communications': (.12, .43), 'Scientific Reports': (.15, .12)})
    if field == 'physics':
        positions.update({'Nature': (.45, .90), 'Science': (.38, .68),
                          'Proceedings of the National Academy of Sciences': (.31, .35),
                          'Nature Communications': (.16, .50), 'Scientific Reports': (.12, .17)})
    short = {'Proceedings of the National Academy of Sciences': 'PNAS',
             'Nature Communications': 'Nat. Commun.', 'Scientific Reports': 'Sci. Rep.'}
    for name, (tx, ty) in positions.items():
        match = sub[sub.journal.eq(name)]
        if match.empty:
            continue
        row = match.iloc[0]; s = journal_size(row.n)
        nature = name == 'Nature'
        ax.scatter(row.topic_diversity, row.international_collaboration_rate*100,
                   s=max(s, 22)+18, facecolor=RED if nature else (PALE_RED if row.publisher_group == 'Nature Portfolio' else 'white'),
                   edgecolor=RED if nature else INK, lw=1.1, zorder=7)
        if nature:
            ax.scatter(row.topic_diversity, row.international_collaboration_rate*100,
                       s=max(s, 22)+55, facecolor='none', edgecolor=RED, lw=.6, zorder=7)
        name_text = short.get(name, name)
        if row.n < 30:
            name_text += f' (n={row.n})'
        ax.annotate(name_text, (row.topic_diversity, row.international_collaboration_rate*100),
                    xytext=(tx, ty), textcoords='axes fraction', fontsize=7.5,
                    weight='bold' if nature else 'normal', color=RED if nature else INK,
                    arrowprops={'arrowstyle': '-', 'color': '#85909B', 'lw': .55}, zorder=9,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .6, 'alpha': .85})


def panel_b_journals(fig: plt.Figure, journals: pd.DataFrame) -> None:
    subtitle(fig, .024, .597, 'b1', 'Journal positions across three fields')
    for k, (field, name, col) in enumerate(zip(FIELDS, FIELD_NAMES, COLORS)):
        x = .057+k*.174
        ax = fig.add_axes([x, .407, .151, .161])
        sub = journals[journals.field.eq(field)].dropna(subset=['topic_diversity'])
        small, main = sub[sub.n.lt(30)], sub[sub.n.ge(30)]
        ax.scatter(small.topic_diversity, small.international_collaboration_rate*100,
                   s=3, c='#8A949E', alpha=.12, edgecolors='none', rasterized=True)
        for brand, color in [('non Nature Portfolio', '#A5ADB5'), ('Nature Portfolio', PALE_RED)]:
            s = main[main.publisher_group.eq(brand)]
            ax.scatter(s.topic_diversity, s.international_collaboration_rate*100,
                       s=journal_size(s.n), color=color, alpha=.50 if brand.startswith('non') else .7,
                       edgecolors='white', lw=.25, rasterized=True)
        unknown = main[~main.publisher_group.isin(['non Nature Portfolio', 'Nature Portfolio'])]
        ax.scatter(unknown.topic_diversity, unknown.international_collaboration_rate*100,
                   s=journal_size(unknown.n), facecolors='none', edgecolors='#8D969F',
                   lw=.5, alpha=.65, rasterized=True)
        annotate_journals(ax, sub, field)
        ax.set(xlim=(-.015, 1.025), ylim=(-2, 102), xticks=[0, .5, 1], yticks=[0, 25, 50, 75, 100],
               xlabel='Topic diversity')
        ax.xaxis.label.set_fontsize(9)
        if k == 0:
            ax.set_ylabel('International collaboration (%)', fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        style_axis(ax)
        fig.add_artist(Rectangle((x, .574), .151, .018, transform=fig.transFigure,
                                facecolor=col, edgecolor='none'))
        text(fig, x+.0755, .579, name, 9, True, color='white', ha='center')
    leg = fig.add_axes([.568, .402, .105, .176]); leg.axis('off'); leg.set(xlim=(0, 1), ylim=(0, 1))
    entries = [('Nature (main journal)', RED, True), ('Other Nature Portfolio', PALE_RED, True),
               ('Other journal', '#A5ADB5', True), ('Unconfirmed / mixed', '#8D969F', False)]
    for k, (name, color, filled) in enumerate(entries):
        leg.scatter(.07, .96-k*.105, s=30, facecolor=color if filled else 'none', edgecolor=color, lw=.7)
        leg.text(.16, .96-k*.105, name, fontsize=7.4, va='center')
    leg.text(.03, .49, 'Sample paper count', fontsize=8)
    for k, n in enumerate([30, 300, 3000, 12000]):
        yy = .38-k*.105
        leg.scatter(.17, yy, s=journal_size(n), color='#A5ADB5', alpha=.6, edgecolor='white', lw=.3)
        leg.text(.40, yy, f'{n:,}', fontsize=7.5, va='center')


def draw_journal_distribution(ax: plt.Axes, data: pd.DataFrame, metric: str,
                              factor: float, rng: np.random.Generator,
                              label_rows: bool) -> None:
    order = [name for name in ['Nature', 'Science', 'Cell', 'PNAS']
             if data.journal.eq(name).any()]
    for j, name in enumerate(order):
        vals = data.loc[data.journal.eq(name), metric].dropna().to_numpy()*factor
        y = len(order)-1-j; color = RED if name == 'Nature' else '#7B8E9F'
        small = len(vals) < 30
        ax.scatter(vals, y+rng.uniform(-.17, .17, len(vals)),
                   s=13 if small else 1.1, c=color, alpha=.95 if small else .20,
                   edgecolors='white' if small else 'none', linewidths=.3,
                   rasterized=not small, zorder=3)
        if not small:
            q10, q25, med, q75, q90 = np.quantile(vals, [.1, .25, .5, .75, .9])
            ax.plot([q10, q90], [y, y], lw=.8, color=color)
            ax.plot([q25, q75], [y, y], lw=3.2, color=color, alpha=.65,
                    solid_capstyle='butt')
            ax.scatter(med, y, s=14, facecolor='white', edgecolor=color, lw=1, zorder=6)
        if label_rows:
            ax.text(-.065, y, f'{name} (n={len(vals):,})',
                    transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=7, weight='bold' if name == 'Nature' else 'normal',
                    color=RED if name == 'Nature' else INK)
    ax.set(ylim=(-.5, len(order)-.5), yticks=[])


def panel_b_distributions(fig: plt.Figure, papers: pd.DataFrame) -> None:
    subtitle(fig, .688, .597, 'b2', 'Four-journal comparison by field')
    rng = np.random.default_rng(20261029)
    specs = [('newly_connected_pair_ratio', 100, (0, 100), [0, 50, 100], 'Newly connected pairs (%)'),
             ('efficiency_change', 1, (0, .52), [0, .25, .5], 'ΔE')]
    row_layout = [(.501, .060, .566), (.440, .043, .488), (.378, .043, .426)]
    for i, (field, name, color) in enumerate(zip(FIELDS, FIELD_NAMES, COLORS)):
        bottom, height, heading = row_layout[i]
        text(fig, .692, heading, name, 8, True, color=color)
        if not papers.loc[papers.field.eq(field), 'journal'].eq('Cell').any():
            text(fig, .772, heading, 'Cell: no sample', 7, color=INK)
        for k, (metric, factor, lim, ticks, title) in enumerate(specs):
            x = .747+k*.126
            ax = fig.add_axes([x, bottom, .100, height])
            draw_journal_distribution(ax, papers[papers.field.eq(field)], metric,
                                      factor, rng, label_rows=k == 0)
            ax.set(xlim=lim, xticks=ticks)
            style_axis(ax, 'x')
            ax.tick_params(axis='x', labelsize=7, labelbottom=i == 2,
                           length=2 if i == 2 else 0, pad=2)
            if i == 0:
                fig.add_artist(Rectangle((x, .578), .100, .013, transform=fig.transFigure,
                                        facecolor='#EDF0F3', edgecolor='none'))
                text(fig, x+.050, .582, title, 7.3, ha='center')
    handles = [Line2D([], [], color='#7B8E9F', lw=.8, label='10–90%'),
               Line2D([], [], color='#7B8E9F', lw=3.2, alpha=.65, label='25–75%'),
               Line2D([], [], marker='o', color='#7B8E9F', mfc='white', lw=0, ms=3, label='Median')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.846, .351), ncol=3,
               frameon=False, fontsize=7, handlelength=1.3, columnspacing=1)


def panel_b(fig: plt.Figure, data: dict[str, pd.DataFrame]) -> None:
    card(fig, 'b', 'Across-journal comparison: topic structure, collaboration and graph changes')
    panel_b_journals(fig, data['journals'])
    panel_b_distributions(fig, data['papers'])


def overall(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.field.eq('all') & frame.online_year.astype(str).eq('all')].copy()


def field_counts(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    sub = frame[frame.field.isin(FIELDS) & frame.online_year.astype(str).eq('all')]
    return sub.pivot(index='entity_id', columns='field', values='full_count').reindex(
        index=ids, columns=FIELDS).fillna(0)


def panel_c_countries(fig: plt.Figure, data: dict[str, pd.DataFrame]) -> None:
    subtitle(fig, .024, .283, 'c1', 'Top 12 countries / regions')
    top = data['country_top'].sort_values('full_rank')
    counts = field_counts(data['countries'], top.entity_id.tolist())
    ax = fig.add_axes([.079, .079, .128, .155]); left = np.zeros(len(top))
    for field, col in zip(FIELDS, COLORS):
        vals = counts[field].to_numpy()/1000
        ax.barh(range(len(top)), vals, left=left, color=col, height=.64, edgecolor='white', lw=.2)
        left += vals
    ax.set(yticks=range(len(top)), yticklabels=[COUNTRY_NAMES[x] for x in top.entity_id],
           xlim=(0, 140), xticks=[0, 50, 100], ylim=(11.7, -.7), xlabel='Sample papers (×10³)')
    ax.xaxis.label.set_fontsize(8)
    ax.tick_params(axis='y', length=0, labelsize=8)
    style_axis(ax, 'x')
    for k, n in enumerate(top.full_count):
        ax.text(n/1000+2, k, f'{n/1000:.1f}', fontsize=7, va='center', color=INK)
    coop = fig.add_axes([.237, .079, .044, .155])
    coop.scatter(top.international_collaboration_rate*100, range(len(top)), s=23,
                 c=COLORS[0], edgecolor='white', lw=.4)
    coop.set(xlim=(0, 100), xticks=[0, 50, 100], yticks=[], ylim=(11.7, -.7))
    style_axis(coop, 'x')
    text(fig, .259, .241, 'International\ncollaboration (%)', 7.3, ha='center', linespacing=1.1)
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in zip(FIELD_NAMES, COLORS)],
               loc='center', bbox_to_anchor=(.152, .257), ncol=3, frameon=False,
               fontsize=7.5, handlelength=1, columnspacing=1)


def panel_c_table(fig: plt.Figure, data: dict[str, pd.DataFrame]) -> None:
    subtitle(fig, .306, .283, 'c2', 'Top 15 institutions')
    top = data['institution_top'].sort_values('full_rank')
    counts = field_counts(data['institutions'], top.entity_id.tolist())
    shares = counts.to_numpy()/top.full_count.to_numpy()[:, None]
    ax = fig.add_axes([.303, .060, .423, .175]); ax.axis('off')
    ax.set(xlim=(0, 1), ylim=(15.5, -.5))
    for k, (_, row) in enumerate(top.iterrows()):
        if k % 2 == 0:
            ax.axhspan(k-.48, k+.48, color='#F5F7F9', zorder=0)
        ax.text(.013, k, str(k+1), va='center', fontsize=7.5, color=INK)
        ax.text(.054, k, SHORT_INSTITUTIONS[k], va='center', fontsize=7.7)
        ax.text(.455, k, f'{row.full_count:,}', va='center', ha='right', fontsize=7.7)
        for j, col in enumerate(COLORS):
            ax.scatter(.523+j*.077, k, s=100*shares[k, j], color=col,
                       edgecolor='white', lw=.3, zorder=4)
        ax.text(.975, k, str(int(row.fractional_rank)), va='center', ha='center',
                fontsize=8, color=RED if abs(row.fractional_rank-row.full_rank) >= 10 else INK)
    for xpos in [.04, .468, .724, .929]:
        ax.plot([xpos, xpos], [-.5, 14.5], lw=.45, color=GRID)
    cols = [( .054, 'Institution'), (.455, 'Paper\ncount'), (.523, 'Bio.'),
            (.600, 'Mat.'), (.677, 'Phys.'), (.975, 'Fractional\nrank')]
    for xpos, name in cols:
        text(fig, .303+.423*xpos, .244, name, 7.4, ha='left' if xpos == .054 else 'center')
    text(fig, .303+.423*.600, .262, 'Field share', 7.8, ha='center')
    coop = fig.add_axes([.303+.423*.753, .060, .423*.143, .175])
    coop.set_facecolor('none')
    coop.scatter(top.international_collaboration_rate*100, np.arange(15),
                 s=16, c=COLORS[0], edgecolors='white', lw=.3)
    coop.set(xlim=(0, 100), ylim=(15.5, -.5), xticks=[0, 50, 100], yticks=[])
    style_axis(coop, 'x'); coop.tick_params(labelsize=7)
    text(fig, .303+.423*.825, .244, 'International\ncollaboration (%)', 7.1, ha='center')
    leg = fig.add_axes([.482, .030, .137, .022]); leg.axis('off'); leg.set(xlim=(0, 1), ylim=(0, 1))
    for k, share in enumerate([.1, .5, 1]):
        leg.scatter(.07+k*.32, .45, s=100*share, color='#AAB6C1')
        leg.text(.14+k*.32, .45, f'{share:.0%}', va='center', fontsize=7)


def institution_type(value: Any) -> str:
    mapping = {'education': 'Education', 'healthcare': 'Healthcare', 'facility': 'Facility',
               'company': 'Company', 'government': 'Government-type'}
    if pd.isna(value) or value == 'unknown':
        return 'Unclassified'
    return mapping.get(value, 'Other')


def panel_c_all(fig: plt.Figure, data: dict[str, pd.DataFrame]) -> None:
    subtitle(fig, .739, .283, 'c3', 'All institutions')
    inst = overall(data['institutions']); inst['type_label'] = inst.entity_type.map(institution_type)
    ax = fig.add_axes([.774, .079, .157, .155])
    for name, col in TYPE_COLORS.items():
        sub = inst[inst.type_label.eq(name)]
        ax.scatter(sub.full_count, sub.international_collaboration_rate*100,
                   s=3, color=col, alpha=.13, edgecolors='none', rasterized=True)
        main = sub[sub.full_count.ge(30)]
        ax.scatter(main.full_count, main.international_collaboration_rate*100,
                   s=5, color=col, alpha=.44, edgecolors='none', rasterized=True)
    top = data['institution_top']
    ax.scatter(top.full_count, top.international_collaboration_rate*100,
               s=27, facecolors='none', edgecolors=INK, lw=.45, zorder=5)
    labels = [('Harvard University', 'Harvard', (.54, .89)),
              ('Massachusetts General Hospital', 'MGH', (.35, .78)),
              ('Lawrence Berkeley National Laboratory', 'LBNL', (.30, .66)),
              ('BGI Group (China)', 'BGI', (.13, .46)),
              ('Chinese Academy of Sciences', 'CAS', (.71, .13)),
              ('Centre National de la Recherche Scientifique', 'CNRS', (.71, .72))]
    for name, short, pos in labels:
        row = inst[inst.name.eq(name)].iloc[0]
        ax.annotate(short, (row.full_count, row.international_collaboration_rate*100),
                    xytext=pos, textcoords='axes fraction', fontsize=7, weight='bold',
                    arrowprops={'arrowstyle': '-', 'lw': .55, 'color': INK},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': .5}, zorder=7)
    ax.set(xscale='log', xlim=(.8, 25000), ylim=(-2, 102),
           xticks=[1, 10, 100, 1000, 10000], yticks=[0, 25, 50, 75, 100],
           xlabel='Sample paper count', ylabel='International collaboration (%)')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_xlabel('Sample paper count (log scale)', fontsize=8)
    ax.set_ylabel('International collaboration (%)', fontsize=8)
    style_axis(ax)
    for k, (name, col) in enumerate(TYPE_COLORS.items()):
        yy = .227-k*.018
        fig.add_artist(Line2D([.942], [yy], marker='o', markersize=4,
                             color=col, linestyle='', transform=fig.transFigure))
        text(fig, .948, yy-.0025, name, 6.5)


def panel_c(fig: plt.Figure, data: dict[str, pd.DataFrame]) -> None:
    card(fig, 'c', 'Countries and institutions: research participation and collaboration')
    panel_c_countries(fig, data)
    panel_c_table(fig, data)
    panel_c_all(fig, data)


def save_figure(fig: plt.Figure, dpi: int) -> None:
    final, panels = OUT / 'final', OUT / 'panels'
    final.mkdir(parents=True, exist_ok=True); panels.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(final / f'Fig8.{ext}', dpi=dpi)
    with plt.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(final / 'Fig8_outlined.svg', dpi=dpi)
    shutil.copyfile(final / 'Fig8.png', final / 'Fig8_600dpi.png')
    width, height = fig.get_size_inches()
    for panel, (x0, y0, x1, y1) in BOUNDS.items():
        bounds = Bbox.from_extents((x0-.004)*width, (y0-.004)*height,
                                  (x1+.004)*width, (y1+.004)*height)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(panels / f'Fig8{panel}.{ext}', dpi=dpi, bbox_inches=bounds)
    delivery = ROOT / 'outputs/fig1-8_reference_final_20261004/Fig8'
    if delivery.is_dir():
        for path in final.iterdir():
            target = delivery / path.name
            if target.exists():
                shutil.copyfile(path, target)
        for path in panels.iterdir():
            if (delivery / 'panels' / path.name).exists():
                shutil.copyfile(path, delivery / 'panels' / path.name)
        preview = delivery.parent / 'previews/Fig8.png'
        if preview.exists():
            shutil.copyfile(final / 'Fig8.png', preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true', help='Read current prepared tables (also the default).')
    parser.add_argument('--preview-only', action='store_true', help='Render the current PNG for layout adjustment.')
    args = parser.parse_args()
    configure()
    for name in ['times.ttf', 'timesbd.ttf']:
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': 'Arial', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    data = load_data()
    fig = plt.figure(figsize=(18, 12.9), facecolor='white')
    text(fig, .014, .973, 'Fig. 8 | Journal-level local knowledge-structure changes and global research participation',
         21, True, fontfamily='Times New Roman', color='#152B43')
    panel_a(fig, data); panel_b(fig, data); panel_c(fig, data)
    if args.preview_only:
        (OUT / 'final').mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / 'final/Fig8.png', dpi=180)
    else:
        save_figure(fig, 600)
    plt.close(fig)
    print(f'Fig8 overwritten: {OUT / "final/Fig8.png"}')


if __name__ == '__main__':
    main()

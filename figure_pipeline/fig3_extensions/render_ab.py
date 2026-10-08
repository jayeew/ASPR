"""Independent, offline Fig3 supplements; never imports the current renderer."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/fig3_reference/study'
OUT = ROOT / 'outputs/fig3_reference/extensions'
COLORS = {'gear': '#E16C35', 'graph': '#1269DF', 'fusion': '#713BCB'}
NAMES = {'gear': 'GEAR', 'graph': 'Graph', 'fusion': 'Full'}
KINDS = ['historical_increment', 'cross_work_relation', 'cross_contribution', 'scope_correction']
KIND_NAMES = ['Historical increment', 'Cross-work relation', 'Cross-contribution', 'Scope correction']
KIND_COLORS = ['#C68A31', '#187CA2', '#71843F', '#9160AC']


def configure() -> None:
    font = Path('/mnt/c/Windows/Fonts/arial.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11,
                         'axes.labelcolor': '#303744', 'text.color': '#243044',
                         'axes.edgecolor': '#A3ACB8', 'axes.linewidth': .7,
                         'xtick.color': '#536070', 'ytick.color': '#536070',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'})


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in ('svg', 'pdf', 'png'):
        path = OUT / f'{name}.{suffix}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=220, facecolor='white')
    plt.close(fig)


def clean(ax: plt.Axes) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)


def paired_samples() -> tuple[list[dict], list[dict]]:
    source = pd.read_csv(SOURCE / 'derived/paper_metrics.csv')
    source = source[source.metric.isin(['recall', 'precision'])]
    wide = source.pivot(index='paper_id', columns=['method', 'metric'], values='value')
    summaries, rows = [], []
    for index, method in enumerate(('gear', 'graph')):
        columns = [(m, t) for m in ('fusion', method) for t in ('recall', 'precision')]
        selected = wide[columns].dropna()
        delta = np.column_stack([selected['fusion', t] - selected[method, t]
                                 for t in ('recall', 'precision')]) * 100
        rng = np.random.default_rng(20260922 + index)
        bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        record = {'comparator': method, 'n': len(delta), 'mean': delta.mean(axis=0),
                  'bootstrap': bootstrap, 'ci': np.quantile(bootstrap, [.025, .975], axis=0)}
        summaries.append(record)
        for paper, values in zip(selected.index, delta):
            rows.append({'paper_id': paper, 'comparator': method,
                         'delta_recall_pp': values[0], 'delta_precision_pp': values[1]})
    return summaries, rows


def density_cloud(ax: plt.Axes, record: dict) -> None:
    samples = record['bootstrap']
    xx, yy = np.meshgrid(np.linspace(-5, 42, 180), np.linspace(-16, 24, 160))
    density = gaussian_kde(samples.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = [ordered[np.searchsorted(cumulative, mass)] for mass in (.95, .5)]
    ax.contourf(xx, yy, density, levels=[levels[0], levels[1], density.max()*1.01],
                colors=['#ECE5F7', '#D5C3EF'], zorder=1)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, c=COLORS['fusion'],
               alpha=.045, edgecolors='none', rasterized=True, zorder=2)
    ax.contour(xx, yy, density, levels=levels, colors=COLORS['fusion'],
               linewidths=[1.0, 1.5], linestyles=['--', '-'], zorder=3)
    ax.scatter(*record['mean'], s=92, marker='D', facecolor=COLORS['fusion'],
               edgecolor='#222733', linewidth=1.1, zorder=5)


def render_a() -> list[dict]:
    summaries, rows = paired_samples()
    pd.DataFrame(rows).to_csv(OUT / 'A_paired_paper_deltas.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.77, wspace=.13)
    fig.text(.035, .95, 'A', fontsize=23, fontweight='bold')
    fig.text(.075, .95, 'Paired gains in contribution identification', fontsize=18, fontweight='bold')
    fig.text(.075, .89, 'Joint paper bootstrap  |  10,000 resamples  |  Complete pairs for both metrics', fontsize=11)
    for ax, record in zip(axes, summaries):
        clean(ax)
        ax.grid(color='#E9ECF0', linewidth=.65)
        ax.axhline(0, color='#637080', lw=1)
        ax.axvline(0, color='#637080', lw=1)
        density_cloud(ax, record)
        ax.set(xlim=(-5, 42), ylim=(-16, 24), xticks=[0, 10, 20, 30, 40],
               yticks=[-10, 0, 10, 20], xlabel='Recall gain (percentage points)')
        ax.set_title(f"Full − {NAMES[record['comparator']]}   ·   n = {record['n']} papers",
                     loc='left', fontsize=13, color=COLORS[record['comparator']], pad=12, fontweight='bold')
        ax.text(.97, .95, f"Mean gain\nRecall {record['mean'][0]:+.1f} pp\nPrecision {record['mean'][1]:+.1f} pp",
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': 3})
    axes[0].set_ylabel('Precision gain (percentage points)')
    handles = [Line2D([], [], marker='D', color='none', markerfacecolor=COLORS['fusion'],
                      markeredgecolor='#222733', label='Observed paired mean'),
               Line2D([], [], color=COLORS['fusion'], lw=1.5, label='50% bootstrap density region'),
               Line2D([], [], color=COLORS['fusion'], ls='--', label='95% bootstrap density region')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .055), ncol=3, frameon=False, fontsize=10)
    save_figure(fig, 'A_paired_gains')
    return [{'comparator': r['comparator'], 'n_papers': r['n'],
             'mean_recall_gain_pp': float(r['mean'][0]), 'mean_precision_gain_pp': float(r['mean'][1]),
             'recall_ci95_pp': r['ci'][:, 0].tolist(), 'precision_ci95_pp': r['ci'][:, 1].tolist()}
            for r in summaries]


def cluster_rows() -> tuple[pd.DataFrame, int]:
    paths = sorted((SOURCE / 'annotations/clusters/papers').glob('*.json'))
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        if set(data.get('missing_methods', [])) & set(NAMES):
            raise ValueError(f'Target method missing: {path}')
        eligible = set(data['eligible_unit_keys'])
        importance = {r['cluster_id']: r['important'] for r in data['importance']}
        for cluster in data['clusters']:
            present = {data['mapping'][key].split('/')[0] for key in cluster['unit_keys'] if key in eligible}
            members = [m for m in NAMES if m in present]
            if members:
                records.append({'paper_id': path.stem, 'cluster_id': cluster['cluster_id'],
                                'membership': '+'.join(members), 'kind': cluster['kind'],
                                'important': importance[cluster['cluster_id']]})
    result = pd.DataFrame(records)
    if result.duplicated(['paper_id', 'cluster_id']).any():
        raise ValueError('Duplicate within-paper cluster key')
    return result, len(paths)


def cluster_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for membership, frame in rows.groupby('membership'):
        record = {'membership': membership, 'clusters': len(frame),
                  'papers': frame.paper_id.nunique(), 'important': int(frame.important.sum()),
                  'important_percent': frame.important.mean()*100}
        record.update({kind: int((frame.kind == kind).sum()) for kind in KINDS})
        records.append(record)
    return pd.DataFrame(records).sort_values('clusters', ascending=False).reset_index(drop=True)


def membership_matrix(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.set(xlim=(-.6, 6.6), ylim=(2.5, -.5), yticks=range(3), yticklabels=list(NAMES.values()))
    for x, row in summary.iterrows():
        selected = [i for i, m in enumerate(NAMES) if m in row.membership.split('+')]
        ax.scatter([x]*3, range(3), s=53, color='#E4E7ED', zorder=2)
        ax.plot([x, x], [min(selected), max(selected)], color='#556070', lw=2.0, zorder=3)
        for y in selected:
            ax.scatter(x, y, s=90, color=COLORS[list(NAMES)[y]], edgecolor='white', lw=.5, zorder=4)
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.spines[:].set_visible(False)
    for label, color in zip(ax.get_yticklabels(), COLORS.values()):
        label.set_color(color)
        label.set_fontweight('bold')


def render_b() -> dict:
    rows, n_papers = cluster_rows()
    summary = cluster_summary(rows)
    rows.to_csv(OUT / 'B_cluster_memberships.csv', index=False)
    summary.to_csv(OUT / 'B_intersection_statistics.csv', index=False)
    fig = plt.figure(figsize=(12.6, 10.1))
    grid = fig.add_gridspec(4, 1, left=.19, right=.96, top=.82, bottom=.095,
                          height_ratios=[2.0, 1.05, 2.0, 1.15], hspace=.26)
    axes = [fig.add_subplot(grid[i]) for i in range(4)]
    fig.text(.035, .952, 'B', fontsize=23, fontweight='bold')
    fig.text(.075, .952, 'Information intersections and attributes', fontsize=18, fontweight='bold')
    fig.text(.075, .90, f'{n_papers} papers  |  {len(rows):,} distinct eligible clusters  |  Exclusive membership among GEAR, Graph and Full', fontsize=11)
    x = np.arange(7)
    for ax in axes:
        ax.set_xlim(-.6, 6.6)
        ax.set_xticks([])
        for position in x:
            ax.axvspan(position-.46, position+.46, color='#F7F8FB', zorder=0)
    ax = axes[0]
    clean(ax)
    ax.spines['bottom'].set_visible(False)
    ax.set(ylim=(0, 1700), yticks=[0, 500, 1000, 1500], ylabel='Eligible clusters')
    ax.grid(axis='y', color='#E8ECF1', lw=.65)
    for position, row in summary.iterrows():
        color = COLORS['fusion'] if 'fusion' in row.membership else '#6B7789'
        ax.plot([position, position], [0, row.clusters], color=color, lw=2)
        ax.scatter(position, row.clusters, color=color, s=70, zorder=3)
        ax.text(position, row.clusters+65, f'{row.clusters:,}', ha='center', fontweight='bold', fontsize=12)
        ax.text(position, 1630, f'{row.papers} papers', ha='center', fontsize=10, color='#586576')
    membership_matrix(axes[1], summary)
    ax = axes[2]
    ax.set(ylim=(3.6, -.6), yticks=range(4), yticklabels=KIND_NAMES)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    for position, row in summary.iterrows():
        for y, (kind, color) in enumerate(zip(KINDS, KIND_COLORS)):
            percent = row[kind] / row.clusters * 100
            ax.scatter(position-.12, y, s=percent*7, color=color, alpha=.8,
                       edgecolors='white', linewidths=.6)
            ax.text(position+.12, y, f'{percent:.0f}%', va='center', ha='left', fontsize=10)
    ax.text(-.19, 1.08, 'Type share', transform=ax.transAxes, fontweight='bold', fontsize=11)
    ax = axes[3]
    clean(ax)
    ax.spines['bottom'].set_visible(False)
    ax.set(ylim=(0, 114), yticks=[0, 50, 100], yticklabels=['0%', '50%', '100%'], ylabel='Important share')
    ax.grid(axis='y', color='#E8ECF1', lw=.65)
    for position, row in summary.iterrows():
        ax.plot([position, position], [0, row.important_percent], color='#AAB4C1', lw=1)
        ax.scatter(position, row.important_percent, s=55, marker='D', color='#3B495F', zorder=3)
        ax.text(position, row.important_percent+7, f'{row.important_percent:.1f}%', ha='center', fontsize=10)
    fig.text(.19, .035, 'Membership uses eligible assertions; absence does not imply no mention. Importance reuses existing AI labels.', fontsize=10, color='#526071')
    save_figure(fig, 'B_information_intersections')
    return {'papers': n_papers, 'distinct_eligible_clusters': len(rows),
            'method_counts': {m: int(rows.membership.str.split('+').apply(lambda v: m in v).sum()) for m in NAMES}}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Existing outputs will not be overwritten: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    a = render_a()
    b = render_b()
    with (OUT / 'AB_statistics.json').open('x') as stream:
        json.dump({'A': a, 'B': b}, stream, indent=2)
    print(json.dumps({'A': a, 'B': b}, indent=2))


if __name__ == '__main__':
    main()

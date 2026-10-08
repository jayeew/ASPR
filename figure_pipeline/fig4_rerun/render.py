"""Keep the reference cards; replace a–d with fixed-core outcomes only."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from figure_pipeline.fig1_reference.export import fonts
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_reference import render as ref

from .config import CONDITIONS, METRICS, Config

ENGLISH = ['Paper only', 'GEAR only', 'Graph only', 'Full system', '−Joint', '−Structure', '−Paths']


class Figure(ref.Figure):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.summary = pd.read_csv(config.output / 'condition_summary_common.csv').set_index('condition').loc[list(CONDITIONS)]
        self.n = int(pd.read_csv(config.output / 'condition_summary.csv').n_papers.max())
        self.common_n = int(self.summary.n_papers.max())
        super().__init__()
        for artist in self.fig.texts:
            if artist.get_gid().startswith('domain_new'):
                cohort = 'Main cohort · 100 enrolled' if (config.output / 'pilot_acceptance.json').exists() else f'Pilot · {self.n} papers'
                artist.set_text(f'{cohort} · a: common n={self.common_n}; b–d: complete pairs · fixed-core evaluation')

    def data(self, name: str) -> pd.DataFrame:
        if name == 'a_condition_endpoints':
            # Used only to reuse the left contrast-map geometry; replace its old table below.
            return self.summary.reset_index().rename(columns={'H': 'V', 'C': 'H', 'S': 'R'})
        return pd.read_csv(ref.OUT / 'data' / f'{name}.csv', escapechar='\\')


def panel_a(f: Figure) -> None:
    ref.panel_a(f)
    for ax in list(f.fig.axes):
        if ax.get_gid() == 'a2':
            ax.remove()
    for artist in list(f.fig.texts):
        if artist.get_gid().startswith('a2_'):
            artist.remove()
    for artist in f.fig.findobj(Text):
        replacements = {'T\nText': 'T\nPaper', 'Same sources and writing task': 'Same paper and report task',
                        'F−M · No metrics': 'F−M · −Structure', 'F−P · No paths': 'F−P · −Paths'}
        if artist.get_text() in replacements:
            artist.set_text(replacements[artist.get_text()])
    f.text('a2', 47.5, 11.5, 'Fixed-core report outcomes', 6.6, bold=True, ha='center')
    ax = f.ax('a2', (29.6, 17, 35.8, 29)); ax.axis('off'); ax.set(xlim=(0, 4.7), ylim=(7.5, -.9))
    for j, title in enumerate(['C ↑', 'H ↑', 'S ↑']):
        ax.add_patch(Rectangle((1.55 + j, -.9), 1, .9, fc=ref.HEADER, ec=ref.EDGE, lw=.4))
        ax.text(2 + j, -.45, title, ha='center', va='center', fontsize=7, weight='bold')
    for i, (_, row) in enumerate(f.summary.iterrows()):
        ax.add_patch(Rectangle((0, i), 1.55, 1, fc='#EDE9F5' if i == 3 else '#F0F3F6', ec=ref.EDGE, lw=.4))
        ax.text(.05, i + .5, ENGLISH[i], va='center', fontsize=6.5)
        for j, metric in enumerate(METRICS):
            value = row[metric]
            ax.add_patch(Rectangle((1.55 + j, i), 1, 1, fc=ref.mpl.colors.to_rgba(ref.PURPLE, .10) if i == 3 else ref.mpl.colors.to_rgba('#4184C0', .08 + .15 * value / 100) if pd.notna(value) else 'white', ec=ref.EDGE, lw=.4))
            ax.text(2 + j, i + .5, f'{value:.1f}' if pd.notna(value) else 'NA', ha='center', va='center', fontsize=6.5)
    ax.text(3, 7.3, 'All endpoints: %', ha='center', fontsize=6)
    f.text('a2', 29.7, 48.2, 'C: response · H: correct history', 6.2)
    f.text('a2', 29.7, 51.1, f'S: correct scope · common n={f.common_n}', 6.2, color=ref.GREY)


def panel_b(f: Figure) -> None:
    papers = pd.read_csv(f.config.output / 'component_effects_paper.csv')
    summary = pd.read_csv(f.config.output / 'component_effects_summary.csv')
    labels = ['GEAR', 'Graph\nbranch', 'Joint', 'Structural\nsummaries', 'Paper\npaths']
    for i, label in enumerate(labels):
        f.text('b_labels', 70, 20.6 + i * 6.65, label, 6.5, color=ref.COMPONENT_COLORS[i], bold=True)
    for index, metric in enumerate(METRICS, 1):
        gid = f'b{index}'; x, _, w, _ = ref.BOXES[gid]
        f.card(gid + '_frame', (x, 11, w, 41))
        f.text(gid, x + w / 2, 11.4, f'Δ{metric} (pp)', 6.5, bold=True, ha='center')
        ax = f.ax(gid, (x + 1, 18, w - 2, 33.5)); ax.set(xlim=(-105, 105), ylim=(4.53, -.50))
        ax.set_yticks([]); ax.set_xticks([-100, -50, 0, 50, 100]); ax.xaxis.tick_top()
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.axvline(0, color=ref.GREY, lw=.55, ls=(0, (2, 2)))
        for i, component in enumerate(ref.COMPONENTS):
            vals = papers[(papers.metric == metric) & (papers.component == component)].delta.dropna().to_numpy()
            row = summary[(summary.metric == metric) & (summary.component == component)].iloc[0]
            color = ref.COMPONENT_COLORS[i]
            ax.axhspan(i - .48, i + .48, color='#EFF4F8' if i % 2 == 0 else 'white', zorder=-3)
            ax.axhline(i + .50, color='#DAE5EE', lw=.45, zorder=-2)
            if len(vals):
                count, edges = np.histogram(vals, bins=[-125, -75, -25, 25, 75, 125])
                ax.bar((edges[:-1] + edges[1:]) / 2, -.23 * count / max(count.max(), 1), width=7, bottom=i + .06, color=color, alpha=.30, lw=0)
                px, py = ref.stacked_points(vals, 'H')
                ax.scatter(px, i + py, s=5 if len(vals) > 10 else 9, c=color, alpha=.65, edgecolors='white', lw=.1)
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.add_patch(Rectangle((q1, i - .045), q3 - q1, .09, fc='white', ec=ref.INK, lw=.5))
                ax.plot([med, med], [i - .065, i + .065], color=ref.INK, lw=.7)
                estimate = f'{row.mean_delta:+.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]'
            else:
                estimate = 'NA'
            ax.text(.98, i + .45, estimate.replace('-', '−'), transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=6)
    f.text('b_labels', 70, 52.3, f'Paired n ≤ {f.n}; positive = improvement · mean [95% paper-bootstrap CI]', 6.1)


def panel_c(f: Figure) -> None:
    stored = read(f.config.output / 'interaction_summary.json')
    values = stored['condition_means']
    if any(v is None for v in values.values()):
        f.text('c1', 4, 65, 'Four-condition complete pairs pending', 6.5)
        return
    xx, yy = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    interaction = values['F'] - values['E'] - values['G'] + values['T']
    additive = values['T'] + (values['E'] - values['T']) * xx + (values['G'] - values['T']) * yy
    ax = f.ax('c1', (3, 63, 39, 23), '3d')
    ax.plot_surface(xx, yy, additive + interaction * xx * yy, cmap=ref.DIVERGING, vmin=0, vmax=100, alpha=.85, linewidth=.15, edgecolor='#BCC5CA')
    ax.plot_wireframe(xx, yy, additive, color=ref.GREY, linewidth=.35, rstride=5, cstride=5)
    for name, x, y, color in [('T', 0, 0, ref.GREY), ('E', 1, 0, ref.COMPONENT_COLORS[0]), ('G', 0, 1, ref.COMPONENT_COLORS[1]), ('F', 1, 1, ref.PURPLE)]:
        ax.scatter(x, y, values[name], s=10, color=color, depthshade=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 100), xticks=[0, 1], yticks=[0, 1], zticks=[0, 50, 100])
    ax.set_zlabel('H (%)', labelpad=-5); ax.tick_params(pad=-2, labelsize=6.5)
    ax.view_init(elev=24, azim=-125); ax.set_proj_type('ortho'); ax.set_box_aspect((1, 1, .7))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False; axis._axinfo['grid']['linewidth'] = .3
    for name, x, y in [('T', 20, 79), ('E', 35, 74), ('G', 5, 67), ('F', 28, 65.7)]:
        f.text('c1', x, y, f'{name}={values[name]:.1f}', 6.5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': .2})
    f.text('c1', 4, 62, 'Correct history and additive reference', 6.2, bold=True)
    f.text('c1', 6, 87, 'Graph (0/1)', 6.5); f.text('c1', 26, 87, 'GEAR (0/1)', 6.5)
    f.text('c1', 3, 91, '4 means; interpolation only', 6.5)
    data = pd.read_csv(f.config.output / 'interaction_paper.csv').I.dropna().to_numpy()
    stats = stored['statistics']
    ax = f.ax('c2', (49, 69, 25, 15))
    ax.hist(data, bins=np.arange(-212.5, 238, 25), color=ref.PURPLE, alpha=.45, edgecolor='white', lw=.3)
    ax.axvspan(stats['low'], stats['high'], color=ref.PURPLE, alpha=.12, lw=0)
    ax.axvline(0, color=ref.GREY, lw=.5, ls='--'); ax.axvline(stats['estimate'], color=ref.PURPLE, lw=.8)
    ax.set(xlim=(-215, 215), xticks=[-200, 0, 200], ylabel='Papers')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, integer=True))
    f.text('c2', 45, 62, 'I = Full − GEAR − Graph + Paper', 6.1, bold=True)
    f.text('c2', 45, 65.4, f"{stats['estimate']:+.1f}; CI [{stats['low']:.1f}, {stats['high']:.1f}]", 6.5)
    f.text('c2', 49, 87, 'Interaction in H (pp)', 6.5)
    f.text('c2', 45, 91, f"+ / 0 / − : {stored['positive']} / {stored['zero']} / {stored['negative']}", 6.5)


def panel_d(f: Figure) -> None:
    data = pd.read_csv(f.config.output / 'joint_effects_paper.csv')
    complete = set(pd.read_csv(f.config.output / 'paper_metrics.csv').query("state == 'complete'").paper_id)
    data = data[data.paper_id.isin(complete)].sort_values(['J_topo', 'paper_id']).copy()
    data['rank'] = np.arange(1, len(data) + 1); data['percent'] = 100 * data.J_topo
    n = len(data)
    if not n:
        return
    ax = f.ax('d1', (89, 64, 90, 14))
    ax.vlines(data['rank'], 0, data.percent, color='#DEE5EB', lw=.4)
    ax.scatter(data['rank'], data.percent, c=data.percent, cmap=ref.DIVERGING, vmin=0, vmax=100, s=9, lw=0)
    ax.set(xlim=(.5, n + .5), ylim=(0, 100), yticks=[0, 50, 100], xticks=sorted({1, max(1, n // 2), n}), ylabel='J (%)')
    ax.grid(axis='y', color=ref.GRID, lw=.35)
    ax.text(.02, .96, f'Median {data.percent.median():.2f}%; range {data.percent.min():.2f}–{data.percent.max():.2f}%', transform=ax.transAxes, va='top', fontsize=6.5)
    f.text('d1', 89, 61.7, f'Joint-exclusive connectivity · {n} papers', 6.5, bold=True)
    ax = f.ax('d2', (89, 83, 90, 5)); ax.axis('off'); ax.set(xlim=(.5, n + .5), ylim=(2, 0))
    for i, col in enumerate(['H_F', 'H_noJ']):
        for row in data.itertuples():
            value = getattr(row, col)
            color = mpl.colors.to_rgba(ref.PURPLE, .12 + .88 * value / 100) if pd.notna(value) else '#E8E8E8'
            ax.add_patch(Rectangle((row.rank - .5, i + .1), .96, .72, fc=color, ec='white', lw=.2, hatch='..' if pd.isna(value) else None))
            if n <= 10:
                ax.text(row.rank, i + .46, f'{value:.0f}%' if pd.notna(value) else 'NA', va='center', ha='center', fontsize=6.3, color='white' if value == 100 else ref.INK)
    f.text('d2', 79, 83.1, 'Full', 6.5); f.text('d2', 79, 85.6, '−Joint', 6.5)
    f.text('d2', 89, 80.6, 'Same paper order · correct historical response H', 6.3)
    f.text('d2', 179, 89, 'H: 0% / 50% / 100% · missing: dotted', 6.3, ha='right')
    paired = data.delta_H.dropna()
    f.text('d2', 79, 92, f'Paired n={len(paired)}; ΔH > / = / < 0: {(paired > 0).sum()} / {(paired == 0).sum()} / {(paired < 0).sum()}', 6.5)


def render(config: Config) -> None:
    import cairosvg

    root = config.output / 'figure'; fonts(root)
    f = Figure(config)
    for draw in [panel_a, panel_b, panel_c, panel_d, ref.panel_e, ref.panel_f]:
        draw(f)
    for artist in f.fig.findobj(Text):
        artist.set_fontsize(artist.get_fontsize() * ref.SCALE)
    for artist in f.fig.findobj(PathCollection):
        artist.set_sizes(artist.get_sizes() * ref.SCALE * ref.SCALE)
    final = root / 'final'; final.mkdir(parents=True, exist_ok=True)
    master = final / 'Fig4.svg'; f.fig.savefig(master); f.fig.savefig(final / 'Fig4.pdf')
    with mpl.rc_context({'svg.fonttype': 'path'}):
        f.fig.savefig(final / 'Fig4_outlined.svg')
    for scale in (4, 6, 10):
        name = 'Fig4.png' if scale == 4 else f'Fig4_{scale}x.png'
        cairosvg.svg2png(url=str(master), write_to=str(final / name), scale=scale)
    for name, box in ref.PANELS.items():
        path = root / 'panels' / f'{name}.svg'
        ref.crop_svg(master, path, (box[0] - 1.2, box[1] - 1.6, box[2] + 2.4, box[3] + 2.4),
                     [name + '_card', name + '_title'] + [k for k in ref.BOXES if k.startswith(name)] + (['b_labels'] if name == 'b' else []))
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix('.pdf')))
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix('.png')), scale=6)
    for name, box in ref.BOXES.items():
        path = root / 'components' / f'{name}.svg'
        ref.crop_svg(master, path, ref.positioned_box(name, box), [name])
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix('.pdf')))
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix('.png')), scale=6)
    write(root / 'source_manifest.json', {'a_d': str(config.output), 'a_d_completed_papers': f.n,
          'e_f': str(ref.OUT / 'data'), 'e_f_scope': 'Earlier Fig3 descriptive reports; not rerun or pooled',
          'metrics': METRICS, 'canvas_mm': [ref.W * ref.SCALE, ref.H * ref.SCALE]})
    historical = root / 'historical_data'; historical.mkdir(exist_ok=True)
    for path in (ref.OUT / 'data').glob('*.csv'):
        if path.name.startswith(('e1_', 'e2_', 'f1_', 'f2_')):
            shutil.copy2(path, historical / path.name)
    plt.close(f.fig)
    package(config)
    print(f'Rendered {master}', flush=True)


def package(config: Config) -> None:
    shutil.copy2(Path(__file__).with_name('README.md'), config.output / 'README.md')
    archive = config.output / ('Fig4_delivery.zip' if (config.output / 'pilot_acceptance.json').exists() else 'Fig4_pilot_delivery.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(config.output.rglob('*')):
            relative = path.relative_to(config.output)
            if path.is_file() and (relative.parts[0] in {'figure', 'reports', 'evaluation'} or len(relative.parts) == 1 and path.suffix in {'.json', '.jsonl', '.csv', '.md'}):
                bundle.write(path, str(relative))
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.is_file():
                bundle.write(path, 'code/' + path.name)

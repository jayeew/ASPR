"""Render the compact two-row Fig.5 from the delivered CSV archive."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/fig5_mechanism_study/Fig5_formal_statistics.zip'
OUT = ROOT / 'fig_reference'
W, H = 420., 218.
INK, GRID = '#10213B', '#E4ECF2'
BLUE, TEAL, ORANGE, RED = '#1269DF', '#219D91', '#E16C35', '#E74C3C'
ASPECTS = ['all', 'historical_verification', 'knowledge_position',
           'joint_contribution', 'structural_resolution', 'citation_contact']
LABELS = ['Overall', 'Historical\nverification', 'Knowledge\nposition',
          'Joint\ncontribution', 'Structural\nresolution', 'Citation\ncontact']
CONDITIONS = ['original_analysis_original_writer', 'replacement_analysis_original_writer',
              'original_analysis_replacement_writer', 'replacement_analysis_replacement_writer']
COLORS = [BLUE, TEAL, ORANGE, RED]
NAMES = ['Original analysis + writing', 'Analysis model replaced',
         'Writing model replaced', 'Both models replaced']
STATES = ['Grounded correct answer', 'Justified withholding',
          'Other incomplete or incorrect answer', 'Scientific judgment unresolved', 'Unsupported assertion']
STATE_COLORS = ['#34A853', '#3498DB', '#B7B7B7', '#8E68C9', '#E53935']
STATE_LABELS = ['Grounded\ncorrect', 'Justified\nwithholding', 'Other incomplete\nor incorrect',
                'Scientific judgment\nunresolved', 'Unsupported\nassertion']
CMAP = LinearSegmentedColormap.from_list('reference_blue', ['#F6FAFD', '#B9D7EA', '#5E9BC6', '#155A94'])


class Figure:
    def __init__(self) -> None:
        for name in ['arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf']:
            path = Path('/mnt/c/Windows/Fonts') / name
            if path.exists():
                font_manager.fontManager.addfont(path)
        mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
                             'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                             'axes.linewidth': .65, 'text.color': INK, 'axes.labelcolor': INK,
                             'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': '#8295A7',
                             'svg.fonttype': 'none', 'pdf.fonttype': 42})
        self.fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
        self.tables: dict[str, pd.DataFrame] = {}
        self.provenance: dict[str, Any] = {}

    def data(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            with zipfile.ZipFile(SOURCE) as archive:
                raw = archive.read(name)
            self.tables[name] = pd.read_csv(io.BytesIO(raw))
            self.provenance[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(self.tables[name])}
        return self.tables[name]

    def text(self, x: float, y: float, value: str, size: float = 8,
             bold: bool = False, color: str = INK, **kw: Any) -> Any:
        return self.fig.text(x / W, 1 - y / H, value, fontsize=size, va='top',
                             weight='bold' if bold else 'normal', color=color, **kw)

    def ax(self, x: float, y: float, w: float, h: float) -> Axes:
        ax = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        ax.set_facecolor('none')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=2)
        return ax

    def card(self, key: str, box: tuple[float, float, float, float], title: str,
             edge: str, fill: str, ink: str) -> None:
        x, y, w, h = box
        self.fig.add_artist(FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H,
                           transform=self.fig.transFigure, boxstyle='round,pad=0,rounding_size=0.004',
                           facecolor='#FCFDFF', edgecolor=edge, linewidth=.75, zorder=-10))
        self.fig.add_artist(Rectangle(((x + .3) / W, 1 - (y + 9.5) / H), (w - .6) / W, 9.2 / H,
                           transform=self.fig.transFigure, facecolor=fill, edgecolor='none', zorder=-9))
        self.text(x + 3, y + 1, key, 20, True)
        self.text(x + 13, y + 2, title, 13.5, True, ink, fontfamily='Times New Roman')

    def resource(self, condition: str, metric: str, cohort: str = 'Model-stage diagnostic') -> pd.Series:
        data = self.data('figure_data/resource_summary.csv')
        return data[(data.cohort == cohort) & (data.condition == condition) & (data.metric == metric)].iloc[0]

    def model(self) -> None:
        self.text(9, 16, '20-paper factorial diagnostic', 9, True)
        self.text(106, 16, '100-paper cohort: analysis-model replacement', 9, True)
        self.text(191, 21, 'Mean [95% interval]', 6.7, True)
        ax = self.ax(30, 27, 63, 45)
        order = [CONDITIONS[0], CONDITIONS[2], CONDITIONS[1], CONDITIONS[3]]
        fills = [BLUE, '#F4B261', TEAL, RED]
        for k, (condition, color) in enumerate(zip(order, fills, strict=True)):
            x, y = k % 2, k // 2
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white', lw=.8))
            val = self.resource(condition, 'grounded_correct_percent')
            delta = self.resource(condition, 'coverage_change_percentage_points')['mean']
            tokens = self.resource(condition, 'processed_token_ratio')['mean']
            ink = INK if k == 1 else 'white'
            ax.text(x + .5, y + .22, f"{val['mean']:.1f}%", ha='center', va='center', fontsize=12, weight='bold', color=ink)
            ax.text(x + .5, y + .59, f"Change {delta:+.1f} points\nTokens {tokens:.2f}×\n{int(val['n'])} papers",
                    ha='center', va='center', fontsize=8.1, color=ink, linespacing=1.4)
        ax.set(xlim=(0, 2), ylim=(2, 0), xticks=[.5, 1.5], yticks=[.5, 1.5],
               xticklabels=['GPT-6.1 Sol', 'GPT-5.6 Luna'], yticklabels=['GPT-6.1 Sol', 'GPT-5.6 Luna'],
               xlabel='Writing model', ylabel='Analysis model')
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        changes = self.data('transfer_validation/paired_changes.csv')
        summaries = self.data('transfer_validation/paired_summary.csv')
        ax = self.ax(132, 25, 55, 48)
        rng = np.random.default_rng(20261004)
        for i, aspect in enumerate(ASPECTS):
            vals = changes[(changes.aspect == aspect) & (changes.metric == 'grounded_answer')].delta_percentage_points
            row = summaries[(summaries.aspect == aspect) & (summaries.metric == 'grounded_answer')].iloc[0]
            ax.scatter(vals, i + rng.uniform(-.18, .18, len(vals)), s=6, color='#6EACE0', alpha=.3, linewidths=0)
            ax.errorbar(row['mean'], i, xerr=[[row['mean'] - row['low']], [row['high'] - row['mean']]],
                        fmt='o', color='#12548E', ms=4, capsize=2, lw=1.3)
            self.text(191, 27 + i * 7.62, f"{row['mean']:+.2f} [{row['low']:+.2f}, {row['high']:+.2f}]\n{int(row['n'])} paired papers", 6.7)
        ax.set_xscale('symlog', linthresh=5, linscale=2, base=10)
        ax.axvspan(-5, 5, color='#EEF6FC', alpha=.5, zorder=-5)
        ax.axvline(0, color='#8B96A4', ls='--', lw=.7)
        ax.set(xlim=(-55, 55), ylim=(5.6, -.7), yticks=range(6), yticklabels=LABELS,
               xticks=[-50, -5, 0, 5, 50], xticklabels=['−50', '−5', '0', '+5', '+50'],
               xlabel='Coverage change (percentage points)\nLinear within ±5; compressed tails')
        ax.grid(axis='x', color=GRID, lw=.45)

    def evidence(self) -> None:
        self.text(224, 16, 'Mean paired coverage change · 20-paper subset', 9, True)
        self.text(372, 16, 'Half-source deletion\nThree deletion schemes', 8.1, True)
        data = self.data('figure_data/dose_summary.csv')
        selected = data[data.metric == 'paired_change_percentage_points']
        bound = max(60, int(np.ceil(-selected['mean'].min() / 10) * 10))
        norm = Normalize(0, bound)
        matrices = [('Historical sources', 'Historical sources retained', ['F', 'E75', 'E50', 'E25'], 255, 59),
                    ('Historical parent papers in graph', 'Graph parent papers retained', ['F', 'G50', 'G25'], 319, 44)]
        for index, (kind, label, conditions, x, width) in enumerate(matrices):
            self.text(x + width / 2, 24, label, 7.8, True, ha='center')
            ax = self.ax(x, 34, width, 38)
            vals = np.array([[selected[(selected.intervention == kind) & (selected.condition == c) &
                             (selected.aspect == a)]['mean'].item() for c in conditions] for a in ASPECTS])
            ax.imshow(np.maximum(-vals, 0), cmap=CMAP, norm=norm, aspect='auto')
            for i in range(6):
                for j in range(len(conditions)):
                    ax.text(j, i, f'{vals[i, j]:+.1f}' if vals[i, j] else '0.0', ha='center', va='center',
                            fontsize=7.8, color='white' if -vals[i, j] > bound * .6 else INK)
            ax.set(xticks=range(len(conditions)), xticklabels=['100%', '75%', '50%', '25%'] if index == 0 else ['100%', '50%', '25%'],
                   yticks=range(6), yticklabels=LABELS if index == 0 else [])
            ax.xaxis.tick_top()
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-.5, len(conditions), 1), minor=True)
            ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
            ax.grid(which='minor', color='white', lw=.65)
            ax.tick_params(which='minor', length=0)
            ax.spines[:].set_visible(False)
        cax = self.ax(272, 79, 74, 2)
        cb = self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal')
        cb.set_ticks(range(0, bound + 1, 20), labels=[str(-i) for i in range(0, bound + 1, 20)])
        cb.ax.tick_params(length=1.5, labelsize=6.5)
        cb.set_label('Coverage change (percentage points)', fontsize=7, labelpad=1)
        summaries = self.data('interventions/paired_summary.csv')
        ax = self.ax(375, 34, 34, 38)
        mask_colors = ['#155A94', '#5798CB', '#A6CEE8']
        lows, highs = [], []
        for i, aspect in enumerate(ASPECTS):
            for j, condition in enumerate(['E50', 'E50_SEED2', 'E50_SEED3']):
                # First scheme must use the same 20 papers as the other two.
                if j == 0:
                    row = selected[(selected.intervention == 'Historical sources') &
                                   (selected.condition == condition) & (selected.aspect == aspect)].iloc[0]
                else:
                    row = summaries[(summaries.condition == condition) & (summaries.aspect == aspect) &
                                    (summaries.metric == 'grounded_answer')].iloc[0]
                y = i + (j - 1) * .22
                ax.barh(y, row['mean'], height=.17, color=mask_colors[j], alpha=.8)
                ax.errorbar(row['mean'], y, xerr=[[row['mean'] - row['low']], [row['high'] - row['mean']]],
                            color=mask_colors[j], capsize=1, lw=.65)
                lows.append(row['low']); highs.append(row['high'])
        ax.set(ylim=(5.5, -.5), yticks=[], xlim=(min(-60, min(lows) - 3), max(2, max(highs) + 2)),
               xticks=[-60, -30, 0], xlabel='Change\n(percentage points)')
        ax.axvline(0, color='#8B96A4', lw=.6)
        for j, col in enumerate(mask_colors):
            self.fig.add_artist(Rectangle(((375 + j * 12) / W, 1 - 88 / H), 2 / W, 1.8 / H,
                                          transform=self.fig.transFigure, color=col, lw=0))
            self.text(378 + j * 12, 86.1, f'{j + 1}', 7)

    def states(self) -> None:
        rows = self.data('figure_data/target_state_counts.csv')
        n = int(rows.n.iloc[0])
        self.text(9, 109, f'State distribution across conditions · {n} paired papers', 9, True)
        self.text(9, 164, 'Per-paper changes: full evidence → critical-source deletion', 9, True)
        ax = self.ax(41, 121, 106, 28)
        conditions = ['F', 'NONCRITICAL', 'CRITICAL', 'RESTORE']
        for i, condition in enumerate(conditions):
            left = 0.
            for state, col in zip(STATES, STATE_COLORS, strict=True):
                count = int(rows[(rows.condition == condition) & (rows.state == state)]['count'].sum())
                if not count:
                    continue
                width = count / n * 100
                ax.barh(i, width, left=left, height=.7, color=col, edgecolor='white', lw=.6)
                label = f'{count} ({count / n:.0%})' if width > 90 else str(count)
                ax.text(left + width / 2, i, label, ha='center', va='center', fontsize=8,
                        color=INK if state == STATES[2] else 'white', weight='bold' if width > 90 else 'normal')
                left += width
        ax.set(xlim=(0, 100), ylim=(3.5, -.5), xticks=range(0, 101, 20), yticks=range(4),
               yticklabels=['Full evidence', 'Noncritical\nsources deleted', 'Critical\nsources deleted', 'Evidence restored'],
               xlabel='Share of papers (%)')
        ax.tick_params(axis='y', length=0)
        for j, (label, col) in enumerate(zip(STATES, STATE_COLORS, strict=True)):
            self.fig.add_artist(Rectangle((154 / W, 1 - (120 + j * 7) / H), 3.7 / W, 3.7 / H,
                                          transform=self.fig.transFigure, facecolor=col, edgecolor='none'))
            wrapped = label.replace('Other incomplete or incorrect answer', 'Other incomplete or\nincorrect answer').replace('Scientific judgment unresolved', 'Scientific judgment\nunresolved')
            self.text(160, 116.2 + j * 7, wrapped, 7.5)
        states = self.data('figure_data/target_states.csv')
        states = states[states.paired_complete]
        wide = states.pivot(index='paper_id', columns='condition', values='state')
        matrix = np.zeros((5, 5), dtype=int)
        for _, row in wide.iterrows():
            matrix[STATES.index(row['F']), STATES.index(row['CRITICAL'])] += 1
        assert int(matrix.sum()) == n
        ax = self.ax(50, 175, 143, 25)
        im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=max(20, int(matrix.max())), aspect='auto')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if matrix[i, j] > 12 else INK)
        ax.set(xticks=range(5), xticklabels=STATE_LABELS, yticks=range(5), yticklabels=STATE_LABELS)
        ax.tick_params(length=0, labelsize=6.5)
        ax.set_xticks(np.arange(-.5, 5), minor=True)
        ax.set_yticks(np.arange(-.5, 5), minor=True)
        ax.grid(which='minor', color='#CBD7E1', lw=.45)
        ax.tick_params(which='minor', length=0)
        self.text(46, 171, 'Full evidence', 7, ha='right')
        self.text(121.5, 210, 'After critical-source deletion', 8, ha='center')
        cb = self.fig.colorbar(im, cax=self.ax(201, 177, 2.4, 22))
        cb.set_ticks([0, 5, 10, 15, 20])
        self.text(202, 167, 'Number\nof papers', 7, ha='center')
        pd.DataFrame(matrix, index=STATES, columns=STATES).to_csv(OUT / 'data/transition_counts.csv')

    def resources(self) -> None:
        self.text(225, 109, 'Per-paper quality–resource trade-off', 9, True)
        self.text(225, 175, 'Configuration summary', 9, True)
        papers = self.data('figure_data/resource_papers.csv')
        ax = self.ax(241, 118, 161, 39)
        offsets = [(6, 7), (-55, 14), (8, -16), (8, -12)]
        table_rows = []
        for condition, col, name, offset in zip(CONDITIONS, COLORS, NAMES, offsets, strict=True):
            values = papers[(papers.cohort == 'Model-stage diagnostic') & (papers.condition == condition)]
            ax.scatter(values.processed_token_ratio, values.coverage_change_percentage_points, s=11,
                       color=col, alpha=.3, linewidths=0)
            tok = self.resource(condition, 'processed_token_ratio')['mean']
            quality = self.resource(condition, 'grounded_correct_percent')['mean']
            delta = self.resource(condition, 'coverage_change_percentage_points')['mean']
            ax.scatter(tok, delta, s=67, color=col, edgecolor='white', lw=.8, zorder=6)
            ax.annotate(f'{name}\n({quality:.1f}%, {tok:.2f}×)', (tok, delta), xytext=offset,
                        textcoords='offset points', fontsize=7.8, color=col, weight='bold')
            table_rows.append((name, quality, tok, delta, col, 'o'))
        condition = CONDITIONS[1]
        cohort = 'Unified main cohort'
        tok = self.resource(condition, 'processed_token_ratio', cohort)['mean']
        quality = self.resource(condition, 'grounded_correct_percent', cohort)['mean']
        row = self.resource(condition, 'coverage_change_percentage_points', cohort)
        col = '#12548E'
        ax.scatter(tok, row['mean'], marker='D', s=62, color=col, edgecolor='white', lw=.8, zorder=7)
        ax.annotate(f"Main cohort · {int(row['n'])} paired\n({quality:.1f}%, {tok:.2f}×)", (tok, row['mean']),
                    xytext=(-77, -25), textcoords='offset points', color=col, fontsize=7.8, weight='bold',
                    arrowprops={'arrowstyle': '-', 'color': col, 'lw': .6})
        table_rows.append((f"Main cohort ({int(row['n'])} paired)", quality, tok, row['mean'], col, 'D'))
        ax.axhline(0, color='#8295A7', ls='--', lw=.6)
        ax.axvline(1, color='#8295A7', ls='--', lw=.6)
        ax.set(xlim=(.2, 1.23), ylim=(-100, 27), xticks=np.arange(.2, 1.21, .2),
               yticks=[-100, -80, -60, -40, -20, 0, 20], xlabel='Processed input + output token ratio',
               ylabel='Coverage change\n(percentage points)')
        tax = self.ax(225, 184, 184, 27)
        tax.set(xlim=(0, 1), ylim=(0, 6.6)); tax.axis('off')
        tax.add_patch(Rectangle((0, 5), 1, 1.6, facecolor='#EEF3F9', edgecolor='none'))
        positions = [.055, .58, .76, .98]
        headers = ['Condition', 'Grounded correct\n(%)', 'Token ratio', 'Coverage change\n(percentage points)']
        for i, (x, title) in enumerate(zip(positions, headers, strict=True)):
            tax.text(x, 5.8, title, ha='left' if i == 0 else 'right', va='center', fontsize=8, weight='bold')
        for i, (name, quality, tok, delta, col, marker) in enumerate(table_rows):
            y = 4.5 - i
            tax.scatter(.025, y, color=col, marker=marker, s=36)
            for j, text in enumerate([name, f'{quality:.1f}', f'{tok:.2f}×', f'{delta:+.2f}']):
                tax.text(positions[j], y, text, ha='left' if j == 0 else 'right', va='center',
                         fontsize=8.1, weight='bold' if i == 1 and j in (1, 2) else 'normal')
        for y in [0, 1, 2, 3, 4, 5, 6.6]:
            tax.axhline(y, color='#C4CFDC', lw=.5)

    def save(self) -> None:
        self.fig.savefig(OUT / 'Fig5_preview.png', dpi=140)
        self.fig.savefig(OUT / 'Fig5.pdf')
        self.fig.savefig(OUT / 'Fig5.svg')
        self.fig.savefig(OUT / 'Fig5.png', dpi=600)
        for name in self.tables:
            path = OUT / 'data' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(SOURCE) as archive:
                path.write_bytes(archive.read(name))
        manifest = {'source_archive': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    'sources': self.provenance, 'canvas_mm': [W, H], 'png_dpi': 600,
                    'notes': ['No global title, takeaway or gray panel commentary.',
                              'All plotted numbers read from the formal statistics archive.',
                              'All individual points retained: model effects use symmetric-log scale, linear within ±5 with linscale 2, limits -55 to 55; resource change axis -100 to 27.',
                              'Two-row layout: C and D side by side, each with vertically stacked component charts.',
                              'Three deletion schemes compared within the same 20-paper subset; scheme 1 from dose_summary.',
                              'Positive matrix changes retain a plus sign and zero-loss shading.',
                              'State transition counts derived by joining the same complete papers.',
                              'Main-cohort and diagnostic absolute scores use their respective evaluation protocols.']}
        (OUT / 'Fig5_source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        plt.close(self.fig)


def main() -> None:
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    figure = Figure()
    figure.card('a', (4, 4, 212, 87), 'Model-stage sensitivity and transfer robustness', '#A9CDE4', '#EEF6FC', '#174580')
    figure.card('b', (219, 4, 197, 87), 'Evidence-reduction dose response', '#A6D5CB', '#EFF9F5', '#1D6056')
    figure.card('c', (4, 95, 212, 119), 'Critical-evidence deletion and judgment states', '#EDC7A7', '#FFF5ED', '#913D22')
    figure.card('d', (219, 95, 197, 119), 'Quality–resource trade-off', '#CABBE9', '#F3EFFC', '#5834A2')
    figure.model()
    figure.evidence()
    figure.states()
    figure.resources()
    figure.save()
    print(OUT / 'Fig5.png')


if __name__ == '__main__':
    main()

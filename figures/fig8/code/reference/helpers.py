"""Numerical and rendering helpers for the fixed Fig.8."""
from __future__ import annotations
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any
import matplotlib.pyplot as plt
from matplotlib import font_manager
INK, MUTED = '#24282D', '#737B85'

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})

def effective_communities(labels: list[int | None]) -> float | None:
    # Unassigned nodes are missing membership, not one additional community.
    labels = [label for label in labels if label is not None]
    if not labels:
        return None
    counts = Counter(labels)
    return math.exp(-sum((n / len(labels)) * math.log(n / len(labels))
                         for n in counts.values()))

def neighborhood_overlap(neighborhoods: list[set[str]]) -> tuple[float | None, int]:
    pairs = [len(a & b) / len(a | b) for a, b in combinations(neighborhoods, 2) if a and b]
    return (mean(pairs), len(pairs)) if pairs else (None, 0)

def configure() -> None:
    for name in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf']:
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10,
                         'axes.labelsize': 10, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'axes.edgecolor': '#91A0AF',
                         'axes.linewidth': .7, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': INK, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})

def label(fig: plt.Figure, x: float, y: float, text: str,
          size: float = 11, bold: bool = False, **kwargs: Any) -> None:
    fig.text(x, y, text, fontsize=size, weight='bold' if bold else 'normal', **kwargs)

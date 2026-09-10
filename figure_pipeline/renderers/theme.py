"""Shared semantic colors and final-size type system; no misleading color aliases."""
from __future__ import annotations
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ('#430258', '#3f387e', '#365d8d', '#2b7d8f', '#30a083', '#51be64', '#9ed73f', '#f8e520')
TOKENS = {
    'ink': PALETTE[0], 'direct_llm': PALETTE[2], 'graph_joint': PALETTE[4],
    'historical_edge': '#999999', 'unknown': '#dddddd', 'neutral': '#f5f5f5',
    'inserted_target': PALETTE[0], 'positive_difference': PALETTE[4],
    'negative_difference': PALETTE[1],
}
SEQUENTIAL = LinearSegmentedColormap.from_list('aspr_sequential', PALETTE)
DIVERGING = LinearSegmentedColormap.from_list('aspr_signed', [PALETTE[1], '#f5f5f5', PALETTE[4]])
WIDTH_MM = 180
MIN_FONT_PT = 7.5

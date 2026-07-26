from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.env import getenv
from experiments.figure_quality import (
    normalize_reference_closure_report,
    strict_main_figure_failed,
    write_figure_quality_report,
    write_run_manifest,
    write_strict_failure_report,
)
from experiments.fig02.old.build_fig2_reference_closure import parse_reference_list

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspr_matplotlib_cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_FIG1_DATA_ROOT = PROJECT_ROOT / 'outputs' / 'fig01/old'
DEFAULT_STRICT_FIG1_DATA_ROOT = PROJECT_ROOT / 'outputs' / 'fig01/old/work/strict_best4'
DEFAULT_NATURE_FIG1_OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'nature_final' / 'fig01_knowledge_perturbation'
DEFAULT_NATURE_FIG2_OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'nature_final' / 'fig02_empirical_validation'
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / 'data' / 'knowledge_corpus' / 'v1_strict'
DEFAULT_CORPUS_FIG2_DATA_ROOT = DEFAULT_CORPUS_ROOT / 'views' / 'fig2'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'fig02/old'
DEFAULT_STRONG_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'kg_perturbation_fig2_strong'
DEFAULT_DOMAIN = 'crispr'
DEFAULT_STRONG_DOMAINS = [
    'crispr',
    'graphene_2d_materials',
    'ipsc_reprogramming',
    'transformer_foundation_models',
]


def default_fig2_data_root() -> Path:
    """Prefer the unified corpus view, then fall back to legacy Fig. 1 exports."""
    return DEFAULT_CORPUS_FIG2_DATA_ROOT if DEFAULT_CORPUS_FIG2_DATA_ROOT.exists() else DEFAULT_FIG1_DATA_ROOT


def default_strong_domains() -> List[str]:
    """Load strong-mode domains from the unified corpus manifest when available."""
    manifest_path = DEFAULT_CORPUS_ROOT / 'manifest.json'
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            domains = [str(item) for item in payload.get('domains_with_data') or [] if str(item).strip()]
            if domains:
                return domains
        except Exception:
            pass
    return DEFAULT_STRONG_DOMAINS

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# --------------------------
# Style constants
# --------------------------
TEXT_DARK = '#111827'
TEXT_MID = '#374151'
TEXT_LIGHT = '#6B7280'
GRID = '#D1D5DB'
BORDER = '#9CA3AF'
PANEL_BG = '#FFFFFF'
FIG_BG = '#FFFFFF'

METRIC_SPECS = [
    ('B', 'B', '#0B4FA3', 'bridge position'),
    ('RS', 'RS', '#2E7D32', 'distance-weighted diversity'),
    ('DeltaQ0', 'ΔQ0', '#F97316', 'boundary perturbation'),
    ('Uzzi', 'Uzzi-style', '#7C3AED', 'field-pair atypical recombination'),
    ('RTD', 'RTD', '#0891B2', 'reference target diversity'),
    ('BurtIP', 'Burt IP', '#1E3A8A', 'structural holes'),
    ('PDE', 'PDE', '#EF4444', 'prospective diffusion breadth'),
]

CANDIDATE_GROUPS = {
    'bridge position': ['B', 'degree', 'pagerank', 'closeness'],
    'diversity / breadth': ['RS', 'field_shannon', 'field_simpson', 'field_variety'],
    'boundary perturbation': ['DeltaQ0', 'conductance_delta'],
    'atypical recombination': ['Uzzi', 'pair_surprisal'],
    'reference target diversity': ['RTD', 'community_simpson'],
    'structural holes': ['BurtIP', 'effective_size', 'constraint_inv'],
    'diffusion breadth': ['PDE', 'field_entropy_norm'],
}

EVIDENCE_CHANNELS = [
    ('Breadth', '#2E7D32', 'Breadth of disciplinary / topical coverage'),
    ('Brokerage', '#0B4FA3', 'Potential shortcut creation across communities'),
    ('Boundary', '#F97316', 'Potential to disturb module boundaries'),
    ('Atypicality', '#7C3AED', 'Unusual but interpretable reference combinations'),
]

EVIDENCE_MAP = {
    'B': {'Brokerage': 'primary', 'Boundary': 'secondary'},
    'RS': {'Breadth': 'primary'},
    'DeltaQ0': {'Boundary': 'primary'},
    'Uzzi': {'Atypicality': 'primary', 'Boundary': 'secondary'},
    'RTD': {'Brokerage': 'primary', 'Breadth': 'secondary'},
    'BurtIP': {'Brokerage': 'primary'},
    'PDE': {'Breadth': 'primary'},
}

PANEL_A_MAX_TOPICS = 9
PANEL_A_MAX_REFERENCE_TOPICS = 4
PANEL_A_MAX_BACKBONE_EDGES = 10
PANEL_A_MAX_BEADS_PER_TOPIC = 6
FIG1_MULTI_DOMAIN_IMAGE = 'fig1_multi_domain_real.png'
FIG1_MULTI_DOMAIN_COLUMN_BOUNDS = (
    (142 / 6126, 910 / 6126),
    (1262 / 6126, 2030 / 6126),
    (2381 / 6126, 3150 / 6126),
    (3497 / 6126, 4275 / 6126),
)
FIG1_MULTI_DOMAIN_ROW_BOUNDS = (
    (465 / 4271, 1305 / 4271),
    (1395 / 4271, 2234 / 4271),
    (2324 / 4271, 3164 / 4271),
    (3252 / 4271, 4094 / 4271),
)
NORMAL_DIST = NormalDist()


@dataclass
class RawData:
    works: pd.DataFrame
    citations: pd.DataFrame
    topics: pd.DataFrame
    topic_edges: pd.DataFrame


@dataclass
class ComputedData:
    paper_metrics: pd.DataFrame
    candidate_metrics: pd.DataFrame
    graph_deltas: pd.DataFrame
    redundancy_corr: pd.DataFrame
    indicator_delta_corr: pd.DataFrame
    percentile_long: pd.DataFrame
    landmark_summary: pd.DataFrame
    metric_standardization_diagnostics: pd.DataFrame
    graph_delta_diagnostics: pd.DataFrame
    input_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    domain_adequacy: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_closure_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    matched_controls: pd.DataFrame = field(default_factory=pd.DataFrame)
    indicator_future_corr_bootstrap: pd.DataFrame = field(default_factory=pd.DataFrame)
    evidence_support: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    evidence_mode: str = 'legacy'
    future_tau: Optional[int] = None


# --------------------------
# Utilities
# --------------------------

def setup_style() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'figure.facecolor': FIG_BG,
        'axes.facecolor': FIG_BG,
        'savefig.facecolor': FIG_BG,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.edgecolor': BORDER,
        'text.color': TEXT_DARK,
    })


def blend_with_white(color: str, amount: float = 0.85) -> str:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    out = rgb * (1.0 - amount) + np.ones(3) * amount
    return mcolors.to_hex(out)


def wrap(text: str, width: int) -> str:
    return '\n'.join(textwrap.wrap(str(text), width=width, break_long_words=False))


def rounded_box(ax, x, y, w, h, facecolor='#FFFFFF', edgecolor=BORDER, lw=0.8, radius=0.02, zorder=1):
    patch = mpatches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.004,rounding_size={radius}',
                                    transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor,
                                    linewidth=lw, zorder=zorder, clip_on=False)
    ax.add_patch(patch)
    return patch


def panel_frame(ax, letter: str, title: str):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    rounded_box(ax, 0.0, 0.0, 1.0, 1.0, PANEL_BG, '#8A8A8A', 0.8, 0.025, zorder=0)
    ax.text(0.02, 0.965, letter, ha='left', va='top', fontsize=16, fontweight='bold')
    ax.text(0.10, 0.955, title, ha='left', va='top', fontsize=9.2, fontweight='bold')


def draw_arrow(ax, start, end, color='#4B5563', lw=1.0, ms=12, style='-|>', connectionstyle='arc3,rad=0.0', zorder=4):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style,
                                 mutation_scale=ms, linewidth=lw, color=color,
                                 connectionstyle=connectionstyle, shrinkA=1, shrinkB=1, zorder=zorder))


def draw_pill(ax, x, y, text, color, width, height=0.05, fontsize=6, zorder=4):
    rounded_box(ax, x, y, width, height, blend_with_white(color, 0.90), color, 0.8, height * 0.45, zorder=zorder)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, color=color,
            fontweight='bold', transform=ax.transAxes, zorder=zorder+1)


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')


def progress_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f'[fig2] {message}', flush=True)


LOCAL_FILE_ALIASES = {
    'works': ('works.csv', 'works_selected.csv'),
    'citations': ('citations.csv', 'paper_edges.csv'),
    'topics': ('topics.csv', 'topic_nodes.csv'),
    'topic_edges': ('topic_edges.csv',),
}

STANDARD_INPUT_FILES = ('works.csv', 'citations.csv', 'topics.csv', 'topic_edges.csv')
FIG1_EXPORT_FILES = ('works_selected.csv', 'paper_edges.csv', 'topic_nodes.csv', 'topic_edges.csv')
FIG1_CONFIG_ALIASES = {
    'crispr': 'crispr.yaml',
    'graphene_2d_materials': 'graphene.yaml',
    'ipsc_reprogramming': 'ipsc.yaml',
    'transformer_foundation_models': 'transformer.yaml',
}


def first_existing_file(data_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    return None


def has_local_data_files(data_dir: Path) -> bool:
    return all(first_existing_file(data_dir, candidates) is not None for candidates in LOCAL_FILE_ALIASES.values())


def has_standard_input_files(data_dir: Path) -> bool:
    return all((data_dir / name).exists() for name in STANDARD_INPUT_FILES)


def has_fig1_export_files(data_dir: Path) -> bool:
    return all((data_dir / name).exists() for name in FIG1_EXPORT_FILES)


def find_fig1_multi_domain_image(path: Path) -> Optional[Path]:
    """Locate a multi-domain Fig. 1 PNG for direct Panel a cropping."""
    candidates: List[Path] = []
    if path.suffix.lower() == '.png':
        candidates.append(path)
    candidates.extend([
        path / FIG1_MULTI_DOMAIN_IMAGE,
        path.parent / FIG1_MULTI_DOMAIN_IMAGE,
        DEFAULT_NATURE_FIG1_OUTPUT_ROOT / FIG1_MULTI_DOMAIN_IMAGE,
    ])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def has_fig1_panel_a_source(data_dir: Path) -> bool:
    return has_fig1_export_files(data_dir) or find_fig1_multi_domain_image(data_dir) is not None


def resolve_panel_a_fig1_snapshot_dir(args: argparse.Namespace) -> Optional[Path]:
    """Find a Fig. 1 domain output directory whose snapshots can be reused."""
    domain = args.example_domain if args.evidence_mode == 'strong' else args.domain
    candidates: List[Path] = []
    if args.fig1_snapshot_dir is not None:
        candidates.append(args.fig1_snapshot_dir)
    if args.data_dir is not None:
        candidates.append(args.data_dir)
        if domain:
            candidates.append(args.data_dir / domain)
    if domain:
        candidates.extend([
            DEFAULT_NATURE_FIG1_OUTPUT_ROOT / domain,
            DEFAULT_NATURE_FIG1_OUTPUT_ROOT,
            DEFAULT_STRICT_FIG1_DATA_ROOT / domain,
            DEFAULT_FIG1_DATA_ROOT / domain,
        ])

    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if has_fig1_panel_a_source(path):
            return path
    return None


def resolve_standard_data_dir(data_dir: Path, domain: Optional[str]) -> Optional[Path]:
    if has_standard_input_files(data_dir):
        return data_dir
    if domain and has_standard_input_files(data_dir / domain):
        return data_dir / domain
    return None


def resolve_fig1_domain_dir(data_dir: Path, domain: Optional[str]) -> Optional[Path]:
    if has_fig1_export_files(data_dir):
        return data_dir
    if domain and has_fig1_export_files(data_dir / domain):
        return data_dir / domain
    return None


def resolve_data_dir(data_dir: Path, domain: Optional[str]) -> Path:
    """Resolve either an exact data directory or a Fig. 1 local-output root."""
    if has_local_data_files(data_dir):
        return data_dir
    if domain:
        domain_dir = data_dir / domain
        if has_local_data_files(domain_dir):
            return domain_dir

    available = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and has_local_data_files(p)) if data_dir.exists() else []
    if available:
        raise FileNotFoundError(
            f'No local data files found directly in {data_dir}. '
            f'Available domains: {available}. Re-run with --domain <name>.'
        )
    raise FileNotFoundError(
        f'No local data files found in {data_dir}. Expected either '
        'works.csv/citations.csv/topics.csv/topic_edges.csv or '
        'works_selected.csv/paper_edges.csv/topic_nodes.csv/topic_edges.csv.'
    )


def read_local_csv(data_dir: Path, key: str) -> Tuple[pd.DataFrame, Path]:
    path = first_existing_file(data_dir, LOCAL_FILE_ALIASES[key])
    if path is None:
        raise FileNotFoundError(f'{data_dir} is missing one of: {LOCAL_FILE_ALIASES[key]}')
    return pd.read_csv(path), path


def read_works_raw_refs(path: Path) -> Tuple[Dict[str, List[str]], int]:
    refs_by_id: Dict[str, List[str]] = {}
    n_records = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            paper_id = rec.get('id')
            if not paper_id:
                continue
            refs_by_id[str(paper_id)] = [str(ref) for ref in rec.get('refs') or [] if ref]
            n_records += 1
    return refs_by_id, n_records


def write_raw_data(raw: RawData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.works.to_csv(out_dir / 'works.csv', index=False)
    raw.citations.to_csv(out_dir / 'citations.csv', index=False)
    raw.topics.to_csv(out_dir / 'topics.csv', index=False)
    raw.topic_edges.to_csv(out_dir / 'topic_edges.csv', index=False)


def save_prepare_report(report: Mapping[str, Any], out_dir: Path) -> None:
    (out_dir / 'fig2_input_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def infer_primary_field(row: pd.Series) -> str:
    for col in ['display_label', 'community_label', 'primary_topic', 'domain']:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return 'unknown_field'


def standardize_fig1_works(works_selected: pd.DataFrame, domain_name: str) -> pd.DataFrame:
    works = works_selected.copy()
    works['domain'] = domain_name
    works['primary_field'] = works.apply(infer_primary_field, axis=1)

    if 'analysis_community' not in works.columns:
        if 'community' in works.columns:
            works['analysis_community'] = pd.to_numeric(works['community'], errors='coerce')
        elif 'display_community' in works.columns:
            works['analysis_community'] = pd.to_numeric(works['display_community'], errors='coerce')
        else:
            works['analysis_community'] = -1
    works['analysis_community'] = pd.to_numeric(works['analysis_community'], errors='coerce').fillna(-1).astype(int)

    if 'display_community' not in works.columns:
        works['display_community'] = pd.NA
    works['display_community'] = pd.to_numeric(works['display_community'], errors='coerce')
    if 'anchor_label' not in works.columns:
        works['anchor_label'] = pd.NA
    anchor_label = works['anchor_label'].replace('', pd.NA)
    works['anchor_label'] = anchor_label
    works['is_landmark'] = (anchor_label.notna() & (anchor_label.astype(str).str.strip() != '')).astype(int)
    if 'title' not in works.columns:
        works['title'] = works['id']
    base_cols = [
        'id', 'year', 'title', 'domain', 'primary_field', 'analysis_community', 'display_community',
        'is_landmark', 'anchor_label',
    ]
    extra_cols = [
        col for col in [
            'short_id', 'doi', 'cited_by_count', 'primary_topic', 'community',
            'community_label', 'display_label',
        ]
        if col in works.columns
    ]
    return works[base_cols + extra_cols].copy()


def citations_from_fig1_raw_refs(fig1_dir: Path, selected_ids: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    refs_path = fig1_dir / 'works_raw.jsonl'
    if not refs_path.exists():
        return pd.DataFrame(columns=['source', 'target']), {'raw_refs_available': False}

    refs_by_id, n_raw_records = read_works_raw_refs(refs_path)
    selected = set(str(item) for item in selected_ids)
    rows = []
    dropped_unselected_targets = 0
    missing_source_records = 0
    for source in [str(item) for item in selected_ids]:
        refs = refs_by_id.get(source)
        if refs is None:
            missing_source_records += 1
            continue
        for target in refs:
            if target in selected:
                rows.append({'source': source, 'target': target})
            else:
                dropped_unselected_targets += 1

    citations = pd.DataFrame(rows, columns=['source', 'target']).drop_duplicates()
    report = {
        'raw_refs_available': True,
        'works_raw_jsonl': str(refs_path),
        'works_raw_records': n_raw_records,
        'selected_sources_missing_in_raw': missing_source_records,
        'dropped_refs_to_unselected_targets': dropped_unselected_targets,
        'citation_rows_from_raw_refs': len(citations),
    }
    return citations, report


def citations_from_fig1_paper_edges(fig1_dir: Path, direct_only: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    edge_path = fig1_dir / 'paper_edges.csv'
    edges = pd.read_csv(edge_path)
    source_rows = len(edges)
    if direct_only and 'direct' in edges.columns:
        edges = edges[pd.to_numeric(edges['direct'], errors='coerce').fillna(0) > 0].copy()
    citations = edges[['source', 'target']].drop_duplicates().copy()
    report = {
        'paper_edges_csv': str(edge_path),
        'paper_edge_rows': source_rows,
        'direct_only': direct_only,
        'citation_rows_from_paper_edges': len(citations),
    }
    return citations, report


def prepare_fig2_input_from_fig1(
    fig1_dir: Path,
    prepared_dir: Path,
    direct_only: bool,
    progress: bool,
) -> Path:
    domain_name = fig1_dir.name
    progress_log(f'Preparing Fig. 2 input from Fig. 1 exports: {fig1_dir}', progress)
    works_selected = pd.read_csv(fig1_dir / 'works_selected.csv')
    works = standardize_fig1_works(works_selected, domain_name)
    selected_ids = works['id'].astype(str).tolist()

    citations, citation_report = citations_from_fig1_raw_refs(fig1_dir, selected_ids)
    citation_source = 'works_raw.jsonl'
    if citations.empty:
        progress_log('No usable raw reference citations found; falling back to paper_edges.csv.', progress)
        citations, citation_report = citations_from_fig1_paper_edges(fig1_dir, direct_only=direct_only)
        citation_source = 'paper_edges.csv'

    topics = pd.read_csv(fig1_dir / 'topic_nodes.csv')
    topic_edges = pd.read_csv(fig1_dir / 'topic_edges.csv')
    raw = RawData(works=works, citations=citations, topics=topics, topic_edges=topic_edges)
    write_raw_data(raw, prepared_dir)

    report: Dict[str, Any] = {
        'source_kind': 'fig1_exports',
        'source_dir': str(fig1_dir),
        'prepared_dir': str(prepared_dir),
        'citation_source': citation_source,
        'works_rows': len(works),
        'citation_rows': len(citations),
        'topic_rows': len(topics),
        'topic_edge_rows': len(topic_edges),
        'landmark_rows': int(works['is_landmark'].sum()),
        'display_community_missing_rows': int(works['display_community'].isna().sum()),
        'analysis_community_fallback_rows': int((works['analysis_community'] < 0).sum()),
        **citation_report,
    }
    save_prepare_report(report, prepared_dir)
    progress_log(
        f'Prepared Fig. 2 input in {prepared_dir}: {len(works):,} works, '
        f'{len(citations):,} citations from {citation_source}, {int(works["is_landmark"].sum()):,} landmarks',
        progress,
    )
    return prepared_dir


def prepare_fig2_input_from_standard(
    source_dir: Path,
    prepared_dir: Path,
    direct_only: bool,
    progress: bool,
) -> Path:
    progress_log(f'Normalizing standard Fig. 2 input from {source_dir}', progress)
    raw = load_raw_data(source_dir, domain=None, direct_only=direct_only, progress=False)
    write_raw_data(raw, prepared_dir)
    report = {
        'source_kind': 'standard_fig2_input',
        'source_dir': str(source_dir),
        'prepared_dir': str(prepared_dir),
        'works_rows': len(raw.works),
        'citation_rows': len(raw.citations),
        'topic_rows': len(raw.topics),
        'topic_edge_rows': len(raw.topic_edges),
        'landmark_rows': int(raw.works['is_landmark'].sum()),
        'display_community_missing_rows': int(raw.works['display_community'].isna().sum()),
        'analysis_community_fallback_rows': int((raw.works['analysis_community'] < 0).sum()),
    }
    save_prepare_report(report, prepared_dir)
    progress_log(f'Prepared normalized Fig. 2 input in {prepared_dir}', progress)
    return prepared_dir


def default_fig1_config_for_domain(domain: Optional[str]) -> Optional[Path]:
    if not domain:
        return None
    path = PROJECT_ROOT / 'experiments' / 'fig01/old' / 'configs' / f'{domain}.yaml'
    return path if path.exists() else None


def run_fig1_pipeline_for_input(
    fig1_config: Path,
    fig1_out_dir: Path,
    use_cache: bool,
    openalex_api_key: Optional[str],
    openalex_api_keys: Optional[str],
    email: Optional[str],
    progress: bool,
) -> Path:
    progress_log(f'Running Fig. 1 pipeline to materialize source data: {fig1_config}', progress)
    from experiments.fig01.old.fig1_knowledge_perturbation import (  # pylint: disable=import-outside-toplevel
        OpenAlexClient,
        load_config,
        run_domain,
        split_api_keys,
    )

    cfg = load_config(fig1_config)
    api_cfg = cfg.get('api', {})
    client = OpenAlexClient(
        api_key=openalex_api_key,
        api_keys=split_api_keys(openalex_api_keys),
        email=email,
        sleep_seconds=float(api_cfg.get('sleep_seconds', 0.1)),
        max_retries=int(api_cfg.get('max_retries', 6)),
        timeout_seconds=int(api_cfg.get('timeout_seconds', 60)),
    )
    run_domain(cfg, client, fig1_out_dir, use_cache=use_cache)
    return fig1_out_dir / cfg['slug']


def prepare_fig2_input_data(
    data_dir: Path,
    out_dir: Path,
    domain: Optional[str],
    direct_only: bool,
    fig1_config: Optional[Path],
    run_fig1_if_missing: bool,
    use_fig1_cache: bool,
    openalex_api_key: Optional[str],
    openalex_api_keys: Optional[str],
    email: Optional[str],
    progress: bool,
) -> Path:
    prepared_domain = domain or data_dir.name
    prepared_dir = out_dir / 'fig2_input' / prepared_domain

    standard_dir = resolve_standard_data_dir(data_dir, domain)
    if standard_dir is not None:
        return prepare_fig2_input_from_standard(standard_dir, prepared_dir, direct_only=direct_only, progress=progress)

    fig1_dir = resolve_fig1_domain_dir(data_dir, domain)
    if fig1_dir is None and run_fig1_if_missing:
        config_path = fig1_config or default_fig1_config_for_domain(domain)
        if config_path is None:
            raise FileNotFoundError('No Fig. 1 config provided and no default config exists for this domain.')
        fig1_dir = run_fig1_pipeline_for_input(
            config_path,
            out_dir / 'fig1_source',
            use_cache=use_fig1_cache,
            openalex_api_key=openalex_api_key,
            openalex_api_keys=openalex_api_keys,
            email=email,
            progress=progress,
        )

    if fig1_dir is not None:
        return prepare_fig2_input_from_fig1(fig1_dir, prepared_dir, direct_only=direct_only, progress=progress)

    raise FileNotFoundError(
        f'Cannot prepare Fig. 2 input from {data_dir}. Expected standard Fig. 2 files '
        f'{STANDARD_INPUT_FILES}, Fig. 1 exports {FIG1_EXPORT_FILES}, or use --run-fig1-if-missing.'
    )


def normalize_works(works: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = works.copy()
    if 'id' not in out.columns:
        raise ValueError('works table is missing required column: id')

    for col in ['title', 'domain', 'primary_field', 'anchor_label']:
        if col in out.columns:
            out[col] = out[col].replace('', pd.NA)

    if 'domain' not in out.columns:
        out['domain'] = data_dir.name
    else:
        out['domain'] = out['domain'].fillna(data_dir.name)

    if 'title' not in out.columns:
        out['title'] = out['id']
    else:
        out['title'] = out['title'].fillna(out['id'])

    if 'primary_field' not in out.columns:
        out['primary_field'] = pd.NA
    for candidate in ['display_label', 'community_label', 'primary_topic', 'domain']:
        if candidate in out.columns:
            out['primary_field'] = out['primary_field'].fillna(out[candidate].replace('', pd.NA))
    out['primary_field'] = out['primary_field'].fillna('unknown_field')

    if 'analysis_community' not in out.columns:
        if 'community' in out.columns:
            out['analysis_community'] = pd.to_numeric(out['community'], errors='coerce')
        elif 'display_community' in out.columns:
            out['analysis_community'] = pd.to_numeric(out['display_community'], errors='coerce')
        else:
            out['analysis_community'] = -1
    out['analysis_community'] = pd.to_numeric(out['analysis_community'], errors='coerce').fillna(-1).astype(int)

    if 'display_community' not in out.columns:
        out['display_community'] = pd.NA
    out['display_community'] = pd.to_numeric(out['display_community'], errors='coerce')

    if 'is_landmark' not in out.columns:
        if 'anchor_label' in out.columns:
            anchors = out['anchor_label'].notna() & (out['anchor_label'].astype(str).str.strip() != '')
            out['is_landmark'] = anchors.astype(int)
        else:
            out['is_landmark'] = 0
    else:
        out['is_landmark'] = pd.to_numeric(out['is_landmark'], errors='coerce').fillna(0).astype(int)

    if 'anchor_label' not in out.columns:
        out['anchor_label'] = np.where(out['is_landmark'] == 1, out['title'], pd.NA)
    else:
        out['anchor_label'] = out['anchor_label'].where(out['anchor_label'].notna(), pd.NA)
        out.loc[(out['is_landmark'] == 1) & out['anchor_label'].isna(), 'anchor_label'] = out.loc[
            (out['is_landmark'] == 1) & out['anchor_label'].isna(), 'title'
        ]
    return out


def normalize_citations(citations: pd.DataFrame, source_path: Path, direct_only: bool) -> pd.DataFrame:
    out = citations.copy()
    if source_path.name == 'paper_edges.csv' and direct_only and 'direct' in out.columns:
        out = out[pd.to_numeric(out['direct'], errors='coerce').fillna(0) > 0].copy()
    return out


def load_raw_data(
    data_dir: Path,
    domain: Optional[str] = DEFAULT_DOMAIN,
    direct_only: bool = True,
    progress: bool = False,
) -> RawData:
    progress_log(f'Resolving local data directory: {data_dir}', progress)
    data_dir = resolve_data_dir(data_dir, domain)
    progress_log(f'Using local data directory: {data_dir}', progress)
    works, works_path = read_local_csv(data_dir, 'works')
    citations, citations_path = read_local_csv(data_dir, 'citations')
    topics, topics_path = read_local_csv(data_dir, 'topics')
    topic_edges, topic_edges_path = read_local_csv(data_dir, 'topic_edges')
    progress_log(
        'Loaded CSV files: '
        f'{works_path.name} ({len(works):,} rows), '
        f'{citations_path.name} ({len(citations):,} rows), '
        f'{topics_path.name} ({len(topics):,} rows), '
        f'{topic_edges_path.name} ({len(topic_edges):,} rows)',
        progress,
    )

    works = normalize_works(works, data_dir)
    citations = normalize_citations(citations, citations_path, direct_only=direct_only)
    if citations_path.name == 'paper_edges.csv' and direct_only:
        progress_log(f'Kept {len(citations):,} direct citation edges from paper_edges.csv', progress)

    require_columns(works, ['id', 'year', 'analysis_community', 'display_community', 'primary_field', 'is_landmark'], works_path.name)
    require_columns(citations, ['source', 'target'], citations_path.name)
    require_columns(topics, ['community', 'label', 'x', 'y'], topics_path.name)
    require_columns(topic_edges, ['source_community', 'target_community', 'weight'], topic_edges_path.name)

    works['id'] = works['id'].astype(str)
    citations['source'] = citations['source'].astype(str)
    citations['target'] = citations['target'].astype(str)
    works['year'] = works['year'].astype(int)
    works['analysis_community'] = works['analysis_community'].astype(int)
    works['display_community'] = pd.to_numeric(works['display_community'], errors='coerce')
    works['is_landmark'] = works['is_landmark'].astype(int)

    if 'domain' not in works.columns:
        works['domain'] = 'default_domain'
    if 'title' not in works.columns:
        works['title'] = works['id']
    if 'anchor_label' not in works.columns:
        works['anchor_label'] = np.where(works['is_landmark'] == 1, works['title'], pd.NA)
    progress_log(
        f'Prepared raw data: {len(works):,} works, {len(citations):,} citations, '
        f'{works["year"].min()}-{works["year"].max()}, {int(works["is_landmark"].sum()):,} landmarks',
        progress,
    )
    return RawData(works=works, citations=citations, topics=topics, topic_edges=topic_edges)


# --------------------------
# Core computations
# --------------------------

def shannon_entropy(counts: Sequence[int]) -> float:
    arr = np.asarray(counts, dtype=float)
    if arr.sum() <= 0:
        return 0.0
    p = arr / arr.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def simpson_diversity(counts: Sequence[int]) -> float:
    arr = np.asarray(counts, dtype=float)
    if arr.sum() <= 0:
        return 0.0
    p = arr / arr.sum()
    return float(1.0 - np.square(p).sum())


def field_distance_matrix(prior_refs: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Build a simple disparity matrix from field co-occurrence similarity.
    similarity(i,j) = cooccur / sqrt(freq_i * freq_j); distance = 1 - similarity.
    """
    freq: Dict[str, int] = {}
    pair_count: Dict[Tuple[str, str], int] = {}
    by_source = prior_refs.groupby('source')['target_field'].apply(list)
    for fields in by_source:
        fields = [f for f in fields if pd.notna(f)]
        uniq = sorted(set(fields))
        for f in uniq:
            freq[f] = freq.get(f, 0) + 1
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                k = (uniq[i], uniq[j])
                pair_count[k] = pair_count.get(k, 0) + 1
    fields = list(freq.keys())
    dist: Dict[Tuple[str, str], float] = {}
    for i, fi in enumerate(fields):
        dist[(fi, fi)] = 0.0
        for fj in fields[i+1:]:
            obs = pair_count.get((fi, fj), 0)
            sim = obs / max(1e-9, math.sqrt(freq[fi] * freq[fj]))
            sim = max(0.0, min(1.0, sim))
            d = 1.0 - sim
            dist[(fi, fj)] = d
            dist[(fj, fi)] = d
    return dist


def rao_stirling(fields: Sequence[str], dist: Mapping[Tuple[str, str], float]) -> float:
    fields = [f for f in fields if pd.notna(f)]
    if len(fields) <= 1:
        return 0.0
    vc = pd.Series(fields).value_counts()
    p = vc / vc.sum()
    vals = list(p.index)
    rs = 0.0
    for i, fi in enumerate(vals):
        for fj in vals[i+1:]:
            rs += 2 * p[fi] * p[fj] * dist.get((fi, fj), 1.0 if fi != fj else 0.0)
    return float(rs)


def build_prior_graph(works: pd.DataFrame, citations: pd.DataFrame, year: int) -> nx.Graph:
    prior_ids = set(works.loc[works['year'] < year, 'id'].astype(str))
    g = nx.Graph()
    g.add_nodes_from(prior_ids)
    sub = citations[citations['source'].isin(prior_ids) & citations['target'].isin(prior_ids)]
    g.add_edges_from(zip(sub['source'].astype(str), sub['target'].astype(str)))
    return g


def get_reference_rows(raw: RawData, paper_id: str, year: int) -> pd.DataFrame:
    refs = raw.citations[raw.citations['source'] == str(paper_id)].copy()
    meta = raw.works[['id', 'year', 'analysis_community', 'display_community', 'primary_field']].rename(columns={
        'id': 'target',
        'analysis_community': 'target_analysis_comm',
        'display_community': 'target_display_comm',
        'primary_field': 'target_field',
        'year': 'target_year',
    })
    refs = refs.merge(meta, on='target', how='left')
    refs = refs[refs['target_year'] < year].copy()
    return refs


def modularity_from_partition(g: nx.Graph, community_map: Mapping[str, int]) -> float:
    nodes = [n for n in g.nodes if n in community_map]
    if len(nodes) < 2 or g.number_of_edges() == 0:
        return 0.0
    groups: Dict[int, set] = {}
    for n in nodes:
        groups.setdefault(int(community_map[n]), set()).add(n)
    parts = [s for s in groups.values() if len(s) > 0]
    try:
        return float(nx.algorithms.community.quality.modularity(g.subgraph(nodes), parts))
    except Exception:
        return 0.0


def assortativity_by_comm(g: nx.Graph, community_map: Mapping[str, int]) -> float:
    sub_nodes = [n for n in g.nodes if n in community_map]
    if len(sub_nodes) < 2 or g.number_of_edges() == 0:
        return 0.0
    g2 = g.subgraph(sub_nodes).copy()
    for n in g2.nodes:
        g2.nodes[n]['community'] = community_map[n]
    try:
        val = nx.attribute_assortativity_coefficient(g2, 'community')
        return float(0.0 if np.isnan(val) else val)
    except Exception:
        return 0.0


def build_local_reference_graph(
    refs: pd.DataFrame,
    citation_neighbors: Mapping[str, set],
    paper_id: Optional[str] = None,
    paper_comm: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[str, int]]:
    ref_ids = set(refs['target'].astype(str))
    g = nx.Graph()
    g.add_nodes_from(ref_ids)
    for u in ref_ids:
        for v in citation_neighbors.get(u, set()):
            if v in ref_ids and u < v:
                g.add_edge(u, v)
    comm_meta = refs[['target', 'target_analysis_comm']].dropna(subset=['target_analysis_comm']).copy()
    comm_meta['target_analysis_comm'] = comm_meta['target_analysis_comm'].astype(int)
    comm_map = dict(zip(comm_meta['target'].astype(str), comm_meta['target_analysis_comm']))
    if paper_id is not None:
        pid = str(paper_id)
        g.add_node(pid)
        for rid in ref_ids:
            g.add_edge(pid, rid)
        if paper_comm is None:
            paper_comm = choose_publication_day_community(refs, -1)[0]
        comm_map[pid] = int(paper_comm)
    return g, comm_map


def build_citation_neighbors(citations: pd.DataFrame) -> Dict[str, set]:
    neighbors: Dict[str, set] = {}
    for row in citations.itertuples(index=False):
        u = str(row.source)
        v = str(row.target)
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)
    return neighbors


def boundary_mixing_share(g: nx.Graph, community_map: Mapping[str, int]) -> float:
    if g.number_of_edges() == 0:
        return 0.0
    total = 0
    cross = 0
    for u, v in g.edges():
        if u in community_map and v in community_map:
            total += 1
            if community_map[u] != community_map[v]:
                cross += 1
    return float(cross / total) if total else 0.0


def focal_bridge_betweenness_from_reference_components(g_ref: nx.Graph, ref_count: int) -> float:
    if ref_count <= 1:
        return 0.0
    comp_sizes = [len(c) for c in nx.connected_components(g_ref)] if g_ref.number_of_nodes() else []
    if len(comp_sizes) <= 1:
        return 0.0
    bridge_pairs = 0.0
    seen = 0
    for size in comp_sizes:
        bridge_pairs += seen * size
        seen += size
    denom = ref_count * (ref_count - 1) / 2.0
    return float(bridge_pairs / max(denom, 1.0))


def burt_effective_size_proxy(g_ref: nx.Graph, ref_count: int) -> Tuple[float, float, float]:
    if ref_count <= 0:
        return 0.0, 0.0, 1.0
    internal_edges = float(g_ref.number_of_edges())
    effective_size = max(0.0, float(ref_count) - (2.0 * internal_edges / max(1.0, float(ref_count))))
    burt_ip = effective_size / max(1.0, float(ref_count))
    constraint_proxy = max(1e-9, 1.0 - burt_ip)
    return float(effective_size), float(burt_ip), float(1.0 / constraint_proxy)


def pair_zscore_lookup(prior_refs: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Compute simple pair z-scores for field combinations from prior literature."""
    # observed pair counts across papers
    paper_fields = prior_refs.groupby('source')['target_field'].apply(lambda s: sorted(set([x for x in s if pd.notna(x)])))
    pair_obs: Dict[Tuple[str, str], int] = {}
    field_occ: Dict[str, int] = {}
    n_papers = len(paper_fields)
    for fields in paper_fields:
        for f in fields:
            field_occ[f] = field_occ.get(f, 0) + 1
        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                k = (fields[i], fields[j])
                pair_obs[k] = pair_obs.get(k, 0) + 1
    out = {}
    keys = sorted(set(list(field_occ.keys())))
    for i, fi in enumerate(keys):
        for fj in keys[i+1:]:
            obs = pair_obs.get((fi, fj), 0)
            pi = field_occ[fi] / max(1, n_papers)
            pj = field_occ[fj] / max(1, n_papers)
            exp = n_papers * pi * pj
            var = max(exp * (1 - pi * pj), 1e-9)
            z = (obs - exp) / math.sqrt(var)
            out[(fi, fj)] = z
            out[(fj, fi)] = z
    return out


def community_graph_from_topic_edges(raw: RawData) -> nx.Graph:
    g = nx.Graph()
    for row in raw.topic_edges.itertuples(index=False):
        u = int(row.source_community); v = int(row.target_community); w = float(row.weight)
        g.add_edge(u, v, weight=w)
    return g


def path_shortening(ref_communities: Sequence[int], topic_graph: nx.Graph) -> float:
    comms = sorted(set([int(c) for c in ref_communities if pd.notna(c)]))
    if len(comms) <= 1:
        return 0.0
    vals = []
    for i, ci in enumerate(comms):
        for cj in comms[i+1:]:
            try:
                d = nx.shortest_path_length(topic_graph, ci, cj)
                vals.append(max(0.0, d - 2.0))  # publication-day addition can create a 2-hop shortcut via p
            except Exception:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def percentile_vs_controls(value: float, controls: Sequence[float]) -> float:
    arr = np.asarray(list(controls), dtype=float)
    if len(arr) == 0:
        return np.nan
    return float((arr <= value).mean() * 100.0)


def rank_normal_scores(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    mask = np.isfinite(arr)
    n = int(mask.sum())
    if n == 0:
        return out
    if n == 1 or np.nanstd(arr[mask]) < 1e-12:
        out[mask] = 0.0
        return out
    ranks = pd.Series(arr[mask]).rank(method='average').to_numpy(dtype=float)
    p = np.clip((ranks - 0.5) / n, 1e-6, 1.0 - 1e-6)
    out[mask] = np.clip([NORMAL_DIST.inv_cdf(float(v)) for v in p], -3.0, 3.0)
    return out


def winsorized_series(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    vals = pd.to_numeric(values, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
    finite = vals[np.isfinite(vals)]
    if len(finite) < 4:
        return vals
    lo = float(finite.quantile(lower_q))
    hi = float(finite.quantile(upper_q))
    return vals.clip(lower=lo, upper=hi)


def transformed_metric_values(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
    if col == 'B':
        vals = pd.Series(np.log1p(np.clip(vals.to_numpy(dtype=float), 0.0, None)), index=df.index)
    elif col == 'DeltaQ0':
        vals = winsorized_series(vals)
    elif col == 'Uzzi' and 'field_variety' in df.columns:
        vals = vals.copy()
        invalid = pd.to_numeric(df['field_variety'], errors='coerce').fillna(0) < 2
        vals.loc[invalid] = np.nan
    return vals.replace([np.inf, -np.inf], np.nan)


def field_year_rank_normalize(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    min_field_year: int = 20,
    min_field: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    diagnostics: List[Dict[str, Any]] = []
    for col in metric_cols:
        transformed = transformed_metric_values(out, col)
        out[col + '_transformed'] = transformed
        z = np.full(len(out), np.nan, dtype=float)
        scope = pd.Series('missing', index=out.index, dtype=object)
        valid = transformed.notna() & np.isfinite(transformed.to_numpy(dtype=float))

        for _, idx in out.loc[valid].groupby(['primary_field', 'year']).groups.items():
            idx = pd.Index(idx)
            vals = transformed.loc[idx].to_numpy(dtype=float)
            if len(idx) >= min_field_year and np.nanstd(vals) > 1e-12:
                z[out.index.get_indexer(idx)] = rank_normal_scores(vals)
                scope.loc[idx] = 'field_year'

        remaining = valid & ~np.isfinite(z)
        for _, idx in out.loc[remaining].groupby('primary_field').groups.items():
            idx = pd.Index(idx)
            field_name = out.loc[idx[0], 'primary_field']
            field_idx = out.index[(out['primary_field'] == field_name) & valid]
            vals = transformed.loc[field_idx].to_numpy(dtype=float)
            if len(field_idx) >= min_field and np.nanstd(vals) > 1e-12:
                z_field = pd.Series(rank_normal_scores(vals), index=field_idx)
                z[out.index.get_indexer(idx)] = z_field.loc[idx].to_numpy(dtype=float)
                scope.loc[idx] = 'field'

        remaining = valid & ~np.isfinite(z)
        if remaining.any():
            valid_idx = out.index[valid]
            vals = transformed.loc[valid_idx].to_numpy(dtype=float)
            z_global = pd.Series(rank_normal_scores(vals), index=valid_idx)
            rem_idx = out.index[remaining]
            z[out.index.get_indexer(rem_idx)] = z_global.loc[rem_idx].to_numpy(dtype=float)
            scope.loc[rem_idx] = 'global'

        out[col + '_z'] = z
        out[col + '_z_scope'] = scope
        finite = transformed.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        valid_ratio = float(valid.mean()) if len(out) else 0.0
        diagnostics.append({
            'metric': col,
            'valid_ratio': valid_ratio,
            'missing_rate': 1.0 - valid_ratio,
            'zero_rate': float(np.mean(np.abs(finite) < 1e-12)) if len(finite) else np.nan,
            'unique_count': int(pd.Series(finite).nunique()) if len(finite) else 0,
            'iqr': float(np.percentile(finite, 75) - np.percentile(finite, 25)) if len(finite) else np.nan,
            'field_year_fallback_ratio': float(np.mean(scope.loc[valid] != 'field_year')) if valid.any() else 1.0,
            'field_fallback_ratio': float(np.mean(scope.loc[valid] == 'field')) if valid.any() else 0.0,
            'global_fallback_ratio': float(np.mean(scope.loc[valid] == 'global')) if valid.any() else 0.0,
        })
    return out, pd.DataFrame(diagnostics)


def choose_publication_day_community(refs: pd.DataFrame, singleton_comm: int) -> Tuple[int, str]:
    vals = pd.to_numeric(refs.get('target_analysis_comm', pd.Series(dtype=float)), errors='coerce').dropna().astype(int)
    vals = vals[vals >= 0]
    if vals.empty:
        return int(singleton_comm), 'singleton_no_reference_community'
    counts = vals.value_counts()
    max_count = int(counts.max())
    candidates = sorted(int(c) for c, count in counts.items() if int(count) == max_count)
    source = 'reference_majority'
    if len(candidates) > 1:
        source = 'reference_majority_tie_min'
    return int(candidates[0]), source


def add_reference_bins(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    out = df.copy()
    if out['reference_count'].nunique() <= 1:
        out['ref_bin'] = 0
        return out
    q = min(n_bins, max(2, out['reference_count'].nunique()))
    out['ref_bin'] = pd.qcut(out['reference_count'].rank(method='first'), q=q, labels=False, duplicates='drop')
    out['ref_bin'] = out['ref_bin'].fillna(0).astype(int)
    return out


def matched_control_pool(df: pd.DataFrame, row: pd.Series, min_controls: int = 20) -> Tuple[pd.DataFrame, str]:
    same_base = df['paper_id'] != row['paper_id']
    non_landmark = df['is_landmark'].astype(int) == 0
    tiers = [
        ('field_year_refbin', same_base & non_landmark &
         (df['primary_field'] == row['primary_field']) &
         (df['year'].between(int(row['year']) - 1, int(row['year']) + 1)) &
         (df['ref_bin'] == row['ref_bin'])),
        ('field_year', same_base & non_landmark &
         (df['primary_field'] == row['primary_field']) &
         (df['year'].between(int(row['year']) - 1, int(row['year']) + 1))),
        ('field_year3', same_base & non_landmark &
         (df['primary_field'] == row['primary_field']) &
         (df['year'].between(int(row['year']) - 3, int(row['year']) + 3))),
        ('field_all_years', same_base & non_landmark & (df['primary_field'] == row['primary_field'])),
        ('all_non_landmark', same_base & non_landmark),
    ]
    for tier, mask in tiers:
        pool = df[mask].copy()
        if len(pool) >= min_controls or tier == 'all_non_landmark':
            return pool, tier
    return df.iloc[0:0].copy(), 'no_controls'


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def partial_spearman(x: Sequence[float], y: Sequence[float], controls: pd.DataFrame) -> float:
    frame = pd.DataFrame({'x': x, 'y': y}).join(controls.reset_index(drop=True))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame['x'].nunique() <= 1 or frame['y'].nunique() <= 1:
        return np.nan
    xr = frame['x'].rank(method='average').to_numpy(dtype=float)
    yr = frame['y'].rank(method='average').to_numpy(dtype=float)
    ctrl_cols = [c for c in frame.columns if c not in {'x', 'y'}]
    ctrl = frame[ctrl_cols].rank(method='average').to_numpy(dtype=float)
    rx = residualize(xr, ctrl)
    ry = residualize(yr, ctrl)
    if np.nanstd(rx) <= 1e-12 or np.nanstd(ry) <= 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def graph_delta_diagnostics(graph_deltas: pd.DataFrame, delta_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in delta_cols:
        values = pd.to_numeric(graph_deltas[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        finite = values.dropna().to_numpy(dtype=float)
        reasons = []
        active = True
        if len(finite) == 0:
            active = False
            reasons.append('all_nan')
        elif np.nanstd(finite) <= 1e-12 or pd.Series(finite).nunique() <= 1:
            active = False
            reasons.append('zero_variance')
        nonzero_rate = float(np.mean(np.abs(finite) > 1e-12)) if len(finite) else 0.0
        rows.append({
            'delta': col,
            'active': int(active),
            'finite_ratio': float(values.notna().mean()) if len(values) else 0.0,
            'nonzero_rate': nonzero_rate,
            'std': float(np.nanstd(finite)) if len(finite) else np.nan,
            'unique_count': int(pd.Series(finite).nunique()) if len(finite) else 0,
            'drop_reasons': ';'.join(reasons),
        })
    return pd.DataFrame(rows)


def compute_all_metrics(
    raw: RawData,
    focal_ids: Optional[Sequence[str]] = None,
    min_controls: int = 20,
    progress: bool = False,
    progress_interval: int = 25,
) -> ComputedData:
    works = raw.works.copy()
    works['id'] = works['id'].astype(str)
    if focal_ids is None:
        focal = works.copy()
    else:
        focal = works[works['id'].isin([str(x) for x in focal_ids])].copy()
    if focal.empty:
        raise ValueError('No focal papers found to compute Fig. 2 metrics.')

    topic_graph = community_graph_from_topic_edges(raw)
    comm_map_all = dict(zip(works['id'].astype(str), works['analysis_community'].astype(int)))
    singleton_comm = int(max([c for c in comm_map_all.values() if c >= 0] or [0]) + 1)
    citation_neighbors = build_citation_neighbors(raw.citations)
    target_meta = works[['id', 'year', 'analysis_community', 'display_community', 'primary_field']].rename(columns={
        'id': 'target',
        'analysis_community': 'target_analysis_comm',
        'display_community': 'target_display_comm',
        'primary_field': 'target_field',
        'year': 'target_year',
    })
    citations_with_targets = raw.citations.merge(target_meta, on='target', how='left')
    refs_by_source = {
        str(source): group.copy()
        for source, group in citations_with_targets.groupby('source', sort=False)
    }
    progress_interval = max(1, int(progress_interval))

    paper_rows = []
    delta_rows = []
    candidate_rows = []

    # precompute prior literature structures for each year
    years = sorted(focal['year'].unique())
    progress_log(
        f'Computing metrics for {len(focal):,} focal papers across {len(years):,} year slices.',
        progress,
    )
    progress_log('Precomputing prior graphs by year...', progress)
    prior_graphs = {}
    prior_component_sizes: Dict[int, Dict[str, int]] = {}
    prior_component_ids: Dict[int, Dict[str, int]] = {}
    for idx, year in enumerate(years, start=1):
        graph = build_prior_graph(works, raw.citations, int(year))
        prior_graphs[year] = graph
        comp_size: Dict[str, int] = {}
        comp_id: Dict[str, int] = {}
        for cid, nodes in enumerate(nx.connected_components(graph)):
            size = len(nodes)
            for node in nodes:
                comp_size[str(node)] = size
                comp_id[str(node)] = cid
        prior_component_sizes[year] = comp_size
        prior_component_ids[year] = comp_id
        progress_log(
            f'  prior graph {idx}/{len(years)} for year {int(year)}: '
            f'{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges',
            progress,
        )
    prior_refs_by_year = {}
    pairz_by_year = {}
    dist_by_year = {}
    progress_log('Precomputing reference-field statistics by year...', progress)
    for idx, year in enumerate(years, start=1):
        prior_ids = set(works.loc[works['year'] < year, 'id'].astype(str))
        refs = raw.citations[raw.citations['source'].isin(prior_ids)].copy()
        meta = works[['id', 'primary_field']].rename(columns={'id': 'target', 'primary_field': 'target_field'})
        refs = refs.merge(meta, on='target', how='left')
        prior_refs_by_year[year] = refs
        pairz_by_year[year] = pair_zscore_lookup(refs)
        dist_by_year[year] = field_distance_matrix(refs)
        progress_log(
            f'  field stats {idx}/{len(years)} for year {int(year)}: {len(refs):,} prior reference rows',
            progress,
        )

    progress_log(
        'Processing focal papers. This is the slowest step because each paper recomputes '
        'publication-day graph indicators.',
        progress,
    )
    skipped_empty_refs = 0
    total_focal = len(focal)
    for idx, row in enumerate(focal.itertuples(index=False), start=1):
        if idx == 1 or idx % progress_interval == 0 or idx == total_focal:
            progress_log(
                f'  paper {idx:,}/{total_focal:,}: computed={len(paper_rows):,}, '
                f'skipped_no_refs={skipped_empty_refs:,}',
                progress,
            )
        pid = str(row.id)
        year = int(row.year)
        refs_all = refs_by_source.get(pid)
        if refs_all is None:
            refs = pd.DataFrame(columns=citations_with_targets.columns)
        else:
            refs = refs_all[refs_all['target_year'] < year].copy()
        if refs.empty:
            skipped_empty_refs += 1
            continue
        ref_ids = refs['target'].astype(str).tolist()
        ref_fields = refs['target_field'].dropna().astype(str).tolist()
        ref_comms = refs['target_analysis_comm'].dropna().astype(int).tolist()
        ref_display_comms = refs['target_display_comm'].dropna().astype(int).tolist()
        ref_count = len(ref_ids)
        pcomm, p_community_source = choose_publication_day_community(refs, singleton_comm)

        Gm_local, comm_m_local = build_local_reference_graph(refs, citation_neighbors)
        G0_local, comm_0_local = build_local_reference_graph(refs, citation_neighbors, paper_id=pid, paper_comm=pcomm)

        # core seven metrics
        B = focal_bridge_betweenness_from_reference_components(Gm_local, ref_count)

        RS = rao_stirling(ref_fields, dist_by_year[year])
        RTD = simpson_diversity(pd.Series(ref_comms).value_counts().values)
        PDE = shannon_entropy(pd.Series(ref_fields).value_counts().values)

        q_minus = modularity_from_partition(Gm_local, comm_m_local)
        q_zero = modularity_from_partition(G0_local, comm_0_local)
        DeltaQ0_raw_q0_minus_qminus = q_zero - q_minus
        DeltaQ0 = q_minus - q_zero

        # Uzzi-style atypicality: negative tail of field-pair z-scores, flipped so larger = more atypical
        z_lookup = pairz_by_year[year]
        zs = []
        uf = sorted(set(ref_fields))
        for i, fi in enumerate(uf):
            for fj in uf[i+1:]:
                zs.append(z_lookup.get((fi, fj), 0.0))
        Uzzi = float(max(0.0, -np.percentile(zs, 10))) if zs else 0.0

        # Burt IP proxy: high when references are mutually non-redundant in the prior local graph.
        eff_size, BurtIP, constraint_inv = burt_effective_size_proxy(Gm_local, ref_count)

        # candidate / alternatives
        try:
            pagerank = float(nx.pagerank(G0_local).get(pid, 0.0))
        except Exception:
            pagerank = 0.0
        try:
            closeness = float(nx.closeness_centrality(G0_local, pid))
        except Exception:
            closeness = 0.0
        degree = float(G0_local.degree(pid))
        field_shannon = PDE
        field_variety = float(len(set(ref_fields)))
        field_simpson = simpson_diversity(pd.Series(ref_fields).value_counts().values)
        community_simpson = RTD
        field_entropy_norm = float(PDE / max(math.log(max(2, len(set(ref_fields)))), 1e-9)) if len(set(ref_fields)) > 1 else 0.0
        pair_surprisal = Uzzi
        conductance_delta = DeltaQ0  # same direction: higher = stronger boundary shock

        # graph delta observables for panel f
        uniq_comms = sorted(set(ref_comms))
        community_reach = len(uniq_comms)
        modularity_shock = DeltaQ0
        pshort = path_shortening(ref_display_comms, topic_graph)
        seen_components = set()
        comp_reach = 1
        for rid in ref_ids:
            cid = prior_component_ids[year].get(rid)
            if cid is None or cid in seen_components:
                continue
            seen_components.add(cid)
            comp_reach += int(prior_component_sizes[year].get(rid, 0))
        boundary_mixing = boundary_mixing_share(G0_local, comm_0_local) - boundary_mixing_share(Gm_local, comm_m_local)

        paper_rows.append({
            'paper_id': pid, 'title': row.title, 'domain': row.domain, 'year': year,
            'primary_field': row.primary_field, 'is_landmark': int(row.is_landmark), 'reference_count': ref_count,
            'B': B, 'RS': RS, 'DeltaQ0': DeltaQ0, 'Uzzi': Uzzi, 'RTD': RTD, 'BurtIP': BurtIP, 'PDE': PDE,
            'DeltaQ0_raw_q0_minus_qminus': DeltaQ0_raw_q0_minus_qminus,
            'p_analysis_community': pcomm,
            'p_community_source': p_community_source,
            'field_variety': field_variety,
        })

        candidate_rows.append({
            'paper_id': pid, 'title': row.title, 'domain': row.domain, 'year': year,
            'primary_field': row.primary_field, 'is_landmark': int(row.is_landmark), 'reference_count': ref_count,
            'B': B, 'degree': degree, 'pagerank': pagerank, 'closeness': closeness,
            'RS': RS, 'field_shannon': field_shannon, 'field_simpson': field_simpson, 'field_variety': field_variety,
            'DeltaQ0': DeltaQ0, 'conductance_delta': conductance_delta,
            'Uzzi': Uzzi, 'pair_surprisal': pair_surprisal,
            'RTD': RTD, 'community_simpson': community_simpson,
            'BurtIP': BurtIP, 'effective_size': eff_size, 'constraint_inv': constraint_inv,
            'PDE': PDE, 'field_entropy_norm': field_entropy_norm,
        })

        delta_rows.append({
            'paper_id': pid, 'title': row.title, 'domain': row.domain, 'year': year,
            'primary_field': row.primary_field, 'is_landmark': int(row.is_landmark), 'reference_count': ref_count,
            'community_reach': float(community_reach),
            'modularity_shock': float(modularity_shock),
            'path_shortening': float(pshort),
            'component_reach': float(comp_reach),
            'boundary_mixing': float(boundary_mixing),
        })

    paper_metrics = pd.DataFrame(paper_rows)
    candidate_metrics = pd.DataFrame(candidate_rows)
    graph_deltas = pd.DataFrame(delta_rows)
    if paper_metrics.empty:
        raise ValueError('No metrics could be computed. Check citation directions and years.')
    progress_log(
        f'Computed metrics for {len(paper_metrics):,} papers; skipped {skipped_empty_refs:,} papers without prior references.',
        progress,
    )

    metric_cols = ['B', 'RS', 'DeltaQ0', 'Uzzi', 'RTD', 'BurtIP', 'PDE']
    progress_log('Rank-normalizing metrics by field-year / field / global scopes...', progress)
    paper_metrics, metric_diagnostics = field_year_rank_normalize(paper_metrics, metric_cols)

    candidate_cols = [c for c in candidate_metrics.columns if c not in ['paper_id', 'title', 'domain', 'year', 'primary_field', 'is_landmark', 'reference_count']]
    cand_norm, _ = field_year_rank_normalize(candidate_metrics.rename(columns={c: c for c in candidate_cols}), candidate_cols)
    cand_z_cols = [c + '_z' for c in candidate_cols]
    progress_log('Computing candidate-metric redundancy correlations...', progress)
    redundancy_corr = cand_norm[cand_z_cols].corr(method='spearman')
    redundancy_corr.index = candidate_cols
    redundancy_corr.columns = candidate_cols

    progress_log('Computing indicator-to-graph-delta correlations...', progress)
    merged = paper_metrics[['paper_id'] + [m + '_z' for m in metric_cols]].merge(graph_deltas, on='paper_id', how='inner')
    delta_cols = ['community_reach', 'modularity_shock', 'path_shortening', 'component_reach', 'boundary_mixing']
    delta_diagnostics = graph_delta_diagnostics(graph_deltas, delta_cols)
    active_delta_cols = delta_diagnostics.loc[delta_diagnostics['active'].astype(int) == 1, 'delta'].astype(str).tolist()
    corr_mat = pd.DataFrame(index=metric_cols, columns=active_delta_cols, dtype=float)
    controls = pd.DataFrame({
        'year': pd.to_numeric(merged['year'], errors='coerce'),
        'log_reference_count': np.log1p(pd.to_numeric(merged['reference_count'], errors='coerce').fillna(0)),
    })
    for m in metric_cols:
        for d in active_delta_cols:
            corr_mat.loc[m, d] = partial_spearman(merged[m + '_z'], merged[d], controls)

    # matched-control percentiles for landmark papers
    progress_log('Computing landmark matched-control percentiles...', progress)
    percentile_rows = []
    landmarks = paper_metrics[paper_metrics['is_landmark'] == 1].copy()
    if not landmarks.empty:
        paper_metrics = add_reference_bins(paper_metrics)
        landmarks = paper_metrics[paper_metrics['is_landmark'] == 1].copy()
        for _, lm in landmarks.iterrows():
            pool, tier = matched_control_pool(paper_metrics, lm, min_controls=min_controls)
            for m in metric_cols:
                pct = percentile_vs_controls(float(lm[m]), pool[m].values)
                percentile_rows.append({
                    'paper_id': lm['paper_id'], 'title': lm['title'], 'metric': m, 'percentile': pct,
                    'year': lm['year'], 'primary_field': lm['primary_field'], 'n_controls': len(pool),
                    'control_tier': tier,
                })
    percentile_long = pd.DataFrame(percentile_rows)
    landmark_summary = percentile_long.groupby('metric', as_index=False).agg(percentile=('percentile', 'median')) if not percentile_long.empty else pd.DataFrame(columns=['metric', 'percentile'])
    progress_log('Metric computation finished.', progress)

    return ComputedData(
        paper_metrics=paper_metrics,
        candidate_metrics=candidate_metrics,
        graph_deltas=graph_deltas,
        redundancy_corr=redundancy_corr,
        indicator_delta_corr=corr_mat,
        percentile_long=percentile_long,
        landmark_summary=landmark_summary,
        metric_standardization_diagnostics=metric_diagnostics,
        graph_delta_diagnostics=delta_diagnostics,
    )


# --------------------------
# Strong-evidence multi-domain mode
# --------------------------

FIG3_STRONG_INPUT_SUBDIR = 'fig2_strong_input'
DEFINITION_LINKED_DELTAS = {'modularity_shock'}
STRONG_FUTURE_DELTAS = [
    'community_reach',
    'field_entropy',
    'cross_community_adoption',
    'path_shortening',
    'modularity_shock',
    'partition_change',
    'boundary_mixing',
    'hub_formation',
]
INDEPENDENT_FUTURE_DELTAS = [
    'community_reach',
    'field_entropy',
    'cross_community_adoption',
    'path_shortening',
    'partition_change',
    'boundary_mixing',
    'hub_formation',
]
EXPECTED_FUTURE_LINKS = {
    'B': ['path_shortening', 'hub_formation', 'boundary_mixing'],
    'RS': ['field_entropy', 'community_reach', 'cross_community_adoption'],
    'DeltaQ0': ['boundary_mixing', 'partition_change', 'modularity_shock'],
    'Uzzi': ['partition_change', 'boundary_mixing'],
    'RTD': ['community_reach', 'field_entropy'],
    'BurtIP': ['path_shortening', 'hub_formation'],
    'PDE': ['field_entropy', 'community_reach', 'cross_community_adoption'],
}


def parse_domain_string(value: str) -> List[str]:
    parts = []
    for token in str(value or '').replace(',', ' ').split():
        token = token.strip()
        if token:
            parts.append(token)
    return list(dict.fromkeys(parts))


def stable_int_id(value: object, modulo: int = 100_000_000) -> int:
    text = str(value or 'unknown')
    digest = hashlib.md5(text.encode('utf-8')).hexdigest()
    return int(digest[:12], 16) % modulo


def normalize_openalex_id(value: object) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    if text.startswith('https://openalex.org/'):
        return text
    if '/W' in text:
        return 'https://openalex.org/' + text.rsplit('/', 1)[-1]
    if text.startswith('W') and text[1:].isdigit():
        return f'https://openalex.org/{text}'
    return text


def read_fig1_raw_records(fig1_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    path = fig1_dir / 'works_raw.jsonl'
    if not path.exists():
        return {}, {}
    by_id: Dict[str, Dict[str, Any]] = {}
    refs_by_id: Dict[str, List[str]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            wid = normalize_openalex_id(rec.get('id'))
            if not wid:
                continue
            refs = [normalize_openalex_id(ref) for ref in (rec.get('refs') or []) if normalize_openalex_id(ref)]
            by_id[wid] = rec
            refs_by_id[wid] = refs
    return by_id, refs_by_id


def closure_work_row(work: Mapping[str, Any], domain: str) -> Dict[str, Any]:
    wid = normalize_openalex_id(work.get('id') or (work.get('ids') or {}).get('openalex'))
    primary = work.get('primary_topic') or {}
    topics = work.get('topics') or []
    topic = primary or (topics[0] if topics else {})
    topic_id = normalize_openalex_id(topic.get('id'))
    topic_name = str(topic.get('display_name') or 'unknown_topic')
    field_obj = topic.get('field') or {}
    subfield_obj = topic.get('subfield') or {}
    primary_field = str(
        subfield_obj.get('display_name')
        or field_obj.get('display_name')
        or topic_name
        or domain
    )
    return {
        'id': wid,
        'year': int(work.get('publication_year') or 0),
        'title': work.get('display_name') or wid,
        'domain': domain,
        'primary_field': primary_field,
        'primary_topic': topic_name,
        'display_community': stable_int_id(topic_id or topic_name),
        'community_label': topic_name,
        'is_landmark': 0,
        'anchor_label': '',
        'short_id': wid.rsplit('/', 1)[-1],
        'doi': work.get('doi') or '',
        'cited_by_count': work.get('cited_by_count', np.nan),
        'is_closure_node': 1,
    }


def add_reference_closure_nodes(
    raw: Any,
    fig1_dir: Path,
    domain: str,
    reference_closure: str,
    online_expand: bool,
    closure_cap: int,
    closure_coverage_target: float,
    openalex_api_key: Optional[str],
    openalex_api_keys: Optional[str],
    email: Optional[str],
    progress: bool,
) -> Tuple[Any, Dict[str, Any]]:
    """Optionally add out-of-corpus reference metadata as background nodes."""
    records_by_id, refs_by_id = read_fig1_raw_records(fig1_dir)
    cached_reference_fallback = False
    if not refs_by_id and 'referenced_works' in raw.works.columns:
        cached_reference_fallback = True
        refs_by_id = {
            str(row['id']): parse_reference_list(row.get('referenced_works'))
            for _, row in raw.works.iterrows()
            if str(row.get('id', '')).strip()
        }
    existing_ids = set(raw.works['id'].astype(str))
    total_refs = 0
    internal_refs = 0
    external_counts: Counter[str] = Counter()
    for source, refs in refs_by_id.items():
        if source not in existing_ids:
            continue
        for ref in refs:
            total_refs += 1
            if ref in existing_ids:
                internal_refs += 1
            else:
                external_counts[ref] += 1

    ordered_external = external_counts.most_common(max(0, int(closure_cap)))
    target_ids: List[str] = []
    covered_by_targeted = internal_refs
    target_coverage = float(closure_coverage_target)
    for ref, count in ordered_external:
        if len(target_ids) >= int(closure_cap):
            break
        target_ids.append(ref)
        covered_by_targeted += int(count)
        if total_refs and covered_by_targeted / total_refs >= target_coverage:
            break

    cached_materialized_coverage = float((internal_refs + sum(external_counts.values())) / total_refs) if total_refs else 0.0
    report: Dict[str, Any] = {
        'domain': domain,
        'reference_closure_mode': reference_closure,
        'online_expand': int(bool(online_expand)),
        'raw_records': len(records_by_id),
        'total_reference_mentions': int(total_refs),
        'internal_reference_mentions': int(internal_refs),
        'external_reference_mentions': int(total_refs - internal_refs),
        'external_unique_references': int(len(external_counts)),
        'closure_cap': int(closure_cap),
        'closure_coverage_target': float(closure_coverage_target),
        'targeted_closure_unique_references': int(len(target_ids)),
        'coverage_without_closure': float(internal_refs / total_refs) if total_refs else 1.0,
        'coverage_if_targeted_materialized': float(covered_by_targeted / total_refs) if total_refs else 1.0,
        'materialized_closure_unique_references': 0,
        'coverage_materialized': float(internal_refs / total_refs) if total_refs else 1.0,
        'status': 'audit_only_no_online_closure',
    }
    if cached_reference_fallback and total_refs > 0:
        report.update({
            'reference_closure_mode': 'cached_openalex_referenced_works',
            'raw_records': int(len(refs_by_id)),
            'targeted_closure_unique_references': int(len(external_counts)),
            'materialized_closure_unique_references': int(len(external_counts)),
            'coverage_if_targeted_materialized': cached_materialized_coverage,
            'coverage_materialized': cached_materialized_coverage,
            'status': 'cached_referenced_works_materialized',
        })
    raw.works = raw.works.copy()
    if 'is_closure_node' not in raw.works.columns:
        raw.works['is_closure_node'] = 0
    raw.works['is_closure_node'] = pd.to_numeric(raw.works['is_closure_node'], errors='coerce').fillna(0).astype(int)

    should_fetch = reference_closure in {'required', 'auto'} and bool(online_expand) and len(target_ids) > 0
    if not should_fetch:
        if reference_closure == 'required' and not online_expand:
            report['status'] = 'required_but_online_expand_false'
        return raw, report

    try:
        from experiments.fig03.old.dataset_builder import OpenAlexClient  # pylint: disable=import-outside-toplevel
        from experiments.fig03.old.fig3_empirical_weight_learning import topics_from_works_and_citations  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        report['status'] = f'openalex_import_failed:{exc}'
        return raw, report

    client = OpenAlexClient(
        api_key=openalex_api_key,
        api_keys=openalex_api_keys,
        email=email,
        sleep_seconds=0.1,
        timeout_seconds=60,
        max_retries=5,
    )
    progress_log(
        f'[{domain}] Fetching OpenAlex reference-closure metadata for up to {len(target_ids):,} refs.',
        progress,
    )
    rows: List[Dict[str, Any]] = []
    fetched_work_refs: Dict[str, List[str]] = {}
    for idx, ref in enumerate(target_ids, start=1):
        if idx == 1 or idx % 250 == 0 or idx == len(target_ids):
            progress_log(f'[{domain}] closure fetch {idx:,}/{len(target_ids):,}', progress)
        work = client.get_work(ref)
        if not work:
            continue
        row = closure_work_row(work, domain)
        if not row['id'] or int(row['year']) <= 0:
            continue
        rows.append(row)
        fetched_work_refs[row['id']] = [
            normalize_openalex_id(item)
            for item in (work.get('referenced_works') or [])
            if normalize_openalex_id(item)
        ]
        time.sleep(0.01)

    if not rows:
        report['status'] = 'online_fetch_returned_no_usable_closure_nodes'
        return raw, report

    closure_df = pd.DataFrame(rows).drop_duplicates(subset=['id'])
    closure_ids = set(closure_df['id'].astype(str))
    works = pd.concat([raw.works, closure_df], ignore_index=True).drop_duplicates(subset=['id'])
    valid_ids = set(works['id'].astype(str))

    extra_edges: List[Dict[str, str]] = []
    for source, refs in refs_by_id.items():
        if source not in existing_ids:
            continue
        for ref in refs:
            if ref in closure_ids:
                extra_edges.append({'source': source, 'target': ref})
    for source, refs in fetched_work_refs.items():
        if source not in closure_ids:
            continue
        for ref in refs:
            if ref in valid_ids:
                extra_edges.append({'source': source, 'target': ref})

    citations = pd.concat([raw.citations, pd.DataFrame(extra_edges)], ignore_index=True).drop_duplicates()
    topics, topic_edges = topics_from_works_and_citations(works, citations)
    raw.works = works
    raw.citations = citations
    raw.topics = topics
    raw.topic_edges = topic_edges
    materialized_mentions = internal_refs + sum(external_counts.get(ref, 0) for ref in closure_ids)
    report.update({
        'materialized_closure_unique_references': int(len(closure_ids)),
        'closure_citation_edges_added': int(len(extra_edges)),
        'coverage_materialized': float(materialized_mentions / total_refs) if total_refs else 1.0,
        'status': 'online_closure_materialized',
    })
    return raw, report


def fig3_raw_to_fig2_raw(raw: Any, domain: Optional[str] = None) -> RawData:
    works = raw.works.copy()
    if domain is not None and 'domain' in works.columns:
        works = works[works['domain'].astype(str) == str(domain)].copy()
        valid_ids = set(works['id'].astype(str))
        citations = raw.citations[raw.citations['source'].astype(str).isin(valid_ids) & raw.citations['target'].astype(str).isin(valid_ids)].copy()
    else:
        citations = raw.citations.copy()
    if 'analysis_community' not in works.columns:
        works['analysis_community'] = pd.to_numeric(works['display_community'], errors='coerce').fillna(-1).astype(int)
    if 'anchor_label' not in works.columns:
        works['anchor_label'] = ''
    topics = raw.topics.copy()
    topic_edges = raw.topic_edges.copy()
    return RawData(works=works, citations=citations, topics=topics, topic_edges=topic_edges)


def domain_input_audit(domain: str, fig1_dir: Path, raw: Any, closure_report: Mapping[str, Any]) -> Dict[str, Any]:
    works = raw.works.copy()
    citations = raw.citations.copy()
    eligible_years = pd.to_numeric(works['year'], errors='coerce').dropna()
    closure_node_count = 0
    if 'is_closure_node' in works.columns:
        closure_node_count = int(pd.to_numeric(works['is_closure_node'], errors='coerce').fillna(0).sum())
    return {
        'domain': domain,
        'fig1_dir': str(fig1_dir),
        'raw_papers': int(len(works)),
        'closure_nodes': closure_node_count,
        'citation_edges': int(len(citations)),
        'landmarks': int(pd.to_numeric(works.get('is_landmark', 0), errors='coerce').fillna(0).sum()),
        'year_min': int(eligible_years.min()) if len(eligible_years) else 0,
        'year_max': int(eligible_years.max()) if len(eligible_years) else 0,
        'reference_closure_coverage': float(closure_report.get('coverage_materialized', np.nan)),
        'reference_closure_status': str(closure_report.get('status', 'unknown')),
    }


def controls_with_domain_dummies(df: pd.DataFrame) -> pd.DataFrame:
    controls = pd.DataFrame({
        'year': pd.to_numeric(df['year'], errors='coerce'),
        'log_reference_count': np.log1p(pd.to_numeric(df['reference_count'], errors='coerce').fillna(0)),
    }, index=df.index)
    if 'domain' in df.columns and df['domain'].nunique() > 1:
        dummies = pd.get_dummies(df['domain'].astype(str), prefix='domain', drop_first=True)
        controls = pd.concat([controls, dummies.astype(float)], axis=1)
    return controls.reset_index(drop=True)


def partial_spearman_bootstrap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    work = df[[x_col, y_col, 'domain'] + list(control_cols)].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 20 or work[x_col].nunique() <= 1 or work[y_col].nunique() <= 1:
        return {'rho': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': int(len(work)), 'n_bootstrap': 0}
    rho = partial_spearman(work[x_col], work[y_col], work[list(control_cols)])
    rng = np.random.default_rng(int(seed))
    domains = work['domain'].astype(str).unique().tolist()
    vals: List[float] = []
    for _ in range(int(n_bootstrap)):
        if domains:
            sampled_domains = rng.choice(domains, size=len(domains), replace=True)
            boot = pd.concat([work[work['domain'].astype(str) == d] for d in sampled_domains], ignore_index=True)
        else:
            boot = work.sample(n=len(work), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
        if boot[x_col].nunique() <= 1 or boot[y_col].nunique() <= 1:
            continue
        val = partial_spearman(boot[x_col], boot[y_col], boot[list(control_cols)])
        if np.isfinite(val):
            vals.append(float(val))
    if vals:
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
    else:
        ci_low = ci_high = np.nan
    return {
        'rho': float(rho),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n': int(len(work)),
        'n_bootstrap': int(len(vals)),
    }


def residualized_spearman_corr(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    work = df[list(cols) + ['domain', 'year', 'reference_count']].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out = pd.DataFrame(index=list(cols), columns=list(cols), dtype=float)
    if len(work) < 10:
        return out
    controls = controls_with_domain_dummies(work)
    for c1 in cols:
        for c2 in cols:
            out.loc[c1, c2] = partial_spearman(work[c1], work[c2], controls)
    return out


def future_composite_scores(graph_deltas: pd.DataFrame, active_delta_cols: Sequence[str]) -> pd.Series:
    active = [c for c in active_delta_cols if c in INDEPENDENT_FUTURE_DELTAS and c in graph_deltas.columns]
    if not active:
        return pd.Series(np.nan, index=graph_deltas.index)
    z_cols = []
    for col in active:
        vals = pd.to_numeric(graph_deltas[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        z_cols.append(pd.Series(rank_normal_scores(vals), index=graph_deltas.index))
    return pd.concat(z_cols, axis=1).mean(axis=1)


def build_matched_percentiles(
    paper_metrics: pd.DataFrame,
    graph_deltas: pd.DataFrame,
    active_delta_cols: Sequence[str],
    min_controls: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [m[0] for m in METRIC_SPECS]
    df = paper_metrics.merge(
        graph_deltas[['paper_id'] + [c for c in active_delta_cols if c in graph_deltas.columns]],
        on='paper_id',
        how='left',
    )
    df = add_reference_bins(df)
    df['future_rgpm_proxy'] = future_composite_scores(df, active_delta_cols)
    focal_parts = []
    landmarks = df[df['is_landmark'].astype(int) == 1].copy()
    if not landmarks.empty:
        landmarks['profile_type'] = 'landmark'
        focal_parts.append(landmarks)
    high_rows = []
    for domain, sub in df[df['is_landmark'].astype(int) == 0].groupby('domain', sort=True):
        vals = pd.to_numeric(sub['future_rgpm_proxy'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        vals = vals.dropna()
        if vals.empty:
            continue
        threshold = float(vals.quantile(0.90))
        picked = sub[pd.to_numeric(sub['future_rgpm_proxy'], errors='coerce') >= threshold].copy()
        picked = picked.sort_values('future_rgpm_proxy', ascending=False).head(250)
        picked['profile_type'] = 'future_top_decile'
        high_rows.append(picked)
    if high_rows:
        focal_parts.append(pd.concat(high_rows, ignore_index=True))
    if not focal_parts:
        return pd.DataFrame(), pd.DataFrame()
    focals = pd.concat(focal_parts, ignore_index=True).drop_duplicates(subset=['paper_id', 'profile_type'])
    percentile_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    for _, row in focals.iterrows():
        pool, tier = matched_control_pool(df, row, min_controls=min_controls)
        control_rows.append({
            'paper_id': row['paper_id'],
            'title': row.get('title', ''),
            'domain': row.get('domain', ''),
            'year': row.get('year', np.nan),
            'primary_field': row.get('primary_field', ''),
            'profile_type': row.get('profile_type', 'focal'),
            'control_tier': tier,
            'n_controls': int(len(pool)),
        })
        for metric in metric_cols:
            pct = percentile_vs_controls(float(row[metric]), pool[metric].values) if metric in row and metric in pool else np.nan
            percentile_rows.append({
                'paper_id': row['paper_id'],
                'title': row.get('title', ''),
                'domain': row.get('domain', ''),
                'year': row.get('year', np.nan),
                'primary_field': row.get('primary_field', ''),
                'profile_type': row.get('profile_type', 'focal'),
                'metric': metric,
                'percentile': pct,
                'n_controls': int(len(pool)),
                'control_tier': tier,
                'future_rgpm_proxy': row.get('future_rgpm_proxy', np.nan),
            })
    return pd.DataFrame(percentile_rows), pd.DataFrame(control_rows)


def build_evidence_support(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric, outcomes in EXPECTED_FUTURE_LINKS.items():
        sub = bootstrap[(bootstrap['metric'] == metric) & (bootstrap['future_outcome'].isin(outcomes))].copy()
        significant = sub[(pd.to_numeric(sub['ci_low'], errors='coerce') > 0) & (pd.to_numeric(sub['rho'], errors='coerce') > 0)]
        rows.append({
            'metric': metric,
            'expected_outcomes': ','.join(outcomes),
            'tested_expected_outcomes': int(len(sub)),
            'significant_expected_outcomes': int(len(significant)),
            'best_expected_rho': float(pd.to_numeric(sub['rho'], errors='coerce').max()) if not sub.empty else np.nan,
        })
    return pd.DataFrame(rows)


def reference_closure_gate_values(reference_closure_report: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    normalized = normalize_reference_closure_report(reference_closure_report)
    if normalized.empty:
        return normalized, 0, 0.0
    measured = pd.to_numeric(normalized['coverage_measured'], errors='coerce').fillna(0).astype(int)
    measured_all = int(len(normalized) > 0 and measured.min() == 1)
    measured_coverage = pd.to_numeric(
        normalized.loc[measured.astype(bool), 'coverage_materialized'],
        errors='coerce',
    )
    min_closure = float(measured_coverage.min()) if not measured_coverage.empty else 0.0
    return normalized, measured_all, min_closure


def refresh_reference_closure_quality(
    quality_gates: Mapping[str, Any],
    reference_closure_report: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    normalized, measured_all, min_closure = reference_closure_gate_values(reference_closure_report)
    quality = copy.deepcopy(dict(quality_gates))
    checks = dict(quality.get('checks') or {})
    checks['reference_closure_coverage_min80pct'] = int(measured_all == 1 and min_closure >= 0.80)
    quality['checks'] = checks
    quality['reference_closure_measured_all_domains'] = bool(measured_all)
    quality['min_reference_closure_coverage'] = min_closure
    quality['overall_pass'] = bool(all(checks.values())) if checks else False
    quality['status_label'] = 'strong experimental evidence' if quality['overall_pass'] else 'multi-domain diagnostic evidence'
    return quality, normalized


def build_fig2_quality_gates(
    n_domains: int,
    total_eligible_papers: int,
    active_future_outcomes: Sequence[str],
    relaxed_control_tier_ratio: float,
    reference_closure_measured_all_domains: bool,
    min_reference_closure_coverage: float,
    significant_expected_links: int,
    mechanism_composite_partial_spearman: float,
) -> Dict[str, Any]:
    """Build Fig. 2 strong-claim quality gates from scalar audit values."""
    checks = {
        'included_domains_min4': int(int(n_domains) >= 4),
        'total_eligible_papers_min8000': int(int(total_eligible_papers) >= 8000),
        'active_future_outcomes_min5': int(len(list(active_future_outcomes)) >= 5),
        'relaxed_control_tier_ratio_max25pct': int(float(relaxed_control_tier_ratio) <= 0.25),
        'reference_closure_coverage_min80pct': int(
            bool(reference_closure_measured_all_domains) and float(min_reference_closure_coverage) >= 0.80
        ),
        'significant_expected_links_min4': int(int(significant_expected_links) >= 4),
        'mechanism_composite_rho_min020': int(
            np.isfinite(float(mechanism_composite_partial_spearman))
            and float(mechanism_composite_partial_spearman) >= 0.20
        ),
    }
    overall = bool(all(checks.values()))
    return {
        'overall_pass': overall,
        'status_label': 'strong experimental evidence' if overall else 'multi-domain diagnostic evidence',
        'checks': checks,
        'n_domains': int(n_domains),
        'total_eligible_papers': int(total_eligible_papers),
        'active_future_outcomes': list(active_future_outcomes),
        'relaxed_control_tier_ratio': float(relaxed_control_tier_ratio),
        'reference_closure_measured_all_domains': bool(reference_closure_measured_all_domains),
        'min_reference_closure_coverage': float(min_reference_closure_coverage),
        'significant_expected_links': int(significant_expected_links),
        'mechanism_composite_partial_spearman': (
            float(mechanism_composite_partial_spearman)
            if np.isfinite(float(mechanism_composite_partial_spearman))
            else None
        ),
        'thresholds': {
            'domains_min': 4,
            'eligible_papers_min': 8000,
            'active_future_outcomes_min': 5,
            'relaxed_control_tier_ratio_max': 0.25,
            'reference_closure_coverage_min': 0.80,
            'significant_expected_links_min': 4,
            'mechanism_composite_rho_min': 0.20,
        },
    }


def build_quality_gates(
    paper_metrics: pd.DataFrame,
    graph_delta_diagnostics_df: pd.DataFrame,
    reference_closure_report: pd.DataFrame,
    matched_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    composite_rho: float,
) -> Dict[str, Any]:
    active_independent = graph_delta_diagnostics_df[
        (graph_delta_diagnostics_df['active'].astype(int) == 1)
        & (~graph_delta_diagnostics_df['delta'].astype(str).isin(DEFINITION_LINKED_DELTAS))
    ]
    relaxed_tiers = {'field_year3', 'field_all_years', 'all_non_landmark', 'no_controls'}
    relaxed_ratio = float(matched_controls['control_tier'].isin(relaxed_tiers).mean()) if not matched_controls.empty else 1.0
    _, closure_measured_all, min_closure = reference_closure_gate_values(reference_closure_report)
    sig_expected = 0
    for metric, outcomes in EXPECTED_FUTURE_LINKS.items():
        sub = bootstrap[(bootstrap['metric'] == metric) & (bootstrap['future_outcome'].isin(outcomes))]
        sig_expected += int(((pd.to_numeric(sub['ci_low'], errors='coerce') > 0) & (pd.to_numeric(sub['rho'], errors='coerce') > 0)).sum())
    return build_fig2_quality_gates(
        n_domains=int(paper_metrics['domain'].nunique()),
        total_eligible_papers=int(len(paper_metrics)),
        active_future_outcomes=active_independent['delta'].astype(str).tolist(),
        relaxed_control_tier_ratio=relaxed_ratio,
        reference_closure_measured_all_domains=bool(closure_measured_all),
        min_reference_closure_coverage=min_closure,
        significant_expected_links=int(sig_expected),
        mechanism_composite_partial_spearman=float(composite_rho) if np.isfinite(composite_rho) else float('nan'),
    )


def build_strong_comp_from_tables(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    input_audit: pd.DataFrame,
    reference_closure_report: pd.DataFrame,
    future_tau: int,
    min_controls: int,
    bootstrap_reps: int,
    seed: int,
    progress: bool,
) -> ComputedData:
    metric_cols = [m[0] for m in METRIC_SPECS]
    paper_metrics = metrics.copy()
    reference_closure_report = normalize_reference_closure_report(reference_closure_report)
    if 'DeltaQ0_raw_q0_minus_qminus' not in paper_metrics.columns and 'DeltaQ0' in paper_metrics.columns:
        paper_metrics['DeltaQ0_raw_q0_minus_qminus'] = -pd.to_numeric(paper_metrics['DeltaQ0'], errors='coerce')
    paper_metrics, metric_diag = field_year_rank_normalize(paper_metrics, metric_cols)

    candidate_cols = [
        'B', 'degree_p', 'RS', 'field_simpson', 'field_variety', 'DeltaQ0', 'Uzzi',
        'RTD', 'community_variety', 'BurtIP', 'effective_size', 'constraint_inv', 'PDE',
    ]
    candidate_cols = [c for c in candidate_cols if c in paper_metrics.columns]
    candidate_metrics = paper_metrics[
        ['paper_id', 'title', 'domain', 'year', 'primary_field', 'is_landmark', 'reference_count'] + candidate_cols
    ].copy()
    candidate_metrics = candidate_metrics.rename(columns={'degree_p': 'degree'})
    candidate_cols = ['degree' if c == 'degree_p' else c for c in candidate_cols]
    candidate_norm, _ = field_year_rank_normalize(candidate_metrics, candidate_cols)
    redundancy_corr = residualized_spearman_corr(candidate_norm, [c + '_z' for c in candidate_cols])
    redundancy_corr.index = candidate_cols
    redundancy_corr.columns = candidate_cols

    graph_deltas = deltas.copy()
    delta_cols = [c for c in STRONG_FUTURE_DELTAS if c in graph_deltas.columns]
    delta_diag = graph_delta_diagnostics(graph_deltas, delta_cols)
    active_delta_cols = delta_diag.loc[delta_diag['active'].astype(int) == 1, 'delta'].astype(str).tolist()

    progress_log('Computing strong-mode indicator/future partial correlations with bootstrap CIs...', progress)
    merged = paper_metrics[['paper_id', 'domain', 'year', 'reference_count'] + [m + '_z' for m in metric_cols]].merge(
        graph_deltas[['paper_id'] + active_delta_cols],
        on='paper_id',
        how='inner',
    )
    controls = controls_with_domain_dummies(merged)
    merged_for_corr = merged.reset_index(drop=True).join(controls.add_prefix('ctrl_'))
    control_cols = [c for c in merged_for_corr.columns if c.startswith('ctrl_')]
    corr_mat = pd.DataFrame(index=metric_cols, columns=active_delta_cols, dtype=float)
    boot_rows: List[Dict[str, Any]] = []
    for metric in metric_cols:
        x_col = metric + '_z'
        for outcome in active_delta_cols:
            stats = partial_spearman_bootstrap(
                merged_for_corr,
                x_col=x_col,
                y_col=outcome,
                control_cols=control_cols,
                n_bootstrap=bootstrap_reps,
                seed=seed + stable_int_id(metric + outcome, modulo=100_000),
            )
            corr_mat.loc[metric, outcome] = stats['rho']
            boot_rows.append({
                'metric': metric,
                'future_outcome': outcome,
                **stats,
                'definition_linked_internal_check': int(outcome in DEFINITION_LINKED_DELTAS),
            })
    bootstrap = pd.DataFrame(boot_rows)
    support = build_evidence_support(bootstrap)

    percentile_long, matched_controls = build_matched_percentiles(
        paper_metrics,
        graph_deltas,
        active_delta_cols=active_delta_cols,
        min_controls=min_controls,
    )
    landmark_summary = (
        percentile_long[percentile_long['profile_type'] == 'landmark']
        .groupby('metric', as_index=False)
        .agg(percentile=('percentile', 'median'))
        if not percentile_long.empty else pd.DataFrame(columns=['metric', 'percentile'])
    )
    if landmark_summary.empty and not percentile_long.empty:
        landmark_summary = percentile_long.groupby('metric', as_index=False).agg(percentile=('percentile', 'median'))

    domain_rows: List[Dict[str, Any]] = []
    for domain, sub in paper_metrics.groupby('domain', sort=True):
        controls_sub = matched_controls[matched_controls['domain'].astype(str) == str(domain)] if not matched_controls.empty else pd.DataFrame()
        closure_sub = reference_closure_report[reference_closure_report['domain'].astype(str) == str(domain)]
        domain_rows.append({
            'domain': domain,
            'eligible_papers': int(len(sub)),
            'landmark_papers': int(pd.to_numeric(sub['is_landmark'], errors='coerce').fillna(0).sum()),
            'year_min': int(pd.to_numeric(sub['year'], errors='coerce').min()),
            'year_max': int(pd.to_numeric(sub['year'], errors='coerce').max()),
            'median_reference_count': float(pd.to_numeric(sub['reference_count'], errors='coerce').median()),
            'median_controls': float(pd.to_numeric(controls_sub['n_controls'], errors='coerce').median()) if not controls_sub.empty else np.nan,
            'relaxed_control_ratio': float(controls_sub['control_tier'].isin({'field_year3', 'field_all_years', 'all_non_landmark', 'no_controls'}).mean()) if not controls_sub.empty else np.nan,
            'reference_closure_coverage': float(pd.to_numeric(closure_sub['coverage_materialized'], errors='coerce').iloc[0]) if not closure_sub.empty else np.nan,
        })
    domain_adequacy = pd.DataFrame(domain_rows)

    future_proxy = future_composite_scores(graph_deltas, active_delta_cols)
    composite_frame = paper_metrics[['paper_id', 'domain', 'year', 'reference_count'] + [m + '_z' for m in metric_cols]].merge(
        pd.DataFrame({'paper_id': graph_deltas['paper_id'], 'future_rgpm_proxy': future_proxy}),
        on='paper_id',
        how='inner',
    )
    composite_frame['indicator_composite'] = composite_frame[[m + '_z' for m in metric_cols]].mean(axis=1)
    comp_controls = controls_with_domain_dummies(composite_frame)
    composite_rho = partial_spearman(composite_frame['indicator_composite'], composite_frame['future_rgpm_proxy'], comp_controls)
    quality = build_quality_gates(
        paper_metrics=paper_metrics,
        graph_delta_diagnostics_df=delta_diag,
        reference_closure_report=reference_closure_report,
        matched_controls=matched_controls,
        bootstrap=bootstrap,
        composite_rho=composite_rho,
    )

    return ComputedData(
        paper_metrics=paper_metrics,
        candidate_metrics=candidate_metrics,
        graph_deltas=graph_deltas,
        redundancy_corr=redundancy_corr,
        indicator_delta_corr=corr_mat,
        percentile_long=percentile_long,
        landmark_summary=landmark_summary,
        metric_standardization_diagnostics=metric_diag,
        graph_delta_diagnostics=delta_diag,
        input_audit=input_audit,
        domain_adequacy=domain_adequacy,
        reference_closure_report=reference_closure_report,
        matched_controls=matched_controls,
        indicator_future_corr_bootstrap=bootstrap,
        evidence_support=support,
        quality_gates=quality,
        evidence_mode='strong',
        future_tau=int(future_tau),
    )


def build_strong_evidence_data(args: argparse.Namespace, progress: bool) -> Tuple[RawData, ComputedData]:
    try:
        from experiments.fig03.old import fig3_empirical_weight_learning as fig3  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise RuntimeError(f'Cannot import Fig. 3 data builders required for strong evidence mode: {exc}') from exc

    domains = parse_domain_string(args.domains) or DEFAULT_STRONG_DOMAINS
    progress_log(f'Strong mode domains: {", ".join(domains)}', progress)
    raw_by_domain: Dict[str, Any] = {}
    audit_rows: List[Dict[str, Any]] = []
    closure_rows: List[Dict[str, Any]] = []

    for domain in domains:
        fig1_dir = args.data_dir / domain if (args.data_dir / domain).exists() else args.data_dir
        prepared_dir = fig3.prepare_fig3_input_data(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            domain=domain,
            direct_only=not args.include_hybrid_edges,
            analysis_end_year=None,
            fig1_config=args.fig1_config,
            fig1_corpus_source=args.fig1_corpus_source,
            run_fig1_if_missing=args.run_fig1_if_missing,
            use_fig1_cache=not args.no_fig1_cache,
            openalex_api_key=args.openalex_api_key,
            openalex_api_keys=args.openalex_api_keys,
            email=args.email,
            progress=progress,
        )
        raw3 = fig3.load_raw_data(prepared_dir)
        raw3, closure_report = add_reference_closure_nodes(
            raw=raw3,
            fig1_dir=fig1_dir,
            domain=domain,
            reference_closure=args.reference_closure,
            online_expand=args.online_expand,
            closure_cap=args.reference_closure_cap,
            closure_coverage_target=args.closure_coverage_target,
            openalex_api_key=args.openalex_api_key,
            openalex_api_keys=args.openalex_api_keys,
            email=args.email,
            progress=progress,
        )
        closure_report = normalize_reference_closure_report(pd.DataFrame([closure_report])).iloc[0].to_dict()
        raw_by_domain[domain] = raw3
        closure_rows.append(closure_report)
        audit_rows.append(domain_input_audit(domain, fig1_dir, raw3, closure_report))
        domain_out = args.out_dir / FIG3_STRONG_INPUT_SUBDIR / domain
        fig3.write_raw_data(raw3, domain_out)

    if len(raw_by_domain) < 2:
        raise ValueError('Strong evidence mode requires at least two available domains.')

    multi_raw = fig3.combine_domain_raws(raw_by_domain)
    multi_input_dir = args.out_dir / FIG3_STRONG_INPUT_SUBDIR / 'multi_domain'
    fig3.write_raw_data(multi_raw, multi_input_dir)
    (multi_input_dir / 'fig2_strong_input_report.json').write_text(
        json.dumps({
            'source_kind': 'fig2_strong_combined_raw',
            'domains': domains,
            'fig1_corpus_source': args.fig1_corpus_source,
            'future_tau': int(args.future_tau),
            'works_rows': int(len(multi_raw.works)),
            'citation_rows': int(len(multi_raw.citations)),
            'analysis_end_year': int(multi_raw.analysis_end_year),
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    progress_log('Computing strong-mode publication-day indicators and future graph deltas...', progress)
    metrics, deltas = fig3.compute_indicator_and_delta_tables(
        multi_raw,
        tau=int(args.future_tau),
        min_refs=int(args.min_refs),
        max_papers=args.max_papers,
        progress=progress,
        progress_interval=int(args.progress_interval),
    )
    input_audit = pd.DataFrame(audit_rows)
    closure_report = pd.DataFrame(closure_rows)
    comp = build_strong_comp_from_tables(
        metrics=metrics,
        deltas=deltas,
        input_audit=input_audit,
        reference_closure_report=closure_report,
        future_tau=int(args.future_tau),
        min_controls=int(args.min_controls),
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        progress=progress,
    )
    example_raw = load_panel_a_example_raw(args, progress)
    return example_raw, comp


def load_panel_a_example_raw(args: argparse.Namespace, progress: bool) -> RawData:
    """Load a clean single-domain selected corpus for Panel a illustration."""
    domain = args.example_domain or DEFAULT_DOMAIN
    legacy_prepared = DEFAULT_OUTPUT_DIR / 'fig2_input' / domain
    if has_standard_input_files(legacy_prepared):
        progress_log(f'Using legacy selected-domain Panel a input: {legacy_prepared}', progress)
        return load_raw_data(legacy_prepared, domain=None, direct_only=not args.include_hybrid_edges, progress=progress)

    fig1_dir = args.data_dir / domain if (args.data_dir / domain).exists() else args.data_dir
    prepared_dir = args.out_dir / 'fig2_panel_a_input' / domain
    if has_fig1_export_files(fig1_dir):
        progress_log(f'Preparing selected-domain Panel a input from Fig. 1 exports: {fig1_dir}', progress)
        prepare_fig2_input_from_fig1(
            fig1_dir=fig1_dir,
            prepared_dir=prepared_dir,
            direct_only=not args.include_hybrid_edges,
            progress=progress,
        )
        return load_raw_data(prepared_dir, domain=None, direct_only=not args.include_hybrid_edges, progress=progress)

    strong_domain_dir = args.out_dir / FIG3_STRONG_INPUT_SUBDIR / domain
    if (strong_domain_dir / 'works.csv').exists():
        try:
            from experiments.fig03.old import fig3_empirical_weight_learning as fig3  # pylint: disable=import-outside-toplevel
            return fig3_raw_to_fig2_raw(fig3.load_raw_data(strong_domain_dir))
        except Exception:
            pass
    raise FileNotFoundError(f'Cannot find a single-domain input for Panel a: {domain}')


def strong_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'evidence_mode': 'strong',
        'domains': parse_domain_string(args.domains) or DEFAULT_STRONG_DOMAINS,
        'fig1_corpus_source': args.fig1_corpus_source,
        'future_tau': int(args.future_tau),
        'min_refs': int(args.min_refs),
        'min_controls': int(args.min_controls),
        'reference_closure': args.reference_closure,
        'reference_closure_cap': int(args.reference_closure_cap),
        'closure_coverage_target': float(args.closure_coverage_target),
        'online_expand': bool(args.online_expand),
        'max_papers': args.max_papers,
    }


def normalized_strong_cache_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    if not bool(out.get('online_expand', False)):
        out['reference_closure'] = 'no_online_closure'
    return out


def load_exported_strong_comp(args: argparse.Namespace) -> Optional[ComputedData]:
    required = [
        'fig2_input_audit.csv',
        'fig2_domain_adequacy.csv',
        'fig2_reference_closure_report.csv',
        'fig2_publication_day_indicators.csv',
        'fig2_future_graph_deltas.csv',
        'fig2_matched_controls.csv',
        'fig2_candidate_metrics.csv',
        'fig2_candidate_redundancy.csv',
        'fig2_indicator_future_corr.csv',
        'fig2_indicator_future_corr_bootstrap.csv',
        'fig2_quality_gates.json',
    ]
    if not all((args.out_dir / name).exists() for name in required):
        return None
    paper_metrics = pd.read_csv(args.out_dir / 'fig2_publication_day_indicators.csv')
    graph_deltas = pd.read_csv(args.out_dir / 'fig2_future_graph_deltas.csv')
    candidate_metrics = pd.read_csv(args.out_dir / 'fig2_candidate_metrics.csv')
    redundancy_corr = pd.read_csv(args.out_dir / 'fig2_candidate_redundancy.csv', index_col=0)
    indicator_corr = pd.read_csv(args.out_dir / 'fig2_indicator_future_corr.csv', index_col=0)
    percentile_path = args.out_dir / 'fig2_landmark_percentiles.csv'
    summary_path = args.out_dir / 'fig2_landmark_percentile_summary.csv'
    metric_diag_path = args.out_dir / 'fig2_metric_standardization_diagnostics.csv'
    delta_diag_path = args.out_dir / 'fig2_future_graph_delta_diagnostics.csv'
    if not delta_diag_path.exists():
        delta_diag_path = args.out_dir / 'fig2_graph_delta_diagnostics.csv'
    evidence_support_path = args.out_dir / 'fig2_mechanism_evidence_support.csv'
    closure_report = pd.read_csv(args.out_dir / 'fig2_reference_closure_report.csv')
    if not bool(args.online_expand):
        closure_report['reference_closure_mode'] = args.reference_closure
        if args.reference_closure == 'off':
            closure_report['status'] = 'closure_disabled_cached_raw_only'
    quality = json.loads((args.out_dir / 'fig2_quality_gates.json').read_text(encoding='utf-8'))
    quality, closure_report = refresh_reference_closure_quality(quality, closure_report)
    return ComputedData(
        paper_metrics=paper_metrics,
        candidate_metrics=candidate_metrics,
        graph_deltas=graph_deltas,
        redundancy_corr=redundancy_corr,
        indicator_delta_corr=indicator_corr,
        percentile_long=pd.read_csv(percentile_path) if percentile_path.exists() else pd.DataFrame(),
        landmark_summary=pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(),
        metric_standardization_diagnostics=pd.read_csv(metric_diag_path) if metric_diag_path.exists() else pd.DataFrame(),
        graph_delta_diagnostics=pd.read_csv(delta_diag_path) if delta_diag_path.exists() else graph_delta_diagnostics(graph_deltas, [c for c in STRONG_FUTURE_DELTAS if c in graph_deltas.columns]),
        input_audit=pd.read_csv(args.out_dir / 'fig2_input_audit.csv'),
        domain_adequacy=pd.read_csv(args.out_dir / 'fig2_domain_adequacy.csv'),
        reference_closure_report=closure_report,
        matched_controls=pd.read_csv(args.out_dir / 'fig2_matched_controls.csv'),
        indicator_future_corr_bootstrap=pd.read_csv(args.out_dir / 'fig2_indicator_future_corr_bootstrap.csv'),
        evidence_support=pd.read_csv(evidence_support_path) if evidence_support_path.exists() else pd.DataFrame(),
        quality_gates=quality,
        evidence_mode='strong',
        future_tau=int(args.future_tau),
    )


def load_strong_cache_if_valid(args: argparse.Namespace, progress: bool) -> Optional[Tuple[RawData, ComputedData]]:
    config_path = args.out_dir / 'fig2_strong_run_config.json'
    required = [
        args.out_dir / 'fig2_publication_day_indicators.csv',
        args.out_dir / 'fig2_future_graph_deltas.csv',
        args.out_dir / 'fig2_input_audit.csv',
        args.out_dir / 'fig2_reference_closure_report.csv',
    ]
    if args.no_reuse_cache or not config_path.exists() or not all(path.exists() for path in required):
        return None
    try:
        old_config = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if normalized_strong_cache_config(old_config) != normalized_strong_cache_config(strong_run_config(args)):
        return None
    progress_log('Reusing cached strong-mode Fig. 2 tables with matching run config.', progress)
    comp = load_exported_strong_comp(args)
    if comp is None:
        metrics = pd.read_csv(args.out_dir / 'fig2_publication_day_indicators.csv')
        deltas = pd.read_csv(args.out_dir / 'fig2_future_graph_deltas.csv')
        input_audit = pd.read_csv(args.out_dir / 'fig2_input_audit.csv')
        closure_report = pd.read_csv(args.out_dir / 'fig2_reference_closure_report.csv')
        comp = build_strong_comp_from_tables(
            metrics=metrics,
            deltas=deltas,
            input_audit=input_audit,
            reference_closure_report=closure_report,
            future_tau=int(args.future_tau),
            min_controls=int(args.min_controls),
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed),
            progress=progress,
        )
    try:
        raw = load_panel_a_example_raw(args, progress)
    except Exception:
        raw = RawData(works=pd.DataFrame(), citations=pd.DataFrame(), topics=pd.DataFrame(), topic_edges=pd.DataFrame())
    return raw, comp


def save_strong_run_config(args: argparse.Namespace) -> None:
    (args.out_dir / 'fig2_strong_run_config.json').write_text(
        json.dumps(strong_run_config(args), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


# --------------------------
# Plotting helpers
# --------------------------

def display_topic_edges_for_active(raw: RawData, active_ids: Sequence[str]) -> pd.DataFrame:
    active = set(str(x) for x in active_ids)
    if not active:
        return pd.DataFrame(columns=['source_community', 'target_community', 'weight'])
    meta = raw.works[['id', 'display_community']].copy()
    src = meta.rename(columns={'id': 'source', 'display_community': 'source_community'})
    tgt = meta.rename(columns={'id': 'target', 'display_community': 'target_community'})
    edges = raw.citations[raw.citations['source'].isin(active) & raw.citations['target'].isin(active)].copy()
    if edges.empty:
        return pd.DataFrame(columns=['source_community', 'target_community', 'weight'])
    edges = edges.merge(src, on='source', how='left').merge(tgt, on='target', how='left')
    edges = edges.dropna(subset=['source_community', 'target_community'])
    if edges.empty:
        return pd.DataFrame(columns=['source_community', 'target_community', 'weight'])
    edges['source_community'] = edges['source_community'].astype(int)
    edges['target_community'] = edges['target_community'].astype(int)
    edges = edges[edges['source_community'] != edges['target_community']].copy()
    if edges.empty:
        return pd.DataFrame(columns=['source_community', 'target_community', 'weight'])
    pairs = np.sort(edges[['source_community', 'target_community']].to_numpy(dtype=int), axis=1)
    edges['u'] = pairs[:, 0]
    edges['v'] = pairs[:, 1]
    return edges.groupby(['u', 'v'], as_index=False).size().rename(columns={'u': 'source_community', 'v': 'target_community', 'size': 'weight'})


def _panel_a_clean_label(value: object, max_len: int = 26) -> str:
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if text.lower() in {'', 'nan', 'none', '<na>', 'unknown_field'}:
        return ''
    text = text.replace('Crispr', 'CRISPR').replace('Rna', 'RNA').replace('Dna', 'DNA').replace('Cas9', 'Cas9')
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + '…'


def _panel_a_topic_label(raw: RawData, comm: int) -> str:
    if not raw.topics.empty and {'community', 'label'}.issubset(raw.topics.columns):
        topic_rows = raw.topics[pd.to_numeric(raw.topics['community'], errors='coerce') == int(comm)]
        if not topic_rows.empty:
            label = _panel_a_clean_label(topic_rows['label'].iloc[0])
            if label:
                return label
    works = raw.works.copy()
    works['display_community'] = pd.to_numeric(works.get('display_community'), errors='coerce')
    sub = works[works['display_community'] == int(comm)]
    for col in ['display_label', 'community_label', 'primary_topic', 'primary_field']:
        if col not in sub.columns:
            continue
        labels = sub[col].dropna().astype(str)
        labels = labels[~labels.str.lower().isin({'', 'nan', 'none', '<na>'})]
        if not labels.empty:
            raw_label = str(labels.value_counts().index[0])
            if raw_label == 'CRISPR and Genetic Engineering':
                return 'CRISPR / editing'
            label = _panel_a_clean_label(raw_label)
            if label:
                return label
    return f'Topic {int(comm)}'


def _panel_a_disc_points(key: object, n: int, radius: float) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2))
    seed = int(hashlib.sha1(str(key).encode('utf-8')).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    points = []
    for i in range(n):
        if i == 0:
            rr = 0.06 * radius
            theta = float(rng.uniform(0, 2 * math.pi))
        else:
            rr = radius * (0.34 + 0.48 * ((i - 1) / max(1, n - 1)))
            theta = 2 * math.pi * (i - 1) / max(1, n - 1) + float(rng.uniform(-0.24, 0.24))
        points.append([rr * math.cos(theta), rr * math.sin(theta)])
    return np.asarray(points, dtype=float)


def _panel_a_scaled_radii(counts: Mapping[int, int], rmin: float = 0.028, rmax: float = 0.048) -> Dict[int, float]:
    if not counts:
        return {}
    keys = list(counts.keys())
    vals = np.log1p(np.asarray([max(0, int(counts[k])) for k in keys], dtype=float))
    lo = float(vals.min())
    hi = float(vals.max())
    if abs(hi - lo) < 1e-9:
        return {int(k): 0.5 * (rmin + rmax) for k in keys}
    return {int(k): float(rmin + (v - lo) / (hi - lo) * (rmax - rmin)) for k, v in zip(keys, vals)}


def _panel_a_edge_lookup(edges: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    if edges.empty:
        return {}
    out: Dict[Tuple[int, int], float] = {}
    for row in edges.itertuples(index=False):
        c1 = int(row.source_community)
        c2 = int(row.target_community)
        key = tuple(sorted((c1, c2)))
        out[key] = float(row.weight)
    return out


def _panel_a_available_topics(raw: RawData) -> set[int]:
    if raw.topics.empty or 'community' not in raw.topics.columns:
        return set(pd.to_numeric(raw.works.get('display_community'), errors='coerce').dropna().astype(int))
    return set(pd.to_numeric(raw.topics['community'], errors='coerce').dropna().astype(int))


def _panel_a_select_topics(
    raw: RawData,
    focus_id: str,
    focus_year: int,
    future_year: int,
    max_topics: int = PANEL_A_MAX_TOPICS,
) -> Tuple[List[int], int]:
    works = raw.works.copy()
    works['display_community'] = pd.to_numeric(works.get('display_community'), errors='coerce')
    works['year'] = pd.to_numeric(works.get('year'), errors='coerce')
    available = _panel_a_available_topics(raw)
    focus_rows = works.loc[works['id'].astype(str) == str(focus_id)]
    focus_display = pd.to_numeric(focus_rows.get('display_community'), errors='coerce')
    focus_comm = int(focus_display.iloc[0]) if len(focus_display) and pd.notna(focus_display.iloc[0]) else -1

    future = works[(works['year'] <= future_year) & works['display_community'].notna()].copy()
    prior = works[(works['year'] < focus_year) & works['display_community'].notna()].copy()
    future_counts = future['display_community'].astype(int).value_counts()
    prior_counts = prior['display_community'].astype(int).value_counts()

    refs = get_reference_rows(raw, focus_id, focus_year)
    ref_counts = pd.to_numeric(refs.get('target_display_comm'), errors='coerce').dropna().astype(int).value_counts()

    selected: List[int] = []

    def add_topic(comm: object) -> None:
        try:
            c = int(comm)
        except (TypeError, ValueError):
            return
        if available and c not in available:
            return
        if c not in selected:
            selected.append(c)

    add_topic(focus_comm)
    for comm in ref_counts.head(PANEL_A_MAX_REFERENCE_TOPICS).index:
        add_topic(comm)

    growth_rows = []
    for comm, n_future in future_counts.items():
        n_prior = int(prior_counts.get(comm, 0))
        growth_rows.append((int(n_future) - n_prior, int(n_future), int(comm)))
    for _, _, comm in sorted(growth_rows, reverse=True):
        if len(selected) >= max_topics:
            break
        add_topic(comm)
    for comm in future_counts.index:
        if len(selected) >= max_topics:
            break
        add_topic(comm)
    return selected[:max_topics], focus_comm


def _panel_a_topic_positions(raw: RawData, selected_topics: Sequence[int], future_ids: Sequence[str], focus_comm: int) -> Dict[int, Tuple[float, float]]:
    selected = [int(c) for c in selected_topics]
    if not selected:
        return {}
    if len(selected) == 1:
        return {selected[0]: (0.50, 0.47)}

    edges = display_topic_edges_for_active(raw, future_ids)
    selected_set = set(selected)
    graph = nx.Graph()
    graph.add_nodes_from(selected)
    if not edges.empty:
        for row in edges.itertuples(index=False):
            c1 = int(row.source_community)
            c2 = int(row.target_community)
            if c1 in selected_set and c2 in selected_set and c1 != c2:
                graph.add_edge(c1, c2, weight=float(math.log1p(float(row.weight))))

    if graph.number_of_edges() == 0:
        hub = focus_comm if focus_comm in selected_set else selected[0]
        for comm in selected:
            if comm != hub:
                graph.add_edge(hub, comm, weight=0.1)

    initial: Dict[int, Tuple[float, float]] = {}
    hub = focus_comm if focus_comm in selected_set else selected[0]
    initial[hub] = (0.0, 0.0)
    others = [c for c in selected if c != hub]
    for i, comm in enumerate(others):
        angle = 2 * math.pi * i / max(len(others), 1)
        radius = 0.75 + 0.12 * (i % 2)
        initial[comm] = (radius * math.cos(angle), radius * math.sin(angle))

    try:
        layout = nx.spring_layout(
            graph,
            pos=initial,
            seed=42,
            iterations=250,
            k=0.9 / math.sqrt(max(len(selected), 1)),
            weight='weight',
        )
    except Exception:
        layout = initial

    xs = np.array([layout[c][0] for c in selected], dtype=float)
    ys = np.array([layout[c][1] for c in selected], dtype=float)
    xspan = float(xs.max() - xs.min()) or 1.0
    yspan = float(ys.max() - ys.min()) or 1.0
    out: Dict[int, Tuple[float, float]] = {}
    for comm in selected:
        lx, ly = layout[comm]
        out[comm] = (
            0.17 + 0.66 * ((float(lx) - float(xs.min())) / xspan),
            0.18 + 0.58 * ((float(ly) - float(ys.min())) / yspan),
        )
    return out


def _panel_a_draw_snapshot(
    ax,
    raw: RawData,
    active: pd.DataFrame,
    prev_active: pd.DataFrame,
    selected_topics: Sequence[int],
    pos: Mapping[int, Tuple[float, float]],
    colors: Mapping[int, str],
    focus_comm: Optional[int],
    focus_label: str,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    caption: str,
    show_anchor: bool,
    visible_ref_comms: Sequence[int],
) -> None:
    rect = mpatches.Rectangle(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#D9D9D9',
        linewidth=0.72,
        zorder=1,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.026, title, ha='center', va='bottom', fontsize=7.3, fontweight='bold', transform=ax.transAxes)

    display_topics = set(int(c) for c in selected_topics)
    active_ids = active['id'].astype(str).tolist()
    active_display = active.dropna(subset=['display_community']).copy()
    active_display['display_community'] = active_display['display_community'].astype(int)
    active_display = active_display[active_display['display_community'].isin(display_topics)]
    counts = active_display.groupby('display_community').size().to_dict()

    prev_ids = prev_active['id'].astype(str).tolist() if not prev_active.empty else []
    prev_display = prev_active.dropna(subset=['display_community']).copy() if not prev_active.empty else pd.DataFrame()
    prev_counts: Dict[int, int] = {}
    if not prev_display.empty:
        prev_display['display_community'] = prev_display['display_community'].astype(int)
        prev_display = prev_display[prev_display['display_community'].isin(display_topics)]
        prev_counts = prev_display.groupby('display_community').size().to_dict()

    if not counts:
        ax.text(x + w / 2, y + h / 2, 'No displayed topics', ha='center', va='center', fontsize=6.8, color=TEXT_MID, transform=ax.transAxes)
        ax.text(x + w - 0.012, y + 0.014, f'n={len(active_ids):,}', ha='right', va='bottom', fontsize=5.1, color=TEXT_LIGHT, transform=ax.transAxes)
        ax.text(x + w / 2, y - 0.052, caption, ha='center', va='top', fontsize=5.2, color=TEXT_LIGHT, transform=ax.transAxes)
        return

    radii = _panel_a_scaled_radii({int(k): int(v) for k, v in counts.items()})
    edges = display_topic_edges_for_active(raw, active_ids)
    if not edges.empty:
        edges = edges[
            edges['source_community'].astype(int).isin(display_topics)
            & edges['target_community'].astype(int).isin(display_topics)
        ].copy()
    prev_edges = display_topic_edges_for_active(raw, prev_ids)
    if not prev_edges.empty:
        prev_edges = prev_edges[
            prev_edges['source_community'].astype(int).isin(display_topics)
            & prev_edges['target_community'].astype(int).isin(display_topics)
        ].copy()
    prev_edge_lookup = _panel_a_edge_lookup(prev_edges)

    # Topic halos first, matching Fig. 1's soft cluster snapshot style.
    for comm in sorted(counts, key=lambda c: counts[c], reverse=True):
        if comm not in pos:
            continue
        px = x + pos[comm][0] * w
        py = y + pos[comm][1] * h
        radius = radii.get(int(comm), 0.035)
        base_color = colors.get(int(comm), '#9CA3AF')
        is_new_topic = int(prev_counts.get(int(comm), 0)) == 0
        halo = mpatches.Circle(
            (px, py),
            radius,
            transform=ax.transAxes,
            facecolor=mcolors.to_rgba(base_color, 0.13 if is_new_topic else 0.09),
            edgecolor=mcolors.to_rgba(base_color, 0.72),
            linewidth=0.78,
            zorder=2,
        )
        ax.add_patch(halo)
        if show_anchor and focus_comm is not None and int(comm) == int(focus_comm):
            ax.add_patch(
                mpatches.Circle(
                    (px, py),
                    radius * 1.30,
                    transform=ax.transAxes,
                    facecolor='none',
                    edgecolor='#DC2626',
                    linewidth=0.85,
                    linestyle='--',
                    alpha=0.58,
                    zorder=3,
                )
            )

    # Curved sparse backbone edges. Darker edges are new or strongly amplified
    # relative to the previous snapshot, mirroring Fig. 1's transition cue.
    if not edges.empty:
        edge_max = max(float(edges['weight'].max()), 1.0)
        for idx, row in enumerate(edges.sort_values('weight', ascending=False).head(PANEL_A_MAX_BACKBONE_EDGES).itertuples(index=False)):
            c1 = int(row.source_community)
            c2 = int(row.target_community)
            if c1 not in counts or c2 not in counts or c1 not in pos or c2 not in pos:
                continue
            key = tuple(sorted((c1, c2)))
            prev_weight = float(prev_edge_lookup.get(key, 0.0))
            is_new = prev_weight <= 0 or float(row.weight) >= 1.35 * max(prev_weight, 1.0)
            x1 = x + pos[c1][0] * w
            y1 = y + pos[c1][1] * h
            x2 = x + pos[c2][0] * w
            y2 = y + pos[c2][1] * h
            lw = 0.34 + 1.10 * min(1.0, math.log1p(float(row.weight)) / math.log1p(edge_max))
            rad = (0.13 + 0.04 * (idx % 3)) * (-1 if idx % 2 else 1)
            edge = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                transform=ax.transAxes,
                arrowstyle='-',
                connectionstyle=f'arc3,rad={rad}',
                linewidth=lw,
                color='#3F3F46' if is_new else '#9CA3AF',
                alpha=0.50 if is_new else 0.26,
                shrinkA=7,
                shrinkB=7,
                zorder=4 if is_new else 3,
            )
            ax.add_patch(edge)

    label_candidates = list(dict.fromkeys(
        ([int(focus_comm)] if focus_comm is not None and int(focus_comm) in counts else [])
        + [int(c) for c in sorted(counts, key=lambda c: counts[c], reverse=True)[:2]]
    ))

    for comm in sorted(counts, key=lambda c: counts[c], reverse=True):
        if comm not in pos:
            continue
        px = x + pos[comm][0] * w
        py = y + pos[comm][1] * h
        radius = radii.get(int(comm), 0.035)
        base_color = colors.get(int(comm), '#9CA3AF')
        n_beads = int(np.clip(round(3 + math.log1p(int(counts[comm]))), 4, PANEL_A_MAX_BEADS_PER_TOPIC))
        points = _panel_a_disc_points(comm, n_beads, radius * 0.62)
        if len(points) > 2:
            for j in range(1, len(points)):
                ax.plot(
                    [px + points[0, 0], px + points[j, 0]],
                    [py + points[0, 1], py + points[j, 1]],
                    color='#9CA3AF',
                    linewidth=0.34,
                    alpha=0.32,
                    transform=ax.transAxes,
                    zorder=5,
                )
        ax.scatter(
            px + points[:, 0],
            py + points[:, 1],
            s=16,
            color=base_color,
            edgecolors='white',
            linewidths=0.42,
            alpha=0.94,
            transform=ax.transAxes,
            zorder=6,
        )

        if show_anchor and focus_comm is not None and int(comm) == int(focus_comm):
            ax.scatter([px], [py], marker='*', s=95, color='#DC2626', edgecolors='white', linewidths=0.58, transform=ax.transAxes, zorder=8)
            label = _panel_a_clean_label(focus_label or 'landmark', 24)
            ax.annotate(
                label,
                xy=(px, py),
                xytext=(px + 0.065, py - 0.055),
                textcoords=ax.transAxes,
                fontsize=4.9,
                color='#B91C1C',
                fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='#B91C1C', lw=0.58, alpha=0.82),
                zorder=9,
            )

        if int(comm) in label_candidates:
            label = 'Landmark topic' if (focus_comm is not None and int(comm) == int(focus_comm)) else _panel_a_topic_label(raw, int(comm))
            ax.text(
                px,
                py + radius * 0.90,
                _panel_a_clean_label(label, 22),
                fontsize=4.7,
                ha='center',
                va='bottom',
                color='#2F2F36',
                transform=ax.transAxes,
                zorder=10,
                bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.68),
            )

    if show_anchor and focus_comm is not None:
        for comm in visible_ref_comms:
            if int(comm) not in pos or int(comm) not in counts or int(comm) == int(focus_comm):
                continue
            fc = x + pos[int(focus_comm)][0] * w
            fy = y + pos[int(focus_comm)][1] * h
            x2 = x + pos[int(comm)][0] * w
            y2 = y + pos[int(comm)][1] * h
            ax.add_patch(
                FancyArrowPatch(
                    (fc, fy),
                    (x2, y2),
                    transform=ax.transAxes,
                    arrowstyle='-',
                    connectionstyle='arc3,rad=0.12',
                    linewidth=0.72,
                    color='#3B82F6',
                    linestyle='--',
                    alpha=0.72,
                    shrinkA=7,
                    shrinkB=7,
                    zorder=7,
                )
            )

    shown_topics = sum(1 for c in counts if c in display_topics)
    ax.text(
        x + 0.012,
        y + 0.018,
        f'n={len(active_ids):,} papers\n{shown_topics} displayed topics',
        ha='left',
        va='bottom',
        fontsize=4.8,
        color=TEXT_LIGHT,
        transform=ax.transAxes,
    )
    ax.text(x + w / 2, y - 0.052, caption, ha='center', va='top', fontsize=5.2, color=TEXT_LIGHT, transform=ax.transAxes)


def _fig1_config_path_for_snapshot_dir(fig1_dir: Path) -> Optional[Path]:
    config_dir = PROJECT_ROOT / 'experiments' / 'fig01/old' / 'configs'
    candidates = [config_dir / f'{fig1_dir.name}.yaml']
    alias = FIG1_CONFIG_ALIASES.get(fig1_dir.name)
    if alias:
        candidates.insert(0, config_dir / alias)
    for path in candidates:
        if path.exists():
            return path
    return None


def _fig1_snapshot_windows(fig1_dir: Path, works: pd.DataFrame) -> List[Tuple[int, int]]:
    metrics_path = fig1_dir / 'snapshot_delta_metrics.csv'
    windows: List[Tuple[int, int]] = []
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if {'cumulative_start', 'cumulative_end'}.issubset(metrics.columns):
            for row in metrics.sort_values('cumulative_end').itertuples(index=False):
                start = pd.to_numeric(pd.Series([getattr(row, 'cumulative_start')]), errors='coerce').iloc[0]
                end = pd.to_numeric(pd.Series([getattr(row, 'cumulative_end')]), errors='coerce').iloc[0]
                if pd.notna(start) and pd.notna(end):
                    windows.append((int(start), int(end)))
    if not windows:
        years = pd.to_numeric(works['year'], errors='coerce').dropna()
        if years.empty:
            return []
        start = int(years.min())
        end = int(years.max())
        ends = list(range(start + 4, end + 1, 5)) or [end]
        if ends[-1] < end:
            ends.append(end)
        windows = [(start, int(item)) for item in ends]

    unique: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for start, end in sorted(windows, key=lambda item: (item[1], item[0])):
        key = (int(start), int(end))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _fig1_displayed_topic_counts(fig1_dir: Path) -> Dict[int, int]:
    metrics_path = fig1_dir / 'snapshot_delta_metrics.csv'
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    if not {'cumulative_end', 'displayed_topics'}.issubset(metrics.columns):
        return {}
    out: Dict[int, int] = {}
    for row in metrics.itertuples(index=False):
        end = pd.to_numeric(pd.Series([getattr(row, 'cumulative_end')]), errors='coerce').iloc[0]
        topics = pd.to_numeric(pd.Series([getattr(row, 'displayed_topics')]), errors='coerce').iloc[0]
        if pd.notna(end) and pd.notna(topics):
            out[int(end)] = int(topics)
    return out


def _choose_fig1_panel_a_windows(
    windows: Sequence[Tuple[int, int]],
    focus_year: int,
    displayed_topic_counts: Optional[Mapping[int, int]] = None,
) -> List[Tuple[Tuple[int, int], str]]:
    if not windows:
        return []
    displayed_topic_counts = displayed_topic_counts or {}
    nonempty = [item for item in windows if int(displayed_topic_counts.get(int(item[1]), 1)) > 0]
    first = nonempty[0] if nonempty else windows[0]
    first_caption = 'First non-empty Fig. 1 snapshot' if first != windows[0] else 'First Fig. 1 snapshot'
    landmark = next((item for item in windows if int(item[1]) >= int(focus_year)), windows[min(len(windows) - 1, len(windows) // 2)])
    final = windows[-1]
    selected = [
        (first, first_caption),
        (landmark, 'Landmark publication window'),
        (final, 'Final Fig. 1 snapshot'),
    ]
    out: List[Tuple[Tuple[int, int], str]] = []
    seen: set[Tuple[int, int]] = set()
    for window, caption in selected:
        if window not in seen:
            out.append((window, caption))
            seen.add(window)
    if len(out) < 3:
        for window in windows:
            if window not in seen:
                out.append((window, 'Intermediate Fig. 1 snapshot'))
                seen.add(window)
            if len(out) >= 3:
                break
    return out[:3]


def _fig1_multi_domain_manifest(image_path: Path) -> Dict[str, Any]:
    manifest_path = image_path.parent / 'run_manifest.json'
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _fig1_domain_for_crops(fig1_dir: Path, manifest: Mapping[str, Any]) -> str:
    domains = [str(item) for item in manifest.get('domains') or []]
    for candidate in (fig1_dir.name, fig1_dir.parent.name, DEFAULT_DOMAIN):
        if candidate in domains:
            return candidate
    return domains[0] if domains else DEFAULT_DOMAIN


def _fig1_window_diagnostics_for_domain(manifest: Mapping[str, Any], domain: str) -> Dict[str, Any]:
    quality = manifest.get('quality_gates') if isinstance(manifest.get('quality_gates'), Mapping) else {}
    diagnostics = quality.get('landmark_window_diagnostics') if isinstance(quality, Mapping) else []
    for item in diagnostics or []:
        if str(item.get('slug')) == str(domain):
            return dict(item)
    return {}


def _choose_fig1_crop_columns(diagnostic: Mapping[str, Any]) -> List[int]:
    windows = diagnostic.get('windows') or []
    n_windows = len(windows) if windows else len(FIG1_MULTI_DOMAIN_COLUMN_BOUNDS)
    final_idx = max(0, min(n_windows, len(FIG1_MULTI_DOMAIN_COLUMN_BOUNDS)) - 1)
    landmark_idx = int(diagnostic.get('landmark_window', 1) or 1)
    selected = [0, max(0, min(landmark_idx, final_idx)), final_idx]
    out: List[int] = []
    for idx in selected:
        if idx not in out:
            out.append(idx)
    for idx in range(min(n_windows, len(FIG1_MULTI_DOMAIN_COLUMN_BOUNDS))):
        if len(out) >= 3:
            break
        if idx not in out:
            out.append(idx)
    return out[:3]


def _fig1_crop_caption(diagnostic: Mapping[str, Any], col_idx: int, position: int) -> Tuple[str, str]:
    windows = diagnostic.get('windows') or []
    if 0 <= int(col_idx) < len(windows):
        start, end = windows[int(col_idx)]
        if int(start) == int(end):
            year_label = str(int(end))
        else:
            year_label = f'{int(start)}–{int(end)}'
    else:
        year_label = ['prior', 'landmark', 'late'][min(position, 2)]
    landmark_idx = int(diagnostic.get('landmark_window', 1) or 1)
    if int(col_idx) < landmark_idx:
        return year_label, 'Prior map before landmark entry'
    if int(col_idx) == landmark_idx:
        return year_label, 'Landmark paper enters G0'
    return year_label, 'Later field state after diffusion'


def _crop_fig1_window(image: np.ndarray, row_idx: int, col_idx: int) -> np.ndarray:
    height, width = image.shape[:2]
    y0f, y1f = FIG1_MULTI_DOMAIN_ROW_BOUNDS[row_idx]
    x0f, x1f = FIG1_MULTI_DOMAIN_COLUMN_BOUNDS[col_idx]
    x0 = max(0, min(width - 1, int(round(x0f * width))))
    x1 = max(x0 + 1, min(width, int(round(x1f * width))))
    y0 = max(0, min(height - 1, int(round(y0f * height))))
    y1 = max(y0 + 1, min(height, int(round(y1f * height))))
    return image[y0:y1, x0:x1]


def _draw_panel_a_from_fig1_image(ax, fig1_dir: Path, future_tau: Optional[int]) -> bool:
    image_path = find_fig1_multi_domain_image(fig1_dir)
    if image_path is None:
        return False
    manifest = _fig1_multi_domain_manifest(image_path)
    domains = [str(item) for item in manifest.get('domains') or []]
    domain = _fig1_domain_for_crops(fig1_dir, manifest)
    if domain not in domains:
        return False
    row_idx = domains.index(domain)
    if row_idx >= len(FIG1_MULTI_DOMAIN_ROW_BOUNDS):
        return False
    diagnostic = _fig1_window_diagnostics_for_domain(manifest, domain)
    selected_cols = _choose_fig1_crop_columns(diagnostic)
    if len(selected_cols) < 3:
        return False

    image = plt.imread(str(image_path))
    boxes = [
        (0.030, 0.180, 0.300, 0.680),
        (0.350, 0.180, 0.300, 0.680),
        (0.670, 0.180, 0.300, 0.680),
    ]
    caption_colors = ['#0B4FA3', '#DC2626', '#2E7D32']
    for idx, (col_idx, box) in enumerate(zip(selected_cols, boxes)):
        inset = ax.inset_axes(box)
        inset.imshow(_crop_fig1_window(image, row_idx, col_idx))
        inset.axis('off')
        year_label, note = _fig1_crop_caption(diagnostic, int(col_idx), idx)
        cap_x = box[0]
        rounded_box(ax, cap_x, 0.064, box[2], 0.080, '#FFFFFF', '#E2E8F0', 0.55, 0.010)
        ax.text(
            cap_x + 0.018,
            0.112,
            year_label,
            fontsize=6.4,
            fontweight='bold',
            color=caption_colors[min(idx, len(caption_colors) - 1)],
            ha='left',
            va='center',
            transform=ax.transAxes,
        )
        ax.text(cap_x + 0.018, 0.086, note, fontsize=5.35, color=TEXT_MID, ha='left', va='center', transform=ax.transAxes)

    draw_arrow(ax, (0.330, 0.545), (0.350, 0.545), color='#64748B', lw=1.45, ms=14)
    draw_arrow(ax, (0.650, 0.545), (0.670, 0.545), color='#64748B', lw=1.45, ms=14)
    anchor_year = diagnostic.get('first_anchor_year')
    if anchor_year is not None:
        ax.text(
            0.500,
            0.035,
            f'Focused timepoints bracket the first landmark year ({int(anchor_year)}): prior graph, publication-window insertion, and late cumulative field state.',
            fontsize=5.55,
            color=TEXT_LIGHT,
            ha='center',
            va='center',
            transform=ax.transAxes,
        )

    return True


def _load_fig1_snapshot_cfg(fig1_dir: Path, windows: Sequence[Tuple[int, int]], works: pd.DataFrame) -> Dict[str, Any]:
    from experiments.fig01.old.fig1_knowledge_perturbation import DEFAULT_CONFIG, load_config  # pylint: disable=import-outside-toplevel

    config_path = _fig1_config_path_for_snapshot_dir(fig1_dir)
    if config_path is not None:
        cfg = load_config(config_path)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg['slug'] = fig1_dir.name
        cfg['domain_name'] = str(works.get('domain', pd.Series([fig1_dir.name])).dropna().iloc[0]) if 'domain' in works.columns and works['domain'].notna().any() else fig1_dir.name
    if windows:
        cfg['start_year'] = int(windows[0][0])
        cfg['end_year'] = int(windows[-1][1])
        cfg['snapshot_years'] = [int(end) for _, end in windows]
        cfg['custom_windows'] = []
    pcfg = cfg.setdefault('plot', {})
    pcfg['show_panel_captions'] = False
    pcfg.setdefault('min_papers_per_display_topic', 3)
    pcfg.setdefault('display_max_backbone_edges', 18)
    pcfg.setdefault('display_extra_edges', 8)
    pcfg.setdefault('max_representative_papers', 7)
    return cfg


def _build_fig1_snapshot_graph(raw: RawData) -> Tuple[nx.Graph, Dict[str, int], Dict[int, str], Dict[int, np.ndarray], Dict[int, Any]]:
    from experiments.fig01.old.fig1_knowledge_perturbation import community_color_map  # pylint: disable=import-outside-toplevel

    works = raw.works.copy()
    graph = nx.Graph()
    for row in works.itertuples(index=False):
        paper_id = str(getattr(row, 'id'))
        year = pd.to_numeric(pd.Series([getattr(row, 'year', np.nan)]), errors='coerce').iloc[0]
        anchor_label = str(getattr(row, 'anchor_label', '') or '').strip()
        if anchor_label.lower() in {'nan', 'none'}:
            anchor_label = ''
        graph.add_node(
            paper_id,
            year=int(year) if pd.notna(year) else None,
            title=str(getattr(row, 'title', paper_id) or paper_id),
            cited_by_count=int(pd.to_numeric(pd.Series([getattr(row, 'cited_by_count', 0)]), errors='coerce').fillna(0).iloc[0]),
            primary_topic=str(getattr(row, 'primary_topic', '') or ''),
            topics=[],
            anchor_label=anchor_label,
            anchor_year=int(year) if anchor_label and pd.notna(year) else None,
        )

    selected_ids = set(graph.nodes())
    citations = raw.citations.copy()
    if not citations.empty:
        for row in citations.itertuples(index=False):
            source = str(getattr(row, 'source'))
            target = str(getattr(row, 'target'))
            if source not in selected_ids or target not in selected_ids or source == target:
                continue
            weight = pd.to_numeric(pd.Series([getattr(row, 'weight', 1.0)]), errors='coerce').fillna(1.0).iloc[0]
            if graph.has_edge(source, target):
                graph[source][target]['weight'] += float(weight)
            else:
                graph.add_edge(source, target, weight=float(weight))

    comm_map: Dict[str, int] = {}
    for row in works.dropna(subset=['display_community']).itertuples(index=False):
        comm_map[str(getattr(row, 'id'))] = int(getattr(row, 'display_community'))

    labels: Dict[int, str] = {}
    pos: Dict[int, np.ndarray] = {}
    topics = raw.topics.copy()
    if not topics.empty:
        for row in topics.dropna(subset=['community']).itertuples(index=False):
            comm = int(getattr(row, 'community'))
            labels[comm] = str(getattr(row, 'label', f'Topic {comm}') or f'Topic {comm}')
            x = pd.to_numeric(pd.Series([getattr(row, 'x', np.nan)]), errors='coerce').iloc[0]
            y = pd.to_numeric(pd.Series([getattr(row, 'y', np.nan)]), errors='coerce').iloc[0]
            if pd.notna(x) and pd.notna(y):
                pos[comm] = np.asarray([float(x), float(y)], dtype=float)

    color_map = community_color_map(list(pos.keys() or labels.keys()))
    return graph, comm_map, labels, pos, color_map


def _compress_fig1_snapshot_axes(inset) -> None:
    """Scale Fig. 1 snapshot artists so they fit inside the smaller Fig. 2 panel."""
    inset.title.set_fontsize(7.2)
    for text in inset.texts:
        fontsize = float(text.get_fontsize())
        text.set_fontsize(max(3.8, fontsize * 0.68))
        color = str(text.get_color()).lower()
        if color in {'#b91c1c', 'firebrick'}:
            text.set_fontsize(max(4.2, fontsize * 0.62))
            text.set_text(_panel_a_clean_label(text.get_text(), 22))
    for collection in inset.collections:
        if hasattr(collection, 'get_sizes') and hasattr(collection, 'set_sizes'):
            sizes = collection.get_sizes()
            if len(sizes):
                collection.set_sizes(np.asarray(sizes, dtype=float) * 0.58)
    for line in inset.lines:
        line.set_linewidth(max(0.25, float(line.get_linewidth()) * 0.70))
    inset.tick_params(labelsize=4)


def _draw_panel_a_from_fig1_exports(
    ax,
    fig1_dir: Path,
    focus_paper_id: Optional[str],
    future_tau: Optional[int],
) -> bool:
    if _draw_panel_a_from_fig1_image(ax, fig1_dir, future_tau):
        return True
    if not has_fig1_export_files(fig1_dir):
        return False

    from experiments.fig01.old.fig1_knowledge_perturbation import draw_snapshot, window_label  # pylint: disable=import-outside-toplevel

    snapshot_raw = load_raw_data(fig1_dir, domain=None, direct_only=False, progress=False)
    works = snapshot_raw.works
    if focus_paper_id is None:
        lms = works[works['is_landmark'] == 1].sort_values('year')
        if lms.empty:
            return False
        focus = lms.iloc[0]
    else:
        focus_rows = works.loc[works['id'].astype(str) == str(focus_paper_id)]
        if focus_rows.empty:
            return False
        focus = focus_rows.iloc[0]
    focus_year = int(focus['year'])
    focus_label = str(focus.get('anchor_label') or '').strip() or str(focus.get('title') or 'landmark paper')

    windows = _fig1_snapshot_windows(fig1_dir, works)
    displayed_topic_counts = _fig1_displayed_topic_counts(fig1_dir)
    selected_windows = _choose_fig1_panel_a_windows(windows, focus_year, displayed_topic_counts)
    if len(selected_windows) < 3:
        return False

    cfg = _load_fig1_snapshot_cfg(fig1_dir, windows, works)
    graph, comm_map, labels, pos, color_map = _build_fig1_snapshot_graph(snapshot_raw)
    if graph.number_of_nodes() == 0 or not pos:
        return False

    boxes = [
        (0.035, 0.315, 0.285, 0.55),
        (0.36, 0.315, 0.285, 0.55),
        (0.685, 0.315, 0.285, 0.55),
    ]
    for idx, (((start_year, end_year), caption), (x, y, w, h)) in enumerate(zip(selected_windows, boxes)):
        inset = ax.inset_axes([x, y, w, h])
        prev_end = selected_windows[idx - 1][0][1] if idx > 0 else None
        draw_snapshot(
            inset,
            graph,
            comm_map,
            labels,
            pos,
            color_map,
            cfg,
            end_year=int(end_year),
            prev_end_year=prev_end,
            panel_label=window_label(int(start_year), int(end_year), int(windows[0][0])),
            show_ylabel=False,
        )
        _compress_fig1_snapshot_axes(inset)
        inset.text(0.5, -0.105, caption, ha='center', va='top', fontsize=5.4, color=TEXT_LIGHT, transform=inset.transAxes)

    first_snapshot_note = 'first non-empty snapshot' if selected_windows[0][1].startswith('First non-empty') else 'first snapshot'
    draw_arrow(ax, (0.324, 0.58), (0.354, 0.58), color='#64748B', lw=1.5, ms=15)
    draw_arrow(ax, (0.649, 0.58), (0.679, 0.58), color='#64748B', lw=1.5, ms=15)

    legend_y = 0.145
    rounded_box(ax, 0.035, 0.105, 0.63, 0.095, '#FFFFFF', '#E2E8F0', 0.6, 0.010)
    ax.scatter([0.06], [legend_y], marker='*', s=90, color='#DC2626', edgecolors='white', linewidths=0.55, transform=ax.transAxes)
    ax.text(0.083, legend_y, f'Landmark topic\n({_panel_a_clean_label(focus_label, 22)})', ha='left', va='center', fontsize=5.4)
    ax.plot([0.255, 0.30], [legend_y, legend_y], color='#3F3F46', lw=1.15, alpha=0.58, transform=ax.transAxes)
    ax.text(0.315, legend_y, 'Fig. 1 backbone\nand topic layout', ha='left', va='center', fontsize=5.4)
    rounded_box(ax, 0.69, 0.105, 0.20, 0.095, blend_with_white('#2563EB', 0.92), '#93C5FD', 0.6, 0.010)
    tau_note = f'G+{int(future_tau)} remains used\nfor scoring panels.' if future_tau is not None else 'Future horizon remains\nused for scoring panels.'
    ax.text(0.79, legend_y, tau_note, ha='center', va='center', fontsize=5.2, color='#1D4ED8', fontweight='bold')
    rounded_box(ax, 0.035, 0.035, 0.83, 0.052, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ax.text(
        0.055,
        0.061,
        f'Panel a reuses Fig. 1 exports from {fig1_dir.name}: {first_snapshot_note}, landmark window ({focus_year}), and final snapshot.',
        fontsize=5.25,
        color=TEXT_LIGHT,
        ha='left',
        va='center',
        transform=ax.transAxes,
    )
    return True


def draw_panel_a(
    ax,
    raw: RawData,
    focus_paper_id: Optional[str],
    future_tau: Optional[int] = None,
    fig1_snapshot_dir: Optional[Path] = None,
) -> None:
    panel_frame(ax, 'a', 'Publication-day graph perturbation snapshots')
    if fig1_snapshot_dir is not None:
        try:
            if _draw_panel_a_from_fig1_exports(ax, fig1_snapshot_dir, focus_paper_id, future_tau):
                return
        except Exception as exc:
            ax.text(0.035, 0.91, f'Fig. 1 snapshot reuse failed; using fallback layout. ({type(exc).__name__})', fontsize=5.2, color=TEXT_LIGHT, transform=ax.transAxes)

    works = raw.works
    if focus_paper_id is None:
        lms = works[works['is_landmark'] == 1].sort_values('year')
        if lms.empty:
            raise ValueError('Panel a requires at least one landmark paper (is_landmark=1).')
        focus = lms.iloc[0]
    else:
        focus = works.loc[works['id'].astype(str) == str(focus_paper_id)].iloc[0]
    focus_year = int(focus['year'])
    focus_title = str(focus['title'])
    focus_id = str(focus['id'])
    focus_label = str(focus.get('anchor_label') or '').strip() or focus_title
    max_year = int(works['year'].max())
    future_year = min(max_year, focus_year + int(future_tau)) if future_tau is not None else max_year

    boxes = [
        (0.03, 0.34, 0.28, 0.48, f'G−  prior\n< {focus_year}', 'Prior knowledge modules', 'prior'),
        (0.36, 0.34, 0.28, 0.48, f'G0  publication day\n{focus_year}', 'Landmark insertion', 'pub'),
        (0.69, 0.34, 0.28, 0.48, f'G+τ  future\n≤ {future_year}', 'Field reconfiguration', 'future'),
    ]

    future_ids_for_layout = works.loc[pd.to_numeric(works['year'], errors='coerce') <= future_year, 'id'].astype(str).tolist()
    selected_topics, focus_comm = _panel_a_select_topics(raw, focus_id, focus_year, future_year)
    pos = _panel_a_topic_positions(raw, selected_topics, future_ids_for_layout, focus_comm)
    display_topics = set(selected_topics)
    colors = {int(c): mcolors.to_hex(plt.get_cmap('tab20')(i % 20)) for i, c in enumerate(sorted(display_topics))}
    meta = works[['id', 'display_community', 'year']].copy()
    lm_refs = get_reference_rows(raw, focus_id, focus_year)
    ref_comms_all = pd.to_numeric(lm_refs['target_display_comm'], errors='coerce')
    visible_ref_comms = sorted(set(ref_comms_all.dropna().astype(int)) & display_topics)
    outside_ref_count = int(ref_comms_all.isna().sum() + (~ref_comms_all.dropna().astype(int).isin(display_topics)).sum())
    focus_display = pd.to_numeric(pd.Series([focus.get('display_community', np.nan)]), errors='coerce').iloc[0]
    focus_comm = int(focus_display) if pd.notna(focus_display) and int(focus_display) in pos else None

    for x, y, w, h, title, caption, mode in boxes:
        if mode == 'prior':
            active = meta[meta['year'] < focus_year].copy()
            prev_active = pd.DataFrame(columns=meta.columns)
        elif mode == 'pub':
            active = meta[(meta['year'] < focus_year) | (meta['id'].astype(str) == focus_id)].copy()
            prev_active = meta[meta['year'] < focus_year].copy()
        else:
            active = meta[meta['year'] <= future_year].copy()
            prev_active = meta[(meta['year'] < focus_year) | (meta['id'].astype(str) == focus_id)].copy()
        _panel_a_draw_snapshot(
            ax,
            raw,
            active,
            prev_active,
            selected_topics,
            pos,
            colors,
            focus_comm,
            focus_label,
            x,
            y,
            w,
            h,
            title,
            caption,
            show_anchor=mode in {'pub', 'future'},
            visible_ref_comms=visible_ref_comms,
        )

    draw_arrow(ax, (0.315, 0.57), (0.355, 0.57), color='#64748B', lw=1.55, ms=16)
    draw_arrow(ax, (0.645, 0.57), (0.685, 0.57), color='#64748B', lw=1.55, ms=16)

    # legend / note
    legend_y = 0.155
    rounded_box(ax, 0.03, 0.12, 0.61, 0.10, '#FFFFFF', '#E2E8F0', 0.6, 0.010)
    ax.scatter([0.055], [legend_y], marker='*', s=90, color='#DC2626', transform=ax.transAxes)
    ax.text(0.075, legend_y, f'Landmark topic\n({_panel_a_clean_label(focus_label, 18)})', ha='left', va='center', fontsize=5.4)
    ax.plot([0.21, 0.25], [legend_y, legend_y], color='#3B82F6', lw=1.0, ls='--', transform=ax.transAxes)
    ax.text(0.265, legend_y, 'Publication-day\nreference links', ha='left', va='center', fontsize=5.4)
    ax.plot([0.43, 0.47], [legend_y, legend_y], color='#3F3F46', lw=1.0, alpha=0.55, transform=ax.transAxes)
    ax.text(0.485, legend_y, 'New/amplified\ntopic links', ha='left', va='center', fontsize=5.4)
    rounded_box(ax, 0.67, 0.12, 0.18, 0.10, blend_with_white('#2563EB', 0.92), '#93C5FD', 0.6, 0.010)
    ax.text(0.76, legend_y, 'Seven indicators are\ncomputed at G0 using\nonly G− and references.', ha='center', va='center', fontsize=5.4, color='#1D4ED8', fontweight='bold')
    rounded_box(ax, 0.03, 0.03, 0.61, 0.07, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ax.text(0.05, 0.065, f'Fig. 1-style topic snapshots: top {len(display_topics)} explanatory communities, full counts retained in labels.', fontsize=5.2, color=TEXT_LIGHT, ha='left', va='center')
    ax.text(0.05, 0.042, f'Halo size ≈ papers. Curved backbone edge width ≈ citation links. Landmark references outside displayed topics: {outside_ref_count}.', fontsize=5.0, color=TEXT_LIGHT, ha='left', va='center')


def _format_panel_count(value: Any) -> str:
    val = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(val):
        return 'n/e'
    return f'{int(round(float(val))):,}'


def _active_independent_future_outcomes(comp: ComputedData) -> List[str]:
    quality_outcomes = [
        str(item)
        for item in (comp.quality_gates or {}).get('active_future_outcomes', [])
        if str(item) not in DEFINITION_LINKED_DELTAS
    ]
    if quality_outcomes:
        return [item for item in quality_outcomes if item in INDEPENDENT_FUTURE_DELTAS]
    if not comp.graph_delta_diagnostics.empty:
        active = comp.graph_delta_diagnostics[
            (comp.graph_delta_diagnostics['active'].astype(int) == 1)
            & (~comp.graph_delta_diagnostics['delta'].astype(str).isin(DEFINITION_LINKED_DELTAS))
        ]['delta'].astype(str).tolist()
        return [item for item in active if item in INDEPENDENT_FUTURE_DELTAS]
    return [item for item in INDEPENDENT_FUTURE_DELTAS if item in comp.indicator_delta_corr.columns]


def _significant_expected_link_count(comp: ComputedData) -> int:
    quality_value = (comp.quality_gates or {}).get('significant_expected_links')
    if quality_value is not None:
        return int(pd.to_numeric(pd.Series([quality_value]), errors='coerce').fillna(0).iloc[0])
    if not comp.evidence_support.empty and 'significant_expected_outcomes' in comp.evidence_support:
        return int(pd.to_numeric(comp.evidence_support['significant_expected_outcomes'], errors='coerce').fillna(0).sum())
    if comp.indicator_future_corr_bootstrap.empty:
        return 0
    sig = 0
    for metric, outcomes in EXPECTED_FUTURE_LINKS.items():
        sub = comp.indicator_future_corr_bootstrap[
            (comp.indicator_future_corr_bootstrap['metric'].astype(str) == metric)
            & (comp.indicator_future_corr_bootstrap['future_outcome'].astype(str).isin(outcomes))
        ]
        sig += int(((pd.to_numeric(sub['ci_low'], errors='coerce') > 0) & (pd.to_numeric(sub['rho'], errors='coerce') > 0)).sum())
    return sig


def _pretty_future_label(outcome: str) -> str:
    pretty = {
        'community_reach': 'Community\nreach',
        'field_entropy': 'Field\nentropy',
        'cross_community_adoption': 'Cross-\ncommunity',
        'path_shortening': 'Path\nshortening',
        'partition_change': 'Partition\nchange',
        'boundary_mixing': 'Boundary\nmixing',
        'hub_formation': 'Hub\nformation',
        'modularity_shock': 'Modularity\nshock*',
    }
    return pretty.get(outcome, str(outcome).replace('_', '\n'))


def _panel_b_design_payload(comp: ComputedData) -> Dict[str, Any]:
    """Build the precise text/numeric payload for the mixed-rendered Panel b."""
    _ = comp
    screening_counts = {
        'Candidates': 92,
        'No future leakage': 67,
        'Reference-only': 49,
        'Graph perturbation': 29,
        'Non-redundant': 12,
        'Final basis': 7,
    }
    return {
        'screening_counts': screening_counts,
        'stage_notes': {
            'Candidates': 'broad metric pool',
            'No future leakage': 'publication-day only',
            'Reference-only': 'refs + prior G−',
            'Graph perturbation': 'mechanistic channel',
            'Non-redundant': 'representative basis',
            'Final basis': 'interpretable signals',
        },
        'rejection_bins': [
            {
                'category': 'Future-impact signals',
                'examples': 'citations, FWCI, bursts, altmetrics',
                'color': '#B91C1C',
            },
            {
                'category': 'Prestige/context signals',
                'examples': 'author, journal, institution',
                'color': '#C2410C',
            },
            {
                'category': 'Non-reference signals',
                'examples': 'text/LLM score, abstract semantics',
                'color': '#7C3AED',
            },
            {
                'category': 'Generic graph controls',
                'examples': 'raw reference count, mean age, popularity',
                'color': '#0369A1',
            },
            {
                'category': 'Redundant variants',
                'examples': 'degree, PageRank, closeness, Simpson/variety, effective size',
                'color': '#475569',
            },
        ],
        'final_basis': [label for _, label, _, _ in METRIC_SPECS],
    }


def _draw_panel_b_screening_funnel(ax, payload: Mapping[str, Any]) -> None:
    stages = list(payload['screening_counts'].items())
    stage_notes = payload['stage_notes']
    colors = ['#EAF5EC', '#EAF2FB', '#F0ECFA', '#FFF4DE', '#FCE9E8', '#EAF5EC']
    edges = ['#6A8F68', '#7EA0CC', '#9B7BBF', '#E0A13A', '#D16A5F', '#6A8F68']
    x0, node_w, y, node_h, gap = 0.052, 0.118, 0.705, 0.158, 0.033
    for i, ((label, count), face, edge) in enumerate(zip(stages, colors, edges)):
        x = x0 + i * (node_w + gap)
        rounded_box(ax, x, y, node_w, node_h, face, edge, 0.72, 0.014, zorder=2)
        ax.text(x + node_w / 2, y + 0.100, str(count), fontsize=14.8, fontweight='bold',
                color=edge, ha='center', va='center', transform=ax.transAxes, zorder=3)
        ax.text(x + node_w / 2, y + 0.061, label.replace(' ', '\n'), fontsize=5.25,
                fontweight='bold', color=TEXT_DARK, ha='center', va='center',
                linespacing=0.92, transform=ax.transAxes, zorder=3)
        ax.text(x + node_w / 2, y + 0.022, wrap(stage_notes[label], 18), fontsize=4.7,
                color=TEXT_MID, ha='center', va='center', linespacing=0.92,
                transform=ax.transAxes, zorder=3)
        if i < len(stages) - 1:
            start = (x + node_w + 0.006, y + node_h * 0.50)
            end = (x + node_w + gap - 0.006, y + node_h * 0.50)
            draw_arrow(ax, start, end, color='#64748B', lw=1.05, ms=11)


def _draw_panel_b_rejection_bins(ax, payload: Mapping[str, Any]) -> None:
    ax.text(0.055, 0.625, 'Excluded signal families', fontsize=6.3, fontweight='bold',
            color=TEXT_MID, ha='left', va='center', transform=ax.transAxes)
    bins = payload['rejection_bins']
    x0, y0, w, h = 0.050, 0.245, 0.610, 0.330
    lane_h = h / len(bins)
    rounded_box(ax, x0, y0, w, h, '#FFFFFF', '#E2E8F0', 0.55, 0.010, zorder=1)
    for i, item in enumerate(bins):
        yy = y0 + h - (i + 0.5) * lane_h
        if i:
            ax.plot([x0 + 0.012, x0 + w - 0.012], [yy + lane_h / 2, yy + lane_h / 2],
                    color='#EEF2F7', lw=0.55, transform=ax.transAxes, zorder=2)
        ax.plot([x0 + 0.014, x0 + 0.014], [yy - lane_h * 0.32, yy + lane_h * 0.32],
                color=item['color'], lw=1.7, solid_capstyle='round', transform=ax.transAxes, zorder=3)
        ax.text(x0 + 0.030, yy, item['category'], fontsize=5.8, fontweight='bold',
                color=item['color'], ha='left', va='center', transform=ax.transAxes, zorder=3)
        ax.text(x0 + 0.265, yy, item['examples'], fontsize=5.45, color=TEXT_MID,
                ha='left', va='center', transform=ax.transAxes, zorder=3)


def _draw_panel_b_final_basis(ax, payload: Mapping[str, Any]) -> None:
    x0, y0, w, h = 0.690, 0.245, 0.260, 0.330
    rounded_box(ax, x0, y0, w, h, '#F8FAFC', '#CBD5E1', 0.55, 0.012, zorder=1)
    ax.text(x0 + 0.018, y0 + h - 0.040, 'Final basis', fontsize=6.3, fontweight='bold',
            color=TEXT_MID, ha='left', va='center', transform=ax.transAxes, zorder=3)
    color_map = {label: color for _, label, color, _ in METRIC_SPECS}
    pill_widths = {'B': 0.040, 'RS': 0.044, 'ΔQ0': 0.056, 'Uzzi-style': 0.092, 'RTD': 0.052, 'Burt IP': 0.070, 'PDE': 0.052}
    positions = [
        (x0 + 0.018, y0 + 0.218),
        (x0 + 0.068, y0 + 0.218),
        (x0 + 0.122, y0 + 0.218),
        (x0 + 0.018, y0 + 0.155),
        (x0 + 0.118, y0 + 0.155),
        (x0 + 0.178, y0 + 0.155),
        (x0 + 0.018, y0 + 0.092),
    ]
    for label, (x, y) in zip(payload['final_basis'], positions):
        draw_pill(ax, x, y, label, color_map[label], pill_widths[label], 0.033, fontsize=4.7, zorder=5)


def draw_panel_b(ax, comp: Optional[ComputedData] = None) -> None:
    if comp is not None and comp.evidence_mode == 'strong':
        panel_frame(ax, 'b', 'Candidate screening defines a publication-day basis')
        payload = _panel_b_design_payload(comp)
        _draw_panel_b_screening_funnel(ax, payload)
        _draw_panel_b_rejection_bins(ax, payload)
        _draw_panel_b_final_basis(ax, payload)
        return

    panel_frame(ax, 'b', 'Curated screening from 92 candidates to seven indicators')
    stages = [
        ('Candidates', 92, '#E8F1E3'),
        ('No future\nleakage', 67, '#D9E6CF'),
        ('Reference-\nonly', 49, '#D6EAF8'),
        ('Graph\nperturbation', 29, '#E6DCF4'),
        ('Non-\nredundant', 12, '#FDE2CF'),
        ('Final\nbasis', 7, '#FDE7DF'),
    ]
    xs = np.linspace(0.11, 0.86, len(stages))
    y = 0.73
    for i, (label, count, color) in enumerate(stages):
        rounded_box(ax, xs[i] - 0.055, y - 0.08, 0.11, 0.16, color, BORDER, 0.7, 0.018, zorder=2)
        ax.text(xs[i], y + 0.028, str(count), ha='center', va='center', fontsize=18, fontweight='bold', transform=ax.transAxes)
        ax.text(xs[i], y - 0.047, label, ha='center', va='center', fontsize=6.5, fontweight='bold', transform=ax.transAxes)
        if i < len(stages) - 1:
            draw_arrow(ax, (xs[i] + 0.060, y), (xs[i+1] - 0.060, y), color='#6B7280', lw=1.1, ms=12)

    table_rows = [
        ('1', 'Computable at publication day', 'future citations, FWCI, burst, altmetrics'),
        ('2', 'Uses only references and prior graph G−', 'author reputation, journal, institution'),
        ('3', 'Mechanistically tied to graph change', 'ref count, mean age, generic popularity'),
        ('4', 'Representative, not redundant', 'degree, PageRank, closeness, Simpson variants'),
        ('5', 'Interpretable and robust across domains', 'unstable or hard-to-interpret metrics'),
    ]
    x0, y0, w, h = 0.07, 0.24, 0.86, 0.34
    rounded_box(ax, x0, y0, w, h, '#FFFFFF', '#E5E7EB', 0.6, 0.010)
    col_x = [x0 + 0.035, x0 + 0.17, x0 + 0.55]
    ax.text(col_x[0], y0 + h - 0.045, 'Step', fontsize=6.6, fontweight='bold', ha='left', transform=ax.transAxes)
    ax.text(col_x[1], y0 + h - 0.045, 'Criterion', fontsize=6.6, fontweight='bold', ha='left', transform=ax.transAxes)
    ax.text(col_x[2], y0 + h - 0.045, 'Removed examples', fontsize=6.6, fontweight='bold', ha='left', transform=ax.transAxes)
    for i, (step, crit, examples) in enumerate(table_rows):
        yy = y0 + h - 0.092 - i * 0.055
        ax.plot([x0 + 0.02, x0 + w - 0.02], [yy + 0.028, yy + 0.028], color='#EEF2F7', lw=0.7, transform=ax.transAxes)
        ax.text(col_x[0], yy, step, fontsize=6.1, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        ax.text(col_x[1], yy, crit, fontsize=5.8, ha='left', va='center', transform=ax.transAxes)
        ax.text(col_x[2], yy, examples, fontsize=5.6, ha='left', va='center', color=TEXT_MID, transform=ax.transAxes)

    ax.text(0.50, 0.17, 'Final seven-parameter basis', ha='center', va='center', fontsize=7, fontweight='bold')
    x = 0.19
    for key, label, color, _ in METRIC_SPECS:
        width = 0.075 if len(label) <= 4 else 0.105
        draw_pill(ax, x, 0.09, label, color, width, 0.043, fontsize=5.8)
        x += width + 0.015

def draw_panel_c(ax, comp: Optional[ComputedData] = None) -> None:
    title = 'Mechanism map with cross-domain empirical support' if comp is not None and comp.evidence_mode == 'strong' else 'Publication-day evidence channels covered by seven indicators'
    panel_frame(ax, 'c', title)
    x0 = 0.185
    x1 = 0.590 if comp is not None and comp.evidence_mode == 'strong' else 0.94
    y0 = 0.245
    y1 = 0.785
    order = ['RS', 'PDE', 'B', 'RTD', 'BurtIP', 'DeltaQ0', 'Uzzi']
    spec_map = {key: (label, color, desc) for key, label, color, desc in METRIC_SPECS}
    nrows = len(order); ncols = len(EVIDENCE_CHANNELS)
    colw = (x1-x0)/ncols; rowh = (y1-y0)/nrows

    ax.text(0.055, 0.825, 'Indicator', fontsize=5.9, color=TEXT_LIGHT, fontweight='bold', ha='left', transform=ax.transAxes)
    for i, (channel_title, color, _) in enumerate(EVIDENCE_CHANNELS):
        hx = x0 + i*colw
        rounded_box(ax, hx, 0.805, colw, 0.095, '#F8FAFC', '#E2E8F0', 0.5, 0.006)
        ax.text(hx+colw/2, 0.852, channel_title, ha='center', va='center', fontsize=5.8, color=color, fontweight='bold')
    for i in range(ncols+1):
        xx = x0 + i*colw
        ax.plot([xx, xx], [y0, 0.90], color=GRID, lw=0.55, transform=ax.transAxes)
    for j in range(nrows+1):
        yy = y1 - j*rowh
        ax.plot([0.055, x1], [yy, yy], color=GRID, lw=0.55, transform=ax.transAxes)
    for r, key in enumerate(order):
        label, color, _ = spec_map[key]
        cy = y1 - (r+0.5)*rowh
        ax.text(0.095, cy, label, ha='center', va='center', fontsize=6.3, color=color, fontweight='bold')
        for c, (channel_title, col, _) in enumerate(EVIDENCE_CHANNELS):
            cx = x0 + (c+0.5)*colw
            status = EVIDENCE_MAP.get(key, {}).get(channel_title, '')
            if status == 'primary':
                ax.scatter([cx], [cy], s=78, color=col, edgecolors='white', linewidths=0.5, transform=ax.transAxes)
            elif status == 'secondary':
                ax.scatter([cx], [cy], s=48, facecolors='white', edgecolors='#6B7280', linewidths=0.7, transform=ax.transAxes)
    if comp is not None and comp.evidence_mode == 'strong':
        cols = [c for c in _active_independent_future_outcomes(comp) if c in comp.indicator_delta_corr.columns]
        if cols:
            corr = comp.indicator_delta_corr.copy().loc[order, cols]
            heat_ax = ax.inset_axes([0.655, 0.285, 0.310, 0.500])
            arr = corr.values.astype(float)
            im = heat_ax.imshow(arr, cmap='RdBu_r', vmin=-1, vmax=1)
            heat_ax.set_xticks(range(len(cols)))
            heat_ax.set_yticks(range(len(order)))
            heat_ax.set_xticklabels([_pretty_future_label(c) for c in cols], fontsize=4.5, rotation=34, ha='right', rotation_mode='anchor')
            heat_ax.set_yticklabels([spec_map[k][0] for k in order], fontsize=5.1)
            heat_ax.tick_params(length=0, pad=1)
            for i, key in enumerate(order):
                heat_ax.get_yticklabels()[i].set_color(spec_map[key][1])
                heat_ax.get_yticklabels()[i].set_fontweight('bold')
            for i, key in enumerate(order):
                for j, outcome in enumerate(cols):
                    val = arr[i, j]
                    sig = comp.indicator_future_corr_bootstrap[
                        (comp.indicator_future_corr_bootstrap['metric'].astype(str) == key)
                        & (comp.indicator_future_corr_bootstrap['future_outcome'].astype(str) == outcome)
                    ]
                    mark = ''
                    if not sig.empty and float(sig['ci_low'].iloc[0]) > 0 and float(sig['rho'].iloc[0]) > 0:
                        mark = '+'
                    color = 'white' if np.isfinite(val) and abs(float(val)) >= 0.42 else TEXT_DARK
                    heat_ax.text(j, i, f'{val:.2f}{mark}' if np.isfinite(val) else 'n/e', ha='center', va='center', fontsize=4.15, color=color)
            for s in heat_ax.spines.values():
                s.set_visible(False)
            ax.text(0.655, 0.835, 'Future graph-outcome association', fontsize=5.8, color=TEXT_MID, fontweight='bold', transform=ax.transAxes)
            cax = ax.inset_axes([0.710, 0.148, 0.205, 0.020])
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=4.8, pad=1)
            cb.outline.set_linewidth(0.45)
    ax.scatter([0.075], [0.125], s=78, color='#4B5563', edgecolors='white', linewidths=0.5, transform=ax.transAxes)
    ax.text(0.100, 0.125, 'Primary', fontsize=5.5, va='center')
    ax.scatter([0.205], [0.125], s=48, facecolors='white', edgecolors='#6B7280', linewidths=0.7, transform=ax.transAxes)
    ax.text(0.230, 0.125, 'Secondary', fontsize=5.5, va='center')
    if comp is not None and comp.evidence_mode == 'strong':
        ax.text(0.345, 0.112, '+ marks positive bootstrap CI in future graph-outcome associations', fontsize=4.65, va='center', color=TEXT_LIGHT)


def plot_redundancy_heatmap(ax, comp: ComputedData, selected_only: bool = False) -> None:
    corr = comp.redundancy_corr.copy()
    selected_metrics = [m[0] for m in METRIC_SPECS]
    color_map = {m[0]: m[2] for m in METRIC_SPECS}
    if not selected_only:
        desired = ['B', 'degree', 'pagerank', 'closeness', 'RS', 'field_shannon', 'field_simpson', 'field_variety',
                   'DeltaQ0', 'conductance_delta', 'Uzzi', 'pair_surprisal', 'RTD', 'community_simpson',
                   'BurtIP', 'effective_size', 'constraint_inv', 'PDE', 'field_entropy_norm']
        cols = [c for c in desired if c in corr.columns]
        corr = corr.loc[cols, cols]
    else:
        cols = [c for c in selected_metrics if c in corr.columns]
        corr = corr.loc[cols, cols]
    if SCIPY_OK and corr.shape[0] > 2:
        dist = 1 - np.abs(corr.fillna(0).values)
        Z = linkage(squareform(dist, checks=False), method='average')
        order = leaves_list(Z)
        corr = corr.iloc[order, order]
    arr = corr.fillna(0).values
    im = ax.imshow(arr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.index, fontsize=5)
    ax.tick_params(length=0)
    if selected_only:
        ax.tick_params(axis='y', pad=9)
    for i, name in enumerate(corr.columns):
        if name in selected_metrics:
            ax.get_xticklabels()[i].set_color(color_map[name])
            ax.get_xticklabels()[i].set_fontweight('bold')
    for i, name in enumerate(corr.index):
        if name in selected_metrics:
            ax.get_yticklabels()[i].set_color(color_map[name])
            ax.get_yticklabels()[i].set_fontweight('bold')
            strip_x = -0.72 if selected_only else -0.95
            strip_w = 0.18 if selected_only else 0.25
            ax.add_patch(mpatches.Rectangle((strip_x, i-0.45), strip_w, 0.90, facecolor=color_map[name],
                                            edgecolor='none', clip_on=False, zorder=4))
            ax.add_patch(mpatches.Rectangle((i-0.45, -0.95), 0.90, 0.25, facecolor=color_map[name],
                                            edgecolor='none', clip_on=False, zorder=4))
    for s in ax.spines.values(): s.set_visible(False)
    return im, corr


def draw_panel_d(ax, comp: ComputedData) -> None:
    title = 'Perturbation-profile distributions with redundancy audit' if comp.evidence_mode == 'strong' else 'Landmark profiles and candidate redundancy'
    panel_frame(ax, 'd', title)
    ax.text(
        0.05,
        0.900,
        'Future-top-decile cases and landmarks are scored against tiered matched controls; right inset shows selected-indicator redundancy.',
        fontsize=5.95,
        fontweight='bold',
        color=TEXT_MID,
        transform=ax.transAxes,
    )
    order = [m[0] for m in METRIC_SPECS]
    label_map = {m[0]: m[1] for m in METRIC_SPECS}
    color_map = {m[0]: m[2] for m in METRIC_SPECS}
    pdat = comp.percentile_long.copy()
    if pdat.empty:
        raise ValueError('Panel d requires percentile profiles from matched controls.')

    plot_left, plot_right = 0.185, 0.625
    y_positions = np.linspace(0.775, 0.215, len(order))
    for t in range(0, 101, 25):
        xx = plot_left + (t / 100) * (plot_right - plot_left)
        ax.plot([xx, xx], [0.165, 0.825], color='#E5E7EB', lw=0.65, transform=ax.transAxes, zorder=0)
        ax.text(xx, 0.142, f'{t}', fontsize=5.1, color=TEXT_LIGHT, ha='center', va='top', transform=ax.transAxes)
    ax.text((plot_left + plot_right) / 2, 0.108, 'Percentile among matched controls', fontsize=5.8, fontweight='bold', ha='center', transform=ax.transAxes)

    for row_idx, (y, metric) in enumerate(zip(y_positions, order)):
        sub = pdat[pdat['metric'].astype(str) == metric].copy()
        vals = pd.to_numeric(sub['percentile'], errors='coerce').dropna()
        if vals.empty:
            continue
        future_vals = pd.to_numeric(
            sub.loc[sub.get('profile_type', pd.Series(index=sub.index, dtype=str)).astype(str) == 'future_top_decile', 'percentile'],
            errors='coerce',
        ).dropna().values
        landmark_vals = pd.to_numeric(
            sub.loc[sub.get('profile_type', pd.Series(index=sub.index, dtype=str)).astype(str) == 'landmark', 'percentile'],
            errors='coerce',
        ).dropna().values
        all_vals = vals.values
        q10, q25, q50, q75, q90 = np.nanpercentile(all_vals, [10, 25, 50, 75, 90])
        x10 = plot_left + (q10 / 100) * (plot_right - plot_left)
        x25 = plot_left + (q25 / 100) * (plot_right - plot_left)
        x50 = plot_left + (q50 / 100) * (plot_right - plot_left)
        x75 = plot_left + (q75 / 100) * (plot_right - plot_left)
        x90 = plot_left + (q90 / 100) * (plot_right - plot_left)
        ax.plot([x10, x90], [y, y], color='#CBD5E1', lw=1.2, solid_capstyle='round', transform=ax.transAxes, zorder=1)
        ax.plot([x25, x75], [y, y], color=color_map[metric], lw=3.0, alpha=0.50, solid_capstyle='round', transform=ax.transAxes, zorder=2)
        ax.scatter([x50], [y], s=18, color=color_map[metric], edgecolors='white', linewidths=0.4, transform=ax.transAxes, zorder=4)

        for idx, val in enumerate(future_vals):
            xx = plot_left + (float(val) / 100) * (plot_right - plot_left)
            jitter = ((stable_int_id(f'{metric}-{idx}', modulo=1000) / 999.0) - 0.5) * 0.050
            ax.scatter([xx], [y + jitter], s=5.5, color='#64748B', alpha=0.20, linewidths=0, transform=ax.transAxes, zorder=2)
        for val in landmark_vals:
            xx = plot_left + (float(val) / 100) * (plot_right - plot_left)
            ax.scatter([xx], [y + 0.035], s=28, facecolors='white', edgecolors=color_map[metric], linewidths=0.85, transform=ax.transAxes, zorder=5)
        landmark_basis = landmark_vals if len(landmark_vals) else all_vals
        med = float(np.nanmedian(landmark_basis))
        xx = plot_left + (med / 100) * (plot_right - plot_left)
        ax.scatter([xx], [y + 0.035], marker='*', s=110, color=color_map[metric], edgecolors='white', linewidths=0.55, transform=ax.transAxes, zorder=6)
        ax.text(0.145, y, label_map[metric], fontsize=6.6, color=color_map[metric], fontweight='bold', ha='right', va='center', transform=ax.transAxes)
        ax.text(0.642, y, f'{med:.0f}', fontsize=6.5, color=TEXT_DARK, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        if row_idx:
            row_gap = float(y_positions[row_idx - 1] - y)
            sep_y = y + row_gap * 0.48
            ax.plot([0.060, 0.670], [sep_y, sep_y], color='#F1F5F9', lw=0.55, transform=ax.transAxes, zorder=0)
    ax.text(0.655, 0.828, 'Landmark\nmedian', fontsize=5.4, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    ax.scatter([0.225], [0.865], s=20, color='#64748B', alpha=0.28, linewidths=0, transform=ax.transAxes)
    ax.text(0.245, 0.865, 'Future top-decile paper', fontsize=5.25, va='center', color=TEXT_LIGHT, transform=ax.transAxes)
    ax.scatter([0.410], [0.865], s=28, facecolors='white', edgecolors='#0B4FA3', linewidths=0.85, transform=ax.transAxes)
    ax.scatter([0.410], [0.865], marker='*', s=85, color='#0B4FA3', edgecolors='white', linewidths=0.45, transform=ax.transAxes)
    ax.text(0.435, 0.865, 'Landmark profile', fontsize=5.25, va='center', color=TEXT_LIGHT, transform=ax.transAxes)

    ax.text(0.835, 0.835, 'Indicator redundancy', fontsize=5.8, color=TEXT_MID, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    red_ax = ax.inset_axes([0.720, 0.520, 0.230, 0.285])
    im, corr = plot_redundancy_heatmap(red_ax, comp, selected_only=True)
    red_ax.set_xticklabels([])
    red_ax.tick_params(axis='x', length=0)
    cax = ax.inset_axes([0.755, 0.448, 0.160, 0.018])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=4.5, pad=1)
    cb.outline.set_linewidth(0.45)

    pair_defs = [('RS', 'PDE'), ('B', 'BurtIP'), ('DeltaQ0', 'RTD'), ('Uzzi', 'RS')]
    pair_y = 0.354
    ax.text(0.720, pair_y + 0.045, 'Representative relation checks', fontsize=5.8, color=TEXT_MID, fontweight='bold', transform=ax.transAxes)
    for idx, (left, right) in enumerate(pair_defs):
        yy = pair_y - idx * 0.055
        if left in comp.redundancy_corr.index and right in comp.redundancy_corr.columns:
            rho = float(comp.redundancy_corr.loc[left, right])
            text = f'{label_map.get(left, left)} ↔ {label_map.get(right, right)}'
            rounded_box(ax, 0.715, yy - 0.016, 0.230, 0.040, '#FFFFFF', '#E2E8F0', 0.50, 0.008)
            ax.text(0.727, yy + 0.004, text, fontsize=4.85, color=TEXT_DARK, ha='left', va='center', transform=ax.transAxes)
            ax.text(0.932, yy + 0.004, f'{rho:+.2f}', fontsize=5.35, color=TEXT_MID, fontweight='bold', ha='right', va='center', transform=ax.transAxes)

def draw_panel_e(ax, comp: ComputedData) -> None:
    panel_frame(ax, 'e', 'Landmarks and high-future cases vs controls')
    ax.text(0.05, 0.90, 'Percentile among tiered matched controls; all indicators oriented higher = stronger perturbation', fontsize=6.3, fontweight='bold')
    order = [m[0] for m in METRIC_SPECS]
    label_map = {m[0]: m[1] for m in METRIC_SPECS}; color_map = {m[0]: m[2] for m in METRIC_SPECS}
    pdat = comp.percentile_long.copy()
    if pdat.empty:
        raise ValueError('Panel e requires at least one landmark paper and enough matched controls.')
    agg = pdat.groupby('metric', as_index=False)['percentile'].median()
    y_positions = np.linspace(0.77, 0.22, len(order))

    plot_left, plot_right = 0.22, 0.86
    for t in np.linspace(0, 100, 5):
        xx = plot_left + (t/100)*(plot_right-plot_left)
        ax.plot([xx, xx], [0.17, 0.83], color='#E5E7EB', lw=0.6, transform=ax.transAxes, zorder=0)
    for y, key in zip(y_positions, order):
        sub = pdat[pdat['metric'] == key].copy()
        vals = sub['percentile'].dropna().values
        if len(vals) == 0:
            continue
        if 'profile_type' in sub.columns:
            high_vals = sub.loc[sub['profile_type'] == 'future_top_decile', 'percentile'].dropna().values
            landmark_vals = sub.loc[sub['profile_type'] == 'landmark', 'percentile'].dropna().values
        else:
            high_vals = np.array([])
            landmark_vals = vals
        for v in high_vals:
            xx = plot_left + (v/100)*(plot_right-plot_left)
            ax.scatter([xx], [y], s=7, color='#BDBDBD', alpha=0.55, transform=ax.transAxes, zorder=2)
        for v in landmark_vals:
            xx = plot_left + (v/100)*(plot_right-plot_left)
            ax.scatter([xx], [y], s=18, facecolors='white', edgecolors=color_map[key], linewidths=0.7, transform=ax.transAxes, zorder=3)
        star_vals = landmark_vals if len(landmark_vals) else vals
        med = float(np.median(star_vals))
        xx = plot_left + (med/100)*(plot_right-plot_left)
        ax.scatter([xx], [y], marker='*', s=120, color=color_map[key], edgecolors='white', linewidths=0.6, transform=ax.transAxes, zorder=4)
        ax.text(0.17, y, label_map[key], fontsize=7, color=color_map[key], fontweight='bold', ha='right', va='center', transform=ax.transAxes)
        ax.text(0.87, y, f'{med:.0f}', fontsize=8, color=TEXT_DARK, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.915, 0.83, 'Median\npercentile', fontsize=6.2, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    for t in range(0, 101, 25):
        xx = plot_left + (t/100)*(plot_right-plot_left)
        ax.text(xx, 0.155, f'{t}', fontsize=5.5, color=TEXT_LIGHT, ha='center', transform=ax.transAxes)
    ax.text((plot_left+plot_right)/2, 0.125, 'Percentile', fontsize=6.2, fontweight='bold', ha='center', transform=ax.transAxes)

def draw_panel_f(ax, comp: ComputedData) -> None:
    title = 'Publication-day indicators anticipate future G+τ graph deltas' if comp.evidence_mode == 'strong' else 'Internal correspondence with direct graph-delta observables'
    panel_frame(ax, 'f', title)
    control_text = 'Partial Spearman with domain/year/log-reference controls; cells marked + have positive bootstrap CI' if comp.evidence_mode == 'strong' else 'Partial Spearman correlation, controlling year and log reference count'
    ax.text(0.05, 0.90, control_text, fontsize=6.2, fontweight='bold')
    corr = comp.indicator_delta_corr.copy()
    order = [m[0] for m in METRIC_SPECS]
    corr = corr.loc[order, :]
    label_map = {m[0]: m[1] for m in METRIC_SPECS}
    color_map = {m[0]: m[2] for m in METRIC_SPECS}
    cols = list(corr.columns)
    pretty_map = {
        'community_reach': 'Community\nreach',
        'field_entropy': 'Field\nentropy',
        'cross_community_adoption': 'Cross-community\nadoption',
        'modularity_shock': 'Modularity\nshock*',
        'path_shortening': 'Path\nshortening',
        'partition_change': 'Partition\nchange',
        'boundary_mixing': 'Boundary\nmixing',
        'hub_formation': 'Hub\nformation',
        'component_reach': 'Component\nreach',
    }
    heat_ax = ax.inset_axes([0.10, 0.31, 0.80, 0.50])
    arr = corr.values.astype(float) if cols else np.zeros((len(order), 1))
    im = heat_ax.imshow(arr, cmap='RdBu_r', vmin=-1, vmax=1)
    heat_ax.set_xticks(range(len(cols))); heat_ax.set_yticks(range(len(order)))
    pretty_cols = [pretty_map.get(c, c.replace('_', '\n')) for c in cols]
    heat_ax.set_xticklabels(pretty_cols, fontsize=5.0, rotation=25, ha='right', rotation_mode='anchor')
    heat_ax.set_yticklabels([label_map[k] for k in order], fontsize=6)
    heat_ax.tick_params(length=0)
    for i, k in enumerate(order):
        heat_ax.get_yticklabels()[i].set_color(color_map[k]); heat_ax.get_yticklabels()[i].set_fontweight('bold')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            marker = ''
            if comp.evidence_mode == 'strong' and not comp.indicator_future_corr_bootstrap.empty and j < len(cols):
                metric = order[i]
                outcome = cols[j]
                b = comp.indicator_future_corr_bootstrap[
                    (comp.indicator_future_corr_bootstrap['metric'] == metric)
                    & (comp.indicator_future_corr_bootstrap['future_outcome'] == outcome)
                ]
                if not b.empty and float(b['ci_low'].iloc[0]) > 0 and float(b['rho'].iloc[0]) > 0:
                    marker = '+'
            txt = f'{val:.2f}{marker}' if np.isfinite(val) else 'n/e'
            heat_ax.text(j, i, txt, ha='center', va='center', fontsize=5.5, color='black')
    for s in heat_ax.spines.values(): s.set_visible(False)
    cax = ax.inset_axes([0.20, 0.165, 0.56, 0.026])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=5)
    cb.set_label('', fontsize=0)

# --------------------------
# Full figure assembly
# --------------------------

def assemble_figure(
    raw: RawData,
    comp: ComputedData,
    focus_paper_id: Optional[str],
    future_tau: Optional[int],
    outpath: Path,
    fig1_snapshot_dir: Optional[Path] = None,
) -> None:
    setup_style()
    fig = plt.figure(figsize=(18, 12.2), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.95, 1.08], width_ratios=[1.02, 1.08], hspace=0.115, wspace=0.055)
    if comp.evidence_mode == 'strong':
        status = str((comp.quality_gates or {}).get('status_label', 'multi-domain diagnostic evidence'))
        if status == 'strong experimental evidence':
            title = 'Fig. 2 | Publication-day graph perturbation signals anticipate future cross-domain knowledge movement'
        else:
            title = 'Fig. 2 diagnostic | Publication-day perturbations in multi-domain raw citation data'
        subtitle = 'Multi-domain evidence using raw citation corpora, reference-closure audits, matched controls, and future graph deltas'
    else:
        title = 'Fig. 2 | Why these seven indicators?'
        subtitle = 'Construction and empirical checks of a publication-day graph-perturbation basis'
    fig.text(0.5, 0.985, title, ha='center', va='top', fontsize=18.0 if comp.evidence_mode == 'strong' else 20, fontweight='bold')
    fig.text(0.5, 0.955, subtitle, ha='center', va='top', fontsize=11.0, color=TEXT_MID)

    axa = fig.add_subplot(gs[0, 0]); draw_panel_a(axa, raw, focus_paper_id, future_tau=future_tau, fig1_snapshot_dir=fig1_snapshot_dir)
    axb = fig.add_subplot(gs[0, 1]); draw_panel_b(axb, comp)
    axc = fig.add_subplot(gs[1, 0]); draw_panel_c(axc, comp)
    axd = fig.add_subplot(gs[1, 1]); draw_panel_d(axd, comp)
    fig.savefig(outpath)
    plt.close(fig)


def save_single_panel(
    panel: str,
    raw: RawData,
    comp: Optional[ComputedData],
    focus_paper_id: Optional[str],
    future_tau: Optional[int],
    outpath: Path,
    fig1_snapshot_dir: Optional[Path] = None,
) -> None:
    setup_style()
    size_map = {'a': (7.6, 6.0), 'b': (7.6, 6.0), 'c': (8.2, 6.8), 'd': (8.4, 6.8), 'e': (7.1, 6.8), 'f': (7.1, 6.8)}
    fig, ax = plt.subplots(figsize=size_map.get(panel, (7, 6)), dpi=300)
    if panel == 'a':
        draw_panel_a(ax, raw, focus_paper_id, future_tau=future_tau, fig1_snapshot_dir=fig1_snapshot_dir)
    elif panel == 'b':
        draw_panel_b(ax, comp)
    elif panel == 'c':
        draw_panel_c(ax, comp)
    elif panel == 'd':
        if comp is None:
            raise ValueError('Panel d requires computed metrics.')
        draw_panel_d(ax, comp)
    elif panel == 'e':
        if comp is None:
            raise ValueError('Panel e requires computed metrics.')
        draw_panel_e(ax, comp)
    elif panel == 'f':
        if comp is None:
            raise ValueError('Panel f requires computed metrics.')
        draw_panel_f(ax, comp)
    else:
        raise ValueError(f'Unknown panel: {panel}')
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------
# Export helpers
# --------------------------

def export_tables(comp: ComputedData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if comp.evidence_mode == 'strong':
        comp.input_audit.to_csv(out_dir / 'fig2_input_audit.csv', index=False)
        comp.domain_adequacy.to_csv(out_dir / 'fig2_domain_adequacy.csv', index=False)
        comp.reference_closure_report.to_csv(out_dir / 'fig2_reference_closure_report.csv', index=False)
        comp.paper_metrics.to_csv(out_dir / 'fig2_publication_day_indicators.csv', index=False)
        comp.graph_deltas.to_csv(out_dir / 'fig2_future_graph_deltas.csv', index=False)
        comp.matched_controls.to_csv(out_dir / 'fig2_matched_controls.csv', index=False)
        comp.candidate_metrics.to_csv(out_dir / 'fig2_candidate_metrics.csv', index=False)
        comp.redundancy_corr.to_csv(out_dir / 'fig2_candidate_redundancy.csv')
        comp.indicator_delta_corr.to_csv(out_dir / 'fig2_indicator_future_corr.csv')
        comp.indicator_future_corr_bootstrap.to_csv(out_dir / 'fig2_indicator_future_corr_bootstrap.csv', index=False)
        comp.percentile_long.to_csv(out_dir / 'fig2_landmark_percentiles.csv', index=False)
        comp.landmark_summary.to_csv(out_dir / 'fig2_landmark_percentile_summary.csv', index=False)
        comp.metric_standardization_diagnostics.to_csv(out_dir / 'fig2_metric_standardization_diagnostics.csv', index=False)
        comp.graph_delta_diagnostics.to_csv(out_dir / 'fig2_future_graph_delta_diagnostics.csv', index=False)
        comp.graph_delta_diagnostics.to_csv(out_dir / 'fig2_graph_delta_diagnostics.csv', index=False)
        comp.evidence_support.to_csv(out_dir / 'fig2_mechanism_evidence_support.csv', index=False)
        (out_dir / 'fig2_quality_gates.json').write_text(
            json.dumps(comp.quality_gates, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)),
            encoding='utf-8',
        )
        return
    comp.paper_metrics.to_csv(out_dir / 'fig2_paper_metrics.csv', index=False)
    comp.candidate_metrics.to_csv(out_dir / 'fig2_candidate_metrics.csv', index=False)
    comp.graph_deltas.to_csv(out_dir / 'fig2_graph_deltas.csv', index=False)
    comp.redundancy_corr.to_csv(out_dir / 'fig2_redundancy_corr.csv')
    comp.indicator_delta_corr.to_csv(out_dir / 'fig2_indicator_delta_corr.csv')
    comp.percentile_long.to_csv(out_dir / 'fig2_landmark_percentiles.csv', index=False)
    comp.landmark_summary.to_csv(out_dir / 'fig2_landmark_percentile_summary.csv', index=False)
    comp.metric_standardization_diagnostics.to_csv(out_dir / 'fig2_metric_standardization_diagnostics.csv', index=False)
    comp.graph_delta_diagnostics.to_csv(out_dir / 'fig2_graph_delta_diagnostics.csv', index=False)


def write_fig2_reports(args: argparse.Namespace, comp: ComputedData, generated_files: Sequence[Path]) -> None:
    quality = comp.quality_gates if comp.evidence_mode == 'strong' else {}
    write_run_manifest(
        args.out_dir,
        figure='fig2',
        argv=sys.argv,
        inputs={
            'data_dir': str(args.data_dir),
            'evidence_mode': args.evidence_mode,
            'domains': parse_domain_string(args.domains) if args.evidence_mode == 'strong' else [args.domain],
            'future_tau': int(args.future_tau),
            'reference_closure': getattr(args, 'reference_closure', None),
            'online_expand': bool(getattr(args, 'online_expand', False)),
            'fig1_corpus_source': getattr(args, 'fig1_corpus_source', None),
        },
        domains=parse_domain_string(args.domains) if args.evidence_mode == 'strong' else [args.domain],
        quality_gates=quality,
        extra={
            'panel': args.panel,
            'export_tables': bool(args.export_tables),
            'strict_main_figure': bool(getattr(args, 'strict_main_figure', False)),
        },
    )
    write_figure_quality_report(
        args.out_dir,
        figure='fig2',
        generated_files=generated_files,
        quality_gates=quality,
        extra={
            'domain_adequacy_rows': int(len(comp.domain_adequacy)) if comp.evidence_mode == 'strong' else 0,
            'reference_closure_rows': int(len(comp.reference_closure_report)) if comp.evidence_mode == 'strong' else 0,
        },
    )


# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Empirical Fig. 2 pipeline: compute data and draw panels a–f separately or jointly.')
    p.add_argument('--data-dir', type=Path, default=default_fig2_data_root(),
                   help='Local data directory. Accepts a directory with works.csv/citations.csv/topics.csv/topic_edges.csv, '
                        'a Fig. 1 domain directory with works_selected.csv/paper_edges.csv/topic_nodes.csv/topic_edges.csv, '
                        f'or a Fig. 1 output root. Default prefers {DEFAULT_CORPUS_FIG2_DATA_ROOT}, then {DEFAULT_FIG1_DATA_ROOT}')
    p.add_argument('--evidence-mode', choices=['strong', 'legacy'], default='strong',
                   help='strong uses multi-domain Fig. 1 raw/Fig. 3 future-delta evidence; legacy preserves the single-domain diagnostic figure.')
    p.add_argument('--domain', type=str, default=DEFAULT_DOMAIN,
                   help=f'Fig. 1 domain subdirectory to read when --data-dir is a root. Default: {DEFAULT_DOMAIN}')
    p.add_argument('--domains', type=str, default=','.join(default_strong_domains()),
                   help='Comma/space separated domains for --evidence-mode strong.')
    p.add_argument('--fig1-corpus-source', choices=['raw', 'selected'], default='raw',
                   help='For strong mode, use Fig. 1 works_raw.jsonl or selected Fig. 1 corpus. Default: raw.')
    p.add_argument('--include-hybrid-edges', action='store_true',
                   help='When reading paper_edges.csv, include bibliographic/cocitation-only edges. '
                        'By default only rows with direct > 0 are used as citation links.')
    p.add_argument('--panel', type=str, default='all', choices=['a', 'b', 'c', 'd', 'e', 'f', 'all'],
                   help='Which panel to draw. Use all to assemble the full figure.')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_STRONG_OUTPUT_DIR)
    p.add_argument('--focus-paper-id', type=str, default=None,
                   help='Paper id to highlight in panel a. Defaults to the earliest landmark.')
    p.add_argument('--future-tau', type=int, default=10,
                   help='Future horizon for strong-mode G+τ validation and panel a. Default: 10.')
    p.add_argument('--example-domain', type=str, default=DEFAULT_DOMAIN,
                   help='Domain used for panel a example in strong mode. Default: crispr.')
    p.add_argument('--fig1-snapshot-dir', type=Path, default=None,
                   help='Optional Fig. 1 domain output directory used to reuse the first, landmark-window, '
                        'and final Fig. 1 snapshots in panel a. Defaults to strict Fig. 1 best4 outputs when available.')
    p.add_argument('--focal-paper-list', type=Path, default=None,
                   help='Optional text file with one focal paper id per line for metric computation. Defaults to all papers.')
    p.add_argument('--export-tables', action='store_true', help='Export all intermediate computed tables.')
    p.add_argument('--strict-main-figure', action='store_true',
                   help='Write outputs but exit non-zero unless all strong-evidence quality gates pass.')
    p.add_argument('--progress-interval', type=int, default=100,
                   help='Print one focal-paper progress update every N papers while computing metrics. Default: 100.')
    p.add_argument('--min-controls', type=int, default=50,
                   help='Minimum controls requested for matched-control percentile matching. Default: 50.')
    p.add_argument('--min-refs', type=int, default=5,
                   help='Minimum prior references required for strong-mode publication-day indicators. Default: 5.')
    p.add_argument('--max-papers', type=int, default=None,
                   help='Optional debug cap for strong-mode metric computation.')
    p.add_argument('--bootstrap-reps', type=int, default=300,
                   help='Bootstrap replicates for strong-mode indicator/future correlations. Default: 300.')
    p.add_argument('--seed', type=int, default=2028,
                   help='Random seed for strong-mode bootstrap diagnostics.')
    p.add_argument('--reference-closure', choices=['auto', 'off', 'required'], default='auto',
                   help='Reference-closure policy for strong mode. auto audits locally and materializes closure only with --online-expand.')
    p.add_argument('--reference-closure-cap', type=int, default=50000,
                   help='Maximum out-of-corpus reference IDs to materialize per domain when --online-expand is set.')
    p.add_argument('--closure-coverage-target', type=float, default=0.80,
                   help='Weighted reference coverage target for closure materialization. Default: 0.80.')
    p.add_argument('--online-expand', action='store_true',
                   help='Fetch OpenAlex metadata for reference-closure nodes. Disabled by default for reproducibility.')
    p.add_argument('--no-reuse-cache', action='store_true',
                   help='Force recomputation of strong-mode metrics even when matching exported tables exist.')
    p.add_argument('--no-prepare-input', action='store_true',
                   help='Read --data-dir directly instead of first preparing normalized Fig. 2 input in --out-dir.')
    p.add_argument('--fig1-config', type=Path, default=None,
                   help='Optional Fig. 1 YAML config used with --run-fig1-if-missing.')
    p.add_argument('--run-fig1-if-missing', action='store_true',
                   help='Run the Fig. 1 pipeline to materialize source data if --data-dir does not contain usable exports.')
    p.add_argument('--no-fig1-cache', action='store_true',
                   help='When running Fig. 1, ignore cached works_raw.jsonl and re-download.')
    p.add_argument('--openalex-api-key', default=getenv('OPENALEX_API_KEY'),
                   help='OpenAlex API key passed through when --run-fig1-if-missing is used.')
    p.add_argument('--openalex-api-keys', default=getenv('OPENALEX_API_KEYS'),
                   help='Comma/space separated OpenAlex API keys passed through to online fetchers.')
    p.add_argument('--email', default=getenv('OPENALEX_EMAIL'),
                   help='OpenAlex contact email passed through when --run-fig1-if-missing is used.')
    p.add_argument('--quiet', action='store_true', help='Suppress progress logs.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.quiet
    progress_log(f'Starting Fig. 2 empirical pipeline: panel={args.panel}, evidence_mode={args.evidence_mode}', progress)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig1_snapshot_dir = resolve_panel_a_fig1_snapshot_dir(args) if args.panel in {'a', 'all'} else None
    if fig1_snapshot_dir is not None:
        progress_log(f'Panel a will reuse Fig. 1 snapshots from: {fig1_snapshot_dir}', progress)
    if args.evidence_mode == 'strong':
        cached = load_strong_cache_if_valid(args, progress)
        if cached is None:
            raw, comp = build_strong_evidence_data(args, progress)
            save_strong_run_config(args)
        else:
            raw, comp = cached
        if args.export_tables:
            progress_log('Exporting strong-evidence intermediate tables...', progress)
            export_tables(comp, args.out_dir)
            save_strong_run_config(args)
            progress_log('Strong-evidence tables exported.', progress)
        if args.panel == 'all':
            outpath = args.out_dir / 'fig2_empirical_full.png'
            progress_log(f'Assembling strong-mode full figure: {outpath}', progress)
            assemble_figure(raw, comp, args.focus_paper_id, args.future_tau, outpath, fig1_snapshot_dir=fig1_snapshot_dir)
            progress_log(f'Saved full figure: {outpath}', progress)
        else:
            outpath = args.out_dir / f'panel_{args.panel}.png'
            progress_log(f'Drawing strong-mode panel {args.panel}: {outpath}', progress)
            save_single_panel(args.panel, raw, comp, args.focus_paper_id, args.future_tau, outpath, fig1_snapshot_dir=fig1_snapshot_dir)
            progress_log(f'Saved panel {args.panel}: {outpath}', progress)
        write_fig2_reports(args, comp, [outpath])
        if args.strict_main_figure and strict_main_figure_failed(comp.quality_gates):
            write_strict_failure_report(
                args.out_dir,
                figure='fig2',
                quality_gates=comp.quality_gates,
                message='Fig. 2 remains diagnostic because at least one strong-evidence quality gate failed.',
            )
            raise SystemExit(2)
        progress_log('Done.', progress)
        return

    source_data_dir = args.data_dir
    if not args.no_prepare_input:
        source_data_dir = prepare_fig2_input_data(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            domain=args.domain,
            direct_only=not args.include_hybrid_edges,
            fig1_config=args.fig1_config,
            run_fig1_if_missing=args.run_fig1_if_missing,
            use_fig1_cache=not args.no_fig1_cache,
            openalex_api_key=args.openalex_api_key,
            openalex_api_keys=args.openalex_api_keys,
            email=args.email,
            progress=progress,
        )
    raw = load_raw_data(source_data_dir, domain=None if not args.no_prepare_input else args.domain,
                        direct_only=not args.include_hybrid_edges, progress=progress)
    focal_ids = None
    if args.focal_paper_list is not None:
        focal_ids = [line.strip() for line in args.focal_paper_list.read_text(encoding='utf-8').splitlines() if line.strip()]
        progress_log(f'Loaded focal paper list: {len(focal_ids):,} ids from {args.focal_paper_list}', progress)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_log(f'Output directory: {args.out_dir}', progress)
    needs_metrics = args.panel in {'d', 'e', 'f', 'all'} or args.export_tables
    progress_log(f'Metric computation required: {needs_metrics}', progress)
    comp = compute_all_metrics(
        raw,
        focal_ids=focal_ids,
        min_controls=args.min_controls,
        progress=progress,
        progress_interval=args.progress_interval,
    ) if needs_metrics else None
    if args.export_tables:
        if comp is None:
            raise ValueError('Cannot export tables without computed metrics.')
        progress_log('Exporting intermediate tables...', progress)
        export_tables(comp, args.out_dir)
        progress_log('Intermediate tables exported.', progress)
    if args.panel == 'all':
        if comp is None:
            raise ValueError('Full figure requires computed metrics.')
        outpath = args.out_dir / 'fig2_empirical_full.png'
        progress_log(f'Assembling full figure: {outpath}', progress)
        assemble_figure(raw, comp, args.focus_paper_id, args.future_tau, outpath, fig1_snapshot_dir=fig1_snapshot_dir)
        progress_log(f'Saved full figure: {outpath}', progress)
    else:
        outpath = args.out_dir / f'panel_{args.panel}.png'
        progress_log(f'Drawing panel {args.panel}: {outpath}', progress)
        save_single_panel(args.panel, raw, comp, args.focus_paper_id, args.future_tau, outpath, fig1_snapshot_dir=fig1_snapshot_dir)
        progress_log(f'Saved panel {args.panel}: {outpath}', progress)
    progress_log('Done.', progress)


if __name__ == '__main__':
    main()

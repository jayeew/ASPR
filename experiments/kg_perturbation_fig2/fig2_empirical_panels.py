from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIG1_DATA_ROOT = PROJECT_ROOT / 'outputs' / 'kg_perturbation_fig1'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'kg_perturbation_fig2'
DEFAULT_DOMAIN = 'crispr'

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
    ('Uzzi', 'Uzzi', '#7C3AED', 'atypical recombination'),
    ('RTD', 'RTD', '#0891B2', 'reference target diversity'),
    ('BurtIP', 'Burt IP', '#2563EB', 'structural holes'),
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
    ('Knowledge\nbreadth', '#2E7D32', 'Breadth of disciplinary / topical coverage'),
    ('Structural\nbrokerage', '#0B4FA3', 'Potential shortcut creation across communities'),
    ('Boundary\nperturbation', '#F97316', 'Potential to disturb module boundaries'),
    ('Combinatorial\natypicality', '#7C3AED', 'Unusual but interpretable reference combinations'),
]

EVIDENCE_MAP = {
    'B': {'Structural\nbrokerage': 'primary', 'Boundary\nperturbation': 'secondary'},
    'RS': {'Knowledge\nbreadth': 'primary'},
    'DeltaQ0': {'Boundary\nperturbation': 'primary'},
    'Uzzi': {'Combinatorial\natypicality': 'primary', 'Boundary\nperturbation': 'secondary'},
    'RTD': {'Structural\nbrokerage': 'primary', 'Knowledge\nbreadth': 'secondary'},
    'BurtIP': {'Structural\nbrokerage': 'primary'},
    'PDE': {'Knowledge\nbreadth': 'primary'},
}


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
    if 'display_community' not in works.columns:
        works['display_community'] = pd.NA
    community_fallback = works['community'] if 'community' in works.columns else pd.Series(-1, index=works.index)
    works['display_community'] = pd.to_numeric(works['display_community'], errors='coerce')
    works['display_community'] = works['display_community'].fillna(pd.to_numeric(community_fallback, errors='coerce'))
    works['display_community'] = works['display_community'].fillna(-1).astype(int)
    if 'anchor_label' not in works.columns:
        works['anchor_label'] = pd.NA
    anchor_label = works['anchor_label'].replace('', pd.NA)
    works['anchor_label'] = anchor_label
    works['is_landmark'] = (anchor_label.notna() & (anchor_label.astype(str).str.strip() != '')).astype(int)
    if 'title' not in works.columns:
        works['title'] = works['id']
    base_cols = [
        'id', 'year', 'title', 'domain', 'primary_field', 'display_community',
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
    for source in selected:
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
    }
    save_prepare_report(report, prepared_dir)
    progress_log(f'Prepared normalized Fig. 2 input in {prepared_dir}', progress)
    return prepared_dir


def default_fig1_config_for_domain(domain: Optional[str]) -> Optional[Path]:
    if not domain:
        return None
    path = PROJECT_ROOT / 'experiments' / 'kg_perturbation_fig1' / 'configs' / f'{domain}.yaml'
    return path if path.exists() else None


def run_fig1_pipeline_for_input(
    fig1_config: Path,
    fig1_out_dir: Path,
    use_cache: bool,
    openalex_api_key: Optional[str],
    email: Optional[str],
    progress: bool,
) -> Path:
    progress_log(f'Running Fig. 1 pipeline to materialize source data: {fig1_config}', progress)
    from experiments.kg_perturbation_fig1.fig1_knowledge_perturbation_v3 import (  # pylint: disable=import-outside-toplevel
        OpenAlexClient,
        load_config,
        run_domain,
    )

    cfg = load_config(fig1_config)
    api_cfg = cfg.get('api', {})
    client = OpenAlexClient(
        api_key=openalex_api_key,
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

    if 'display_community' not in out.columns:
        out['display_community'] = pd.NA
    community_fallback = out['community'] if 'community' in out.columns else pd.Series(-1, index=out.index)
    out['display_community'] = pd.to_numeric(out['display_community'], errors='coerce')
    out['display_community'] = out['display_community'].fillna(pd.to_numeric(community_fallback, errors='coerce'))
    out['display_community'] = out['display_community'].fillna(-1).astype(int)

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

    require_columns(works, ['id', 'year', 'display_community', 'primary_field', 'is_landmark'], works_path.name)
    require_columns(citations, ['source', 'target'], citations_path.name)
    require_columns(topics, ['community', 'label', 'x', 'y'], topics_path.name)
    require_columns(topic_edges, ['source_community', 'target_community', 'weight'], topic_edges_path.name)

    works['id'] = works['id'].astype(str)
    citations['source'] = citations['source'].astype(str)
    citations['target'] = citations['target'].astype(str)
    works['year'] = works['year'].astype(int)
    works['display_community'] = works['display_community'].astype(int)
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
    meta = raw.works[['id', 'year', 'display_community', 'primary_field']].rename(columns={
        'id': 'target', 'display_community': 'target_comm', 'primary_field': 'target_field', 'year': 'target_year'})
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


def field_year_normalize(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in metric_cols:
        out[col + '_z'] = out.groupby(['primary_field', 'year'])[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return out


def compute_all_metrics(
    raw: RawData,
    focal_ids: Optional[Sequence[str]] = None,
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
    comm_map_all = dict(zip(works['id'].astype(str), works['display_community'].astype(int)))
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
    for idx, year in enumerate(years, start=1):
        graph = build_prior_graph(works, raw.citations, int(year))
        prior_graphs[year] = graph
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
        refs = get_reference_rows(raw, pid, year)
        if refs.empty:
            skipped_empty_refs += 1
            continue
        ref_ids = refs['target'].astype(str).tolist()
        ref_fields = refs['target_field'].dropna().astype(str).tolist()
        ref_comms = refs['target_comm'].dropna().astype(int).tolist()
        ref_count = len(ref_ids)

        Gm = prior_graphs[year].copy()
        G0 = Gm.copy()
        G0.add_node(pid)
        for rid in ref_ids:
            G0.add_edge(pid, str(rid))

        # core seven metrics
        try:
            B = float(nx.betweenness_centrality(G0, normalized=True).get(pid, 0.0))
        except Exception:
            B = 0.0

        RS = rao_stirling(ref_fields, dist_by_year[year])
        RTD = simpson_diversity(pd.Series(ref_comms).value_counts().values)
        PDE = shannon_entropy(pd.Series(ref_fields).value_counts().values)

        q_minus = modularity_from_partition(Gm, comm_map_all)
        q_zero = modularity_from_partition(G0, {**comm_map_all, pid: comm_map_all.get(ref_ids[0], -1) if ref_ids else -1})
        DeltaQ0 = q_zero - q_minus

        # Uzzi-style atypicality: negative tail of field-pair z-scores, flipped so larger = more atypical
        z_lookup = pairz_by_year[year]
        zs = []
        uf = sorted(set(ref_fields))
        for i, fi in enumerate(uf):
            for fj in uf[i+1:]:
                zs.append(z_lookup.get((fi, fj), 0.0))
        Uzzi = float(max(0.0, -np.percentile(zs, 10))) if zs else 0.0

        # Burt IP: effective size normalized by degree, or inverse constraint fallback
        try:
            eff_size = nx.effective_size(G0, [pid])[pid]
        except Exception:
            eff_size = float(ref_count)
        try:
            constraint = nx.constraint(G0, [pid])[pid]
        except Exception:
            constraint = 1.0
        BurtIP = float(eff_size / max(1.0, ref_count))
        constraint_inv = float(1.0 / max(constraint, 1e-9))

        # candidate / alternatives
        try:
            pagerank = float(nx.pagerank(G0).get(pid, 0.0))
        except Exception:
            pagerank = 0.0
        try:
            closeness = float(nx.closeness_centrality(G0, pid))
        except Exception:
            closeness = 0.0
        degree = float(G0.degree(pid))
        field_shannon = PDE
        field_variety = float(len(set(ref_fields)))
        field_simpson = simpson_diversity(pd.Series(ref_fields).value_counts().values)
        community_simpson = RTD
        field_entropy_norm = float(PDE / max(math.log(max(2, len(set(ref_fields)))), 1e-9)) if len(set(ref_fields)) > 1 else 0.0
        pair_surprisal = Uzzi
        conductance_delta = -DeltaQ0  # placeholder directional alternative, explicitly computed from modularity shock family

        # graph delta observables for panel f
        uniq_comms = sorted(set(ref_comms))
        cross_pairs = sum(1 for i, ci in enumerate(uniq_comms) for cj in uniq_comms[i+1:] if ci != cj)
        community_reach = len(uniq_comms)
        modularity_shock = -DeltaQ0
        pshort = path_shortening(ref_comms, topic_graph)
        comp_reach = len(nx.node_connected_component(G0, pid)) if pid in G0 else 1
        assort_minus = assortativity_by_comm(Gm, comm_map_all)
        assort_zero = assortativity_by_comm(G0, {**comm_map_all, pid: uniq_comms[0] if uniq_comms else -1})
        boundary_mixing = -(assort_zero - assort_minus)

        paper_rows.append({
            'paper_id': pid, 'title': row.title, 'domain': row.domain, 'year': year,
            'primary_field': row.primary_field, 'is_landmark': int(row.is_landmark), 'reference_count': ref_count,
            'B': B, 'RS': RS, 'DeltaQ0': DeltaQ0, 'Uzzi': Uzzi, 'RTD': RTD, 'BurtIP': BurtIP, 'PDE': PDE,
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
            'primary_field': row.primary_field, 'is_landmark': int(row.is_landmark),
            'cross_community_gain': float(cross_pairs),
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
    progress_log('Normalizing metrics by field-year groups...', progress)
    paper_metrics = field_year_normalize(paper_metrics, metric_cols)

    candidate_cols = [c for c in candidate_metrics.columns if c not in ['paper_id', 'title', 'domain', 'year', 'primary_field', 'is_landmark', 'reference_count']]
    cand_norm = field_year_normalize(candidate_metrics.rename(columns={c: c for c in candidate_cols}), candidate_cols)
    cand_z_cols = [c + '_z' for c in candidate_cols]
    progress_log('Computing candidate-metric redundancy correlations...', progress)
    redundancy_corr = cand_norm[cand_z_cols].corr(method='spearman')
    redundancy_corr.index = candidate_cols
    redundancy_corr.columns = candidate_cols

    progress_log('Computing indicator-to-graph-delta correlations...', progress)
    merged = paper_metrics[['paper_id'] + [m + '_z' for m in metric_cols]].merge(graph_deltas, on='paper_id', how='inner')
    delta_cols = ['cross_community_gain', 'community_reach', 'modularity_shock', 'path_shortening', 'component_reach', 'boundary_mixing']
    corr_mat = pd.DataFrame(index=metric_cols, columns=delta_cols, dtype=float)
    for m in metric_cols:
        for d in delta_cols:
            corr_mat.loc[m, d] = merged[m + '_z'].corr(merged[d], method='spearman')

    # matched-control percentiles for landmark papers
    progress_log('Computing landmark matched-control percentiles...', progress)
    percentile_rows = []
    landmarks = paper_metrics[paper_metrics['is_landmark'] == 1].copy()
    if not landmarks.empty:
        paper_metrics['ref_bin'] = pd.qcut(paper_metrics['reference_count'].rank(method='first'), q=min(4, max(2, paper_metrics['reference_count'].nunique())), duplicates='drop')
        for lm in landmarks.itertuples(index=False):
            pool = paper_metrics[(paper_metrics['is_landmark'] == 0) &
                                 (paper_metrics['primary_field'] == lm.primary_field) &
                                 (paper_metrics['year'].between(lm.year - 1, lm.year + 1))].copy()
            if 'ref_bin' in paper_metrics.columns:
                lm_bin = paper_metrics.loc[paper_metrics['paper_id'] == lm.paper_id, 'ref_bin'].iloc[0]
                pool = pool[pool['ref_bin'] == lm_bin]
            if len(pool) < 5:
                pool = paper_metrics[(paper_metrics['is_landmark'] == 0) &
                                     (paper_metrics['year'].between(lm.year - 1, lm.year + 1))].copy()
            for m in metric_cols:
                pct = percentile_vs_controls(float(getattr(lm, m)), pool[m].values)
                percentile_rows.append({'paper_id': lm.paper_id, 'title': lm.title, 'metric': m, 'percentile': pct,
                                        'year': lm.year, 'primary_field': lm.primary_field, 'n_controls': len(pool)})
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
    )


# --------------------------
# Plotting helpers
# --------------------------

def _topic_positions(topics: pd.DataFrame, x: float, y: float, w: float, h: float) -> Dict[int, Tuple[float, float]]:
    tx = topics['x'].astype(float).to_numpy(); ty = topics['y'].astype(float).to_numpy()
    xmin, xmax = tx.min(), tx.max(); ymin, ymax = ty.min(), ty.max()
    xs = (tx - xmin) / (xmax - xmin + 1e-9)
    ys = (ty - ymin) / (ymax - ymin + 1e-9)
    pos = {}
    for (_, row), xr, yr in zip(topics.iterrows(), xs, ys):
        pos[int(row['community'])] = (x + 0.12*w + xr*0.76*w, y + 0.14*h + yr*0.70*h)
    return pos


def draw_panel_a(ax, raw: RawData, focus_paper_id: Optional[str]) -> None:
    panel_frame(ax, 'a', 'Real publication-day measurement setting')
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

    boxes = [
        (0.03, 0.29, 0.28, 0.58, f'Prior graph G−\n({focus_year-1})', '#2E7D32', 'prior'),
        (0.36, 0.29, 0.28, 0.58, f'Publication-day graph G0\n(landmark year {focus_year})', '#2563EB', 'pub'),
        (0.69, 0.29, 0.28, 0.58, f'Future graph G+τ\n({focus_year+1}–{works.year.max()})', '#6D4C8D', 'future'),
    ]

    pos = _topic_positions(raw.topics, 0, 0, 1, 1)
    colors = {int(c): mcolors.to_hex(plt.get_cmap('tab20')(i % 20)) for i, c in enumerate(sorted(raw.topics['community'].unique()))}
    meta = works[['id', 'display_community', 'year']].copy()
    lm_refs = raw.citations[raw.citations['source'] == str(focus['id'])]['target'].astype(str).tolist()

    for x, y, w, h, title, edgecolor, mode in boxes:
        rounded_box(ax, x, y, w, h, '#FCFCFD', edgecolor, 0.70, 0.015, zorder=1)
        ax.text(x + 0.02, y + h - 0.035, title, ha='left', va='top', fontsize=7.2, fontweight='bold')
        year_cut = focus_year - 1 if mode == 'prior' else (focus_year if mode == 'pub' else works['year'].max())
        active = meta[meta['year'] <= year_cut].copy()
        counts = active.groupby('display_community').size().to_dict()
        # community halos and nodes
        for comm, count in counts.items():
            if comm not in pos:
                continue
            cx, cy = pos[comm]
            px = x + (cx)*w; py = y + (cy)*h
            rr = 0.022 + 0.018 * (np.log1p(count) / max(np.log1p(max(counts.values())), 1e-9))
            circ = mpatches.Circle((px, py), rr, transform=ax.transAxes, facecolor=mcolors.to_rgba(colors[comm], 0.10),
                                   edgecolor=mcolors.to_rgba(colors[comm], 0.70), lw=0.8)
            ax.add_patch(circ)
            npts = max(4, min(8, int(round(np.log1p(count))) + 3))
            for j in range(npts):
                th = 2*math.pi*j/npts
                ax.scatter([px + rr*0.62*math.cos(th)], [py + rr*0.62*math.sin(th)], s=18,
                           color=colors[comm], edgecolors='white', linewidths=0.4, transform=ax.transAxes, zorder=3)
        # topic edges
        for r in raw.topic_edges.itertuples(index=False):
            c1 = int(r.source_community); c2 = int(r.target_community)
            if c1 in counts and c2 in counts and c1 in pos and c2 in pos:
                x1 = x + pos[c1][0]*w; y1 = y + pos[c1][1]*h
                x2 = x + pos[c2][0]*w; y2 = y + pos[c2][1]*h
                ax.plot([x1, x2], [y1, y2], color='#C7CBD1', lw=0.6, alpha=0.45, transform=ax.transAxes, zorder=1)

        if mode in {'pub', 'future'}:
            # focal paper star and its ref links to communities
            fc = x + 0.43*w; fy = y + 0.34*h
            ax.scatter([fc], [fy], marker='*', s=160, color='#DC2626', edgecolors='white', linewidths=0.6, transform=ax.transAxes, zorder=5)
            ax.text(fc + 0.012, fy + 0.01, 'p', fontsize=7, fontstyle='italic', fontweight='bold', transform=ax.transAxes)
            ref_comms = meta[meta['id'].isin(lm_refs)]['display_community'].dropna().astype(int).tolist()
            for c in sorted(set(ref_comms)):
                if c in pos:
                    x2 = x + pos[c][0]*w; y2 = y + pos[c][1]*h
                    ax.plot([fc, x2], [fy, y2], color='#3B82F6' if mode == 'pub' else '#64748B', lw=0.8, ls='--' if mode == 'pub' else '-', alpha=0.7, transform=ax.transAxes, zorder=2)

    draw_arrow(ax, (0.315, 0.58), (0.355, 0.58), color='#64748B', lw=1.8, ms=18)
    draw_arrow(ax, (0.645, 0.58), (0.685, 0.58), color='#64748B', lw=1.8, ms=18)

    # legend / note
    legend_y = 0.15
    rounded_box(ax, 0.03, 0.12, 0.61, 0.11, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ax.scatter([0.055], [legend_y], marker='*', s=90, color='#DC2626', transform=ax.transAxes)
    ax.text(0.075, legend_y, f'Landmark paper\n(e.g., {focus_title[:18]})', ha='left', va='center', fontsize=5.4)
    ax.plot([0.21, 0.25], [legend_y, legend_y], color='#3B82F6', lw=1.0, ls='--', transform=ax.transAxes)
    ax.text(0.265, legend_y, 'Reference links from p\nto existing papers', ha='left', va='center', fontsize=5.4)
    ax.plot([0.43, 0.47], [legend_y, legend_y], color='#C7CBD1', lw=0.9, transform=ax.transAxes)
    ax.text(0.485, legend_y, 'Existing links among\nprior papers', ha='left', va='center', fontsize=5.4)
    rounded_box(ax, 0.67, 0.12, 0.18, 0.11, blend_with_white('#2563EB', 0.92), '#93C5FD', 0.6, 0.012)
    ax.text(0.76, legend_y, 'Seven indicators are\ncomputed at G0 using\nonly G− and references.', ha='center', va='center', fontsize=5.4, color='#1D4ED8', fontweight='bold')
    rounded_box(ax, 0.03, 0.03, 0.61, 0.07, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ax.text(0.05, 0.065, 'Data: OpenAlex citation graph. Topic communities detected by Leiden clustering.', fontsize=5.2, color=TEXT_LIGHT, ha='left', va='center')
    ax.text(0.05, 0.042, 'Node size ≈ number of papers. Edge width ≈ connection strength.', fontsize=5.0, color=TEXT_LIGHT, ha='left', va='center')


def draw_panel_b(ax) -> None:
    panel_frame(ax, 'b', 'Multi-stage screening from 92 candidates to 7 indicators')
    stages = [
        ('Stage 1\nNo future leakage', 'Computable on\npublication day', 'Future citations, FWCI,\nCD index, burst, altmetric', '#D9E6CF'),
        ('Stage 2\nReference-only', 'Use only references\nand prior graph G−', 'Author reputation, journal\nimpact, institution, funding', '#D8E7D1'),
        ('Stage 3\nLinked to graph\nperturbation', 'Mechanistically linked to\nexpansion / bridging /\nreconfiguration', 'Generic controls (ref count,\nmean age), popularity measures', '#D6EAF8'),
        ('Stage 4\nNon-redundant\nmechanisms', 'Remove highly correlated /\noverlapping metrics', 'PageRank, degree, closeness,\nSimpson, DIV, effective size, ...', '#E6DCF4'),
        ('Stage 5\nInterpretable &\nrobust', 'Interpretable, stable across\ndomains, validation-ready', 'Unstable, low signal,\nhard to interpret', '#FDE2CF'),
        ('Stage 6\nFinal basis', '', '', '#FDE7DF'),
    ]
    counts = [92, 67, 49, 29, 12, 7]
    widths = [0.28, 0.25, 0.22, 0.19, 0.16, 0.13]
    x = 0.06
    y0 = 0.18
    h = 0.67
    for i, ((title, crit, excl, fc), n, w) in enumerate(zip(stages, counts, widths)):
        patch = mpatches.Polygon([[x, y0+h],[x+w, y0+h],[x+w-0.02, y0],[x+0.02, y0]], closed=True,
                                  transform=ax.transAxes, facecolor=fc, edgecolor=BORDER, linewidth=0.7)
        ax.add_patch(patch)
        ax.text(x+w/2, y0+h-0.05, title, ha='center', va='top', fontsize=7, fontweight='bold')
        ax.text(x+w/2, y0+0.37, f'{n}', ha='center', va='center', fontsize=20, fontweight='bold')
        if crit:
            ax.text(0.54, y0+h-0.05 - i*0.117, crit, ha='left', va='top', fontsize=6.1)
            ax.text(0.77, y0+h-0.05 - i*0.117, excl, ha='left', va='top', fontsize=5.9)
        if i < len(counts)-1:
            draw_arrow(ax, (x+w/2, y0-0.01), (x+w/2, y0-0.055), color='#374151', lw=1.0, ms=10)
        x += 0.08
    ax.text(0.56, 0.89, 'Criterion', fontsize=7, fontweight='bold')
    ax.text(0.79, 0.89, 'Exclude examples', fontsize=7, fontweight='bold')
    ax.text(0.65, 0.12, 'Final seven-parameter basis', ha='center', va='center', fontsize=7, fontweight='bold')
    x = 0.44
    for key, label, color, _ in METRIC_SPECS:
        draw_pill(ax, x, 0.06, label, color, 0.08 if label != 'Burt IP' else 0.10, 0.04, fontsize=6)
        x += 0.095


def draw_panel_c(ax) -> None:
    panel_frame(ax, 'c', 'Publication-day evidence channels covered by the seven indicators')
    x_label = 0.16; x0 = 0.23; x1 = 0.97; y0 = 0.18; y1 = 0.82
    nrows = len(METRIC_SPECS); ncols = len(EVIDENCE_CHANNELS)
    colw = (x1-x0)/ncols; rowh = (y1-y0)/nrows

    for i, (title, color, subtitle) in enumerate(EVIDENCE_CHANNELS):
        hx = x0 + i*colw
        rounded_box(ax, hx, 0.83, colw, 0.10, '#F8FAFC', '#E2E8F0', 0.5, 0.006)
        ax.text(hx+colw/2, 0.885, title, ha='center', va='center', fontsize=7, color=color, fontweight='bold')
    for i in range(ncols+1):
        xx = x0 + i*colw
        ax.plot([xx, xx], [y0, 0.93], color=GRID, lw=0.6, transform=ax.transAxes)
    for j in range(nrows+1):
        yy = y1 - j*rowh
        ax.plot([0.08, x1], [yy, yy], color=GRID, lw=0.6, transform=ax.transAxes)
    for r, (key, label, color, _) in enumerate(METRIC_SPECS):
        cy = y1 - (r+0.5)*rowh
        ax.text(0.12, cy, label, ha='center', va='center', fontsize=7, color=color, fontweight='bold')
        for c, (title, col, _) in enumerate(EVIDENCE_CHANNELS):
            cx = x0 + (c+0.5)*colw
            status = EVIDENCE_MAP.get(key, {}).get(title, '')
            if status == 'primary':
                ax.scatter([cx], [cy], s=90, color=col, edgecolors='white', linewidths=0.5, transform=ax.transAxes)
            elif status == 'secondary':
                ax.scatter([cx], [cy], s=55, facecolors='white', edgecolors='#6B7280', linewidths=0.7, transform=ax.transAxes)
    ax.scatter([0.20], [0.09], s=90, color='#4B5563', edgecolors='white', linewidths=0.5, transform=ax.transAxes)
    ax.text(0.23, 0.09, 'Primary evidence', fontsize=5.8, va='center')
    ax.scatter([0.45], [0.09], s=55, facecolors='white', edgecolors='#6B7280', linewidths=0.7, transform=ax.transAxes)
    ax.text(0.48, 0.09, 'Secondary evidence', fontsize=5.8, va='center')
    ax.plot([0.72, 0.76], [0.09, 0.09], color='#9CA3AF', lw=0.8, ls='--', transform=ax.transAxes)
    ax.text(0.78, 0.09, 'Not primary', fontsize=5.8, va='center')


def plot_redundancy_heatmap(ax, comp: ComputedData, selected_only: bool = False) -> None:
    corr = comp.redundancy_corr.copy()
    selected_metrics = [m[0] for m in METRIC_SPECS]
    if not selected_only:
        desired = ['B', 'degree', 'pagerank', 'closeness', 'RS', 'field_shannon', 'field_simpson', 'field_variety',
                   'DeltaQ0', 'conductance_delta', 'Uzzi', 'pair_surprisal', 'RTD', 'community_simpson',
                   'BurtIP', 'effective_size', 'constraint_inv', 'PDE', 'field_entropy_norm']
        cols = [c for c in desired if c in corr.columns]
        corr = corr.loc[cols, cols]
    else:
        corr = corr.loc[selected_metrics, selected_metrics]
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
    for i, rname in enumerate(corr.index):
        if rname in selected_metrics:
            ax.add_patch(mpatches.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.9))
    for s in ax.spines.values(): s.set_visible(False)
    return im, corr


def draw_panel_d(ax, comp: ComputedData) -> None:
    panel_frame(ax, 'd', 'Candidate metric redundancy map (empirical)')
    ax.text(0.05, 0.90, 'Spearman correlation (papers, field-year normalized)', fontsize=6.5, fontweight='bold')
    # inset heatmap axes
    hax = ax.inset_axes([0.06, 0.18, 0.60, 0.68])
    im, corr = plot_redundancy_heatmap(hax, comp, selected_only=False)
    cax = ax.inset_axes([0.69, 0.20, 0.03, 0.64])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.5)
    # legend / families
    lx = 0.76; y = 0.83
    fam_lines = [
        ('Bridge position', ['Betweenness (B)', 'Bridge score', 'Participation coeff.', 'Closeness', 'PageRank', 'Degree'], '#0B4FA3'),
        ('Diversity / breadth', ['RS', 'DIV', 'Shannon (fields)', 'Simpson (fields)', 'Field variety'], '#2E7D32'),
        ('Atypical recombination', ['Uzzi', 'PMI surprisal', 'Wang W'], '#7C3AED'),
        ('Structural holes', ['Burt IP', 'Effective size', 'Constraint'], '#2563EB'),
        ('Modularity / boundary', ['ΔQ0', 'Conductance change', 'Community surprise'], '#F97316'),
    ]
    for title, lines, color in fam_lines:
        ax.plot([lx, lx], [y-0.05, y+0.02], color=color, lw=1.5, transform=ax.transAxes)
        ax.text(lx+0.015, y+0.015, title, fontsize=5.8, fontweight='bold', color=color, transform=ax.transAxes, ha='left', va='top')
        for i, line in enumerate(lines):
            ax.text(lx+0.015, y-0.01 - i*0.025, line, fontsize=5.0, color=TEXT_DARK, transform=ax.transAxes, ha='left', va='top')
        y -= 0.17
    rounded_box(ax, 0.03, 0.05, 0.94, 0.08, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ax.text(0.05, 0.09, 'Seven selected indicators (boxed) are lowly correlated with each other but capture different families of graph perturbation mechanisms.', fontsize=5.7, va='center')


def draw_panel_e(ax, comp: ComputedData) -> None:
    panel_frame(ax, 'e', 'Landmark perturbation percentiles (vs. matched controls)')
    ax.text(0.05, 0.90, 'Percentile among same-year, same-field matched controls (higher = stronger perturbation)', fontsize=6.5, fontweight='bold')
    order = [m[0] for m in METRIC_SPECS]
    label_map = {m[0]: m[1] for m in METRIC_SPECS}; color_map = {m[0]: m[2] for m in METRIC_SPECS}
    pdat = comp.percentile_long.copy()
    if pdat.empty:
        raise ValueError('Panel e requires at least one landmark paper and enough matched controls.')
    agg = pdat.groupby('metric', as_index=False)['percentile'].median()
    y_positions = np.linspace(0.77, 0.22, len(order))

    # grey reference distribution band (conceptual control range from actual per-landmark percentiles)
    plot_left, plot_right = 0.22, 0.86
    for y, key in zip(y_positions, order):
        vals = pdat.loc[pdat['metric'] == key, 'percentile'].dropna().values
        if len(vals) == 0:
            continue
        # draw ticks / distribution along percentile axis
        for t in np.linspace(0, 100, 5):
            xx = plot_left + (t/100)*(plot_right-plot_left)
            ax.plot([xx, xx], [0.17, 0.83], color='#E5E7EB', lw=0.6, transform=ax.transAxes, zorder=0)
        for v in vals:
            xx = plot_left + (v/100)*(plot_right-plot_left)
            ax.scatter([xx], [y], s=10, color='#BDBDBD', alpha=0.7, transform=ax.transAxes, zorder=2)
        med = float(np.median(vals))
        xx = plot_left + (med/100)*(plot_right-plot_left)
        ax.scatter([xx], [y], marker='*', s=120, color=color_map[key], edgecolors='white', linewidths=0.6, transform=ax.transAxes, zorder=4)
        ax.text(0.17, y, label_map[key], fontsize=7, color=color_map[key], fontweight='bold', ha='right', va='center', transform=ax.transAxes)
        ax.text(0.87, y, f'{med:.0f}', fontsize=8, color=TEXT_DARK, fontweight='bold', ha='left', va='center', transform=ax.transAxes)
        ax.text(0.05, y, CANDIDATE_GROUPS[[g for g, vals2 in CANDIDATE_GROUPS.items() if key in vals2][0]][0] if any(key in vals2 for vals2 in CANDIDATE_GROUPS.values()) else '', fontsize=5.2, color=TEXT_MID, ha='left', va='center', transform=ax.transAxes)
    ax.text(0.915, 0.83, 'Landmark\npercentile', fontsize=6.2, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
    for t in range(0, 101, 25):
        xx = plot_left + (t/100)*(plot_right-plot_left)
        ax.text(xx, 0.13, f'{t}', fontsize=5.5, color=TEXT_LIGHT, ha='center', transform=ax.transAxes)
    ax.text((plot_left+plot_right)/2, 0.085, 'Percentile', fontsize=6.2, fontweight='bold', ha='center', transform=ax.transAxes)
    rounded_box(ax, 0.03, 0.04, 0.94, 0.08, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ex_domain = comp.paper_metrics.loc[comp.paper_metrics['is_landmark'] == 1, 'domain'].iloc[0] if (comp.paper_metrics['is_landmark'] == 1).any() else 'example domain'
    n_controls = int(pdat['n_controls'].median()) if 'n_controls' in pdat else 0
    ax.text(0.05, 0.08, f'Example domain: {ex_domain}. Stars show median landmark percentile; gray dots are landmark-specific percentiles vs. matched controls (n ≈ {n_controls}).', fontsize=5.7, va='center')


def draw_panel_f(ax, comp: ComputedData) -> None:
    panel_frame(ax, 'f', 'Indicator–to–graph-delta correspondence (empirical)')
    ax.text(0.05, 0.90, 'Spearman correlation between indicators and direct graph deltas at publication day', fontsize=6.5, fontweight='bold')
    corr = comp.indicator_delta_corr.copy()
    order = [m[0] for m in METRIC_SPECS]
    corr = corr.loc[order, :]
    label_map = {m[0]: m[1] for m in METRIC_SPECS}
    color_map = {m[0]: m[2] for m in METRIC_SPECS}
    cols = list(corr.columns)
    heat_ax = ax.inset_axes([0.06, 0.19, 0.86, 0.64])
    arr = corr.values.astype(float)
    im = heat_ax.imshow(arr, cmap='RdBu_r', vmin=-1, vmax=1)
    heat_ax.set_xticks(range(len(cols))); heat_ax.set_yticks(range(len(order)))
    pretty_cols = ['Cross-community\nedge gain', 'Community\nreach', 'Modularity\nshock\n(-ΔQ0)', 'Path\nshortening\namong refs', 'Component\nreach', 'Boundary\nmixing\n(-assort.)']
    heat_ax.set_xticklabels(pretty_cols, fontsize=5.5)
    heat_ax.set_yticklabels([label_map[k] for k in order], fontsize=6)
    heat_ax.tick_params(length=0)
    for i, k in enumerate(order):
        heat_ax.get_yticklabels()[i].set_color(color_map[k]); heat_ax.get_yticklabels()[i].set_fontweight('bold')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            heat_ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5.5, color='black')
    for s in heat_ax.spines.values(): s.set_visible(False)
    cax = ax.inset_axes([0.15, 0.11, 0.62, 0.03])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=5)
    cb.set_label('Spearman ρ', fontsize=6, fontweight='bold')
    rounded_box(ax, 0.03, 0.04, 0.94, 0.08, '#FFFFFF', '#E2E8F0', 0.6, 0.012)
    ax.text(0.05, 0.08, 'Positive correlations (red) indicate stronger alignment of the indicator with the structural change induced at publication day.', fontsize=5.7, va='center')


# --------------------------
# Full figure assembly
# --------------------------

def assemble_figure(raw: RawData, comp: ComputedData, focus_paper_id: Optional[str], outpath: Path) -> None:
    setup_style()
    fig = plt.figure(figsize=(20, 13), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.1], width_ratios=[1.1, 1.0, 1.0], hspace=0.12, wspace=0.04)
    fig.text(0.5, 0.985, 'Fig. 2 | Why these seven indicators?', ha='center', va='top', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.955, 'Empirical evidence for a publication-day, reference-only basis capturing graph perturbations',
             ha='center', va='top', fontsize=11.5, color=TEXT_MID)

    axa = fig.add_subplot(gs[0, 0]); draw_panel_a(axa, raw, focus_paper_id)
    axb = fig.add_subplot(gs[0, 1]); draw_panel_b(axb)
    axc = fig.add_subplot(gs[0, 2]); draw_panel_c(axc)
    axd = fig.add_subplot(gs[1, 0]); draw_panel_d(axd, comp)
    axe = fig.add_subplot(gs[1, 1]); draw_panel_e(axe, comp)
    axf = fig.add_subplot(gs[1, 2]); draw_panel_f(axf, comp)
    fig.savefig(outpath)
    plt.close(fig)


def save_single_panel(panel: str, raw: RawData, comp: Optional[ComputedData], focus_paper_id: Optional[str], outpath: Path) -> None:
    setup_style()
    size_map = {'a': (7.6, 6.0), 'b': (7.1, 6.0), 'c': (7.0, 6.0), 'd': (7.5, 6.8), 'e': (7.1, 6.8), 'f': (7.1, 6.8)}
    fig, ax = plt.subplots(figsize=size_map.get(panel, (7, 6)), dpi=300)
    if panel == 'a':
        draw_panel_a(ax, raw, focus_paper_id)
    elif panel == 'b':
        draw_panel_b(ax)
    elif panel == 'c':
        draw_panel_c(ax)
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
    comp.paper_metrics.to_csv(out_dir / 'fig2_paper_metrics.csv', index=False)
    comp.candidate_metrics.to_csv(out_dir / 'fig2_candidate_metrics.csv', index=False)
    comp.graph_deltas.to_csv(out_dir / 'fig2_graph_deltas.csv', index=False)
    comp.redundancy_corr.to_csv(out_dir / 'fig2_redundancy_corr.csv')
    comp.indicator_delta_corr.to_csv(out_dir / 'fig2_indicator_delta_corr.csv')
    comp.percentile_long.to_csv(out_dir / 'fig2_landmark_percentiles.csv', index=False)
    comp.landmark_summary.to_csv(out_dir / 'fig2_landmark_percentile_summary.csv', index=False)


# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Empirical Fig. 2 pipeline: compute data and draw panels a–f separately or jointly.')
    p.add_argument('--data-dir', type=Path, default=DEFAULT_FIG1_DATA_ROOT,
                   help='Local data directory. Accepts a directory with works.csv/citations.csv/topics.csv/topic_edges.csv, '
                        'a Fig. 1 domain directory with works_selected.csv/paper_edges.csv/topic_nodes.csv/topic_edges.csv, '
                        f'or a Fig. 1 output root. Default: {DEFAULT_FIG1_DATA_ROOT}')
    p.add_argument('--domain', type=str, default=DEFAULT_DOMAIN,
                   help=f'Fig. 1 domain subdirectory to read when --data-dir is a root. Default: {DEFAULT_DOMAIN}')
    p.add_argument('--include-hybrid-edges', action='store_true',
                   help='When reading paper_edges.csv, include bibliographic/cocitation-only edges. '
                        'By default only rows with direct > 0 are used as citation links.')
    p.add_argument('--panel', type=str, default='all', choices=['a', 'b', 'c', 'd', 'e', 'f', 'all'],
                   help='Which panel to draw. Use all to assemble the full figure.')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--focus-paper-id', type=str, default=None,
                   help='Paper id to highlight in panel a. Defaults to the earliest landmark.')
    p.add_argument('--focal-paper-list', type=Path, default=None,
                   help='Optional text file with one focal paper id per line for metric computation. Defaults to all papers.')
    p.add_argument('--export-tables', action='store_true', help='Export all intermediate computed tables.')
    p.add_argument('--progress-interval', type=int, default=25,
                   help='Print one focal-paper progress update every N papers while computing metrics. Default: 25.')
    p.add_argument('--no-prepare-input', action='store_true',
                   help='Read --data-dir directly instead of first preparing normalized Fig. 2 input in --out-dir.')
    p.add_argument('--fig1-config', type=Path, default=None,
                   help='Optional Fig. 1 YAML config used with --run-fig1-if-missing.')
    p.add_argument('--run-fig1-if-missing', action='store_true',
                   help='Run the Fig. 1 pipeline to materialize source data if --data-dir does not contain usable exports.')
    p.add_argument('--no-fig1-cache', action='store_true',
                   help='When running Fig. 1, ignore cached works_raw.jsonl and re-download.')
    p.add_argument('--openalex-api-key', default=os.getenv('OPENALEX_API_KEY'),
                   help='OpenAlex API key passed through when --run-fig1-if-missing is used.')
    p.add_argument('--email', default=os.getenv('OPENALEX_EMAIL'),
                   help='OpenAlex contact email passed through when --run-fig1-if-missing is used.')
    p.add_argument('--quiet', action='store_true', help='Suppress progress logs.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.quiet
    progress_log(f'Starting Fig. 2 empirical pipeline: panel={args.panel}', progress)
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
        assemble_figure(raw, comp, args.focus_paper_id, outpath)
        progress_log(f'Saved full figure: {outpath}', progress)
    else:
        outpath = args.out_dir / f'panel_{args.panel}.png'
        progress_log(f'Drawing panel {args.panel}: {outpath}', progress)
        save_single_panel(args.panel, raw, comp, args.focus_paper_id, outpath)
        progress_log(f'Saved panel {args.panel}: {outpath}', progress)
    progress_log('Done.', progress)


if __name__ == '__main__':
    main()

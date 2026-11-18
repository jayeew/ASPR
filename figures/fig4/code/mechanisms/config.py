"""Seven explicit information conditions and existing Luna execution settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.config import Config as BaseConfig, ROOT

CONDITIONS = ('T', 'E', 'G', 'F', 'F_noJ', 'F_noM', 'F_noP')
COMPONENTS = {'GEAR': 'G', 'Graph branch': 'E', 'Joint': 'F_noJ',
              'Structural summaries': 'F_noM', 'Paper paths': 'F_noP'}
PATH_METRICS = {'direct_citation_neighbor_count', 'two_hop_neighbor_count', 'co_citation_neighbor_count'}
SIMILARITY_METRICS = {'nearest_prior_similarity', 'mean_top5_similarity'}
PATH_FIELDS = {'direct_citation', 'two_hop_path_count', 'shared_reference_count',
               'shared_reference_salton', 'parent_id_cache_hit'}
RISK_TYPES = ('scope_inflation', 'false_antecedence', 'semantic_causal', 'other_new_error',
              'unsupported_downgrade', 'wrong_modification', 'context_dropped', 'omission_only')


class Config(BaseConfig):
    source: Path = ROOT / 'outputs/fig3_reference/study'
    output: Path = ROOT / 'outputs/fig4_mechanisms'
    download_fulltext: bool = False
    material_max_chars: int = 256000
    request_max_chars: int = 320000
    request_max_bytes: int = 1280000


def condition_rows() -> list[dict[str, Any]]:
    slots = {'T': (0, 0, 0, 0, 0), 'E': (1, 0, 0, 0, 0), 'G': (0, 1, 1, 1, 1),
             'F': (1, 1, 1, 1, 1), 'F_noJ': (1, 1, 0, 1, 1),
             'F_noM': (1, 1, 1, 0, 1), 'F_noP': (1, 1, 1, 1, 0)}
    fields = ('gear_on', 'single_graph_on', 'joint_on', 'metrics_on', 'paths_on')
    return [{'condition_id': key, 'label': key.replace('_no', '−'), **dict(zip(fields, slots[key])),
             'replacement_stage': 'ordinary_text_analysis', 'report_length_target': None,
             'interpretation': 'fixed_existing_analysis_information_effect'} for key in CONDITIONS]

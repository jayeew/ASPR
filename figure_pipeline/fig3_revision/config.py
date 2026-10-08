"""Explicit experiment settings; no dependency or configuration comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('config.json')
METHODS = ('direct_a', 'eacl', 'reviewgrounder', 'gear', 'graph', 'fusion')
COMPARATORS = METHODS[:-1]
REUSED: tuple[str, ...] = ()
EFFORTS = {
    'claims': 'high', 'read': 'high', 'core': 'high', 'baseline': 'high', 'search': 'high', 'review_sections': 'medium',
    'tone': 'medium', 'checklist': 'xhigh', 'reference': 'xhigh', 'extract': 'high',
    'support': 'xhigh', 'novelty': 'xhigh', 'concerns': 'xhigh', 'clusters': 'high',
    'importance': 'xhigh', 'fusion': 'xhigh', 'preference': 'high', 'recheck': 'xhigh',
    'quality_checklist': 'xhigh', 'quality': 'xhigh', 'review_dynamics': 'xhigh',
}


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset: Path = ROOT/'outputs/fig3_reference/dataset'
    output: Path = ROOT/'outputs/fig3_reference/study'
    model: Literal['gpt-5.6-luna'] = 'gpt-5.6-luna'
    efforts: dict[str, str] = EFFORTS
    workers: int = 64
    cli_limit: int = 64
    memory_reserve_gib: float = 4.0
    task_memory_gib: float = 0.5
    workers_per_cpu: int = 4
    timeout_seconds: int = 600
    synthesis_timeout_seconds: int = 900
    cli_initial: int = 16
    network_limit: int = 16
    block_chars: int = 3000
    material_target_chars: int = 8000
    material_max_chars: int = 12000
    request_max_chars: int = 24000
    request_max_bytes: int = 64000
    executable: str = 'codex'
    repair_attempts: int = 1
    network_retries: int = 2
    unit_batch_size: int = 8
    review_chunk_chars: int = 3000
    max_queries: int = 4
    lexical_candidates: int = 20
    semantic_candidates: int = 30
    openalex_candidates: int = 20
    candidate_limit: int = 120
    rerank_limit: int = 100
    rerank_top_k: int = 10
    download_fulltext: bool = True
    download_max_bytes: int = 25*1024*1024
    fulltext_max_pages: int = 100
    fulltext_max_chars: int = 30000
    network_timeout_seconds: int = 40
    graph_assets: Path = ROOT/'data/claim_graph'
    embedding_model: Path = ROOT/'data/models/Qwen3-Embedding-4B'
    reranker_model: Path = Path('/home/jayee/models/OpenScholar_Reranker')
    device: str = 'cuda:0'
    bootstrap_repeats: int = 10000
    seed: int = 20260922
    recheck_papers: int = 20
    figure_width_mm: float = 420
    figure_height_mm: float = 760


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    config = Config.model_validate(json.loads(path.read_text(encoding='utf-8')))
    for name in ('dataset', 'output', 'graph_assets', 'embedding_model', 'reranker_model'):
        value = getattr(config, name)
        if not value.is_absolute():
            setattr(config, name, ROOT/value)
    return config

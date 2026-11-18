"""Read original manuscript/reviewer files, never old target analyses or reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .materials import blocks
from .storage import Store, roster, write


def resolve_source(config: Config, value: str, kind: str) -> Path:
    supplied = Path(value)
    copied = config.dataset/'sources'/kind/supplied.name
    return copied if copied.is_file() else supplied if supplied.is_absolute() else config.dataset/supplied


def prepare_paper(config: Config, row: dict[str, Any]) -> dict[str, Any]:
    manuscript_path = resolve_source(config, row['paper_path'], 'manuscripts')
    review_path = resolve_source(config, row['review_path'], 'peer_reviews')
    manuscript = manuscript_path.read_text(encoding='utf-8')
    reviews = review_path.read_text(encoding='utf-8')
    return {'paper': row, 'cutoff': row['publication_date'], 'manuscript': manuscript,
            'manuscript_path': str(manuscript_path), 'reviews': reviews, 'review_path': str(review_path),
            'manuscript_blocks': blocks(manuscript, 'M', config.block_chars),
            'review_blocks': blocks(reviews, 'R', config.block_chars), 'claims': []}


def prepare(config: Config, overwrite: bool = False, paper_ids: set[str] | None = None) -> None:
    for row in roster(config):
        if paper_ids is not None and row['paper_id'] not in paper_ids:
            continue
        store = Store(config)
        if not store.existing('prepare', row['paper_id'], overwrite=overwrite):
            store.put('prepare', row['paper_id'], prepare_paper(config, row))
    write(config.output/'inputs/roster.json', roster(config))


def generation_input(data: dict[str, Any]) -> dict[str, Any]:
    return {'title': data['paper']['title'], 'doi': data['paper'].get('doi', ''),
            'cutoff': data['cutoff'], 'manuscript': data['manuscript']}

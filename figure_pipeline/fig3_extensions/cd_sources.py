"""Read-only source locations and append-only CD tables."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/fig3_reference/study'
OUTPUT = ROOT / 'outputs/fig3_reference/extensions/cd'
METHODS = ('gear', 'graph', 'fusion')


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def stage_path(paper: str, stage: str, variant: str = 'papers') -> Path:
    area = ('inputs' if stage == 'prepare' else 'evidence' if stage in {'reference', 'evidence_pool'}
            else 'reports' if stage in {'gear', 'graph', 'full'} else 'annotations')
    return SOURCE / area / stage / variant / f'{paper}.json'


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Append absent deterministic keys; never replace an existing row."""
    existing = set()
    if path.exists():
        existing = {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        for row in rows:
            if row['id'] not in existing:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                existing.add(row['id'])


def normalize(text: str) -> tuple[str, list[int]]:
    chars, positions = [], []
    for position, char in enumerate(text):
        for normalized in unicodedata.normalize('NFKC', char):
            if not normalized.isspace():
                chars.append(normalized)
                positions.append(position)
    return ''.join(chars), positions


class Locator:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[str, list[int]]] = {}

    def locate(self, text: str, quote: str) -> dict:
        if not quote.strip():
            return {'state': 'empty', 'start': None, 'end': None, 'source_quote': None}
        position = text.find(quote)
        if position >= 0:
            return {'state': 'exact', 'start': position, 'end': position+len(quote), 'source_quote': quote}
        if text not in self.cache:
            self.cache[text] = normalize(text)
        normalized, offsets = self.cache[text]
        query, _ = normalize(quote)
        position = normalized.find(query) if query else -1
        if position >= 0:
            start, end = offsets[position], offsets[position+len(query)-1]+1
            return {'state': 'normalized_whitespace_unicode', 'start': start, 'end': end,
                    'source_quote': text[start:end]}
        return {'state': 'not_located', 'start': None, 'end': None, 'source_quote': None}


class PaperSources:
    def __init__(self, paper: str, evidence_rows: list[dict]) -> None:
        self.paper, self.rows = paper, evidence_rows
        self.data = read(stage_path(paper, 'prepare'))
        self.pool = read(stage_path(paper, 'evidence_pool'))
        self.blocks = {b['block_id']: b['text'] for b in self.pool['blocks']}
        self.owners: dict[str, list[dict]] = {}
        self.sources: dict[str, list[dict]] = {}
        for source in self.pool['sources']:
            self.sources.setdefault(source['source_id'], []).append(source)
            for block in source['block_ids']:
                self.owners.setdefault(block['block_id'], []).append(source)
        self.locator = Locator()
        self.seen: dict[tuple, str] = {}
        self.local_count = 0

    def add_text(self, namespace: str, quote: str, text: str, source_file: str,
                 pointer: str, metadata: dict | None = None) -> str:
        key = (namespace, source_file, pointer, quote)
        if key in self.seen:
            return self.seen[key]
        self.local_count += 1
        ident = f'{self.paper}/E{self.local_count:05d}'
        self.seen[key] = ident
        self.rows.append({'id': ident, 'paper_id': self.paper, 'namespace': namespace,
                          'source_file': source_file, 'json_pointer': pointer,
                          'requested_quote': quote, 'location': self.locator.locate(text, quote),
                          'metadata': metadata or {}})
        return ident

    def original(self, kind: str, quote: str) -> str:
        key = 'manuscript' if kind == 'manuscript' else 'reviews'
        return self.add_text(kind, quote, self.data[key], self.data['manuscript_path' if key == 'manuscript' else 'review_path'], '')

    def report(self, method: str, quote: str, body: str) -> str:
        return self.add_text('report/'+method, quote, body,
                             str(stage_path(self.paper, 'full' if method == 'fusion' else method)), '/body')

    def evidence(self, value: dict) -> list[str]:
        source_id, quote = value['source_id'], value.get('quote', '')
        if source_id == 'MANUSCRIPT':
            return [self.original('manuscript', quote)]
        if source_id in self.blocks:
            owners = self.owners.get(source_id, [])
            return [self.add_text('evidence_pool/block/'+source_id, quote, self.blocks[source_id],
                                 str(stage_path(self.paper, 'evidence_pool')), '/blocks/'+source_id,
                                 {'owners': [self.metadata(s) for s in owners],
                                  'evaluated_source_type': value.get('source_type'),
                                  'declared_location': value.get('location')})]
        sources = self.sources.get(source_id, [])
        result = []
        for i, source in enumerate(sources):
            text = ''.join(self.blocks[b['block_id']] for b in source['block_ids'])
            result.append(self.add_text('evidence_pool/source/'+source_id, quote, text,
                                       str(stage_path(self.paper, 'evidence_pool')), f'/sources/{source_id}/{i}',
                                       {**self.metadata(source), 'offset_basis': 'concatenated_source_blocks',
                                        'declared_location': value.get('location')}))
        if not sources:
            result.append(self.add_text('unresolved_source/'+source_id, quote, '',
                                       str(stage_path(self.paper, 'evidence_pool')), '',
                                       {'source_id': source_id, 'source_resolution': 'unresolved',
                                        'declared_location': value.get('location')}))
        return result

    def metadata(self, source: dict) -> dict:
        keys = ('source_id', 'source_type', 'title', 'doi', 'publication_date',
                'work_id', 'url', 'original_path', 'fulltext_status', 'retrieval_limitation')
        result = {k: source.get(k) for k in keys}
        result['block_ids'] = source.get('block_ids', [])
        result['target_doi_match'] = bool(source.get('doi') and
                                          source['doi'].lower() == str(self.data['paper'].get('doi', '')).lower())
        result['independent_antecedence_eligibility'] = 'pending_scope_date_and_version_review'
        return result

    def refs(self, values: list[dict]) -> list[str]:
        return list(dict.fromkeys(e for value in values for e in self.evidence(value)))


def chunks(text: str, size: int = 3500, overlap: int = 160) -> list[dict]:
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start+size)
        parts.append({'start': start, 'end': end, 'text': text[start:end]})
        if end == len(text):
            break
        start = end-overlap
    return parts or [{'start': 0, 'end': 0, 'text': ''}]

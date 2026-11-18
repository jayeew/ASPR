"""Small text blocks and evidence packets; ordinary identities, no fingerprints."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import Config


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def browser_notice(text: str) -> bool:
    return len(text) < 1000 and bool(re.search(
        r'JavaScript is disabled|enable JavaScript to proceed', text, re.I))


def blocks(text: str, prefix: str = 'T', size: int = 3000) -> list[dict[str, Any]]:
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cuts = [text.rfind(mark, start + size//2, end) for mark in ('\n', '. ', '。', '; ')]
            cut = max(cuts)
            if cut >= start + size//2:
                end = cut + 1
        result.append({'block_id': f'{prefix}{len(result)+1:05d}', 'start': start,
                       'end': end, 'text': text[start:end]})
        start = end
    return result


def catalog(sources: list[dict[str, Any]], size: int = 3000) -> dict[str, Any]:
    texts: dict[str, str] = {}
    records, chunks = [], []
    for source in sources:
        if browser_notice(source.get('passage', '')):
            source = {**source, 'passage': '', 'source_type': 'metadata_only',
                      'fulltext_status': 'browser_notice_only'}
        record = {k: v for k, v in source.items() if k not in {'passage', 'blocks'}}
        record['block_ids'] = []
        for part in blocks(source.get('passage', ''), size=size):
            text = part['text']
            if text not in texts:
                key = f'T{len(chunks)+1:06d}'
                texts[text] = key
                chunks.append({'block_id': key, 'text': text})
            record['block_ids'].append({'block_id': texts[text], 'start': part['start'], 'end': part['end']})
        records.append(record)
    return {'blocks': chunks, 'sources': records}


def terms(text: str) -> set[str]:
    words = set(re.findall(r'[a-z0-9]{2,}', text.lower()))
    chinese = ''.join(re.findall(r'[\u4e00-\u9fff]', text))
    words.update(chinese[i:i+2] for i in range(len(chinese)-1))
    return words


class PacketIndex:
    """Reuse lexical sets and locators without changing ranking or passage selection."""
    def __init__(self, index: dict[str, Any]) -> None:
        self.index = index
        self.tokens = {part['block_id']: terms(part['text']) for part in index['blocks']}
        self.references: dict[bool, dict[str, list[dict[str, Any]]]] = {}

    def refs(self, historical: bool) -> dict[str, list[dict[str, Any]]]:
        if historical in self.references:
            return self.references[historical]
        refs: dict[str, list[dict[str, Any]]] = {}
        for source in self.index['sources']:
            if historical and source.get('source_type') in {'review', 'graph_observation'}:
                continue
            for part in source['block_ids']:
                ref = {k: source.get(k) for k in ('source_id', 'source_type', 'title', 'publication_date',
                                                  'original_path', 'doi', 'fulltext_origin') if source.get(k) is not None}
                refs.setdefault(part['block_id'], []).append({**ref, 'start': part['start'], 'end': part['end']})
        self.references[historical] = refs
        return refs


def packet(index: dict[str, Any], query: Any, budget: int = 6500,
           historical: bool = False, prepared: PacketIndex | None = None) -> dict[str, Any]:
    prepared = prepared or PacketIndex(index)
    query_text = encoded(query)
    tokens = terms(query_text)
    refs = prepared.refs(historical)
    ranked = []
    for part in index['blocks']:
        key = part['block_id']
        if key not in refs:
            continue
        overlap = len(tokens & prepared.tokens[key])
        cited = any(r['source_id'] in query_text for r in refs[key])
        # Negative/limitation passages remain candidates, not only report-selected citations.
        ranked.append((overlap + 10*int(cited), key, part))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if historical:
        # Give relevant independent works room before additional target passages.
        manuscript = [row for row in ranked if all(r['source_type'] == 'manuscript' for r in refs[row[1]])]
        history = [row for row in ranked if any(r['source_type'] != 'manuscript' for r in refs[row[1]])]
        diversified, repeated, seen = [], [], set()
        for row in history:
            names = {r.get('doi') or r['source_id'] for r in refs[row[1]] if r['source_type'] != 'manuscript'}
            if names-seen:
                diversified.append(row)
                seen.update(names)
            else:
                repeated.append(row)
        ranked = manuscript[:1]+diversified+repeated+manuscript[1:]
    selected, used = [], 0
    for _, key, part in ranked:
        entry = {**part, 'provenance': refs[key]}
        cost = len(encoded(entry))
        if used + cost <= budget:
            selected.append(entry)
            used += cost
    return {'evidence_blocks': selected, 'coverage': {'available_blocks': len(ranked),
            'selected_blocks': len(selected), 'limitation': 'Selected passages, not exhaustive. '
            'Absence here does not establish absence in the full paper or literature.'}}


def fits(config: Config, payload: Any, instruction: str = '', schema: dict[str, Any] | None = None) -> bool:
    material = encoded(payload)
    total = instruction + material + encoded(schema or {})
    return (len(material) <= config.material_max_chars and len(total) <= config.request_max_chars
            and len(total.encode('utf-8')) <= config.request_max_bytes)


def batches(rows: list[Any], maximum: int = 4, chars: int = 5000) -> list[list[Any]]:
    result, current = [], []
    for row in rows:
        if current and (len(current) >= maximum or len(encoded(current + [row])) > chars):
            result.append(current)
            current = []
        current.append(row)
    if current:
        result.append(current)
    return result


def fragments(value: Any, size: int = 3000, path: str = '$', context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Split structured records without losing the parent source/object identity."""
    context = context or {}
    if len(encoded(value)) <= size:
        return [{'path': path, 'identity': context, 'value': value}]
    if isinstance(value, dict):
        identity = {**context, **{k: v for k, v in value.items() if k in {
            'source_id', 'claim_id', 'unit_id', 'core_id', 'concern_id', 'block_id', 'source_type', 'publication_date'} and isinstance(v, (str, int))}}
        return [part for key, child in value.items() for part in fragments(child, size, path+'.'+key, identity)]
    if isinstance(value, list):
        return [part for i, child in enumerate(value) for part in fragments(child, size, f'{path}[{i}]', context)]
    return [{'path': path, 'identity': context, **part} for part in blocks(str(value), size=size)]

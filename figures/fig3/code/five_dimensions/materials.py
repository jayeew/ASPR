"""Read-only adapters for existing method materials and neutral source packets."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import DAMAGE_ID, METHODS, OUTPUT, ROOT, STUDY
from .io import encoded, read, write

ORIGINAL_TYPES = {'manuscript', 'abstract', 'fulltext', 'metadata_only', 'reference_only'}
PRIVATE_KEYS = {'original_path', 'local_fulltext', 'paper_path', 'manuscript_path',
                'review_path', 'output_dir', 'work_dir', 'log_path'}
GRAPH_NOTICE = ('Recorded graph observations and generated node metadata. Node claim_text is not '
                'an original manuscript/historical quotation. Semantic and citation edges do not '
                'establish support, antecedence or causality. Use original scientific text for those judgments.')


def clean_structure(value: Any) -> Any:
    """Remove local location fields, not scientifically relevant citation-path facts."""
    if isinstance(value, dict):
        return {key: clean_structure(item) for key, item in value.items()
                if key not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [clean_structure(item) for item in value]
    return value


def paper_metadata(data: dict[str, Any], cohort: str) -> dict[str, Any]:
    source = data.get('paper', data)
    return {'paper_id': source['paper_id'], 'title': source['title'],
            'doi': source.get('doi', ''),
            'cutoff': data.get('cutoff', source.get('cutoff_date', source.get('publication_date', ''))),
            'journal': source.get('journal_name', source.get('venue', '')),
            'field': source.get('field_name', ''), 'cohort': cohort}


def historical_status(source: dict[str, Any], paper: dict[str, Any]) -> str:
    if source.get('source_type') in {'manuscript', 'graph_observation', 'historical_claim_summary'}:
        return 'not_an_independent_historical_original'
    doi = str(source.get('doi', '') or '').casefold().removeprefix('https://doi.org/')
    target = str(paper.get('doi', '') or '').casefold().removeprefix('https://doi.org/')
    if (doi and doi == target) or source.get('article_id') == paper['paper_id']:
        return 'target_paper_not_antecedent'
    date = str(source.get('publication_date', '') or '')
    cutoff = str(paper.get('cutoff', '') or '')
    if len(date) >= 10 and len(cutoff) >= 10:
        return 'eligible_date' if date[:10] <= cutoff[:10] else 'after_cutoff_not_antecedent'
    return 'date_not_established'


class SourceCollector:
    def __init__(self, paper: dict[str, Any]) -> None:
        self.paper = paper
        self.sources: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []
        self.excluded: list[dict[str, Any]] = []
        self.by_id: dict[str, list[tuple[str, str]]] = {}

    def add(self, source: dict[str, Any], origin: str, text: str | None = None) -> None:
        kind = str(source.get('source_type', ''))
        if kind not in ORIGINAL_TYPES | {'graph_observation', 'historical_claim_summary'}:
            self.excluded.append({'source_id': source.get('source_id'), 'source_type': kind,
                                  'reason': 'not_original_text_or_graph_fact', 'origin': origin})
            return
        body = text if text is not None else str(source.get('passage', source.get('text', '')) or '')
        if kind == 'graph_observation':
            body = encoded(clean_structure(json.loads(body)))
        original_id = str(source.get('source_id') or f'SOURCE{len(self.sources)+1:05d}')
        digest = hashlib.sha256(body.encode()).hexdigest()
        versions = self.by_id.setdefault(original_id, [])
        existing = next((ident for fingerprint, ident in versions if fingerprint == digest), None)
        source_id = existing or (original_id if not versions else f'{original_id}::SEG{len(versions)+1:03d}')
        provenance = {'source_id': source_id, 'original_source_id': original_id,
                      'origin': origin, 'text_sha256': digest, 'characters': len(body),
                      'source_metadata': source}
        self.provenance.append(provenance)
        if existing:
            return
        versions.append((digest, source_id))
        item = {'source_id': source_id, 'original_source_id': original_id,
                'source_type': kind, 'text': body,
                'historical_use': historical_status(source, self.paper)}
        for key in ('title', 'doi', 'publication_date', 'year', 'work_id', 'article_id'):
            if source.get(key) is not None:
                item[key] = source[key]
        if kind in {'graph_observation', 'historical_claim_summary'}:
            item['source_notice'] = GRAPH_NOTICE
        if kind in {'metadata_only', 'reference_only'}:
            item['source_notice'] = 'Bibliographic metadata only; not evidence of scientific findings.'
        self.sources.append(item)


def add_manuscript(collector: SourceCollector, manuscript: str, origin: Path) -> None:
    collector.add({'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                   'title': collector.paper['title'], 'doi': collector.paper.get('doi', '')},
                  str(origin), manuscript)


def add_native_graph(collector: SourceCollector, facts: dict[str, Any], origin: Path) -> None:
    for card in facts.get('cards', []):
        claim_id = card.get('claim', {}).get('claim_id', '')
        collector.add({'source_id': f'GRAPH:{claim_id}', 'source_type': 'graph_observation'},
                      str(origin), encoded(card))
    if facts.get('joint'):
        collector.add({'source_id': f'JOINT_GRAPH:{collector.paper["paper_id"]}',
                       'source_type': 'graph_observation'}, str(origin), encoded(facts['joint']))


def pool_text(source: dict[str, Any], blocks: dict[str, str]) -> str:
    pieces = sorted(source.get('block_ids', []), key=lambda row: row.get('start', 0))
    output, end = '', None
    for piece in pieces:
        text = blocks[piece['block_id']]
        start = piece.get('start')
        if end is not None and start != end:
            output += '\n\n'
        output += text
        end = piece.get('end')
    return output


def prepare_generation(output: Path, paper: dict[str, Any], prepared: dict[str, Any],
                       method: str, study: Path = STUDY) -> dict[str, Any]:
    paper_id = paper['paper_id']
    path = study / 'reports' / method / 'papers' / f'{paper_id}.json'
    report = read(path)
    collector = SourceCollector(paper)
    add_manuscript(collector, prepared['manuscript'], Path(prepared['manuscript_path']))
    for source in report.get('sources', []):
        collector.add(source, str(path))
    payload = {'paper': paper, 'sources': collector.sources, 'analysis': report['body']}
    write(output / 'inputs/generation' / method / f'{paper_id}.json', payload)
    previous = output / 'inputs/original_reports' / method / f'{paper_id}.md'
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text(report['body'], encoding='utf-8')
    audit = {'paper_id': paper_id, 'method': method, 'report_path': str(path),
             'structured_analysis_path': str(path), 'structured_analysis_keys':
             [key for key in ('analysis', 'analyses', 'joint_analysis', 'fusion_details') if key in report],
             'prior_report_sha256': hashlib.sha256(report['body'].encode()).hexdigest(),
             'source_count': len(collector.sources), 'sources': collector.provenance,
             'excluded': collector.excluded, 'input_characters': len(encoded(payload)),
             'source_characters': sum(len(source['text']) for source in collector.sources),
             'analysis_characters': len(report['body'])}
    write(output / 'inputs/audit/generation' / method / f'{paper_id}.json', audit)
    return {key: audit[key] for key in ('paper_id', 'method', 'source_count', 'input_characters',
                                       'source_characters', 'analysis_characters')}


def prepare_reference(output: Path, paper: dict[str, Any], prepared: dict[str, Any],
                      study: Path = STUDY) -> dict[str, Any]:
    paper_id = paper['paper_id']
    collector = SourceCollector(paper)
    add_manuscript(collector, prepared['manuscript'], Path(prepared['manuscript_path']))
    path = study / 'evidence/evidence_pool/papers' / f'{paper_id}.json'
    pool = read(path)
    blocks = {block['block_id']: block['text'] for block in pool['blocks']}
    for source in pool['sources']:
        # Native graph facts are loaded separately; the pool's graph interpretations are not used.
        if source.get('source_type') in ORIGINAL_TYPES - {'manuscript'}:
            collector.add(source, str(path), pool_text(source, blocks))
    graph_path = study / 'evidence/native_graph' / f'{paper_id}.json'
    if graph_path.exists():
        add_native_graph(collector, read(graph_path), graph_path)
    core_path = study / 'annotations/core/papers' / f'{paper_id}.json'
    cores = [{key: item[key] for key in ('core_id', 'description', 'manuscript_quote') if key in item}
             for item in read(core_path)['items']]
    payload = {'paper': paper, 'sources': collector.sources, 'core_contributions': cores}
    write(output / 'inputs/reference' / f'{paper_id}.json', payload)
    audit = {'paper_id': paper_id, 'source_count': len(collector.sources),
             'input_characters': len(encoded(payload)), 'sources': collector.provenance,
             'excluded': collector.excluded, 'core_source': str(core_path),
             'no_method_reports_or_reviewer_judgments': True}
    write(output / 'inputs/audit/reference' / f'{paper_id}.json', audit)
    return {key: audit[key] for key in ('paper_id', 'source_count', 'input_characters')}


def prepare_main_paper(path: Path, output: Path, study: Path = STUDY) -> dict[str, Any]:
    prepared = read(path)
    paper = paper_metadata(prepared, 'main')
    generated = [prepare_generation(output, paper, prepared, method, study) for method in METHODS]
    reference = prepare_reference(output, paper, prepared, study)
    return {'roster': {**paper, 'methods': list(METHODS)}, 'generation': generated,
            'reference': reference}


def add_damage_work(collector: SourceCollector, item: dict[str, Any], origin: str) -> None:
    payload, key = item['payload'], item['evidence_id']
    metadata = {name: payload[name] for name in ('title', 'doi', 'publication_date', 'year', 'work_id')
                if name in payload}
    if payload.get('abstract'):
        collector.add({'source_id': key + ':ABSTRACT', 'source_type': 'abstract', **metadata},
                      origin, payload['abstract'])
    for index, span in enumerate(payload.get('spans', []), 1):
        kind = 'fulltext' if 'fulltext' in str(span.get('source', '')) else 'abstract'
        collector.add({'source_id': f'{key}:P{index:02d}', 'source_type': kind, **metadata},
                      origin, str(span.get('text', '')))


def damage_sources(paper: dict[str, Any], directory: Path, report: dict[str, Any],
                   manuscript: dict[str, Any]) -> SourceCollector:
    collector = SourceCollector(paper)
    add_manuscript(collector, manuscript['markdown'], directory / 'shared/paper_ir.json')
    for source in report.get('references', []):
        copied = dict(source)
        if copied.get('source_type') == 'historical_claim':
            copied['source_type'] = 'historical_claim_summary'
        collector.add(copied, str(ROOT / 'outputs/fig6_damage_study/original_report' / f'{DAMAGE_ID}.json'))
    for path in sorted((directory / 'gear').glob('*/evidence_trace.jsonl')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            item = json.loads(line)
            if item['kind'] in {'retrieved_work', 'retrieved_work_fulltext'}:
                add_damage_work(collector, item, f'{path}:{number}')
    cards = []
    for path in sorted((directory / 'graph').glob('*/evidence_trace.jsonl')):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item['kind'] == 'graph_fact':
                cards.append(item['payload'])
    add_native_graph(collector, {'cards': cards, 'joint': read(directory / 'graph/joint/facts.json')},
                     directory / 'graph')
    return collector


def prepare_damage(output: Path) -> dict[str, Any]:
    directory = ROOT / 'outputs/fig6_damage_study/papers' / DAMAGE_ID
    paper = paper_metadata(read(directory / 'innovation_input.json'), 'damage')
    manuscript = read(directory / 'shared/paper_ir.json')
    report_path = ROOT / 'outputs/fig6_damage_study/original_report' / f'{DAMAGE_ID}.json'
    report = read(report_path)
    collector = damage_sources(paper, directory, report, manuscript)
    # The saved report's final paragraph is known to be truncated. Keep it in the attachment,
    # but do not use its incomplete conclusion as an analysis input.
    complete = report['body'].rsplit('\n\n', 1)[0]
    generation = {'paper': paper, 'sources': collector.sources, 'analysis': complete}
    write(output / 'inputs/generation/full' / f'{DAMAGE_ID}.json', generation)
    previous = output / 'inputs/original_reports/full' / f'{DAMAGE_ID}.md'
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text(report['body'], encoding='utf-8')
    claims = read(directory / 'shared/claims.json')['claims']
    span_map = {span['span_id']: span['text'] for span in manuscript['spans']}
    cores = [{'core_id': claim['claim_id'], 'description': claim['normalized_claim_text'],
              'manuscript_quotes': [span_map[ident] for ident in claim['support_span_ids'] if ident in span_map]}
             for claim in claims]
    reference = {'paper': paper, 'sources': collector.sources, 'core_contributions': cores}
    write(output / 'inputs/reference' / f'{DAMAGE_ID}.json', reference)
    audit = {'paper_id': DAMAGE_ID, 'method': 'full', 'report_path': str(report_path),
             'prior_report_truncated': True, 'generation_uses_complete_paragraphs_only': True,
             'prior_report_sha256': hashlib.sha256(report['body'].encode()).hexdigest(),
             'input_characters': len(encoded(generation)), 'source_count': len(collector.sources),
             'source_characters': sum(len(item['text']) for item in collector.sources),
             'analysis_characters': len(complete), 'sources': collector.provenance,
             'excluded': collector.excluded}
    write(output / 'inputs/audit/generation/full' / f'{DAMAGE_ID}.json', audit)
    write(output / 'inputs/audit/reference' / f'{DAMAGE_ID}.json',
          {'paper_id': DAMAGE_ID, 'sources': collector.provenance,
           'no_method_reports_or_reviewer_judgments': True,
           'core_source': str(directory / 'shared/claims.json')})
    return {'roster': {**paper, 'methods': ['full']},
            'generation': [{key: audit[key] for key in ('paper_id', 'method', 'input_characters',
                                                       'source_count', 'source_characters', 'analysis_characters')}],
            'reference': {'paper_id': DAMAGE_ID, 'source_count': len(collector.sources),
                          'input_characters': len(encoded(reference))}}


def distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {'n': len(values), 'min': ordered[0], 'median': ordered[len(values)//2],
            'p90': ordered[min(len(values)-1, int(.9*len(values)))], 'max': ordered[-1]}


def prepare_all(output: Path = OUTPUT) -> dict[str, Any]:
    paths = sorted((STUDY / 'inputs/prepare/papers').glob('*.json'))
    if len(paths) != 100:
        raise ValueError(f'Expected the fixed 100-paper cohort, found {len(paths)}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda path: prepare_main_paper(path, output), paths))
    records.append(prepare_damage(output))
    write(output / 'inputs/roster.json', [record['roster'] for record in records])
    generation = [row for record in records for row in record['generation']]
    references = [record['reference'] for record in records]
    summary = {'main_papers': 100, 'damage_papers': 1, 'generation_tasks': len(generation),
               'reference_tasks': len(references), 'methods': list(METHODS),
               'generation_size_by_method': {method: distribution([row['input_characters'] for row in generation
                                                                  if row['method'] == method]) for method in METHODS},
               'reference_size': distribution([row['input_characters'] for row in references]),
               'generation': generation, 'reference': references,
               'source_policy': 'No model calls, retrieval, reviewer judgments or evaluation answers in generation.'}
    write(output / 'inputs/material_manifest.json', summary)
    return summary

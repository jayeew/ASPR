"""Eight bounded forecasts, one consolidation, at most one additional attempt."""
from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from figure_pipeline.fig3_revision.client import Client
from figure_pipeline.fig3_revision.config import Config as BaseConfig
from .data import OUT, ROOT, compact, read, write

CATEGORIES = ('Method', 'Problem', 'Combination')


class Config(BaseConfig):
    model: str = 'gpt-6.1-sol'
    output: Path = OUT
    executable: str = str(ROOT / 'figure_pipeline/fig4_explanation_study/codex_session')
    efforts: dict[str, str] = {'forecast': 'medium', 'merge': 'medium'}
    material_max_chars: int = 448000
    request_max_chars: int = 480000
    request_max_bytes: int = 1800000
    timeout_seconds: int = 1800
    synthesis_timeout_seconds: int = 1800
    repair_attempts: int = 0


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Citation(Strict):
    source_id: str
    quote: str


class Label(Strict):
    category: Literal['Method', 'Problem', 'Combination']
    phrase: str


class Direction(Strict):
    direction: str
    graph_refs: list[str] = Field(min_length=1)
    graph_basis: str
    citations: list[Citation] = Field(min_length=2)
    future_extension: str
    labels: list[Label] = Field(min_length=1, max_length=6)


class Forecast(Strict):
    directions: list[Direction] = Field(max_length=12)
    limitations: str


class Retained(Strict):
    candidate_id: str
    labels: list[Label] = Field(min_length=1, max_length=6)


class Exclusion(Strict):
    candidate_id: str
    reason: str


class Consolidated(Strict):
    retained: list[Retained]
    excluded: list[Exclusion]
    limitations: str


FORECAST_PROMPT = '''Generate up to 12 specific, source-grounded FUTURE research directions from this
whole-graph packet. Analyze all group summary rows; select promising directions across distinct
scientific topics, not just the largest communities. These are prospective hypotheses from a
2023-2025 abstract-claim corpus, not calibrated predictions, validated novelty, or retrospective
forecast accuracy. Use only the supplied material; do not add knowledge from later papers.

Tables use group_columns/evidence_columns/edge_columns to define array rows (including edge_examples). C-prefixed groups are fixed
whole-period communities; U-prefixed groups organize unassigned claims by topic metadata and
are NOT discovered communities. annual rows correspond to years and annual_columns. Edge year
is the later claim's year; outgoing edges link that later group to strictly older eligible
claims. cross_group_fraction is the fraction of those edges ending in a DIFFERENT group,
including metadata buckets. Neighbor counts are undirected incident cross-group edge counts.
method_incident_edges counts incident edges with at least one METHOD endpoint (not necessarily
inside the focal group). method_neighbor_groups counts other groups on those edges. Edges mean
semantic similarity, NOT causal/support/antecedent relations or demonstrated method transfer.
Corpus/year coverage varies. Interpret paper SHARES alongside counts. No observed edge does not
establish a research gap. Existing whole-period community labels cannot establish community births.

For each direction cite existing group IDs and/or E-prefixed edge IDs in graph_refs. Explain
which supplied graph observations motivate it, and use at least TWO DIFFERENT source papers
from evidence. Citations use N-prefixed source IDs and exact contiguous quotes from their claim
strings. Explain a concrete FUTURE extension beyond what these claims already report. Mark it
as a proposal, not a fact or existing achievement. Avoid numerical claims not explicit in input.
When interpreting a particular edge, cite its endpoints. All graph_refs must occur in this packet.

Provide concise scientific phrases (ideally 3-6 words, <=55 characters) for applicable categories:
Method = a prospective method/technical route; Problem = a specific prospective research problem;
Combination = a plausible future combination of method, mechanism or scientific object.
Use English throughout, with accurate scientific qualifiers. Do not force all categories onto
every direction. Prefer reusable precise phrases where directions actually share a research
axis, but never fabricate repeated themes or generic words just to make a word cloud look varied.
Use a maximum of two phrases per applicable category. Avoid bare words like AI, retrieval,
multimodal, innovation. No quotas by field; return fewer directions if evidence is inadequate.'''

MERGE_PROMPT = '''Consolidate the supplied candidate FUTURE research directions, using supplied content
only. Keep concrete future extensions supported by their graph observations and at least two
source papers. Exclude duplicates, generic themes, scientifically unjustified extrapolations,
and restatements of demonstrated past results. Select a representative candidate ID when
several are the SAME proposed direction; do not count duplicates as independent predictions.
Do not create new directions, change source IDs, add evidence, or infer that evidence is verified.

For each retained candidate return canonical labels for applicable Method, Problem, Combination
categories. Normalize true synonyms consistently across candidates so occurrence counts are
meaningful. Preserve scientific object, scope and qualifiers; do not conflate unrelated methods
or diseases. Phrases ideally 3-6 words, <=55 characters. Up to two labels per category. Labels
must be supported by THAT candidate's proposal. Do not broaden labels solely to create repetition.
Every input candidate ID must appear EXACTLY ONCE, either retained or excluded with a reason.
Return no directions or scientific facts of your own. Word sizes will later count distinct
retained directions containing each phrase, not confidence or evidence strength.'''


def packet_rows(packet: dict[str, Any], field: str) -> list[dict[str, Any]]:
    columns = packet['group_columns' if field == 'groups' else 'evidence_columns']
    rows = [dict(zip(columns, row)) for row in packet[field]]
    if field == 'groups':
        for row in rows:
            row['edge_examples'] = [dict(zip(packet['edge_columns'], e)) for e in row['edge_examples']]
    return rows


def valid_candidates(packet: dict[str, Any], forecast: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = {r['id']: r for r in packet_rows(packet, 'evidence')}
    groups = packet_rows(packet, 'groups')
    group_ids = {g['id'] for g in groups}
    neighbor_ids = set(packet.get('neighbor_labels', {})) | {s['group'] for s in sources.values()}
    edges = {e['id']: e for g in groups for e in g['edge_examples']}
    accepted, rejected = [], []
    for i, candidate in enumerate(forecast['directions']):
        candidate = {**candidate, 'candidate_id': f'{packet["packet_id"]}-D{i+1:02d}', 'packet_id': packet['packet_id']}
        cites = candidate['citations']
        errors = []
        if any(c['source_id'] not in sources for c in cites):
            errors.append('unknown_source')
        elif any(not c['quote'].strip() or c['quote'] not in sources[c['source_id']]['claim'] for c in cites):
            errors.append('non_exact_quote')
        elif len({sources[c['source_id']]['paper_id'] for c in cites}) < 2:
            errors.append('fewer_than_two_papers')
        refs = set(candidate['graph_refs'])
        if not refs <= group_ids | neighbor_ids | set(edges):
            errors.append('unknown_graph_reference')
        cited = {c['source_id'] for c in cites}
        if any(not {edges[r]['earlier'], edges[r]['later']} <= cited for r in refs & set(edges)):
            errors.append('edge_endpoints_not_cited')
        if refs & group_ids and not any(sources[c]['group'] in refs for c in cited if c in sources):
            errors.append('group_not_linked_to_cited_claim')
        if errors:
            rejected.append({'candidate_id': candidate['candidate_id'], 'errors': errors})
        else:
            accepted.append(candidate)
    return accepted, rejected


def call_once(task: str, ident: str, instruction: str, payload: Any, schema: type[BaseModel]) -> dict[str, Any]:
    path = OUT / 'calls.jsonl'
    entries = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if len(entries) >= 10:
        raise RuntimeError('The authorized ten-call budget is exhausted; missing work needs discussion.')
    entry = {'attempt': len(entries) + 1, 'task': task, 'id': ident,
             'time': datetime.now(timezone.utc).isoformat(), 'input_chars': len(compact(payload))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(compact(entry) + '\n')
    print(f'CALL {entry["attempt"]}/10: {ident}, {entry["input_chars"]} input chars', flush=True)
    result = Client(Config(), context={'fig7_stage': ident}).call(task, instruction, payload, schema)
    write(OUT / 'predictions' / f'{ident}.json', result)
    print(f'COMPLETED {ident}', flush=True)
    return result


def obtain(task: str, ident: str, instruction: str, payload: Any, schema: type[BaseModel]) -> dict[str, Any]:
    path = OUT / 'predictions' / f'{ident}.json'
    if path.exists():
        return schema.model_validate(read(path)).model_dump()
    entries = [json.loads(s) for s in (OUT / 'calls.jsonl').read_text().splitlines()] if (OUT / 'calls.jsonl').exists() else []
    used_extra = len(entries) - len({e['id'] for e in entries})
    if any(e['id'] == ident for e in entries) and used_extra >= 1:
        raise RuntimeError('This missing stage has exhausted the single extra attempt; discuss recovery.')
    try:
        return call_once(task, ident, instruction, payload, schema)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        write(OUT / 'failures' / f'{ident}.json', {'error': str(exc)})
        if used_extra >= 1:
            raise
        print(f'Using the single additional attempt for {ident}: {type(exc).__name__}', flush=True)
        return call_once(task, ident, instruction, payload, schema)


def check_merge(candidates: list[dict[str, Any]], result: dict[str, Any]) -> None:
    expected = {c['candidate_id'] for c in candidates}
    ids = [r['candidate_id'] for r in result['retained'] + result['excluded']]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Merge must account for every candidate exactly once')


def frequencies(directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: dict[tuple[str, str], set[str]] = {}
    for direction in directions:
        for label in direction['labels']:
            phrase = ' '.join(label['phrase'].split())
            labels.setdefault((label['category'], phrase), set()).add(direction['candidate_id'])
    rows = [{'category': cat, 'phrase': phrase, 'frequency': len(ids), 'direction_ids': sorted(ids)}
            for (cat, phrase), ids in labels.items()]
    shown: set[tuple[str, str]] = set()
    for cat in CATEGORIES:
        ordered = sorted([r for r in rows if r['category'] == cat], key=lambda r: r['phrase'])
        random.Random(20261003 + CATEGORIES.index(cat)).shuffle(ordered)
        selected = sorted(ordered, key=lambda r: -r['frequency'])[:20]
        shown.update((r['category'], r['phrase']) for r in selected)
    maximum = max((r['frequency'] for r in rows if (r['category'], r['phrase']) in shown), default=1)
    for row in rows:
        row['displayed'] = (row['category'], row['phrase']) in shown
        row['font_size_pt'] = 8 + 10 * math.sqrt(row['frequency'] / maximum)
    return sorted(rows, key=lambda r: (CATEGORIES.index(r['category']), -r['frequency'], r['phrase']))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: compact(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def generate() -> None:
    candidates, rejected = [], []
    for i in range(1, 9):
        ident = f'P{i:02d}'
        packet = read(OUT / 'packets' / f'{ident}.json')
        forecast = obtain('forecast', ident, FORECAST_PROMPT, packet, Forecast)
        good, bad = valid_candidates(packet, forecast)
        candidates.extend(good)
        rejected.extend(bad)
    write(OUT / 'data/candidates.json', candidates)
    write(OUT / 'data/rejected_candidates.json', rejected)
    # Include the actual referenced records to let consolidation inspect the evidence.
    records = read(OUT / 'data/evidence.json')
    selected_sources = sorted({c['source_id'] for d in candidates for c in d['citations']})
    payload = {'candidates': candidates, 'sources': {k: records[k] for k in selected_sources}}
    result = obtain('merge', 'merge', MERGE_PROMPT, payload, Consolidated)
    try:
        check_merge(candidates, result)
    except ValueError as exc:
        result = call_once('merge', 'merge', MERGE_PROMPT + '\nRepair the candidate accounting: ' + str(exc),
                           payload, Consolidated)
        check_merge(candidates, result)
    by_id = {c['candidate_id']: c for c in candidates}
    final = [{**by_id[r['candidate_id']], 'labels': r['labels']} for r in result['retained']]
    write(OUT / 'data/directions.json', final)
    write(OUT / 'qa/source_validation.json', {
        'completed_forecast_packets': 8, 'source_checked_candidates': len(candidates),
        'source_check_exclusions': len(rejected), 'merge_exclusions': len(result['excluded']),
        'retained_directions': len(final), 'exact_claim_quotes_checked': True,
        'two_distinct_source_papers_checked': True, 'graph_reference_ids_checked': True,
        'scientific_truth_or_predictive_accuracy_validated': False})
    rows = frequencies(final)
    write(OUT / 'data/wordcloud_frequencies.json', rows)
    csv_write(OUT / 'tables/wordcloud_frequencies.csv', rows)
    csv_write(OUT / 'tables/predicted_directions.csv', final)
    source_rows = [{**records[c['source_id']], 'candidate_id': d['candidate_id'], 'cited_quote': c['quote']}
                   for d in final for c in d['citations']]
    csv_write(OUT / 'tables/direction_sources.csv', source_rows)
    print(f'FINAL: {len(final)} retained directions, {len(rejected)} source-check exclusions', flush=True)

"""Bounded prospective forecasts from 2026 claims and their historical neighborhoods."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from figure_pipeline.fig3_revision.client import Client
from .data import compact, read, write
from .predict import Config, Strict, Citation, Label, Direction, check_merge, csv_write, frequencies


class FutureLabel(Label):
    display_label: str = Field(min_length=3, max_length=40)


class FutureDirection(Direction):
    observed_2026_basis: str
    labels: list[FutureLabel] = Field(min_length=1, max_length=6)


class FutureForecast(Strict):
    directions: list[FutureDirection] = Field(max_length=12)
    limitations: str


class Retained(Strict):
    candidate_id: str
    labels: list[FutureLabel] = Field(min_length=1, max_length=6)


class Excluded(Strict):
    candidate_id: str
    reason: str


class FutureMerge(Strict):
    retained: list[Retained]
    excluded: list[Excluded]
    limitations: str


PROMPT = '''Propose up to 12 concrete FUTURE research directions beyond the supplied 2026 papers.
The corpus is 2026-01-10 through 2026-05-08. Historical graph context is 2023–2025 abstract claims.
This is a prospective hypothesis exercise, not retrospective keyword statistics or calibrated forecasting.
Read all target-paper/claim rows; do not simply select the largest fields. For every proposal:
1. State what the cited 2026 full-text claims already establish, respecting narrowed supported scope.
2. Explain actual graph observations (neighbors, communities, local connectivity), separately from interpretation.
3. Specify a testable NEXT question, method development, or combination beyond those existing results.
4. Cite exact contiguous substrings of at least two DIFFERENT papers: at least one 2026_fulltext claim
and at least one historical abstract claim. Cite the target and one historical endpoint of EACH graph fact used.
Do not cite neighbor IDs lacking an evidence row. References in graph_refs must be provided G-prefixed IDs.

Tables are column-coded with evidence_columns, graph_columns, metric_columns and neighbor_columns.
If a target paper has citation_metadata_available=false, zero citation-path flags mean unavailable
reference metadata, not a demonstrated absence of citation. Semantic neighbors remain usable. All target claims
except internally unsupported ones are represented. All actual neighbor edges are listed, but historical
original text is representative, not exhaustive. Complete source spans and graph facts are stored separately.
The graph links target claims to historical claims only. No new-paper-to-new-paper edges were computed.
Do not invent those edges, infer causality or antecedence from proximity, treat lack of an edge as a gap,
or call fixed communities verified disciplines. Semantically similar claims can concern different organisms,
materials, mechanisms and scales: do not silently equate them. Fewer directions is fine when evidence is thin.

labels: Method = next method/technical route; Problem = next concrete research question;
Combination = proposed combination to investigate. Only applicable categories, maximum two per category.
phrase: scientifically specific 3–7 words; display_label: <=40 characters, preferably <=32, compact and
future-task-oriented (e.g., 'Resolve ion-solvation kinetics', not just 'Lithium'). Keep the scientific object.
Do not force every proposal into all three types. Reuse the same label only for a genuinely shared direction
axis. Avoid rebranding demonstrated 2026 results as predictions. All output in English.'''

MERGE = '''Consolidate these prospective directions using only supplied candidates and sources.
Exclude duplicate directions, restatements of observed 2026 results, and unsupported extrapolations.
Keep the concrete future task and correct scientific object/qualifiers. Never add sources or directions.
Every candidate_id must occur EXACTLY ONCE, retained or excluded with a reason.
Normalize true synonymous phrases and their display_label consistently, without conflating distinct science.
Method/Problem/Combination labels are optional by applicability, at most two of each per direction.
Display labels should express future investigation/development tasks, <=40 characters, preferably <=32.
Full phrases preserve scope in 3–7 words (at most 70 characters); do not put whole proposal sentences in phrase. This is source-grounded hypothesis generation, not validated prediction accuracy.'''


def unpack(packet: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    return {row[0]: dict(zip(packet[field+'_columns'], row)) for row in packet[field]}


def validate(packet: dict[str, Any], forecast: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources, graph = unpack(packet, 'evidence'), unpack(packet, 'graph')
    good, bad = [], []
    for i, candidate in enumerate(forecast['directions'], 1):
        row = {**candidate, 'candidate_id': f'{packet["packet_id"]}-D{i:02d}', 'packet_id': packet['packet_id']}
        cites = {c['source_id'] for c in row['citations']}
        errors = []
        if not cites <= sources.keys():
            errors.append('unknown_source')
        else:
            if any(not c['quote'].strip() or c['quote'] not in sources[c['source_id']]['claim'] for c in row['citations']):
                errors.append('quote_not_exact')
            if len({sources[k]['paper_id'] for k in cites}) < 2:
                errors.append('fewer_than_two_papers')
            if {sources[k]['provenance'] for k in cites} != {'2026_fulltext','2023_2025_abstract'}:
                errors.append('requires_2026_and_historical_sources')
        if not row['graph_refs'] or not set(row['graph_refs']) <= graph.keys():
            errors.append('unknown_graph_reference')
        else:
            for ref in row['graph_refs']:
                fact = graph[ref]
                if fact['target'] not in cites or not {n[0] for n in fact['neighbors']} & cites:
                    errors.append('graph_endpoints_not_cited')
        (bad if errors else good).append({'candidate_id': row['candidate_id'], 'errors': errors} if errors else row)
    return good, bad


def obtain(destination: Path, ident: str, instruction: str, payload: Any, schema: type[Strict], budget: int) -> dict[str, Any]:
    path = destination/'predictions'/f'{ident}.json'
    if path.exists():
        return schema.model_validate(read(path)).model_dump()
    ledger = destination/'calls.jsonl'
    entries = ledger.read_text().splitlines() if ledger.exists() else []
    if len(entries) >= budget:
        raise RuntimeError(f'Forecast call cap {budget} reached; no automatic extra calls')
    destination.mkdir(parents=True, exist_ok=True)
    with ledger.open('a') as handle:
        handle.write(compact({'id':ident, 'input_chars':len(compact(payload)), 'attempt':len(entries)+1})+'\n')
    config = Config(output=destination, material_max_chars=900000, request_max_chars=940000,
                    request_max_bytes=2000000, timeout_seconds=1800)
    task = 'merge' if ident=='merge' else 'forecast'
    print(f'Forecast call {len(entries)+1}/{budget}: {ident}, {len(compact(payload))} chars', flush=True)
    result = Client(config, context={'fig7_stage':ident, 'cohort':'2026'}).call(task,instruction,payload,schema)
    write(path,result)
    return result


def predict(destination: Path, budget: int) -> None:
    candidates, rejected, all_sources = [], [], {}
    for path in sorted((destination/'packets').glob('P*.json')):
        packet = read(path)
        forecast = obtain(destination,packet['packet_id'],PROMPT,packet,FutureForecast,budget)
        good, bad = validate(packet,forecast)
        # Prefix packet-local IDs so that merge/source tables cannot alias records from other packets.
        prefix = packet['packet_id']+':'
        sources = read(destination/'sources'/path.name)
        all_sources.update({prefix+k:v for k,v in sources.items()})
        for c in good:
            c['citations'] = [{**x,'source_id':prefix+x['source_id']} for x in c['citations']]
            c['graph_refs'] = [prefix+k for k in c['graph_refs']]
        candidates.extend(good)
        rejected.extend(bad)
    write(destination/'data/candidates.json', candidates)
    write(destination/'qa/rejected_candidates.json', rejected)
    if not candidates:
        raise ValueError('No source-valid future directions; no cloud can be drawn')
    cited = {c['source_id'] for d in candidates for c in d['citations']}
    result = obtain(destination,'merge',MERGE,{'candidates':candidates,
        'sources':{k:v for k,v in all_sources.items() if k in cited}},FutureMerge,budget)
    check_merge(candidates,result)
    by_id = {c['candidate_id']:c for c in candidates}
    final = [{**by_id[r['candidate_id']],'labels':r['labels']} for r in result['retained']]
    write(destination/'data/directions.json',final)
    write(destination/'data/evidence.json',all_sources)
    write(destination/'data/wordcloud_frequencies.json',frequencies(final))
    csv_write(destination/'tables/predicted_directions.csv',final)
    source_rows = [
        {**all_sources[c['source_id']], 'source_id':c['source_id'], 'candidate_id':d['candidate_id'], 'quote':c['quote']}
        for d in final for c in d['citations']]
    columns = sorted({k for row in source_rows for k in row})
    csv_write(destination/'tables/direction_sources.csv',
              [{k:row.get(k) for k in columns} for row in source_rows])
    write(destination/'qa/source_validation.json',{'candidate_count':len(candidates),'rejected':len(rejected),
        'retained':len(final),'merge_excluded':len(result['excluded']),
        'exact_quotes_and_graph_endpoints_checked':True,'predictive_accuracy_validated':False})
    print(f'Retained {len(final)} future directions; source exclusions {len(rejected)}',flush=True)

"""Fresh generation and object-level evaluation through a shared async engine."""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from . import models as m
from .blinding import blind
from .config import METHODS, Config
from .data import generation_input
from .evidence import EvidenceStore
from .execution import Engine
from .materials import PacketIndex, batches, catalog, encoded, packet
from .prompts import PROMPTS
from .retrieval import Retriever, eligible, retrieval_needs_retry
from .storage import Store, read, roster, write

METHOD_STAGES = {'extract', 'support', 'novelty', 'quality', 'concerns'}
STAGES = ('claims', 'manuscript_notes', 'review_sections', 'checklist', 'tone', 'review_dynamics',
          'direct_a', 'gear', 'graph', 'full', 'eacl', 'reviewgrounder', 'evidence_pool', 'core',
          'reference', 'extract', 'support', 'novelty', 'quality_checklist', 'quality', 'concerns',
          'clusters', 'fusion', 'preference', 'recheck')


def recheck_ids(config: Config) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in roster(config):
        groups[row.get('field_name') or '其他'].append(row['paper_id'])
    merged: dict[str, list[str]] = defaultdict(list)
    for field, ids in groups.items():
        merged[field if len(ids) >= 5 else '其他'].extend(ids)
    total = sum(map(len, merged.values()))
    if not total:
        return []
    n = min(config.recheck_papers, total)
    quotas = {f: n*len(ids)/total for f, ids in merged.items()}
    seats = {f: int(q) for f, q in quotas.items()}
    for field in sorted(merged, key=lambda f: (-(quotas[f]-seats[f]), f))[:n-sum(seats.values())]:
        seats[field] += 1
    rng, result = random.Random(config.seed), []
    for field in sorted(merged):
        ids = sorted(merged[field]); rng.shuffle(ids)
        result.extend(ids[:seats[field]])
    return result


class Stages:
    def __init__(self, config: Config, ident: str, engine: Engine, native: Any = None) -> None:
        self.config, self.ident, self.engine, self.native = config, ident, engine, native
        self.store = Store(config)
        self.data = engine.load(self.store.path('prepare', ident))
        self.stage, self.method = '', ''
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}
        index_key = 'manuscript_index/'+ident
        if index_key not in engine.artifacts:
            engine.artifacts[index_key] = catalog([{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                'passage': self.data['manuscript'], 'original_path': self.data['manuscript_path']}], min(1500, config.block_chars))
        self.manuscript_index = engine.artifacts[index_key]

    def get(self, stage: str, method: str = '') -> dict[str, Any]:
        key = (stage, method)
        if key not in self.cache:
            self.cache[key] = self.engine.load(self.store.path(stage, self.ident, method))
        return self.cache[key]

    def report(self, method: str) -> dict[str, Any]:
        return self.get('full' if method == 'fusion' else method)

    async def run(self, stage: str, method: str = '') -> dict[str, Any]:
        self.stage, self.method = stage, method
        if stage in {'direct_a', 'gear', 'graph', 'full', 'eacl', 'reviewgrounder'}:
            return await self.generate(stage)
        return await getattr(self, stage)(method)

    async def ask(self, role: str, payload: Any, schema: type[m.Record], step: str,
                  prompt: str | None = None, synthesis: bool = False) -> dict[str, Any]:
        return await self.engine.ask(self.ident, self.stage, self.method, step, role,
                                     prompt or PROMPTS[role], payload, schema, synthesis)

    async def map(self, name: str, payloads: list[Any], action: Any) -> list[Any]:
        return await self.engine.map(self.ident, self.stage, self.method, name, payloads, action)

    async def notes(self, namespace: str) -> dict[str, Any]:
        rows = await self.map(namespace+'_reading', batches(self.data['manuscript_blocks'], 2, 6500),
            lambda part, i: self.ask('read', {'manuscript_blocks': part}, m.Digest, f'{namespace}_read_{i:05d}',
                'Read manuscript only. Preserve scientific contributions, methods/results, exact short quotations '
                'with block IDs, cited work titles/DOIs and necessary limitations. Do not infer from omitted sections. '
                'Write a compact factual digest, at most 900 characters.'))
        return {'reading': await self.engine.reduce(self.ident, self.stage, self.method, namespace+'_overview', rows),
                'coverage': {'blocks_read': len(self.data['manuscript_blocks']), 'total_blocks': len(self.data['manuscript_blocks'])}}

    async def manuscript_notes(self, _: str) -> dict[str, Any]:
        return await self.notes('neutral')

    async def claims(self, _: str) -> dict[str, Any]:
        candidates = await self.map('贡献候选', batches(self.data['manuscript_blocks'], 2, 6500),
            lambda part, i: self.ask('claims', {'manuscript_blocks': part}, m.SharedClaims, f'candidate_{i:05d}',
                'Extract only explicit scientific contribution candidates, not background or future work. '
                'Exclude author role assignments, funding acquisition, acknowledgments and administrative declarations. '
                'Use block_id as source_span_ids, exact manuscript_quote, author_claim_text and normalized_claim_text. '
                'Keep scope. claim_type METHOD/FINDING/MECHANISM/RESOURCE/THEORY. Return empty claims if none.'))
        result = await self.ask('claims', {'candidates': candidates}, m.SharedClaims, 'consolidate',
            'Consolidate full-paper contribution candidates into at most 8 distinct grounded contributions. '
            'Exclude author roles, funding acquisition and administrative declarations; these are not scientific contributions. '
            'Retain manuscript quotes, block IDs and qualifications. Do not require exactly 8. No outside evidence.', True)
        for i, claim in enumerate(result['claims'], 1):
            claim['claim_id'] = f'{self.ident}::CLAIM::{i:02d}'
        # Prepared artifact carries only the newly mined identities for the existing exporter.
        data = self.get('prepare'); data['claims'] = result['claims']; data['reports'] = {}
        self.store.put('prepare', self.ident, data)
        return result

    async def core(self, _: str) -> dict[str, Any]:
        from .scientific_tasks import select_core
        return await select_core(self)

    async def review_sections(self, _: str) -> dict[str, Any]:
        rows = await self.map('审稿分段', self.data['review_blocks'],
            lambda part, i: self.ask('review_sections', {'current_text': part['text'], 'block_id': part['block_id'],
                'position': part['start'], 'preceding_context': self.data['reviews'][max(0, part['start']-500):part['start']]}, m.ReviewSections, f'section_{i:05d}',
                PROMPTS['review_sections']+' If identity/round is not explicit in this fragment use unknown/0; never guess.'))
        from .review_origins import bind_origins
        sections = bind_origins(self.data['reviews'], [(value['sections'], block)
            for value, block in zip(rows, self.data['review_blocks'])], self.data.get('review_attachment_spans'))
        for i, row in enumerate(sections, 1):
            row['section_id'] = f'R{i:04d}'
        return {'sections': sections}

    def manuscript_material(self, query: Any, budget: int = 6000) -> dict[str, Any]:
        return packet(self.manuscript_index, query, budget)

    def material(self, query: Any, budget: int = 6000, historical: bool = False) -> dict[str, Any]:
        key = 'evidence_index/'+self.ident
        if key not in self.engine.artifacts:
            self.engine.artifacts[key] = PacketIndex(self.get('evidence_pool'))
        index = self.engine.artifacts[key]
        cache_key = (self.ident, encoded(query), budget, historical)
        if cache_key not in self.engine.material_cache:
            self.engine.material_cache[cache_key] = packet(index.index, query, budget, historical, index)
        return self.engine.material_cache[cache_key]

    async def checklist(self, _: str) -> dict[str, Any]:
        sections = [r for r in self.get('review_sections')['sections'] if r['role'] == 'reviewer']
        grouped = batches(sections, 2, 4500)
        values = await self.map('真人问题', grouped, lambda group, i: self.ask('checklist',
            {'sections': group, **self.manuscript_material(group, 5500), 'review_source': self.data['review_path']},
            m.Checklist, f'check_{i:05d}'))
        rows = [r for value in values for r in value['concerns']]
        for i, row in enumerate(rows, 1):
            row['concern_id'] = f'C{i:04d}'
        reviewer_ids = {s['section_id'] for s in sections}
        concerns, context = [], []
        for row in rows:
            keys = set(re.findall(r'R\d{4}', row['section_id']))
            (concerns if keys and keys <= reviewer_ids else context).append(row)
        return {'concerns': concerns, 'manuscript_context': context}

    async def tone(self, _: str) -> dict[str, Any]:
        sections = [s for s in self.get('review_sections')['sections'] if s['role'] == 'reviewer']
        values = await self.map('审稿语气', batches(sections, 4),
            lambda group, i: self.ask('tone', {'sections': group}, m.Tones, f'tone_{i:05d}'))
        return {'items': [r for value in values for r in value['items']]}

    async def review_dynamics(self, _: str) -> dict[str, Any]:
        from .scientific_tasks import review_dynamics
        return await review_dynamics(self)

    def reuse_retrieval_fulltext(self, directory: Path, previous: dict[str, Any]) -> None:
        from .retrieval import doi
        from .materials import browser_notice
        def keep(source: Any) -> None:
            if not isinstance(source, dict) or source.get('source_type') != 'fulltext':
                return
            if browser_notice(source.get('passage', '')):
                return
            key = doi(source.get('doi')) or source.get('work_id') or source.get('pdf_url')
            if key and key not in self.engine.downloads:
                ready = asyncio.get_running_loop().create_future()
                ready.set_result(source)
                self.engine.downloads[key] = ready
        for source in previous.get('sources', []):
            keep(source)
        # Raw source records survive interruption and changes to positional source IDs.
        raw = directory/'evidence.jsonl'
        if raw.exists():
            with raw.open() as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    keep(record.get('payload'))
        for path in directory.glob('fulltext_*.json'):
            keep(read(path))
            path.unlink()

    async def search(self, name: str, analysis: Any, cited: list[str] | None = None) -> dict[str, Any]:
        queries = await self.ask('search', {'title': self.data['paper']['title'], 'own_reading': analysis},
                                m.Queries, name+'_queries')
        directory = self.config.output/'evidence/retrieval'/self.ident/name
        final = directory/'result.json'
        previous: dict[str, Any] = {}
        if final.exists():
            previous = read(final)
            ranking_present = (directory/'ranking.json').exists()
            if (not retrieval_needs_retry(previous) and ranking_present) or self.stage not in {'gear', 'evidence_pool', 'reference'}:
                return previous
            # Reuse successful request files, but rank the recovered candidate pool anew.
            (directory/'ranking.json').unlink(missing_ok=True)
        self.reuse_retrieval_fulltext(directory, previous)
        # A partial rerank must not leave an old final result eligible for resume.
        final.unlink(missing_ok=True)
        result = await Retriever(self.config).search_async(self.engine, generation_input(self.data), queries['queries'],
            list(dict.fromkeys(queries['cited_works']+(cited or []))), name.upper()+'_', name, directory)
        if self.stage == 'gear':
            # Recovered sources require fresh dependent judgments, not old source-index labels.
            for area in ('annotations/checkpoints', 'inputs/tasks'):
                root = self.config.output/area/'gear'/'papers'/self.ident
                for path in root.glob(name+'*.json'):
                    if any(label in path.stem for label in ('relations', 'antecedent', 'assessment', '历史关系', '先例独立')):
                        path.unlink()
                        self.engine.shared.pop(str(path), None)
            index = int(name.removeprefix('gear_'))
            # A recovered GEAR claim changes both its report section and its Full inputs.
            for stage in ('gear', 'full'):
                for area in ('annotations/checkpoints', 'inputs/tasks'):
                    root = self.config.output/area/stage/'papers'/self.ident
                    for path in root.glob('*.json'):
                        step = path.stem
                        dependent = step in {'贡献报告原文', 'writer_joint'} or step.startswith((
                            f'writer_claim_{index:03d}', f'writer_locations_{index:03d}',
                            f'writer_outcomes_{index:03d}'))
                        dependent |= stage == 'full' and (step == f'fuse_{index:05d}' or
                            step.startswith(f'finding_decisions_{index}_'))
                        if dependent:
                            path.unlink()
                            self.engine.shared.pop(str(path), None)
        if self.stage == 'reference' and name.startswith('core_'):
            index = next(i for i, core in enumerate(self.get('core')['items']) if name == 'core_'+core['core_id'])
            root = self.config.output/'annotations/checkpoints/reference/papers'/self.ident
            for path in root.glob('*.json'):
                match = re.fullmatch(rf'compare_(?:locations_)?{index}_(\d+)', path.stem)
                if (match and int(match[1]) >= 10) or path.stem in {
                        f'reference_{index:05d}', f'reference_completed_{index:05d}'}:
                    path.unlink()
                    self.engine.shared.pop(str(path), None)
            (self.config.output/'inputs/tasks/reference/papers'/self.ident/f'补充历史比较_{index}.json').unlink(missing_ok=True)
        evidence = EvidenceStore(directory)
        for source in result['sources']:
            evidence.add(source['source_id'], source['source_type'], source)
        write(final, result)
        return result

    async def read_sources(self, name: str, sources: list[dict[str, Any]]) -> Any:
        index = catalog(sources, self.config.block_chars)
        payloads = []
        for part in index['blocks']:
            provenance = [{k: v for k, v in s.items() if k != 'block_ids'} for s in index['sources']
                          if any(b['block_id'] == part['block_id'] for b in s['block_ids'])]
            payloads.append({'block': part, 'provenance': provenance})
        values = await self.map(name+'_sources', batches(payloads, 4, 6500),
            lambda p, i: self.ask('read', p, m.Digest, f'{name}_source_{i:05d}',
                'Read historical source passage. Retain concrete methods, findings, scope, necessary limitations, '
                'counterevidence and short exact quotes with source IDs and locations. No novelty verdict. Be compact.'))
        return await self.engine.reduce(self.ident, self.stage, self.method, name+'_source_overview', values)

    async def generate(self, method: str) -> dict[str, Any]:
        from gear.innovation.analysis import COMMON, GRAPH
        from gear.innovation.contracts import Assessment
        from gear.innovation.joint_graph import JointAnalysis
        sources: list[dict[str, Any]] = []
        fusion_details: list[dict[str, Any]] = []
        if method == 'full':
            gear, graph = self.report('gear'), self.report('graph')
            from .scientific_tasks import fuse_claims
            values, fusion_details = await fuse_claims(self, gear, graph)
            analysis = {'claims': values, 'joint_graph': graph['joint_analysis'], 'fusion_decisions': fusion_details}
            sources = gear['sources']+graph['sources']
        elif method == 'graph':
            claims = self.get('claims')['claims']
            path = self.config.output/'evidence/native_graph'/f'{self.ident}.json'
            if path.exists():
                facts = read(path)
            else:
                facts = await self.engine.io(lambda: self.native.compute(self.data, claims), gpu=True)
                evidence = EvidenceStore(path.parent/self.ident)
                for card in facts['cards']:
                    evidence.add('GRAPH:'+card['claim']['claim_id'], 'graph_fact', card)
                evidence.add('JOINT_GRAPH:'+self.ident, 'joint_graph_fact', facts['joint'])
                write(path, facts)
            values = await self.map('Graph贡献', facts['cards'], lambda fact, i: self.ask('baseline',
                {'claim_id': fact['claim']['claim_id'], 'claim_text': fact['claim'].get('claim_text', ''),
                 'source_id': 'GRAPH:'+fact['claim']['claim_id'], 'fact': fact}, Assessment, f'graph_{i:05d}', COMMON+'\n'+GRAPH))
            joint = await self.ask('baseline', {'paper_id': self.ident, 'source_id': 'JOINT_GRAPH:'+self.ident, 'union_facts': facts['joint'], 'claim_analyses': values},
                JointAnalysis, 'joint', 'Analyze the union neighborhood jointly. Do not sum single-claim changes or '
                'invent target-to-target edges. Separate actual topology and tentative scientific relationships. '
                'Explain missing neighborhoods and uncertainty. No novelty score.', True)
            for card in facts['cards']:
                sources.append({'source_id': 'GRAPH:'+card['claim']['claim_id'], 'source_type': 'graph_observation', 'passage': encoded(card)})
            sources.append({'source_id': 'JOINT_GRAPH:'+self.ident, 'source_type': 'graph_observation', 'passage': encoded(facts['joint'])})
            analysis = {'claims': values, 'joint_graph': joint}
        elif method == 'gear':
            claims = self.get('claims')['claims']
            from .gear_adapter import analyze_claim
            rows = await self.map('GEAR贡献', claims, lambda claim, i: analyze_claim(self, claim, i))
            values = [r['analysis'] for r in rows]
            sources = [s for r in rows for s in r['sources']]
            analysis = {'claims': values}
        else:
            from .baselines import STEPS
            own = await self.notes(method)
            if method == 'direct_a':
                analysis = await self.ask('baseline', own, m.Analysis, 'direct_analysis',
                    'Write a manuscript-only innovation analysis from the complete fragment readings. '
                    'Separate claims from established evidence; retain scope and uncertainty. No external prior art, '
                    'shared-claim inventory, reviewer comments or other method outputs.', True)
            else:
                steps: dict[str, Any] = {}
                retrieval: dict[str, Any] = {}
                reading: Any = []
                for index, (name, instruction) in enumerate(STEPS[method]):
                    if index == 1:
                        retrieval = await self.search(method, {'manuscript_reading': own, 'initial_analysis': steps})
                        sources = retrieval['sources']
                        reading = await self.read_sources(method, sources)
                    steps[name] = await self.ask('baseline', {'manuscript_reading': own, 'previous_steps': steps,
                        'historical_reading': reading, 'retrieval_limitations': retrieval.get('failures', [])},
                        m.Analysis, name, instruction+' Preserve exact evidence IDs and limitations; no outside judgments.', True)
                analysis = steps
        if method in {'gear', 'graph', 'full'}:
            from .scientific_tasks import write_scientific_report
            report = await write_scientific_report(self, method, analysis, sources)
        else:
            report = await self.ask('baseline', {'method_analysis': analysis}, m.Report, 'writer',
                'Render only the supplied scientific judgments as a Chinese innovation analysis, target 1200–2000 '
                'Chinese characters. Preserve source IDs, limitations and disagreements. No new conclusions or method name. '
                'No numeric novelty score.', True)
        result = {**report, 'sources': sources, 'model': self.config.model, 'analysis': analysis}
        if method == 'full':
            result['fusion_details'] = fusion_details
        if method in {'gear', 'graph', 'full'}:
            result['analyses'] = values
        if method == 'graph':
            result['joint_analysis'] = joint
        return result

    async def evidence_pool(self, _: str) -> dict[str, Any]:
        checklist = self.get('checklist')
        cited = [q for c in checklist['concerns']+checklist.get('manuscript_context', [])
                 for q in c['prior_work_queries']]
        retrieved = await self.search('history', self.get('manuscript_notes'), cited)
        sources = [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'passage': self.data['manuscript'],
                    'original_path': self.data['manuscript_path']}]+retrieved['sources']
        for method in ('gear', 'graph', 'eacl', 'reviewgrounder'):
            sources.extend(self.report(method)['sources'])
        selected, excluded = [], []
        for source in sources:
            if source['source_type'] in {'manuscript', 'graph_observation'} or eligible(source, generation_input(self.data)):
                selected.append(source)
            else:
                excluded.append({k: v for k, v in source.items() if k != 'passage'})
        result = catalog(selected, min(1500, self.config.block_chars))
        return {**result, 'excluded': excluded, 'limitations': retrieved['failures']}

    async def reference(self, _: str) -> dict[str, Any]:
        from .scientific_tasks import reference_core
        values = await self.map('历史参考', self.get('core')['items'], lambda core, i: reference_core(self, core, i))
        extra = []
        for path in (self.config.output/'evidence/retrieval'/self.ident).glob('core_*/result.json'):
            extra.extend(read(path)['sources'])
        if extra:
            from .materials import catalog
            pool = self.get('evidence_pool')
            additions = catalog(extra, 1500)
            by_text = {b['text']: b['block_id'] for b in pool['blocks']}
            remap = {}
            for block in additions['blocks']:
                if block['text'] not in by_text:
                    key = f'T{len(pool["blocks"])+1:06d}'
                    by_text[block['text']] = key
                    pool['blocks'].append({'block_id': key, 'text': block['text']})
                remap[block['block_id']] = by_text[block['text']]
            existing = {s['source_id'] for s in pool['sources']}
            for source in additions['sources']:
                if source['source_id'] not in existing:
                    for block in source['block_ids']:
                        block['block_id'] = remap[block['block_id']]
                    pool['sources'].append(source)
                    existing.add(source['source_id'])
            self.store.put('evidence_pool', self.ident, pool)
            self.engine.artifacts.pop('evidence_index/'+self.ident, None)
            self.engine.material_cache = {k: v for k, v in self.engine.material_cache.items() if k[0] != self.ident}
        return {'items': [r for v in values for r in v['items']]}

    async def extract(self, method: str) -> dict[str, Any]:
        from .scientific_tasks import extract_report
        return await extract_report(self, method)

    def object_material(self, objects: list[dict[str, Any]], budget: int = 3000) -> dict[str, Any]:
        selected, coverage = {}, []
        for obj in objects:
            result = self.material(obj, budget)
            coverage.append(result['coverage'])
            for part in result['evidence_blocks']:
                selected.setdefault(part['block_id'], part)
        return {'evidence_blocks': list(selected.values()), 'coverage': coverage}

    def support_payload(self, method: str, units: list[Any]) -> dict[str, Any]:
        return {'units': units, 'original_report_citations': self.report(method)['cited_source_ids'],
                **self.object_material(units)}

    async def support_group(self, method: str, group: list[dict[str, Any]], index: int) -> dict[str, Any]:
        name = f'来源核验超时_{index:05d}'
        split = self.config.output/'inputs/tasks/support'/method/self.ident/(name+'.json')
        if not split.exists():
            try:
                return await self.ask('support', lambda: self.support_payload(method, group),
                                      m.Support, f'support_{index:05d}')
            except TimeoutExpired:
                if len(group) == 1:
                    raise
        from .recheck import missing_schema
        values = await self.map(name, group, lambda unit, i: self.ask('support',
            lambda: self.support_payload(method, [unit]), missing_schema('support', unit['unit_id']),
            f'support_{index:05d}_unit_{i:05d}'))
        return {'units': [row for value in values for row in value['units']]}

    async def support(self, method: str) -> dict[str, Any]:
        units = [u for u in self.get('extract', method)['units'] if u['needs_verification'] or u['substantive']]
        # Group neighboring report units; shared locators sort together to avoid repeated evidence.
        units.sort(key=lambda u: (u['original_locator'], u['unit_id']))
        groups = []
        for candidate in batches(units, self.config.unit_batch_size, 4500):
            pending = [candidate]
            while pending:
                group = pending.pop(0)
                if len(encoded(self.support_payload(method, group))) > 64000 and len(group) > 1:
                    middle = len(group)//2
                    pending[0:0] = [group[:middle], group[middle:]]
                else:
                    groups.append(group)
        values = await self.map('来源核验', groups, lambda group, i: self.support_group(method, group, i))
        from .recheck import missing_schema, unique_rows
        found = unique_rows([r for v in values for r in v['units']], 'unit_id', {u['unit_id'] for u in units})
        missing = [u for u in units if u['unit_id'] not in found]
        async def finish(unit: dict[str, Any], _: int) -> dict[str, Any]:
            ident = unit['unit_id']
            result = await self.ask('support', self.support_payload(method, [unit]),
                missing_schema('support', ident), 'support_missing_'+ident)
            rows = unique_rows(result['units'], 'unit_id', {ident})
            if ident not in rows:
                raise ValueError('Support judgment missing: '+ident)
            return rows[ident]
        for row in await self.map('来源核验缺项', missing, finish):
            found[row['unit_id']] = row
        return {'units': [found[u['unit_id']] for u in units]}

    def novelty_payload(self, method: str, core: dict[str, Any]) -> dict[str, Any]:
        ident = core['core_id']
        return {'core': [core], 'reference': {'items': [r for r in self.get('reference')['items'] if r['core_id'] == ident]},
                'predictions': [p for p in self.get('extract', method)['predictions'] if p['core_id'] == ident],
                'report': self.report(method)['body'], **self.material(core, 5000, historical=True)}

    async def novelty_one(self, method: str, core: dict[str, Any], index: int) -> dict[str, Any]:
        from gear.codex_cli import _strict_response_schema

        from .client import BASE
        from .materials import fits
        from .recheck import missing_schema, unique_rows
        ident = core['core_id']
        value = await self.ask('novelty', lambda: self.novelty_payload(method, core),
                               m.Novelty, f'novelty_{index:05d}')
        valid = unique_rows(value['items'], 'core_id', {ident})
        if ident in valid:
            return {'items': [valid[ident]]}
        schema = missing_schema('novelty', ident)
        formatted = _strict_response_schema(schema.model_json_schema())
        instruction = PROMPTS['novelty']
        material = self.novelty_payload(method, core)
        if not fits(self.config, material, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
            reading = await self.engine.reduce(self.ident, self.stage, self.method,
                                               f'novelty_{index:05d}_read', material)
            anchors = {k: material[k] for k in ('core', 'reference', 'predictions')}
            material = {**anchors, 'bounded_reading': reading}
            if not fits(self.config, material, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
                digest = await self.ask('read', {'reading': reading}, m.Digest,
                    f'novelty_context_{ident}', 'Condense neutral evidence only. Preserve quotations, source IDs, '
                    'scope and uncertainty. Do not make an evaluation. Aim below 900 characters.')
                material = {**anchors, 'bounded_reading': digest}
        if not fits(self.config, material, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
            raise ValueError(f'核心评价材料仍超预算: {ident}')
        value = await self.ask('novelty', material, schema, f'novelty_missing_{ident}', instruction)
        valid = unique_rows(value['items'], 'core_id', {ident})
        if ident not in valid:
            raise ValueError(f'核心评价仍缺少对象: {ident}')
        return {'items': [valid[ident]]}

    async def novelty(self, method: str) -> dict[str, Any]:
        from .scientific_tasks import effective_increment
        values = await self.map('核心判断', self.get('core')['items'],
                                lambda core, i: self.novelty_one(method, core, i))
        predictions = {p['core_id']: p for p in self.get('extract', method)['predictions']}
        refs = {r['core_id']: r for r in self.get('reference')['items']}
        rows = [r for v in values for r in v['items']]
        for row in rows:
            ident = row['core_id']
            row['effective_increment'] = effective_increment(refs[ident], predictions[ident], row)
        return {'items': rows}

    async def quality_checklist(self, _: str) -> dict[str, Any]:
        return await self.ask('quality_checklist', {'manuscript_reading': self.get('manuscript_notes'),
            **self.material(self.get('core'), 5000, historical=True)}, m.QualityChecklist, 'quality_checklist', synthesis=True)

    async def quality(self, method: str) -> dict[str, Any]:
        from .scientific_tasks import quality_report
        return await quality_report(self, method)

    def concern_payload(self, method: str, concerns: list[Any]) -> dict[str, Any]:
        return {'checklist': {'concerns': concerns}, 'report': self.report(method)['body'],
                **self.object_material(concerns, 3000)}

    async def concerns(self, method: str) -> dict[str, Any]:
        from .packing import enabled, evaluate
        if enabled(self.config):
            return await evaluate(self, method)
        values = await self.map('真人问题匹配', batches(self.get('checklist')['concerns'], 2, 4000),
            lambda group, i: self.ask('concerns', lambda: self.concern_payload(method, group),
                                      m.ConcernMatches, f'concerns_{i:05d}'))
        return {'matches': [r for v in values for r in v['matches']]}

    async def clusters(self, _: str) -> dict[str, Any]:
        units, mapping, eligible_keys = [], {}, []
        missing_methods = []
        for method in METHODS:
            if method not in {'gear', 'graph', 'fusion'} and any(
                    not self.store.path(stage, self.ident, method).exists() for stage in ('extract', 'support')):
                missing_methods.append(method)
                continue
            extracted = {u['unit_id']: u for u in self.get('extract', method)['units']}
            for row in self.get('support', method)['units']:
                unit = extracted.get(row['unit_id'])
                if unit is None:
                    continue
                key = f'U{len(mapping)+1:05d}'
                mapping[key] = method+'/'+row['unit_id']
                item = {'unit_key': key, 'quote': unit['quote'], 'errors': row['errors'], 'insight_type': unit['insight_type']}
                if unit['substantive'] and not unit['paraphrase']:
                    units.append(item)
                    if row['support'] == 'supported' and row['scope_correct'] and row['nonparaphrase_insight']:
                        eligible_keys.append(key)
        random.Random(self.config.seed).shuffle(units)
        candidates = await self.map('信息候选簇', batches(units, 8, 6500), lambda group, i: self.ask('clusters',
            {'units': group}, m.Clusters, f'clusters_{i:05d}'))
        result = await self.ask('clusters', {'candidate_clusters': candidates}, m.Clusters, 'cluster_merge',
            PROMPTS['clusters']+' Align synonyms ACROSS all candidate groups; preserve original unit_keys, '
            'never simply concatenate local clusters. Distinct scientific increments must remain distinct.', True) if units else {'clusters': []}
        for i, row in enumerate(result['clusters'], 1):
            row['cluster_id'] = f'I{i:04d}'
        important = await self.map('信息重要性', batches(result['clusters'], 4, 4500), lambda group, i: self.ask('importance',
            {'clusters': group, **self.material(group)}, m.Importance, f'importance_{i:05d}'))
        return {**result, 'importance': [r for v in important for r in v['items']],
                'eligible_unit_keys': eligible_keys, 'mapping': mapping, 'missing_methods': missing_methods}

    async def fusion(self, _: str) -> dict[str, Any]:
        clusters = self.get('clusters')
        from .scientific_tasks import fusion_errors
        judgments = await fusion_errors(self)
        memberships = {c['cluster_id']: {clusters['mapping'].get(k, '').split('/')[0] for k in c['unit_keys']}
                       for c in clusters['clusters']}
        eligible = set(clusters.get('eligible_unit_keys', clusters['mapping']))
        valid = {c['cluster_id']: {clusters['mapping'][k].split('/')[0] for k in c['unit_keys'] if k in eligible}
                 for c in clusters['clusters']}
        union = {key for key, methods in valid.items() if methods & {'gear', 'graph'}}
        full = {key for key, methods in valid.items() if 'fusion' in methods}
        branch_mentions = {key for key, methods in memberships.items() if methods & {'gear', 'graph'}}
        full_mentions = {key for key, methods in memberships.items() if 'fusion' in methods}
        important = {r['cluster_id'] for r in clusters['importance'] if r['important']}
        return {**judgments,
                'sets': {'union': sorted(union), 'full': sorted(full), 'retained': sorted(union & full),
                         'not_retained': sorted(union-full), 'added': sorted(full-branch_mentions),
                         'newly_supported': sorted((full-union) & branch_mentions),
                         'mentioned_without_support': sorted((union & full_mentions)-full),
                         'important_union': sorted(union & important), 'important_retained': sorted(union & full & important)}}

    async def preference(self, variant: str) -> dict[str, Any]:
        method, order = variant.rsplit('_', 1)
        names = ['fusion', method] if order == 'AB' else [method, 'fusion']
        # The query does not depend on report order or the compared method.
        selected = {}
        for core in self.get('core')['items']:
            for part in self.material(core, 6500, historical=True)['evidence_blocks']:
                selected[part['block_id']] = part
        facts = read(self.config.output/'evidence/native_graph'/f'{self.ident}.json')
        graph_facts = [{'source_id': 'GRAPH:'+card['claim']['claim_id'],
                        'source_type': 'graph_observation', 'claim_id': card['claim']['claim_id'],
                        'metrics': card['metrics'], 'notes': card['notes']} for card in facts['cards']]
        graph_facts.append({'source_id': 'JOINT_GRAPH:'+self.ident, 'source_type': 'graph_observation',
                            'observations': {k: v for k, v in facts['joint'].items()
                                             if not isinstance(v, (dict, list))},
                            'notes': facts['joint'].get('notes', [])})
        neutral = {'evidence_blocks': list(selected.values()), 'graph_observations': graph_facts}
        identifiers = sorted(set(self.report('fusion')['cited_source_ids']+self.report(method)['cited_source_ids']
                                 +[r['source_id'] for r in graph_facts]))
        payload, source_mapping = blind({'sources': [{'source_id': key} for key in identifiers], **neutral, 'reports':
            {letter: self.report(name)['body'] for letter, name in zip(('A', 'B'), names)}})
        return {**await self.ask('preference', payload, m.Preference, 'preference',
                PROMPTS['preference']+' Graph observations establish recorded local topology only, '
                'not scientific novelty, causal relations or verified disciplinary impact.', synthesis=True),
                'mapping': dict(zip(('A', 'B'), names)), 'order': order, 'source_mapping': source_mapping}

    async def recheck(self, _: str) -> dict[str, Any]:
        from .recheck import run
        return await run(self)

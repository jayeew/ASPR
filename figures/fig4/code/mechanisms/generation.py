"""Condition-specific projections, reused judgments and common uncapped writers."""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from gear.innovation.analysis import COMMON, FUSION, GRAPH
from gear.innovation.contracts import Assessment
from gear.innovation.joint_graph import JointAnalysis
from figure_pipeline.fig3_revision import models as old
from figure_pipeline.fig3_revision.execution import Engine
from figure_pipeline.fig3_revision.evidence import EvidenceStore
from figure_pipeline.fig3_revision.materials import PacketIndex, batches, catalog, encoded, packet
from figure_pipeline.fig3_revision.scientific_tasks import locate_quotes
from figure_pipeline.fig3_revision.storage import Store

from .config import PATH_FIELDS, PATH_METRICS, SIMILARITY_METRICS, Config
from .io import artifact, read, record, write
from .models import Outcomes, TracedReport


def project_facts(facts: dict[str, Any], condition: str) -> dict[str, Any]:
    row = copy.deepcopy(facts)
    if condition in {'T', 'E'}:
        return {'cards': [], 'joint': None}
    if condition == 'F_noJ':
        row['joint'] = None
    if condition not in {'F_noM', 'F_noP'}:
        return row
    for card in row['cards']:
        card.pop('notes', None)
        if condition == 'F_noM':
            card['metrics'] = [m for m in card['metrics'] if m['name'] in SIMILARITY_METRICS | PATH_METRICS]
        else:
            card['metrics'] = [m for m in card['metrics'] if m['name'] not in PATH_METRICS]
            for neighbor in card['neighbors']:
                for key in PATH_FIELDS:
                    neighbor.pop(key, None)
                neighbor['edge_type'] = 'semantic_only'
    if condition == 'F_noM':
        row['joint'] = {k: v for k, v in row['joint'].items()
                        if k in {'historical_edges', 'insertion_edges', 'claims_without_neighbors'}}
    else:
        row['joint'].pop('notes', None)
    return row


def graph_sources(facts: dict[str, Any], paper: str) -> list[dict[str, Any]]:
    sources = [{'source_id': 'GRAPH:'+c['claim']['claim_id'], 'source_type': 'graph_observation',
                'passage': encoded(c)} for c in facts['cards']]
    if facts['joint'] is not None:
        sources.append({'source_id': 'JOINT_GRAPH:'+paper, 'source_type': 'graph_observation',
                        'passage': encoded(facts['joint'])})
    return sources


def joint_input_findings(paper: str, joint: dict[str, Any]) -> list[dict[str, Any]]:
    return [{'finding_key': f'graph:JOINT:{paper}:{i:03d}', 'origin_branch': 'graph',
             'claim_id': finding['claim_ids'][0], 'claim_ids': finding['claim_ids'],
             'exposure_stage': 'paper_synthesis', 'finding': {'text': finding['observation']+'\n'+finding['interpretation'],
             'evidence_keys': finding['evidence_keys']}} for i, finding in enumerate(joint.get('findings', []))]


class Generation:
    def __init__(self, config: Config, paper: str, engine: Engine) -> None:
        self.config, self.paper, self.engine = config, paper, engine
        self.store = Store(config)
        self.common = read(config.output/'inputs/common'/f'{paper}.json')
        self.fixed = read(config.output/'inputs/fixed_analysis'/f'{paper}.json')
        self.facts = read(config.output/'evidence/native_graph'/f'{paper}.json')
        self.claims = self.common['claims']

    async def ask(self, condition: str, step: str, payload: Any, schema: Any,
                  instruction: str, role: str = 'baseline') -> dict[str, Any]:
        stage = 'fig4_writer' if step == 'writer' else 'full'
        if step == 'synthesis':
            previous = self.config.output/'annotations/checkpoints/full'/condition/self.paper/'synthesis.json'
            if not previous.exists():
                stage = 'fig4_synthesis'
        return await self.engine.ask(self.paper, stage, condition, step, role, instruction,
                                     payload, schema, True)

    async def text_analysis(self, slot: str) -> list[dict[str, Any]]:
        path = artifact(self.config.output, 'text_analysis', self.paper, slot)
        if path.exists():
            return read(path)['analyses']
        async def one(claim: dict[str, Any], index: int) -> dict[str, Any]:
            result = await self.ask(slot, f'text_{index:03d}', {'claim': claim,
                **self.common['claim_materials'][claim['claim_id']]}, old.Analysis,
                'Read the original manuscript and historical passages for this contribution. '
                'Make an ordinary text-based '+('historical comparison' if slot == 'text_gear' else 'knowledge-relation analysis')+
                '. No branch judgments or graph observations are supplied. Preserve concrete comparisons, '
                'necessary scope, original source IDs and locations, and unresolved limits. No length target.')
            return {'claim_id': claim['claim_id'], **result}
        values = await self.engine.map(self.paper, 'full', slot, 'ordinary_text', self.claims, one)
        write(path, {'analyses': values})
        return values

    async def text_joint(self) -> dict[str, Any]:
        path = artifact(self.config.output, 'text_joint', self.paper)
        if path.exists():
            return read(path)
        pool = self.store.get('evidence_pool', self.paper)
        evidence = packet(pool, [c['normalized_claim_text'] for c in self.claims], 24000, True)
        value = await self.ask('text_joint', 'joint', {'claims': self.claims, **evidence}, old.Analysis,
            'Synthesize relationships across these contributions by reading the original passages. '
            'No graph topology or branch judgments are supplied. Distinguish source-supported relationships '
            'from conjectures, identify at least two claim IDs for a cross-contribution relation, '
            'preserve original source locations and limitations. No length target.')
        write(path, value)
        return value

    async def masked_graph(self, condition: str, facts: dict[str, Any]) -> dict[str, Any]:
        path = artifact(self.config.output, 'masked_graph', self.paper, condition)
        if path.exists():
            return read(path)
        async def one(card: dict[str, Any], index: int) -> dict[str, Any]:
            return await self.ask(condition, f'graph_{index:03d}',
                {'claim_id': card['claim']['claim_id'], 'claim_text': card['claim']['claim_text'],
                 'source_id': 'GRAPH:'+card['claim']['claim_id'], 'fact': card}, Assessment, COMMON+'\n'+GRAPH)
        values = await self.engine.map(self.paper, 'full', condition, 'masked_graph', facts['cards'], one)
        joint = await self.ask(condition, 'masked_joint', {'paper_id': self.paper,
            'source_id': 'JOINT_GRAPH:'+self.paper, 'union_facts': facts['joint'], 'claim_analyses': values},
            JointAnalysis, 'Interpret only the supplied projected union-neighborhood facts. '
            'Do not recover hidden numeric summaries or paths from previous outputs. '
            'Separate observed topology from tentative scientific interpretations; no novelty score.')
        result = {'analyses': values, 'joint_analysis': joint}
        write(path, result)
        return result

    async def masked_fusion(self, condition: str, graph: dict[str, Any], facts: dict[str, Any]) -> list[dict]:
        path = artifact(self.config.output, 'masked_fusion', self.paper, condition)
        if path.exists():
            return read(path)['analyses']
        gear = {r['claim_id']: r for r in self.fixed['gear']['analyses']}
        mapped = {r['claim_id']: r for r in graph['analyses']}
        cards = {r['claim']['claim_id']: r for r in facts['cards']}
        async def one(claim: dict[str, Any], index: int) -> dict[str, Any]:
            ident = claim['claim_id']
            return await self.ask(condition, f'fusion_{index:03d}', {'claim_id': ident,
                'claim_text': claim['normalized_claim_text'], 'gear': gear[ident], 'graph': mapped[ident],
                'graph_fact': cards[ident], **self.common['claim_materials'][ident]}, Assessment, COMMON+'\n'+FUSION)
        values = await self.engine.map(self.paper, 'full', condition, 'masked_fusion', self.claims, one)
        write(path, {'analyses': values})
        return values

    async def inputs(self, condition: str) -> dict[str, Any]:
        path = artifact(self.config.output, 'condition_inputs', self.paper, condition)
        if path.exists():
            return read(path)
        facts = project_facts(self.facts, condition)
        gear = self.fixed['gear']['analyses'] if condition != 'G' and condition != 'T' else await self.text_analysis('text_gear')
        if condition in {'T', 'E'}:
            graph = {'analyses': await self.text_analysis('text_graph')}
        elif condition in {'F_noM', 'F_noP'}:
            graph = await self.masked_graph(condition, facts)
        else:
            graph = self.fixed['graph']
        joint = await self.text_joint() if condition in {'T', 'E', 'F_noJ'} else graph['joint_analysis']
        fused = []
        if condition in {'F', 'F_noJ'}:
            fused = self.fixed['full']['analyses']
        elif condition in {'F_noM', 'F_noP'}:
            fused = await self.masked_fusion(condition, graph, facts)
        gear_on, graph_on = condition not in {'T', 'G'}, condition not in {'T', 'E'}
        findings = []
        for branch, rows, enabled in [('gear', gear, gear_on), ('graph', graph['analyses'], graph_on)]:
            if enabled:
                for row in rows:
                    findings.extend({'finding_key': f'{branch}:{row["claim_id"]}:{i:03d}',
                                     'origin_branch': branch, 'claim_id': row['claim_id'],
                                     'claim_ids': [row['claim_id']], 'exposure_stage': 'claim_fusion', 'finding': f}
                                    for i, f in enumerate(row['findings']))
        if condition not in {'T', 'E', 'F_noJ'}:
            findings.extend(joint_input_findings(self.paper, joint))
        result = {'condition': condition, 'gear': gear, 'graph': graph['analyses'],
                  'joint': joint, 'fused': fused, 'findings': findings, 'facts': facts,
                  'reused_fusion_decisions': self.fixed['full'].get('fusion_details', []) if condition in {'F', 'F_noJ'} else []}
        evidence_store = EvidenceStore(self.config.output/'evidence/conditions'/condition/self.paper)
        for card in facts['cards']:
            evidence_store.add('GRAPH:'+card['claim']['claim_id'], 'graph_fact', card)
        if facts['joint'] is not None:
            evidence_store.add('JOINT_GRAPH:'+self.paper, 'joint_graph_fact', facts['joint'])
        write(path, result)
        return result

    async def generate(self, condition: str) -> None:
        target = self.config.output/'reports'/condition/'papers'/f'{self.paper}.json'
        if target.exists():
            record(self.config.output, 'generate', self.paper, condition, 'reused')
            return
        inputs = await self.inputs(condition)
        gear = {r['claim_id']: r for r in inputs['gear']}
        graph = {r['claim_id']: r for r in inputs['graph']}
        fused = {r['claim_id']: r for r in inputs['fused']}
        cards = {c['claim']['claim_id']: c for c in inputs['facts']['cards']}
        async def one(claim: dict[str, Any], index: int) -> dict[str, Any]:
            ident = claim['claim_id']
            payload = {'claim': claim, 'assessment': {'first_analysis': gear[ident], 'second_analysis': graph[ident],
                'existing_fusion': fused.get(ident)}, 'graph_fact': cards.get(ident),
                **self.common['claim_materials'][ident]}
            value = await self.ask(condition, f'organize_{index:03d}', payload, old.Analysis,
                'Organize the supplied analyses for this contribution using actual original evidence. '
                'Preserve distinct supported information, necessary scope and scientific disagreements; '
                'correct only with cited source evidence. Missing support remains unresolved. '
                'Existing analyses are candidates, not scientific ground truth. No length target or novelty score.')
            return {'claim_id': ident, **value}
        organized = await self.engine.map(self.paper, 'full', condition, 'organize', self.claims, one)
        synthesis = await self.ask(condition, 'synthesis', {'contribution_sections': organized,
            'joint_analysis': inputs['joint'], 'joint_facts': inputs['facts']['joint']}, old.Analysis,
            'Synthesize the whole paper from the supplied organized contributions and cross-contribution analysis. '
            'Preserve supported complementarity, limitations and unresolved disagreements, with source locations. '
            'Graph connectivity is not scientific causality. No length target; remove repetition only.')
        pool = self.store.get('evidence_pool', self.paper)
        evidence = packet(pool, [c['normalized_claim_text'] for c in self.claims], 24000, True)
        report = await self.ask(condition, 'writer', {'assessment': synthesis, 'organized_contributions': organized,
            'findings': inputs['findings'], 'graph_facts': inputs['facts'], **evidence}, TracedReport,
            'Write a complete Chinese scientific innovation analysis from the supplied organized material. '
            'There is NO word or character target. Retain distinct evidence-supported information and necessary '
            'qualifications, remove repetition, preserve original source_id@start:end locations. No system names '
            'or numeric novelty score. For EVERY supplied finding_key record its actual final report outcome '
            'retained/merged/corrected/omitted/unresolved with a verbatim report_quote and reason. '
            'Omitted/unresolved findings may have an empty quote; never fabricate a quote. '
            'Treat source-supported topology separately from scientific relations; do not invent new claims.')
        expected = {f['finding_key'] for f in inputs['findings']}
        excluded_outcomes = [o for o in report['finding_outcomes'] if o['finding_key'] not in expected]
        report['finding_outcomes'] = [o for o in report['finding_outcomes'] if o['finding_key'] in expected]
        if excluded_outcomes:
            record(self.config.output, 'writer_alignment', self.paper, condition, 'corrected',
                   reason='Unknown finding keys excluded from actual-input lineage; body unchanged',
                   excluded_outcomes=excluded_outcomes)
        counts = Counter(o['finding_key'] for o in report['finding_outcomes'])
        good = [o for o in report['finding_outcomes'] if counts[o['finding_key']] == 1]
        missing = expected-{o['finding_key'] for o in good}
        if missing:
            from .evaluation import Evaluation
            from .tasks import exact_items
            alignment = Evaluation(self.config, self.paper, self.engine)
            alignment.method = condition
            pending = [f for f in inputs['findings'] if f['finding_key'] in missing]
            repaired = []
            for index, group in enumerate(batches(pending, 8, 10000)):
                value = await exact_items(alignment, 'extract',
                    {'report': report['body'], 'findings': group}, Outcomes,
                    f'writer_missing_outcomes_{index:03d}',
                    'Align EACH supplied finding_key to this unchanged final report. Return retained/merged/'
                    'corrected/omitted/unresolved with a verbatim report_quote and reason. Do not rewrite the '
                    'report or judge scientific support. For omissions or uncertain matches leave quote empty. '
                    'Return every supplied finding_key exactly once.', 'finding_key',
                    {f['finding_key'] for f in group})
                repaired.extend(value['items'])
            report['finding_outcomes'] = good+repaired
        if ({o['finding_key'] for o in report['finding_outcomes']} != expected
                or len(report['finding_outcomes']) != len(expected)):
            raise ValueError('Writer finding outcomes do not match actual input findings')
        from .evaluation import Evaluation
        locator = Evaluation(self.config, self.paper, self.engine)
        locator.stage, locator.method = 'full', condition
        quoted = {o['finding_key']: o['report_quote'] for o in report['finding_outcomes'] if o['report_quote']}
        quote_rows = [{'key': key, 'quote': quote} for key, quote in quoted.items()]
        async def locate(group: list[dict[str, str]], index: int) -> dict[str, str | None]:
            return await locate_quotes(locator, report['body'], {r['key']: r['quote'] for r in group},
                                       f'writer_outcome_locations_batch_{index:03d}')
        groups = await self.engine.map(self.paper, 'full', condition, 'writer_locations',
                                       batches(quote_rows, 20, 12000), locate)
        located = {key: quote for group in groups for key, quote in group.items()}
        for outcome in report['finding_outcomes']:
            if outcome['report_quote']:
                quote = located.get(outcome['finding_key'])
                if quote is None:
                    outcome.update(action='unresolved', report_quote='', reason=outcome['reason']+'; quote not located')
                else:
                    outcome['report_quote'] = quote
        result = {**report, 'paper_id': self.paper, 'condition': condition, 'model': self.config.model,
                  'excluded_unknown_finding_outcomes': excluded_outcomes,
                  'report_length_target': None, 'body_chars': len(report['body']), 'organized': organized,
                  'synthesis': synthesis, 'input_artifact': str(artifact(self.config.output, 'condition_inputs', self.paper, condition))}
        write(target, result)
        target.with_suffix('.md').write_text(report['body']+'\n', encoding='utf-8')
        record(self.config.output, 'generate', self.paper, condition, 'completed', body_chars=len(report['body']))

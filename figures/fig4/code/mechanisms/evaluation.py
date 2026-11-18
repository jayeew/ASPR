"""Reuse Fig.3 evaluators with seven report identities and condition-bound graph evidence."""
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision import models as old
from figure_pipeline.fig3_revision.execution import Engine
from figure_pipeline.fig3_revision.materials import PacketIndex, batches, catalog, encoded, packet
from figure_pipeline.fig3_revision.prompts import PROMPTS
from figure_pipeline.fig3_revision.scientific_tasks import locate_quotes
from figure_pipeline.fig3_revision.stages import Stages
from figure_pipeline.fig3_revision.storage import Store

from .config import CONDITIONS, Config
from .generation import graph_sources, project_facts
from .io import artifact, read, record, write
from .models import FindingAtomBatches, Relations
from .tasks import ask as packed_ask, exact_items, identity_groups


class ConditionPacketIndex(PacketIndex):
    """Share unchanged original-text lexical sets and source locators."""

    def __init__(self, index: dict[str, Any], common: PacketIndex, graph: PacketIndex) -> None:
        self.index = index
        self.tokens = {**common.tokens, **graph.tokens}
        self.references = {h: {**common.refs(h), **graph.refs(h)} for h in (False, True)}


class Evaluation(Stages):
    def __init__(self, config: Config, paper: str, engine: Engine) -> None:
        super().__init__(config, paper, engine)
        self.facts = read(config.output/'evidence/native_graph'/f'{paper}.json')
        self.pool_condition = ''

    def report(self, method: str) -> dict[str, Any]:
        if method == 'branch_input':
            return read(artifact(self.config.output, 'branch_report', self.ident))
        return self.engine.load(self.config.output/'reports'/method/'papers'/f'{self.ident}.json')

    def material(self, query: Any, budget: int = 6000, historical: bool = False) -> dict[str, Any]:
        condition = self.pool_condition or self.method
        key = ('fig4_pool', self.ident, condition)
        if key not in self.engine.artifacts:
            pool = self.get('evidence_pool')
            common_key = ('fig4_common_pool', self.ident)
            if common_key not in self.engine.artifacts:
                self.engine.artifacts[common_key] = PacketIndex(pool)
            projected = project_facts(self.facts, condition if condition in CONDITIONS else 'F')
            extra = catalog(graph_sources(projected, self.ident), 1500)
            remap = {b['block_id']: 'Q'+b['block_id'] for b in extra['blocks']}
            for b in extra['blocks']:
                b['block_id'] = remap[b['block_id']]
            for s in extra['sources']:
                for b in s['block_ids']:
                    b['block_id'] = remap[b['block_id']]
            index = {'blocks': pool['blocks']+extra['blocks'], 'sources': pool['sources']+extra['sources']}
            self.engine.artifacts[key] = ConditionPacketIndex(
                index, self.engine.artifacts[common_key], PacketIndex(extra))
        index = self.engine.artifacts[key]
        return packet(index.index, query, budget, historical, index)

    async def stage_result(self, stage: str, condition: str) -> dict[str, Any]:
        path = self.store.path(stage, self.ident, condition)
        if path.exists():
            return read(path)
        self.stage, self.method = stage, condition
        value = await getattr(self, stage)(condition)
        write(path, value)
        record(self.config.output, stage, self.ident, condition, 'completed')
        return value

    async def evaluate_condition(self, condition: str) -> None:
        for stage in ('extract', 'support', 'novelty'):
            await self.stage_result(stage, condition)

    async def cluster_reports(self) -> dict[str, Any]:
        path = artifact(self.config.output, 'information_clusters', self.ident)
        if path.exists():
            return read(path)
        self.stage, self.method = 'clusters', ''
        units, mapping, missing = [], {}, []
        for condition in CONDITIONS:
            extract_path = self.store.path('extract', self.ident, condition)
            support_path = self.store.path('support', self.ident, condition)
            if not extract_path.exists() or not support_path.exists():
                missing.append(condition)
                continue
            labels = {r['unit_id']: r for r in read(support_path)['units']}
            for unit in read(extract_path)['units']:
                if unit['substantive'] and not unit['paraphrase']:
                    key = f'X{len(mapping)+1:05d}'
                    mapping[key] = {'condition': condition, 'unit_id': unit['unit_id'], 'support': labels.get(unit['unit_id'])}
                    units.append({'unit_key': key, 'quote': unit['quote']})
        if missing:
            result = {'clusters': [], 'mapping': mapping, 'missing_conditions': missing}
            write(artifact(self.config.output, 'partial_information_clusters', self.ident), result)
            return result
        random.Random(self.config.seed).shuffle(units)
        async def local(group: list[dict], index: int) -> dict:
            value = await self.ask('clusters', {'units': group}, old.Clusters,
                                   f'partition_{index:04d}', PROMPTS['clusters'])
            expected = {u['unit_key'] for u in group}
            assigned = [k for c in value['clusters'] for k in c['unit_keys']]
            if set(assigned) != expected or len(assigned) != len(expected):
                value = await packed_ask(self, 'clusters', {'units': group}, old.Clusters,
                    f'partition_complete_{index:04d}', PROMPTS['clusters']+
                    ' Preserve every supplied unit_key exactly once; do not omit uncertain or erroneous wording.')
                assigned = [k for c in value['clusters'] for k in c['unit_keys']]
                if set(assigned) != expected or len(assigned) != len(expected):
                    raise ValueError('Local information partition still has missing/duplicate units')
            return value
        values = await self.map('anonymous_information', batches(units, 8, 6500), local)
        originals = {f'L{i:04d}': c for i, c in enumerate(
            (c for v in values for c in v['clusters']), 1)}
        candidates = [{'candidate_id': k, 'summary': c['summary'], 'kind': c['kind']}
                      for k, c in originals.items()]
        groups = await identity_groups(self, candidates, 'merge_identities')
        result = {'clusters': [{'cluster_id': f'I{i:04d}', 'summary': g['summary'],
            'kind': g['kind'], 'scope': g['scope'], 'unit_keys': [u for k in g['member_ids']
            for u in originals[k]['unit_keys']]} for i, g in enumerate(groups, 1)]}
        if any(c['kind'] not in {'historical_increment', 'cross_work_relation', 'scope_correction',
                                'cross_contribution'} for c in result['clusters']):
            raise ValueError('Unknown information identity kind')
        result.update(mapping=mapping, missing_conditions=missing)
        # Partial preparation is useful, but cannot become a completed seven-condition partition.
        if not missing:
            write(path, result)
        else:
            write(artifact(self.config.output, 'partial_information_clusters', self.ident), result)
        return result

    async def cross_relations(self, clusters: dict[str, Any]) -> dict[str, Any]:
        path = artifact(self.config.output, 'cross_relations', self.ident)
        if path.exists():
            return read(path)
        candidates = []
        for cluster in clusters['clusters']:
            if cluster['kind'] != 'cross_contribution':
                continue
            for key in cluster['unit_keys']:
                ref = clusters['mapping'][key]
                unit = next(u for u in self.get('extract', ref['condition'])['units'] if u['unit_id'] == ref['unit_id'])
                candidates.append({'unit_key': key, 'cluster_id': cluster['cluster_id'], 'condition': ref['condition'], 'unit': unit})
        async def condition_job(condition: str) -> list[dict[str, Any]]:
            rows = [c for c in candidates if c['condition'] == condition]
            evaluation = Evaluation(self.config, self.ident, self.engine)
            evaluation.stage, evaluation.method = 'support', condition
            async def one(group: list[dict], index: int) -> dict:
                anonymous = [{'unit_key': r['unit_key'], 'unit': r['unit']} for r in group]
                return await evaluation.ask('support', {'units': anonymous, 'claims': evaluation.get('claims')['claims'],
                    **evaluation.object_material([r['unit'] for r in group], 6000)}, Relations,
                    f'cross_relations_{index:04d}', 'For each supplied unit_key verify the cross-contribution relation '
                    'ACTUALLY stated in its original quote. Identify at least two distinct target claim IDs only '
                    'when the relation explicitly involves them. Verify using manuscript and original history, '
                    'not generated graph claim_text. Do not narrow or repair the assertion to make it supported. '
                    'If no such supported relation is present, retain not_verifiable/partly_supported and explain.')
            values = await evaluation.map('cross_relations', batches(rows, 4, 6500), one)
            returned = [r for v in values for r in v['items']]
            if {r['unit_key'] for r in returned} != {r['unit_key'] for r in rows}:
                raise ValueError('Cross-contribution verification missing input units')
            allowed = {c['claim_id'] for c in evaluation.get('claims')['claims']}
            for relation in returned:
                candidate = next(r for r in rows if r['unit_key'] == relation['unit_key'])
                relation.update(condition=condition, cluster_id=candidate['cluster_id'])
                relation['claim_ids'] = sorted(set(relation['claim_ids']) & allowed)
                relation['original_quote'] = candidate['unit']['quote']
                relation['valid'] = (relation['support'] == 'supported' and relation['scope_correct'] is True
                                     and len(relation['claim_ids']) >= 2 and bool(relation['quote'])
                                     and relation['quote'] in candidate['unit']['quote'])
            return returned
        values = await asyncio.gather(*(condition_job(c) for c in CONDITIONS))
        results = [r for rows in values for r in rows]
        result = {'items': results, 'missing_conditions': clusters['missing_conditions']}
        if not clusters['missing_conditions']:
            write(path, result)
        return result

    async def branch_inputs(self) -> dict[str, Any]:
        path = artifact(self.config.output, 'branch_units', self.ident)
        if path.exists():
            return read(path)
        inputs = read(artifact(self.config.output, 'condition_inputs', self.ident, 'F'))
        findings = inputs['findings']
        self.stage, self.method, self.pool_condition = 'extract', 'branch_input', 'F'
        await self.prepare_branch_atoms(findings)
        async def one(finding: dict[str, Any], index: int) -> dict[str, Any]:
            value = read(self.atom_checkpoint(index))
            located = await locate_quotes(self, finding['finding']['text'],
                {str(i): u['quote'] for i, u in enumerate(value['units'])}, f'branch_locations_{index:04d}')
            if any(q is None for q in located.values()):
                raise ValueError('Branch atomic assertion could not be located in actual input')
            for i, unit in enumerate(value['units']):
                unit.update(quote=located[str(i)], finding_key=finding['finding_key'],
                            origin_branch=finding['origin_branch'], needs_verification=True)
            return value
        values = await self.map('actual_branch_inputs', findings, one)
        units = [u for v in values for u in v['units']]
        for i, unit in enumerate(units, 1):
            unit['unit_id'] = f'B{i:05d}'
        write(artifact(self.config.output, 'branch_report', self.ident),
              {'body': '\n'.join(f['finding']['text'] for f in findings),
               'cited_source_ids': sorted({key for f in findings for key in f['finding']['evidence_keys']})})
        self.store.put('extract', self.ident, {'units': units, 'predictions': []}, 'branch_input')
        old_store = Store(self.config.model_copy(update={'output': self.config.source}))
        reusable = {}
        for branch in ('gear', 'graph'):
            original = {u['unit_id']: u for u in old_store.get('extract', self.ident, branch)['units']}
            for label in old_store.get('support', self.ident, branch)['units']:
                unit = original[label['unit_id']]
                reusable[(branch, unit['quote'], tuple(sorted(unit['claim_ids'])))] = label
        labels, pending = [], []
        for unit in units:
            key = (unit['origin_branch'], unit['quote'], tuple(sorted(unit['claim_ids'])))
            if key in reusable:
                labels.append({**reusable[key], 'unit_id': unit['unit_id'], 'evaluation_reused': True})
            else:
                pending.append(unit)
        self.stage = 'support'
        values = await self.map('branch_support', batches(pending, 8, 4500),
            lambda group, i: self.support_group('branch_input', group, i))
        labels.extend(r for v in values for r in v['units'])
        if {u['unit_id'] for u in units} != {r['unit_id'] for r in labels}:
            raise ValueError('Actual branch input support labels are incomplete')
        self.store.put('support', self.ident, {'units': labels}, 'branch_input')
        result = {'units': units, 'support': labels, 'findings': findings}
        write(path, result)
        self.pool_condition = ''
        return result

    def atom_checkpoint(self, index: int) -> Path:
        return (self.config.output/'annotations/checkpoints/extract/branch_input'/self.ident/
                f'branch_atoms_{index:04d}.json')

    async def prepare_branch_atoms(self, findings: list[dict[str, Any]]) -> None:
        grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
        for index, finding in enumerate(findings):
            if self.atom_checkpoint(index).exists():
                continue
            ids = finding.get('claim_ids', [finding['claim_id']])
            key = (finding['origin_branch'], tuple(sorted(ids)))
            grouped.setdefault(key, []).append({'index': index, 'finding_key': finding['finding_key'],
                'original_finding': finding['finding']['text'], 'claim_ids': ids})
        groups = [group for rows in grouped.values() for group in batches(rows, 4, 12000)]
        async def one(group: list[dict[str, Any]], index: int) -> dict[str, Any]:
            if all(self.atom_checkpoint(r['index']).exists() for r in group):
                return {}
            result = await exact_items(self, 'extract', {'findings': group}, FindingAtomBatches,
                f'branch_atoms_batch_{group[0]["index"]:04d}',
                'For EACH finding_key independently extract atomic scientific assertions verbatim ONLY '
                'from its original_finding, including Joint observations, interpretations and substantive '
                'limitations. Return extraction.predictions=[]; use only that finding supplied claim_ids. '
                'Keep source locators and necessary conditions and qualifiers. Never merge findings, '
                'transfer assertions between findings or rewrite '
                'claims. Return every finding_key exactly once.', 'finding_key',
                {r['finding_key'] for r in group})
            values = {r['finding_key']: r['extraction'] for r in result['items']}
            for row in group:
                path = self.atom_checkpoint(row['index'])
                if not path.exists():
                    write(path, values[row['finding_key']])
            return result
        await self.map('actual_branch_atom_batches', groups, one)

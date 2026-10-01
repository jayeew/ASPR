"""Offline Fig.3 regression tests: no real CLI, models, network or graph loading."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from figure_pipeline.fig3_revision import models as m
from figure_pipeline.fig3_revision.aggregate import costs, paired_preference
from figure_pipeline.fig3_revision.client import Client
from figure_pipeline.fig3_revision.config import METHODS, Config
from figure_pipeline.fig3_revision.data import prepare_paper
from figure_pipeline.fig3_revision.execution import Engine
from figure_pipeline.fig3_revision.materials import (
    blocks,
    catalog,
    encoded,
    fragments,
)
from figure_pipeline.fig3_revision.runner import (
    GROUPS,
    execute,
    fresh,
    selected_ids,
    tasks,
)
from figure_pipeline.fig3_revision.storage import Store, read, write


@pytest.fixture
def config(tmp_path: Path) -> Config:
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    paper, review = dataset/'paper.md', dataset/'review.md'
    paper.write_text('We introduce a bounded method. It improves accuracy on two datasets. Limitations apply.')
    review.write_text('Reviewer 1, round 1: compare prior methods. Author: we added a comparison.')
    row = {'paper_id': 'p001', 'title': 'A method', 'doi': '10.test/target', 'publication_date': '2026-01-01',
           'paper_path': str(paper), 'review_path': str(review), 'field_name': 'Science'}
    (dataset/'papers.jsonl').write_text(json.dumps(row)+'\n')
    return Config(dataset=dataset, output=tmp_path/'output', device='cpu', workers=8, cli_limit=8,
                  cli_initial=8, memory_reserve_gib=0, task_memory_gib=.001, bootstrap_repeats=10)


def fake_transport(**kwargs: Any) -> dict[str, Any]:
    schema, payload = kwargs['response_schema']['title'], json.loads(kwargs['user'])
    if schema == 'Digest':
        return {'text': 'Bounded method improves two datasets; limitations remain. Source MANUSCRIPT M00001.',
                'source_ids': ['MANUSCRIPT'], 'limitations': []}
    if schema == 'SharedClaims':
        return {'claims': [{'claim_id': 'C', 'claim_type': 'METHOD', 'normalized_claim_text': 'Bounded method',
                            'author_claim_text': 'We introduce a bounded method.', 'source_span_ids': ['M00001'],
                            'manuscript_quote': 'We introduce a bounded method.'}]}
    if schema == 'ManuscriptSupport':
        return {'status': 'supported', 'supported_scope': 'two datasets', 'evidence': [], 'reason': 'manuscript'}
    if schema == 'HistoricalRelations':
        return {'relations': [{'work_id': prior['work_id'], 'relation': 'PARTIAL_ANTECEDENT',
            'common_dimensions': ['method'], 'difference_dimensions': ['scope'], 'essential_facet_coverage': .5,
            'evidence': [], 'rationale': 'partial'} for prior in payload['prior_works']]}
    if schema == 'AntecedentVerification':
        return {'confirmed': False, 'missing_facets': ['second dataset'], 'rationale': 'scope differs'}
    if schema == 'Assessment':
        return {'claim_id': payload['claim_id'], 'claim_text': payload['claim_text'], 'supported_scope': 'two datasets',
            'findings': [{'dimension': 'increment', 'text': 'bounded, unresolved', 'evidence_keys': ['MANUSCRIPT'], 'stance': 'unresolved'}],
            'overall_stance': 'unresolved', 'overall_reason': 'limited evidence', 'limitations': []}
    if schema == 'JointAnalysis':
        return {'paper_id': 'p001', 'knowledge_summary': 'No eligible neighbors', 'findings': [
            {'question': 'joint structure', 'claim_ids': ['p001::CLAIM::01'], 'observation': 'empty union',
             'interpretation': 'unresolved', 'evidence_keys': ['JOINT_GRAPH:p001'], 'limitations': ['empty']}], 'limitations': ['empty']}
    if schema == 'Analysis':
        return {'analysis': 'Bounded contribution with unresolved historical comparison.', 'evidence': [], 'limitations': []}
    if schema == 'Report':
        return {'body': 'We introduce a bounded method. Historical comparison remains unresolved.', 'cited_source_ids': ['MANUSCRIPT']}
    if schema == 'Queries':
        return {'queries': ['prior bounded method'], 'cited_works': []}
    if schema == 'Core':
        return {'status': 'selected', 'reason': 'Main contribution', 'items': [{'core_id': 'C01', 'description': 'Bounded method',
            'manuscript_quote': 'We introduce a bounded method.', 'selection_reason': 'Main method'}]}
    if schema == 'CoreMapping':
        return {'items': [{'core_id': 'C01', 'claim_ids': ['p001::CLAIM::01'], 'reason': 'same method'}]}
    if schema == 'ReviewSections':
        return {'sections': [{'section_id': 'R', 'role': 'reviewer', 'reviewer_id': '1', 'round_number': 1,
                             'manuscript_scope': 'current', 'identity_explicit': True, 'quote': 'Compare prior methods.'}]}
    if schema == 'Checklist':
        return {'concerns': [{'concern_id': 'C0001', 'section_id': 'R0001', 'reviewer_id': '1', 'round_number': 1,
            'quote': 'Compare prior methods.', 'object_description': 'method', 'scope': 'two datasets', 'dimension': 'novelty',
            'stance': 'unresolved', 'reasons': ['comparison absent'], 'reason_quotes': ['Compare prior methods.'],
            'prior_work_queries': [], 'applies_to_input': 'yes', 'manuscript_scope': 'current', 'version_reason': 'explicit', 'evidence': []}]}
    if schema == 'Tones':
        return {'items': [{'section_id': 'R0001', 'tone': 'neutral', 'scientific_stance': 'unresolved', 'quote': 'Compare prior methods.', 'reason': 'request'}]}
    if schema == 'ReviewDynamics':
        return {'changes': []}
    if schema == 'Reference':
        return {'items': [{'core_id': 'C01', 'state': 'insufficient_material', 'common_ground': 'method', 'difference': 'uncertain',
            'necessary_scope': 'two datasets', 'historical_comparison_applicable': True, 'scope_applicable': True,
            'known_antecedent_ids': [], 'evidence': [], 'limitations': ['limited literature']}]}
    if schema == 'Units':
        return {'units': [{'unit_id': 'U0001', 'quote': 'We introduce a bounded method.', 'claim_ids': ['p001::CLAIM::01'],
            'needs_verification': True, 'kind': 'manuscript_fact', 'insight_type': 'scope_correction', 'substantive': True,
            'paraphrase': False, 'original_locator': 'MANUSCRIPT'}], 'predictions': [{'core_id': 'C01', 'state': 'explicit_abstention',
            'quotes': ['Historical comparison remains unresolved.'], 'conflicting_quotes': [], 'asserts_increment': False,
            'difference': '', 'stated_scope': 'two datasets'}]}
    if schema == 'Support':
        return {'units': [{'unit_id': u['unit_id'], 'support': 'supported', 'scope_correct': True, 'evidence': [],
                           'originally_substantiated': True, 'original_locator': 'MANUSCRIPT', 'nonparaphrase_insight': True,
                           'errors': [], 'reason': 'bounded'} for u in payload['units']]}
    if schema == 'Novelty':
        return {'items': [{'core_id': 'C01', 'difference_correct': None, 'scope_correct': True, 'historical_comparison_correct': None,
            'false_firstness': False, 'false_antecedence': False, 'effective_increment': None, 'material_overclaim': False,
            'antecedent_ids_covered': [], 'evidence': [], 'reason': 'insufficient'}]}
    if schema in {'QualityChecklist', 'Quality'}:
        from figure_pipeline.fig3_revision.aggregate import DIMENSIONS
        return {'dimensions': [{'dimension': d, **({'criteria': ['faithfulness'], 'evidence': []} if schema == 'QualityChecklist'
                else {'score': 2, 'report_quotes': ['bounded'], 'evidence': [], 'reason': 'bounded'})} for d in DIMENSIONS]}
    if schema == 'ConcernMatches':
        return {'matches': [{'concern_id': c['concern_id'], 'scope': 'same', 'report_quote': 'Historical comparison remains unresolved.',
            'reason_coverage': 'partial', 'stance': 'unresolved', 'disagreement': 'unresolved', 'evidence': [], 'reason': 'limited'}
            for c in payload['checklist']['concerns']]}
    if schema == 'Clusters':
        keys = [u['unit_key'] for u in payload.get('units', [])]
        keys += [k for group in payload.get('candidate_clusters', []) for c in group['clusters'] for k in c['unit_keys']]
        return {'clusters': [{'cluster_id': 'I1', 'unit_keys': keys, 'summary': 'scope', 'kind': 'scope_correction'}] if keys else []}
    if schema == 'Importance':
        return {'items': [{'cluster_id': c['cluster_id'], 'important': True, 'reason': 'scope'} for c in payload['clusters']]}
    if schema == 'ErrorClusters':
        return {'clusters': []}
    if schema == 'FusionJudgment':
        return {'errors': [], 'newly_introduced_error_keys': [], 'reason': 'no identified errors'}
    if schema == 'Preference':
        return {**{k: 'tie' for k in ('overall', 'increment_clarity', 'evidence_traceability', 'knowledge_usefulness', 'appropriate_limitations')},
                'quote_a': 'bounded', 'quote_b': 'bounded', 'reason': 'same'}
    if schema == 'Controls':
        return {'pairs': [{'kind': 'scope_deletion', 'original': 'Works on two datasets', 'altered': 'Works',
                           'original_source_ids': ['MANUSCRIPT'], 'altered_source_ids': ['MANUSCRIPT'], 'alteration_reason': 'deleted scope'}]}
    if schema == 'ControlVerdict':
        return {'support': 'partly_supported', 'scope_correct': False, 'errors': ['omitted_scope'], 'reason': 'scope', 'evidence': []}
    raise AssertionError(schema)


def test_blocks_preserve_every_character_and_bound_single_long_line() -> None:
    text = 'a'*10001+'\n中文。'*4000
    parts = blocks(text)
    assert ''.join(p['text'] for p in parts) == text
    assert max(len(p['text']) for p in parts) <= 3000
    assert all(p['text'] == text[p['start']:p['end']] for p in parts)


def test_duplicate_text_retains_both_source_identities() -> None:
    result = catalog([{'source_id': 'S1', 'passage': 'same text', 'source_type': 'abstract'},
                      {'source_id': 'alias', 'passage': 'same text', 'source_type': 'fulltext'}])
    assert len(result['blocks']) == 1
    assert len(result['sources']) == 2
    assert result['sources'][0]['block_ids'] == result['sources'][1]['block_ids']
    parts = fragments({'source_id': 'S1', 'passage': 'x'*9000})
    assert all(p['identity']['source_id'] == 'S1' for p in parts)


def test_prepare_never_reads_old_target_results(config: Config) -> None:
    row = json.loads((config.dataset/'papers.jsonl').read_text())
    (config.dataset/'data/inputs').mkdir(parents=True)
    (config.dataset/'data/inputs/p001.json').write_text('BROKEN OLD MODEL INPUT')
    value = prepare_paper(config, row)
    assert value['claims'] == []
    assert 'reports' not in value and 'raw_retrieval' not in value
    assert value['manuscript'].startswith('We introduce')


def test_fresh_removes_outputs_not_input_assets(config: Config) -> None:
    for name in ('inputs', 'reports', 'annotations', 'evidence', 'derived', 'figures', 'logs'):
        write(config.output/name/'old.json', {'old': True})
    fresh(config)
    assert not list(config.output.rglob('*.json'))
    assert (config.dataset/'paper.md').is_file()
    assert selected_ids(config) == {'p001'}
    assert len(tasks(config, 'preference')) == 10


def test_parallel_children_resume_and_no_heartbeat(config: Config, capsys: Any) -> None:
    current = peak = 0
    lock = threading.Lock()
    def transport(**kwargs: Any) -> dict[str, Any]:
        nonlocal current, peak
        with lock:
            current += 1; peak = max(peak, current)
        time.sleep(.03)
        with lock:
            current -= 1
        return fake_transport(**kwargs)
    async def run() -> None:
        engine = Engine(config, 4, transport)
        async def action(payload: Any, i: int) -> Any:
            return await engine.ask('p001', 'test', '', f'part_{i}', 'read', 'Read', payload, m.Digest)
        await engine.map('p001', 'test', '', 'parts', [{'x': i} for i in range(8)], action)
        await engine.close()
        engine2 = Engine(config, 4, lambda **kw: pytest.fail('Completed step invoked again'))
        for i in range(8):
            await engine2.ask('p001', 'test', '', f'part_{i}', 'read', 'Read', {}, m.Digest)
        await engine2.close()
    asyncio.run(run())
    assert 2 <= peak <= 4
    output = capsys.readouterr().out
    assert '[parts 8/8]' in output and '心跳' not in output


def test_queued_slot_survives_poll_timeout(config: Config) -> None:
    async def run() -> None:
        engine = Engine(config, 1)
        try:
            await engine.acquire()
            waiting = asyncio.create_task(engine.acquire())
            await asyncio.sleep(1.15)
            assert not waiting.done()
            await engine.release('completed')
            await asyncio.wait_for(waiting, 1)
            assert engine.active == 1
            await engine.release('completed')
        finally:
            await engine.close()
    asyncio.run(run())


def test_cancelled_children_count_as_finished(config: Config, capsys: Any) -> None:
    async def run() -> None:
        engine = Engine(config, 1)
        started = asyncio.Event()
        async def action(payload: Any, index: int) -> None:
            started.set()
            await asyncio.Event().wait()
        batch = asyncio.create_task(engine.map('p001', 'test', '', 'cancel_parts', [1, 2], action))
        await started.wait()
        batch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch
        await engine.close()
    asyncio.run(run())
    output = capsys.readouterr().out
    assert '[cancel_parts 2/2]' in output and '取消2' in output
    assert '[cancel_parts 0/2]' not in output


def test_large_payload_and_repair_never_bypass_budget(config: Config) -> None:
    observed = []
    def transport(**kwargs: Any) -> dict[str, Any]:
        assert len(kwargs['user']) <= config.material_max_chars
        total = kwargs['system']+kwargs['user']+encoded(kwargs['response_schema'])
        assert len(total) <= config.request_max_chars
        assert len(total.encode()) <= config.request_max_bytes
        observed.append(len(kwargs['user']))
        return fake_transport(**kwargs)
    async def run() -> None:
        engine = Engine(config, 8, transport)
        await engine.ask('p001', 'test', '', 'big', 'baseline', 'Analyze the evidence',
                         {'source_id': 'S', 'passage': 'some text '*20000}, m.Analysis)
        await engine.close()
    asyncio.run(run())
    assert len(observed) > 2
    calls = 0
    def bad(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {'unparseable': 'x'*30000}
    with pytest.raises(ValueError, match='拆分'):
        Client(config, bad).call('read', 'Read', {'x': 'small'}, m.Digest)
    assert calls == 1


def test_dependency_failure_blocks_descendants_only(config: Config) -> None:
    class FakeStages:
        def __init__(self, *args: Any) -> None:
            pass
        async def run(self, stage: str, method: str) -> Any:
            if stage == 'review_sections':
                raise RuntimeError('expected failure')
            if stage == 'checklist':
                pytest.fail('Blocked child ran')
            return {'ok': True}
    result = asyncio.run(execute(config, ('prepare', 'review_sections', 'checklist', 'direct_a'), 4,
                                 {'p001'}, stage_factory=FakeStages))
    assert result == {'completed': 2, 'failed': 1, 'blocked': 1}


def test_complete_pipeline_with_fake_models(config: Config, monkeypatch: Any) -> None:
    from figure_pipeline.fig3_revision.native_graph import NativeGraph
    from figure_pipeline.fig3_revision.retrieval import Retriever
    async def search(self: Any, *args: Any, **kwargs: Any) -> Any:
        return {'sources': [{'source_id': 'PRIOR', 'source_type': 'abstract', 'passage': 'Prior method only tested one dataset.',
                            'doi': '10.test/prior', 'publication_date': '2020-01-01', 'title': 'Prior method'}],
                'queries': ['prior'], 'failures': [], 'network_calls': 0, 'fulltext_count': 0, 'candidate_count': 1}
    def graph(self: Any, data: Any, claims: Any) -> Any:
        return {'cards': [{'claim': {'claim_id': c['claim_id']}, 'neighbors': []} for c in claims],
                'joint': {'historical_neighbor_count': 0, 'claims_without_neighbors': [c['claim_id'] for c in claims]}}
    monkeypatch.setattr(Retriever, 'search_async', search)
    monkeypatch.setattr(NativeGraph, 'compute', graph)
    captured = []
    def transport(**kwargs: Any) -> Any:
        captured.append((kwargs['response_schema']['title'], json.loads(kwargs['user'])))
        return fake_transport(**kwargs)
    for group in ('generate', 'evaluate', 'recheck'):
        result = asyncio.run(execute(config, GROUPS[group], 8, {'p001'}, transport=transport))
        assert not result.get('failed') and not result.get('blocked'), (group, result)
    store = Store(config)
    for method in METHODS:
        assert store.path('full' if method == 'fusion' else method, 'p001').is_file()
    assert len(store.get('claims', 'p001')['claims']) == 1
    assert len(store.get('recheck', 'p001')['checks']) == 18
    for title, payload in captured:
        if title == 'ControlVerdict':
            assert not {'kind', 'side', 'original', 'altered', 'first', 'second'} & payload.keys()
        if title in {'Support', 'Novelty', 'ConcernMatches'}:
            assert 'first' not in payload and 'second' not in payload
    a = store.get('preference', 'p001', 'gear_AB'); b = store.get('preference', 'p001', 'gear_BA')
    assert paired_preference(a, b, 'overall') == 'tie'
    from figure_pipeline.fig3_revision.aggregate import aggregate
    aggregate(config, {'p001'})
    summary = read(config.output/'derived/summary.json')
    assert summary['papers'] == 1 and summary['shared_claims'] == 1
    assert len(list((config.output/'derived/components').glob('*.json'))) == 40


def test_broken_log_does_not_block_costs(config: Config) -> None:
    path = config.output/'logs/calls/calls.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('{broken\n'+json.dumps({'paper_id': 'p001', 'stage': 'test', 'state': 'failed', 'usage': None, 'seconds': 1})+'\n')
    costs(config)
    assert len(read(config.output/'derived/unreadable_calls.json')) == 1
    assert ',0,0,' not in (config.output/'derived/costs.csv').read_text()


def test_cli_stream_files_error_and_timeout(config: Config, tmp_path: Path) -> None:
    script = tmp_path/'fake-codex'
    script.write_text('#!/usr/bin/env python3\nimport sys,json\nfrom pathlib import Path\nsys.stdin.read()\n'
        'print(json.dumps({"type":"thread.started","thread_id":"fake"}),flush=True)\n'
        'Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text(json.dumps({"text":"ok","source_ids":[],"limitations":[]}))\n'
        'print(json.dumps({"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}),flush=True)\n')
    script.chmod(0o755)
    config.executable = str(script)
    result = Client(config).call('read', 'Read', {}, m.Digest)
    assert result['text'] == 'ok'
    record = read(next((config.output/'logs/calls').glob('*/record.json')))
    assert record['session_id'] == 'fake' and record['usage']['input_tokens'] == 12
    assert record['last_event_at'] >= record['first_event_at']


def test_cli_timeout_reaps_process_group(config: Config, tmp_path: Path) -> None:
    import os
    import subprocess
    script = tmp_path/'fake-hanging-codex'
    pidfile = tmp_path/'child.pid'
    script.write_text('#!/usr/bin/env python3\nimport subprocess,sys,time\nfrom pathlib import Path\n'
        'sys.stdin.read()\np=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
        f'Path({str(pidfile)!r}).write_text(str(p.pid))\ntime.sleep(60)\n')
    script.chmod(0o755)
    config.executable, config.timeout_seconds = str(script), 1
    with pytest.raises(subprocess.TimeoutExpired):
        Client(config).call('read', 'Read', {}, m.Digest)
    pid = int(pidfile.read_text())
    status = Path(f'/proc/{pid}/status')
    if status.exists():
        assert 'State:\tZ' in status.read_text()
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_adaptive_concurrency_policy(config: Config) -> None:
    config.cli_limit = 64
    config.cli_initial = 16
    async def run() -> None:
        engine = Engine(config, 64)
        for _ in range(32):
            engine.active = 1
            await engine.release('completed')
        assert engine.limit == 24
        engine.active = 1
        await engine.release('rate_limit')
        assert engine.limit == 12
        engine.window = []
        for i in range(32):
            engine.active = 1
            await engine.release('timeout' if i < 4 else 'completed')
        assert engine.limit == 4
        await engine.close()
    asyncio.run(run())


def test_script_defaults_fresh_all_and_resume_preserves(config: Config, monkeypatch: Any, tmp_path: Path) -> None:
    from figure_pipeline.fig3_revision import runner
    from figure_pipeline.fig3_revision.__main__ import main
    config_path = tmp_path/'config.json'
    config_path.write_text(config.model_dump_json())
    old = config.output/'reports/old.json'
    write(old, {'old': True})
    seen = []
    def sequence(*args: Any, **kwargs: Any) -> dict[str, int]:
        seen.append((args[1], args[3], old.exists()))
        return {'failed': 1}
    monkeypatch.setattr(runner, 'sequence', sequence)
    assert main(['--config', str(config_path), 'workflow', '--group', 'all']) == 1
    assert seen == [('generate', 'all', False)]
    write(old, {'new_checkpoint': True})
    seen.clear()
    assert main(['--config', str(config_path), 'workflow', '--group', 'all', '--resume']) == 1
    assert seen == [('generate', 'all', True)]


def test_preference_uses_identical_neutral_material_in_both_orders(config: Config) -> None:
    from figure_pipeline.fig3_revision.stages import Stages
    row = json.loads((config.dataset/'papers.jsonl').read_text())
    store = Store(config)
    store.put('prepare', 'p001', prepare_paper(config, row))
    store.put('core', 'p001', {'items': [{'core_id': 'C01', 'description': 'method'}]})
    store.put('evidence_pool', 'p001', catalog([{'source_id': 'GEAR_SOURCE', 'source_type': 'abstract',
              'passage': 'Prior method', 'publication_date': '2020-01-01'}]))
    store.put('full', 'p001', {'body': 'Full report', 'cited_source_ids': []})
    store.put('gear', 'p001', {'body': 'GEAR report', 'cited_source_ids': []})
    captured = []
    def transport(**kw: Any) -> Any:
        captured.append(json.loads(kw['user']))
        return fake_transport(**kw)
    async def run() -> None:
        engine = Engine(config, 4, transport)
        await asyncio.gather(*(Stages(config, 'p001', engine).run('preference', order) for order in ('gear_AB', 'gear_BA')))
        await engine.close()
    asyncio.run(run())
    assert captured[0]['evidence_blocks'] == captured[1]['evidence_blocks']
    assert captured[0]['reports']['A'] == captured[1]['reports']['B']
    assert captured[0]['reports']['B'] == captured[1]['reports']['A']
    assert 'GEAR_SOURCE' not in encoded(captured)


def test_native_graph_adapter_uses_union_and_threshold_policy(config: Config, monkeypatch: Any) -> None:
    import numpy as np

    import gear.claim_attribution as native
    from figure_pipeline.fig3_revision.native_graph import NativeGraph
    from figure_pipeline.fig3_revision.retrieval import Retriever
    from gear.review_contracts import GraphFactCard, GraphNeighbor
    calls = []
    class Encoder:
        def encode(self, texts: Any, **kwargs: Any) -> Any:
            calls.append(('encode', len(texts)))
            return np.ones((len(texts), 2))
    class DB:
        def execute(self, query: str, args: Any) -> Any:
            class Row:
                def fetchone(self) -> Any:
                    return (int(args[0][-1]),)
            return Row()
    class Runtime:
        def __init__(self, root: Any, embedding: Any, top_k: int, threshold: float) -> None:
            calls.append(('policy', top_k, threshold))
            self._claim_db = DB()
        def insert_vector(self, claim: Any, item: Any, vector: Any) -> Any:
            suffix = claim.claim_id[-1]
            neighbor = GraphNeighbor(claim_id='H'+suffix, parent_paper_id='prior', claim_type='METHOD',
                claim_text='prior', publication_date='2020-01-01', cosine_similarity=.8, semantic_rank=1)
            return GraphFactCard(claim=claim, neighbors=[neighbor], metrics=[])
        def _connections(self) -> None:
            pass
        def _neighbor_edges(self, neighbors: Any) -> Any:
            return {(1, 2)}
        def close(self) -> None:
            pass
    monkeypatch.setattr(native, 'ClaimGraphRuntime', Runtime)
    monkeypatch.setattr(Retriever, 'local', lambda self: {'encoder': Encoder()})
    row = json.loads((config.dataset/'papers.jsonl').read_text())
    data = prepare_paper(config, row)
    claims = [{'claim_id': 'C'+str(i), 'claim_type': 'METHOD', 'normalized_claim_text': 'method',
               'source_span_ids': ['M00001'], 'manuscript_quote': 'bounded'} for i in (1, 2)]
    facts = NativeGraph(config).compute(data, claims)
    assert ('policy', 10, .5) in calls and ('encode', 2) in calls
    assert facts['joint']['historical_neighbor_count'] == 2
    assert facts['joint']['historical_edges'] == [['H1', 'H2']]
    assert facts['joint']['insertion_edges'] == [['C1', 'H1'], ['C2', 'H2']]
    assert ['C1', 'C2'] not in facts['joint']['insertion_edges']


def test_ranking_accepts_missing_source_title(config: Config, monkeypatch: Any) -> None:
    from figure_pipeline.fig3_revision.retrieval import Retriever
    observed = []
    class Ranker:
        def predict(self, pairs: list[list[str]], batch_size: int) -> list[float]:
            observed.extend(pairs)
            return [0.5] * len(pairs)
    retriever = Retriever(config)
    monkeypatch.setattr(retriever, 'local', lambda: {'reranker': Ranker()})
    sources = [{'title': None, 'passage': 'Evidence'}, {'title': 'Prior work', 'passage': None}]
    result = retriever.rank(sources, {'title': 'Target', 'manuscript': 'Body'}, [], False)
    assert len(result) == 2
    assert [pair[1] for pair in observed] == ['\nEvidence', 'Prior work\n']


def seed_concern_recovery(config: Config) -> list[dict[str, Any]]:
    from figure_pipeline.fig3_revision.storage import roster
    store = Store(config)
    store.put('prepare', 'p001', prepare_paper(config, roster(config)[0]))
    sample = fake_transport(response_schema={'title': 'Checklist'}, user='{}')['concerns'][0]
    concerns = [{**sample, 'concern_id': f'C{i:04d}', 'quote': 'long exact reviewer quote '*80} for i in range(1, 7)]
    store.put('checklist', 'p001', {'concerns': concerns})
    store.put('evidence_pool', 'p001', catalog([{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'passage': 'Bounded method.'}]))
    for method in METHODS:
        store.put('full' if method == 'fusion' else method, 'p001', {'body': 'Complete report body.', 'cited_source_ids': []})
        write(config.output/'inputs/tasks/concerns'/method/'p001/真人问题匹配.json', {'tasks': [[c] for c in concerns]})
        row = fake_transport(response_schema={'title': 'ConcernMatches'}, user=encoded({'checklist': {'concerns': concerns[:1]}}))
        write(config.output/'annotations/checkpoints/concerns'/method/'p001/concerns_00000.json', row)
    return concerns


def test_object_recovery_rejects_foreign_ids_and_conflicts(config: Config) -> None:
    from figure_pipeline.fig3_revision.packing import recover_paper
    concerns = seed_concern_recovery(config)
    path = config.output/'annotations/checkpoints/concerns/gear/p001/concerns_00001.json'
    row = fake_transport(response_schema={'title': 'ConcernMatches'}, user=encoded({'checklist': {'concerns': concerns[1:2]}}))
    write(path, {'matches': row['matches'] + [{**row['matches'][0], 'concern_id': 'ALIEN'}]})
    Store(config).put('concerns', 'p001', {'matches': [{**row['matches'][0], 'reason': 'different'}]}, 'gear')
    result = recover_paper(config, 'p001')
    assert set(result['accepted']['gear']) == {'C0001'}
    assert result['conflicts']['gear'] == ['C0002']
    assert result['counts']['outside_batch'] == 1


def test_packed_concerns_share_preparation_resume_and_preserve_reports(config: Config) -> None:
    from figure_pipeline.fig3_revision.packing import activate, evaluate
    from figure_pipeline.fig3_revision.stages import Stages
    seed_concern_recovery(config)
    existing = (config.output/'annotations/checkpoints/concerns/gear/p001/concerns_00000.json').read_bytes()
    first = activate(config)
    assert first['counts']['retained'] == 6
    assert activate(config) == first
    seen = []
    def transport(**kwargs: Any) -> Any:
        payload = json.loads(kwargs['user']); title = kwargs['response_schema']['title']
        seen.append((title, payload))
        if title == 'ConcernBriefs':
            return {'concerns': [{'concern_id': c['concern_id'], 'brief': 'Prior comparison, narrow scope, unknown applicability.'} for c in payload['concerns']]}
        assert payload['report'] == 'Complete report body.'
        assert all(c['concern_id'] != 'C0001' for c in payload['checklist']['concerns'])
        assert len(payload['checklist']['concerns']) <= 4
        return fake_transport(**kwargs)
    async def run() -> None:
        engine = Engine(config, 4, transport)
        async def method_work(method: str) -> Any:
            stages = Stages(config, 'p001', engine); stages.stage = 'concerns'; stages.method = method
            return await evaluate(stages, method)
        results = await asyncio.gather(*(method_work(meth) for meth in METHODS))
        assert all(len(r['matches']) == 6 for r in results)
        await engine.close()
        engine = Engine(config, 4, lambda **kw: pytest.fail('Finished objects were called again'))
        for meth in METHODS:
            stages = Stages(config, 'p001', engine); stages.stage = 'concerns'; stages.method = meth
            assert len((await evaluate(stages, meth))['matches']) == 6
        await engine.close()
    asyncio.run(run())
    briefs = [p for t, p in seen if t == 'ConcernBriefs']
    assert sum(len(p['concerns']) for p in briefs) == 5
    assert len([t for t, p in seen if t == 'ConcernMatches']) == 12
    assert (config.output/'annotations/checkpoints/concerns/gear/p001/concerns_00000.json').read_bytes() == existing


def test_packet_index_preserves_selection(config: Config) -> None:
    from figure_pipeline.fig3_revision.materials import PacketIndex, packet
    index = catalog([{'source_id': 'A', 'source_type': 'abstract', 'passage': 'prior method limitations'},
                     {'source_id': 'B', 'source_type': 'review', 'passage': 'method similarity'}])
    cached = PacketIndex(index)
    for historical in (True, False):
        assert packet(index, 'A method', 1000, historical) == packet(index, 'A method', 1000, historical, cached)


def test_scheduler_round_robin_does_not_starve_other_stages(config: Config) -> None:
    async def run() -> None:
        engine = Engine(config, 1)
        await engine.acquire()
        order = []
        async def worker(stage: str, paper: str) -> None:
            await engine.acquire(stage, paper)
            order.append((stage, paper))
            await engine.release('completed')
        jobs = [asyncio.create_task(worker('concerns', 'p1')) for _ in range(8)]
        jobs.append(asyncio.create_task(worker('core', 'p2')))
        await asyncio.sleep(.02)
        await engine.release('completed')
        await asyncio.gather(*jobs)
        assert order.index(('core', 'p2')) <= 1
        await engine.close()
    asyncio.run(run())


def test_call_estimate_is_read_only(config: Config) -> None:
    from figure_pipeline.fig3_revision.packing import estimate
    seed_concern_recovery(config)
    before = {str(p): p.read_bytes() for p in config.output.rglob('*') if p.is_file()}
    result = estimate(config)
    after = {str(p): p.read_bytes() for p in config.output.rglob('*') if p.is_file()}
    assert before == after
    assert result['objects'] == 36 and result['retained'] == 6
    assert result['new_evaluation_max_briefs'] < result['old_unfinished_answer_batches_lower_bound']


def test_identical_neutral_reads_share_but_reports_remain_independent(config: Config) -> None:
    called = []
    def transport(**kwargs: Any) -> Any:
        called.append(json.loads(kwargs['user']))
        time.sleep(.03)
        return fake_transport(**kwargs)
    async def run() -> None:
        engine = Engine(config, 4, transport)
        async def call(method: str, path: str, suffix: str) -> Any:
            return await engine.ask('p001', 'concerns', method, suffix, 'read', 'Neutral reading',
                {'parts': [{'path': path, 'value': 'same words'}]}, m.Digest)
        await asyncio.gather(call('gear', '$.checklist.concerns[0]', 'neutral'),
                             call('graph', '$.checklist.concerns[0]', 'neutral'))
        assert len(called) == 1
        await asyncio.gather(call('gear', '$.report', 'report'), call('graph', '$.report', 'report'))
        assert len(called) == 3
        await engine.close()
    asyncio.run(run())


def test_recovery_skips_damaged_call_logs_and_preserves_files(config: Config) -> None:
    from figure_pipeline.fig3_revision.packing import recover_calls
    root = config.output/'logs/calls'
    metadata = {'state': 'completed', 'stage': 'concerns', 'paper_id': 'p001', 'method': 'gear', 'step': 'valid'}
    write(root/'good/record.json', metadata)
    write(root/'good/response.json', {'matches': []})
    (root/'empty').mkdir(parents=True)
    (root/'empty/record.json').write_text('')
    write(root/'bad_response/record.json', {**metadata, 'step': 'bad'})
    (root/'bad_response/response.json').write_text('{unfinished')
    write(root/'wrong_shape/record.json', [])
    write(root/'missing_id/record.json', {'state': 'completed', 'stage': 'concerns'})
    original = {str(p): p.read_bytes() for p in root.rglob('*.json')}
    assert recover_calls(config) == 1
    assert read(config.output/'annotations/checkpoints/concerns/gear/p001/valid.json') == {'matches': []}
    assert not (config.output/'annotations/checkpoints/concerns/gear/p001/bad.json').exists()
    assert len(read(config.output/'logs/recovery_errors.json')['errors']) == 4
    assert {str(p): p.read_bytes() for p in root.rglob('*.json')} == original
    assert recover_calls(config) == 0


def test_missing_ids_are_supplemented_without_repeating_completed_objects(config: Config) -> None:
    from figure_pipeline.fig3_revision.packing import activate, evaluate
    from figure_pipeline.fig3_revision.stages import Stages
    seed_concern_recovery(config)
    activate(config)
    calls = []
    omitted_match = False
    def transport(**kwargs: Any) -> Any:
        nonlocal omitted_match
        data = json.loads(kwargs['user']); schema = kwargs['response_schema']; title = schema['title']
        calls.append((title, data))
        if title in {'ConcernBriefs', 'MissingConcernBrief'}:
            source = data['concerns']
            if title == 'ConcernBriefs':
                source = [c for c in source if c['concern_id'] != 'C0002']
            else:
                assert len(source) == 1 and source[0]['concern_id'] == 'C0002'
                assert schema['$defs']['RequestedConcern']['properties']['concern_id']['enum'] == ['C0002']
            return {'concerns': [{'concern_id': c['concern_id'], 'brief': 'Scope, reason and uncertainty.'} for c in source]}
        result = fake_transport(response_schema={'title': 'ConcernMatches'}, user=kwargs['user'])
        if title == 'ConcernMatches' and not omitted_match:
            result['matches'] = result['matches'][1:]
            omitted_match = True
        elif title == 'MissingConcernMatch':
            assert len(data['checklist']['concerns']) == 1
            assert data['report'] == 'Complete report body.'
        return result
    async def run() -> None:
        engine = Engine(config, 4, transport)
        stages = Stages(config, 'p001', engine); stages.stage = 'concerns'; stages.method = 'gear'
        result = await evaluate(stages, 'gear')
        assert {x['concern_id'] for x in result['matches']} == {f'C{i:04d}' for i in range(1, 7)}
        await engine.close()
        engine = Engine(config, 4, lambda **kw: pytest.fail('Completed object evaluated again'))
        stages = Stages(config, 'p001', engine); stages.stage = 'concerns'; stages.method = 'gear'
        assert await evaluate(stages, 'gear') == result
        await engine.close()
    asyncio.run(run())
    assert sum(title == 'MissingConcernBrief' for title, _ in calls) == 1
    assert sum(title == 'MissingConcernMatch' for title, _ in calls) == 1


def test_recheck_batch_missing_object_resume_and_explanation_materials(config: Config) -> None:
    from figure_pipeline.fig3_revision.recheck import repeat_batch

    class Stub:
        ident = 'p001'

        def __init__(self) -> None:
            self.config = config
            self.calls: list[Any] = []
            self.cache: dict[str, Any] = {}
            self.fail_missing = True

        def support_payload(self, method: str, units: list[Any]) -> dict[str, Any]:
            return {'units': units, 'original_report_citations': ['SOURCE'],
                    'evidence_blocks': [{'block_id': 'B', 'text': 'original evidence'}]}

        def get(self, stage: str, method: str = '') -> dict[str, Any]:
            return {'units': [self.row('U1'), self.row('U2')]}

        def row(self, ident: str) -> dict[str, Any]:
            return {'unit_id': ident, 'support': 'supported', 'scope_correct': True, 'evidence': [],
                    'originally_substantiated': True, 'original_locator': 'L', 'nonparaphrase_insight': False,
                    'errors': [], 'reason': 'original'}

        async def ask(self, role: str, payload: Any, schema: Any, step: str,
                      prompt: str, synthesis: bool = False) -> Any:
            if step in self.cache:
                return self.cache[step]
            material = payload() if callable(payload) else payload
            self.calls.append((step, material))
            await asyncio.sleep(.001)
            if step.startswith('repeat_batch'):
                assert 'first' not in material and 'second' not in material
                assert len(material['evidence_blocks']) == 1
                result = {'units': [self.row('U1')]}
            elif step.startswith('missing'):
                assert len(material['units']) == 1 and material['units'][0]['unit_id'] == 'U2'
                if self.fail_missing:
                    raise RuntimeError('offline simulated failure')
                result = {'units': [{**self.row('U2'), 'support': 'contradicted'}]}
                schema.model_validate(result)
            else:
                assert synthesis
                assert material['original_task_materials']['original_report_citations'] == ['SOURCE']
                assert material['original_task_materials']['evidence_blocks'][0]['text'] == 'original evidence'
                result = {'analysis': 'A difference', 'evidence': [], 'limitations': []}
            self.cache[step] = result
            return result

    stages = Stub()
    batch = {'jobs': [{'method': 'gear', 'kind': 'support', 'object': {'unit_id': ident}} for ident in ('U1', 'U2')]}
    with pytest.raises(RuntimeError, match='未完成对象'):
        asyncio.run(repeat_batch(stages, batch, 0))
    path = config.output/'annotations/recheck_details/p001/gear/support/U1.json'
    assert path.exists()
    saved = path.read_bytes()
    stages.fail_missing = False
    result = asyncio.run(repeat_batch(stages, batch, 0))
    assert len(result) == 2 and result[1]['different']
    assert path.read_bytes() == saved
    calls = len(stages.calls)
    assert asyncio.run(repeat_batch(stages, batch, 0)) == result
    assert len(stages.calls) == calls
    assert sum(step.startswith('repeat_batch') for step, _ in stages.calls) == 1


def test_recheck_packing_preserves_methods_objects_and_budget(config: Config) -> None:
    from figure_pipeline.fig3_revision.recheck import fits_batch, merge_inputs, plan

    class Stub:
        def __init__(self) -> None:
            self.config = config

        def support_payload(self, method: str, units: list[Any]) -> dict[str, Any]:
            return {'units': units, 'original_report_citations': [method],
                    'evidence_blocks': [{'block_id': 'same', 'text': 'shared evidence'}],
                    'coverage': [{'selected_blocks': 1}]}

    stages = Stub()
    jobs = [{'method': method, 'kind': 'support', 'object': {'unit_id': ident, 'quote': 'x'*1000}}
            for method in ('gear', 'graph') for ident in ('U1', 'U2')]
    groups = plan(stages, jobs)
    assert len(groups) == 2
    assert [j for group in groups for j in group['jobs']] == jobs
    for group in groups:
        assert len({j['method'] for j in group['jobs']}) == 1
        payload = merge_inputs([stages.support_payload(j['method'], [j['object']]) for j in group['jobs']])
        assert len(payload['units']) == 2 and len(payload['evidence_blocks']) == 1
        assert fits_batch(stages, 'support', payload)
    config.material_max_chars = 1600
    assert len(plan(stages, jobs)) == 4


def test_recheck_controls_select_real_bounded_candidates_without_labels(config: Config) -> None:
    from figure_pipeline.fig3_revision.recheck import control_candidates

    class Stub:
        def get(self, stage: str, method: str) -> dict[str, Any]:
            return {'units': [{'unit_id': f'U{i}', 'quote': method+' bounded finding '+str(i)+'x'*600,
                              'kind': 'scope', 'original_locator': str(i)} for i in range(40)]}

        def report(self, method: str) -> dict[str, Any]:
            return {'cited_source_ids': [method+'_SOURCE']}

    stages = Stub()
    candidates = control_candidates(stages)
    assert len(encoded(candidates)) <= 7000
    assert 6 <= len(candidates['units']) <= 12
    assert {r['unit_key'].split('/')[0] for r in candidates['units']} == set(METHODS)
    for row in candidates['units']:
        method, ident = row['unit_key'].split('/')
        raw = next(u for u in stages.get('extract', method)['units'] if u['unit_id'] == ident)
        assert row['quote'] == raw['quote']
        assert not {'support', 'errors', 'reason', 'first', 'second'} & row.keys()


def test_recheck_missing_first_keeps_repeat_without_false_agreement(config: Config) -> None:
    from figure_pipeline.fig3_revision.recheck import finish

    class Stub:
        ident = 'p001'

        def __init__(self) -> None:
            self.config = config

        def get(self, stage: str, method: str) -> Any:
            return {'items': [{'core_id': 'CLAIM::01'}]}

        async def ask(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError('Missing first judgment must not trigger explanation or fabricate first')

    second = {'core_id': 'C02', 'reason': 'independent repeat'}
    job = {'method': 'gear', 'kind': 'novelty', 'object': {'core_id': 'C02'}}
    result = asyncio.run(finish(Stub(), job, second))
    assert result['first'] is None and result['different'] is None
    assert result['second'] == second and result['comparison_status'] == 'first_missing'
    assert result['explanation'] is None
    assert read(config.output/'annotations/recheck_details/p001/gear/novelty/C02.json') == result


def test_recheck_missing_first_excluded_from_agreement_denominator(config: Config) -> None:
    from figure_pipeline.fig3_revision.aggregate import Aggregator, component_data
    a = Aggregator(config)
    a.tables['rechecks'] = [
        {'paper_id': 'p001', 'task': 'novelty', 'different': False, 'object_id': 'C01'},
        {'paper_id': 'p001', 'task': 'novelty', 'different': None, 'object_id': 'C02'},
    ]
    result = component_data(a, [], [])['h5']
    assert result['rows'][0]['denominator'] == 1
    assert result['rows'][0]['count'] == 1
    assert result['uncomparable_rechecks'][0]['object_id'] == 'C02'


def test_novelty_missing_core_preserves_identity_after_reading(config: Config) -> None:
    from figure_pipeline.fig3_revision.materials import fits
    from figure_pipeline.fig3_revision.stages import Stages

    class NeutralEngine:
        async def reduce(self, *args: Any) -> Any:
            return [{'text': 'Existing neutral evidence', 'source_ids': ['S'], 'limitations': []}]

    class Stub(Stages):
        def __init__(self) -> None:
            self.config = config
            self.ident, self.stage, self.method = 'p001', 'novelty', 'gear'
            self.engine = NeutralEngine()
            self.calls: list[Any] = []

        def novelty_payload(self, method: str, core: Any) -> Any:
            return {'core': [core], 'reference': {'items': [{'core_id': 'C02', 'text': 'history'}]},
                    'predictions': [{'core_id': 'C02', 'text': 'prediction'}], 'report': 'x'*30000}

        async def ask(self, role: str, payload: Any, schema: Any, step: str,
                      prompt: Any = None, synthesis: bool = False) -> Any:
            if step == 'novelty_00001':
                return {'items': []}
            self.calls.append(step)
            assert payload['core'][0]['core_id'] == 'C02'
            assert payload['reference']['items'][0]['core_id'] == 'C02'
            assert payload['predictions'][0]['core_id'] == 'C02'
            assert 'bounded_reading' in payload and fits(config, payload)
            assert 'first' not in payload and 'second' not in payload
            row = {'core_id': 'C02', 'difference_correct': None, 'scope_correct': None,
                   'historical_comparison_correct': None, 'false_firstness': None,
                   'false_antecedence': None, 'material_overclaim': None, 'effective_increment': None,
                   'antecedent_ids_covered': [], 'evidence': [], 'reason': 'uncertain'}
            return schema.model_validate({'items': [row]}).model_dump()

    stages = Stub()
    result = asyncio.run(stages.novelty_one('gear', {'core_id': 'C02', 'description': 'Target'}, 1))
    assert result['items'][0]['core_id'] == 'C02'
    assert stages.calls == ['novelty_missing_C02']


def test_export_recheck_coverage_uses_frozen_objects_and_control_pairs(config: Config) -> None:
    from figure_pipeline.fig3_revision.aggregate import Aggregator
    write(config.output/'inputs/tasks/recheck/papers/p001/repeat_batches.json', {'tasks': [
        {'jobs': [{'method': 'gear', 'kind': 'novelty', 'object': {'core_id': x}} for x in ('C01', 'C02', 'C03')]}]})
    Store(config).put('recheck', 'p001', {'checks': [
        {'method': 'gear', 'task': 'novelty', 'object_id': 'C01', 'different': False},
        {'method': 'gear', 'task': 'novelty', 'object_id': 'C02', 'different': None}],
        'controls': [{'kind': 'scope_deletion', 'judgments': {'original': {}, 'altered': {}}}]})
    a = Aggregator(config)
    a.collect_recheck('p001')
    row = a.tables['recheck_coverage'][0]
    assert (row['expected'], row['observed'], row['comparable'], row['uncomparable']) == (3, 2, 1, 1)
    assert row['missing_objects'] == ['C03']
    assert {r['object_id'] for r in a.tables['missing']} == {'C02', 'C03'}
    assert sum(r['constructed'] for r in a.tables['control_coverage']) == 1
    assert sum(r['observed_judgments'] for r in a.tables['control_coverage']) == 2


def test_export_costs_handles_nonobject_records_and_unknown_usage(config: Config) -> None:
    import csv
    write(config.output/'logs/calls/broken/record.json', [])
    write(config.output/'logs/calls/failed/record.json', {'stage': 'recheck', 'method': '',
          'state': 'failed', 'seconds': None, 'usage': None})
    costs(config)
    assert len(read(config.output/'derived/unreadable_calls.json')) == 1
    with (config.output/'derived/costs.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert row['usage_complete'] == 'False' and row['usage_missing'] == '1'
    assert row['input_tokens'] == '' and row['output_tokens'] == ''


def test_export_display_handles_nulls_without_mutating_raw_rows(config: Config) -> None:
    import csv
    import xml.etree.ElementTree as ET

    from figure_pipeline.fig3_revision.aggregate import write_csv
    from figure_pipeline.fig3_revision.render import root, text
    rows = [{'quote': 'a\x00b', 'evidence': [{'quote': 'c\x00d'}]}]
    path = config.output/'display.csv'
    write_csv(path, rows)
    with path.open() as handle:
        shown = next(csv.DictReader(handle))
    assert shown['quote'] == 'a\ufffdb'
    assert json.loads(shown['evidence'])[0]['quote'] == 'c\x00d'
    assert rows[0]['quote'] == 'a\x00b'
    node = root(100, 100, 'test')
    text(node, 1, 1, 'a\x00b\x01c')
    ET.fromstring(ET.tostring(node))


def test_scientific_original_report_survives_old_packet_limit(config: Config) -> None:
    original = '原报告的具体贡献与限定条件。'*1000
    observed = []
    def transport(**kwargs: Any) -> dict[str, Any]:
        observed.append(json.loads(kwargs['user']))
        return {'body': original, 'cited_source_ids': []}
    async def run() -> None:
        engine = Engine(config, 1, transport)
        try:
            result = await engine.ask('p001', 'full', '', 'writer_original', 'baseline',
                'Preserve original report.', {'report': original, 'evidence': 'separate evidence'}, m.Report)
            assert result['body'] == original
        finally:
            await engine.close()
    asyncio.run(run())
    assert observed == [{'report': original, 'evidence': 'separate evidence'}]


def test_original_quote_copies_pdf_typography_not_paraphrase() -> None:
    from figure_pipeline.fig3_revision.scientific_tasks import original_quote
    assert original_quote('An all fibre design.', 'An all\nﬁbre design.') == 'An all\nﬁbre design.'
    with pytest.raises(ValueError):
        original_quote('The method is globally first.', 'The method works on two datasets.')


def test_scientific_extraction_does_not_exempt_substantive_assertions() -> None:
    from types import SimpleNamespace
    from figure_pipeline.fig3_revision.scientific_tasks import extract_report
    async def ask(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = fake_transport(response_schema={'title': 'Units'}, user='{}')
        row['units'][0]['needs_verification'] = False
        return row
    async def map_rows(name: str, rows: list[Any], action: Any) -> list[Any]:
        return [await action(row, i) for i, row in enumerate(rows)]
    stages = SimpleNamespace(ask=ask, map=map_rows,
        report=lambda method: {'body': 'We introduce a bounded method.', 'cited_source_ids': []},
        get=lambda stage: {'claims': [], 'items': []})
    result = asyncio.run(extract_report(stages, 'fusion'))
    assert result['units'][0]['needs_verification'] is True
    assert result['units'][0]['unit_id'] == 'U0001'


def test_reviewer_tone_and_followup_use_actual_speakers_and_later_rounds() -> None:
    from types import SimpleNamespace
    from figure_pipeline.fig3_revision.scientific_tasks import review_dynamics
    from figure_pipeline.fig3_revision.stages import Stages
    sections = [{'section_id': 'R1', 'role': 'author', 'quote': 'Everything is resolved.'},
                {'section_id': 'R2', 'role': 'reviewer', 'quote': 'Compare prior work.',
                 'reviewer_id': '1', 'identity_explicit': True, 'round_number': 1}]
    concern = {'concern_id': 'C0001', 'reviewer_id': '1', 'round_number': 2, 'stance': 'challenged'}
    async def ask(role: str, payload: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert role == 'tone'
        assert [s['section_id'] for s in payload['sections']] == ['R2']
        return {'items': []}
    async def map_rows(name: str, rows: list[Any], action: Any) -> list[Any]:
        return [await action(row, i) for i, row in enumerate(rows)]
    stages = SimpleNamespace(ask=ask, map=map_rows, get=lambda stage:
        {'sections': sections} if stage == 'review_sections' else {'concerns': [concern]})
    asyncio.run(Stages.tone(stages, ''))
    result = asyncio.run(review_dynamics(stages))
    assert result['changes'][0]['concern_id'] == 'C0001'
    assert result['changes'][0]['later_round'] is None
    assert result['changes'][0]['resolution'] == 'no_explicit_followup'


def test_review_versions_and_rebuttal_quotes_keep_source_roles() -> None:
    from figure_pipeline.fig3_revision.review_origins import bind_origins
    text = ('This file contains all reviewer reports in order by version, followed by all author rebuttals in order by version.\n'
            'Version 0:\nReviewer #1\nPlease compare the older method.\n'
            'Version 1:\nReviewer #1\nThe comparison is now adequate.\n'
            'This Peer Review File is licensed\nWe added the comparison in our response.')
    quotes = ['Please compare the older method.', 'The comparison is now adequate.',
              'We added the comparison in our response.']
    groups = [([{'section_id': f'R{i:04d}', 'quote': quote, 'role': 'reviewer',
                 'reviewer_id': 'unknown', 'round_number': 0, 'identity_explicit': False}],
               {'start': text.index(quote), 'end': text.index(quote)+len(quote)})
              for i, quote in enumerate(quotes, 1)]
    result = bind_origins(text, groups)
    assert [(r['reviewer_id'], r['round_number']) for r in result[:2]] == [('1', 1), ('1', 2)]
    assert result[2]['role'] == 'author'

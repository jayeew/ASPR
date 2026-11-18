"""One same-model retry of recorded capacity failures within the original 30-call cap."""
from __future__ import annotations

import asyncio

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_revision.models import Evaluation
from figure_pipeline.fig5_robustness.pipeline import align_evaluation
from figure_pipeline.fig5_robustness.runtime import Runner

from .evaluate import PROMPT, config
from .prepare import OUT


async def retry(runner: Runner, paper: str, condition: str) -> None:
    data = read(OUT / 'dispatched_inputs' / condition / f'{paper}.json')
    try:
        result = await runner.call(paper, 'evaluation', condition + '__capacity_retry', PROMPT,
                                   data, Evaluation, additional=True)
        current = align_evaluation(result, data)
        for c in current['candidates']:
            for q in c['questions']:
                for p in q['parts']:
                    expected = {'substantive_answer': False, 'omission': False,
                                'target_abstention': True, 'unresolved': None}[p['response_type']]
                    if p['explicitly_abstains'] is not expected:
                        p['technical_state'] = 'invalid_response_taxonomy'
        path = OUT / 'aligned' / f'{paper}.json'
        prior = read(path)['candidates'] if path.exists() else []
        replacing = {c['report_id'] for c in current['candidates']}
        write(path, {'candidates': [c for c in prior if c['report_id'] not in replacing] + current['candidates']})
    except (OSError, ValueError, RuntimeError) as exc:
        runner.status(paper, 'evaluation', condition + '__capacity_retry', 'unresolved', reason=str(exc))
        print(f'UNRESOLVED {paper}/{condition}: {str(exc)[:200]}', flush=True)


async def main() -> None:
    runner = Runner(config())
    records = [read(p) for p in (OUT / 'logs/calls').glob('*/record.json')]
    if any(r['state'] == 'running' for r in records):
        raise RuntimeError('Wait until primary dispatch completes; one ledger writer at a time')
    failed = {(r['paper_id'], r['method']) for r in records if r['state'] == 'failed'
              and 'Selected model is at capacity' in r.get('cli_error', '') and '__capacity_retry' not in r['method']}
    if len(runner.entries) + len(failed) > 30:
        raise RuntimeError('Required retries exceed hard cap; do not dispatch')
    await asyncio.gather(*(retry(runner, paper, condition) for paper, condition in sorted(failed)))


if __name__ == '__main__':
    asyncio.run(main())

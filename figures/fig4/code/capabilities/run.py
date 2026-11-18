from __future__ import annotations

import asyncio
import copy
import subprocess
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_mechanisms.io import record
from figure_pipeline.fig4_rerun.config import CONDITIONS, NAMES, Report
from figure_pipeline.fig4_rerun.engine import Engine
from figure_pipeline.fig4_rerun.run import check_evaluation

from .config import DIMENSIONS, Config, Evaluation
from .materials import payload
from .prompts import EVALUATOR, WRITER


def align_result(raw: dict[str, Any], material: dict[str, Any], body: str) -> dict[str, Any]:
    """Reuse existing quote alignment; preserve each independent failed judgment as missing."""
    result = copy.deepcopy(raw)
    try:
        check_evaluation(result, material, body)
        result['core_status'] = 'completed'
        result['core_error'] = ''
    except ValueError as exc:
        result['core_status'] = 'technical_missing'
        result['core_error'] = str(exc)
    expected = list(DIMENSIONS)
    items = {d['dimension']: d for d in result['dimensions']}
    if set(items) != set(expected) or len(result['dimensions']) != len(expected):
        raise ValueError('Dimension IDs incomplete or duplicated; raw response retained, no repair call')
    for dimension, item in items.items():
        item['model_score'], item['model_status'] = item['score'], item['status']
        try:
            applicable = material['fixed_opportunities'][dimension]
            if not applicable and (item['status'] != 'not_applicable' or item['score'] is not None):
                raise ValueError('Judgment conflicts with fixed not-applicable opportunity')
            if applicable and item['status'] == 'not_applicable':
                raise ValueError('Applicable dimension incorrectly marked not applicable')
            if (item['status'] == 'assessed') != (item['score'] is not None):
                raise ValueError('Score/status mismatch')
            if item['score'] and not item['report_quotes']:
                raise ValueError('Positive explanation score lacks a report quote')
            aligned = {'items': [{'core_id': dimension, 'addressed': bool(item['report_quotes']),
                                  'report_quotes': item['report_quotes']}]}
            check_evaluation(aligned, {'cores': [{'core_id': dimension}]}, body)
            item['report_quotes'] = aligned['items'][0]['report_quotes']
            item['quote_restorations'] = aligned['quote_restorations']
            item['quote_status'], item['technical_error'] = 'completed', ''
        except ValueError as exc:
            item['score'], item['status'] = None, 'technical_missing'
            item['quote_status'], item['technical_error'] = 'failed', str(exc)
    result['evaluation_status'] = 'completed' if (
        result['core_status'] == 'completed' and all(d['quote_status'] == 'completed'
                                                  for d in result['dimensions'])) else 'partial'
    return result


async def run(config: Config) -> None:
    ids = read(config.output / 'pilot.json')['paper_ids']
    engine = Engine(config, ids, 70)
    errors = (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError)

    async def one(paper: str, condition: str) -> None:
        stage = 'generate'
        try:
            path = config.output / 'reports' / condition / f'{paper}.json'
            if path.exists():
                report = read(path)
            else:
                raw = await engine.once(paper, condition, stage, WRITER, payload(config, paper, condition), Report)
                if not raw['body'].strip():
                    raise ValueError('Empty report; raw response retained, no retry')
                report = {**raw, 'paper_id': paper, 'condition': condition, 'configuration': NAMES[condition],
                          'model': config.model, 'effort': 'high', 'body_chars': len(raw['body']),
                          'report_length_target': None}
                write(path, report)
                path.with_suffix('.md').write_text(report['body'] + '\n', encoding='utf-8')
                record(config.output, stage, paper, condition, 'completed', body_chars=report['body_chars'])
            stage = 'evaluate'
            path = config.output / 'evaluation' / condition / f'{paper}.json'
            if not path.exists():
                material = read(config.output / 'inputs/evaluation' / f'{paper}.json')
                raw = await engine.once(paper, condition, stage, EVALUATOR,
                                        {'report': report['body'], **material}, Evaluation)
                result = align_result(raw, material, report['body'])
                write(path, {'paper_id': paper, 'condition': condition, 'configuration': NAMES[condition],
                             'model': config.model, 'effort': 'xhigh', **result})
                record(config.output, stage, paper, condition, result['evaluation_status'],
                       missing_dimensions=[d['dimension'] for d in result['dimensions']
                                           if d['status'] == 'technical_missing'], core_status=result['core_status'])
            print(f'COMPLETE {paper} {NAMES[condition]}; calls={len(engine.entries)}/70', flush=True)
        except errors as exc:
            record(config.output, stage, paper, condition, 'failed', error=str(exc))
            print(f'FAILED {stage} {paper} {NAMES[condition]}: {exc}', flush=True)

    try:
        await asyncio.gather(*(one(paper, condition) for paper in ids for condition in CONDITIONS))
    finally:
        await engine.close()

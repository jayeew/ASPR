from __future__ import annotations

import asyncio
import re
import subprocess
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig3_revision.scientific_tasks import original_quote
from figure_pipeline.fig4_mechanisms.io import record

from .config import Config, Evaluation, Report
from .engine import Engine, adopted_entries
from .materials import payload

WRITER = '''Write a complete Chinese scientific innovation analysis using only the supplied materials.
Use the SAME four sections: 主要科学贡献; 与已有研究的具体区别; 科学关系与证据; 适用范围与限制.
Identify and address the manuscript's principal contributions. State concrete distinctions where supported.
There is NO word or character target. Do not compress distinct contributions merely to shorten the report.
Retain necessary qualifications and usable source IDs/locations. Manuscript citations may use MANUSCRIPT.
If a section lacks evidence, say so briefly; do not fabricate missing history or removed graph information.
Graph proximity/connectivity/citation paths are observations, not proof of novelty, derivation or causality.
Do not mention system names, input configuration, evaluation metrics or numeric novelty scores.
Return body and cited_source_ids only. No finding tracking, supplementary analysis or intermediate summary.'''

EVALUATOR = '''Evaluate this anonymous report against EACH supplied core contribution, fixed reference,
original manuscript and original historical passages. Return exactly one item per supplied core_id.
addressed is true only if a verbatim report passage actually responds to that contribution, including an
explicit scoped abstention. A generic statement about the whole paper does not cover an unmentioned core.
Apply ONE uniform content-based coverage standard to every anonymous report:
Identify the core's scientific object, main result/relation and essential experimental context.
A concrete synonymous description that identifies this content counts as addressed. Do NOT require
verbatim repetition of the reference, a separate heading, or every reference number/method detail.
Omitting a resolution, throughput, percentage or symmetry label alone does NOT make a described
scientific result unaddressed. Require a specific quantitative detail only when it defines the core's
scientific content (for example a claimed performance advantage whose meaning depends on that number).
Do not turn a mere number omission into scope error; judge the scope actually stated by the report.
Conversely, generic high-resolution/high-performance language without the core's object and result
does not establish coverage. Do not infer missing science from the manuscript or reference.
An explicit, contribution-specific evidence limitation may address the core but does not establish
correct historical comparison. Keep addressed, historical correctness and scope correctness distinct.
Judge historical_comparison_correct and scope_correct independently; retain null for unresolved evidence.
Correct history requires a supported concrete comparison, not generic firstness or novelty language.
For an unaddressed core return addressed=false, report_quotes=[], both correctness fields=null.
Keep ALL material conflicting report statements; a correct sentence cannot erase a coexisting overclaim.
Do not rewrite, narrow or infer an absent report answer. Existing reference applicability is fixed.
Return exact report quotes, original source IDs/quotes, reasons and unresolved reasons.
Copy short, complete contiguous report sentences EXACTLY, including their words and citations.
Use separate quote entries for separate passages; never join passages or move their citations.
Evaluate only these cores. No full-report assertion extraction, information counts or relation statistics.'''


def paragraph_quote(quote: str, body: str) -> str:
    """Recover an exact sentence plus its moved citation within the same paragraph."""
    tail = re.search(r'(\[[^\]]+\]|（[^（）]+）|\([^()]+\))[。.]?$', quote.strip())
    if tail is None:
        raise ValueError('Quotation content differs from report; original response retained')
    content = quote[:tail.start()].rstrip(' 。.')
    for paragraph in body.split('\n\n'):
        try:
            sentence = original_quote(content, paragraph)
            start = paragraph.index(sentence)
            citation = original_quote(tail.group(), paragraph[start + len(sentence):])
            end = paragraph.index(citation, start + len(sentence)) + len(citation)
            return paragraph[start:end]
        except ValueError:
            continue
    raise ValueError('Quotation content differs from report; original response retained')


def check_evaluation(result: dict[str, Any], material: dict[str, Any], body: str) -> None:
    expected = {c['core_id'] for c in material['cores']}
    returned = [item['core_id'] for item in result['items']]
    if set(returned) != expected or len(returned) != len(expected):
        raise ValueError('Core IDs incomplete or duplicated; original response retained, no repair call')
    repairs = []
    # Only restore source typography/Markdown; never accept paraphrased content.
    omitted = {i for match in re.finditer(r'\*\*|(?m:^\s*\d+\.\s*)', body)
               for i in range(match.start(), match.end())}
    positions = [i for i in range(len(body)) if i not in omitted]
    plain = ''.join(body[i] for i in positions)
    for item in result['items']:
        if item['addressed'] and not item['report_quotes']:
            raise ValueError('Addressed core lacks original report quote')
        if not item['addressed'] and item['report_quotes']:
            raise ValueError('Unaddressed core unexpectedly has report quotes')
        for index, quote in enumerate(item['report_quotes']):
            method = 'local_typography_or_markdown'
            if not quote:
                raise ValueError('Empty evaluation quote; original response retained')
            try:
                restored = original_quote(quote, body)
            except ValueError:
                needle = re.sub(r'^\s*\d+\.\s*', '', quote).replace('**', '')
                try:
                    matched = original_quote(needle, plain)
                    start = plain.index(matched)
                    restored = body[positions[start]:positions[start + len(matched) - 1] + 1]
                except ValueError:
                    restored = paragraph_quote(quote, body)
                    method = 'exact_content_and_citation_same_paragraph_expanded'
            if restored != quote:
                repairs.append({'core_id': item['core_id'], 'raw_quote': quote,
                                'source_quote': restored, 'method': method})
                item['report_quotes'][index] = restored
    result['quote_restorations'] = repairs


async def run(config: Config, cohort: list[str], conditions: list[str], ceiling: int) -> None:
    engine = Engine(config, cohort, ceiling)
    errors = (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError)

    async def one(paper: str, condition: str) -> None:
        stage = 'generate'
        try:
            report_path = config.output / 'reports' / condition / f'{paper}.json'
            if report_path.exists():
                report = read(report_path)
            else:
                report = await engine.once(paper, condition, stage, WRITER, payload(config, paper, condition), Report)
                if not report['body'].strip():
                    raise ValueError('Empty report; no repair call')
                report = {**report, 'paper_id': paper, 'condition': condition, 'body_chars': len(report['body']),
                          'model': config.model, 'effort': 'high', 'report_length_target': None}
                write(report_path, report)
                report_path.with_suffix('.md').write_text(report['body'] + '\n')
                record(config.output, stage, paper, condition, 'completed', body_chars=report['body_chars'])
            stage = 'evaluate'
            evaluation_path = config.output / 'evaluation' / condition / f'{paper}.json'
            if not evaluation_path.exists():
                original = read(config.output / 'inputs/evaluation' / f'{paper}.json')
                result = await engine.once(paper, condition, stage, EVALUATOR,
                                           {'report': report['body'], **original}, Evaluation)
                check_evaluation(result, original, report['body'])
                write(evaluation_path, {'paper_id': paper, 'condition': condition, 'model': config.model,
                                        'effort': 'xhigh', 'evaluation_round': config.evaluation_round, **result})
                record(config.output, stage, paper, condition, 'completed')
            print(f'COMPLETE {paper} {condition}; adopted_calls={len(adopted_entries(engine.entries, config))}/1400; '
                  f'historical_calls={len(engine.entries)}', flush=True)
        except errors as exc:
            record(config.output, stage, paper, condition, 'failed', error=str(exc))
            print(f'FAILED {stage} {paper} {condition}: {exc}', flush=True)

    try:
        await asyncio.gather(*(one(p, c) for p in cohort for c in conditions))
    finally:
        await engine.close()

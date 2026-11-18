"""Paper-level summaries of five-dimensional analysis performance; no total score."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from . import prompts
from .config import DIMENSIONS, METHODS, METHOD_LABELS, MODEL, OUTPUT, PROTOCOL
from .evaluation import validate_checklist, validate_evaluation
from .io import fingerprint, read, write
from .models import Checklist, Evaluation, Report
from .packing import packet_sources
from .partial_reference import validate_partial_checklist, verify_partial_reference
from .selector_evaluation import EVALUATION_PROTOCOL, SELECTOR_INSTRUCTION, prepare_selector_payload
from .runner import (JUDGE_PROVENANCE_PROTOCOL, blind, build_evaluation_input, input_payload, replace_ids)

SUPPORT = ('supported', 'partly_supported', 'not_verifiable', 'contradicted')
BOOTSTRAP_REPEATS = 10000


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def estimate(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'estimate': None, 'low': None, 'high': None}
    array = np.asarray(values, dtype=float)
    # Computation is repeatable; no scientific interpretation is attached to the random state.
    rng = np.random.default_rng(20261117)
    samples = rng.integers(0, len(array), size=(BOOTSTRAP_REPEATS, len(array)))
    means = array[samples].mean(axis=1)
    return {'n': len(values), 'estimate': float(array.mean()),
            'low': float(np.quantile(means, .025)), 'high': float(np.quantile(means, .975))}


def current_request(output: Path, stage: str, paper: str, method: str,
                    expected: dict[str, Any], instruction: str | None = None) -> dict[str, Any]:
    """Require the saved successful artifact to match today's inputs and instructions."""
    folder, prompt, effort = {
        'generation': ('reports', prompts.REPORT, 'high'),
        'reference': ('references', prompts.REFERENCE, 'xhigh'),
        'evaluation': ('evaluations', prompts.EVALUATE + SELECTOR_INSTRUCTION, 'xhigh'),
    }[stage]
    if stage == 'generation':
        prompt = prompts.report_instruction(expected)
    if instruction is not None:
        if stage != 'evaluation':
            raise ValueError('Only evaluation may have a partial-reference instruction')
        prompt = instruction
    suffix = Path(method) / f'{paper}.json' if stage != 'reference' else Path(f'{paper}.json')
    request_path = output / 'requests' / stage / suffix
    meta_path = output / 'provenance' / folder / suffix
    if not request_path.is_file() or not meta_path.is_file():
        raise ValueError(f'missing_{stage}_request_or_provenance')
    request, meta = read(request_path), read(meta_path)
    if request != expected:
        raise ValueError(f'stale_{stage}_request: current inputs or upstream results changed')
    citation_protocol = EVALUATION_PROTOCOL if stage == 'evaluation' else JUDGE_PROVENANCE_PROTOCOL
    prefix = [PROTOCOL, MODEL] + ([] if stage == 'generation' else [citation_protocol])
    digest = fingerprint(prefix + [prompt, request])
    if (meta.get('request_hash') != digest or meta.get('protocol') != PROTOCOL
            or meta.get('model') != MODEL or meta.get('effort') != effort):
        raise ValueError(f'stale_{stage}_artifact: request or configuration mismatch')
    if stage != 'generation' and meta.get('provenance_protocol') != citation_protocol:
        raise ValueError(f'stale_{stage}_artifact: citation protocol mismatch')
    return request


def verify_current_result(output: Path, paper: str, method: str, report: dict[str, Any],
                          checklist: dict[str, Any], evaluation: dict[str, Any],
                          partial_reference: dict[str, Any] | None = None) -> None:
    """Reconstruct the runner's dependencies without launching a model or altering files."""
    own = read(output / 'inputs/generation' / method / f'{paper}.json')
    neutral = read(output / 'inputs/reference' / f'{paper}.json')
    current_request(output, 'generation', paper, method, input_payload(own, 'report'))
    if partial_reference is None:
        common = current_request(output, 'reference', paper, method, input_payload(neutral, 'reference'))
        validate_checklist(checklist, packet_sources(common))
    else:
        # Failed references have no successful provenance meta. verify_partial_reference
        # reconstructs their original request, model checkpoint and quote-only recovery.
        common = read(output / 'requests/reference' / f'{paper}.json')
        validate_partial_checklist(partial_reference,
                                   common['source_passages'] + partial_reference['supplemental_sources'])
    inputs = build_evaluation_input(output, paper, method, report, checklist, partial_reference)
    payload, sources = inputs['payload'], inputs['sources']
    anonymous, mapping = blind(payload)
    pointer_payload, pointer_index = prepare_selector_payload(anonymous)
    current_request(output, 'evaluation', paper, method, pointer_payload, inputs['instruction'])
    mapping_path = output / 'requests/evaluation_mapping' / method / f'{paper}.json'
    if not mapping_path.is_file() or read(mapping_path) != mapping:
        raise ValueError('stale_evaluation_mapping: anonymous-to-original mapping differs')
    if replace_ids(anonymous, mapping) != payload:
        raise ValueError('invalid_evaluation_mapping: anonymous payload does not restore exactly')
    index_path = output / 'requests/evaluation_index' / method / f'{paper}.json'
    if not index_path.is_file() or read(index_path) != pointer_index:
        raise ValueError('stale_evaluation_index: clause index differs from current material')
    meta = read(output / 'provenance/evaluations' / method / f'{paper}.json')
    if partial_reference is not None:
        failures = partial_reference['technical_failures']
        expected = {'partial_reference_fingerprint': partial_reference['artifact_fingerprint'],
                    'reference_technical_failures': failures,
                    'inference_performed': bool(inputs['allowed_dimensions']),
                    'model_evaluated_dimensions': list(inputs['allowed_dimensions']),
                    'programmatic_dimensions': [r['dimension'] for r in failures]}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError('stale_partial_evaluation: reference availability provenance differs')
    pointer_audit = meta.get('audit', {}).get('pointer_resolution')
    if not isinstance(pointer_audit, dict):
        raise ValueError('missing_evaluation_pointer_resolution')
    if pointer_audit.get('scientific_judgments_and_scores_changed') is not False:
        raise ValueError('invalid_evaluation_pointer_resolution: scientific fields changed')
    # Outputs already use original identifiers. Judge quotation failures exclude the
    # evaluation; invalid citations made by report authors remain scientific errors.
    validate_evaluation(evaluation, report, sources, checklist, inputs['technical_failures'])
    verify_pointer_audit(pointer_audit, evaluation)


def verify_pointer_audit(audit: dict[str, Any], evaluation: dict[str, Any]) -> None:
    """Require copied quotations and restored source IDs to agree with saved output."""
    expected = {}
    for row in evaluation['dimensions']:
        dimension = row['dimension']
        for i, quote in enumerate(row['report_quotes']):
            expected[(dimension, f'report_quotes[{i}]')] = ('report', quote, None)
        for i, check in enumerate(row['checks']):
            expected[(dimension, f'checks[{i}].report_quote')] = ('report', check['report_quote'], None)
            for j, source in enumerate(check['evidence']):
                expected[(dimension, f'checks[{i}].evidence[{j}]')] = (
                    'source', source['quote'], source['source_id'])
    quotes = audit.get('quotations', [])
    actual = {(r['dimension'], r['path']): (r['kind'], r['text'], r.get('source_id')) for r in quotes}
    if (len(actual) != len(quotes) or audit.get('quotation_count') != len(quotes)
            or actual != expected):
        raise ValueError('invalid_evaluation_pointer_resolution: copied quotations or restored IDs differ')


def load_result(output: Path, paper: str, method: str) -> tuple[dict[str, Any] | None, str | None]:
    paths = {'report': output / 'reports' / method / f'{paper}.json',
             'checklist': output / 'references' / f'{paper}.json',
             'evaluation': output / 'evaluations' / method / f'{paper}.json'}
    partial = not paths['checklist'].is_file() and (output / 'partial_references' / f'{paper}.json').is_file()
    if partial:
        paths['checklist'] = output / 'partial_references' / f'{paper}.json'
    absent = [kind for kind, path in paths.items() if not path.is_file()]
    if absent:
        return None, 'missing_' + '_and_'.join(absent)
    try:
        report = Report.model_validate(read(paths['report'])).model_dump()
        envelope = verify_partial_reference(output, paper) if partial else None
        checklist = Checklist.model_validate(envelope['checklist'] if envelope else read(paths['checklist'])).model_dump()
        evaluation = Evaluation.model_validate(read(paths['evaluation'])).model_dump()
        verify_current_result(output, paper, method, report, checklist, evaluation, envelope)
    except (OSError, ValueError, ValidationError, KeyError, TypeError) as exc:
        return None, f'invalid_artifact: {exc}'
    return evaluation, None


def collect(output: Path, roster: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    scores, support, missing = [], [], []
    for paper in roster:
        paper_id, cohort = paper['paper_id'], paper['cohort']
        for method in paper['methods']:
            result, error = load_result(output, paper_id, method)
            common = {'paper_id': paper_id, 'cohort': cohort, 'method': method}
            if error:
                error = rerun_failure_context(output, paper_id, method, error)
                missing.append({**common, 'reason': error})
            by_dimension = {r['dimension']: r for r in result['dimensions']} if result else {}
            for dimension in DIMENSIONS:
                row = by_dimension.get(dimension)
                scores.append({**common, 'dimension': dimension,
                               'status': row['status'] if row else 'missing',
                               'score': row['score'] if row else None,
                               'reason': row['reason'] if row else error})
                if row is not None and row['status'] != 'technical_failure':
                    support.append(support_row(common, dimension, row))
    return scores, support, missing


def rerun_failure_context(output: Path, paper: str, method: str, error: str) -> str:
    path = output / 'status/full_rerun_calls.json'
    if method != 'full' or not path.exists():
        return error
    ledger = read(path)
    tasks_path = output / 'status/tasks.json'
    tasks = read(tasks_path) if tasks_path.exists() else {}
    for stage in ('report', 'evaluation'):
        key = f'{stage}/full/{paper}'
        row = ledger.get(key, {})
        if row.get('state') != 'needs_discussion':
            continue
        detail = row.get('error', '')
        kind = ('provider_content_refusal' if 'limited access to this content for safety reasons' in detail
                else tasks.get(key, {}).get('error_type', 'model_call_failure'))
        downstream = '; downstream evaluation not run' if stage == 'report' else ''
        return f'{stage}: {kind}{downstream}; previous result excluded ({error})'
    return error


def support_row(common: dict, dimension: str, row: dict) -> dict[str, Any]:
    checks = row['checks']
    counts = Counter(check['support'] for check in checks)
    return {**common, 'dimension': dimension, 'evaluation_status': row['status'],
            'assertion_count': len(checks), **{f'{key}_count': counts[key] for key in SUPPORT},
            'supported_scope_correct_count': sum(c['support'] == 'supported' and c['scope_correct'] is True
                                                 for c in checks),
            'scope_correct_count': sum(c['scope_correct'] is True for c in checks),
            'scope_incorrect_count': sum(c['scope_correct'] is False for c in checks),
            'scope_unresolved_count': sum(c['scope_correct'] is None for c in checks)}


def summarise_scores(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['cohort'], row['method'], row['dimension'])].append(row)
    result = []
    for (cohort, method, dimension), group in grouped.items():
        statuses = Counter(r['status'] for r in group)
        values = [float(r['score']) for r in group if r['status'] == 'assessed']
        result.append({'cohort': cohort, 'method': method, 'dimension': dimension,
                       'dimension_zh': DIMENSIONS[dimension], 'expected_papers': len(group),
                       **estimate(values), 'n_unresolved': statuses['unresolved'],
                       'n_not_applicable': statuses['not_applicable'],
                       'n_technical_failure': statuses['technical_failure'], 'n_missing': statuses['missing']})
    return result


def paired_effects(rows: list[dict]) -> list[dict]:
    main = [r for r in rows if r['cohort'] == 'main']
    by_key = {(r['paper_id'], r['method'], r['dimension']): r for r in main}
    expected = {method: {r['paper_id'] for r in main if r['method'] == method} for method in METHODS}
    result = []
    for other in METHODS:
        if other == 'full':
            continue
        papers = sorted(expected['full'] & expected[other])
        for dimension in DIMENSIONS:
            differences = []
            for paper in papers:
                a, b = by_key[(paper, 'full', dimension)], by_key[(paper, other, dimension)]
                if a['status'] == b['status'] == 'assessed':
                    differences.append(float(a['score']) - float(b['score']))
            result.append({'cohort': 'main', 'method': 'full', 'comparator': other,
                           'comparison_priority': 'primary' if other in {'gear', 'graph'} else 'secondary',
                           'dimension': dimension, 'dimension_zh': DIMENSIONS[dimension],
                           'expected_pairs': len(papers), 'non_assessed_pairs': len(papers) - len(differences),
                           **estimate(differences), 'unit': 'score_points_0_to_3'})
    return result


def summarise_support(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['cohort'], row['method'], row['dimension'])].append(row)
    result = []
    for (cohort, method, dimension), group in grouped.items():
        valid = [r for r in group if r['assertion_count'] > 0]
        row = {'cohort': cohort, 'method': method, 'dimension': dimension,
               'evaluated_reports': len(group), 'reports_with_checks': len(valid),
               'assertion_count': sum(r['assertion_count'] for r in group)}
        for key in (*SUPPORT, 'supported_scope_correct'):
            row[f'{key}_count'] = sum(r[f'{key}_count'] for r in group)
            fractions = [r[f'{key}_count'] / r['assertion_count'] for r in valid]
            stats = estimate(fractions)
            row.update({f'{key}_{name}': stats[name] for name in ('estimate', 'low', 'high')})
        result.append(row)
    return result


def format_number(value: float | None) -> str:
    return 'NA' if value is None else f'{value:.3f}'


def completion_counts(scores: list[dict], cohort: str) -> dict[str, int]:
    """Separate full scientific assessments from partial reference failures."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in scores:
        if row['cohort'] == cohort:
            grouped[(row['paper_id'], row['method'])].append(row['status'])
    counts = Counter()
    for states in grouped.values():
        if 'missing' in states:
            counts['missing'] += 1
        elif all(s == 'technical_failure' for s in states):
            counts['technical_failure'] += 1
        elif 'technical_failure' in states:
            counts['partial'] += 1
        else:
            counts['completed'] += 1
    result = {f'{cohort}_{state}_evaluations': counts[state]
              for state in ('completed', 'partial', 'technical_failure', 'missing')}
    result[f'{cohort}_available_evaluations'] = counts['completed'] + counts['partial']
    return result


def markdown(summary: dict[str, Any]) -> str:
    counts = summary['completion']
    lines = ['# 五维创新分析表现：统一主比较', '',
             '评分评价报告分析得是否准确、完整、有据，不评价论文创新程度，也不计算总分。', '',
             f"主队列：{counts['main_papers']}篇，{counts['main_completed_evaluations']}/"
             f"{counts['main_expected_evaluations']}份报告具有可读取的完整评价。", '',
             f"另有{counts['main_partial_evaluations']}份部分评价，"
             f"{counts['main_technical_failure_evaluations']}份全部维度参考技术失败，"
             f"{counts['main_missing_evaluations']}份缺失。"
             '部分评价仅将确实失引维度记为technical_failure，其余维度正常保留。', '',
             '## Full与分支的逐维配对差异', '',
             '| 对照 | 维度 | 配对数 | Full−对照 | 95%区间 |',
             '|---|---|---:|---:|---|']
    if summary.get('full_rerun'):
        lines[2:2] = ['本轮仅覆盖100篇Full报告与核验；其他五种方法的500份报告及评分复用。'
                      'Full使用完整稿件和已有证据选段并集，正文不限篇幅；核验指令、清单和评分规则不变。'
                      'DAMAGE未重跑，论文草稿及其中图件副本未更新。', '']
    for row in summary['paired_effects']:
        if row['comparison_priority'] != 'primary':
            continue
        lines.append(f"| {METHOD_LABELS[row['comparator']]} | {row['dimension_zh']} | {row['n']} | "
                     f"{format_number(row['estimate'])} | [{format_number(row['low'])}, {format_number(row['high'])}] |")
    lines.extend(['', '配对分母逐维确定。未决、不适用和技术缺失不计为零分；差值单位为0–3级评分的分值。',
                  '支持状态比例先逐论文计算，再对有检查项的论文等权汇总；各维度可检查同一科学命题，不能跨维度相加解释为独立断言数。',
                  'DAMAGE单独列为案例，不纳入主队列配对比较。来源定位成功仅说明原文可查，不构成独立的人类科学验证。', '',
                  '## 数据文件', '', '- `tables/dimension_scores.csv`：所有预期槽位及状态。',
                  '- `tables/summary.csv`：逐维评分与有效分母。',
                  '- `tables/paired_effects.csv`：Full与五种对照的配对差异。',
                  '- `tables/support_summary.csv`：判断支持状态。',
                  '- `tables/technical_failures.csv`：逐维参考技术失败，独立于科学未决。',
                  '- `tables/missing.csv`：技术缺失或非法产物。', ''])
    if summary.get('missing'):
        lines.extend(['## 缺失记录', ''])
        lines.extend(f"- {row['paper_id']} / {row['method']}：{row['reason']}"
                     for row in summary['missing'])
        lines.append('')
    return '\n'.join(lines)


def aggregate(output: Path = OUTPUT) -> dict[str, Any]:
    roster = read(output / 'inputs/roster.json')
    ids = [(r['cohort'], r['paper_id']) for r in roster]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate paper in roster')
    if any(len(r['methods']) != len(set(r['methods'])) for r in roster):
        raise ValueError('Duplicate method in roster')
    scores, support, missing = collect(output, roster)
    summaries, pairs = summarise_scores(scores), paired_effects(scores)
    completion = {'main_papers': sum(r['cohort'] == 'main' for r in roster),
                  'main_expected_evaluations': sum(len(r['methods']) for r in roster if r['cohort'] == 'main'),
                  'damage_expected_evaluations': sum(len(r['methods']) for r in roster if r['cohort'] == 'damage'),
                  **completion_counts(scores, 'main'), **completion_counts(scores, 'damage')}
    technical = [row for row in scores if row['status'] == 'technical_failure']
    result = {'protocol': PROTOCOL, 'unit': 'paper', 'bootstrap_repeats': BOOTSTRAP_REPEATS,
              'score_meaning': 'Accuracy, completeness and grounding of innovation analysis; not innovation magnitude.',
              'completion': completion, 'dimensions': DIMENSIONS, 'summary': summaries,
              'paired_effects': pairs, 'support_summary': summarise_support(support),
              'technical_failures': technical, 'missing': missing}
    if (output / 'status/full_rerun.json').exists():
        run = read(output / 'status/full_rerun.json')
        result['full_rerun'] = {'generation_tasks': 100, 'evaluation_tasks': 100,
                                'baseline_reports_and_scores_reused': 500,
                                'evaluation_protocol_unchanged': True, 'damage_rerun': False,
                                'manuscript_updated': False, 'report_length_cap': None,
                                'actual_cli_calls': run.get('actual_cli_calls'),
                                'authorised_extra_calls': run.get('extra_calls', 0),
                                'completed_reports': run.get('completed_reports'),
                                'completed_evaluations': run.get('completed_evaluations')}
    tables = output / 'tables'
    write_csv(tables / 'dimension_scores.csv', scores)
    write_csv(tables / 'summary.csv', summaries)
    write_csv(tables / 'paired_effects.csv', pairs)
    write_csv(tables / 'support_by_paper.csv', support,
              list(support[0]) if support else ['paper_id', 'cohort', 'method', 'dimension', 'assertion_count'])
    write_csv(tables / 'support_summary.csv', result['support_summary'],
              list(result['support_summary'][0]) if result['support_summary'] else ['cohort', 'method', 'dimension'])
    write_csv(tables / 'missing.csv', missing, ['paper_id', 'cohort', 'method', 'reason'])
    write_csv(tables / 'technical_failures.csv', technical,
              ['paper_id', 'cohort', 'method', 'dimension', 'status', 'score', 'reason'])
    write(output / 'summary.json', result)
    (output / 'summary.md').write_text(markdown(result), encoding='utf-8')
    return result

"""Locate quotations without replacing scientific judgments with string matches."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .models import Checklist, Evaluation, Report


def line_hyphens(text: str) -> set[int]:
    return {match.start() + len(match[1]) for match in
            re.finditer(r'([A-Za-z]{3,})-[ \t]*\r?\n(?=[a-z])', text)}


def compact_typography(text: str, drop_line_hyphens: bool = False) -> tuple[str, list[int]]:
    """Map each normalised character back to the original source position."""
    soft_hyphens = line_hyphens(text) if drop_line_hyphens else set()
    chars, positions = [], []
    for position, char in enumerate(text):
        if position in soft_hyphens:
            continue
        for expanded in unicodedata.normalize('NFKC', char):
            if not expanded.isspace():
                chars.append(expanded)
                positions.append(position)
    return ''.join(chars), positions


def find_typography(needle: str, compact: str, optional_hyphens: set[int]) -> tuple[int, int] | None:
    start = compact.find(needle)
    if start >= 0:
        return start, start + len(needle)
    start = compact.find(needle[0]) if needle else -1
    while start >= 0:
        source_position, quote_position = start, 0
        while source_position < len(compact) and quote_position < len(needle):
            if compact[source_position] == needle[quote_position]:
                quote_position += 1
            elif source_position not in optional_hyphens:
                break
            source_position += 1
        if quote_position == len(needle):
            return start, source_position
        start = compact.find(needle[0], start + 1)
    return None


def typographic_span(quote: str, text: str) -> dict[str, Any] | None:
    compact, positions = compact_typography(text)
    soft_positions = line_hyphens(text)
    optional_hyphens = {i for i, position in enumerate(positions) if position in soft_positions}
    variants = [(quote, False)]
    leading = re.match(r'\s*(The|the|This|this|These|these|Those|those|We|we|Our|our)\b', quote)
    if leading:
        index = leading.start(1)
        variants.append((quote[:index] + quote[index].swapcase() + quote[index + 1:], True))
    for candidate, changed_case in variants:
        for drop_quote_hyphens in (False, True):
            needle, _ = compact_typography(candidate, drop_quote_hyphens)
            location = find_typography(needle, compact, optional_hyphens) if needle else None
            if location is not None:
                start, end = positions[location[0]], positions[location[1] - 1] + 1
                transformations = ['NFKC', 'PDF_line_hyphenation', 'whitespace']
                if changed_case:
                    transformations.append('leading_grammatical_word_case')
                return {'start': start, 'end': end, 'text': text[start:end],
                        'normalisation': 'typography:' + '+'.join(transformations)}
    return None


def locate_quote(quote: str, text: str) -> dict[str, Any] | None:
    """Return an original span, permitting audited typography but no semantic edits."""
    if not quote.strip():
        return None
    start = text.find(quote)
    if start >= 0:
        return {'start': start, 'end': start + len(quote), 'text': quote,
                'normalisation': 'none'}
    pattern = r'\s+'.join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text)
    if match is None:
        span = typographic_span(quote, text)
        if span is None and any(token in quote for token in ('\\n', '\\r', '\\t')):
            decoded = quote.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
            span = locate_quote(decoded, text)
            if span is not None:
                span['normalisation'] += '+literal_JSON_whitespace_escape'
        return span
    return {'start': match.start(), 'end': match.end(), 'text': match.group(),
            'normalisation': 'whitespace_only'}


def source_index(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        key = str(source['source_id'])
        index.setdefault(key, []).append(source)
    return index


def evidence_audit(evidence: dict[str, Any], index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    key = evidence['source_id']
    sources = index.get(key, [])
    source, span, quote_field = None, None, None
    for candidate in sources:
        candidate_span = locate_quote(evidence['quote'], candidate['text'])
        if candidate_span is not None:
            source, span = candidate, candidate_span
            span['source_start'] = int(candidate.get('start', 0)) + span['start']
            span['source_end'] = int(candidate.get('start', 0)) + span['end']
            quote_field = 'text'
            break
    if span is None:
        for candidate in sources:
            title_span = locate_quote(evidence['quote'], candidate.get('title', ''))
            if title_span is not None:
                source, span, quote_field = candidate, title_span, 'title'
                span.update(source_start=None, source_end=None)
                break
    return {'source_id': key, 'quote': evidence['quote'],
            'located': span is not None,
            'source_type': source.get('source_type') if source else None,
            'source_field': quote_field, 'quote_field': quote_field,
            'title_only': quote_field == 'title',
            'location_meaning': ('Visible bibliographic title only; does not verify the full scientific assertion.'
                                 if quote_field == 'title' else 'Text location does not establish scientific support.'),
            'span': span,
            'problem': ('unknown_source_id' if not sources else
                        'quotation_not_in_source' if span is None else None)}


def audit_report(report: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Author citation failures remain observable report errors, not exclusions."""
    value = Report.model_validate(report).model_dump()
    index = source_index(sources)
    rows = [dict(dimension=dimension['dimension'], evidence_index=i,
                 **evidence_audit(evidence, index))
            for dimension in value['dimensions']
            for i, evidence in enumerate(dimension['evidence'])]
    return {'schema_valid': True, 'citation_count': len(rows),
            'located_count': sum(row['located'] for row in rows),
            'unlocated_count': sum(not row['located'] for row in rows),
            'citations': rows, 'eligible_for_scientific_evaluation': True,
            'meaning': 'Quotation location only; neither scientific support nor correctness is certified.'}


def validate_checklist(checklist: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    value = Checklist.model_validate(checklist).model_dump()
    index = source_index(sources)
    rows = [dict(dimension=dimension['dimension'], evidence_index=i,
                 **evidence_audit(evidence, index))
            for dimension in value['dimensions']
            for i, evidence in enumerate(dimension['evidence'])]
    errors = [f"{row['dimension']} evidence {row['evidence_index']}: {row['problem']}"
              for row in rows if not row['located']]
    if errors:
        raise ValueError('Checklist citation failures: ' + '; '.join(errors))
    return {'schema_valid': True, 'citations': rows, 'all_quoted_sources_located': True,
            'meaning': 'Source-bound checklist; string matches do not establish scientific correctness.'}


def dimension_prose(row: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Actual contiguous rendered prose, stopping before the status suffix and evidence."""
    text, offsets = '', {}
    for key in ('judgment', 'rationale', 'scope'):
        if key != 'judgment':
            text += '\n\n'
        if key == 'scope':
            text += '**成立范围与判断状态：** '
        offsets[key] = len(text)
        text += row[key]
    return text, offsets


def report_fragments(report: dict[str, Any]) -> list[tuple[str, str]]:
    """Exclude evidence quotations so they cannot masquerade as author assertions."""
    result = [(key, report[key]) for key in ('contribution_overview', 'synthesis')]
    for row in report['dimensions']:
        result.extend((f"{row['dimension']}.{key}", row[key])
                      for key in ('judgment', 'rationale', 'scope'))
        result.append((f"{row['dimension']}.rendered_prose", dimension_prose(row)[0]))
    return result


def report_quote_audit(quote: str, fragments: list[tuple[str, str]]) -> dict[str, Any]:
    for field, text in fragments:
        span = locate_quote(quote, text)
        if span is not None:
            return {'quote': quote, 'located': True, 'field': field, 'span': span}
    return {'quote': quote, 'located': False, 'field': None, 'span': None}


def check_applicability(evaluation: dict[str, Any], checklist: dict[str, Any],
                        technical_failures: list[dict[str, Any]] | None = None) -> None:
    applicability = {r['dimension']: r['applicability'] for r in checklist['dimensions']}
    failed = {row['dimension']: row for row in technical_failures or []}
    for row in evaluation['dimensions']:
        if row['dimension'] in failed:
            expected = failed[row['dimension']]
            if (row['status'] != 'technical_failure' or row.get('origin') != expected['origin']
                    or row['reason'] != expected['reason']):
                raise ValueError(f"{row['dimension']}: technical failure differs from reference availability audit")
            continue
        if row['status'] == 'technical_failure':
            raise ValueError(f"{row['dimension']}: unauthorised technical failure without reference audit")
        expected_na = applicability[row['dimension']] == 'not_applicable'
        if expected_na != (row['status'] == 'not_applicable'):
            raise ValueError(f"{row['dimension']}: applicability differs from the common checklist")
        if applicability[row['dimension']] == 'unresolved' and row['status'] != 'unresolved':
            raise ValueError(f"{row['dimension']}: unresolved common reference cannot receive a score")
        if applicability[row['dimension']] == 'applicable' and row['status'] != 'assessed':
            raise ValueError(f"{row['dimension']}: an applicable common reference requires an assessed report")


def validate_evaluation(evaluation: dict[str, Any], report: dict[str, Any],
                        sources: list[dict[str, Any]],
                        checklist: dict[str, Any] | None = None,
                        technical_failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    value = Evaluation.model_validate(evaluation).model_dump()
    report_value = Report.model_validate(report).model_dump()
    if checklist is not None:
        check_applicability(value, Checklist.model_validate(checklist).model_dump(), technical_failures)
    elif any(row['status'] == 'technical_failure' for row in value['dimensions']):
        raise ValueError('Technical failure requires a common checklist and reference availability audit')
    index, fragments = source_index(sources), report_fragments(report_value)
    reports, evidence = [], []
    for row in value['dimensions']:
        dimension = row['dimension']
        reports.extend(dict(dimension=dimension, kind='dimension',
                            **report_quote_audit(q, fragments)) for q in row['report_quotes'])
        for i, check in enumerate(row['checks']):
            if check['support'] != 'not_verifiable' and not check['evidence']:
                raise ValueError(f'{dimension}: a supported/contradicted assertion needs source evidence')
            reports.append(dict(dimension=dimension, kind='assertion', check_index=i,
                                **report_quote_audit(check['report_quote'], fragments)))
            evidence.extend(dict(dimension=dimension, check_index=i,
                                 **evidence_audit(e, index)) for e in check['evidence'])
        if row['score'] is not None and row['score'] > 0 and not (row['report_quotes'] or row['checks']):
            raise ValueError(f'{dimension}: positive analysis score requires a report quotation')
    errors = [f"{r['dimension']}: evaluator quote absent from report: {r['quote'][:100]}"
              for r in reports if not r['located']]
    errors.extend(f"{r['dimension']}: {r['problem']} ({r['source_id']})"
                  for r in evidence if not r['located'])
    if errors:
        raise ValueError('Evaluation quotation failures: ' + '; '.join(errors))
    return {'schema_valid': True, 'report_quotations': reports, 'source_quotations': evidence,
            'all_report_quotes_located': True, 'all_source_quotes_located': True,
            'meaning': 'Evaluator provenance checked; scientific labels remain model judgments.'}

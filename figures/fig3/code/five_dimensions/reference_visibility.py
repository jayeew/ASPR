"""Map already-visible manuscript passages and core quotes to original positions."""
from __future__ import annotations

import copy
from typing import Any

from .evaluation import locate_quote
from .io import fingerprint


def core_quotes(packet: dict[str, Any]) -> list[tuple[str, str]]:
    result = []
    for i, core in enumerate(packet.get('core_contributions', [])):
        if isinstance(core.get('manuscript_quote'), str):
            result.append((f'core_contributions[{i}].manuscript_quote', core['manuscript_quote']))
        for j, quote in enumerate(core.get('manuscript_quotes', [])):
            if isinstance(quote, str):
                result.append((f'core_contributions[{i}].manuscript_quotes[{j}]', quote))
    return result


def unique_location(quote: str, text: str) -> tuple[dict[str, Any] | None, str]:
    span = locate_quote(quote, text)
    if span is None:
        return None, 'not_located_in_original_manuscript'
    if locate_quote(quote, text[span['start'] + 1:]) is not None:
        return None, 'ambiguous_original_position'
    return span, 'located'


def components(packet: dict[str, Any], manuscript: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parts, audit = [], []
    for i, source in enumerate(packet['source_passages']):
        if source['source_id'] != 'MANUSCRIPT':
            continue
        start, end = source.get('start'), source.get('end')
        exact = (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(manuscript)
                 and manuscript[start:end] == source['text'])
        if exact:
            parts.append({'start': start, 'end': end, 'input_field': f'source_passages[{i}].text'})
        else:
            audit.append({'input_field': f'source_passages[{i}].text', 'state': 'unverified_original_offsets'})
    for field, quote in core_quotes(packet):
        span, state = unique_location(quote, manuscript)
        row = {'input_field': field, 'state': state, 'visible_quote': quote, 'original_span': span}
        audit.append(row)
        if span is not None:
            parts.append({'start': span['start'], 'end': span['end'], 'input_field': field})
    return parts, audit


def continuous_groups(parts: list[dict[str, Any]], manuscript: str) -> list[dict[str, Any]]:
    groups = []
    for part in sorted(parts, key=lambda row: (row['start'], row['end'])):
        previous = groups[-1] if groups else None
        if previous is not None and (part['start'] <= previous['end']
                or not manuscript[previous['end']:part['start']].strip()):
            if part['start'] > previous['end']:
                previous['layout_only_gaps'].append({'start': previous['end'], 'end': part['start'],
                                                     'text': manuscript[previous['end']:part['start']]})
            previous['end'] = max(previous['end'], part['end'])
            previous['components'].append(part)
        else:
            groups.append({'start': part['start'], 'end': part['end'],
                           'components': [part], 'layout_only_gaps': []})
    return groups


def reference_visible_sources(packet: dict[str, Any], original_sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Do not alter prompts, scientific judgments or add unseen scientific text.

    The full manuscript is used only to prove locations and gap contents of quotes already
    present in this request. Unseen original paragraphs are never added to visible material.
    """
    sources = copy.deepcopy(packet['source_passages'])
    originals = {source['text'] for source in original_sources
                 if source['source_id'] == 'MANUSCRIPT' and source['source_type'] == 'manuscript'}
    if len(originals) != 1:
        return {'sources': sources, 'audit': {'state': 'no_unique_original_manuscript',
                                             'additional_fragments': [], 'core_locations': []}}
    manuscript = next(iter(originals))
    parts, core_audit = components(packet, manuscript)
    groups = continuous_groups(parts, manuscript)
    additions = []
    present = {(source['source_id'], source.get('start'), source.get('end')) for source in sources}
    for group in groups:
        identity = ('MANUSCRIPT', group['start'], group['end'])
        if identity in present:
            continue
        fragment = {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                    'start': group['start'], 'end': group['end'],
                    'text': manuscript[group['start']:group['end']],
                    'visibility_origin': 'already_supplied_manuscript_passages_and_core_quotes',
                    'visible_input_fields': sorted({part['input_field'] for part in group['components']})}
        sources.append(fragment)
        additions.append({**group, 'source_id': 'MANUSCRIPT', 'text_sha256': fingerprint(fragment['text'])})
        present.add(identity)
    return {'sources': sources, 'audit': {
        'state': 'checked', 'visibility_protocol': 'visible_manuscript_intervals_v2',
        'original_packet_sources': len(packet['source_passages']), 'returned_sources': len(sources),
        'core_locations': core_audit, 'additional_fragments': additions,
        'request_changed': False, 'scientific_judgments_changed': False,
        'meaning': ('Only manuscript passages and core quotes already supplied in the request, located in the original. '
                    'Continuous unions include only visible intervals and verified whitespace gaps; no unseen scientific text.')}}


def required_reference_sources(packet: dict[str, Any], original_sources: list[dict[str, Any]],
                               checklist: dict[str, Any]) -> dict[str, Any]:
    """Expose only already-visible manuscript spans necessary to validate this checklist.

    Successful references return their original source packet byte-for-byte equivalent.
    This helper never calls a model or replays a location request with changed candidates.
    """
    from .evaluation import evidence_audit, source_index
    original_index = source_index(packet['source_passages'])
    missing = [(f'dimensions[{i}].evidence[{j}]', evidence)
               for i, row in enumerate(checklist['dimensions']) for j, evidence in enumerate(row['evidence'])
               if not evidence_audit(evidence, original_index)['located']]
    if not missing:
        return {'sources': copy.deepcopy(packet['source_passages']), 'audit': {
            'state': 'original_packet_sufficient', 'additional_fragments': [], 'remaining_failures': [],
            'request_changed': False, 'scientific_judgments_changed': False}}
    expanded = reference_visible_sources(packet, original_sources)
    count = len(packet['source_passages'])
    supplemental_index = source_index(expanded['sources'][count:])
    sources, additions, failures, seen = copy.deepcopy(packet['source_passages']), [], [], set()
    for path, evidence in missing:
        located = evidence_audit(evidence, supplemental_index)
        if not located['located']:
            failures.append({'path': path, 'source_id': evidence['source_id'], 'quote': evidence['quote']})
            continue
        span = located['span']
        identity = (evidence['source_id'], span['source_start'], span['source_end'])
        additions.append({'path': path, 'source_id': evidence['source_id'], 'original_span': span})
        if identity not in seen:
            sources.append({'source_id': evidence['source_id'], 'source_type': located['source_type'],
                            'start': span['source_start'], 'end': span['source_end'], 'text': span['text'],
                            'visibility_origin': 'already_supplied_manuscript_intervals', 'checklist_evidence_path': path})
            seen.add(identity)
    return {'sources': sources, 'audit': {
        'state': 'checked_missing_citations_only', 'additional_fragments': additions,
        'remaining_failures': failures, 'visibility_audit': expanded['audit'],
        'request_changed': False, 'scientific_judgments_changed': False}}


def figure_citation_location(quote: str, text: str) -> dict[str, Any] | None:
    """Permit omission only of syntactically pure parenthesised figure references.

    Return the complete original span INCLUDING its figure references. No scientific
    parentheses, measurements, negatives, conditions or non-citation numbers are removed.
    """
    import re
    panel = r'\d+[A-Za-z]?(?:\s*[-–—]\s*(?:\d+)?[A-Za-z])?'
    name = r'(?:Supplementary\s+)?Figs?\.\s*'
    reference = name + panel + r'(?:(?:\s*,\s*|\s+and\s+)(?:' + name + r')?' + panel + r')*'
    pattern = re.compile(r'\(\s*' + reference + r'\s*\)')
    removed = list(pattern.finditer(text))
    if not removed:
        return None
    chars, positions, cursor = [], [], 0
    for match in removed:
        chars.extend(text[cursor:match.start()])
        positions.extend(range(cursor, match.start()))
        cursor = match.end()
    chars.extend(text[cursor:])
    positions.extend(range(cursor, len(text)))
    span = locate_quote(quote, ''.join(chars))
    if span is None:
        return None
    start, end = positions[span['start']], positions[span['end'] - 1] + 1
    citations = [{'start': m.start(), 'end': m.end(), 'text': m.group()}
                 for m in removed if start <= m.start() and m.end() <= end]
    if not citations:
        return None
    return {'start': start, 'end': end, 'text': text[start:end], 'figure_citations_restored': citations,
            'normalisation': span['normalisation'] + '+omitted_pure_figure_reference_restored'}


def repair_visible_reference_locations(packet: dict[str, Any], original_sources: list[dict[str, Any]],
                                       checklist: dict[str, Any]) -> dict[str, Any]:
    """Copy verified original quotation spans, never regenerate scientific judgments."""
    from .evaluation import evidence_audit, source_index
    visible = reference_visible_sources(packet, original_sources)
    index = source_index(visible['sources'])
    result, repairs = copy.deepcopy(checklist), []
    for i, dimension in enumerate(result['dimensions']):
        for j, evidence in enumerate(dimension['evidence']):
            if evidence_audit(evidence, index)['located']:
                continue
            for source in index.get(evidence['source_id'], []):
                span = figure_citation_location(evidence['quote'], source['text'])
                if span is None:
                    continue
                span['source_start'] = source.get('start', 0) + span['start']
                span['source_end'] = source.get('start', 0) + span['end']
                repairs.append({'path': f'dimensions[{i}].evidence[{j}].quote', 'source_id': evidence['source_id'],
                                'original_quote': evidence['quote'], 'copied_original_span': span})
                evidence['quote'] = span['text']
                break
    required = required_reference_sources(packet, original_sources, result)
    return {'value': result, 'sources': required['sources'], 'audit': {
        'mechanical_quote_repairs': repairs, 'visibility_audit': required['audit'],
        'request_changed': False, 'scientific_judgments_changed': False}}

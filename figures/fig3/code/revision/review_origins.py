"""Bind review identities to explicit version/speaker headings in the source file."""
from __future__ import annotations

import re
from typing import Any


def bind_origins(text: str, groups: list[tuple[list[dict[str, Any]], dict[str, Any]]],
                 attachment_spans: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from .scientific_tasks import original_quote
    license_start = re.search(r'This Peer Review File is licensed', text, re.IGNORECASE)
    boundary = license_start.start() if license_start else len(text)
    versions = list(re.finditer(r'(?im)^\s*Version\s+(\d+)\s*:', text[:boundary]))
    speakers = list(re.finditer(r'(?im)^\s*(?:Reviewer|Referee)\s*#?\s*(\d+)\b', text[:boundary]))
    ordered = 'followed by all author rebuttals in order by version' in text
    attachments = 'Attachments originally included by the reviewers' in text
    rows = []
    for sections, block in groups:
        for original in sections:
            row = dict(original)
            if not row['quote'].strip() or row.get('source_region') in {'empty_source_quote', 'unresolved_source_quote'}:
                row.update(role='unknown', identity_explicit=False, round_number=0)
                rows.append(row)
                continue
            start, end = max(0, block['start']-500), block['end']
            try:
                quote = original_quote(row['quote'], text[start:end])
                position = start+text[start:end].index(quote)
                row['quote'] = quote
            except ValueError:
                # Use a block location only when neither a version nor a speaker changes inside it.
                headers = [m.start() for m in versions+speakers]+[boundary]
                if any(block['start'] < p < end for p in headers):
                    rows.append(row)
                    continue
                position = block['start']
            row['source_start'] = position
            if position >= boundary and ordered:
                row['source_region'] = 'rebuttal_or_attachment' if attachments else 'author_rebuttal'
                spans = [s for s in attachment_spans or [] if s['start'] <= position < s['end']]
                if spans:
                    span = spans[0]
                    row.update(role='reviewer', reviewer_id=span.get('reviewer_id', 'unknown'),
                               round_number=span.get('round_number', 0), source_region='review_attachment',
                               identity_explicit=span.get('reviewer_id', 'unknown') != 'unknown')
                    try:
                        original_quote(row['quote'], text[:boundary])
                    except ValueError:
                        pass
                    else:
                        row.update(role='unknown', source_region='duplicate_review_attachment')
                elif row['role'] in {'reviewer', 'unknown'}:
                    row.update(role='unknown' if attachments and attachment_spans is None else 'author', identity_explicit=False,
                               round_number=0)
            else:
                version = next((m for m in reversed(versions) if m.start() <= position), None)
                speaker = next((m for m in reversed(speakers) if m.start() <= position), None)
                if version and speaker and speaker.start() > version.start() and row['role'] != 'editor':
                    row.update(role='reviewer', reviewer_id=speaker.group(1), identity_explicit=True,
                               round_number=int(version.group(1))+1, source_version=int(version.group(1)),
                               source_region='review_report')
            rows.append(row)
    return rows


def bind_concerns(checklist: dict[str, Any], sections: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {s['section_id']: s for s in sections}
    retained, excluded = [], list(checklist.get('quoted_or_unresolved_context', []))
    for original in checklist['concerns']:
        row = dict(original)
        ids = set(re.findall(r'R\d{4}', row['section_id']))
        linked = [sources[key] for key in ids if key in sources]
        if not linked or any(s['role'] != 'reviewer' for s in linked):
            excluded.append(row)
            continue
        identities = {(s['reviewer_id'], s['round_number']) for s in linked}
        if len(identities) == 1:
            row['reviewer_id'], row['round_number'] = next(iter(identities))
        else:
            row.update(reviewer_id='unknown', round_number=0)
        retained.append(row)
    return {**checklist, 'concerns': retained, 'quoted_or_unresolved_context': excluded}


async def complete_concerns(stages: Any) -> dict[str, Any]:
    from . import models as m
    from .materials import batches
    from .storage import write
    checklist = stages.get('checklist')
    pending = set(checklist.get('pending_source_sections', []))
    if not pending:
        return checklist
    sections = stages.get('review_sections')['sections']
    selected = [s for s in sections if s['section_id'] in pending]
    groups = batches(selected, 2, 4500)
    write(stages.config.output/'inputs/tasks/checklist/papers'/stages.ident/'恢复审稿来源问题.json', {'tasks': groups})
    values = await stages.map('恢复审稿来源问题', groups,
        lambda group, i: stages.ask('checklist', {'sections': group,
            **stages.manuscript_material(group, 5500)}, m.Checklist,
            'source_restored_'+'_'.join(s['section_id'] for s in group)))
    recovered = bind_concerns({'concerns': [r for v in values for r in v['concerns']]}, sections)
    old = checklist['concerns']+checklist.get('quoted_or_unresolved_context', [])+checklist.get('manuscript_context', [])
    counter = max((int(r['concern_id'][1:]) for r in old), default=0)
    for row in recovered['concerns']:
        keys = set(re.findall(r'R\d{4}', row['section_id']))
        if keys and keys <= pending:
            counter += 1
            checklist['concerns'].append({**row, 'concern_id': f'C{counter:04d}'})
    checklist['pending_source_sections'] = []
    return checklist

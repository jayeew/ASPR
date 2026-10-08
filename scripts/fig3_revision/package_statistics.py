"""Package derived statistics and source code only; never copy experiment texts/logs."""
from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'outputs/fig3_reference/study'
DEST = ROOT/'outputs/fig3_reference/statistics_package'


def rows(name: str) -> list[dict[str, str]]:
    with (RUN/'derived'/f'{name}.csv').open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def table(name: str, values: list[dict[str, Any]]) -> None:
    path = DEST/'statistics'/f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def counts(name: str, fields: list[str], source: list[dict[str, Any]] | None = None) -> None:
    counter = Counter(tuple(row.get(key, '') for key in fields) for row in (source if source is not None else rows(name)))
    table(name+'_counts', [{**dict(zip(fields, key)), 'count': n} for key, n in sorted(counter.items())])


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    direct = ['paper_metrics', 'summary', 'paired_differences', 'costs', 'retrieval_costs',
              'recheck_coverage', 'control_coverage', 'preference_pairs']
    for name in direct:
        table(name, rows(name))
    for name, fields in {
        'reference': ['state'], 'support': ['method', 'support', 'scope_correct'],
        'concern_matches': ['method', 'scope', 'reason_coverage', 'stance', 'disagreement'],
        'quality': ['method', 'dimension', 'score'], 'fusion_errors': ['status'],
        'review_dynamics': ['resolution'], 'preference_pairs': ['opponent', 'category'],
        'rechecks': ['method', 'task', 'different'], 'control_coverage': ['kind', 'status'],
    }.items():
        counts(name, fields)
    # Use the same last-row-per-core convention as aggregate.core_metrics; discard non-core IDs.
    core_ids = {(r['paper_id'], r['core_id']) for r in rows('core')}
    canonical = {(r['paper_id'], r['method'], r['core_id']): r for r in rows('verdicts')
                 if (r['paper_id'], r['core_id']) in core_ids}
    counts('verdicts', ['method', 'difference_correct', 'scope_correct', 'effective_increment', 'material_overclaim'],
           list(canonical.values()))
    controls = []
    for row in rows('controls'):
        for side, verdict in json.loads(row['judgments']).items():
            controls.append({'kind': row['kind'], 'side': side, 'support': verdict['support'],
                             'scope_correct': str(verdict['scope_correct'])})
    counts('control_verdicts', ['kind', 'side', 'support', 'scope_correct'], controls)
    summary = json.loads((RUN/'derived/summary.json').read_text())
    costs = rows('costs')
    overview = {'papers': summary['papers'], 'shared_claims': summary['shared_claims'],
        'core_contributions': len(core_ids), 'core_method_objects': len(canonical), 'methods': 6,
        'reports': 600, 'paired_preferences': len(rows('preference_pairs')), 'preference_orders': len(rows('preference_orders')),
        'recheck': summary['recheck'], 'control_categories_planned': len(rows('control_coverage')),
        'control_pairs_constructed': sum(r['constructed']=='True' for r in rows('control_coverage')),
        'calls_with_readable_records': sum(int(r['calls']) for r in costs),
        'calls_without_usage': sum(int(r['usage_missing']) for r in costs),
        'known_input_tokens': sum(int(r['input_tokens'] or 0) for r in costs),
        'known_output_tokens': sum(int(r['output_tokens'] or 0) for r in costs),
        'unreadable_call_records': len(json.loads((RUN/'derived/unreadable_calls.json').read_text())),
        'bootstrap': summary['bootstrap']}
    (DEST/'statistics/overview.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2)+'\n')
    # Numeric/category component tables only. Quotation cases and free-text transition matrices excluded.
    for path in sorted((RUN/'derived/components').glob('*.csv')):
        if path.stem in {'c6', 'f5', 'e4', 'e5'} or path.stem.startswith('a'):
            continue
        target = DEST/'statistics/components'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    roster_rows = [json.loads(line) for line in (ROOT/'outputs/fig3_reference/dataset/papers.jsonl').read_text().splitlines() if line.strip()]
    paper_stats, report_stats, claim_types = [], [], Counter()
    for row in roster_rows:
        ident = row['paper_id']
        claims = json.loads((RUN/'annotations/claims/papers'/f'{ident}.json').read_text())['claims']
        cores = {key for paper, key in core_ids if paper == ident}
        checklist = json.loads((RUN/'annotations/checklist/papers'/f'{ident}.json').read_text())['concerns']
        paper_stats.append({'paper_id': ident, 'field': row.get('field_name', ''), 'claims': len(claims),
                            'core_contributions': len(cores), 'reviewer_concerns': len(checklist),
                            'applicable_concerns': sum(x['applies_to_input']=='yes' for x in checklist)})
        claim_types.update(x['claim_type'] for x in claims)
        for method in ('direct_a', 'gear', 'graph', 'full', 'eacl', 'reviewgrounder'):
            report = json.loads((RUN/'reports'/method/'papers'/f'{ident}.json').read_text())
            report_stats.append({'paper_id': ident, 'method': 'fusion' if method == 'full' else method,
                                 'report_characters': len(report['body']), 'cited_sources': len(report['cited_source_ids'])})
    table('paper_overview', paper_stats)
    table('report_sizes', report_stats)
    table('claim_type_counts', [{'claim_type': k, 'count': v} for k, v in sorted(claim_types.items())])
    sources = []
    for directory in ['figure_pipeline/fig3_revision', 'gear', 'configs/gear']:
        sources.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and
                       p.suffix in ({'.py'} if directory == 'gear' else {'.py', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}) and '__pycache__' not in p.parts)
    sources.extend((ROOT/'scripts/fig3_revision').glob('*.sh'))
    sources.extend([Path(__file__).resolve(), ROOT/'tests/figure_pipeline/test_fig3_revision.py',
                    ROOT/'figure_pipeline/__init__.py', ROOT/'requirements.txt', ROOT/'requirements-figures.txt'])
    for source in sources:
        if not source.exists():
            continue
        target = DEST/'code'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for doc in (ROOT/'docs/fig3_statistics_package').glob('*.md'):
        shutil.copyfile(doc, DEST/doc.name)
    inventory = []
    for path in sorted((DEST/'statistics').rglob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            n = sum(1 for _ in reader)
            inventory.append({'table': str(path.relative_to(DEST)), 'rows': n, 'columns': reader.fieldnames})
    (DEST/'statistics/table_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n')
    archive = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                handle.write(path, Path(DEST.name)/path.relative_to(DEST))
    print(json.dumps(overview, ensure_ascii=False, indent=2))
    print(f'Archive: {archive}; bytes: {archive.stat().st_size}')


if __name__ == '__main__':
    main()

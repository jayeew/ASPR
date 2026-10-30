"""Apply single-paper dated publisher evidence without extrapolating an annual rule."""
from __future__ import annotations

import gzip
import json
import sqlite3

import pyarrow.dataset as ds
from lxml import etree

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import imprint_dates, portfolio_label
from .fig8b_qualification_evidence import parse_front
from .jats_annual_copyright import named_publisher
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/other_dated_permissions/article_imprints'
    folder.mkdir(exist_ok=True)
    review = json.loads((folder.parent / 'inventory.json').read_text())
    reviewed = {r['work_id']: r for r in review['papers']}
    rows = ds.dataset(ROOT / 'fields/qualification/decisions', format='parquet').to_table(
        columns=['work_id', 'doi', 'online_year'], filter=ds.field('work_id').isin(reviewed) &
        (ds.field('eligibility_status') == 'included_verified_research') &
        (~ds.field('nature_portfolio_at_verified_online_year').isin(['True', 'False']))).to_pylist()
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    sources, audit = {}, []
    for row in rows:
        primary = [json.loads(p) for p, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],))]
        for e in primary:
            if e.get('family') != 'archived_jats' or e.get('identity') != 'matched':
                continue
            raw = gzip.decompress((ROOT / 'fields/qualification' / e['receipt']).read_bytes())
            doc = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
            tree = etree.fromstring(doc, etree.XMLParser(resolve_entities=False, no_network=True))
            if tree.tag != 'article':
                articles = tree.xpath('.//article')
                if len(articles) != 1:
                    continue
                tree = articles[0]
                doc = etree.tostring(tree)
            named = named_publisher(tree, row['online_year'])
            if not named or named[0] not in {'Elsevier', 'Oxford University Press', 'American Chemical Society', 'American Association for Cancer Research', 'Wolters Kluwer', 'S. Karger', 'BMJ'}:
                continue
            native = parse_front(doc)
            extra = {k: v for k, v in e.items() if k not in {'id', 'class', 'identity', 'work_id', 'doi'}}
            extra.update(publishers=[named[0]], publication_dates=imprint_dates(native['dates']),
                publisher_statement=named[1], publisher_statement_rule='Explicit dated named published-by statement; copyright holder alone insufficient',
                publisher_statement_locator='front/article-meta/permissions/copyright-statement')
            trial = {**e, **extra}
            label, _ = portfolio_label('', [*primary, trial], row['online_year'])
            audit.append({'work_id': row['work_id'], 'doi': row['doi'], 'receipt': e['receipt'],
                          'status': 'native_imprint_date_agrees' if label == 'False' else 'publisher_statement_known_but_native_date_not_corroborated',
                          'annual_extrapolation': False})
            if label == 'False':
                sources.setdefault(row['doi'], []).append(extra)
                break
    db.close()
    atomic_json(folder / 'review.json', {'qualified_unknown_scope': len(rows), 'papers': audit,
                                       'network_requests': 0, 'annual_rules_added': 0})
    result = apply_evidence(set(sources), folder, explicit_sources=sources)
    atomic_json(folder / 'application.json', result)
    from .derived_refresh import refresh
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)

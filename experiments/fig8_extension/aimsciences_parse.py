"""Read explicit AIMS Sciences article history and labelled byline connections."""
from __future__ import annotations

from datetime import datetime

from lxml import html

from .fig8b_qualification_build import plain
from .supplement_sources import parse_publisher


def text(node: html.HtmlElement) -> str:
    return ' '.join(node.text_content().split())


def native_metadata(raw: bytes) -> dict:
    tree = html.fromstring(raw)
    base = parse_publisher(raw)
    base['issns'] = [v.strip() for v in base['issns'] if v.strip()]
    labels = [text(n.getparent()) for n in tree.xpath('//div[contains(@class,"article-date")]//b')
              if text(n).rstrip(':').strip() == 'Published online']
    dates = set()
    for label in labels:
        value = label.partition(':')[2].strip()
        for pattern, precision, size in [('%B %d, %Y', 'day', 10), ('%B %Y', 'month', 7)]:
            try:
                dates.add((datetime.strptime(value, pattern).date().isoformat()[:size], precision))
                break
            except ValueError:
                continue
    # Never replace this explicit history field with issue/print/Early access dates.
    online, precision = next(iter(dates)) if len(dates) == 1 else ('', 'missing_or_conflicting_online_history')
    base.update(online_date=online, date_precision=precision,
                online_history_labels=labels)
    descriptions = tree.xpath('//meta[@name="dc.description"]/@content')
    base['abstract'] = plain(descriptions[0]) if len(descriptions) == 1 else base['abstract']
    slots = []
    for node in tree.xpath('//li[contains(@class,"article-author-address")]'):
        names = node.xpath('.//span[contains(@class,"aff-nmuber")]')
        body = node.xpath('./p')
        slots.append((text(names[0]).rstrip('.') if len(names) == 1 else '',
                      ' '.join(text(n) for n in body)))
    rows = tree.xpath('//ul[contains(concat(" ",normalize-space(@class)," ")," article-author ")]/li')
    shared = len(slots) == 1 and not slots[0][0] and not any(
        r.xpath('.//a[@data-tagval]') for r in rows)
    authors = []
    for row in rows:
        names = row.xpath('./a[@type="authors.authorNameEn"]')
        if len(names) != 1:
            continue
        refs = list(dict.fromkeys(row.xpath('.//a[@data-tagval]/@data-tagval')))
        missing = [v for v in refs if sum(s[0] == v for s in slots) != 1]
        linked = [s for s in slots if s[0] in refs and s[0] not in missing] if not shared else slots
        authors.append({'name': text(names[0]), 'orcid': '', 'nonpersonal': False,
            'affiliations': [{'text': s[1], 'countries': [], 'identifiers': [], 'names': []}
                             for s in linked if s[1]],
            'unresolved_affiliation_refs': missing or ([] if linked else ['native_link_absent'])})
    base.update(authors=authors, byline_has_etal=not rows or len(authors) != len(rows),
                native_author_adapter='AIMS Sciences explicit author data-tagval to numbered affiliation; only a unique unlabelled shared address can be shared')
    return base

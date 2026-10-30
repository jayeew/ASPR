"""Recompute every acquired Pensoft DOI using explicit publisher heading guards."""
from __future__ import annotations

import gzip
import json

from lxml import etree

from .fig8b_qualification_decide import NONPRIMARY, SPECIAL, key
from .pensoft_bulk import FOLDER
from .publisher_delta_apply import apply_available
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder = FOLDER / 'type_repair'
    staged = folder / 'staged'
    staged.mkdir(parents=True, exist_ok=True)
    sources = {}
    for relative in ['staged', 'native/staged', 'alternate_hosts/staged']:
        for path in (FOLDER / relative).glob('*.json'):
            record = json.loads(path.read_text())
            sources[record['doi']] = (path, record)
    jobs = []
    headings_count = {}
    for doi, (path, record) in sources.items():
        parsed = record['parsed']
        if record.get('source') == 'publisher_jats':
            raw = gzip.decompress((ROOT / 'fields/qualification' / record['raw_receipt']).read_bytes())
            tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            headings = tree.xpath('./front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject/text()')
            for heading in headings:
                headings_count[heading] = headings_count.get(heading, 0) + 1
            allowed = NONPRIMARY | SPECIAL | {'researcharticle', 'originalresearch', 'originalresearcharticle', 'conferenceabstract'}
            types = [parsed['article_type'], *headings]
            if not headings or any(key(h) not in allowed for h in headings):
                types.append('pensoft_unreviewed_heading')
            parsed = {**parsed, 'article_types': types, 'publisher_headings': headings}
        target = staged / path.name
        atomic_json(target, {**record, 'parsed': parsed,
                    'supersedes_type_interpretation_receipt': str(path.relative_to(ROOT)),
                    'type_rule': 'Generic research-article cannot override conference abstracts or an unreviewed explicit publisher heading'})
        jobs.append((doi, str(target.relative_to(ROOT))))
    result = apply_available(supplied_jobs=jobs)
    atomic_json(folder / 'application.json', {'source_dois': len(sources), 'native_headings': headings_count, 'application': result, 'network_requests': 0})
    return result


if __name__ == '__main__':
    print(json.dumps(run()))

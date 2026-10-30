"""Plan candidate-scoped PMC journal batches using retained archive identities."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree

from .fig8b_qualification_decide import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def journal_identity(data: bytes) -> tuple[str, set[str], set[str]]:
    """Read journal domain and identity from retained JATS/OAI front matter."""
    root = etree.fromstring(data, etree.XMLParser(recover=True, resolve_entities=False, no_network=True))
    if root is None:
        return '', set(), set()
    domains = root.xpath('//*[local-name()="journal-id"][@journal-id-type="pmc-domain"]/text()')
    issns = set(root.xpath('//*[local-name()="journal-meta"]//*[local-name()="issn"]/text()'))
    titles = {key(t) for t in root.xpath('//*[local-name()="journal-title"]/text()')}
    collection = root.xpath(
        '//*[local-name()="custom-meta"][*[local-name()="meta-name"][text()="pmc-is-collection-domain"]]/*[local-name()="meta-value"]/text()')
    if any(value.strip().casefold() == 'yes' for value in collection):
        return '', issns, titles  # Archive collection is not a journal-specific OAI set.
    return domains[0] if len(set(domains)) == 1 else '', issns, titles


def plan(root: Path = ROOT) -> dict:
    folder = root/'supplement'
    with (folder/'pmc_remaining_by_journal_year.csv').open(encoding='utf-8-sig') as f:
        pending = list(csv.DictReader(f))
    counts = Counter()
    names = {}
    for r in pending:
        counts[r['source_id']] += int(r['remaining_pmc_links'])
        names[r['source_id']] = r['journal']
    issn_sources: dict[str, set[str]] = defaultdict(set)
    title_sources: dict[str, set[str]] = defaultdict(set)
    with (root/'fields/journal_registry.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            sid = r['source_id']
            if sid not in counts:
                continue
            for issn in json.loads(r['issn']):
                issn_sources[issn].add(sid)
            title_sources[key(names[sid])].add(sid)
    proofs: dict[str, dict] = {}
    conflicts = []
    q = root/'fields/qualification'
    inspected = 0
    for folder_name in ['jats_raw', 'oai_raw']:
        for path in (q/folder_name).glob('*.gz'):
            domain, issns, titles = journal_identity(gzip.decompress(path.read_bytes()))
            inspected += 1
            if not domain:
                continue
            ids = set().union(*(issn_sources.get(i,set()) for i in issns)) if issns else set()
            if not ids:
                ids = set().union(*(title_sources.get(t,set()) for t in titles))
            if len(ids) != 1:
                continue
            sid = next(iter(ids))
            proof = {'domain':domain,'receipt':str(path.relative_to(root)), 'issns':sorted(issns),'title_keys':sorted(titles)}
            if sid in proofs and proofs[sid]['domain'] != domain:
                conflicts.append({'source_id':sid,'first':proofs[sid],'other':proof})
            else:
                proofs.setdefault(sid,proof)
    conflicting = {r['source_id'] for r in conflicts}
    rows = []
    for sid, n in counts.most_common():
        p = proofs.get(sid,{})
        rows.append({'source_id':sid,'journal':names[sid],'remaining_pmc_links':n,'proposed_set':p.get('domain',''),
                     'status':'domain_conflict_review' if sid in conflicting else 'candidate_set_requires_api_validation' if p else 'no_cached_domain_mapping',
                     'evidence_receipt':p.get('receipt',''),'from_release_date':'2023-01-01',
                     'publication_year_filter':'Only matched existing candidates; article online date verified separately'})
    with (folder/'pmc_journal_batch_plan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'journal_n':len(rows),'mapped_journal_n':len(proofs),'conflicting_journal_n':len(conflicting),'cached_fronts_inspected':inspected,
            'mapped_target_links':sum(r['remaining_pmc_links'] for r in rows if r['status']=='candidate_set_requires_api_validation'),
            'status':'planned_not_collected','policy':'Official OAI pmc_fm, serial off-peak retrieval; candidate DOI/PMCID identity filter before application'}
    atomic_json(folder/'pmc_journal_batch_plan.json',result)
    atomic_json(folder/'pmc_journal_domain_proofs.json',proofs)
    atomic_json(folder/'pmc_journal_domain_conflicts.json',conflicts)
    return result


if __name__ == '__main__':
    print(json.dumps(plan()),flush=True)

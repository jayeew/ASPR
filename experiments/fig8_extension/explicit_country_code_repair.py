"""Reprocess fixed qualified signatures whose explicit GE code was previously lost."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict

import pyarrow.parquet as pq
from lxml import etree

from .derived_refresh import refresh
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata


def run(scope_name: str = 'explicit_country_code_repair') -> dict:
    folder = ROOT / 'supplement' / scope_name
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    db = connect()
    if not scope_path.exists():
        groups = defaultdict(set)
        for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE status='included_verified_research' AND country_gap=1"):
            groups[shard].add(wid)
        targets = []
        for shard, ids in groups.items():
            rows = pq.read_table(ROOT / 'fields/papers' / shard, columns=['work_id','doi','authors_json'],
                                 filters=[('work_id','in',list(ids))]).to_pylist()
            for paper in rows:
                sources = set()
                slots = 0
                for author in json.loads(paper['authors_json']):
                    if author.get('nature_v1_slots') is not None:
                        continue
                    for slot in author.get('publisher_affiliation_slots') or []:
                        if not slot.get('country_codes') and any(c.get('code','').upper() == 'GE' for c in slot.get('countries',[])):
                            sources.add(author['publisher_affiliation_source'])
                            slots += 1
                if sources:
                    targets.append({'work_id':paper['work_id'], 'doi':paper['doi'], 'shard':shard,
                                    'source_receipts':sorted(sources), 'slots':slots})
        atomic_json(scope_path, {'papers':targets, 'network_requests':0, 'selection':'Existing explicit GE country tags with empty parsed codes; country-gap qualified papers only'})
    targets = json.loads(scope_path.read_text())['papers']
    jobs, held = [], []
    for target in targets:
        if len(target['source_receipts']) != 1:
            held.append({**target, 'reason':'Multiple primary sources require joint review'})
            continue
        raw_path = ROOT / target['source_receipts'][0]
        if not raw_path.name.endswith('.xml.gz'):
            held.append({**target, 'reason':'Non-JATS primary source requires separate adapter'})
            continue
        raw = gzip.decompress(raw_path.read_bytes())
        document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
        tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
        native = etree.QName(tree).localname == 'Publisher'
        if native:
            from .springer_native_xml import parse
            parsed = parse(raw)
        else:
            parsed = jats_metadata(document)
        if parsed['doi'] != [target['doi']]:
            held.append({**target, 'reason':'Raw DOI identity mismatch'})
            continue
        parsed['issns'] = parsed.get('issns', parsed.get('issn', []))
        record = {'source':'publisher_native_xml' if native else 'pmc', 'route':'explicit_country_code_cache', 'doi':target['doi'],
                  'raw_receipt':os.path.relpath(raw_path,ROOT/'fields/qualification'),
                  'raw_sha256':hashlib.sha256(raw).hexdigest(), 'parsed':parsed,
                  'scope_receipt':str(scope_path.relative_to(ROOT)), 'parse_authors_from_raw':True}
        token = hashlib.sha256(target['doi'].encode()).hexdigest()
        path = folder / (token + '.json')
        atomic_json(path, record)
        jobs.append((target['work_id'],target['shard'],str(path.relative_to(ROOT))))
    db.close()
    result = apply_authors(selected_jobs=jobs, workers=4, reprocess_selected=True) if jobs else {}
    output = {'fixed_papers':len(targets), 'fixed_slots':sum(t['slots'] for t in targets),
              'application':result, 'held':held, 'network_requests':0}
    atomic_json(folder/'status.json',output)
    if result.get('papers_changed'):
        refresh()
    print(json.dumps(output),flush=True)
    return output


if __name__ == '__main__':
    run()

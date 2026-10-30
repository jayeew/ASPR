"""Resolve Genes & Diseases annual branding from dated native publication statements."""
from __future__ import annotations

import gzip
import hashlib
import json
import re

from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import electronic_date
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import jats_metadata


def run() -> dict:
    folder = ROOT / 'supplement/keai_annual_cache'; folder.mkdir(exist_ok=True)
    sid = 'https://openalex.org/S2764642421'
    db = connect(); papers = {d:t for d,t in db.execute('SELECT doi,title FROM papers WHERE source_id=?',(sid,))}; db.close()
    rules = {}
    for path in (ROOT / 'supplement/named_native_formats/staged').glob('*.json'):
        record = json.loads(path.read_text()); doi = record['doi']
        if doi not in papers: continue
        raw_path = ROOT / 'fields/qualification' / record['raw_receipt']
        raw = gzip.decompress(raw_path.read_bytes())
        if raw.rstrip().endswith(b'</front>'): raw += b'</article>'
        native = jats_metadata(raw); native['issns'] = native['issn']
        date, _ = electronic_date(native['dates'])
        if not date or date[:4] not in {'2023','2024','2025'}: continue
        if doi not in native['doi'] or not signature_match({'title':papers[doi],'journal_name':'Genes & Diseases'},native,{'2352-3042','2352-4820'}):continue
        tree = etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True))
        statements = ' '.join(tree.xpath('./front/article-meta/permissions/copyright-statement//text()'))
        years = set(tree.xpath('./front/article-meta/permissions/copyright-year/text()'))
        if years != {date[:4]} or 'Chongqing Medical University' not in native.get('publisher',[]):continue
        if not re.search(r'Publishing services by Elsevier B\.V\. on behalf of KeAi Communications',statements):continue
        proof = {'doi':doi,'publication_date':date,'statement':statements,'source_url':record['url'],
                 'raw_receipt':str(raw_path.relative_to(ROOT)),'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                 'locator':'front/journal-meta/publisher and front/article-meta/permissions',
                 'issns':native['issns']}
        year = int(date[:4])
        if year not in rules:
            rules[year] = {'source_id':sid,'year':year,'label':'False','journal':'Genes & Diseases',
                           'publisher':'Chongqing Medical University / KeAi; Elsevier publishing services',
                           'source_url':record['url'],'raw_receipt':proof['raw_receipt'],'raw_sha256':proof['raw_sha256'],
                           'first_dated_proof':proof,'last_dated_proof':proof,'supporting_dois':[],
                           'scope':'Working annual rule based on exact native journal identity, electronic publication year and matching dated explicit KeAi/Elsevier statement; not a Springer Nature group inference; reopen on transfer or conflicting branding evidence.'}
        rule = rules[year]; rule['supporting_dois'].append(doi)
        if date < rule['first_dated_proof']['publication_date']:rule['first_dated_proof']=proof
        if date > rule['last_dated_proof']['publication_date']:rule['last_dated_proof']=proof
    return apply_proposals(list(rules.values()),folder,[],rule_status='dated_native_keai_publication_statement',
                           reason='Native electronic year and dated copyright identify Chongqing/KeAi with Elsevier publishing services',route='native_keai_jats')


if __name__ == '__main__':
    print(json.dumps(run()),flush=True)

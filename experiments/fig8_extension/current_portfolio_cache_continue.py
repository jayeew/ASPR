from __future__ import annotations
import gzip,hashlib,json,sqlite3
from collections import defaultdict,Counter
from pathlib import Path
from urllib.parse import urlparse
import pyarrow.parquet as pq
from lxml import etree
from experiments.fig8_extension.portfolio_current_review import FOLDER, native_article
from experiments.fig8_extension.supplement_queue import ROOT
from experiments.fig8_extension.supplement_sources import atomic_json,jats_metadata
from experiments.fig8_extension.three_journal_apply import read_csv
from experiments.fig8_extension.additional_dated_imprints import ENTITIES,date_supported
from experiments.fig8_extension.jats_annual_copyright import named_publisher
from experiments.fig8_extension.fig8b_qualification_decide import electronic_date

CONTEXT=json.loads((FOLDER/'publisher_context.json').read_text())['sources']
ENTITIES=ENTITIES | set(CONTEXT)
# Review every outstanding group, reusing raw evidence only. No network calls.
rows=pq.read_table(ROOT/'analysis/cohort/selection.parquet',filters=[('included','=',True)]).to_pylist()
groups=defaultdict(list)
for r in rows:
 if r['portfolio'] not in {'True','False'}: groups[r['source_id'],r['online_year']].append(r)
reg={r['source_id']:r for r in read_csv(ROOT/'fields/journal_registry.csv')}
db=sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro',uri=True)
rules=json.loads((FOLDER/'rules.json').read_text()); done={(r['source_id'],r['year']) for r in rules}; audit=[]
for pair,papers in groups.items():
 if pair in done:continue
 proofs=[]; reasons=Counter(); families=Counter()
 for p in papers:
  es=[json.loads(x[0]) for x in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'",(p['work_id'],))]
  for e in es:
   families[e.get('family','')]+=1
   if e.get('family') not in {'archived_jats','publisher_pensoft'} or e.get('identity')!='matched' or not e.get('receipt'):continue
   path=ROOT/'fields/qualification'/e['receipt']
   if not path.exists():reasons['missing_receipt']+=1;continue
   raw=path.read_bytes()
   if raw[:2]==b'\x1f\x8b':raw=gzip.decompress(raw)
   try:
    tree=native_article(raw,p['doi'])
    native=jats_metadata(etree.tostring(tree))
   except (ValueError,TypeError,etree.XMLSyntaxError):reasons['parse_failed']+=1;continue
   if p['doi'] not in {x.lower() for x in native.get('doi',[])} or not set(native.get('issn',[]))&set(json.loads(reg[pair[0]]['issn'])):
    reasons['identity_mismatch']+=1;continue
   date,_=electronic_date(native.get('dates',[]))
   if not date:
    dates=[x.get('online_date','') for x in es if x.get('identity')=='matched' and str(x.get('online_date',''))[:4]==str(pair[1]) and x.get('receipt')]
    date=next((d for d in dates if date_supported(native,d,es)),None)
   if not date or date[:4]!=str(pair[1]):reasons['no_supported_matching_online_year']+=1;continue
   named=named_publisher(tree,pair[1]);pubs=set(native.get('publisher',[])); method=''
   if named:publisher,statement=named;method='explicit_dated_named_published_by'
   elif pubs and pubs<=ENTITIES:publisher=' | '.join(sorted(pubs));statement=publisher;method='existing_accepted_native_imprint'
   elif e['family']=='publisher_pensoft' and urlparse(e['url']).hostname and urlparse(e['url']).hostname.endswith('.pensoft.net'):
    publisher='Pensoft / '+' | '.join(sorted(pubs));statement='Native DOI, ISSN and dated article XML delivered by the journal on publisher Pensoft domain';method='pensoft_native_journal_delivery'
   else:reasons['native_imprint_requires_publisher_context']+=1;continue
   proofs.append({'doi':p['doi'],'work_id':p['work_id'],'evidence_id':e['id'],'online_date':date,'publisher':publisher,'statement':statement,'publisher_context':{v:CONTEXT[v] for v in pubs if v in CONTEXT},'method':method,'source_url':e['url'],'raw_receipt':str(path.relative_to(ROOT)),'raw_sha256':hashlib.sha256(raw).hexdigest()});break
  if len(proofs)>=2:break
 if proofs:
  rule={'source_id':pair[0],'year':pair[1],'journal':reg[pair[0]]['journal_name'],'label':'False','publisher':' | '.join(sorted({p['publisher'] for p in proofs})),'source_url':proofs[0]['source_url'],'raw_receipt':proofs[0]['raw_receipt'],'raw_sha256':proofs[0]['raw_sha256'],'proofs':proofs,'scope':'Exact DOI and journal ISSN with supported online year; explicit native published-by statement, existing accepted imprint, or Pensoft native journal delivery. Annual working rule; reopen on transfer/conflicting evidence.'}
  rules.append(rule);done.add(pair)
 audit.append({'source_id':pair[0],'year':pair[1],'journal':reg[pair[0]]['journal_name'],'paper_n':len(papers),'current_registry_publisher':reg[pair[0]]['host_name'],'status':'confirmed_non_portfolio' if proofs else 'unresolved_after_extended_cache_review','reasons':dict(reasons),'evidence_families':dict(families)})
 if len(audit)%300==0:print(json.dumps({'reviewed':len(audit),'new_rules':len(rules)-236}),flush=True)
atomic_json(FOLDER/'rules.json',rules)
old_audit=json.loads((FOLDER/'cache_review_all_remaining.json').read_text()) if (FOLDER/'cache_review_all_remaining.json').exists() else []
updated={(r['source_id'],r['year']):r for r in old_audit}
updated.update({(r['source_id'],r['year']):r for r in audit})
atomic_json(FOLDER/'cache_review_all_remaining.json',list(updated.values()))
atomic_json(FOLDER/'unmatched.json',[r for r in updated.values() if (r['source_id'],r['year']) not in done])
print(json.dumps({'rules_total':len(rules),'remaining_groups':sum(r['status']!='confirmed_non_portfolio' for r in audit),'new_resolved_papers':sum(r['paper_n'] for r in audit if r['status']=='confirmed_non_portfolio')}),flush=True)

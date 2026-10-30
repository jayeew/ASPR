"""Assess exact current identity and field scope without silently appending candidates."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict

from .fig8b_qualification_build import key, plain
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder=ROOT/'supplement/boundary_review';targets=json.loads((folder/'current_metadata_targets.json').read_text())
    with (ROOT/'supplement/boundary_candidates.csv').open(encoding='utf-8-sig') as stream:directory={r['doi']:r for r in csv.DictReader(stream)}
    found=defaultdict(dict);queried={}
    for path in (folder/'current_raw').glob('*.gz'):
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        for doi in r['requested_dois']:queried[doi]=(str(path.relative_to(ROOT)),r.get('status','source_unresolved'))
        if r.get('status')!='success':continue
        for item in r['response'].get('results',[]):
            doi=(item.get('doi') or '').lower().removeprefix('https://doi.org/')
            found[doi][item['id']]=(item,str(path.relative_to(ROOT)))
    results=[];stats=Counter()
    for target in targets:
        doi=target['doi'];matches=list(found[doi].values());row={**target,'current_receipt':queried.get(doi,('',''))[0]}
        if not matches:
            status='batch_not_yet_processed' if doi not in queried else 'no_current_record_returned' if queried[doi][1]=='success' else 'current_source_retrieval_unresolved'
        elif len(matches)!=1:status='multiple_current_work_identities'
        else:
            current,receipt=matches[0];primary=current.get('primary_topic') or {};source=(current.get('primary_location') or {}).get('source') or {}
            field=(primary.get('field') or {}).get('id','').rsplit('/',1)[-1]
            row.update(current_work_id=current['id'],current_field_id=(primary.get('field') or {}).get('id'),current_type=current.get('type'),current_language=current.get('language'),current_source_id=source.get('id'),current_receipt=receipt)
            if key(plain(current.get('title') or ''))!=key(directory[doi]['title']) or source.get('id')!=target['source_id']:status='current_identity_requires_review'
            elif not field:status='current_field_unknown'
            elif field not in {'13','25','31'}:status='current_outside_three_fields_not_appended'
            elif current.get('language')!='en':status='current_language_requires_review'
            elif current.get('is_retracted') or current.get('is_paratext'):status='current_integrity_or_paratext_flag_requires_primary_review'
            else:status='current_in_scope_append_candidate_requires_primary_requalification'
        row['current_assessment']=status;results.append(row);stats[status]+=1
    atomic_json(folder/'current_assessment.json',results);result={'targets':len(targets),**stats,'appended':0}
    atomic_json(folder/'assessment_status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()

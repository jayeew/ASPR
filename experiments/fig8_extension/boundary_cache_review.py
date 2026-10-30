"""Match fixed publisher-directory omissions to existing local metadata clues."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run(root: Path = ROOT) -> dict:
    folder=root/'supplement/boundary_review';folder.mkdir(exist_ok=True)
    with (root/'supplement/boundary_candidates.csv').open(encoding='utf-8-sig') as stream:boundary={r['doi']:r for r in csv.DictReader(stream)}
    source=root/'metadata/canonical_target_works_ids_restored.parquet';rows=[];targets=[];stats=Counter()
    for cached in pq.read_table(source).to_pylist():
        doi=str(cached['doi']).lower().removeprefix('https://doi.org/')
        if doi not in boundary:continue
        official=boundary[doi];field=str(cached['field_id']).rsplit('/',1)[-1]
        title_match=key(official['title'])==key(cached['title'])
        journal_match=not official['source_id'] or official['source_id']==cached['source_id']
        status='historical_other_field_requires_current_scope_check'
        if not title_match or not journal_match:status='identity_conflict_requires_review'
        elif field in {'13','25','31'}:status='historical_in_scope_requires_current_metadata_confirmation'
        row={'doi':doi,'work_id':'https://openalex.org/'+cached['work_id'].rsplit('/',1)[-1],
            'source_id':cached['source_id'],'historical_field_id':cached['field_id'],'historical_type':cached['work_type'],
            'publisher_online_date':official['online_date'],'title_match':title_match,'journal_match':journal_match,
            'status':status,'local_metadata_receipt':str(source.relative_to(root))+'#'+cached['work_id'],
            'publisher_receipt':official['receipt']}
        rows.append(row);stats[status]+=1
        if status=='historical_in_scope_requires_current_metadata_confirmation':targets.append(row)
    atomic_json(folder/'cached_matches.json',rows);atomic_json(folder/'current_metadata_targets.json',targets)
    result={'fixed_directory_omissions':len(boundary),'local_metadata_matches':len(rows),**stats,
            'not_found_in_this_cache':len(boundary)-len({r['doi'] for r in rows}),'appended':0,
            'scope_note':'Historical field assignments are clues; neither exclusions nor additions are decided from changed or mismatched taxonomy alone'}
    atomic_json(folder/'cache_status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()

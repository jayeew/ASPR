"""Gather cached, publisher-verified sources for the complete closure scope."""
from __future__ import annotations

import csv
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import nature_boundary_sources as sources
from . import nature_clinical_review as prior
from .stage0 import write_json

ROOT = prior.OUT.parent
OUT = ROOT/'nature'
INPUT = prior.OUT/'research_roster_clinical_review.parquet'
FIELD, SLOTS = 'authorships_release_json', 'institution_release_entities'


def publisher(doi: str) -> dict[str, Any]:
    folders = [OUT,prior.OUT,prior.PREV,ROOT/'nature_cutoff_review',ROOT/'nature_joint_branch_review',
        ROOT/'nature_university_boundary_review',ROOT/'nature_weighted_review',ROOT/'nature_institution_review',ROOT/'nature_metadata_backfill']
    for folder in folders:
        path=folder/'publisher_records'/(doi.split('/')[1]+'.json')
        if not path.exists():
            continue
        result=json.loads(path.read_text())
        if any('excluded_present_address_ids' not in a for a in result['authors']):
            raw=Path(result['source_receipt']).with_suffix('.gz')
            parsed=sources.parse_page(gzip.decompress(raw.read_bytes()).decode('utf-8'),doi)
            parsed.update(source_url=result['source_url'],source_receipt=result['source_receipt'])
            result=parsed
        write_json(OUT/'publisher_records'/path.name,result)
        return result
    from . import nature_conflict_review as fetch
    old=fetch.OUT
    fetch.OUT=OUT
    try:
        return fetch.publisher(doi)
    finally:
        fetch.OUT=old


def main() -> None:
    rows=pq.read_table(INPUT).to_pylist()
    missing={r['doi'] for r in csv.DictReader((prior.OUT/'remaining_papers.csv').open(encoding='utf-8-sig'))}
    priority={r['doi'] for f in ['unresolved_branches.csv','additional_rollup_unresolved.csv']
        for r in csv.DictReader((prior.OUT/f).open(encoding='utf-8-sig'))}
    country={r['doi'] for r in rows for a in json.loads(r[prior.FIELD]) if any(not(s.get('country_code') or s.get('site_country_codes')) for s in a[prior.SLOTS])}
    scope={'missing348':sorted(missing),'priority48':sorted(priority),'country27':sorted(country),'union':sorted(missing|priority|country)}
    assert [len(scope[k]) for k in ['missing348','priority48','country27','union']]==[348,48,27,400]
    write_json(OUT/'scope.json',scope)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(publisher,sorted(missing|priority|country,key=lambda d:(d not in country,d not in priority,d)))):
            if i%50==0:
                print(f'Publisher records {i}/400',flush=True)


if __name__=='__main__':
    main()

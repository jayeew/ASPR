"""Separate metadata presence from original-research eligibility; portable offline audit."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

NONPRIMARY = {'news','newsviews','newsfeature','researchhighlight','review','reviewarticle',
    'comment','commentary','editorial','correspondence','correction','authorcorrection',
    'publishercorrection','retraction','retractionnote','obituary','careercolumn','booksarts',
    'bookreview','worldview','careers','newsqanda','newsandviews','researchbriefing'}


def disposition(row: dict, retracted: set[str], duplicate_dois: set[str]) -> str:
    if row['doi'] in retracted and row['source_id'] == 'https://openalex.org/S137773608':
        return 'excluded_known_publisher_retraction'
    if row['year_publisher_verified'] and row['year_publisher_verified'] not in [2023,2024,2025]:
        return 'excluded_verified_online_year_outside_window'
    kind = re.sub('[^a-z]', '', (row['publisher_type'] or '').lower())
    if kind in NONPRIMARY:
        return 'excluded_publisher_nonprimary_type'
    if row['doi'] in duplicate_dois:
        return 'pending_doi_version_resolution'
    if row['type_status'] == 'nature_v1_original_article_verified':
        return 'included_inherited_nature_v1_verified_research'
    if kind in ['article','researcharticle'] and row['year_publisher_verified']:
        return 'publisher_type_date_confirmed_further_eligibility_review_pending'
    return 'pending_publisher_type_date_or_protocol_review'


def build(root: Path) -> dict:
    with (root / 'inherited_evidence/nature_excluded_retractions.csv').open(encoding='utf-8-sig') as stream:
        retracted = {r['doi'].removeprefix('https://doi.org/').lower() for r in csv.DictReader(stream)}
    with (root / 'duplicate_doi_queue.csv').open(encoding='utf-8-sig') as stream:
        duplicate_dois = {r['doi'] for r in csv.DictReader(stream)}
    counts: Counter = Counter()
    columns = ['work_id','doi','source_id','field','publisher_type','type_status','year_publisher_verified']
    with gzip.open(root / 'research_eligibility.csv.gz','wt',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=[*columns,'eligibility_status'])
        writer.writeheader()
        for path in sorted((root / 'papers').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(columns=columns,batch_size=5000):
                for row in batch.to_pylist():
                    status = disposition(row,retracted,duplicate_dois)
                    counts[status] += 1
                    writer.writerow({**row,'eligibility_status':status})
    result = {'statuses': dict(counts), 'total': sum(counts.values()),
        'scope': 'Known exclusions and inherited Nature v1 eligibility only; other publishers still require explicit screening. Candidate coverage denominator is not silently replaced.'}
    (root / 'eligibility_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path,nargs='?',default=Path('.'))
    print(json.dumps(build(parser.parse_args().root),ensure_ascii=False))


if __name__ == '__main__':
    main()

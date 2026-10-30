"""Replay and source-check a fixed applied native-metadata batch."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree

from .arrow_row_patch import selected_rows
from .publisher_delta_apply import updated_decision
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def validate_sources(jobs: list[list[str]]) -> None:
    batches: dict[str, set[tuple[str, str]]] = {}
    for doi, receipt in jobs:
        record = json.loads((ROOT/receipt).read_text())
        assert record['status'] == 'parsed' and doi in record['parsed']['doi'], doi
        raw = gzip.decompress((ROOT/'fields/qualification'/record['raw_receipt']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record['raw_sha256'], doi
        if record.get('source_batch_receipt'):
            source = record['source_batch_receipt']
            if source not in batches:
                batch = gzip.decompress((ROOT/source).read_bytes())
                assert hashlib.sha256(batch).hexdigest() == record['source_batch_sha256'], source
                root = etree.fromstring(batch, etree.XMLParser(resolve_entities=False, no_network=True))
                batches[source] = {
                    (value.strip().casefold(), hashlib.sha256(etree.tostring(article)).hexdigest())
                    for article in root.findall('article')
                    for value in article.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')}
            assert (doi, record['raw_sha256']) in batches[source], doi


def validate(jobs_path: Path, output: Path) -> None:
    jobs = json.loads(jobs_path.read_text())
    validate_sources(jobs)
    counts = Counter()
    groups = defaultdict(set)
    errors = []
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for doi, _ in jobs:
            rows = db.execute('select work_id,shard,status from papers where doi=?',(doi,)).fetchall()
            assert len(rows) == 1, doi
            wid, shard, status = rows[0]
            groups[shard].add(wid)
            counts[status] += 1
        annual = {(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(ROOT/'fields/journal_publisher_by_year.csv')}
        for shard, ids in groups.items():
            for row in selected_rows(pq.read_table(ROOT/'fields/qualification/decisions'/shard),ids):
                primary = []
                secondary = []
                for kind,payload in db.execute('select kind,payload from current_evidence_index where work_id=?',(row['work_id'],)):
                    if kind == 'primary':
                        primary.append(json.loads(payload))
                    else:
                        secondary = json.loads(payload)
                if row != updated_decision(row,primary,secondary,annual):
                    errors.append({'doi':row['doi'],'issue':'offline_replay_mismatch'})
                if row['eligibility_status'] == 'included_verified_research':
                    assert row['online_year'] in {2023,2024,2025}, row['doi']
        totals = dict(db.execute('select status,count(*) from papers group by status'))
        db.close()
        assert not errors, errors[:5]
        publish_progress(totals, output)
    atomic_json(output, {'at':datetime.now(timezone.utc).isoformat(), 'papers':len(jobs),
        'statuses':dict(counts), 'raw_source_hashes_match':True, 'DOI_unique_in_unified_table':True,
        'included_online_years_in_window':True, 'offline_replay_equal':True,
        'validation_scope':'Specified applied batch only; not full Fig8 acceptance', 'errors':errors})
    print(json.dumps({'papers':len(jobs),'statuses':dict(counts),'offline_replay_equal':True}),flush=True)


def refresh_observed_coverage(progress: dict, root: Path = ROOT) -> None:
    """Use completed coverage reports without confusing them with newer cohort counts."""
    marker = root/'supplement/derived_refresh_status.json'
    if not marker.exists():
        return
    state = json.loads(marker.read_text())
    if (state.get('status') != 'complete' or state.get('finished_at')
            == progress.get('last_completed_full_refresh', {}).get('finished_at')):
        return
    rows = [r for r in read_csv(root/'fields/qualification/coverage.csv')
            if r['population'] == 'verified_research' and int(r['year']) == 0]
    fields = ['papers', 'abstract', 'every_author_institution', 'every_author_country',
              'author_records', 'authors_with_institution', 'authors_with_country']
    coverage = {field: sum(int(r[field]) for r in rows) for field in fields}
    replay = json.loads((root/'fields/qualification/recomputed_results.json').read_text())
    if coverage['papers'] != replay['verified_research_count']:
        raise ValueError('Completed coverage denominator differs from completed qualification replay')
    progress['observed_metadata_coverage'] = coverage
    progress['metadata_gaps'] = {field: coverage['papers'] - coverage[field]
                               for field in ['abstract', 'every_author_institution', 'every_author_country']}
    progress['allocation_diagnostics'] = json.loads((root/'supplement/panel_readiness.json').read_text())['diagnostics']
    progress['last_completed_full_refresh'] = state
    progress['metadata_coverage_source'] = 'fields/qualification/coverage.csv; verified_research field totals (year=0), not annual rows'


def publish_progress(totals: dict[str,int], evidence: Path) -> None:
    path = ROOT/'supplement/current_progress.json'
    progress = json.loads(path.read_text())
    refresh_observed_coverage(progress)
    included = totals['included_verified_research']
    progress.update(at=datetime.now(timezone.utc).isoformat(),
        scope='Current qualification; publisher-native and authorized NLM indexed evidence distinguished. Metadata coverage awaits coalesced refresh.',
        statuses=totals, candidate_count=sum(totals.values()), included=included,
        excluded=sum(v for k,v in totals.items() if k.startswith('excluded_')),
        pending=sum(v for k,v in totals.items() if k.startswith('pending_')),
        remaining_to_target=max(0,500000-included), metadata_coverage_refresh_pending=True,
        latest_qualification_validation=str(evidence.relative_to(ROOT)), goal_complete=False)
    progress['cohort_delta_since_coverage_refresh'] = included - progress.get('observed_metadata_coverage', {}).get('papers', included)
    progress['post_refresh_changes'] = {
        'qualified_cohort_net_change': progress['cohort_delta_since_coverage_refresh'],
        'latest_scoped_validation': str(evidence.relative_to(ROOT)),
        'metadata_coverage_requires_next_coalesced_refresh': True,
    }
    atomic_json(path,progress)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    validate(args.jobs,args.output)

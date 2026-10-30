"""Replay the reviewed MDPI subset and synchronize its derived gap flags."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .arrow_row_patch import selected_rows
from .fig8b_qualification_recompute import metrics
from .mdpi_brief_content_apply import FOLDER
from .publisher_delta_apply import updated_decision
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(review_path: Path | None = None, scope_path: Path | None = None,
        output_folder: Path | None = None) -> dict:
    reviews = json.loads((review_path or FOLDER / 'content_reviews.json').read_text())['records']
    scope = json.loads((scope_path or FOLDER / 'content_scope.json').read_text())['records']
    annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year']
              for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    groups = defaultdict(set); errors = []; counts = Counter(); coverage = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = connect()
        for review in reviews:
            shard = db.execute('SELECT shard FROM papers WHERE work_id=?', (review['work_id'],)).fetchone()[0]
            groups[shard].add(review['work_id'])
        for shard, ids in groups.items():
            papers = {r['work_id']: r for r in selected_rows(pq.read_table(ROOT / 'fields/papers' / shard), ids)}
            decisions = selected_rows(pq.read_table(ROOT / 'fields/qualification/decisions' / shard), ids)
            for row in decisions:
                primary = []; secondary = []
                for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (row['work_id'],)):
                    if kind == 'primary':
                        primary.append(json.loads(payload))
                    else:
                        secondary = json.loads(payload)
                if updated_decision(row, primary, secondary, annual) != row:
                    errors.append({'doi': row['doi'], 'error': 'qualification_replay'})
                counts[row['eligibility_status']] += 1
                if row['eligibility_status'] != 'included_verified_research':
                    continue
                paper = papers[row['work_id']]; actual = metrics(paper); coverage.update(actual)
                for field, value in actual.items():
                    if 'after_' + field in paper and paper['after_' + field] != value:
                        errors.append({'doi': row['doi'], 'error': 'stored_metric', 'field': field})
                db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',
                    (not actual['abstract'], not actual['every_author_institution'],
                     not actual['every_author_country'], row['work_id']))
        db.commit(); db.close()
    result = {'reviewed': len(reviews), 'statuses': dict(counts),
        'fixed_scope_remaining_unreviewed': len(scope) - len(reviews),
        'offline_replay_and_qualified_metrics_equal': not errors, 'errors': errors,
        'qualified_coverage': dict(coverage), 'paper_gap_flags_synchronized': True,
        'network_requests': 0, 'final_task_closure': False}
    atomic_json((output_folder or FOLDER) / 'content_validation.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)

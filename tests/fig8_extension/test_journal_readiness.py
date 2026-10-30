from __future__ import annotations

from pathlib import Path

import pytest

from experiments.fig8_extension.journal_goal_worker import ordered_jobs
from experiments.fig8_extension.journal_readiness import abstract_screen


def test_native_absence_provenance_review_is_bound_to_current_text() -> None:
    import hashlib

    text = 'A potentially misattributed abstract ' + 'word ' * 50
    digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
    flagged = {'10.1/example': digest}
    assert abstract_screen(text, '10.1/example', provenance_review=flagged) == 'quality_review'
    assert abstract_screen(text + 'Updated.', '10.1/example', provenance_review=flagged) == 'screen_pass_unvalidated'
    assert abstract_screen(text, '10.1/other', provenance_review=flagged) == 'screen_pass_unvalidated'
    assert abstract_screen(text, '10.1/example', flagged, flagged) == 'source_text_confirmed'


def test_quality_screen_does_not_claim_validation() -> None:
    assert abstract_screen(None) == 'missing'
    assert abstract_screen('A graphical abstract teaser.') == 'quality_review'
    assert abstract_screen('or when ' + 'word ' * 50) == 'quality_review'
    assert abstract_screen('A complete candidate abstract ' + 'word ' * 50) == 'screen_pass_unvalidated'


def test_priority_preserves_long_tail_jobs_without_duplicate_requests() -> None:
    jobs = [({'doi': 'tail'}, 'pmc'), ({'doi': 'pnas'}, 'pmc'), ({'doi': 'science'}, 'pmc')]
    ordered = ordered_jobs(jobs, {'science': 0, 'pnas': 2})
    assert [i['doi'] for i, _ in ordered] == ['science', 'pnas', 'tail']
    assert len(ordered) == len(jobs)


@pytest.mark.parametrize('lock_held', [False, True])
def test_journal_report_keeps_pending_out_of_abstract_denominator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_held: bool) -> None:
    import csv
    import sqlite3

    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.fig8_extension.journal_readiness import run

    for folder in ['fields/papers', 'fields/qualification/decisions', 'supplement']:
        (tmp_path / folder).mkdir(parents=True)
    db = sqlite3.connect(tmp_path / 'supplement/tasks.sqlite')
    db.execute('CREATE TABLE tasks(kind TEXT,entity_key TEXT,status TEXT,reason TEXT,receipt TEXT)')
    db.close()
    papers = [
        {'work_id': 'W1', 'doi': '10.test/1', 'source_id': 'S1', 'journal_name': 'Science', 'field': 'physics', 'abstract': 'word ' * 60, 'abstract_source': 'publisher'},
        {'work_id': 'W2', 'doi': '10.test/2', 'source_id': 'S1', 'journal_name': 'Science', 'field': 'physics', 'abstract': 'word ' * 60, 'abstract_source': 'publisher'},
        {'work_id': 'W3', 'doi': '10.test/3', 'source_id': 'S2', 'journal_name': 'Cell', 'field': 'biochemistry', 'abstract': '', 'abstract_source': ''},
    ]
    decisions = [
        {'work_id': p['work_id'], 'eligibility_status': 'pending_primary_type' if i == 1 else 'included_verified_research',
         'online_year': 0 if i == 1 else 2024, 'year_openalex': 2024, 'reason': 'fixture',
         'nature_portfolio_at_verified_online_year': 'unresolved' if i == 2 else 'False',
         'secondary_pmcids': '[]', 'evidence_ids': '[]'}
        for i, p in enumerate(papers)
    ]
    pq.write_table(pa.Table.from_pylist(papers), tmp_path / 'fields/papers/sample.parquet')
    pq.write_table(pa.Table.from_pylist(decisions), tmp_path / 'fields/qualification/decisions/sample.parquet')
    if lock_held:
        monkeypatch.setattr('experiments.fig8_extension.journal_readiness.fcntl.flock',
                            lambda *_: pytest.fail('Nested lock acquisition would deadlock the parent writer'))
    result = run(tmp_path, lock_held=lock_held)
    assert result['candidate_n'] == 3
    with (tmp_path / 'analysis/journal_readiness/priority_journals.csv').open(encoding='utf-8-sig') as stream:
        rows = {r['journal']: r for r in csv.DictReader(stream)}
    assert rows['Science']['candidate_n'] == '2'
    assert rows['Science']['verified_n'] == '1'
    assert rows['Science']['pending_n'] == '1'
    assert rows['Science']['abstract_nonempty_rate'] == '1.0'
    assert rows['Science']['formal_comparison_status'] == 'blocked_pending_eligibility'
    assert rows['Cell']['abstract_nonempty_rate'] == '0.0'
    assert rows['Cell']['brand_unknown_n'] == '1'
    assert rows['Cell']['abstract_usable_validated_n'] == ''

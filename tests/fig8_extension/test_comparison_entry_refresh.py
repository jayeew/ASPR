import csv
from pathlib import Path

from experiments.fig8_extension.journal_readiness import refresh_comparison_entries, write_csv


def test_comparison_export_tracks_new_counts_without_promoting_readiness(tmp_path: Path) -> None:
    for verified, pending in [(1, 9), (8, 2)]:
        rows = [{'journal': 'Test', 'candidate_n': 10, 'verified_n': verified,
                 'pending_n': pending, 'formal_comparison_status': 'blocked_pending_eligibility'}]
        write_csv(tmp_path / 'priority_journals.csv', rows)
        refresh_comparison_entries(tmp_path, rows)
        with (tmp_path / 'comparison_entry_assessment.csv').open(encoding='utf-8-sig') as stream:
            actual = next(csv.DictReader(stream))
        assert int(actual['verified_n']) == verified
        assert int(actual['pending_n']) == pending
        assert float(actual['qualification_resolved_rate']) == (10 - pending) / 10
        assert actual['formal_ready'] == 'False'

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from experiments.fig8_extension import fig8b_qualification_recompute as recompute


def test_gap_queue_uses_current_research_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / 'qualification'
    root.mkdir()
    (root / 'decide.py').write_text("def decide(evidence, secondary, duplicate):\n return evidence[0]\ndef portfolio_label(label,evidence,year):\n return '',[]\n")
    (tmp_path / 'journal_publisher_by_year.csv').write_text('source_id,year,nature_portfolio_year\n')
    (root / 'coverage.csv').write_text('field,population,year,papers\n')
    rows = []
    for i, status in enumerate(['included_verified_research', 'included_verified_research', 'pending_no_primary_type_date']):
        author = {'institution_ids_or_verified_local_entities': ['known'], 'countries': ['US']}
        paper = {'work_id': str(i), 'doi': '10.test/' + str(i), 'field': 'physics', 'source_id': 'journal',
                 'journal_name': 'Journal', 'title': str(i), 'year_openalex': 2024,
                 'authors_json': json.dumps([author] if i == 1 else []), 'abstract': 'present' if i == 1 else '',
                 'abstract_source': 'receipt', 'raw_receipt': 'raw', 'author_list_may_be_truncated': False}
        decision = {'work_id': str(i), 'eligibility_status': status, 'duplicate_of': '', 'online_year': 2024,
                    'nature_portfolio_at_verified_online_year': ''}
        rows.append((Path('one.parquet'), paper, decision))
    with gzip.open(root / 'decision_evidence.jsonl.gz', 'wt') as stream:
        for _, paper, decision in rows:
            stream.write(json.dumps({'work_id': paper['work_id'], 'status': decision['eligibility_status'], 'online_year': 2024}) + '\n')
    with gzip.open(root / 'secondary_evidence.jsonl.gz', 'wt'):
        pass
    monkeypatch.setattr(recompute, 'iter_papers', lambda _: iter(rows))
    (root / 'research_metadata_gaps_summary.json').write_text('{"missing_abstract":999}')
    result = recompute.run(root, refresh=True)
    gaps = json.loads((root / 'research_metadata_gaps_summary.json').read_text())
    assert result['verified_research_count'] == gaps['qualified_denominator'] == 2
    assert all(gaps[k] == 1 for k in ['missing_abstract', 'missing_institution', 'missing_country', 'any_metadata_gap'])
    with gzip.open(root / 'research_metadata_gaps.csv.gz', 'rt') as stream:
        assert [r['work_id'] for r in csv.DictReader(stream)] == ['0']
    before = (root / 'research_metadata_gaps.csv.gz').read_bytes()
    rows.append(rows[0])
    with pytest.raises(ValueError, match='Qualification errors'):
        recompute.run(root, refresh=True)
    assert (root / 'research_metadata_gaps.csv.gz').read_bytes() == before

from __future__ import annotations
import csv
import json
from pathlib import Path
from experiments.fig8_extension.target_native_validate import refresh_observed_coverage


def test_field_totals_not_double_counted_and_current_cohort_preserved(tmp_path: Path) -> None:
    q = tmp_path/'fields/qualification'; q.mkdir(parents=True)
    s = tmp_path/'supplement'; s.mkdir()
    (s/'derived_refresh_status.json').write_text(json.dumps({'status':'complete','finished_at':'2026-10-26'}))
    (q/'recomputed_results.json').write_text(json.dumps({'verified_research_count':3}))
    (s/'panel_readiness.json').write_text(json.dumps({'diagnostics':{'research_papers':3}}))
    row = dict(population='verified_research', year=0, papers=3, abstract=2,
               every_author_institution=1, every_author_country=2, author_records=6,
               authors_with_institution=4, authors_with_country=5)
    with (q/'coverage.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
        writer.writerow(row);writer.writerow({**row,'year':2023})
    progress={'included':4}
    refresh_observed_coverage(progress,tmp_path)
    assert progress['included']==4
    assert progress['observed_metadata_coverage']['papers']==3
    assert progress['metadata_gaps']['abstract']==1


def test_running_refresh_does_not_publish_partial_reports(tmp_path: Path) -> None:
    s=tmp_path/'supplement';s.mkdir()
    (s/'derived_refresh_status.json').write_text(json.dumps({'status':'running'}))
    progress={'observed_metadata_coverage':{'papers':2}}
    refresh_observed_coverage(progress,tmp_path)
    assert progress=={'observed_metadata_coverage':{'papers':2}}

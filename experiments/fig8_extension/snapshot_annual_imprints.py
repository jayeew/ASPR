"""Reuse newly recovered PMC statements for still-unknown journal-year labels."""
from __future__ import annotations

import json
import sqlite3

from . import jats_annual_fixed_scope as annual_apply
from .fig8b_qualification_decide import electronic_date
from .pmc_cloud_inventory import FOLDER as CLOUD
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> None:
    folder = FOLDER / 'annual_imprints'
    folder.mkdir(exist_ok=True)
    scope = folder / 'scope.json'
    if not scope.exists():
        unknown = {(r['source_id'], int(r['year'])) for r in
                   read_csv(ROOT / 'fields/journal_publisher_by_year.csv')
                   if r['nature_portfolio_year'] not in {'True', 'False'}}
        state = sqlite3.connect(f'file:{FOLDER}/archive_jobs.sqlite?mode=ro', uri=True)
        identifiers = [r[0] for r in state.execute("SELECT pmcid FROM jobs WHERE status='metadata_and_available_xml_path_processed'")]
        state.close()
        current = connect()
        records = []
        for identifier in identifiers:
            result = json.loads((CLOUD / 'xml_results' / (identifier + '.1.json')).read_text())
            if not result.get('parsed_receipt'):
                continue
            source = json.loads((ROOT / result['parsed_receipt']).read_text())
            parsed = source['parsed']
            online, _ = electronic_date(parsed.get('dates', []))
            if not online or int(online[:4]) not in {2023, 2024, 2025}:
                continue
            for doi in {v.lower() for v in parsed.get('doi', [])}:
                for wid, title, journal, sid in current.execute('SELECT work_id,title,journal,source_id FROM papers WHERE doi=?', (doi,)):
                    if (sid, int(online[:4])) not in unknown:
                        continue
                    records.append({'work_id': wid, 'doi': doi, 'title': title, 'journal': journal,
                        'source_id': sid, 'evidence': {'online_date': online, 'receipt': source['raw_receipt'],
                                                    'url': source['url']}})
        current.close()
        atomic_json(scope, {'records': records, 'archive_identifiers_reviewed': len(identifiers),
                    'network_requests': 0, 'scope': 'Existing current candidates, recovered published PMC captures and unknown 2023-2025 journal-year labels only. Native identity, date and dated named publisher rechecked before rule proposal.'})
    if (folder / 'application_status.json').exists():
        if json.loads((folder / 'application_status.json').read_text()).get('status') == 'fixed_annual_cache_applied':
            return
    annual_apply.FOLDER = folder
    annual_apply.run()


if __name__ == '__main__':
    run()

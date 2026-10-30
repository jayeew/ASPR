"""Join cached official preprint IDs as leads, never equating different DOIs."""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict

from .elife_batch import FOLDER
from .elife_text import plain
from .fig8b_qualification_build import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    scope = json.loads((FOLDER / 'scope.json').read_text())['papers']
    targets: dict[str, list[dict]] = defaultdict(list)
    for paper in scope:
        targets[paper['doi'].split('elife.', 1)[1].split('.', 1)[0]].append(paper)
    records = {}
    stats: Counter = Counter()
    for path in sorted((FOLDER / 'reviewed_preprints/raw').glob('*.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        payload = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
        for item in payload.get('items', []):
            for candidate in targets.get(item.get('id'), []):
                matched = key(candidate['title']) == key(plain(item.get('title', '')))
                records[candidate['doi']] = {
                    'candidate': candidate, 'native': item, 'raw_receipt': receipt['raw_receipt'],
                    'status': 'official_id_and_title_match' if matched else 'official_id_title_conflict',
                    'limitation': 'Preprint DOI is distinct; neither DOI equivalence nor journal publication date inferred',
                    'qualification_unchanged': True}
    for record in records.values():
        stats[record['status']] += 1
    result = {'records': list(records.values()), 'summary': dict(stats), 'network_requests': 0,
              'scope_candidates': len(scope), 'unmatched_candidates': len(scope) - len(records)}
    atomic_json(FOLDER / 'reviewed_preprints/id_crosswalk_review.json', result)
    print(json.dumps(result['summary']), flush=True)
    return result


if __name__ == '__main__':
    run()

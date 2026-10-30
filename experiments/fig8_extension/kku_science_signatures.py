"""Apply reviewed chemistry/physics Abha signatures; retain centers and other branches."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from . import physics_university_signatures as worker
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/official_entities/kku_science'
    candidates = json.loads((folder / 'candidate_signatures.json').read_text())
    accepted, held = [], []
    for name in candidates['signature_counts']:
        prefix = re.match(r'^department of (chemistry|physics), (faculty|college) of sciences?, king khalid university, ', name)
        if (prefix and 'abha' in name and 'saudi arabia' in name
                and not any(token in name for token in ['alqura', 'mohayel', 'dhahran', 'center', 'centre', 'hospital'])):
            accepted.append(name)
        else:
            held.append(name)
    atomic_json(folder / 'reviewed_signatures.json', {'accepted': accepted, 'held': held,
                'held_reason': 'Unsupported branch, research-center compound, missing college qualification or nonmatching signature'})
    url = 'https://www.science.kku.edu.sa/en/node/257'
    worker.FOLDER = folder
    worker.SPECS = [(candidates['entity_id'], 'King Khalid University', url, accepted)]
    worker.run(review={
        'url': url, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'additional_sources': ['https://www.science.kku.edu.sa/en/node/220',
            'https://science.kku.edu.sa/sites/science.kku.edu.sa/files/2024-03/Booklet-Chemistry%20Department%20-02042014%20%281%29.pdf'],
        'access': 'Official college and department text reviewed with web tool',
        'findings': 'KKU College of Science includes Chemistry and Physics departments; physics page gives POB960, 61421 Abha; chemistry booklet corroborates historical college departments in Abha.',
        'temporal_scope': 'Official department identity corroborates explicit 2023–2025 signed university/college/department names; no inferred author employment beyond the signature.',
        'excluded': 'Research centers, compound institutions, other branches, contaminated or incomplete names; countries unchanged.'})


if __name__ == '__main__':
    run()

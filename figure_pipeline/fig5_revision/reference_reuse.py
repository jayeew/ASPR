from __future__ import annotations

import copy
import re
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write

from .materials import digest
from .models import Config


def relabel_reference(reference: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    result = copy.deepcopy(reference)
    pattern = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(re.escape(k) for k in mapping) + r')(?![A-Za-z0-9_])') if mapping else None
    def replace(text: str) -> str:
        return pattern.sub(lambda match: mapping[match.group()], text) if pattern else text
    for part in result['parts']:
        for key in ('expected_content', 'acceptable_variants', 'reason'):
            part[key] = replace(part[key])
        for evidence in part['evidence']:
            for key in ('source_id', 'location'):
                evidence[key] = replace(evidence[key])
    return result


def sync_order_references(config: Config) -> None:
    """An information-preserving transform cannot change reference answerability."""
    for paper in read(config.output / 'cohort.json')['small']:
        full = config.output / 'validated_reference/F' / f'{paper}.json'
        target = config.output / 'validated_reference/ORDER' / f'{paper}.json'
        if not full.exists():
            continue
        state_path = config.output / 'task_status/reference_final/ORDER' / f'{paper}.json'
        if state_path.exists() and read(state_path)['state'] in ('running', 'waiting_memory'):
            continue
        mapping = read(config.output / 'order_mapping' / f'{paper}.json')
        value = relabel_reference(read(full), mapping)
        if target.exists() and read(target) == value:
            continue
        if list((config.output / 'aligned/ORDER').glob(f'*/{paper}.json')):
            write(config.output / 'repair_needed/order_reference' / f'{paper}.json', {'reason': 'ORDER must inherit F answerability; invalidate/regrade affected ORDER only before replacement'})
            continue
        if target.exists():
            write(config.output / 'superseded_order_reference' / f'{paper}.json', read(target))
        write(target, value)
        write(config.output / 'reference_reuse/ORDER' / f'{paper}.json', {'source': str(full), 'source_hash': digest(read(full)),
              'mapping': mapping, 'reason': 'identical evidence content and graph; only order and block labels change', 'answerability_invariant': True})

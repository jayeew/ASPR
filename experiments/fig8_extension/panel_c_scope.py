"""Apply independent country and any-known-institution masks to the current cohort."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def any_entity(authors_json: str) -> bool:
    authors = json.loads(authors_json)
    return any(bool(a.get('institution_ids_or_verified_local_entities')) for a in authors)


def run() -> None:
    folder = ROOT / 'analysis/cohort'
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = folder / 'selection.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        # Existing all-author entity coverage proves the weaker any-author condition.
        known = {r['work_id']: True for r in rows if r['included'] and r['all_personal_authors_have_institution']}
        pending = defaultdict(set)
        for r in rows:
            if r['included'] and r['work_id'] not in known:
                pending[r['paper_shard']].add(r['work_id'])
        for shard, ids in pending.items():
            raw = pq.read_table(ROOT / 'fields/papers' / shard,
                columns=['work_id', 'authors_json'], filters=[('work_id', 'in', sorted(ids))])
            for r in raw.to_pylist():
                known[r['work_id']] = any_entity(r['authors_json'])
        assert len(known) == sum(r['included'] for r in rows)
        country = [bool(r['included']) for r in rows]
        institutions = [bool(r['included'] and known[r['work_id']]) for r in rows]
        for name, values in [('panel_c_country_included', country), ('panel_c_institution_included', institutions)]:
            array = pa.array(values, type=pa.bool_())
            index = table.schema.get_field_index(name)
            table = table.append_column(name, array) if index < 0 else table.set_column(index, name, array)
        temp = path.with_suffix('.tmp.parquet')
        pq.write_table(table, temp, compression='zstd')
        temp.replace(path)
        fields = defaultdict(Counter)
        for r, c, i in zip(rows, country, institutions, strict=True):
            if c:
                fields[r['field']]['country_papers'] += 1
                fields[r['field']]['institution_papers'] += i
                fields[r['field']]['no_identifiable_institution'] += not i
        result = {
            'country_papers': sum(country), 'institution_papers': sum(institutions),
            'excluded_from_institution_only': sum(country) - sum(institutions),
            'fields': dict(fields),
            'institution_rule': 'Include if at least one recorded author has an existing institution ID or verified local institution entity; exclude from institution panel only if none do.',
            'country_rule': 'Use current country-qualified cohort independently of institution linkage.',
            'entity_field': 'authors_json[].institution_ids_or_verified_local_entities',
            'counting_rule': 'Deduplicate each institution within each paper; count known entities only; unknown entities unassigned.',
            'institution_types': 'All types: universities, hospitals, institutes, companies and other identifiable institutions.',
            'denominator_note': 'Observed current cohort; entity presence is not a claim that every affiliation or ownership relationship is verified.',
            'specialized_affiliation_acquisition': False, 'llm_calls': 0,
            'selection': 'analysis/cohort/selection.parquet',
        }
        assert result['country_papers'] == 419081
        atomic_json(folder / 'panel_c_summary.json', result)
        for metadata in [folder / 'summary.json', ROOT / 'analysis/analysis_scope.json']:
            content = json.loads(metadata.read_text())
            content['panel_c'] = result
            if metadata.name == 'summary.json':
                content['counts'].pop('retained_missing_institution', None)
                content['university_rule'] = 'All institution types are compared independently; no automatic hospital/university ownership merge.'
            atomic_json(metadata, content)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()

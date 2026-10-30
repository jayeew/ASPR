"""Patch identified rows without materializing every column of a metadata shard."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def selected_rows(table: pa.Table, work_ids: set[str]) -> list[dict]:
    keys = pa.array(sorted(work_ids), type=table.schema.field('work_id').type)
    return table.filter(pc.is_in(table['work_id'], value_set=keys)).to_pylist()


def patch_rows(table: pa.Table, updates: dict[str, dict]) -> pa.Table:
    if not updates:
        return table
    existing = set(table['work_id'].to_pylist())
    if not set(updates) <= existing:
        raise ValueError('Cannot patch absent Work identities')
    fields = sorted({name for values in updates.values() for name in values})
    if 'work_id' in fields:
        raise ValueError('Row patch cannot change Work identity')
    result = table
    for name in fields:
        field = table.schema.field(name)
        keys = [wid for wid, values in updates.items() if name in values]
        positions = pc.index_in(table['work_id'], value_set=pa.array(keys, type=table.schema.field('work_id').type))
        values = pa.array([updates[wid][name] for wid in keys], type=field.type)
        replacement = pc.take(values, positions)
        column = pc.if_else(pc.is_valid(positions), replacement, table[name])
        result = result.set_column(table.schema.get_field_index(name), field, column)
    return result

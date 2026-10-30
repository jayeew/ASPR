import pyarrow as pa
import pytest

from experiments.fig8_extension.arrow_row_patch import patch_rows, selected_rows


def test_sparse_patch_preserves_order_types_metadata_and_unselected_values() -> None:
    schema = pa.schema([pa.field('work_id', pa.string()), pa.field('authors_json', pa.string()),
                        pa.field('flag', pa.bool_()), pa.field('ids', pa.list_(pa.string()))], metadata={b'scope': b'current'})
    rows = [{'work_id': 'W2', 'authors_json': 'old', 'flag': False, 'ids': ['I2']},
            {'work_id': 'W1', 'authors_json': None, 'flag': True, 'ids': []},
            {'work_id': 'W3', 'authors_json': 'keep', 'flag': None, 'ids': None}]
    table = pa.Table.from_pylist(rows, schema=schema)
    table = pa.concat_tables([table.slice(0, 1), table.slice(1)])
    updates = {'W1': {'authors_json': 'α', 'ids': ['I1']}, 'W2': {'flag': None}}
    expected = [{**r, **updates.get(r['work_id'], {})} for r in rows]
    result = patch_rows(table, updates)
    assert result.equals(pa.Table.from_pylist(expected, schema=schema), check_metadata=True)
    assert patch_rows(result, updates).equals(result, check_metadata=True)
    assert selected_rows(table, {'W3', 'W1'}) == rows[1:]
    with pytest.raises(ValueError):
        patch_rows(table, {'missing': {'flag': True}})
    with pytest.raises(ValueError):
        patch_rows(table, {'W1': {'work_id': 'W9'}})

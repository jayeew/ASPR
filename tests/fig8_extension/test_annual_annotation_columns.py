import pyarrow as pa

from experiments.fig8_extension.cell_press_annual_cache import ANNOTATION_COLUMNS, replace_annotation_columns


def test_annotation_update_preserves_unrelated_columns_and_schema() -> None:
    schema = pa.schema([pa.field('authors_json', pa.string()), pa.field('abstract', pa.string()),
                        *[pa.field(c, pa.string(), metadata={b'origin': b'current'}) for c in ANNOTATION_COLUMNS]],
                       metadata={b'dataset': b'unified'})
    before = pa.Table.from_pylist([{'authors_json': '[{"name":"A"}]', 'abstract': None,
                                  **{c: '' for c in ANNOTATION_COLUMNS}},
                                 {'authors_json': '[]', 'abstract': 'Original text',
                                  **{c: 'unchanged' for c in ANNOTATION_COLUMNS}}], schema=schema)
    projected = before.select(ANNOTATION_COLUMNS).to_pylist()
    projected[0].update(nature_portfolio_at_candidate_year='False', annual_imprint='ACS')
    after = replace_annotation_columns(before, projected)
    assert after.schema.equals(before.schema, check_metadata=True)
    for name in ['authors_json', 'abstract']:
        assert after[name].equals(before[name])
    assert after['annual_imprint'].to_pylist() == ['ACS', 'unchanged']
    assert after.to_pylist()[1] == before.to_pylist()[1]

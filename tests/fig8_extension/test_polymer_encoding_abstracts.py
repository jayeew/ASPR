import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fig8_extension import polymer_encoding_abstracts as module


def test_only_matching_original_source_is_corrected_and_replay_safe(tmp_path, monkeypatch) -> None:
    folder = tmp_path / 'review'; folder.mkdir()
    (tmp_path / 'supplement').mkdir()
    papers = tmp_path / 'fields/papers'; papers.mkdir(parents=True)
    old_text = 'A sufficiently long experimental abstract describes measured polymer properties and Î± chains in a controlled study.'
    new_text = old_text.replace('Î±', 'α')
    old = {'raw_receipt': 'raw.html.gz', 'parsed': {'abstract': old_text}}
    new = {**old, 'parsed': {'abstract': new_text}, 'encoding_review': {'utf8_roundtrip_verified': True}}
    (tmp_path / 'old.json').write_text(json.dumps(old)); (tmp_path / 'new.json').write_text(json.dumps(new))
    rows = [{'work_id': w, 'abstract': old_text, 'abstract_source': s} for w, s in [
        ('matched', 'matched_publisher:fields/qualification/raw.html.gz'), ('other', 'other_source')]]
    pq.write_table(pa.Table.from_pylist(rows), papers / 'part.parquet')
    (folder / 'audit.json').write_text(json.dumps({'records': [
        {'status': 'matched_cache_decoding_repair', 'job': [w, 'part.parquet', 'new.json'], 'receipt': 'old.json'}
        for w in ['matched', 'other']]}))
    db_path = tmp_path / 'db.sqlite'; db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE changes(work_id,field,before_value,after_value,receipt,reason)'); db.close()
    monkeypatch.setattr(module, 'ROOT', tmp_path); monkeypatch.setattr(module, 'FOLDER', folder)
    monkeypatch.setattr(module, 'connect', lambda: sqlite3.connect(db_path)); monkeypatch.setattr(module, 'refresh', lambda: None)
    module.run(); module.run()
    result = pq.read_table(papers / 'part.parquet').to_pylist()
    assert result[0]['abstract'] == new_text
    assert result[1]['abstract'] == old_text
    db = sqlite3.connect(db_path)
    assert db.execute('SELECT count(*) FROM changes').fetchone()[0] == 1
    db.close()

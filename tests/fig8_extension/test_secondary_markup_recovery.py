from experiments.fig8_extension.secondary_markup_recovery import title_key


def test_encoded_scientific_markup_recovers_exact_title_only() -> None:
    assert title_key('SOD1 is an essential H&lt;sub&gt;2&lt;/sub&gt;S detoxifying enzyme.') == title_key('SOD1 is an essential H 2 S detoxifying enzyme')
    assert title_key('V&amp;lt;sub&amp;gt;m&amp;lt;/sub&amp;gt;') == title_key('V m')
    assert title_key('H&lt;sub&gt;2&lt;/sub&gt;S') != title_key('H2O')
    assert title_key('α &amp; β') == title_key('α & β')


def test_recovered_lead_ledger_is_idempotent_and_does_not_qualify(tmp_path, monkeypatch) -> None:
    import json
    from experiments.fig8_extension import secondary_markup_recovery as recovery
    from experiments.fig8_extension import supplement_queue

    root = tmp_path / 'data'
    folder = root / 'supplement/secondary_markup_recovery'
    (folder / 'parts').mkdir(parents=True)
    (folder / 'status.json').write_text(json.dumps({'status': 'fixed_cache_discovery_processed'}))
    (folder / 'parts/one.json').write_text(json.dumps({'leads': [{
        'work_id': 'W1', 'doi': '10.1/test', 'pmcid': 'PMC123', 'position': 2,
        'receipt': 'raw/cache.gz', 'identity_basis': 'Exact DOI and decoded title'}]}))
    original_connect = supplement_queue.connect
    monkeypatch.setattr(recovery, 'ROOT', root)
    monkeypatch.setattr(recovery, 'FOLDER', folder)
    monkeypatch.setattr(supplement_queue, 'connect', lambda: original_connect(root))
    db = original_connect(root)
    supplement_queue.task(db, 'qualification', '10.1/test', 'publisher', {}, 'W1')
    db.commit()
    db.close()
    recovery.register_leads()
    recovery.register_leads()
    db = original_connect(root)
    assert db.execute('SELECT count(*) FROM attempts').fetchone()[0] == 1
    assert db.execute('SELECT status,attempt_n FROM tasks').fetchone() == ('pending', 1)
    db.close()

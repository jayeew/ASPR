import gzip
import hashlib
import json
import sqlite3

from experiments.fig8_extension.scoap3_cache_index import index


def test_cached_identity_and_idempotence(tmp_path):
    folder = tmp_path / 'supplement/scoap3'
    folder.mkdir(parents=True)
    db = sqlite3.connect(tmp_path / 'supplement/tasks.sqlite')
    db.execute('CREATE TABLE papers(work_id TEXT,doi TEXT,title TEXT,journal TEXT)')
    db.executemany('INSERT INTO papers VALUES(?,?,?,?)', [
        ('W1','10.1/test','Test title','Test Journal'),
        ('W2','10.1/test','Wrong title','Test Journal')])
    db.commit()
    db.close()
    row = {'id':1,'doi':'10.1/test','title':'Test title',
           'publication_info':[{'journal_title':'Test Journal','material':'article'}]}
    payload = json.dumps({'props':{'pageProps':{'results':[row,row]}}})
    raw = ('<html><script id="__NEXT_DATA__">'+payload+'</script></html>').encode()
    (folder/'page.html.gz').write_bytes(gzip.compress(raw))
    (folder/'page.json').write_text(json.dumps({'url':'https://repo.scoap3.org/search',
        'http_status':200,'sha256':hashlib.sha256(raw).hexdigest(),
        'raw_receipt':'supplement/scoap3/page.html.gz'}))
    index(tmp_path)
    first = json.loads((folder/'cache_index_status.json').read_text())
    index(tmp_path)
    assert json.loads((folder/'cache_index_status.json').read_text()) == first
    assert first['records'] == first['record_variants'] == first['matched_candidates'] == 1
    assert first['match_statuses'] == {'identity_matched':1,'title_or_journal_mismatch':1}
    assert first['qualification_applied'] is False


def test_journal_leading_article_does_not_merge_other_tokens():
    from experiments.fig8_extension.scoap3_cache_index import journal_variants
    variants=journal_variants(['European Physical Journal C'])
    assert 'The European Physical Journal C' in variants
    assert 'European Physical Journal B' not in variants
    assert 'oretical Physics' not in journal_variants(['Theoretical Physics'])

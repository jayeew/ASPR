import json

from experiments.fig8_extension import ojs_metadata_harvest as transport


def test_existing_access_denial_reused_without_new_network(tmp_path, monkeypatch) -> None:
    folder=tmp_path/'supplement/official_entities';folder.mkdir(parents=True)
    raw=folder/'blocked.raw.gz';raw.write_bytes(b'retained response')
    receipt={'url':'https://example.org/author-guidelines','http_status':403,
             'raw_receipt':str(raw.relative_to(tmp_path)),'attempts':[{'http_status':403}]}
    (folder/'policy.json').write_text(json.dumps(receipt))
    out=tmp_path/'out';out.mkdir()
    monkeypatch.setattr(transport,'ROOT',tmp_path);monkeypatch.setattr(transport,'RAW',out)
    def forbidden():
        raise AssertionError('No network session should be opened')
    monkeypatch.setattr(transport.requests,'Session',forbidden)
    result=transport.fetch(receipt['url'],{})
    assert result['http_status']==403 and result['reused_receipt']
    assert result['reused_from']=='supplement/official_entities/policy.json'
    assert transport.fetch(receipt['url'],{})==result


def test_parameterized_requests_and_missing_raw_are_not_reused(tmp_path) -> None:
    folder=tmp_path/'supplement/official_entities';folder.mkdir(parents=True)
    (folder/'a.json').write_text(json.dumps({'url':'https://example.org/a','http_status':200,'raw_receipt':'missing'}))
    (folder/'b.json').write_text(json.dumps({'url':'https://example.org/b','http_status':403,'params':{'verb':'GetRecord'},'raw_receipt':'missing'}))
    assert transport.official_receipts(str(tmp_path))=={}

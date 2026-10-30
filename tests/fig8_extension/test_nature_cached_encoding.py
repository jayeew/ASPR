from __future__ import annotations

import gzip
import json

from experiments.fig8_extension import nature_cached_encoding as review


def test_only_matched_utf8_cache_is_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    monkeypatch.setattr(review, 'FOLDER', tmp_path / 'review')
    (tmp_path / 'review').mkdir()
    raw_folder = tmp_path / 'fields/qualification'
    raw_folder.mkdir(parents=True)
    raw = '''<html><meta name="citation_doi" content="10.1038/test"><meta name="citation_title" content="Étude"><meta name="citation_issn" content="1234-5678"><meta name="citation_journal_title" content="Nature Test"><meta name="citation_author" content="Müller, Léa"><ul data-test="authors-list"><li><a data-test="author-name">Léa Müller</a><sup><a href="#Aff1">1</a></sup></li></ul><li id="Aff1"><p class="c-article-author-affiliation__address">Université, France</p></li></html>'''.encode('utf-8')
    (raw_folder / 'raw.gz').write_bytes(gzip.compress(raw))
    record = {'route': 'nature_cached_page', 'raw_receipt': 'raw.gz'}
    (tmp_path / 'receipt.json').write_text(json.dumps(record))
    job = ('W1', 'receipt.json', 'shard.parquet', '10.1038/test', 'Étude', 'Nature Test', 'S1')
    result = review.inspect_receipt(job, {'S1': {'1234-5678'}})
    assert result['status'] == 'matched_cache_decoding_repair'
    staged = json.loads((tmp_path / result['job'][2]).read_text())
    assert staged['parsed']['title'] == ['Étude']
    assert staged['encoding_review']['utf8_roundtrip_verified']
    assert review.inspect_receipt((*job[:3], '10.1038/wrong', *job[4:]), {'S1': {'1234-5678'}})['status'] == 'corrected_identity_unconfirmed'
    record['route'] = 'pmc_archive_priority'
    (tmp_path / 'receipt.json').write_text(json.dumps(record))
    assert review.inspect_receipt(job, {})['status'] == 'other_adapter_not_in_scope'

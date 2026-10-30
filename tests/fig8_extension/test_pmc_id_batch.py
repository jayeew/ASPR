from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from outputs.fig8_data.supplement.target_500k.pmc_id_bulk import consume_batches as pipeline


def test_batch_split_preserves_identity_source_chain_and_normalizes_native_doi(tmp_path: Path, monkeypatch) -> None:
    root=tmp_path/'data';pmc=root/'pmc';(pmc/'staged').mkdir(parents=True)
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'PMC',pmc)
    raw=b'''<pmc-articleset><article article-type="research-article"><front><journal-meta><journal-title-group><journal-title>Journal A</journal-title></journal-title-group></journal-meta><article-meta><article-id pub-id-type="pmcid">PMC123</article-id><article-id pub-id-type="doi">10.1234/ABC</article-id><title-group><article-title>First paper</article-title></title-group></article-meta></front></article><article article-type="review-article"><front><journal-meta><journal-title-group><journal-title>Journal B</journal-title></journal-title-group></journal-meta><article-meta><article-id pub-id-type="pmcid">PMC124</article-id><article-id pub-id-type="doi">10.1234/def</article-id><title-group><article-title>Second paper</article-title></title-group></article-meta></front></article></pmc-articleset>'''
    receipt=root/'batch.xml.gz';receipt.write_bytes(gzip.compress(raw))
    jobs=pipeline.stage(raw,receipt,'https://eutils.ncbi.nlm.nih.gov/',[
        {'doi':'10.1234/abc','pmcid':'PMC123'}, {'doi':'10.1234/def','pmcid':'PMC999'}])
    assert len(jobs)==1
    record=json.loads((root/jobs[0][1]).read_text())
    assert record['parsed']['doi']==['10.1234/abc']
    assert record['parsed']['article_type']=='research-article'
    assert record['source_batch_sha256']==hashlib.sha256(raw).hexdigest()
    article_raw=gzip.decompress((pmc/'PMC123.efetch_article.xml.gz').read_bytes())
    assert b'First paper' in article_raw and b'Second paper' not in article_raw
    assert record['raw_sha256']==hashlib.sha256(article_raw).hexdigest()

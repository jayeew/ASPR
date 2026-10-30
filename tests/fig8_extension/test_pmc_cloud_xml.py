import hashlib

import pytest

from experiments.fig8_extension.pmc_cloud_xml import parsed_publication

RAW = b'''<article article-type="research-article"><front><journal-meta>
<journal-title>Example</journal-title><issn>1234-5678</issn></journal-meta><article-meta>
<article-id pub-id-type="doi">10.1/example</article-id><title-group><article-title>Example</article-title></title-group>
<article-categories><subj-group><subject>Perspective</subject></subj-group></article-categories>
<abstract><p>Source abstract.</p></abstract></article-meta></front></article>'''


def test_preserves_conflicting_type_category_for_shared_qualifier() -> None:
    result = parsed_publication(RAW, {'doi': '10.1/example', 'md5': hashlib.md5(RAW).hexdigest()})
    assert result['article_type'] == 'research-article'
    assert result['categories'] == ['Perspective']
    assert result['dates'] == []


def test_checksum_or_doi_conflict_is_not_applied() -> None:
    with pytest.raises(ValueError, match='checksum'):
        parsed_publication(RAW, {'doi': '10.1/example', 'md5': '0' * 32})
    with pytest.raises(ValueError, match='doi_mismatch'):
        parsed_publication(RAW, {'doi': '10.1/other', 'md5': hashlib.md5(RAW).hexdigest()})

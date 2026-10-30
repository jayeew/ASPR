"""Native PMID leads must belong to the exact article, not references or another DOI."""
from __future__ import annotations

from experiments.fig8_extension.mdpi_pubmed_increment import pmid_lead


def test_native_pmid_lead_requires_exact_article_doi() -> None:
    raw = b'<article><front><article-meta><article-id pub-id-type="doi">10.3390/example</article-id><article-id pub-id-type="pmid">123</article-id></article-meta></front>'
    assert pmid_lead('10.3390/example', raw) == ['123']
    assert pmid_lead('10.3390/other', raw) == []
    assert pmid_lead('10.3390/example', raw.replace(b'123', b'not-a-pmid')) == []


def test_conflicting_native_pmids_remain_visible() -> None:
    raw = b'<article><front><article-meta><article-id pub-id-type="doi">10.3390/example</article-id><article-id pub-id-type="pmid">123</article-id><article-id pub-id-type="pmid">456</article-id></article-meta></front></article>'
    assert pmid_lead('10.3390/example', raw) == ['123', '456']

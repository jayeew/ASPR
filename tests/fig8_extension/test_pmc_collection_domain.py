from experiments.fig8_extension.pmc_journal_batch import journal_identity


def test_archive_collection_is_not_journal_domain() -> None:
    raw = b'''<article><front><journal-meta><journal-id journal-id-type="pmc-domain">nihpa</journal-id>
    <journal-title>PNAS</journal-title><issn>1091-6490</issn><custom-meta-group><custom-meta>
    <meta-name>pmc-is-collection-domain</meta-name><meta-value>yes</meta-value>
    </custom-meta></custom-meta-group></journal-meta></front></article>'''
    domain, issns, titles = journal_identity(raw)
    assert domain == '' and issns == {'1091-6490'} and titles == {'pnas'}
    assert journal_identity(raw.replace(b'nihpa', b'pnas').replace(b'>yes<', b'>no<'))[0] == 'pnas'

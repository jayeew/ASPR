from experiments.fig8_extension.pmc_journal_batch import journal_identity
from experiments.fig8_extension.pmc_journal_harvest import wanted_records
from experiments.fig8_extension.supplement_oai import parse_oai


def test_front_identity_and_truncated_outer_article():
    raw=b'<article><front><journal-meta><journal-id journal-id-type="pmc-domain">testj</journal-id><journal-title>Test Journal</journal-title><issn>1234-5678</issn></journal-meta></front>'
    assert journal_identity(raw)==('testj',{'1234-5678'},{'testjournal'})


def test_only_frozen_pmc_targets_retained_with_parseable_provenance():
    raw=b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
    <record><header><identifier>oai:pubmedcentral.nih.gov:123</identifier></header><metadata><article><front><journal-meta><journal-title>Test Journal</journal-title></journal-meta><article-meta><article-id pub-id-type="doi">10.1/test</article-id><title-group><article-title>Test</article-title></title-group></article-meta></front></article></metadata></record>
    <record><header><identifier>oai:pubmedcentral.nih.gov:456</identifier></header><metadata><article/></metadata></record><resumptionToken>next</resumptionToken></ListRecords></OAI-PMH>'''
    records,token,errors=wanted_records(raw,{'PMC123'})
    assert token=='next' and errors==[]
    assert [p for p,_ in records]==['PMC123']
    assert parse_oai(records[0][1])['doi']==['10.1/test']


def test_invalid_set_preserved_as_error_not_success():
    records,token,errors=wanted_records(b'<OAI-PMH><error code="badArgument">Unknown set</error></OAI-PMH>',{'PMC123'})
    assert records==[] and token=='' and errors==['Unknown set']

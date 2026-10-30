from __future__ import annotations

import json

import pytest

from experiments.fig8_extension.fig8b_qualification_build import match
from experiments.fig8_extension.fig8b_qualification_decide import decide, type_class


def fixture() -> tuple[dict, dict, dict]:
    row={'work_id':'work','doi':'10.1234/test','title':'Candidate title variant','source_id':'journal','journal_name':'Journal'}
    item={'doi':row['doi'],'title':'Published title variant','family':'archived_jats',
          'kind':'research-article','issns':['1234-5678'],'categories':[],
          'native_title_identity_bridge':{'rule':'exact_doi_issn_complete_original_byline',
          'candidate_title':row['title'],'native_title':'Published title variant',
          'native_author_keys':['alice','bob','carol'],'candidate_author_keys':['alice','bob','carol'],
          'candidate_source_receipt':'census_raw/original.json.gz','candidate_raw_sha256':'abc','native_raw_sha256':'def'}}
    return row,item,{'issn':json.dumps(['1234-5678'])}


def test_complete_original_byline_bridge_retains_native_title() -> None:
    row,item,journal=fixture();result=match(row,item,journal)
    assert result['identity']=='matched'
    assert result['title']=='Published title variant'


@pytest.mark.parametrize('mutation',['doi','issn','byline','original_receipt'])
def test_partial_or_unrelated_identity_cannot_match(mutation: str) -> None:
    row,item,journal=fixture()
    if mutation=='doi':item['doi']='10.1234/other'
    if mutation=='issn':item['issns']=['9999-9999']
    if mutation=='byline':item['native_title_identity_bridge']['native_author_keys']=['alice','bob']
    if mutation=='original_receipt':item['native_title_identity_bridge']['candidate_source_receipt']='supplement/updated.json'
    assert match(row,item,journal)['identity']!='matched'


def test_migration_online_metadata_cannot_establish_year() -> None:
    research={'id':'type','family':'publisher_cpb_native','identity':'matched','class':'research',
              'kind':'Research Article','title':'Measured material properties','categories':[],'online_date':''}
    migration={**research,'id':'migration','class':'unresolved_type','kind':'unspecified',
               'online_date':'2025','native_online_dates':['2025-05-23']}
    assert decide([research,migration],[])['status']=='pending_primary_online_date'
    accepted={**migration,'id':'posted','online_date':'2024-10-03',
              'date_role':'publisher_accepted_manuscript_online'}
    accepted.pop('native_online_dates')
    result=decide([research,migration,accepted],[])
    assert result['status']=='included_verified_research' and result['online_year']==2024


def test_publisher_genres_are_not_generic_iop_or_cpl_quality_tags() -> None:
    assert type_class('Letter','publisher_cpl_native')=='research'
    assert type_class('Rapid Communication','publisher_cpb_native')=='research'
    assert type_class("Editors' Suggestion",'publisher_cpl_native')!='research'
    assert type_class('Paper','crossref_publisher_deposit')!='research'


def test_complete_abstract_bridge_requires_equal_entire_text_and_shared_author() -> None:
    row,item,journal=fixture();b=item['native_title_identity_bridge']
    b.update(rule='exact_doi_issn_complete_original_abstract_shared_author',
             native_author_keys=['alice','different'],candidate_author_keys=['alice','other'],
             original_abstract_key_sha256='wholetext',native_abstract_key_sha256='wholetext',abstract_key_length=800)
    assert match(row,item,journal)['identity']=='matched'
    b['native_abstract_key_sha256']='differenttext'
    assert match(row,item,journal)['identity']!='matched'


def test_initialised_byline_requires_entire_order_and_one_full_name() -> None:
    row,item,journal=fixture();b=item['native_title_identity_bridge']
    b.update(rule='exact_doi_issn_complete_byline_initialisation',native_author_keys=['alice','b','carol'],
             native_bibliographic_author_keys=['smith:a','jones:b','lee:c'],
             candidate_bibliographic_author_keys=['smith:a','jones:b','lee:c'])
    assert match(row,item,journal)['identity']=='matched'
    b['native_bibliographic_author_keys']=['smith:a','lee:c','jones:b']
    assert match(row,item,journal)['identity']!='matched'

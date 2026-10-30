from __future__ import annotations

from experiments.fig8_extension.fig8b_qualification_decide import (
    decide,
    electronic_date,
    type_class,
)
from experiments.fig8_extension.fig8b_qualification_evidence import parse_front


def evidence(kind: str = 'Article', date: str = '2024-01-02', identity: str = 'matched') -> dict:
    return {'id':'e1','kind':kind,'class':type_class(kind,'nature_catalogue'),
            'online_date':date,'identity':identity}


def test_crossref_journal_article_is_not_original_research() -> None:
    assert type_class('journal-article','crossref')=='unresolved_type'
    assert type_class('article','archived_jats')=='unresolved_type'


def test_online_year_not_print_collection_or_receipt() -> None:
    dates=[{'attributes':{'pub-type':'ppub'},'year':'2024'},
           {'attributes':{'pub-type':'epub'},'year':'2022','month':'12','day':'31'},
           {'attributes':{'pub-type':'collection'},'year':'2025'}]
    assert electronic_date(dates)==('2022-12-31','day')
    assert decide([evidence(date='2022-12-31')],[])['status']=='excluded_primary_online_year_outside_window'
    assert electronic_date([{'attributes':{},'year':'2024'}])[0]==''


def test_conflicting_sources_do_not_silently_include() -> None:
    assert decide([evidence(),evidence('Review')],[])['status']=='pending_conflicting_primary_types'
    assert decide([evidence(),evidence(date='2025-01-02')],[])['status']=='pending_conflicting_primary_online_years'
    assert decide([evidence()],[{'types':['Review']}])['status']=='pending_primary_secondary_type_conflict'


def test_exact_identity_and_date_required() -> None:
    assert decide([evidence()],[])['status']=='included_verified_research'
    assert decide([evidence(date='')],[])['status']=='pending_primary_online_date'
    assert decide([evidence(identity='title_or_journal_mismatch')],[])['status']=='pending_primary_identity_mismatch'
    assert decide([], [{'types':['Journal Article']}])['status']=='pending_no_primary_type_date'


def test_special_subtype_and_duplicate_are_not_included() -> None:
    assert type_class('research-article','archived_jats',['Study Protocol'])=='subtype_review_required'
    assert decide([evidence('Protocol')],[])['status']=='pending_special_subtype_review'
    assert decide([evidence()],[],'Wcanonical')['status']=='excluded_duplicate_wrong_journal'


def test_nature_photo_profile_requires_exact_journal_and_format() -> None:
    item = {**evidence('Where I Work'), 'family': 'nature_catalogue',
            'source_id': 'https://openalex.org/S137773608'}
    assert decide([item], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([{**item, 'source_id': 'other'}], [])['status'] == 'pending_primary_type'
    assert decide([{**item, 'identity': 'mismatch'}], [])['status'] == 'pending_primary_identity_mismatch'
    research = {**evidence(), 'id': 'different-evidence'}
    assert decide([item, research], [])['status'] == 'pending_conflicting_primary_types'


def test_literal_nature_news_and_review_formats_preserve_conflicts() -> None:
    for label in ['Career Q&A', 'News Q&A', 'ReviewPaper', 'Product News']:
        assert type_class(label, 'nature_page') == 'nonprimary'
        item = {**evidence(label), 'class': 'unresolved_type', 'family': 'nature_page'}
        assert decide([item], [])['status'] == 'excluded_primary_nonresearch'
        assert decide([item, {**evidence(), 'id': 'research'}], [])['status'] == 'pending_conflicting_primary_types'
    assert type_class('OriginalPaper', 'nature_page') == 'unresolved_type'


def test_jats_front_preserves_publication_semantics() -> None:
    xml=b'<article article-type="research-article"><front><journal-meta><issn>1234-5678</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.1/x</article-id><title-group><article-title>A title</article-title></title-group><pub-date publication-format="electronic" date-type="pub"><year>2023</year></pub-date><pub-date pub-type="ppub"><year>2024</year></pub-date></article-meta></front></article>'
    result=parse_front(xml)
    assert result['doi']==['10.1/x']
    assert electronic_date(result['dates'])==('2023','year')


def test_aps_letter_is_research_but_rmp_article_is_review() -> None:
    from experiments.fig8_extension.fig8b_qualification_build import match

    assert type_class('letter','aps')=='research'
    row={'title':'A colloquium','journal_name':'Reviews of Modern Physics','source_id':'S1','work_id':'W1'}
    item={'family':'aps','title':'A colloquium','journal_titles':['Reviews of Modern Physics'],
          'aps_journal':'RMP','kind':'article','online_date':'2023-01-01','categories':[]}
    result=match(row,item,{'issn':'["0034-6861"]'})
    assert result['identity']=='matched'
    assert result['class']=='nonprimary'
    assert decide([result],[])['status']=='excluded_primary_nonresearch'


def test_nonresearch_retains_observed_online_date() -> None:
    result=decide([evidence('Review','2024-03-15')],[])
    assert result['status']=='excluded_primary_nonresearch'
    assert result['online_year']==2024
    assert result['online_date']=='2024-03-15'


def test_publisher_deposit_date_can_support_independent_research_type() -> None:
    research=evidence(date='')
    deposited={'id':'date','kind':'journal-article','class':'unresolved_type',
               'online_date':'2024-04-02','identity':'matched'}
    assert decide([deposited],[])['status']=='pending_primary_type'
    assert decide([research,deposited],[])['status']=='included_verified_research'


def test_current_metadata_is_joined_without_subset_copy(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.fig8_extension.fig8b_qualification_recompute import (
        iter_papers,
        metrics,
    )

    papers = tmp_path / 'papers'
    qualification = tmp_path / 'qualification'
    papers.mkdir()
    (qualification / 'decisions').mkdir(parents=True)
    identity = {'work_id': 'W1', 'doi': '10.1/a', 'source_id': 'S1', 'field': 'physics', 'title': 'A'}
    paper = {**identity, 'abstract': '', 'authors_json': json.dumps([])}
    decision = {**identity, 'eligibility_status': 'included_verified_research'}
    pq.write_table(pa.Table.from_pylist([paper]), papers / 'one.parquet')
    pq.write_table(pa.Table.from_pylist([decision]), qualification / 'decisions/one.parquet')
    assert metrics(next(iter_papers(qualification, True))[1])['abstract'] == 0
    paper['abstract'] = 'Publisher abstract added later'
    pq.write_table(pa.Table.from_pylist([paper]), papers / 'one.parquet')
    assert metrics(next(iter_papers(qualification, True))[1])['abstract'] == 1
    assert not (qualification / 'research_papers').exists()


def test_current_join_rejects_added_unclassified_shard(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from experiments.fig8_extension.fig8b_qualification_recompute import iter_papers

    (tmp_path / 'papers').mkdir()
    qualification = tmp_path / 'qualification'
    (qualification / 'decisions').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{'work_id': 'new'}]), tmp_path / 'papers/new.parquet')
    with pytest.raises(ValueError, match='shards differ'):
        list(iter_papers(qualification))


def test_pnas_brief_report_is_journal_specific_and_retains_holds() -> None:
    e = {**evidence('Brief Report'), 'family': 'archived_jats',
         'issns': ['1091-6490'], 'title': 'An experimental result'}
    assert decide([e], [])['status'] == 'included_verified_research'
    assert decide([{**e, 'issns': []}], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**e, 'online_date': ''}], [])['status'] == 'pending_primary_online_date'
    assert decide([{**e, 'categories': ['Review']}], [])['status'] == 'pending_special_subtype_review'
    assert decide([e], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([e, {**evidence('Review'), 'id': 'other'}], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([{**e, 'identity': 'mismatch'}], [])['status'] == 'pending_primary_identity_mismatch'


def test_pnas_archived_full_journal_identity_without_issn() -> None:
    e = {**evidence('brief-report'), 'family': 'archived_jats',
         'journal_titles': ['Proceedings of the National Academy of Sciences of the United States of America'],
         'publishers': ['National Academy of Sciences']}
    assert decide([e], [])['status'] == 'included_verified_research'
    assert decide([{**e, 'journal_titles': ['PNAS Nexus']}], [])['status'] == 'pending_special_subtype_review'

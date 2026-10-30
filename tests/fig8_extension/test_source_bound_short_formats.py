from __future__ import annotations

import pytest

from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


def native(issn: str, category: str) -> dict:
    return {'id':'native','family':'archived_jats','identity':'matched','class':'subtype_review_required',
            'kind':'brief-report','title':'Measurement of a novel mechanism','categories':[category],
            'issns':[issn],'journal_titles':['Verified journal'],'online_date':'2024-02-01'}


@pytest.mark.parametrize('issn,genre',[('2666-9102','Short Communication'),('2213-6711','Report'),
    ('0950-1991','Research Report'),('0021-9533','Short Report'),('0003-2700','Technical Note')])
def test_official_source_bound_original_formats(issn: str,genre: str) -> None:
    item=native(issn,genre)
    assert journal_research_format(item)
    assert decide([item],[])['status']=='included_verified_research'
    assert decide([item],[{'types':['Review']}])['status']=='pending_primary_secondary_type_conflict'
    assert decide([{**item,'scope_review_required':True}],[])['status']=='pending_special_subtype_review'
    assert decide([{**item,'online_date':''}],[])['status']=='pending_primary_online_date'


@pytest.mark.parametrize('issn,genre',[('2161-5063','Technical Note'),('2666-9102','Technical Note'),
    ('0021-9533','Tools and Resources'),('0950-1991','Techniques and Resources'),('2213-6711','Letter')])
def test_unreviewed_resource_or_other_genres_stay_unresolved(issn: str,genre: str) -> None:
    assert not journal_research_format(native(issn,genre))

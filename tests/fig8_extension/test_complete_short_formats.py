from __future__ import annotations
import pytest
from experiments.fig8_extension.fig8b_qualification_decide import journal_research_format, decide


def item(journal: str = 'Journal of Plant Pathology', **changes: object) -> dict:
    national = journal == 'National Academy Science Letters'
    value = dict(id='e1', family='publisher_springer', identity='matched',
                 journal_titles=[journal], issns=['2250-1754' if national else '2239-7264'],
                 doi='10.1007/s40009-023-01207-0' if national else '10.1007/s42161-022-01292-y',
                 title='Experimental characterization of a pathogen', kind='BriefCommunication',
                 categories=['Short Communication', 'BriefCommunication'],
                 online_date='2023-01-10', date_precision='day',
                 **{'class': 'subtype_review_required'})
    value.update(changes)
    return value


@pytest.mark.parametrize('journal', ['Journal of Plant Pathology', 'National Academy Science Letters'])
@pytest.mark.parametrize('kind', ['BriefCommunication', 'Short Communication'])
def test_named_complete_research_format(journal: str, kind: str) -> None:
    evidence = item(journal, kind=kind)
    assert journal_research_format(evidence)
    assert decide([evidence], [])['status'] == 'included_verified_research'


@pytest.mark.parametrize('changes', [
    {'categories': ['Disease Note', 'BriefCommunication']},
    {'categories': ['Genome Resource Paper', 'BriefCommunication']},
    {'categories': ['Short Communication', 'News', 'BriefCommunication']},
    {'categories': ['BriefCommunication']},
    {'issns': ['0000-0000']}, {'doi': '10.1007/s99999-023-00000'},
    {'journal_titles': ['Unrelated Journal']}, {'scope_review_required': True},
    {'title': 'A systematic review of plant pathogens'},
])
def test_ambiguous_or_excluded_content_not_promoted(changes: dict) -> None:
    assert not journal_research_format(item(**changes))


def test_date_and_secondary_conflict_guards_remain() -> None:
    assert decide([item(online_date='2022-12-31')], [])['status'] == 'excluded_primary_online_year_outside_window'
    assert decide([item(online_date='')], [])['status'] == 'pending_primary_online_date'
    assert decide([item()], [{'types': ['Review'], 'receipt': 'index.json'}])['status'] == 'pending_primary_secondary_type_conflict'

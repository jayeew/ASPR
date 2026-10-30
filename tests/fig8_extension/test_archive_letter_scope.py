from __future__ import annotations
from experiments.fig8_extension.fig8b_qualification_decide import decide


def item(**changes: object) -> dict:
    e=dict(id='native',identity='matched',family='archived_jats',kind='research-article',categories=['Article'],title='A study of biological signalling',online_date='2024-01-01',date_precision='day',**{'class':'research'})
    e.update(changes)
    return e


def test_generic_archive_does_not_resolve_indexed_letter() -> None:
    secondary=[{'types':['Letter','research-article'],'receipt':'nlm.json'}]
    assert decide([item()],secondary)['status']=='pending_primary_secondary_type_conflict'
    assert decide([item()],[])['status']=='included_verified_research'


def test_specific_publisher_research_type_is_preserved() -> None:
    secondary=[{'types':['Letter'],'receipt':'nlm.json'}]
    publisher=item(id='publisher',family='publisher_springer',kind='OriginalPaper',categories=['OriginalPaper'])
    assert decide([item(),publisher],secondary)['status']=='included_verified_research'
    assert decide([item(categories=['Research Article'])],secondary)['status']=='included_verified_research'


def test_outside_window_and_nonresearch_are_still_resolved() -> None:
    secondary=[{'types':['Letter'],'receipt':'nlm.json'}]
    assert decide([item(online_date='2022-12-31')],secondary)['status']=='excluded_primary_online_year_outside_window'
    assert decide([item(kind='commentary',**{'class':'nonprimary'})],secondary)['status']=='excluded_primary_nonresearch'

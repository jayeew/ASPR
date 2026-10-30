from __future__ import annotations

import json

from experiments.fig8_extension.analysis_cohort import assess, author_flags


def row(authors: list[dict], truncated: bool = False) -> dict:
    return {'authors_json':json.dumps(authors),'author_list_may_be_truncated':truncated}


def test_one_known_country_does_not_make_all_authors_complete() -> None:
    flags,_=author_flags(row([{'countries':['US']},{'countries':[]}]))
    assert 'author_country_missing' in flags


def test_truncation_conflict_and_unknown_recorded_slot_are_excluded() -> None:
    flags,_=author_flags(row([{'countries':['US'],'secondary_country_conflict':True,
                              'publisher_affiliation_slots':[{'country_codes':[]}]}],True))
    assert set(flags)=={'author_list_truncated','author_country_conflict','affiliation_country_missing'}


def test_verified_collective_author_is_not_a_missing_personal_country() -> None:
    flags,_=author_flags(row([{'countries':['GB']},{'nonpersonal_verified':True,'countries':[]}]))
    assert not flags


def test_old_claim_failures_do_not_change_input_eligibility() -> None:
    paper=row([{'countries':['US']}]) | {'work_id':'W1','doi':'10.1234/example','title':'Example',
          'journal_name':'Any journal','abstract':' '.join(['Valid abstract content.']*30),
          'legacy_status':'not_reusable_current_text_or_schema'}
    decision={'field':'physics','source_id':'S1','online_year':2024,'base_shard':'physics.parquet',
              'nature_portfolio_at_verified_online_year':'False'}
    result=assess(paper,decision,{}, {})
    assert result['included']
    assert not result['old_claims_used']

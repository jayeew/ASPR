from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from experiments.fig8_extension import three_journal_round as round_
from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_cap_blocks_even_a_journal_page_without_doi(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(round_, 'AUDIT', tmp_path)
    state = {'status': 'running', 'deadline_utc': (datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
             'requests': 4, 'dois': [f'10.1/{n}' for n in range(200)], 'blocked_hosts': []}
    (tmp_path/'budget.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match='cap'):
        round_.get('https://example.org/journal')
    assert json.loads((tmp_path/'budget.json').read_text()) == state


def test_alias_cannot_override_doi_or_issn_conflict() -> None:
    sid = round_.SIDS[2]
    d = {'source_id':sid,'doi':'10.1/test','title':'Exact title'}
    e = {'id':'old','identity':'title_or_journal_mismatch','title':'Exact title','doi':'10.1/wrong',
         'issns':[],'journal_titles':[round_.ALIASES[sid]]}
    rules = {sid:{'alias':round_.ALIASES[sid]}}
    assert round_.repair_alias(d,e,rules) == e
    e['doi'] = d['doi']
    e['issns'] = ['wrong-issn']
    assert round_.repair_alias(d,e,rules) == e
    e['issns'] = []
    fixed = round_.repair_alias(d,e,rules)
    assert fixed['identity'] == 'matched' and fixed['supersedes'] == 'old'


def test_generic_date_and_type_do_not_qualify_paper() -> None:
    assert round_.deposit_evidence({'published':{'date-parts':[[2024,1,1]]}},'receipt','url') == []
    e = round_.deposit_evidence({'published-online':{'date-parts':[[2024,1,1]]},'type':'journal-article'},'receipt','url')[0]
    e.update(identity='matched',id='date-only',**{'class':'unresolved_type'})
    assert decide([e],[])['status'] == 'pending_primary_type'


def test_short_format_remains_pending_after_alias_and_date() -> None:
    e = {'identity':'matched','id':'brief','kind':'brief-report','class':'subtype_review_required','online_date':'2024-01-01'}
    assert decide([e],[])['status'] == 'pending_special_subtype_review'


def test_pnas_editorial_heading_overrides_generic_jats_research() -> None:
    from experiments.fig8_extension.fig8b_qualification_build import match
    row={'work_id':'W1','doi':'10.1/x','source_id':round_.SIDS[2],'title':'A matching title','journal_name':'PNAS'}
    source={'issn':'["0027-8424"]'}
    item={'family':'archived_jats','title':row['title'],'issns':['0027-8424'],'kind':'research-article',
          'categories':['Opinion'],'online_date':'2024-01-01'}
    e=match(row,item,source)
    assert e['identity']=='matched' and e['class']=='subtype_review_required'
    assert decide([e],[])['status']=='pending_special_subtype_review'


def test_curated_scope_hold_survives_rebuild() -> None:
    from experiments.fig8_extension.fig8b_qualification_build import match
    row={'work_id':'W1','doi':'10.1/x','source_id':round_.SIDS[0],'title':'A matching title','journal_name':'Science'}
    item={'family':'archived_jats','title':row['title'],'issns':['0036-8075'],'kind':'research-article',
          'categories':['Article'],'online_date':'2024-01-01','scope_review_required':True}
    e=match(row,item,{'issn':'["0036-8075"]'})
    assert e['class']=='subtype_review_required'

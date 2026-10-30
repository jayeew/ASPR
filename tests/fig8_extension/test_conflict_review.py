"""Publisher author references, not another coauthor's addresses, govern conflict repair."""
from __future__ import annotations

from typing import Any

import pytest

from experiments.fig8_extension.nature_conflict_review import corrected_slots
from experiments.fig8_extension.nature_cutoff_review import SLOTS


def fixtures() -> tuple[dict[str,Any],dict[str,Any]]:
    author={'author':{'display_name':'Jiaming Li'},'raw_author_name':'Jiaming Li',
            SLOTS:[{'display_name':'Institute A'},{'display_name':'Hospital B'}]}
    pub={'author_affiliation_parse_complete':True,
         'authors':[{'name':'Jiaming Li','affiliation_ids':['Aff1']},{'name':'Jian Li','affiliation_ids':['Aff2']}],
         'affiliations':[{'affiliation_id':'Aff1','address':'Institute A, China'},
                         {'affiliation_id':'Aff2','address':'Hospital B, China'}]}
    return author,pub


def test_similar_author_names_do_not_share_institutions() -> None:
    a,p=fixtures()
    slots,addresses=corrected_slots(a,p,['Institute A'])
    assert slots==[{'display_name':'Institute A'}]
    assert addresses==['Institute A, China']
    assert len(a[SLOTS])==2
    with pytest.raises(ValueError,match='lacks explicit'):
        corrected_slots(a,p,['Hospital B'])


def test_incomplete_publisher_refs_do_not_authorize_deletion() -> None:
    a,p=fixtures();p['author_affiliation_parse_complete']=False
    with pytest.raises(ValueError,match='complete publisher'):
        corrected_slots(a,p,['Institute A'])


def test_retained_unknown_unit_is_not_silently_removed() -> None:
    a,p=fixtures();a[SLOTS].append({'display_name':'Consortium C','identity_status':'identity_unresolved'})
    p['affiliations'][0]['address']='Institute A, Consortium C, China'
    slots,_=corrected_slots(a,p,['Institute A','Consortium C'])
    assert slots[1]['identity_status']=='identity_unresolved'


def test_present_address_removed_only_when_absent_from_primary() -> None:
    from experiments.fig8_extension.nature_conflict_review import SLOTS as NEW_SLOTS
    from experiments.fig8_extension.nature_conflict_review import remove_present
    a,p=fixtures();a[NEW_SLOTS]=a[SLOTS]
    p['authors'][0]['excluded_present_address_ids']=['Aff3']
    slots,ev=remove_present(a,p,{'Aff3':'Present address: Hospital B, China'})
    assert slots==[{'display_name':'Institute A'}]
    assert ev['status']=='removed_present_only'
    p['affiliations'][0]['address']='Institute A and Hospital B, China'
    slots,ev=remove_present(a,p,{'Aff3':'Present address: Hospital B, China'})
    assert len(slots)==2
    assert ev['status']=='no_exact_present_only_slot'

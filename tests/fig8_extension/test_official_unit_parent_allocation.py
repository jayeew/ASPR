import json

from experiments.fig8_extension.official_units_apply import resolve
from experiments.fig8_extension.supplement_allocation import SCENARIOS, allocate


def test_confirmed_unit_keeps_entity_but_allocates_parent() -> None:
    rule={'entity_id':'institute','entity_type':'research_organization','ownership_status':'officially_confirmed_university_unit','university_targets':{'university':1.0}}
    author={'institution_ids_or_verified_local_entities':[], 'publisher_affiliation_slots':[{'text':'explicit unit','institution_ids':[],'identity_status':'unresolved','country_codes':['RU']}]}
    revised,count=resolve(author,{'explicit unit':rule},'official/statute.json')
    assert count and revised['publisher_affiliation_slots'][0]['institution_ids']==['institute']
    assert resolve(revised,{'explicit unit':rule},'official/statute.json')[1]==0
    known={s:set() for s in SCENARIOS}
    for s in SCENARIOS:
        countries,universities,_=allocate({'authors_json':json.dumps([revised])},s,known)
        assert dict(countries)=={'RU':1.0}
        assert dict(universities)=={'university':1.0}
    revised['publisher_affiliation_slots'][0].pop('official_identity_evidence')
    assert dict(allocate({'authors_json':json.dumps([revised])},SCENARIOS[0],known)[1])=={'unknown':1.0}


def test_conflicting_parent_is_not_overwritten() -> None:
    rule={'entity_id':'institute','entity_type':'research_organization','ownership_status':'officially_confirmed_university_unit','university_targets':{'university':1.0}}
    author={'institution_ids_or_verified_local_entities':['institute'],'publisher_affiliation_slots':[{'text':'explicit unit','institution_ids':['institute'],'official_university_targets':{'other':1.0}}]}
    assert resolve(author,{'explicit unit':rule},'official/statute.json')==(author,0)

import json

from experiments.fig8_extension.official_entity_apply import CERN, resolve
from experiments.fig8_extension.supplement_allocation import allocate


def author(text='CERN'):
    return {'name':'A','institution_ids_or_verified_local_entities':[], 'countries':[],
            'publisher_affiliation_slots':[{'text':text,'institution_ids':[],
                'identity_status':'unresolved','country_codes':[]}]}


def test_nonuniversity_identity_does_not_infer_country_or_university():
    rule={'entity_id':CERN,'exact_names':['CERN']}
    value,n=resolve(author(),rule)
    assert n==1
    assert value['countries']==[]
    assert value['publisher_affiliation_slots'][0]['country_codes']==[]
    c,u,stats=allocate({'authors_json':json.dumps([value])},'main_personal_direct',{'main_personal_direct':set()})
    assert c['unknown']==1
    assert not u
    assert not stats['identity_unresolved_slots']
    assert resolve(value,rule)==(value,0)


def test_joint_collaboration_and_nature_slots_preserved():
    rule={'entity_id':CERN,'exact_names':['CERN']}
    for text in ['CERN and University of X','CERN collaboration','CERN, Geneva']:
        original=author(text)
        assert resolve(original,rule)==(original,0)
    original=author();original['nature_v1_slots']=[{'entity_id':'existing'}]
    assert resolve(original,rule)==(original,0)


def test_later_metadata_preserves_review_or_retains_conflict():
    from experiments.fig8_extension.supplement_apply import preserve_official_slots
    old=resolve(author(),{'entity_id':CERN,'exact_names':['CERN']})[0]['publisher_affiliation_slots']
    incoming=[{'text':'CERN','institution_ids':[],'identity_status':'unresolved','country_codes':[]}]
    updated,conflict=preserve_official_slots(old,incoming)
    assert not conflict
    assert updated[0]['institution_ids']==[CERN]
    assert updated[0]['university_ownership_status']=='officially_confirmed_nonuniversity'
    assert incoming[0]['institution_ids']==[]
    assert preserve_official_slots(old,[{'text':'Other unit','institution_ids':[]}])[1]
    assert preserve_official_slots(old,[{'text':'CERN','institution_ids':['different']}])[1]

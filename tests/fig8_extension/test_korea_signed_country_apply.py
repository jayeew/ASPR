from experiments.fig8_extension import korea_signed_country_apply as module


def test_country_fill_preserves_ids_and_existing_countries(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path/'source').write_text('retained source')
    address='University of Seoul, Seoul, Korea'
    author={'name':'A','author_id':'id','publisher_affiliation_source':'source','countries':[],
            'publisher_affiliation_slots':[{'text':address,'institution_ids':['unit'],'country_codes':[]}]}
    updated,n=module.fill(author,{address})
    assert n==1 and updated['countries']==['KR']
    assert updated['author_id']=='id'
    assert updated['publisher_affiliation_slots'][0]['institution_ids']==['unit']
    assert module.fill(updated,{address})==(updated,0)
    author['publisher_affiliation_slots'][0]['country_codes']=['US']
    assert module.fill(author,{address})==(author,0)
    author['nature_v1_slots']=[]
    assert module.fill(author,{address})==(author,0)

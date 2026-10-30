from experiments.fig8_extension.scoap3_metadata_apply import merge, metadata


def test_archive_country_uses_signature_not_repository_annotation():
    source=metadata({'id':1,'title':'t','authors':[{'name':'Jane Doe','affiliations':[
        {'value':'Laboratory, United Kingdom','country':{'code':'US'},'ror':'unverified'}]}]},'receipt.json')
    author={'name':'Jane Doe','countries':[],'institution_ids_or_verified_local_entities':[]}
    result,stats=merge([author],[source],{}, {}, {'united kingdom':'GB'})
    assert result[0]['countries']==['GB']
    assert result[0]['country_source']=='scoap3_explicit_affiliation_country'
    assert stats['countries_filled']==1
    assert not result[0]['institution_ids_or_verified_local_entities']
    repeated,delta=merge(result,[source],{}, {}, {'united kingdom':'GB'})
    assert repeated==result
    assert not delta['author_statements_added']


def test_archive_does_not_replace_primary_or_duplicate_name():
    source=metadata({'id':1,'authors':[{'name':'Jane Doe','affiliations':[{'value':'Lab, UK'}]}]},'receipt.json')
    primary={'name':'Jane Doe','countries':['US'],'institution_ids_or_verified_local_entities':[],
             'publisher_affiliation_slots':[{'text':'Existing'}]}
    other={'name':'Jane Doe','countries':[],'institution_ids_or_verified_local_entities':[]}
    result,stats=merge([primary,other],[source],{},{},{'uk':'GB'})
    assert result==[primary,other]

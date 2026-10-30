from experiments.fig8_extension.secondary_author_cache import enrich, metadata


def test_no_initials_or_inferred_country_and_preserve_primary():
    source=metadata({'title':'Test','authorList':{'author':[{'fullName':'Li A'}, {'firstName':'Alice','lastName':'Li','authorAffiliationDetailsList':{'authorAffiliation':[{'affiliation':'Example University, Canada'}]}}]}},'cached.json.gz')
    assert len(source['authors'])==1
    author={'name':'Alice Li','countries':[],'institution_ids_or_verified_local_entities':[]}
    result,stats=enrich([author],[source],{}, {}, {'canada':'CA'})
    assert result[0]['countries']==['CA']
    assert result[0]['secondary_affiliation_evidence']['authority'].startswith('Europe PMC')
    assert 'publisher_affiliation_slots' not in result[0]
    assert stats['author_country_filled']==1
    repeated,_=enrich(result,[source],{}, {}, {'canada':'CA'})
    assert repeated==result
    primary={**author,'publisher_affiliation_slots':[{'text':'original'}]}
    assert enrich([primary],[source],{},{},{'canada':'CA'})[0]==[primary]


def test_conflicting_affiliations_and_duplicate_authors_stay_unresolved():
    a={'name':'Alice Li','countries':[],'institution_ids_or_verified_local_entities':[]}
    source={'authors':[{'name':'Alice Li','affiliations':[{'text':'Canada'}]}]}
    other={'authors':[{'name':'Alice Li','affiliations':[{'text':'USA'}]}]}
    assert enrich([a],[source,other],{},{},{})[0]==[a]
    assert enrich([a,a],[source],{},{},{})[0]==[a,a]

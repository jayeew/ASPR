from experiments.fig8_extension.orcid_cached_links import canonical_orcid,link_authors


def test_checksum():
    assert canonical_orcid('https://orcid.org/0000-0002-1271-3089')=='0000-0002-1271-3089'
    assert canonical_orcid('0000-0002-1271-3088')==''


def test_unique_orcid_link_preserves_names_and_rejects_ambiguity():
    a={'name':'S. Example','author_id':'A1','countries':[],'institution_ids_or_verified_local_entities':[]}
    donor={'name':'Susan Example','orcid':'0000-0002-1271-3089','nonpersonal':False,'unresolved_affiliation_refs':[],
        'affiliations':[{'text':'Example University, Canada','countries':[],'names':[],'identifiers':[]}]}
    source={'authors':[donor],'evidence_receipt':'front.xml'}
    result,stats=link_authors([a],[source],{'A1':donor['orcid']},{},{},{'canada':'CA'})
    assert result[0]['name']=='S. Example'
    assert result[0]['publisher_author_original_name']=='Susan Example'
    assert result[0]['countries']==['CA']
    assert stats['publisher_author_links_recovered']==1
    assert link_authors([a],[{**source,'authors':[donor,donor]}],{'A1':donor['orcid']},{},{},{})[0]==[a]
    assert link_authors([a,a],[source],{'A1':donor['orcid']},{},{},{})[0]==[a,a]


def test_orcid_source_conflict_and_existing_links_preserved() -> None:
    author = {'name': 'S. Example', 'author_id': 'A1', 'countries': [],
              'institution_ids_or_verified_local_entities': []}
    donor = {'name': 'Susan Example', 'orcid': '0000-0002-1271-3089',
             'nonpersonal': False, 'unresolved_affiliation_refs': [],
             'affiliations': [{'text': 'Example University, Canada',
                               'countries': [], 'names': [], 'identifiers': []}]}
    first = {'authors': [donor], 'evidence_receipt': 'first.xml'}
    second = {'authors': [{**donor, 'affiliations': [
        {'text': 'Different Institute, France', 'countries': [],
         'names': [], 'identifiers': []}]}], 'evidence_receipt': 'second.xml'}
    ids = {'A1': donor['orcid']}
    result, stats = link_authors([author], [first, second], ids, {}, {}, {})
    assert result == [author]
    assert stats['orcid_affiliation_source_conflicts'] == 1
    for field in ('publisher_affiliation_slots', 'nature_v1_slots'):
        retained = {**author, field: [{'text': 'Existing reviewed affiliation'}]}
        assert link_authors([retained], [first, second], ids, {}, {}, {})[0] == [retained]

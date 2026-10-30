from experiments.fig8_extension.raw_author_name_recovery import reconcile


def test_original_raw_name_links_without_fuzzy_matching() -> None:
    current = [{'name': 'Jessica Valentina Montgomery', 'author_id': 'A1'}]
    native = [{'name': 'Jessica V. Montgomery', 'affiliations': [{'text': 'Published institute'}]}]
    original = {'authorships': [{'author': {'id': 'A1', 'display_name': current[0]['name']},
                                'raw_author_name': native[0]['name']}]}
    updated, links = reconcile(current, native, original, {'work_id': 'W1'})
    assert len(links) == 1 and updated[0]['name'] == current[0]['name']
    assert updated[0]['native_name'] == native[0]['name']
    assert native[0]['name'] == 'Jessica V. Montgomery'
    assert updated[0]['affiliations'] == native[0]['affiliations']
    for changed in [
        {'authorships': [{**original['authorships'][0], 'raw_author_name': 'J. Montgomery'}]},
        {'authorships': [{**original['authorships'][0], 'author': {'id': 'A2', 'display_name': current[0]['name']}}]},
        {'authorships': original['authorships'] * 2},
    ]:
        assert reconcile(current, native, changed, {})[1] == []
    assert reconcile(current, native * 2, original, {})[1] == []
    assert reconcile([{**current[0], 'publisher_affiliation_slots': [{'text': 'Preserved'}]}], native, original, {})[1] == []


def test_alternate_source_fills_only_missing_slots() -> None:
    from experiments.fig8_extension.primary_byline_comparison import fill_missing_links
    existing = {'name': 'Existing Author', 'author_id': 'A0', 'countries': ['GB'],
                'institution_ids_or_verified_local_entities': ['I0'],
                'publisher_affiliation_slots': [{'text': 'Reviewed institute', 'country_codes': ['GB']}],
                'publisher_affiliation_source': 'retained', 'nonpersonal_verified': False}
    missing = {'name': 'Expanded Author', 'author_id': 'A1', 'countries': [],
               'institution_ids_or_verified_local_entities': [], 'nonpersonal_verified': False}
    proof = {'author_id': 'A1', 'raw_author_name': 'E. Author'}
    source = {'evidence_receipt': 'alternate', 'authors': [
        {'name': 'Existing Author', 'nonpersonal': False, 'unresolved_affiliation_refs': [],
         'affiliations': [{'text': 'Different institute', 'countries': [], 'identifiers': [], 'names': []}]},
        {'name': 'Expanded Author', 'nonpersonal': False, 'unresolved_affiliation_refs': [],
         'raw_name_identity_evidence': proof,
         'affiliations': [{'text': 'Declared institute', 'countries': [], 'identifiers': [], 'names': []}]},
    ]}
    result, count = fill_missing_links([existing, missing], source, {}, {}, {})
    assert count == 1 and result[0] == existing
    assert result[1]['publisher_affiliation_slots'][0]['text'] == 'Declared institute'
    assert result[1]['publisher_author_identity_evidence'] == proof
    assert fill_missing_links(result, source, {}, {}, {}) == (result, 0)

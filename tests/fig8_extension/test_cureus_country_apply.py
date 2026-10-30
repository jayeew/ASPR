from experiments.fig8_extension.cureus_country_apply import enrich


def test_country_fill_requires_original_author_address_and_preserves_known_values() -> None:
    address = 'Department, University, City, IND'
    source = {'authors': [{'name': 'A', 'affiliations': [{'text': address}]}]}
    person = {'name': 'A', 'countries': [], 'publisher_affiliation_source': 'raw',
              'publisher_affiliation_slots': [{'text': address, 'country_codes': []}]}
    result, slots, people = enrich([person], source, 'raw', {address: 'IN'})
    assert (slots, people) == (1, 1)
    assert result[0]['countries'] == ['IN']
    assert person['countries'] == []
    assert enrich(result, source, 'raw', {address: 'IN'}) == (result, 0, 0)
    assert enrich([person], source, 'other', {address: 'IN'}) == ([person], 0, 0)
    known = {**person, 'countries': ['GB']}
    conflict = enrich([known], source, 'raw', {address: 'IN'})[0][0]
    assert conflict['countries'] == ['GB']
    assert conflict['country_conflict_retained']
    reviewed = {**person, 'nature_v1_slots': []}
    assert enrich([reviewed], source, 'raw', {address: 'IN'}) == ([reviewed], 0, 0)
    assert enrich([person, person], source, 'raw', {address: 'IN'}) == ([person, person], 0, 0)


def test_name_punctuation_does_not_expand_initials_or_change_stored_names() -> None:
    address = 'Department, University, City, IND'
    source = {'authors': [{'name': 'Piyush C Rathi', 'affiliations': [{'text': address}]}]}
    person = {'name': 'Piyush  C. Rathi', 'countries': [], 'publisher_affiliation_source': 'raw',
              'publisher_affiliation_slots': [{'text': address, 'country_codes': []}]}
    result, slots, _ = enrich([person], source, 'raw', {address: 'IN'})
    assert slots == 1 and result[0]['name'] == person['name']
    person['name'] = 'Piyush Chandra Rathi'
    assert enrich([person], source, 'raw', {address: 'IN'}) == ([person], 0, 0)


def test_terminal_country_rule_keeps_its_actual_provenance() -> None:
    address = 'Department, University, City, Japan'
    source = {'authors': [{'name': 'A', 'affiliations': [{'text': address}]}]}
    person = {'name': 'A', 'countries': [], 'publisher_affiliation_source': 'raw',
              'publisher_affiliation_slots': [{'text': address, 'country_codes': []}]}
    result, slots, people = enrich([person], source, 'raw', {address: 'JP'},
                                   'terminal_scope.json', 'explicit_terminal_country_name')
    assert (slots, people) == (1, 1)
    evidence = result[0]['publisher_affiliation_slots'][0]['country_evidence']
    assert evidence['rule'] == 'terminal_scope.json'
    assert 'ISO' not in evidence['basis']
    assert result[0]['country_source'] == 'all_publisher_slots_explicit_terminal_country'

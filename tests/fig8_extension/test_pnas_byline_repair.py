from experiments.fig8_extension.pnas_byline_repair import reconcile


def native(name):
    return {'name': name, 'nonpersonal': False, 'affiliations': [], 'unresolved_affiliation_refs': []}


def test_duplicate_inherited_ids_not_transferred():
    old = [{'name': 'Benita', 'author_id': 'wrong', 'position': 0},
           {'name': 'Benita', 'author_id': 'wrong', 'position': 1},
           {'name': 'Other', 'author_id': 'keep', 'position': 2}]
    source = {'authors': [native('Benita'), native('John'), native('Other')], 'evidence_receipt': 'raw.xml'}
    out, status = reconcile(old, source)
    assert status == 'publisher_roster_reconciled'
    assert [a['author_id'] for a in out] == ['', '', 'keep']
    assert [a['name'] for a in out] == ['Benita', 'John', 'Other']
    assert reconcile(out, source)[0] == out
    assert old[1]['name'] == 'Benita'


def test_existing_primary_ambiguous_identity_held():
    old = [{'name': 'A', 'publisher_affiliation_slots': [{'text': 'confirmed'}]}, {'name': 'A'}]
    source = {'authors': [native('A'), native('B')], 'evidence_receipt': 'raw.xml'}
    assert reconcile(old, source) == (old, 'existing_primary_identity_conflict')


def test_unique_primary_record_preserved_when_position_changes():
    confirmed = {'name': 'Known', 'author_id': 'known-id', 'position': 0,
                 'countries': ['GB'], 'publisher_affiliation_source': 'raw.xml',
                 'publisher_affiliation_slots': [{'text': 'Reviewed address', 'country_codes': ['GB']}]}
    source = {'authors': [native('New'), native('Known')], 'evidence_receipt': 'raw.xml'}
    out, status = reconcile([confirmed], source)
    assert status == 'publisher_roster_reconciled'
    assert out[1] == {**confirmed, 'position': 1}
    assert out[0]['author_id'] == ''


def test_incomplete_native_list_never_replaces_current():
    old = [{'name': 'Existing'}]
    source = {'authors': [native('New')], 'byline_has_etal': True, 'evidence_receipt': 'raw.xml'}
    assert reconcile(old, source) == (old, 'native_byline_incomplete_or_ambiguous')


def test_collective_front_does_not_disprove_member_roster():
    old = [{'name': 'Member'}, {'name': 'Consortium'}]
    group = {**native('Consortium'), 'nonpersonal': True}
    source = {'authors': [group], 'evidence_receipt': 'front.xml'}
    assert reconcile(old, source) == (old, 'group_membership_requires_full_byline_evidence')

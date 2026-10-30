from experiments.fig8_extension.supplement_apply import preserve_official_slots


def test_country_review_survives_reparse_without_official_institution_rule() -> None:
    evidence = {'rule': 'scope.json', 'raw_signature': 'native.xml.gz'}
    old = [{'text': 'Department, City, Japan', 'country_codes': ['JP'],
            'country_evidence': evidence, 'institution_ids': []}]
    incoming = [{'text': 'Department, City, Japan', 'country_codes': [],
                 'institution_ids': ['newly_matched_institution']}]
    result, conflict = preserve_official_slots(old, incoming)
    assert not conflict
    assert result[0]['country_codes'] == ['JP']
    assert result[0]['country_evidence'] == evidence
    assert result[0]['institution_ids'] == ['newly_matched_institution']
    assert incoming[0]['country_codes'] == []
    assert preserve_official_slots(result, incoming) == (result, False)
    incoming[0]['country_codes'] = ['CN']
    assert preserve_official_slots(old, incoming)[1]
    incoming[0]['text'] = 'Different unit, China'
    assert preserve_official_slots(old, incoming)[1]

from experiments.fig8_extension.official_units_apply import resolve


def test_known_identity_ownership_queue_reuses_only_exact_rules():
    import sqlite3
    from experiments.fig8_extension.official_units_apply import affected_papers
    db = sqlite3.connect(':memory:')
    db.executescript('''
        CREATE TABLE tasks(task_id TEXT,kind TEXT,entity_key TEXT);
        CREATE TABLE task_papers(task_id TEXT,work_id TEXT);
        CREATE TABLE papers(work_id TEXT,shard TEXT,status TEXT);
        INSERT INTO tasks VALUES('t','university_ownership','institute');
        INSERT INTO task_papers VALUES('t','included'),('t','excluded');
        INSERT INTO papers VALUES('included','a','included_verified_research'),
                                 ('excluded','b','excluded_nonresearch');
    ''')
    rule = {'entity_id': 'institute', 'entity_type': 'research_organization',
            'ownership_status': 'officially_confirmed_nonuniversity'}
    aliases = {'institute, city': rule}
    assert affected_papers(db, aliases) == {'a': {'included'}}
    author = {'institution_ids_or_verified_local_entities': ['institute'],
              'publisher_affiliation_slots': [
                  {'text': 'Institute, City', 'institution_ids': ['institute']},
                  {'text': 'Institute and University, City', 'institution_ids': ['institute']}]}
    updated, count = resolve(author, aliases, 'reviewed.json')
    assert count == 1
    assert updated['publisher_affiliation_slots'][0]['official_identity_evidence'] == 'reviewed.json'
    assert updated['publisher_affiliation_slots'][1] == author['publisher_affiliation_slots'][1]
    assert resolve(updated, aliases, 'reviewed.json') == (updated, 0)
    db.close()


def test_combined_rules_preserve_individual_sources_and_repeat_safely():
    aliases = {name: {'entity_id': name, 'entity_type': 'research_organization',
                      'ownership_status': 'officially_confirmed_nonuniversity',
                      '_rule_receipt': name + '.json'} for name in ['a', 'b']}
    author = {'institution_ids_or_verified_local_entities': [], 'countries': ['IT'],
              'publisher_affiliation_slots': [{'text': name, 'institution_ids': [],
                                               'country_codes': ['IT']} for name in aliases]}
    updated, count = resolve(author, aliases, 'fallback.json')
    assert count == 4
    assert [s['official_identity_evidence'] for s in updated['publisher_affiliation_slots']] == ['a.json', 'b.json']
    assert resolve(updated, aliases, 'fallback.json') == (updated, 0)
    assert updated['countries'] == ['IT']


def test_confirmed_local_identity_does_not_confirm_university_ownership():
    from experiments.fig8_extension.supplement_allocation import SCENARIOS, slots_for
    rule = {'entity_id': 'reviewed-local:hefei', 'entity_type': 'research_organization',
            'ownership_status': 'unverified'}
    author = {'institution_ids_or_verified_local_entities': [], 'countries': ['CN'],
              'publisher_affiliation_slots': [{'text': 'Hefei National Laboratory, Hefei 230088, China',
                                               'institution_ids': [], 'country_codes': ['CN']}]}
    updated, count = resolve(author, {'hefei national laboratory, hefei 230088, china': rule}, 'official.json')
    assert count == 2
    known = {s: set() for s in SCENARIOS}
    for scenario in SCENARIOS:
        slot = slots_for(updated, scenario, known)[0]
        assert not slot['identity_missing']
        assert slot['unknown_university'] and slot['targets'] == {}
        assert slot['country_codes'] == ['CN']


def test_conflicting_combined_rules_fail_before_database_write(tmp_path, monkeypatch):
    import json
    import pytest
    from experiments.fig8_extension import official_units_apply as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    for name in ['a', 'b']:
        rule = {'entity_id': name, 'entity_type': 'research_organization',
                'ownership_status': 'officially_confirmed_nonuniversity',
                'exact_names': ['same whole signature'], 'university_targets': {}}
        (tmp_path / (name + '.json')).write_text(json.dumps({'status': 'reviewed', 'rules': [rule]}))
    with pytest.raises(ValueError, match='Conflicting reviewed rules'):
        module.run(rule_paths=['a.json', 'b.json'])
    assert not (tmp_path / 'supplement').exists()


def test_independent_unit_only_preserves_country_and_conflicts():
    rule = {'entity_id': 'infn-pisa', 'entity_type': 'research_organization',
            'ownership_status': 'officially_confirmed_nonuniversity'}
    aliases = {'infn sezione di pisa, pisa, italy': rule}
    author = {'institution_ids_or_verified_local_entities': [], 'countries': [],
              'publisher_affiliation_slots': [
                  {'text': 'INFN Sezione di Pisa, Pisa, Italy', 'institution_ids': [], 'country_codes': []},
                  {'text': 'INFN Sezione di Pisa and University', 'institution_ids': []}]}
    updated, n = resolve(author, aliases)
    assert n == 2
    assert updated['countries'] == []
    assert updated['publisher_affiliation_slots'][0]['country_codes'] == []
    assert updated['publisher_affiliation_slots'][1]['institution_ids'] == []
    assert resolve(updated, aliases)[1] == 0
    author['publisher_affiliation_slots'][0]['institution_ids'] = ['different-unit']
    assert resolve(author, aliases)[1] == 0
    author['nature_v1_slots'] = []
    assert resolve(author, aliases) == (author, 0)


def test_official_explicit_university_signature_enters_allocation_without_parent_rule():
    from experiments.fig8_extension.supplement_allocation import SCENARIOS, slots_for
    rule = {'entity_id': 'university', 'entity_type': 'education',
            'ownership_status': 'officially_confirmed_university_self'}
    author = {'institution_ids_or_verified_local_entities': [], 'countries': ['DE'],
              'publisher_affiliation_slots': [{'text': 'Department, University, Germany',
                                              'institution_ids': [], 'country_codes': ['DE']}]}
    updated, count = resolve(author, {'department, university, germany': rule}, 'reviewed_rule.json')
    assert count == 2
    for scenario in SCENARIOS:
        slot = slots_for(updated, scenario, {s: set() for s in SCENARIOS})[0]
        assert slot['targets'] == {'university': 1.0}
        assert not slot['unknown_university']
        assert slot['country_codes'] == ['DE']
    updated['publisher_affiliation_slots'][0].pop('official_identity_evidence')
    assert slots_for(updated, SCENARIOS[0], {s: set() for s in SCENARIOS})[0]['targets'] == {}


def test_shared_rule_receipt_is_not_reappended_after_another_rule() -> None:
    from experiments.fig8_extension.official_units_apply import has_rule
    previous = {'official_rule': 'b', 'previous_identity_evidence': {'official_rule': 'a'}}
    assert has_rule(previous, 'a')
    assert has_rule(previous, 'b')
    assert not has_rule(previous, 'c')
    assert not has_rule({}, 'a')


def test_equivalent_rule_keeps_existing_official_provenance() -> None:
    rule = {'entity_id': 'unit', 'entity_type': 'research_organization',
            'ownership_status': 'officially_confirmed_nonuniversity'}
    original = {'institution_ids_or_verified_local_entities': [],
                'publisher_affiliation_slots': [{'text': 'Unit, Italy', 'institution_ids': [], 'country_codes': ['IT']}]}
    first, count = resolve(original, {'unit, italy': rule}, 'first-source.json')
    assert count > 0
    second, count = resolve(first, {'unit, italy': dict(rule, _rule_receipt='equivalent-source.json')})
    assert count == 0 and second == first
    revised = dict(rule, ownership_status='unverified', _rule_receipt='changed-scope.json')
    changed, count = resolve(first, {'unit, italy': revised})
    assert count > 0
    assert changed['publisher_affiliation_slots'][0]['official_identity_evidence'] == 'changed-scope.json'


def test_incremental_selection_does_not_require_refreshed_gap_links() -> None:
    import sqlite3
    from experiments.fig8_extension.official_units_apply import selected_papers
    db = sqlite3.connect(':memory:')
    db.executescript('''
        CREATE TABLE papers(work_id TEXT PRIMARY KEY,shard TEXT,status TEXT);
        INSERT INTO papers VALUES('new','a','included_verified_research'),
            ('excluded','a','excluded_nonresearch'),
            ('pending','b','pending_type'),
            ('unselected','b','included_verified_research');
    ''')
    assert selected_papers(db, {'new', 'excluded', 'pending', 'absent'}) == {'a': {'new'}}
    assert selected_papers(db, set()) == {}
    db.close()

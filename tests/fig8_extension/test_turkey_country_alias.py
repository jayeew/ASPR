from experiments.fig8_extension.nature_backfill_resolve import address_country, country_names
from experiments.fig8_extension.supplement_apply import countries


def test_old_and_current_country_name_and_explicit_site() -> None:
    names = country_names()
    assert address_country('Department of Physics, Adana, Turkey', names) == 'TR'
    assert address_country('Department of Physics, Adana, Türkiye', names) == 'TR'
    assert address_country('Turkey Research Centre, London, United Kingdom', names) == 'GB'
    assert address_country('University, Adana', names) is None
    assert countries([{'text': 'University, Adana, Turkey', 'countries': []}], names) == ['TR']


def test_same_source_country_repair_preserves_reviewed_institution() -> None:
    from experiments.fig8_extension.supplement_apply import enrich_authors

    address = 'University, Adana, Turkey'
    author = {'name': 'A. Author', 'countries': [], 'institution_ids_or_verified_local_entities': ['known'],
              'publisher_affiliation_slots': [{'text': address, 'country_codes': [], 'institution_ids': ['known'],
                  'official_identity_evidence': 'reviewed.json', 'identity_status': 'official_exact_declared_name',
                  'entity_type': 'education', 'university_ownership_status': 'officially_confirmed_university_self'}]}
    source = {'evidence_receipt': 'raw.html.gz', 'authors': [{'name': 'A. Author', 'nonpersonal': False,
              'unresolved_affiliation_refs': [], 'affiliations': [{'text': address, 'countries': [], 'identifiers': [], 'names': []}]}]}
    revised, counts = enrich_authors([author], source, {}, {}, country_names())
    assert revised[0]['countries'] == ['TR']
    assert revised[0]['publisher_affiliation_slots'][0]['institution_ids'] == ['known']
    assert revised[0]['publisher_affiliation_slots'][0]['official_identity_evidence'] == 'reviewed.json'
    assert counts['country_author_filled'] == 1

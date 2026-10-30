from experiments.fig8_extension.ustc_deposited_self import confirmed_links


def test_exact_address_must_belong_to_the_same_unique_author_and_source():
    receipt = 'fields/crossref_raw/batch.json.gz'
    aliases = {'university of science and technology of china'}
    old = [{'name': 'Alice Wang', 'publisher_affiliation_source': receipt,
            'publisher_affiliation_slots': [{'text': next(iter(aliases))}]}]
    source = {'authors': [{'name': 'Alice Wang', 'affiliations': [{'text': next(iter(aliases))}]}]}
    assert confirmed_links(old, source, receipt, aliases)
    assert not confirmed_links(old, source, receipt + '.other', aliases)
    source['authors'][0]['name'] = 'Bob Wang'
    assert not confirmed_links(old, source, receipt, aliases)
    source['authors'][0]['name'] = 'Alice Wang'
    source['authors'].append(source['authors'][0])
    assert not confirmed_links(old, source, receipt, aliases)


def test_existing_nature_mapping_is_not_reopened():
    old = [{'name': 'Alice Wang', 'nature_v1_slots': [],
            'publisher_affiliation_slots': [{'text': 'university'}]}]
    assert not confirmed_links(old, {'authors': []}, 'source', {'university'})

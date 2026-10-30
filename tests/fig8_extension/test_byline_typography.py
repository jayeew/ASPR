from experiments.fig8_extension.byline_typography import compare_typography, name_key


def test_punctuation_keeps_word_boundaries():
    assert name_key('Jin‐Ju Jeong') == name_key('Jin-Ju Jeong') == name_key('Jin Ju Jeong')
    assert name_key('Ceylan E. Tanes') == name_key('Ceylan E Tanes')
    assert name_key('Hongying Li') != name_key('Hong-Ying Li')
    assert name_key('A Smith') != name_key('Anna Smith')


def test_same_names_require_exact_addresses_and_nontruncated_roster():
    old = [{'name': 'Jin‐Ju Jeong', 'publisher_affiliation_slots': [{'text': 'University A'}]}]
    incoming = {'authors': [{'name': 'Jin-Ju Jeong', 'nonpersonal': False,
                            'affiliations': [{'text': 'University A'}], 'unresolved_affiliation_refs': []}]}
    assert compare_typography(old, incoming) == 'equivalent_complete_declared_links_entity_mapping_not_asserted'
    incoming['byline_has_etal'] = True
    assert compare_typography(old, incoming) == 'incoming_byline_completeness_unconfirmed'
    incoming['byline_has_etal'] = False
    incoming['authors'][0]['affiliations'][0]['text'] = 'University B'
    assert compare_typography(old, incoming) == 'affiliation_statements_differ'


def test_normalized_duplicate_names_remain_unresolved():
    old = [{'name': 'Jin-Ju Jeong'}, {'name': 'Jin Ju Jeong'}]
    assert compare_typography(old, {'authors': old}) == 'duplicate_author_names_require_review'

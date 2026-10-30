"""Typography variants cannot erase institution, campus or address distinctions."""
from experiments.fig8_extension.exact_signature_variants import normalized_signature


def test_punctuation_labels_and_diacritics_are_equivalent() -> None:
    assert normalized_signature('12 INFN, Sezione di Bologna, Bologna, Italy') == normalized_signature('INFN Sezione di Bologna, Bologna, Italy')
    assert normalized_signature('ETH Zürich') == normalized_signature('ETH Zurich')
    assert normalized_signature('Università di Milano Bicocca') == normalized_signature('Università di Milano-Bicocca')


def test_entity_branch_and_address_words_remain_distinct() -> None:
    for a, b in [('University of California, Irvine', 'University of California, Berkeley'),
                 ('Harvard Medical School', 'Harvard University'),
                 ('Nikhef and University of Amsterdam', 'Nikhef'),
                 ('CERN, Geneva, Switzerland', 'CERN'),
                 ('University of Tokyo Hospital', 'University of Tokyo'),
                 ('University, Road 12', 'University, Road 21')]:
        assert normalized_signature(a) != normalized_signature(b)

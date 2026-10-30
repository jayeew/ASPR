from experiments.fig8_extension.latin_america_signed_self import classify


def test_named_entities_and_typographic_accents():
    assert classify('Universidad de Sonora (UNISON), Hermosillo, Mexico') == 'sonora'
    assert classify('Universidade Federal do Rio de Janeiro (UFRJ), Rio de Janeiro, Brazil') == 'ufrj'
    assert classify('Centro Brasileiro de Pesquisas Físicas (CBPF), Rio de Janeiro, Brazil') == 'cbpf'


def test_similar_names_joint_affiliations_and_departments_are_not_inferred():
    for name in [
        'Universidade Federal do Estado do Rio de Janeiro, Rio de Janeiro, Brazil',
        'Universidade Federal Rural do Rio de Janeiro, Rio de Janeiro, Brazil',
        'Universidad Estatal de Sonora, Hermosillo, Mexico',
        'Universidade Federal do Rio de Janeiro, Instituto de Física, Rio de Janeiro, Brazil',
        'Centro Brasileiro de Pesquisas Físicas and UFRJ, Rio de Janeiro, Brazil',
        'Universidad de Sonora, Campus Caborca, Mexico',
    ]:
        assert classify(name) is None

from experiments.fig8_extension.fig8b_qualification_build import plain
from experiments.fig8_extension.supplement_apply import signature_match


def test_literal_inequality_preserves_title_identity() -> None:
    title = 'Magnetic behavior of perovskites (0.0<x<0.5)'
    assert plain(title) == title
    assert signature_match(
        {'title': title, 'journal_name': 'Materials'},
        {'title': [title], 'issns': ['1234-5678']},
        {'1234-5678'},
    )


def test_actual_title_markup_is_still_removed() -> None:
    assert plain('Properties of <i>Fe</i><sub>2</sub>O<sub>3</sub>') == 'Properties of Fe2O3'

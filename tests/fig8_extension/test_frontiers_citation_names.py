from experiments.fig8_extension.frontiers_authors import citation_sequence_matches


def test_only_structurally_explained_name_formats() -> None:
    authors = [{'givenNames': 'Jane', 'surname': 'Doe', 'suffix': 'III'},
               {'givenNames': 'Pooja', 'surname': ''},
               {'givenNames': 'José', 'surname': 'García'}]
    assert citation_sequence_matches(authors, ['Doe, Jane', 'Pooja, Pooja', 'Garcia, Jose'])
    assert citation_sequence_matches(authors, ['Doe III, Jane', 'Pooja, null', 'García, José'])
    assert not citation_sequence_matches(authors, ['Doe, J.', 'Pooja, Pooja', 'Garcia, Jose'])
    assert not citation_sequence_matches(authors, ['Doe, Jane', 'Pooja, Other', 'Garcia, Jose'])
    assert not citation_sequence_matches(authors, ['Garcia, Jose', 'Pooja, Pooja', 'Doe, Jane'])
    assert not citation_sequence_matches(authors, ['Doe, Jane'])
    assert citation_sequence_matches([{'givenNames': '', 'surname': 'Navpreet'}], ['Navpreet, null'])
    assert citation_sequence_matches([{'givenNames': '', 'surname': 'Pooja'}], ['Pooja, Pooja'])


def test_ambiguous_collisions_and_missing_declared_suffix_remain_unmatched() -> None:
    assert not citation_sequence_matches([{'givenNames': 'Jose', 'surname': 'Garcia'},
        {'givenNames': 'José', 'surname': 'García'}], ['Garcia, Jose', 'García, José'])
    assert not citation_sequence_matches([{'givenNames': 'Jane', 'surname': 'Doe III'}], ['Doe, Jane'])
    assert not citation_sequence_matches([{'givenNames': 'Jane', 'surname': 'Doe', 'suffix': 'Jr.'},
        {'givenNames': 'Jane', 'surname': 'Doe', 'suffix': 'Sr.'}], ['Doe, Jane', 'Doe, Jane'])

from experiments.fig8_extension.postal_country_cache import explicit_country


def test_explicit_country_not_postcode_or_institution_inference():
    assert explicit_country('Department, Changsha, Hunan 410008 People’s Republic of China') == 'CN'
    assert explicit_country("Department, Beijing, 100190 People's Republic of China.") == 'CN'
    for text in ['Department, China University, 100190', 'Department, Changsha, 410008',
                 'Department, Taipei, 100190 Republic of China',
                 "Department; other address, City, 100190 People's Republic of China",
                 "Department, City, 1100190 People's Republic of China"]:
        assert explicit_country(text) is None

from experiments.fig8_extension.aps_annual_cache import copyright_year


def test_copyright_and_publication_year_and_named_publisher_must_agree() -> None:
    article = {'date': '2023-04-01', 'publisher': {'name': 'APS'}, 'rights': {
        'copyrightYear': 2023, 'rightsStatement': '© 2023 American Physical Society',
        'copyrightHolders': [{'name': 'American Physical Society'}]}}
    assert copyright_year(article) == 2023
    assert copyright_year({**article, 'date': '2024-01-01'}) is None
    assert copyright_year({**article, 'publisher': {'name': 'Other'}}) is None
    assert copyright_year({**article, 'rights': {**article['rights'], 'copyrightHolders': [{'name': 'Authors'}]}}) is None


def test_open_access_explicit_publishing_body_does_not_require_copyright_ownership():
    article = {'date':'2023-02-06','publisher':{'name':'APS'},'rights':{
        'copyrightYear':2023,'rightsStatement':'Published by the American Physical Society',
        'creativeCommons':True,'copyrightHolders':[]}}
    assert copyright_year(article) == 2023
    assert copyright_year({**article,'rights':{**article['rights'],'rightsStatement':'The Authors'}}) is None
    assert copyright_year({**article,'rights':{**article['rights'],'copyrightYear':2024}}) is None
    assert copyright_year({**article,'publisher':{'name':'Other'}}) is None

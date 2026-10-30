from experiments.fig8_extension.elife_authors import metadata


def test_native_affiliation_country_and_group_preserved() -> None:
    article = {'doi': '10.7554/eLife.1', 'title': 'Study', 'authors': [
        {'type': 'person', 'name': {'preferred': 'A Smith'}, 'affiliations': [
            {'name': ['Example Institute'], 'address': {'formatted': ['Paris', 'France'],
                                                       'components': {'country': 'France'}}}]},
        {'type': 'group', 'name': 'Study Consortium', 'affiliations': []}]}
    parsed = metadata(article)
    assert parsed['authors'][0]['affiliations'][0]['countries'] == [{'code': '', 'text': 'France'}]
    assert parsed['authors'][1]['nonpersonal']
    assert not parsed['byline_has_etal']
    assert parsed['doi'] == ['10.7554/elife.1']

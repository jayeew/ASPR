import json

import pytest

from experiments.fig8_extension.frontiers_content_cache import content_text, extract


def page(body_doi: str = '123') -> bytes:
    values = []

    def encode(value: object) -> int:
        index = len(values); values.append(None)
        if isinstance(value, dict):
            values[index] = {k: encode(v) for k, v in value.items()}
        elif isinstance(value, list):
            values[index] = [encode(v) for v in value]
        else:
            values[index] = value
        return index

    encode({'currentArticle': {'doi': '10.3389/test.123', 'title': 'Title',
                              'journal': {'name': 'Journal', 'electronicISSN': '1234-5678'}},
            'xmlParsedArticleContent': {'doi': body_doi, 'title': [{'#text': 'Title'}],
                'journal': {'title': 'Journal', 'issn': '1234-5678'},
                'authors': [{'givenNames': 'Jane', 'surname': 'Doe', 'affiliationsIndexes': [1]}],
                'affiliations': [{'index': 1, 'textContent': 'University, France'}],
                'abstractContent': [{'p': [{'#text': 'Abstract'}]}],
                'bodyContent': [{'p': [{'#text': 'Original '}, {'italic': [{'#text': 'body'}]}]}],
                'references': [{'#text': 'Unrelated reference content'}]}})
    return (f'<meta name="citation_doi" content="10.3389/test.123">'
            f'<meta name="citation_title" content="Title">'
            f'<meta name="citation_author" content="Doe, Jane">'
            f'<script id="__NUXT_DATA__">{json.dumps(values)}</script>').encode()


def test_extracts_only_identity_bound_body() -> None:
    result = extract(page(), '10.3389/test.123')
    assert result['body'] == 'Original body'
    assert result['abstract'] == 'Abstract'
    with pytest.raises(ValueError, match='disagreement'):
        extract(page('999'), '10.3389/test.123')
    with pytest.raises(ValueError, match='DOI'):
        extract(page(), '10.3389/other.123')


def test_repeated_text_is_preserved_but_cycles_rejected() -> None:
    assert content_text([[1, 1], {'#text': 2}, 'x'], 0) == 'xx'
    with pytest.raises(ValueError, match='cyclic'):
        content_text([[0]], 0)


def test_inner_author_links_require_bound_article_and_complete_name_sequence() -> None:
    from experiments.fig8_extension.frontiers_authors import parse_authors
    result = parse_authors(page(), '10.3389/test.123')
    assert result['authors'][0]['name'] == 'Jane Doe'
    assert result['authors'][0]['affiliations'][0]['text'] == 'University, France'
    assert not result['byline_has_etal']
    assert parse_authors(page('999'), '10.3389/test.123')['authors'] == []
    assert parse_authors(page().replace(b'Doe, Jane', b'Other, Person'), '10.3389/test.123')['authors'] == []


def test_comparison_adapter_uses_native_frontiers_links() -> None:
    from experiments.fig8_extension.primary_byline_comparison import incoming_metadata
    record = {'route': 'frontiers', 'source': 'publisher', 'doi': '10.3389/test.123'}
    parsed = incoming_metadata(record, page())
    assert parsed['authors'][0]['name'] == 'Jane Doe'
    assert not parsed['byline_has_etal']
    assert incoming_metadata(record, page('999'))['authors'] == []

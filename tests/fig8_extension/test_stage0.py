from __future__ import annotations

from typing import Any

import pytest

from experiments.fig8_extension.stage0 import Client, entity_id, filters
from experiments.fig8_extension.stage0_roster import abstract_text


def test_entity_namespaces_are_not_conflated() -> None:
    assert entity_id('https://openalex.org/S137773608', 'sources') == 'https://openalex.org/S137773608'
    assert entity_id('https://openalex.org/fields/13', 'fields') == 'https://openalex.org/fields/13'
    assert entity_id('T12345', 'topics') == 'https://openalex.org/T12345'
    assert entity_id('W12345', 'sources') is None
    assert entity_id('S12345', 'works') is None


def test_scope_uses_paper_primary_field_and_explicit_years() -> None:
    query = filters('materials', 'has_abstract:true')
    assert 'primary_topic.field.id:25' in query
    assert 'publication_year:2023-2025' in query
    assert 'primary_location.source.type:journal' in query
    assert query.endswith('has_abstract:true')


def test_abstract_reconstruction_preserves_order_and_repetition() -> None:
    assert abstract_text({'the': [0, 2], 'study': [1], 'result': [3]}) == 'the study the result'
    assert abstract_text(None) is None


class FakeClient(Client):
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = iter(payloads)

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        assert 'select' not in params
        assert 'cursor' in params
        return next(self.payloads)


def test_groups_fetch_all_pages_and_preserve_unknown() -> None:
    client = FakeClient([
        {'meta': {'count': 9, 'next_cursor': 'next'}, 'group_by': [{'key': 'S1', 'count': 7}]},
        {'meta': {'count': 9, 'next_cursor': None}, 'group_by': [{'key': 'unknown', 'count': 2}]},
    ])
    total, rows = client.groups('filter', 'source')
    assert total == 9 and sum(row['count'] for row in rows) == 9


def test_group_count_drift_is_not_silently_accepted() -> None:
    client = FakeClient([
        {'meta': {'count': 9, 'next_cursor': 'next'}, 'group_by': [{'key': 'S1', 'count': 7}]},
        {'meta': {'count': 10, 'next_cursor': None}, 'group_by': [{'key': 'S2', 'count': 3}]},
    ])
    with pytest.raises(RuntimeError, match='Inconsistent'):
        client.groups('filter', 'source')

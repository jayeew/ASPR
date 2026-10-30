"""Decode the publisher's HTML text fields, including entity-only strings."""
from __future__ import annotations

from lxml import html


def plain(value: str) -> str:
    return html.fragment_fromstring(value, create_parent='div').text_content()

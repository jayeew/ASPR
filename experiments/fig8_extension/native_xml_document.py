"""Strict native XML parsing with one documented missing MathML binding repair."""
from __future__ import annotations

import re

from lxml import etree


def document(raw: bytes) -> tuple[etree._Element, list[str]]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        return etree.fromstring(raw, parser), []
    except etree.XMLSyntaxError:
        errors = list(parser.error_log)
        opening = re.search(rb'<article\b[^>]*>', raw)
        if (not errors or not opening
                or b'xmlns="http://www.elsevier.com/xml/ja/dtd"' not in opening.group()
                or b'xmlns:mml=' in opening.group()
                or any(e.type_name != 'NS_ERR_UNDEFINED_NAMESPACE'
                       or not e.message.startswith('Namespace prefix mml on ') for e in errors)):
            raise
        repaired = (raw[:opening.end()-1]
                    + b' xmlns:mml="http://www.w3.org/1998/Math/MathML"'
                    + raw[opening.end()-1:])
        tree = etree.fromstring(repaired, etree.XMLParser(resolve_entities=False, no_network=True))
        return tree, ['Declared missing mml prefix as standard MathML namespace; original bytes retained']

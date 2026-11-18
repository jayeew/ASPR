"""Mask source namespaces and explicit method labels without rewriting science."""
from __future__ import annotations

import re
from typing import Any


def blind(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    sources = payload.get('sources', []) + [r for b in payload.get('evidence_blocks', []) for r in b.get('provenance', [])]
    mapping = {source['source_id']: f'E{i+1:05d}' for i, source in enumerate(sources)}
    pattern = re.compile('|'.join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))) if mapping else None
    labels = re.compile(r'\b(?:ReviewGrounder|GEAR|Direct-A|EACL|Full system|fusion system)\b', re.IGNORECASE)

    def transform(value: Any) -> Any:
        if isinstance(value, str):
            text = pattern.sub(lambda match: mapping[match.group()], value) if pattern else value
            return labels.sub('the analysis system', text)
        if isinstance(value, list):
            return [transform(v) for v in value]
        if isinstance(value, dict):
            return {key: transform(v) for key, v in value.items()}
        return value

    return transform(payload), mapping

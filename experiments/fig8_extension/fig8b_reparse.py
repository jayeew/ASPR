"""Refresh parsed fields from frozen publisher bytes without new retrieval."""
from __future__ import annotations

import gzip
import json

from .fig8b_collect import OUT
from .fig8b_publisher import parse
from .stage0 import write_json


def main() -> None:
    updated = 0
    for path in sorted((OUT / 'publisher_parsed').glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('status') != 'publisher_doi_issn_verified':
            continue
        raw = OUT / record['raw_receipt']
        with gzip.open(raw, 'rt') as stream:
            parsed = parse(stream.read(), record['doi'], record['issn'])
        write_json(path, {**parsed, 'url': record['url'], 'raw_receipt': record['raw_receipt']})
        updated += 1
    print('Reparsed frozen pages', updated, flush=True)


if __name__ == '__main__':
    main()

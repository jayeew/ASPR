"""Verify and package a completed full run; never invoke a model."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import read, write
from .frontier_data import OUT, claim_objects, fact_card, inventory
from .frontier_predict import validate


def finalize() -> dict[str, Any]:
    destination = OUT/'full'
    coverage = read(destination/'coverage.json')
    rows = inventory()
    expected = {r['paper_id'] for r in rows}
    assert coverage['full_1000'] and len(expected)==1000
    assert set(coverage['paper_ids'])==expected
    counts: Counter[str] = Counter()
    for row in rows:
        claims, _ = claim_objects(row)
        counts['papers'] += 1
        for claim in claims.claims:
            card = fact_card(Path(row['graph']),claim.claim_id)
            assert card is not None and card.claim.claim_text==claim.normalized_claim_text
            assert card.claim.claim_id==claim.claim_id
            assert card.insertion_policy=='threshold_parent_path:k=10:cosine>0.5'
            assert len(card.neighbors)<=10
            assert all(n.cosine_similarity>.5 and n.publication_date.isoformat()<row['date']
                       and n.parent_paper_id!=row['paper_id'] for n in card.neighbors)
            counts['claims'] += 1
            counts['semantic_edges'] += len(card.neighbors)
            counts['claims_without_neighbors'] += not card.neighbors
            counts['internally_unsupported_claims'] += claim.internal_support.value=='internally_unsupported'
    assert counts['claims']-counts['internally_unsupported_claims']==coverage['target_claim_count']
    good_ids = set()
    for packet_path in sorted((destination/'packets').glob('P*.json')):
        packet = read(packet_path)
        good,_ = validate(packet,read(destination/'predictions'/packet_path.name))
        good_ids.update(c['candidate_id'] for c in good)
    directions = read(destination/'data/directions.json')
    assert {d['candidate_id'] for d in directions}<=good_ids
    word_rows = read(destination/'data/wordcloud.json')
    shown = [r for r in word_rows if r['displayed']]
    import pymupdf
    from PIL import Image
    with pymupdf.open(destination/'final/Fig7.pdf') as pdf:
        text = ' '.join(pdf[0].get_text().split())
        assert all(' '.join(r['display_label'].split()) in text for r in shown)
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(1,1)).save(destination/'qa/pdf_preview.png')
    with Image.open(destination/'final/Fig7_600dpi.png') as image:
        assert image.size[0]==10000 and abs(image.info['dpi'][0]-600)<.1
    calls = [read(p) for p in (destination/'logs/calls').glob('*/record.json')]
    assert len(calls)<=10
    usage = {key:sum((r.get('usage') or {}).get(key,0) for r in calls)
             for key in ('input_tokens','output_tokens','cached_input_tokens')}
    usage['calls_without_token_usage'] = sum(r.get('usage') is None for r in calls)
    status = read(destination/'run_status.json')
    end = datetime.now(timezone.utc)
    elapsed = (end-datetime.fromisoformat(status['started_at_utc'])).total_seconds()
    result = {**dict(counts),'retained_directions':len(directions),'total_phrases':len(word_rows),
              'displayed_phrases':len(shown),'displayed_categories':dict(Counter(r['category'] for r in shown)),
              'forecast_calls':len(calls),'usage':usage,'elapsed_seconds':elapsed,
              'coverage_source_graph_export_checks_passed':True,'predictive_accuracy_validated':False}
    write(destination/'qa/final_validation.json',result)
    write(destination/'run_status.json',{**status,'status':'complete','completed_at_utc':end.isoformat(),**result})
    paths = []
    for folder in ('final','tables','data','packets','sources','predictions','qa','timings'):
        paths.extend(p for p in (destination/folder).rglob('*') if p.is_file())
    paths += [destination/'methods_and_caption.md',destination/'coverage.json',destination/'run_status.json']
    with zipfile.ZipFile(destination/'Fig7_2026_full_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path,path.relative_to(destination))
        for path in sorted((OUT/'materials').glob('*.json')):
            archive.write(path,Path('full_source_materials')/path.name)
    with zipfile.ZipFile(destination/'Fig7_2026_full_bundle.zip') as archive:
        assert archive.testzip() is None
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    finalize()

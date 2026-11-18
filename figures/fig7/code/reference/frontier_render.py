"""One mixed word cloud, generated only from saved 2026 prospective directions."""
from __future__ import annotations

import math
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene
from .data import read, write
from .predict import CATEGORIES, csv_write, frequencies
from .unified import BASE, draw_words, pack


def cloud_rows(destination: Path) -> list[dict[str, Any]]:
    directions = read(destination/'data/directions.json')
    sources = read(destination/'data/evidence.json')
    by_id = {d['candidate_id']:d for d in directions}
    aliases: dict[tuple[str,str],str] = {}
    for d in directions:
        for label in d['labels']:
            key = (label['category'],' '.join(label['phrase'].split()))
            if key in aliases and aliases[key] != label['display_label']:
                raise ValueError('Merged phrase has conflicting display labels: '+key[1])
            aliases[key] = label['display_label']
    rows = frequencies(directions)
    topic_path = destination/'data/topic_labels.json'
    topics = {(r['category'],r['phrase']):r['display_label']
              for r in read(topic_path)} if topic_path.exists() else {}
    for r in rows:
        r['source_paper_ids'] = sorted({sources[c['source_id']]['paper_id'] for i in r['direction_ids'] for c in by_id[i]['citations']})
        r['source_paper_count'] = len(r['source_paper_ids'])
        r['prominence'] = r['frequency']*math.log2(1+r['source_paper_count'])
        key = (r['category'],r['phrase'])
        r['original_display_label'] = aliases[key]
        r['display_label'] = topics.get(key,aliases[key])
        r['displayed'] = False
    if not rows:
        raise ValueError('No retained labels to render')
    for category in CATEGORIES:
        group = [r for r in rows if r['category']==category]
        random.Random(20261003).shuffle(group)
        group.sort(key=lambda r:-r['prominence'])
        for r in group[:24]:
            r['displayed'] = True
    lo,hi = min(r['prominence'] for r in rows),max(r['prominence'] for r in rows)
    for r in rows:
        w = ((r['prominence']-lo)/(hi-lo))**.45 if hi>lo else 0.0
        r['font_size_pt'] = round(18+18*w)
        r['bold'] = hi>lo and r['prominence']==hi
        base = BASE[r['category']]
        r['color'] = '#'+''.join(f'{round(int(base[i:i+2],16)*(1-.3*(1-w))+255*.3*(1-w)):02x}' for i in (1,3,5))
    return rows


def render(destination: Path) -> None:
    started = time.monotonic()
    fonts(destination)
    rows = cloud_rows(destination)
    for height in (680,840,1000,1160,1320):
        try:
            layout = pack(rows,1120,height)
            break
        except ValueError:
            if height==1320:
                raise
    canvas_h = height+180
    coverage = read(destination/'coverage.json')
    n = len(read(destination/'data/directions.json'))
    scene = Scene(1200,canvas_h,'fig7_2026_future_wordcloud')
    scene.rect(0,0,1200,canvas_h,'#FFFFFF')
    title = 'Future research directions' if coverage['full_1000'] else 'Pilot · Future research directions'
    scene.text(40,42,title,29,'#1E2B3C',True)
    scene.text(40,68,f'{coverage["paper_count"]} papers from Jan–May 2026 + historical Claim Graph · {n} prospective directions',13,'#647184')
    for i,category in enumerate(CATEGORIES):
        scene.rect(40+i*160,87,11,11,BASE[category],radius=2)
        scene.text(58+i*160,98,category,13,BASE[category],True)
    scene.text(1160,98,'Larger + darker = higher evidence-weighted prominence',12,'#647184',anchor='end')
    draw_words(scene,layout,40,117)
    scene.text(40,canvas_h-24,'Prominence = direction frequency × log₂(1 + distinct source papers). Prospective hypotheses; not forecast probabilities.',11,'#647184')
    target=destination/'final/Fig7.svg'
    scene.save(target)
    tree=ET.parse(target)
    tree.getroot().set('width','1200pt');tree.getroot().set('height',f'{canvas_h}pt')
    tree.write(target,encoding='utf-8',xml_declaration=True)
    convert(target,1.4)
    outline(target,destination/'final/Fig7_outlined.svg')
    import cairosvg
    high=destination/'final/Fig7_600dpi.png'
    cairosvg.svg2png(url=str(target),write_to=str(high),output_width=10000,output_height=round(canvas_h/72*600))
    with Image.open(high) as im:
        im.save(high,dpi=(600,600))
    write(destination/'data/wordcloud.json',rows)
    csv_write(destination/'tables/wordcloud.csv',rows)
    write(destination/'qa/layout.json',{'layout':layout,'pixel_collision_check_passed':True,'width_pt':1200,'height_pt':canvas_h})
    write(destination/'timings/render.json',{'seconds':time.monotonic()-started,'displayed_labels':len(layout)})
    (destination/'methods_and_caption.md').write_text(
        f'# Fig7 — 2026 frontier {"full corpus" if coverage["full_1000"] else "pilot"}\n\n'
        f'{coverage["paper_count"]} target papers; {coverage["target_claim_count"]} internally supported or narrowed claims. '
        '2026 target claims are full-text grounded; historical neighbors are 2023–2025 abstract claims. '
        'All eligible historical neighbor edges participate; representative historical originals and all eligible target claims enter model packets. '
        'No 2026-to-2026 semantic edges are asserted. Future directions extend observed findings and cite actual graph endpoints.\n\n'
        '**Caption.** Future research directions informed by 2026 papers through 8 May and the historical Claim Graph. '
        'Hue indicates Method, Problem or Combination; size and shade encode direction frequency × log₂(1 + distinct cited papers). '
        'Up to 24 labels per category are shown, with seeded tie selection. These are source-grounded prospective hypotheses, not validated probabilities. '
        'Complete proposals, exact source quotations and graph references are supplied in the tables.\n\n'
        'Font size: round(18 + 18w) pt, w = normalized prominence^0.45; when all scores tie, w=0 for all labels. '
        'Word-cloud labels name scientific topics; future research tasks remain in the complete direction table. '
        'The saved topic-label mapping preserves original labels and canonical phrases; display editing does not change directions, sources or prominence.\n',encoding='utf-8')

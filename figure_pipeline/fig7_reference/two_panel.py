"""Two-panel Fig7: full historical phrase statistics and saved 2026 forecasts."""
from __future__ import annotations

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene
from .data import ROOT, read, write
from .frontier_render import cloud_rows
from .historical_statistics import prepare
from .predict import CATEGORIES, csv_write
from .unified import BASE, draw_words, pack

OUT = ROOT/'outputs/fig7_reference/two_panel'
FORECAST = ROOT/'outputs/fig7_reference/frontier_2026/full'
W, H = 1400, 890
CARD_W, CARD_H = 682, 816
CLOUD_W, CLOUD_H = 1000, 1080


def panel(key: str, title: str, rows: list[dict[str, Any]]) -> tuple[Scene,list[dict[str,Any]]]:
    scene = Scene(CARD_W,CARD_H,'panel_'+key)
    scene.rect(0,0,CARD_W,CARD_H,'#FFFFFF','#90b8cf',3,sw=.8)
    scene.rect(1,1,CARD_W-2,42,'url(#bluewash)')
    scene.text(10,29,key,28,'#000000',True,sans=True)
    scene.text(43,29,title,23,'#101d3b',True)
    for i, category in enumerate(CATEGORIES):
        x = 115+i*160
        scene.rect(x,58,10,10,BASE[category],radius=2)
        scene.text(x+17,69,category,17,BASE[category],True)
    if key=='a':
        maximum = max(r['frequency'] for r in rows)
        rows = [{**r,'horizontal_only':r['frequency']>=maximum*.7} for r in rows]
    layout = pack(rows,CLOUD_W,CLOUD_H,gap=6 if key=='a' else 0,rectangular=True)
    cloud = Scene(CLOUD_W,CLOUD_H,'cloud_'+key)
    draw_words(cloud,layout,0,0)
    # Replace generic forecast tooltips with the appropriate statistical unit.
    if key=='a':
        titles = list(cloud.root.iter('{http://www.w3.org/2000/svg}title'))
        for title_node, row in zip(titles,layout):
            title_node.text = f'{row["display_label"]}: {row["frequency"]} distinct historical papers; {row["claim_count"]} claims'
    group = copy.deepcopy(cloud.root)
    left = min(r['x'] for r in layout)
    top = min(r['y'] for r in layout)
    width = max(r['x']+r['width'] for r in layout)-left
    height = max(r['y']+r['height'] for r in layout)-top
    scale = min((CARD_W-32)/width,(CARD_H-108)/height)
    ox = (CARD_W-width*scale)/2-left*scale
    oy = 88+(CARD_H-108-height*scale)/2-top*scale
    group.set('transform',f'translate({ox} {oy}) scale({scale})')
    for row in layout:
        row['panel_scale'] = scale
        row['rendered_font_size_pt'] = row['font_size_pt']*scale
        row['panel_x'] = ox+row['x']*scale
        row['panel_y'] = oy+row['y']*scale
        assert row['panel_x']>=16-.01 and row['panel_x']+row['width']*scale<=CARD_W-16+.01
        assert row['panel_y']>=88-.01 and row['panel_y']+row['height']*scale<=CARD_H-20+.01
    scene.add(group)
    return scene,layout


def render() -> None:
    fonts(OUT)
    history = read(OUT/'data/historical_wordcloud.json')
    future = cloud_rows(FORECAST)
    write(OUT/'data/forecast_wordcloud.json',future)
    csv_write(OUT/'tables/forecast_wordcloud.csv',future)
    figure = Scene(W,H,'fig7_historical_and_future')
    figure.rect(0,0,W,H,'#FFFFFF')
    figure.text(12,35,'Fig. 7 | Historical research topics and future directions',33,'#101d3b',True)
    layouts = {}
    for key,title,rows,x in (
        ('a','Historical topics (2023–2025)',history,10),
        ('b','Future directions (1,000 papers, 2026)',future,708)):
        card,layout = panel(key,title,rows)
        card.save(OUT/f'panels/{key}.svg')
        figure.use(card,x,60)
        layouts[key] = layout
    target = OUT/'final/Fig7.svg'
    figure.save(target)
    tree = ET.parse(target)
    tree.getroot().set('width',f'{W}pt')
    tree.getroot().set('height',f'{H}pt')
    tree.write(target,encoding='utf-8',xml_declaration=True)
    convert(target,1.5)
    outline(target,OUT/'final/Fig7_outlined.svg')
    import cairosvg
    high = OUT/'final/Fig7_600dpi.png'
    cairosvg.svg2png(url=str(target),write_to=str(high),output_width=round(W/72*600),output_height=round(H/72*600))
    with Image.open(high) as image:
        image.save(high,dpi=(600,600))
    write(OUT/'qa/layout.json',{'width_pt':W,'height_pt':H,'cloud_scaling':'uniform fit to each occupied bounding box',
                              'glyph_collision_check_passed':True,'panels':layouts})
    for name in ('directions.json','evidence.json','topic_labels.json'):
        shutil.copy2(FORECAST/'data'/name,OUT/'data'/name)
    for name in ('predicted_directions.csv','direction_sources.csv'):
        shutil.copy2(FORECAST/'tables'/name,OUT/'tables'/name)
    (OUT/'methods_and_caption.md').write_text(
        '# Fig7 — historical statistics and prospective directions\n\n'
        '**a.** Corpus-ranked research-topic noun phrases from all 70,034 historical abstract-derived claims in '
        '24,914 papers dated 2023–2025. spaCy 3.8.7 / en_core_web_sm 3.8.0 extracts complete noun phrases '
        'with 3–6 lexical tokens. Determiners and listed generic leading modifiers are trimmed; generic '
        'measurement/reporting heads and clausal expressions are filtered using fixed linguistic rules. '
        'An explicit plural-normalization dictionary is applied. No topic whitelist is used. '
        'Every candidate is then matched across all claims; each distinct parent paper counts once. '
        'All candidates occurring in at least five papers are ranked by paper frequency descending, with '
        'canonical phrase alphabetical order breaking ties. If a candidate is contained in a longer '
        'candidate with at least 80% of its frequency, the shorter expression is suppressed from display '
        'but retained in the full table. The first 68 remaining are displayed, without category '
        'quotas, manual additions, replacements or semantic synonym merging. Display text is the most '
        'frequent extracted original surface form. Full candidate ranking, exact matched spans and source '
        'IDs are supplied. This is a reproducible automatic noun-phrase statistic, not a validated exhaustive '
        'semantic classification of scientific topics. All nodes are scanned, '
        'including nodes without community assignments. Glyph size is round(18 + 18 × (frequency/max)^0.65) '
        'before uniform scaling to fill its panel content area. Actual scales and rendered font sizes '
        'are saved in qa/layout.json. Hue indicates a rule-based primary category: Method '
        '(technical or intervention route), Problem (research object, process or mechanism), or '
        'Combination (coupled processes or integrated material/device systems). These describe '
        'historical topics, not future tasks. Categories are not inferred from graph node roles; '
        'explicit coupling/architecture cues take precedence over method cues; remaining research objects '
        'and processes are assigned Problem. This coarse display classification does not alter ranking. '
        'The same blue, teal '
        'and orange category hues are used in both panels. Frequency and font size do not depend on category.\n\n'
        '**b.** The existing 96 prospective directions inferred from 1,000 papers dated 10 January–8 May '
        '2026 and their actual 2023–2025 Claim Graph neighborhoods; 68 scientific-topic labels are displayed. '
        'Method, Problem and Combination use blue, teal and orange. The prior selection and source-based '
        'prominence are unchanged: direction frequency × log₂(1 + distinct cited papers). Font sizes and '
        'shades follow the existing renderer, with uniform scaling to fill its panel content area. Future tasks and sources '
        'remain in the full direction table. These are source-grounded hypotheses, not calibrated predictions.\n\n'
        'The panels share historical evidence context; panel a’s phrase counts were not supplied to the '
        'forecast model and are not an input to panel b. The two font scales measure different quantities '
        'and should not be compared as growth or forecast probability. The historical graph was read only; '
        'only a local grammatical parser was run; no generative forecast calls were made for this revision.\n\n'
        'Caption: **Fig. 7 | Historical research topics and future directions.** '
        '(a) Highest-frequency eligible research-topic noun phrases across the 2023–2025 Claim Graph corpus. '
        '(b) Prospective research topics grounded in the 2026 paper cohort and historical graph evidence. '
        'Panel a size represents distinct-paper frequency; panel b size and shade represent evidence-weighted '
        'direction prominence. Full definitions and claim-level provenance accompany the figure.\n',encoding='utf-8')


def verify_and_package() -> None:
    import pymupdf
    layout = read(OUT/'qa/layout.json')
    with pymupdf.open(OUT/'final/Fig7.pdf') as pdf:
        content = ' '.join(pdf[0].get_text().split())
        for rows in layout['panels'].values():
            assert all(' '.join(r['display_label'].split()) in content for r in rows)
        pdf[0].get_pixmap().save(OUT/'qa/pdf_preview.png')
    assert len(layout['panels']['a'])==68 and len(layout['panels']['b'])==68
    assert (OUT/'data/directions.json').read_bytes()==(FORECAST/'data/directions.json').read_bytes()
    write(OUT/'qa/export_checks.json',{'historical_labels':68,'forecast_labels':68,
          'all_pdf_labels_present':True,'forecast_directions_unchanged':True,'new_model_calls':0})
    with zipfile.ZipFile(OUT/'Fig7_two_panel_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('final','panels','tables','data','qa'):
            for path in sorted((OUT/folder).rglob('*')):
                if path.is_file():
                    archive.write(path,path.relative_to(OUT))
        archive.write(OUT/'methods_and_caption.md','methods_and_caption.md')
    with zipfile.ZipFile(OUT/'Fig7_two_panel_bundle.zip') as archive:
        assert archive.testzip() is None


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','render','all'],default='all',nargs='?')
    stage = parser.parse_args().stage
    if stage in ('prepare','all'):
        prepare(OUT)
    if stage in ('render','all'):
        render()
        verify_and_package()


if __name__=='__main__':
    main()

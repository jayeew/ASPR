"""Source-bound DAMAGE figure; preserves the original study artifacts."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from figure_pipeline.fig6_computer import build as base
from figure_pipeline.fig1_reference.svg import Scene, mix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig6_reference'
PID = 's41467-026-69179-5'
CASE = ROOT / 'outputs/fig6_damage_study/papers' / PID
base.OUT = OUT
base.PID = PID
base.TITLES = ['Manuscript and grounded contributions', 'Historical evidence and differences',
               'Local neighborhoods and joint context', 'Evidence-constrained judgments',
               'Traceable report and scientific takeaways']
LABELS = ['RNA-to-death circuit', 'Effector engineering', 'Cell selectivity',
          'Target breadth', 'Sequence discrimination', 'RNA delivery']
TEXTS = ['RNA recognition triggers a\nswitch for cell-death output.',
         'Engineered effector and\nintegrated expression architecture.',
         'Target-dependent effects\nin the tested cell models.',
         'One framework evaluated\nacross distinct RNA contexts.',
         'Sequence matching and transcript\ncontext constrain the response.',
         'RNA-format and mRNA/LNP\ndelivery evaluated in cells.']
WORKS = [('W4308430559',1,'RNA protease','2022'),('W4292971025',2,'Native output','2022*'),
         ('W4391953677',3,'Logic gate','2024')]
SELECTED = ['s41467-024-52962-7::C01','s41467-023-41006-1::C02','s41467-023-41006-1::C03',
's41467-024-50243-x::C01','s41467-024-55083-3::C01','s41467-024-47634-5::C03',
's41467-024-52110-1::C03','s41467-023-43707-z::C01','s41467-025-62350-4::C01',
's41467-023-39938-9::C01','s41467-023-39938-9::C02','s41467-023-39938-9::C03']
NAMES = ['RNA-IN / RNA-OUT','RNA detection','Mutation detection','RNA-guided cleavage',
         'Regulated gasdermin','Controlled pyroptosis','Cell-death switch','Gasdermin inhibition',
         'KRAS targeting','mRNA/LNP delivery','Pyroptosis induction','Tumor-model evidence']


def prepare() -> dict:
    (OUT/'data').mkdir(parents=True,exist_ok=True)
    claims=base.read(CASE/'shared/claims.json')['claims']
    spans={s['span_id']:s for s in base.read(CASE/'shared/paper_ir.json')['spans']}
    cards={f'C{i:02}':base.trace(CASE/f'graph/{i:02}/evidence_trace.jsonl')[f'GRAPH:{PID}::CLAIM::{i:02}'] for i in range(1,7)}
    history={n['claim_id']:n for c in cards.values() for n in c['neighbors']}
    aliases={n:f'H{i:02}' for i,n in enumerate(SELECTED+sorted(set(history)-set(SELECTED)),1)}
    facts=base.read(CASE/'graph/joint/facts.json')
    local={tuple(sorted(e)) for c in cards.values() for e in c['neighbor_edges']}
    extra=[e for e in facts['historical_edges'] if tuple(sorted(e)) not in local]
    gear={f'C{i:02}':base.trace(CASE/f'gear/{i:02}/evidence_trace.jsonl') for i in range(1,7)}
    works={f'E{j:02}':gear[f'C{ci:02}'][f'WORK:{PID}::CLAIM::{ci:02}:https://openalex.org/{wid}'] for j,(wid,ci,_,_) in enumerate(WORKS,1)}
    relations=[{'claim':f'C{i:02}','work':f'E{j:02}','assessment':gear[f'C{i:02}'].get(f'RELATION:{PID}::CLAIM::{i:02}:https://openalex.org/{wid}')} for i in range(1,7) for j,(wid,_,_,_) in enumerate(WORKS,1)]
    report=base.read(ROOT/f'outputs/innovation_200_20260907/reports/fusion/{PID}.json')
    d=dict(claims=claims,spans=spans,cards=cards,history=history,aliases=aliases,facts=facts,extra=extra,works=works,relations=relations,report=report)
    assert len(history)==facts['historical_neighbor_count']==47
    assert len(facts['insertion_edges'])==60
    base.write(OUT/'data/snapshot.json',d)
    return d


def manuscript(d: dict) -> Scene:
    s=base.panel('a');p=base.para
    p(s,14,56,'A synthetic system for RNA-responsive pyroptosis based on type III-E CRISPR nuclease-protease',258,20,23,True)
    s.text(14,173,'He et al. · Nat. Commun. (2026)',16,base.MUTED)
    s.text(14,196,'DAMAGE · 6 grounded claims',17,base.INK,True)
    s.line(283,44,283,224,base.BORDER)
    for i,c in enumerate(d['claims']):
        x,y=296+(i%3)*257,43+(i//3)*93;color=base.COLORS[i]
        s.rect(x,y,246,85,mix(color,'#FFFFFF',.96),mix(color,'#FFFFFF',.6),3)
        s.text(x+8,y+19,f'C{i+1:02}  {LABELS[i]}',17,color,True)
        p(s,x+8,y+40,TEXTS[i],230,16,18)
    return s


def historical(d: dict) -> Scene:
    s=base.panel('b');p=base.para
    s.text(14,55,'Selected works × saved claim-level assessments',17,base.INK,True)
    for j,(_,_,name,year) in enumerate(WORKS):
        x=136+j*113;s.text(x,81,f'E{j+1:02} · {year}',16,base.MUTED,anchor='middle');s.text(x,102,name,16,base.INK,True,anchor='middle')
    labels={'PARTIAL_ANTECEDENT':'Part','BUILDING_BLOCK':'Base','PARALLEL':'Parallel'}
    rel={(r['claim'],r['work']):r['assessment'] for r in d['relations']}
    for i in range(6):
        y=113+i*28;s.text(16,y+19,f'C{i+1:02}',17,base.COLORS[i],True)
        for j in range(3):
            x=82+j*113;r=rel[(f'C{i+1:02}',f'E{j+1:02}')]
            s.rect(x,y,108,26,'#EEF4F8','#DFE6EC',2)
            txt=labels.get(r['relation_label'],r['relation_label'])+' / '+('F' if r['evidence_level']=='fulltext_evidence' else 'A') if r else '—'
            s.text(x+54,y+18,txt,16,base.INK,anchor='middle')
    p(s,14,302,'Part: partial antecedent · Base: building block\nA: abstract · F: full text · —: no saved assessment\n* E02 is a preprint; not an extra independent result.',418,15.5,18)
    s.line(14,352,431,352,base.BORDER)
    s.text(14,375,'What changes relative to RNA proteases?',18,base.INK,True)
    p(s,14,402,'Earlier work\nRNA-triggered cleavage\nand native death output.',197,17,21)
    p(s,235,402,'DAMAGE\nAn engineered gasdermin\noutput in mammalian cells.',197,17,21)
    s.line(220,390,220,457,base.BORDER)
    p(s,14,480,'Supported increment: coupling and output engineering; absolute firstness is not established.',416,17,20,True)
    return s


def positions() -> dict:
    pos={f'H{i+1:02}':(100+(i//4)*196,62+(i%4)*51) for i in range(12)}
    pos.update({f'C{i+1:02}':(24+(i%3)*260,12+(i//3)*265) for i in range(6)})
    return pos


def focus(d: dict, claim: str | None = None) -> Scene:
    sub=dict(d);sub['aliases']={n:a for n,a in d['aliases'].items() if n in SELECTED}
    keep=set(SELECTED)|{c['claim_id'] for c in d['claims']}
    sub['facts']={**d['facts'],'historical_edges':[e for e in d['facts']['historical_edges'] if set(e)<=keep], 'insertion_edges':[e for e in d['facts']['insertion_edges'] if set(e)<=keep]}
    sub['cards']={c:{**v,'neighbors':[n for n in v['neighbors'] if n['claim_id'] in SELECTED]} for c,v in d['cards'].items()}
    scene=base.graph_scene(sub,positions(),claim)
    if claim is None:
        short=['RNA circuit','RNA detection','Point mutation','RNA cleavage','GSDMD control','Photopyroptosis','Death switch','Pore inhibition','KRAS targeting','mRNA/LNP','Cell pyroptosis','Tumor response']
        for i,name in enumerate(short):
            x,y=positions()[f'H{i+1:02}']
            scene.rect(x-62,y+12,130,18,'#FCFDFF',opacity=.92)
            scene.text(x+3,y+26,name,14,base.MUTED,anchor='middle')
    return scene


def graph(d: dict) -> Scene:
    s=base.panel('c')
    for j,c in enumerate(['C01','C03','C06']):
        s.text(14+j*198,54,c+' · same selected nodes',15,base.COLORS[int(c[1:])-1],True)
        s.use(focus(d,c),12+j*198,60,.30)
    s.line(14,153,602,153,base.BORDER)
    s.text(14,175,'Joint view: all 6 claims + 12 of 47 historical nodes',18,base.INK,True)
    s.use(focus(d),10,182,.96)
    s.text(14,479,'Colored: semantic insertions · gray: historical edges',15.5,base.MUTED)
    s.line(15,495,43,495,'#405A70',1.8,'5 3');s.text(49,500,'Dashed: existing edges absent from each local view',15.5,base.MUTED)
    s.text(14,521,f'Full union supplied: 47 history nodes · 51 history edges · 60 insertions',15,base.INK)
    return s


def judgments(d: dict) -> Scene:
    s=base.panel('d');p=base.para
    p(s,14,55,'Evidence supports different levels of conclusion.\nThese are scoped judgments, not a novelty score.',418,16,19,color=base.MUTED)
    rows=[('SUPPORTED','A coupled functional system','C01 + C02 · M01 + M02 · E01 + E02','RNA recognition is coupled to an engineered gasdermin output in tested mammalian cells.',0),
          ('SCOPED','Selectivity in tested models','C03–C05 · manuscript and report','Sequence and transcript context matter. Target-dependent responses do not imply universal selectivity.',3),
          ('NOT ESTABLISHED','Therapeutic efficacy in vivo','C06 · delivery evidence and report','Cell-based delivery supports feasibility; organism-level efficacy, safety and tissue targeting remain open.',2)]
    for i,(status,title,links,body,ci) in enumerate(rows):
        y=97+i*113;color=base.COLORS[ci]
        s.rect(13,y,420,105,mix(color,'#FFFFFF',.96),mix(color,'#FFFFFF',.6),3)
        s.text(23,y+20,status,15,color,True,sans=True);s.text(23,y+43,title,18,base.INK,True)
        p(s,23,y+64,body,397,16.5,18)
    return s


def report(d: dict) -> Scene:
    s=base.panel('e');p=base.para
    s.text(14,55,'English summaries of complete saved report passages',16,base.MUTED)
    for i,row in enumerate(d['translations']):
        y=71+i*91;s.rect(13,y,591,83,'#F5F8FB','#E2EAF1',3)
        s.text(23,y+20,row['id']+'  '+row['title'],17,base.INK,True)
        p(s,23,y+42,row['english'],563,16.5,19)
    return s


def render(d: dict) -> None:
    main=Scene(1100,1375,'fig6_damage');main.rect(0,0,1100,1375,'#FFFFFF')
    main.text(14,29,'Fig. 6 | From an RNA-responsive system to a traceable report',27,base.INK,True)
    main.text(14,49,'DAMAGE · grounded claims, independent historical comparisons, joint context and scope',17,base.MUTED)
    issues=[]
    for key,panel in zip(base.PANELS,[manuscript(d),historical(d),graph(d),judgments(d),report(d)]):
        main.use(panel,*base.PANELS[key][:2]);base.save_scene(panel,OUT/f'panels/{key}.svg')
        issues += [{'panel':key,**b} for b in panel.text_boxes if b['x']<0 or b['x']+b['width']>panel.width or b['y']+b['height']>panel.height]
    base.save_scene(main,OUT/'final/Fig6.svg',True)
    base.write(OUT/'data/layout_checks.json',{'out_of_bounds':issues});assert not issues,issues
    base.fonts(OUT);base.convert(OUT/'final/Fig6.svg',scale=2)
    base.cairosvg.svg2png(url=str(OUT/'final/Fig6.svg'),write_to=str(OUT/'final/Fig6_600dpi.png'),output_width=4252,output_height=5315)
    base.outline(OUT/'final/Fig6.svg',OUT/'final/Fig6_outlined.svg')
    pos={a:(66+(i%7)*77,52+(i//7)*36) for i,a in enumerate(d['aliases'].values())}
    pos.update({f'C{i+1:02}':(40+i*103,316) for i in range(6)})
    allscene=base.graph_scene(d,pos);allscene.height=350
    base.save_scene(allscene,OUT/'supplement/full_union.svg');base.convert(OUT/'supplement/full_union.svg',scale=2)


def package(d: dict) -> None:
    lines=['# Figure 6 — DAMAGE: source companion','','An illustrative retrospective case, not an optimality claim or a controlled system-benefit experiment. DOI: 10.1038/s41467-026-69179-5.','',
    'Original branches are unchanged copies of the 20260907 study. The original report ends mid-sentence at 4000 characters. Two separate regeneration attempts also ended mid-sentence and were rejected. This figure uses only complete passages from the original report; its unfinished concluding paragraph is excluded. The original report and attempt status are preserved.','',
    'RFdiffusion2 was screened externally but not selected: its 2025 first-public date precedes the native graph cutoff (2025-12-31). A valid historical graph requires a rebuilt snapshot, not just neighbor filtering. Selection favored traceability and a readable contribution chain; it does not establish best scientific novelty or best system performance.','',
    'The graph focus contains 12 explicitly selected historical claims and all six targets. All edges induced among those nodes are retained. The supplement contains all 47 historical nodes, 51 historical edges and 60 insertion edges. Dashed edges already exist historically and are absent from all single-claim induced neighborhoods. Editorial layout is not a community or distance embedding. No target-to-target edges were invented.','',
    'The report summaries below are editorial English summaries of specified complete source passages. They are not model-generated English quotations. Reader takeaways are not measured reader outcomes. The graph is context, not proof of firstness or causal improvement in report quality.','']
    for i,c in enumerate(d['claims'],1):
        lines += [f'## C{i:02} / M{i:02}',c['normalized_claim_text']]
        for sid in c['support_span_ids']:lines += [sid,d['spans'][sid]['text']]
    lines+=['','## Historical works']
    for eid,w in d['works'].items():lines += [f"{eid}: {w['title']} | {w.get('doi')} | {w.get('publication_date')}"]
    lines+=['','## Historical node map']
    for nid,a in d['aliases'].items():lines += [f"- {a}: {nid} — {d['history'][nid]['claim_text']}"]
    lines+=['','## Report-summary mappings']
    for row in d['translations']:lines += [row['id'],row['source'],row['english'],row['links'],'']
    (OUT/'source_companion.md').write_text('\n'.join(lines))
    (OUT/'caption_en.md').write_text('''# Figure 6 | From an RNA-responsive system to a traceable report

Illustrative retrospective analysis of DAMAGE (He et al., Nature Communications, 2026; doi:10.1038/s41467-026-69179-5). (a) Six grounded contributions, shown as editorial summaries. (b) Saved GEAR assessments against three selected historical works; A denotes abstract evidence and the dash denotes no saved pairwise assessment. The 2022 preprint is identified explicitly. Labels are model judgments without passed independent verification. (c) Three local views and a joint view share coordinates and a selected set of 12 historical nodes; all six target claims are shown. The full union is provided separately. Colored edges are semantic insertions, gray edges are historical links, and dashed edges are existing historical edges not visible in any single-claim induced view. (d) Supported integration, model-dependent selectivity, and unestablished in vivo therapeutic scope. (e) English editorial summaries of source passages in complete passages of the saved report; its truncated conclusion is excluded. The figure illustrates evidence tracing and scoped interpretation; it does not measure system superiority, scientific firstness, or reader benefit. See the companion for source identities and report-generation provenance.
''')
    manifest=[]
    sources=list(CASE.rglob('*.json'))+list(CASE.rglob('*.jsonl'))+[ROOT/f'outputs/innovation_200_20260907/reports/fusion/{PID}.json',ROOT/'outputs/fig6_damage_study/provenance.json',ROOT/f'outputs/fig6_damage_study/original_report/{PID}.json']
    for src in sources:
        target=OUT/'sources'/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        manifest.append({'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
    base.write(OUT/'data/source_manifest.json',manifest);base.write(OUT/'data/report_summary_map.json',d['translations'])
    for src in [*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('report_rows.json')]:
        dest=OUT/'reproduce'/src.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(src,dest)
    with zipfile.ZipFile(OUT/'Fig6_reference_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.rglob('*'):
            if f.is_file() and f.suffix!='.zip':z.write(f,f.relative_to(OUT))


def main() -> None:
    d=prepare();d['translations']=base.read(Path(__file__).with_name('report_rows.json'))
    for row in d['translations']:assert row['source'] in d['report']['body'],row['id']
    render(d);package(d);print(OUT/'final/Fig6.png')


if __name__=='__main__':main()

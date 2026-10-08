"""Reproduce a source-bound TESB case figure without changing study results."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import cairosvg
import networkx as nx

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene, mix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig6_computer'
STUDY = ROOT / 'outputs/innovation_200_20260907'
PID = 's42005-026-02538-2'
BASE = STUDY / 'papers' / PID
INK, MUTED, BORDER = '#24282D', '#5F7182', '#A9CDE4'
COLORS = ['#1269DF', '#178F9D', '#713BCB', '#D47626', '#56739C', '#9D557D', '#61754A']
PANELS = {'a': (12, 57, 1076, 238), 'b': (12, 307, 446, 548),
          'c': (470, 307, 618, 548), 'd': (12, 867, 446, 472), 'e': (470, 867, 618, 472)}
TITLES = ['Manuscript and grounded contributions', 'Historical evidence and differences',
          'From claim neighborhoods to joint structure', 'Evidence-constrained judgments',
          'Traceable report and scientific takeaways']
LABELS = ['Two-stage solver', 'History penalty', 'Mini-batch update', 'Max-Cut results',
          'Large-scale task', 'Budget setting', 'Geometry limit']
TEXTS = ['Warm up with SB;\nthen run tabu checking.', 'Penalize regions near\nstored suboptimal states.',
         'Sample stored solutions\nto build each penalty.', 'Up to 1,000× lower TTS\non selected instances.',
         'Track reconstruction:\n109,498 spin variables.', '90% checking was best\nin the reported scan.',
         'Penalties may suppress\nnearby high-quality states.']
WORKS = [('W2054870037', 1, 'Tabu / Ising', '1995'),
         ('W2938847643', 2, 'Original SB', '2019'),
         ('W3016234529', 1, 'Hybrid QA', '2021')]
ROWS = [
    {'id': 'R01', 'title': 'A specific algorithmic combination', 'paragraph': 1,
     'source': '证据支持的是一个具体的两阶段算法结构及其数学实现',
     'english': 'The evidence supports a specific two-stage algorithmic structure and its mathematical implementation.',
     'links': 'C01 · M01', 'takeaway': 'What is added?\nA concrete two-stage\nalgorithm.'},
    {'id': 'R02', 'title': 'Separate foundations from the increment', 'paragraph': 3,
     'source': '本文相对于已识别研究的可确认增量是具体机制的组合与落地',
     'english': 'Relative to the identified studies, the supported increment is the combination and implementation of specific mechanisms.',
     'links': 'C01 + C02 · E01–E03', 'takeaway': 'What was known?\nSB and tabu search\nare earlier foundations.'},
    {'id': 'R03', 'title': 'Treat the claims as complementary roles', 'paragraph': 7,
     'source': 'SB基础—历史引导惩罚—随机小批量实现—基准性能—大规模应用—适用边界',
     'english': 'SB foundations → history-guided penalties → stochastic mini-batches → benchmarks → large-scale application → limits.',
     'links': 'C01–C07 · report synthesis', 'takeaway': 'Why one paper?\nMethod, tests and limits\nplay different roles.'},
    {'id': 'R04', 'title': 'Keep empirical gains within scope', 'paragraph': 5,
     'source': '不能推出对所有组合优化问题的普遍优越性。',
     'english': 'The results do not establish superiority for all combinatorial optimization problems.',
     'links': 'C04 + C05 · M04 + M05', 'takeaway': 'How far does it go?\nGains on tested tasks;\nno universal guarantee.'},
]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def trace(path: Path) -> dict[str, Any]:
    return {r['evidence_id']: r['payload'] for line in path.read_text().splitlines()
            if (r := json.loads(line))}


def screening() -> None:
    manifest = ROOT / 'data/nature_2026_testset/manifest.jsonl'
    roster = {p['paper_id']: p for line in (STUDY / 'papers.jsonl').read_text().splitlines()
              if (p := json.loads(line))}
    pattern = r'language model|machine learning|neural network|deep learning|computer vision|federated|reasoning|reinforcement learning|graph learning|graph dissimilarity|combinatorial optimization|online learning'
    rows = []
    for line in manifest.read_text().splitlines():
        p = json.loads(line)
        pid = p['article_id']
        field = roster.get(pid, {}).get('field_name', '')
        hit = bool(re.search(pattern, p['title'], re.I)) or field == 'Computer Science'
        rows.append({'paper_id': pid, 'title': p['title'], 'field_if_available': field,
                     'title_or_field_candidate': hit, 'in_downstream_roster': pid in roster,
                     'has_fusion_report': (STUDY / f'reports/fusion/{pid}.json').exists(),
                     'selected': pid == PID})
    with (OUT / 'data/corpus_screening.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(OUT / 'data/selection.json', {'manifest_papers': len(rows), 'downstream_roster': len(roster),
          'candidate_count': sum(r['title_or_field_candidate'] for r in rows),
          'method': 'Title/available-field screen of all 1,000 records, followed by detailed content review of three available cases and scope review of one quantum-algorithm alternative. Not a full scientific audit of all records; no outcome-score selection.',
          'shortlist': [
              {'paper_id': PID, 'decision': 'Selected: explicit optimization algorithm, distinct prior foundations, complete case records, interpretable method/implementation/test/limit roles.'},
              {'paper_id': 's42005-026-02523-9', 'decision': 'Alternative: graph dissimilarity; mathematical-property and benchmark qualifications require more space.'},
              {'paper_id': 's41467-026-68292-9', 'decision': 'Alternative: medical image compression; medically specific scope and broad historical comparators make the main story less direct.'},
              {'paper_id': 's42005-026-02528-4', 'decision': 'Not preferred: quantum variational-algorithm framing is further from the requested computer-science emphasis.'}],
          'scope': 'Illustrative retrospective case; no claim that graph structure or this case proves system superiority.'})


def prepare() -> dict[str, Any]:
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    screening()
    claims = read(BASE / 'shared/claims.json')['claims']
    ir = read(BASE / 'shared/paper_ir.json')
    spans = {s['span_id']: s for s in ir['spans']}
    gear = {f'C{i:02}': trace(BASE / f'gear/{i:02}/evidence_trace.jsonl') for i in range(1, 8)}
    cards, history = {}, {}
    for i in range(1, 8):
        key = f'C{i:02}'
        cards[key] = trace(BASE / f'graph/{i:02}/evidence_trace.jsonl')[f'GRAPH:{PID}::CLAIM::{i:02}']
        history.update({n['claim_id']: n for n in cards[key]['neighbors']})
    aliases = {n: f'H{i:02}' for i, n in enumerate(sorted(history), 1)}
    facts = read(BASE / 'graph/joint/facts.json')
    local = {tuple(sorted(e)) for c in cards.values() for e in c['neighbor_edges']}
    extra = [e for e in facts['historical_edges'] if tuple(sorted(e)) not in local]
    works, relations = {}, []
    for ei, (wid, ci, _, _) in enumerate(WORKS, 1):
        works[f'E{ei:02}'] = gear[f'C{ci:02}'][f'WORK:{PID}::CLAIM::{ci:02}:https://openalex.org/{wid}']
        for i in range(1, 8):
            key = f'RELATION:{PID}::CLAIM::{i:02}:https://openalex.org/{wid}'
            relations.append({'claim': f'C{i:02}', 'work': f'E{ei:02}', 'key': key,
                              'assessment': gear[f'C{i:02}'].get(key)})
    report = read(STUDY / f'reports/fusion/{PID}.json')
    for row in ROWS:
        assert row['source'] in report['body'], row['id']
    data = {'paper': read(BASE / 'innovation_input.json'), 'claims': claims, 'spans': spans,
            'cards': cards, 'history': history, 'aliases': aliases, 'facts': facts,
            'extra': extra, 'works': works, 'relations': relations, 'report': report,
            'translations': ROWS}
    write(OUT / 'data/snapshot.json', data)
    sources = list(BASE.glob('shared/*.json')) + list(BASE.glob('gear/**/*.json'))
    sources += list(BASE.glob('gear/**/evidence_trace.jsonl')) + list(BASE.glob('graph/**/*.json*'))
    sources += [BASE / 'innovation_input.json', BASE / 'graph/joint/report.md',
                STUDY / f'reports/fusion/{PID}.json', STUDY / f'reports/fusion/{PID}.md']
    sources += [ROOT / f'data/nature_2026_testset/{folder}/{PID}{suffix}.md'
                for folder, suffix in [('paper', ''), ('peer_review', '_r')]]
    manifest = []
    for src in sorted(set(sources)):
        dest = OUT / 'sources' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
        manifest.append({'original': str(src), 'copy': str(dest.relative_to(OUT)),
                         'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
    write(OUT / 'data/source_manifest.json', manifest)
    return data


def para(s: Scene, x: float, y: float, text: str, width: float, size: float = 17,
         leading: float = 20, bold: bool = False, color: str = INK) -> float:
    return s.paragraph(x, y, text, width, size, leading, bold, color=color)


def panel(key: str) -> Scene:
    _, _, w, h = PANELS[key]
    s = Scene(w, h, 'panel_' + key)
    s.rect(0, 0, w, h, '#FCFDFF', BORDER, 4, sw=1)
    s.rect(1, 1, w - 2, 32, '#EEF6FC')
    s.text(10, 24, key, 27, INK, True)
    s.text(36, 23, TITLES[ord(key)-97], 21, INK, True)
    return s


def manuscript(d: dict[str, Any]) -> Scene:
    s = panel('a')
    para(s, 14, 57, 'Tabu-Enhanced Simulated Bifurcation for combinatorial optimization', 257, 20, 22, True)
    s.text(14, 130, 'Tao et al. · Commun. Phys. (2026)', 16, MUTED)
    s.text(14, 151, 'Saved cutoff: 10 Feb 2026', 16, MUTED)
    para(s, 14, 174, 'Can search history help an optimizer escape local traps?', 255, 18, 21, True)
    s.text(14, 224, 'Published text · retrospective analysis', 15, MUTED)
    s.line(283, 45, 283, 225, BORDER)
    for i, claim in enumerate(d['claims']):
        x, y = 296 + (i % 4)*194, 43 + (i//4)*93
        color = COLORS[i]
        s.rect(x, y, 186, 85, mix(color, '#ffffff', .965), mix(color, '#ffffff', .60), 3)
        s.text(x+7, y+19, f'C{i+1:02}  {LABELS[i]}', 16.3, color, True)
        para(s, x+7, y+41, TEXTS[i], 174, 16.2, 18)
        p = d['spans'][claim['support_span_ids'][0]]['page']
        s.text(x+7, y+77, f'M{i+1:02} · {claim["claim_type"].title()} · p. {p}', 14.5, MUTED)
    para(s, 885, 158, 'Seven source-grounded claims; complementary roles, not seven independent inventions.', 174, 16, 19, True)
    return s


def historical(d: dict[str, Any]) -> Scene:
    s = panel('b')
    s.text(14, 55, 'Saved GEAR relations · selected independent works', 17, INK, True)
    for j, (_, _, name, year) in enumerate(WORKS):
        x = 136 + j*113
        s.text(x, 81, f'E{j+1:02} · {year}', 16, MUTED, anchor='middle')
        s.text(x, 102, name, 17, INK, True, anchor='middle')
    labels = {'PARTIAL_ANTECEDENT': 'Part', 'BUILDING_BLOCK': 'Base', 'PARALLEL': 'Parallel'}
    rel = {(r['claim'], r['work']): r['assessment'] for r in d['relations']}
    for i in range(7):
        y=110+i*27
        s.text(16, y+19, f'C{i+1:02}', 17, COLORS[i], True)
        for j in range(3):
            x=82+j*113; r=rel[(f'C{i+1:02}',f'E{j+1:02}')]
            s.rect(x,y,108,25,'#EEF4F8' if r else '#FAFBFC','#DFE6EC',2)
            txt=labels[r['relation_label']] + (' / F' if r['evidence_level']=='fulltext_evidence' else ' / A') if r else '?'
            s.text(x+54,y+18,txt,16,INK if r else '#97A1AC',anchor='middle')
    para(s,14,320,'Part: partial antecedent · Base: building block\nA: abstract · F: full text · ?: no saved assessment',419,15.5,18)
    s.line(14,351,431,351,BORDER)
    s.text(14,373,'A closer comparison: C02 versus E02',18,INK,True)
    para(s,14,397,'Earlier SB: nonlinear Hamiltonian dynamics seek low-energy solutions.',198,17,20)
    para(s,232,397,'TESB: stored solutions generate a history-guided penalty.',198,17,20)
    s.line(220,385,220,450,BORDER)
    para(s,14,475,'Residual difference: memory-based landscape modification within the existing SB route.',416,17,20,True)
    para(s,14,516,'Saved model labels; independent verification not passed.\nNo claim of exhaustive prior-art coverage.',417,15,17,color=MUTED)
    return s


def graph_positions(d: dict[str, Any]) -> dict[str, tuple[float, float]]:
    # Editorial grid keeps every historical label readable; not a metric embedding.
    pos = {alias: (84 + (i % 5)*100, 50 + (i//5)*42)
           for i, alias in enumerate(d['aliases'].values())}
    pos.update({'C01': (110, 9), 'C02': (350, 9), 'C03': (557, 257),
                'C04': (557, 100), 'C05': (557, 195),
                'C06': (18, 117), 'C07': (18, 221)})
    return pos


def graph_scene(d: dict[str, Any], pos: dict[str, tuple[float,float]], claim: str | None = None) -> Scene:
    s=Scene(596,304,'joint_graph' if claim is None else 'local_'+claim)
    ids={c['claim_id']:f'C{i:02}' for i,c in enumerate(d['claims'],1)} | d['aliases']
    keep=set(ids.values()) if claim is None else {claim}|{d['aliases'][n['claim_id']] for n in d['cards'][claim]['neighbors']}
    extra={tuple(sorted(e)) for e in d['extra']}
    for a,b in d['facts']['historical_edges']:
        u,v=ids[a],ids[b]
        if u in keep and v in keep:
            ex=tuple(sorted((a,b))) in extra
            s.line(*pos[u],*pos[v],'#405A70' if ex else '#A3ADB7',2.2 if ex else .85,'5 3' if ex else None)
    for a,b in d['facts']['insertion_edges']:
        u,v=ids[a],ids[b]
        if u in keep and v in keep:s.line(*pos[u],*pos[v],COLORS[int(u[1:])-1],1,opacity=.48)
    for n in sorted(keep):
        x,y=pos[n]; target=n.startswith('C'); color=COLORS[int(n[1:])-1] if target else '#E4EBF1'
        s.node(x,y,fill=color,radius=13 if target else 7,stroke='#FFFFFF' if target else '#8C9BAB',sw=1.2)
        if target:s.text(x,y+4,n,13,'#FFFFFF',True,anchor='middle',sans=True)
        else:s.text(x+9,y+4,n,15,MUTED)
    return s


def graph(d: dict[str, Any]) -> Scene:
    s=panel('c'); pos=graph_positions(d)
    for j,ci in enumerate(['C01','C04','C05']):
        s.text(16+j*198,54,ci+' local neighborhood',16,COLORS[int(ci[1:])-1],True)
        s.use(graph_scene(d,pos,ci),10+j*198,60,.30)
    s.line(14,154,602,154,BORDER)
    s.text(14,176,'Complete union: 7 targets + 25 historical nodes',18,INK,True)
    s.use(graph_scene(d,pos),10,179,.96)
    s.text(395,449,'C03: no eligible neighbors',15,COLORS[2])
    s.line(15,465,42,465,'#405A70',1.8,'5 3')
    s.text(48,470,f'{len(d["extra"])} existing historical edge visible only jointly',15.5,MUTED)
    s.line(15,486,42,486,COLORS[0],1.3)
    s.text(48,491,f'{len(d["facts"]["insertion_edges"])} semantic insertions; gray = {len(d["facts"]["historical_edges"])} historical edges',15.5,MUTED)
    shared = d['aliases']['s41467-024-53270-w::C01']
    s.text(14,514,f'C01/C04/C05 share {shared}: p-bit Ising architecture (context only).',15.5,INK)
    s.text(14,536,'Layout is not distance; connections do not establish priority or causality.',15,MUTED)
    write(OUT/'data/graph_layout.json',pos)
    return s


def judgments(d: dict[str, Any]) -> Scene:
    s=panel('d')
    para(s,14,55,'Statement-level treatments in the saved report.\nMethod, performance and generality are assessed separately.',418,16,19,color=MUTED)
    blocks=[('SUPPORTED','Concrete method increment','C01–C03 · M01–M03 · E01 + E02',
             'Two-stage SB with history-based penalties and stochastic mini-batch construction.',COLORS[0]),
            ('SCOPED','Reported performance advantage','C04–C06 · report paragraphs 5–6',
             'Gains depend on the instances, GPU, budget and baselines. The 90% setting is empirical.',COLORS[3]),
            ('WITHHELD','Universal advantage / absolute priority','C07 · M07 · report paragraphs 3, 6–7',
             'Dense near-optimal clusters may be penalized. Global superiority and firstness are not established.',COLORS[2])]
    for i,(status,title,links,body,color) in enumerate(blocks):
        y=96+i*111
        s.rect(13,y,420,102,mix(color,'#FFFFFF',.968),mix(color,'#FFFFFF',.65),3)
        s.text(23,y+21,status,15,color,True,sans=True)
        s.text(23,y+43,title,18,INK,True)
        para(s,23,y+64,body,398,16.5,18)
        s.text(23,y+96,links,14.5,MUTED)
    para(s,14,452,'Source scope is retained; performance is not novelty.',416,16,19,True)
    return s


def report(d: dict[str, Any]) -> Scene:
    s=panel('e')
    s.text(14,55,'Faithful English excerpts from the saved Chinese report',16,MUTED)
    for i,row in enumerate(ROWS):
        y=69+i*91
        s.rect(13,y,591,85,'#F5F8FB','#E2EAF1',3)
        s.text(23,y+19,row['id']+'  '+row['title'],17,INK,True)
        para(s,23,y+36,row['english'],377,16,16)
        s.text(23,y+84,row['links'],14,MUTED)
        s.line(408,y+27,408,y+77,BORDER)
        para(s,419,y+37,row['takeaway'],174,15.7,17)
    para(s,14,450,'Specific hybrid mechanism; empirical gains with explicit limits.',589,18,20,True)
    return s


def save_scene(scene: Scene, path: Path, physical: bool = False) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    root=scene.document()
    if physical:root.set('width','180mm');root.set('height','225mm')
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def render(d: dict[str, Any]) -> None:
    main=Scene(1100,1375,'fig6_computer')
    main.rect(0,0,1100,1375,'#FFFFFF')
    main.text(14,29,'Fig. 6 | From an optimization paper to a traceable report',28,INK,True)
    main.text(14,49,'Tabu-enhanced simulated bifurcation · algorithm, evidence, joint context and scope',17,MUTED)
    panels=[manuscript(d),historical(d),graph(d),judgments(d),report(d)]
    issues=[]
    for key,p in zip(PANELS,panels):
        main.use(p,*PANELS[key][:2]);save_scene(p,OUT/f'panels/{key}.svg')
        for box in p.text_boxes:
            if box['x']<0 or box['x']+box['width']>p.width or box['y']+box['height']>p.height:
                issues.append({'panel':key,**box})
    main.text(14,1363,'M: manuscript anchors · E: historical works · C: shared claims · R: report excerpts. Raw evidence accompanies the figure.',15,MUTED)
    save_scene(main,OUT/'final/Fig6.svg',True)
    write(OUT/'data/layout_checks.json',{'out_of_bounds':issues,'text_boxes':main.text_boxes})
    assert not issues,issues
    fonts(OUT)
    for path in list((OUT/'panels').glob('*.svg'))+[OUT/'final/Fig6.svg']:convert(path,scale=2)
    cairosvg.svg2png(url=str(OUT/'final/Fig6.svg'),write_to=str(OUT/'final/Fig6_600dpi.png'),output_width=4252,output_height=5315)
    outline(OUT/'final/Fig6.svg',OUT/'final/Fig6_outlined.svg')


def package(d: dict[str, Any]) -> None:
    lines=['# Figure 6 — Computer science case', '',
           'Tao et al., Tabu-Enhanced Simulated Bifurcation for combinatorial optimization. Communications Physics 9, 100 (2026). DOI: 10.1038/s42005-026-02538-2.', '',
           '## Selection and temporal scope','',
           'All 1,000 local manifest titles were screened with title terms and available study fields. The complete downstream roster contains 200 papers, not 1,000. See data/corpus_screening.csv and data/selection.json. Selection is qualitative and illustrative; it is not based on a system performance score.', '',
           'The saved run uses cutoff 10 February 2026. The publisher distinguishes online publication (10 February) from the version of record (19 March 2026). The saved full text is a published article; this figure is retrospective, not a reconstruction of submission-time knowledge. No exhaustive earliest-preprint determination is claimed. The three selected historical works date to 1995, 2019 and 2021 and are distinct from the target paper.', '',
           '## Provenance and interpretations','',
           'Original saved study files are copied unchanged with export-time SHA-256 hashes. No new branch/model run or altered historical graph is presented. Saved relation labels are model outputs; independent_verification_passed is false. Author-paper versions and unrelated retrieval hits are not shown as independent foundations. Full retrieval traces remain available for inspection.', '',
           'The graph contains ALL 25 historical nodes, 28 historical edges, all seven targets and every saved insertion edge. All local views use the same coordinates. C03 has no eligible neighbor; absence is not novelty. Dashed lines are existing historical edges absent from every single-claim induced edge set. They were not created by the target paper. Target-to-target scientific edges are never invented.', '',
           'The joint analysis groups C01–C03 as method/mechanism/implementation, C04–C05 as benchmark/application, and C06–C07 as parameter/limit. This is a source-bound synthesis, not proof that graph connectivity caused the report or improved reader understanding. R03 has no added graph citation. No controlled benefit claim is made.', '',
           '## Manuscript anchors','']
    for i,c in enumerate(d['claims'],1):
        lines += [f'### C{i:02} / M{i:02}',c['normalized_claim_text'],'']
        for sid in c['support_span_ids']:
            span=d['spans'][sid];lines += [f"Page {span['page']} · {sid}",span['text'],'']
    lines+=['## Independent historical works','']
    for key,w in d['works'].items():lines += [f"### {key} · {w['title']}",f"DOI: {w['doi']} | Date: {w['publication_date']}",w['abstract'],'']
    lines+=['## Historical node identity map','']
    for nid,alias in d['aliases'].items():lines += [f"- {alias} · {nid}: {d['history'][nid]['claim_text']}"]
    lines+=['','## Report translation audit','']
    for row in ROWS:lines += [f"### {row['id']} · paragraph {row['paragraph']}",row['source'],row['english'],f"Display links: {row['links']}",'']
    (OUT/'source_companion.md').write_text('\n'.join(lines),encoding='utf-8')
    caption='''# Figure 6 | From an optimization paper to a traceable report

An illustrative retrospective case of Tabu-Enhanced Simulated Bifurcation (TESB; Tao et al., Communications Physics, 2026; doi:10.1038/s42005-026-02538-2).
(a) Seven grounded claims describe the algorithm, proposed mechanism, mini-batch implementation, benchmark and application results, parameter setting and applicability limit. TTS denotes time-to-solution. Summaries are not author quotations.
(b) Saved GEAR relation assessments against three selected independent historical works. A/F indicates abstract/full-text evidence; ? indicates no saved assessment for that pair. These model labels have not passed independent verification. The comparison distinguishes original SB dynamics from TESB's use of stored search history.
(c) Three local neighborhoods and the complete union of all seven target claims and 25 historical claims, with the same layout. Colored edges are semantic insertions, gray edges are existing historical edges, and dashed edges are existing edges visible only in the union. The isolated C03 explicitly preserves missing graph evidence. Layout, similarity and connectivity do not establish scientific priority, derivation or causality.
(d) Statement-specific treatments in the saved report: supported method construction, scoped empirical performance, and unresolved generality/priority. These are not fabricated intermediate revisions.
(e) Faithful English translations of exact saved Chinese report substrings, with shortened evidence references and editorial reader takeaways. R03 is a report synthesis, not a graph-caused conclusion. A single illustrative case does not establish system superiority or measured gains in reader understanding.

All displayed network nodes and edges are retained in the source package; source_companion.md and data/snapshot.json contain exact identities, manuscript spans and translation mappings. The displayed cutoff is the saved run cutoff; this is not a historical submission-time assessment.
'''
    (OUT/'caption_en.md').write_text(caption)
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Figure 6 — TESB</title><style>body{max-width:1100px;margin:30px auto;font:18px Georgia;color:#24282d}img{width:100%}a{color:#1269df}</style><h1>Figure 6 — Computer science case</h1><p>Tabu-Enhanced Simulated Bifurcation for combinatorial optimization</p><p><a href="final/Fig6.pdf">Vector PDF</a> · <a href="final/Fig6.svg">Editable SVG</a> · <a href="final/Fig6_600dpi.png">600 dpi PNG</a> · <a href="source_companion.md">Evidence map</a></p><img src="final/Fig6.png" alt="Five-panel TESB case study">')
    with zipfile.ZipFile(OUT/'Fig6_computer_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))


def main() -> None:
    d=prepare();render(d);package(d)
    print(OUT/'final/Fig6.png')


if __name__=='__main__':
    main()

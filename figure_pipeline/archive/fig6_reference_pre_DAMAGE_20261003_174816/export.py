"""Publication exports and an English, source-linked handoff."""
from __future__ import annotations

import html
import zipfile
from pathlib import Path

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from .data import OUT, ROOT, read
from .editorial import REPORT_ROWS
from .render import PANELS, TITLES


def export() -> None:
    import cairosvg

    fonts(OUT)
    for folder in ['final', 'panels', 'components']:
        for path in sorted((OUT / folder).glob('*.svg')):
            if path.stem.endswith('_outlined'):
                continue
            convert(path, scale=2 if folder == 'final' else 2.5)
    path = OUT / 'final/Fig6.svg'
    for dpi in (600, 900):
        cairosvg.svg2png(url=str(path), write_to=str(OUT / f'final/Fig6_{dpi}dpi.png'),
                        output_width=round(180 / 25.4 * dpi), output_height=round(225 / 25.4 * dpi))
    outline(path, OUT / 'final/Fig6_outlined.svg')


def companion() -> None:
    data = read(OUT / 'data/snapshot.json')
    text = ['# Figure 6: sources and evidence map', '',
            f"**Case:** {data['paper']['title']}. Nature Communications (2026). DOI: {data['paper']['doi']}.", '',
            '## Case selection', '',
            f"Screened {data['selection']['screened']} saved study cases; {data['selection']['complete']} have the required claim, GEAR, joint-graph and fusion-report artifacts.",
            data['selection']['basis'], '',
            'Selection focused on physics, biochemistry and computation. '
            'The RNA-prediction candidate was not used because its report treated a closely matching preprint as prior art. '
            'Af-CUT&Tag was also inspected; the selected cryo-EM case provides a clearer physics-to-structural-biology narrative '
            'and multiple reconstructible cross-neighborhood edges. This is an illustrative case, not a representative estimate or performance ranking.', '',
            '## Provenance and limits', '',
            'All scientific artifacts come from the selected paper in outputs/innovation_200_20260907. '
            'No retrieval, model calls or new scientific evaluations were run. The study contains resumed and recovered stages; '
            'this package is not described as an uninterrupted or uniformly full-text run. The exported SHA-256 manifest '
            'records file contents at packaging time, not a pre-existing frozen release.', '',
            'A/F in panel b identifies the actual relation evidence level. The selected relation records have '
            'independent_verification_passed=false. Their labels are saved model assessments, not independently validated scientific truth. '
            'An unassessed matrix cell means no saved relation for that pair, not no historical relationship.', '',
            'The main SVG, caption, gallery and explanatory material are English. Raw saved artifacts and original report '
            'sentences retain their original language to preserve source fidelity.', '',
            '## Manuscript anchors', '']
    for key, source in data['manuscript'].items():
        text += [f"### {key} · primary anchor page {source['page']} · {source['span_id']}", '', 'Grouped supporting spans: ' + ', '.join(source['supporting_span_ids']), '', source['text'], '']
    text += ['## Historical works in panel b', '']
    for key, work in data['works'].items():
        text += [f"### {key} · {work['title']} ({work['publication_year']})", '',
                 f"DOI: {work['doi']}", '', work['abstract'], '']
    text += ['## Displayed historical claim nodes', '']
    for key, node in data['historical_nodes'].items():
        text += [f"- **{key}** — `{node['claim_id']}` ({node['publication_date']}): {node['claim_text']}"]
    text += ['', '## Graph facts and drawing rules', '',
             f"The full union contains {data['joint']['historical_neighbor_count']} historical claim nodes, "
             f"{len(data['joint']['historical_edges'])} existing historical edges and {len(data['joint']['insertion_edges'])} insertion edges. "
             f"{len(data['joint_only_edges'])} historical edges are absent from the union of single-claim induced edge sets.", '',
             'The main figure displays all six targets and ten explicitly selected historical nodes. '
             'Every insertion and historical edge among those displayed nodes is retained. '
             'The three local views use the same coordinates and show only the corresponding claim and its retained displayed neighbors. '
             'Node positions are editorial layout, not scientific distance or importance. The complete network accompanies the figure as JSON.', '',
             '**G01:** C01 and C04 both connect to H01 (foam-film sample preparation) and H02 (interface localization/preferred orientation). '
             '**G02:** the stored H05–H06 historical edge connects the C05 ribosome-assembly neighbor to the C04 in situ-ribosome-imaging neighbor; '
             'neither endpoint pair appears together in any single-claim induced edge set. '
             '**G03:** C05 and C06 share H08, a non-canonical L1-stalk result in a different ribosome system. '
             'Sharing that neighbor does not establish the helix-68 hypothesis.', '',
             'Colored lines are semantic insertion edges; gray lines are existing historical graph edges; '
             'dashed dark lines are existing historical edges visible only in the joint view. '
             'No target-to-target edges are introduced. Parent-paper citation paths are not drawn. '
             'The full insertion policy is at most ten eligible neighbors per target, cosine strictly greater than 0.5.', '',
             '## Judgment records', '',
             '**Supported, C04:** manuscript performance plus saved GEAR assessment; scoped to tested specimens. '
             '**Narrowed, C02:** the shared extraction candidate used “sample damage”; the saved grounding record narrows this to liquid-film breakup. '
             'The candidate is system-extracted, not a verbatim author sentence. '
             '**Withheld, C06:** the final report explicitly states that helix 68 causality remains unresolved. '
             'These are statement-specific treatments, not three whole-claim quality grades.', '',
             '## Report translation audit', '',
             'Each quoted English excerpt below translates the indicated exact substring of the saved Chinese fusion report. '
             'The reader takeaway is an editorial paraphrase. Evidence IDs were shortened for this figure; '
             'the G02 link attached to R03 is a figure-added cross-reference, not an original report citation. '
             'It supplies structural context, not support for the biological temperature-response result.', '']
    for row in REPORT_ROWS:
        text += [f"### {row['id']} · report paragraph {row['paragraph']}", '', row['english'], '',
                 f"Original: {row['source']}", '', f"Figure links: {row['links']}", '']
    (OUT / 'source_companion.md').write_text('\n'.join(text), encoding='utf-8')


def package() -> None:
    companion()
    caption = '''# Figure 6 | From a scientific paper to an evidence-constrained report

An illustrative case from Curtis et al., *Ultrathin liquid cells for microsecond time-resolved cryo-EM* (Nature Communications, 2026; doi:10.1038/s41467-026-68515-z).
**a**, Six shared contributions with manuscript anchors, covering sample sealing, pulse timing, interfaces, image quality and L1-stalk dynamics. The specimen cross-section is schematic, not an experimental image.
**b**, Saved GEAR relation assessments for selected contributions and four independent prior works. A and F indicate abstract and full-text evidence, respectively; question marks mark pairs without saved assessments. The source comparison distinguishes an established laser-melting route from the new sealed-cell, short-pulse implementation. Relation labels are not independent verification.
**c**, Three local views and a selected joint subgraph with fixed node positions. All six targets and ten of fifty historical neighbors are shown, retaining all recorded edges among displayed nodes. G01 marks shared sample/interface context; G02 identifies an existing historical edge between ribosome-assembly and in situ-imaging neighbors that is only exposed in the joint view; G03 identifies a shared L1-stalk neighbor. These facts do not establish causality, claim derivation or priority. Full node identities and the complete network are supplied.
**d**, Evidence-supported, narrowed and withheld statements drawn from the saved grounding, GEAR and report artifacts. The narrowed wording belongs to a system-extracted candidate, not an author quotation.
**e**, Faithful English translations of exact excerpts from the saved fusion report, with editorial reader takeaways. Short evidence IDs map to the accompanying source files; the G02 cross-reference is added for this figure. This case illustrates traceability and scope control, not population-level system effectiveness.
'''
    (OUT / 'caption_en.md').write_text(caption, encoding='utf-8')
    (OUT / 'README.md').write_text('''# Fig.6 reference

English, five-panel end-to-end case study: ultrathin liquid cells for microsecond time-resolved cryo-EM.

- `final/Fig6.svg`: editable text and vector objects, 180 × 225 mm.
- `final/Fig6_outlined.svg`: portable font-outlined copy.
- `final/Fig6.pdf`: vector PDF; `Fig6.png`: preview; 600/900-dpi PNG exports.
- `panels/` and `components/`: separate editable SVG, PDF and PNG assets.
- `data/`: screening table, source snapshot, identity map, translation map and source manifest.
- `sources/`: unchanged saved scientific records (original language preserved).
- `source_companion.md`: manuscript anchors, prior works, graph identities, selection rationale and caveats.
- `caption_en.md`: English caption. `index.html`: local gallery.

Reproduce from the repository root:

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig6_reference all
```

Stages: `prepare`, `render`, `package`; `all` runs them in order. No model or external retrieval calls are needed.
To redraw from the extracted delivery alone, use `render`: it reads the bundled snapshot. `prepare` requires the original study.
Dependencies: NumPy, SciPy, pandas, NetworkX, Pillow, Matplotlib and CairoSVG; Times New Roman and Arial are read from Windows fonts and are not redistributed.
The pipeline reuses the Fig.1 reference SVG scene graph, font setup and exporters. It does not change the source study or other figures.
Body text is approximately 8 pt at the stated physical size; evidence IDs and compact graph labels are smaller.
''', encoding='utf-8')
    gallery()
    files = [p for p in OUT.rglob('*') if p.is_file() and p.suffix != '.zip']
    with zipfile.ZipFile(OUT / 'Fig6_reference_delivery.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, 'outputs/fig6_reference/' + str(path.relative_to(OUT)))
        for path in sorted((ROOT / 'figure_pipeline/fig6_reference').glob('*.py')):
            archive.write(path, str(path.relative_to(ROOT)))
        for path in [ROOT / 'figure_pipeline/__init__.py'] + list((ROOT / 'figure_pipeline/fig1_reference').glob('*.py')):
            archive.write(path, str(path.relative_to(ROOT)))


def gallery() -> None:
    cards = ''.join(f'<article><h2>{k} · {html.escape(TITLES[k])}</h2><a href="panels/{k}.svg"><img src="panels/{k}.png" alt="Panel {k}"></a><p><a href="panels/{k}.pdf">PDF</a> · <a href="panels/{k}.svg">Editable SVG</a></p></article>' for k in PANELS)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure 6 · Cryo-EM case study</title><style>body{{max-width:1180px;margin:36px auto;padding:0 24px;color:#24282d;font:17px Arial,sans-serif;line-height:1.5}}h1,h2{{font-family:Georgia,serif}}a{{color:#1269df}}img{{width:100%;height:auto}}.hero{{max-width:1000px;display:block;margin:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Figure 6 · From a scientific paper to an evidence-constrained report</h1><p>Ultrathin liquid cells for microsecond time-resolved cryo-EM · Curtis et al. (2026)</p>
<p><a href="final/Fig6.pdf">Vector PDF</a> · <a href="final/Fig6.svg">Editable SVG</a> · <a href="final/Fig6_900dpi.png">900-dpi PNG</a> · <a href="caption_en.md">Caption</a> · <a href="source_companion.md">Sources</a></p>
<img class="hero" src="final/Fig6.png" alt="Five panels tracing manuscript contributions through historical evidence and a joint graph to a bounded report"><div class="grid">{cards}</div></html>'''
    (OUT / 'index.html').write_text(document, encoding='utf-8')

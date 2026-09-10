"""Assemble explicitly labelled draft handoff and publication evidence inventory."""
from __future__ import annotations
import html
import json
from pathlib import Path
from pypdf import PdfReader, PdfWriter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/FROM_WEB_v2'


def assemble() -> None:
    panels={
        '4a':'说明 Graph + joint 与 Direct LLM 的保存 AI 匹配表现，按指标比较两种颜色的论文均值和95%区间，并留意各项有效论文数与好坏方向。',
        '4b':'说明同一论文的两方法差异，点代表论文、菱形代表配对均值及区间，前三项正值较好而矛盾率负值较好。',
        '4c':'说明同一参考ID在两方法之间的范围匹配转移，按行读Direct、按列读Graph + joint的计数，失配不直接等于经独立核验的错误。',
        'S1a':'说明各期刊保存卡片的完成结构，按颜色读全部、部分或无卡论文数，无卡不自动等于未运行或没有科学证据。',
        'S1b':'说明执行覆盖与可用比较采用不同分母，横轴读保存卡数/请求数、纵轴读可用比较数/已保存卡数，点面积和数字为重合论文数，缺卡论文另列unknown。',
    }
    graph=(OUT/'data/graph_panel_captions_zh.md').read_text()
    case=(OUT/'fig02_fig08_fig09_fig10_notes_zh.md').read_text()
    text='# 新版逐 panel 说明：要说明什么、怎么看\n\n'+graph+'\n'+case+'\n## Fig.4 与 Supplement S1\n\n'
    text+='\n'.join(f'- **{k}**：{v}' for k,v in panels.items())
    text+='\n\n## 尚未绘成科学结果的目标 panels\n\n'
    questions=json.loads((ROOT/'figure_pipeline/specs/panels.json').read_text())
    for spec in questions:
        if spec['figure'] in [5,6,7]:
            text+=f"- **Fig.{spec['figure']}{spec['panel']}（未测）**：计划回答“{spec['question']}”，目前缺所需实验或独立核验，不能从S1进度推断其结果。\n"
    text+='\nFig3c当前为零模型诊断，并未完成原计划的独立解释核验；Fig8当前仍为标题短语基线，尚未完成claim方向和未来采纳验证。\n'
    (OUT/'PANEL_GUIDE_ZH.md').write_text(text)
    files=sorted(OUT.glob('fig*.png'))+[OUT/'supplement_S1_readiness.png']
    writer=PdfWriter()
    for path in files:
        for page in PdfReader(path.with_suffix('.pdf')).pages: writer.add_page(page)
    with (OUT/'REVISED_DRAFT_FIGURES.pdf').open('wb') as handle: writer.write(handle)
    cards=''.join(f'<article><h2>{html.escape(p.stem)}</h2><a href="{p.name}"><img src="{p.name}" alt="{html.escape(p.stem)}"></a><p><a href="{p.with_suffix(".svg").name}">SVG</a> · <a href="{p.with_suffix(".pdf").name}">PDF</a></p></article>' for p in files)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ASPR 修订草稿</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;color:#430258;background:#fafafa}main{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}article{background:white;padding:18px;border:1px solid #ddd}img{width:100%;height:500px;object-fit:contain}h2{font-size:17px}p{line-height:1.6}</style><h1>ASPR：数据审计与重绘草稿</h1><p>冻结原200篇。七幅可由当前快照支持的重绘与S1；Fig5/6/7效果图未生成。Fig3为描述诊断，Fig4为旧版未盲化AI匹配，Fig8为标题基线。<a href="PANEL_GUIDE_ZH.md">逐panel说明</a> · <a href="REVISED_DRAFT_FIGURES.pdf">合并PDF</a></p><main>'+cards+'</main>')
    artifacts={name:True for name in ['clean_atlas','frozen_local_cases','mechanism_source','terminal_source_trace','structural_diagnostics','paired_benchmark','execution_evidence_separation','deduplicated_phrase_baselines','field_provenance','paper_normalized_roles','traceable_case','main_reviewer_reference']}
    (OUT/'audit/artifact_attestations.json').write_text(json.dumps(artifacts,indent=2))
    from figure_pipeline.specs.contract import REQUIREMENTS, publication_missing
    (OUT/'audit/publication_status.json').write_text(json.dumps({str(f):{'required_artifacts':REQUIREMENTS[f],'missing':publication_missing(f,artifacts)} for f in range(1,11)},indent=2))


if __name__=='__main__': assemble()

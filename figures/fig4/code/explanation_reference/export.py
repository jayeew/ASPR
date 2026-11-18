"""Source companion, caption, local file gallery and figure delivery archive."""
from __future__ import annotations

import html
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .data import OUT, ROOT, SILK, THZ, POLICY, read
from .render import BOXES, PANELS, TITLES

CAPTION = """# Fig. 4 | Complementary branch contributions and graph component effects

Seven configurations complete matched public scientific tasks using the same report writer. Paper only receives the manuscript and public tasks; GEAR only and Graph only receive their own branch materials and interpretations; full system receives both. Three ablations remove the joint graph, structural values, or citation-path information. Missing branches receive no substitute analysis and legitimate reconstruction from remaining material is allowed. These are controlled information comparisons using existing factual assets, not a rerun of retrieval or graph construction.

a, Correct scientific-part coverage for seven configurations and five task aspects. Each paper uses its fixed applicable parts and papers receive equal weight. Common seven-condition complete sets contain 98 papers for the first four aspects and 74 for citation contact. All 100 candidates passed scientific screening; 694/700 reports are complete. Technical gaps remain missing, not zero.

b, Paired full-system minus single-branch or ablated-condition differences. Faint dots are individual paper differences, diamonds are means, and intervals are 95% CIs from 10,000 paper-bootstrap draws; the horizontal unit is percentage points. The paired branch sample is 99 for the first four aspects and 74 for citation contact; component comparisons use 98, 98 and 74 papers. Hence branch differences need not equal subtraction of the common-set means in a. Small offsets separate discrete observations; no density model is fitted. Different task requirements preclude ranking general component importance by effect size alone.

c, Descriptive counts of scientific parts retained, gained, lost or not correct in either report, comparing each single branch with the full system on complete pairs. Each row is normalized to its own N, which is a part count, not a paper count. Loss of correct coverage may reflect omission, error or missing scope. These counts are not the paper-weighted endpoint.

d, Additional historical edges visible in the union neighborhood but absent from the union of single-card internal edges. Papers are sorted by joint-only edge count and then paper ID; 84/100 have extra edges and 34/100 have fewer post-insertion components when those edges are present. Coverage tiles beneath the dot plot use exactly the same paper positions; 0–100% is shown by the sequential scale and hatching denotes a technical gap. Full-system versus joint-deletion means and the effect use 98 complete pairs. Extra graph edges do not imply scientific causality or automatically higher report coverage.

e, Quantitative-structure coverage for full system and structural-value deletion, with the paired effect and stored 95% CI (n=98). This task requires specific distribution-, distance- and historical-combination comparisons with scientific interpretation and scope. Low coverage after deletion does not imply inability to reconstruct local density or any structural quantity: some required centroid-related or historical-frequency information cannot be recovered from the retained nodes and edges.

f, Of 100 candidates, 74 have positive citation-contact records supporting the prespecified task; 26 are not applicable, not globally citation-free. Full-system and path-deletion coverage tiles show those 74 papers in ID order using the same sequential coverage scale as a/d. The summary gives absolute means and the paired effect with its 95% CI. Shared references, direct citation, two-hop paths and semantic similarity are distinct records; none by itself establishes scientific support or causal inheritance.

This compact layout removes all case illustrations and report excerpts from panels c–f, the structural case table, and grey bottom annotations. Layout is two rows: a/b above, c/d and vertically stacked e/f below. Sample definitions, uncertainty and source scope remain documented here. Original case evidence remains supplementary material rather than a displayed panel.

Scientific labels are model evaluations. Branch/report generation used gpt-6.1-sol/medium; references/evaluation used gpt-6.1-sol/high. Five development papers are included, with existing summaries excluding them supplied. Scientific uncertainty is not confirmed coverage, not-applicable parts leave the denominator, and technical gaps are not zero. All values and evaluations are reused; this layout revision introduces no model calls or new scoring.
"""

README = """# 新版 Fig4：两行精简布局

按最新要求更新排版，数据和评分不变。

- 第一行：a、b保持原布局，仅删除底部灰色说明。
- 第二行：从左到右为c、d，最右侧e在上、f在下。
- c只保留要点状态统计；d只保留额外历史边和逐论文覆盖率。
- e只保留完整系统与屏蔽结构数值的覆盖率比较；删除全部丝素案例、左下表格和报告摘要。
- f只保留适用性横条、汇总数值及两行色块；色块下方内容全部删除。
- 全图底部灰色说明删除。必要的坐标刻度、样本数和数据图例保留。

## 交付文件

最终归档目录：`outputs/fig4_reference/`。`experiment/`集中保存本轮百篇正式报告、原文/图输入、参考、逐要点评价、聚合表、原始响应与调用账本；`source/`保存代码快照。旧Fig4实验版本及重复数据包已删除，见`CLEANUP.md`。`Fig4_final_delivery.zip`为轻量图件包，不重复打包experiment原始材料。

画布324 × 181 mm。`final/`为整图可编辑SVG、转路径SVG、矢量PDF、300/600/900 dpi PNG；`panels/`为6个独立panel；`components/`为11个独立子图。`data/`含不变的本轮源表；`caption_en.md`说明指标、分母、区间和缺失处理。

`PANEL_GUIDE_zh.md`为详细科学阅读说明；其中原有案例现已从主图移除，仅供补充理解。`case_evidence_zh.md`保留既有案例原句及评价依据，不表示主图仍展示这些案例。`design/`保留最初设计，最新排版以本说明与实际图件为准。`index.html`是本地图件浏览页。

## 数据口径

100篇科学筛选均保留，694份报告完成。a前四方面使用98篇七配置共同完整论文，引用接触74篇。b按自身完整配对取样，差值单位为百分点、区间为95%论文bootstrap区间。c是要点计数，不是论文数或论文等权均值。d上下图同列对应同篇论文，斜线NA不是0。e的低覆盖针对指定完整结构问题，不代表没有任何结构推理能力。f的26篇为本次材料未检出正向引用接触、未纳入该项评价，不代表论文没有参考文献。

## 复现

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_reference all
```

阶段为prepare/render/package。原始材料只读，无新增模型调用或评分。校对采用代码逻辑阅读和实际导出查看，不编写或运行回归测试。正文说明从画布移入图注，没有改变科学数据。
"""


def companion(data: dict[str, Any]) -> None:
    lines=['# Fig4 案例原句与评分依据', '', '图内英文引句为明确标注的节译；report summary 为内容概述。以下保留完整已有原句，不新增科学评价。', '']
    for row in data['case_parts']:
        lines.extend([f'## {row["paper_id"]} · {row["configuration"]} · {row["part_id"]}',
                      '',f'状态：{row["status"]}；必要范围：{row["scope_correct"]}；证据落实：{row["grounding"]}。',
                      '', '### 实际报告原句', ''])
        lines.extend('> '+q.replace('\n','\n> ')+'\n' for q in row['report_quotes'])
        lines.extend(['### 既有参考与判定', '',row['expected_content'],'',row['reason'],'',
                      '来源键：'+', '.join(row['evidence_source_ids']),''])
    (OUT/'case_evidence_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def gallery() -> None:
    body=['''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>新版 Fig4 · 百篇数据</title><style>body{font:16px Arial,sans-serif;background:#f4f8fb;color:#24282d;max-width:1450px;margin:30px auto;padding:0 24px}p{line-height:1.6}a{color:#1269df}.hero{width:100%;background:white}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:#fff;border:1px solid #a9cde4;border-radius:8px;padding:14px}article img{width:100%;height:310px;object-fit:contain}h2{margin-top:32px}</style>
<h1>Fig4 · 两分支互补贡献与图组件作用</h1><p>本轮百篇数据 · 694份报告 · 6 panels · 11独立子图 · 324 × 181 mm</p>
<p><a href="final/Fig4.svg">可编辑SVG</a> · <a href="final/Fig4_outlined.svg">转路径SVG</a> · <a href="final/Fig4.pdf">矢量PDF</a> · <a href="final/Fig4_600dpi.png">600 dpi PNG</a> · <a href="final/Fig4_900dpi.png">900 dpi PNG</a></p>
<p><a href="README_zh.md">中文说明</a> · <a href="PANEL_GUIDE_zh.md">逐panel详细讲解</a> · <a href="caption_en.md">英文图注</a> · <a href="case_evidence_zh.md">真实案例原句与评分</a> · <a href="data/">绘图源表</a></p>
<img class="hero" src="final/Fig4.png" alt="新版Fig4整图"><h2>独立 panels</h2><div class="grid">''']
    for key in PANELS:
        body.append(f'<article><h3>{key} · {html.escape(TITLES[key])}</h3><a href="panels/{key}.svg"><img src="panels/{key}.png" alt="panel {key}"></a><p><a href="panels/{key}.pdf">PDF</a> · <a href="panels/{key}.svg">SVG</a> · <a href="panels/{key}_outlined.svg">转路径SVG</a></p></article>')
    body.append('</div><h2>独立子图</h2><div class="grid">')
    for key in BOXES:
        body.append(f'<article><h3>{key}</h3><a href="components/{key}.svg"><img src="components/{key}.png" alt="component {key}"></a></article>')
    body.append('</div></html>')
    (OUT/'index.html').write_text('\n'.join(body),encoding='utf-8')


def package() -> Path:
    (OUT/'caption_en.md').write_text(CAPTION,encoding='utf-8')
    (OUT/'README_zh.md').write_text(README,encoding='utf-8')
    companion(read(OUT/'data/snapshot.json'))
    (OUT/'design').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'docs/Fig4_新版百篇数据_逐panel绘图方案.md',OUT/'design/Fig4_逐panel绘图方案.md')
    gallery()
    destination=OUT/'Fig4_final_delivery.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip' and path.relative_to(OUT).parts[0] not in {'experiment', 'source'}:
                archive.write(path,Path('Fig4')/path.relative_to(OUT))
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.suffix not in {'.py', '.md'}:
                continue
            archive.write(path,Path('source/figure_pipeline/fig4_explanation_reference')/path.name)
        for name in ['__init__.py', '__main__.py', 'README.md']:
            path=ROOT/'figure_pipeline/fig4_reference'/name
            archive.write(path,Path('source/figure_pipeline/fig4_reference')/name)
    print(f'Delivery: {destination} ({destination.stat().st_size/1e6:.1f} MB)',flush=True)
    return destination

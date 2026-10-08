from __future__ import annotations

import html
import zipfile
from pathlib import Path

from .data import OUT, read
from .render import BOXES, PANELS, TITLES

CAPTION = '''# Fig. 5 | Robustness, evidence boundaries and computational cost

The figure combines a descriptive analysis of the existing 100-paper Fig.4 complete-system condition (a,b), a fixed stratified 20-paper stress test (c,d,f), and five independently repeated complete-system runs (e). These are distinct populations and evaluation passes. The 20-paper sample excludes five development papers and deliberately includes ten papers with at least one sparse claim neighborhood and ten with all neighborhoods at the cap. It is not an estimate of average perturbation effects in all 100 papers. Each scientific aspect is reported separately; there is no overall innovation score. Rates are computed within paper using fixed applicable scientific parts and then averaged with equal paper weights. Scientific judgments are model evaluations.

a, Four domain groups (life sciences, 40; physical sciences/engineering, 38; medicine, 14; earth/environment, 8). Left: correct content with appropriate scope. Right: correct, appropriately scoped content grounded in evidence, from the accepted Fig.4 F evaluations. Cell labels give percentages and actual evaluable paper counts. Candidate counts remain visible even when technical evaluations are missing or an aspect is inapplicable. Both heatmaps share a 0–100% scale. This observational domain comparison is not a held-out domain-transfer experiment. Summaries excluding the five development papers accompany the figure.

b, Four descriptive historical-coverage strata, using the same Fig.4 F evidence-grounded endpoint. Sparse-claim fraction is split at zero (a sparse claim has fewer than ten eligible neighbors); mean nearest-neighbor similarity and the number of deduplicated readable historical works are split at their respective 100-paper medians, with equal values assigned together; full-text-source fraction is split into <1 versus =1. Open grey circles and blue diamonds mark the two strata defined above each column. Horizontal lines are 95% percentile intervals from 10,000 paper-bootstrap draws, seed 20261002. The horizontal axis is explicitly focused on 70–100%, not zero-based. N denotes candidate papers; per-aspect n, exact thresholds and observed ranges are in b_stratum_summary.csv. Group membership is fixed from coverage alone. These are associations, not causal effects or validated failure thresholds; sparse graph neighborhoods are not equivalent to inadequate scientific evidence.

c, Paired condition-minus-F changes in evidence-grounded correct coverage (percentage points). E50 retains floor(N/2) independent historical source works and regenerates both branch interpretations and the report; actual retention may be below 50% for odd N. K5 changes the eligible-neighbor cap from ten to five, recomputes native graph facts and Graph interpretation, holds the historical text pool fixed, and reuses GEAR. LUNA regenerates both branches and the report with gpt-5.6-luna/medium; other generation uses gpt-6.1-sol/medium. F is the matched Fig.4 report, re-evaluated alongside perturbations. Small points are individual paired differences; diamonds and lines are means and stored 95% paper-bootstrap CIs (10,000 draws, seed 20261002). E50 has 19 valid pairs for the first four aspects and 13 for citation contact; K5 and LUNA have 20 and 14. One E50 Graph request for s41467-026-69411-2 was rejected by the provider's content restriction; its dependent report is technically missing, not zero. K5 is evaluated against its actual K5 facts, not K10 answers. Scientific evaluation uncertainty is not confirmed correct coverage.

d, Left: explicit abstention rates, split into reasonable, unnecessary and unresolved appropriateness, for F, E50, K5 and LUNA. All segments use the original applicable-part denominator, not just abstentions. Omission or generic caution is not explicit abstention; the unfilled bar area is not labelled correct. Right: confirmed unsupported definitive-assertion rates (0–30% color scale). Each cell reports confirmed / unresolved percentages for that assertion judgment and its evaluable-paper count; unresolved is not treated as confirmed absence. These are absolute paper-weighted rates over each condition's own technically complete set, so E50 has one fewer paper. They are not paired causal estimates. Other scientific uncertainty (e.g. scope or correctness) remains separately available in the source tables and is not conflated with uncertainty of the unsupported-assertion flag. The scientific metrics overlap and must not be added into a total score.

e, Five repeated complete-system runs, retained independently without selecting the better report. Left: each marker identifies the same paper across aspects and plots its F_REPEAT-minus-F grounded-coverage difference; symbols are vertically offset solely for visibility. Citation contact applies to four of these papers. Right: observed minimum–maximum changes in explicit abstention and confirmed unsupported assertions. No confidence interval, estimated noise band, or significance cutoff is inferred from these five repeats. Paper identities and full paired differences accompany the figure.

f, Per-paper ratios of complete attributable fixed-material generation costs to F. Left: input and output tokens remain separate (open circle / filled diamond); right: the sum of branch and report call durations. Small points show paired paper ratios; large marks and lines show the median and interquartile range. E50 has 19 complete cost pairs; K5 and LUNA have 20. The reused GEAR call is charged to K5's attributable workflow; historical F calls are attributable costs but not newly incurred expenditure. Thus this comparison is not simply the number of newly launched calls. Call-time sums differ from elapsed wall time and reflect model, transport and concurrent-service conditions, not hardware-normalized latency. Retrieval, embeddings and historical asset construction were not rerun and are excluded. Model token counts do not imply equivalent dollar prices or FLOPs. Unknown usage is not zero. New spending, references/evaluation, cache tokens and available event-wall times are separately provided in the data companion.

All 20 private conditional references were prepared before new reports. Public questions and part identities remain fixed. Evaluation used gpt-6.1-sol/high with anonymous candidate and material identifiers. There are 84 valid reports out of 85 planned objects, 100 paper-by-aspect evaluation slots (including inapplicable slots), 288 new request attempts, 287 successful requests and one failed request; no additional-budget requests were needed. Technical missingness, original inapplicability and scientific uncertainty remain distinct. This plotting step performs no model calls or scientific relabelling.
'''

README = '''# Fig5：鲁棒性、证据边界与计算代价

按确认方案绘制三行六panel，324 × 252 mm，15个独立组件；与Fig4保持字体、浅蓝边框、标题栏和数据编码风格一致。

- a：100篇既有Fig4完整条件的跨领域正确覆盖与证据落实覆盖。
- b：四个历史覆盖变量分别分层的证据落实覆盖；不合成覆盖分数，不解释为因果或失效阈值。
- c：20篇压力样本中E50、K5、LUNA相对F的有依据正确覆盖配对差；点为论文，菱形为均值，线为95%论文bootstrap区间。
- d：合理／不必要／未决的明确保留，以及已确认无依据断言。右侧单元格同时列出该断言判定的未决比例，其他科学未决保留在源表。
- e：5篇原配置重复的真实差异和范围，无置信区间、显著性界限或优选结果。
- f：完整可归属生成流程的同篇成本比值，输入、输出token与调用耗时分别展示，中位数和四分位范围不冒充置信区间。

`final/`：可编辑SVG、转路径SVG、矢量PDF、300/600/900 dpi PNG。
`panels/`：6个独立panel；`components/`：15个独立组件，均提供SVG/PDF/PNG。
`data/`：复制的现有源表及绘图时产生的简单聚合；保留每篇身份、分母、未决和缺失原因。
`caption_en.md`：完整英文图注与统计口径。
`index.html`：本地图件浏览页。

主图100篇分层与20篇扰动来自不同评价集合，不混合。20篇为压力样本，不代表百篇总体扰动效应。84/85份报告有效，s41467-026-69411-2的E50因服务方内容限制缺失，其19条适用要点不计零分。百篇来源另有原有技术缺失与不适用，均保留。成本不包含检索、嵌入或建库，不换算美元；调用时间之和不是墙钟时间。

复现：

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_reference all
```

也可单独执行prepare/render/package。只读取outputs/fig5_robustness，不修改Fig1–Fig4或Fig5实验产物，不产生新的模型评分。校对采用源表算术核对与实际导出目视检查，不添加独立验证框架。
'''


def package() -> Path:
    (OUT/'caption_en.md').write_text(CAPTION, encoding='utf-8')
    cohort = read(OUT/'data/cohort.json')
    repeat = '\n'.join(f'- {mark}: `{paper}`' for mark, paper in zip(['○','□','△','◇','▽'], cohort['repeated']))
    (OUT/'README_zh.md').write_text(README+'\n重复样本符号：\n\n'+repeat+'\n', encoding='utf-8')
    body = ['''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fig5 · 鲁棒性与运行代价</title><style>body{font:16px Arial,sans-serif;background:#f4f8fb;color:#24282d;max-width:1450px;margin:30px auto;padding:0 24px}p{line-height:1.6}a{color:#1269df}.hero{width:100%;background:white}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:white;border:1px solid #a9cde4;padding:14px}article img{width:100%;height:280px;object-fit:contain}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>Fig5 · 鲁棒性、证据边界与计算代价</h1><p>100篇分层 · 20篇压力测试 · 5篇重复 · 84/85份报告有效</p>
<p><a href="final/Fig5.svg">可编辑SVG</a> · <a href="final/Fig5.pdf">PDF</a> · <a href="final/Fig5_600dpi.png">600 dpi PNG</a> · <a href="README_zh.md">中文说明</a> · <a href="caption_en.md">英文图注</a> · <a href="data/">绘图源表</a></p>
<img class="hero" src="final/Fig5.png" alt="Fig5整图"><h2>独立panel</h2><div class="grid">''']
    for key in PANELS:
        body.append(f'<article><h3>{key} · {html.escape(TITLES[key])}</h3><a href="panels/{key}.svg"><img src="panels/{key}.png" alt="panel {key}"></a><p><a href="panels/{key}.pdf">PDF</a> · <a href="panels/{key}.svg">SVG</a></p></article>')
    body.append('</div><h2>独立组件</h2><div class="grid">')
    for key in BOXES:
        body.append(f'<article><h3>{key}</h3><a href="components/{key}.svg"><img src="components/{key}.png" alt="component {key}"></a></article>')
    body.append('</div></html>')
    (OUT/'index.html').write_text('\n'.join(body),encoding='utf-8')
    target=OUT/'Fig5_final_delivery.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix!='.zip': archive.write(path,Path('Fig5')/path.relative_to(OUT))
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.suffix in ('.py','.md'): archive.write(path,Path('source/figure_pipeline/fig5_reference')/path.name)
        archive.write(Path(__file__).parents[1]/'fig1_reference/export.py', 'source/figure_pipeline/fig1_reference/export.py')
    print(f'Delivery: {target}',flush=True)
    return target

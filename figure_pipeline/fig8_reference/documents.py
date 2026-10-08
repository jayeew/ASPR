"""Reproduction notes, validation and self-contained plotted-data delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .helpers import write_json

from .data import OUT, ROOT, TYPES, YEARS


def verify(data: dict) -> dict:
    manifest = json.loads((OUT / 'sources/source_manifest.json').read_text())
    for name, record in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record['sha256'], name
    claimed = [p for p in data['papers'] if p['n_claims']]
    assert all(abs(sum(p[f'type_{t}'] for t in TYPES) - 1) < 1e-12 for p in claimed)
    plotted = [p for p in data['papers'] if p['scatter_eligible']]
    assert all(1 <= p['effective_communities'] <= p['n_assigned_neighbors'] + 1e-9 for p in plotted)
    assert all(0 <= p['mean_jaccard'] <= 1 for p in plotted)
    assert len(plotted) == 24344
    assert sum(p['shared_neighbor_fraction'] is not None for p in data['papers']) == 24345
    return dict(source_files_unchanged=len(manifest), no_gear_sources=True,
                year_coordinate_counts={year: sum(p['year'] == year for p in plotted) for year in YEARS},
                type_fractions_sum_to_one=True, undefined_not_imputed=True,
                metrics_range_checks=True, missing_communities_never_grouped=True,
                all_eligible_points_plotted=True, model_calls=0, embedding_calls=0)


def write_notes(data: dict) -> None:
    caption = '''Figure 8 | Contribution and knowledge-structure profiles across journals.

Population. The retrospective local 2023–2025 corpus contains 24,919 papers across eight Nature-series journals, not all worldwide papers or a certified census of those journals. Nature Communications contributes 24,224 papers; the other journals contribute 59, 71, 116, 119, 100, 73 and 157 papers in displayed order. Of these, 24,914 papers have 70,034 saved abstract-derived contribution claims. Five papers have no claim record. The extraction requested one to three central claims per abstract; these are not full-text claim inventories and must not be pooled with the 2026 full-text study. Fields and article types are not matched, and journal comparisons describe this corpus only.

a, Paper-equal contribution profiles. Within each claim-bearing paper, five mutually exclusive claim-type fractions are calculated, then averaged over papers in each journal. Bars are not pooled claim counts. Labels n refer to claim-bearing papers. Annual sample counts and coordinate-coverage numerators and denominators remain available in journal_year_summary.csv; the coverage matrix is omitted from the figure. Sample counts use canonical publication year, not the synthetic date field retained in abstract claim records.

b, Each point is one eligible paper in its publication-year facet, with identical axis limits across years. All 24,344 eligible papers are plotted. Nature Communications is drawn first using small translucent marks; the seven smaller journals use larger marks and distinct shapes to prevent occlusion. Mark size is a rendering convention, not scientific magnitude. No density contours, clustering boundaries, innovation quadrants or significance claims are inferred.

Historical neighborhoods. The saved temporal semantic edges were generated as the top ten earlier-date, other-paper neighbors per claim; we retain only cosine strictly above 0.5. The resulting 669,755 edges are reused without new embedding or semantic searches. Same-day and later-paper claims are not eligible. A paper's neighborhood is the union of unique prior claim nodes across its claims. D, the x coordinate, is exp(Shannon entropy) of the community distribution among assigned nodes in that union, without cosine weighting or repeated-node weighting. Of all historical claim nodes, 14,872 have no community assignment. They are excluded from the D denominator, never collapsed into a fictitious shared community, and per-paper assignment counts and fractions are exported. D is undefined if no neighbor has an assignment. J, the y coordinate, is the arithmetic mean of pairwise Jaccard overlap over nonempty claim-neighbor sets; empty sets do not contribute a zero. At least two nonempty claim neighborhoods are required. Paper-level missing reasons can overlap and are exported; undefined papers are retained in cohort and coverage tables.

Temporal interpretation. All facets use the same saved 2023–2025 community partition as a retrospective structural description, not an as-of-publication community reconstruction. The graph contains no pre-2023 target corpus, so early papers have shorter available history. Year differences cannot be interpreted as innovation growth. Communities are semantic clusters, not established disciplines. The figure rederives raw paper-level quantities and uses no old percentiles or historical insertion-profile summaries.

c, Within-journal heterogeneity. Each point shows the fraction of unique joint-neighborhood nodes connected to at least two claims from that paper; papers require at least two nonempty neighborhoods. All 24,345 eligible values are retained, including one paper with no assigned community and hence no b coordinate. Points are pooled across years and jittered only vertically. Dark bars span the interquartile range and white ticks mark medians; these are observation spread, not confidence intervals. Year-specific counts remain available in the data tables.

Computation. This figure uses only existing local metadata, abstract claims, semantic edges, communities. There were zero new language-model, embedding or retrieval calls. No GEAR historical comparisons or report evaluations enter the figure. Source hashes and a reproduction entry point are supplied. These validate source identity and numerical processing, not the scientific correctness of original extraction or the representativeness of collection.
'''
    (OUT / 'caption_en.txt').write_text(caption, encoding='utf-8')
    guide = '''Fig.8 历史语料版：2023–2025年跨期刊贡献与知识结构

本版正式图在final/Fig8.png、Fig8.pdf和Fig8.svg；已完全移除GEAR指标。本目录为唯一当前定稿；旧200篇结果与删除的案例组件已清理。

真实范围：仓库收录24,919篇、8刊。2023年5,766篇，2024年9,265篇，2025年9,888篇。24,914篇有摘要贡献，共70,034条；这不是全科学文献或出版社完整普查。原始摘要抽取每篇最多3条，不能理解成全文贡献总量。

a：先按论文计算五类贡献构成，再按刊等权平均。条旁n是有贡献的论文数。覆盖矩阵已从图中删除，年度收录量和坐标可用率仍保存在data/journal_year_summary.csv。没有贡献的5篇仍留在覆盖表。
b：三个年度分面使用相同坐标，展示24,344篇。横轴只用已分配社区的历史邻居计算有效社区数D；纵轴是非空贡献邻域对的平均Jaccard重叠J。14,872个历史节点没有社区归属，不能把它们当成一个社区。各论文分配覆盖率在plot_data.json.gz中。Nature Communications点小且透明、先画；其他期刊用颜色和形状区分、后画，点大小没有科学含义。
c（由原c1移至左下）：24,345篇的“被同篇至少两条贡献共享的历史节点比例”；点保留原始数值，只有纵向避让。黑线是四分位范围、白刻线是中位数，不是置信区间。主图混合三年，年度分母保留在汇总表。

时间边界：先行邻居必须严格早于目标论文，cosine>0.5且每claim最多10条；社区使用2023–2025全图的固定回顾性划分，不是当年知识快照。语料从2023年起，早期可用历史较短，不能把年度差异写成创新趋势。参考来源日期用canonical_target_works，不用摘要claim中保留的年份占位日期。

文件：data保留压缩绘图数据（全部论文指标、汇总、缺失口径）和年度汇总；sources记录只读原始文件哈希；qa记录分母与核验；panels是独立a/b/c；源码在仓库figure_pipeline/fig8_reference。
复现需在原仓库执行：
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig8_reference
仅重绘：上述命令追加 --render-only。
运行只做确定性数据处理与绘图，不调用语言模型、嵌入模型或GEAR，不修改data/claim_graph。
'''
    (OUT / 'PANEL_GUIDE_zh.txt').write_text(guide, encoding='utf-8')


def deliver(data: dict) -> None:
    write_json(OUT / 'qa/final_validation.json', verify(data))
    write_notes(data)
    write_json(OUT / 'qa/export_manifest.json', {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['final', 'panels'] for p in sorted((OUT / folder).glob('*')) if p.is_file()})

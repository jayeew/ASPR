# 真人审稿材料：七类理论归并与统一统计

本地中文报告位于 `report_zh.md`。重建所需的原始材料、编码和已生成结果由本地工作区提供。全部结果统一使用994篇有效论文。当前及后续分析不按材料收集阶段拆分统计或进行前后比较。

创新评价维度及原五角度对照位于 `open_discovery/innovation_dimensions_zh.md`。该分析以两个创新核心类别的10个主题为依据，提出三个核心判断及四种贡献形式，保留14个经原文定位核验的典型与边界案例。相关频数为主题出现率，不是维度认可率；论文与既有编码未修改。

## 主要文件

- `open_discovery/theoretical_codebook.json`：七类定义、100项映射、跨类解释边界及统计口径。
- `open_discovery/theoretical_framework_zh.md`：完整归并规范。
- `open_discovery/theoretical_category_statistics.csv`：七类总体频数和比例。
- `open_discovery/evaluation_role_statistics.csv`：创新核心、研究质量、研究责任和评价条件的去重并集。
- `open_discovery/theme_statistics.csv`：100项具体关注点的总体频数和比例。
- `open_discovery/theoretical_category_by_group.csv`：七类的期刊与领域分布。
- `open_discovery/theme_by_group.csv`：具体关注点的期刊与领域分布。
- `open_discovery/paper_theoretical_category_matrix.csv`：逐篇七类矩阵。
- `open_discovery/theme_to_theoretical_category.csv`：全部具体关注点的主要归属与跨类条件。
- `open_discovery/theme_evidence.jsonl`：原文、身份、位置及修正状态。
- `open_discovery/protocol.md`：当前统计和归并方法。
- `open_discovery/theoretical_aggregation_audit.json`、`final_validation.json`：汇总核验。
- `open_discovery/innovation_dimension_theme_statistics.csv`：创新评价相关10个主题的统一统计。
- `open_discovery/innovation_dimension_framework.json`：三个核心判断、四种贡献形式及原五角度映射。
- `open_discovery/innovation_dimension_examples.json`：典型原文、完整来源片段、位置和解释。
- `open_discovery/innovation_dimension_analysis_validation.json`：相关计数与引文核验。

## 当前重建命令

在项目根目录运行，以下命令不调用模型：

```bash
python3 outputs/reviewer_open_coding_pilot_20261116/open_discovery/merge_screening_passes.py
python3 outputs/reviewer_open_coding_pilot_20261116/open_discovery/aggregate_themes.py --input-dir outputs/reviewer_open_coding_pilot_20261116/open_discovery/complete_screening --sample outputs/reviewer_open_coding_pilot_20261116/sample.jsonl --output-dir outputs/reviewer_open_coding_pilot_20261116/open_discovery --require-complete
python3 outputs/reviewer_open_coding_pilot_20261116/open_discovery/consolidate_themes.py
python3 outputs/reviewer_open_coding_pilot_20261116/open_discovery/validate_final_distribution.py
```

`open_discovery/write_discovery_report.py`和`write_findings.py`均转入同一当前报告生成器。不要用根目录的历史`write_report.py`或`compile_analysis.py`重建当前报告：它们属于此前归纳流程，未采用当前理论归并口径。

独立重建创新评价维度分析（不调用模型、不修改原始编码）：

```bash
python3 outputs/reviewer_open_coding_pilot_20261116/open_discovery/analyse_innovation_dimensions.py
```

## 保留的来源记录

名册、原文评审块、附件、开放编码、原始模型输出、语义调整账本及历史审计均保留。历史阶段标记只用于追溯，不进入当前统计输出。论文正文、终版图件和科学实验结果均未改动。

七类是可解释的概念归并，不是已经验证的七因子量表；分类频率表示文本讨论，不是评审一致率或创新得分。

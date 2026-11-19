# 数据目录指南

`data/` 存放多个流程共用的输入和检索缓存。以下路径以仓库根目录为基准。图谱、模型和大宗研究材料由本地工作区提供。

| 位置 | 内容与用途 |
| --- | --- |
| `data/claim_graph/` | 历史主张图谱、父论文引文图及嵌入索引。运行时使用该图谱进行临时插入。 |
| `data/knowledge_corpus/`、`data/nature_markdown/` | 共享文献语料和稿件 Markdown。`data/nature_markdown/` 是指向本地语料的符号链接。 |
| `data/nature_2026_testset/` | 论文与公开审稿来源对、名册和来源记录。研究运行使用各自的 `papers.jsonl`。 |
| `data/models/` | 本地嵌入和检索模型资产。 |
| `data/cache/`、`data/gear_cache/` | 检索与运行缓存。判断能否清理时，先查已保存的证据、运行状态和复现入口。 |
| `data/calibration/`、`data/doi_coverage_audit/` | 历史校准、发布与回放资产，以及 DOI 覆盖核对记录。相应配置和历史研究引用这些目录。 |

单图证据和绘图输入放在 `figures/figN/data/`，可编辑母版与正式导出放在 `figures/figN/final/`。稿件及版本化图件位于 `manuscript/nature_innovation/`。 [图件入口](../figures/README.md) · [v10 稿件](../manuscript/nature_innovation/README.md)

跨图研究和 GEAR 运行结果放在 `outputs/` 下各自的运行目录，日志与检查点随运行保存。旧版本研究及冻结发布的索引在 `archive/`。 [输出与重绘指南](../outputs/README.md) · [历史研究档案](../archive/README.md)

数据来源、评价范围以及大文件的本地存储方式分别见对应指南。 [数据来源](../docs/data_sources.md) · [存储指南](../docs/HDD_STORAGE.md)

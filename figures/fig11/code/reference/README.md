# Fig.11 — Claim 分类依据与实际提示词

补充方法说明表，不是新增实验结果。采用 Fig.1–Fig.8 的可编辑 SVG
绘图及导出工具、Times New Roman 字体、深蓝文字、浅蓝表头和五类节点形状。

```bash
python3 -m figure_pipeline.fig11_reference
```

本机包含完整绘图依赖的环境：
`/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig11_reference`。

输出位于 `outputs/fig11_reference/`：`final/Fig11.svg`（可编辑文字）、
`final/Fig11_outlined.svg`（文字转路径）、`final/Fig11.pdf`、
`final/Fig11.png`、`prompts.json`（实际提示词完整原文）、`qa.json`。
不调用模型，不修改运行时、历史图或既有交付清单。

## 图中内容与来源

表格第二列“Classification criteria and distinctions”（分类依据与区别）合并
历史摘要提示词的定义与易混淆边界，属于概括说明，不再标为逐字引用。
第三列“Hypothetical example”（假设例子）围绕细胞成像给出五类示例；
这些例子仅用于说明分类，不是研究中的实际结果。
历史摘要输出标签通过 `TYPE_ALIASES` 映射到图中的五个大写类别。

| 类别 | 核心区别 |
| --- | --- |
| METHOD | 可复用的方法、算法、系统、材料或干预设计本身；应用后的效果属于 FINDING。 |
| FINDING | 观测、测量、估计、比较、结构、性能、关联或干预结果。 |
| MECHANISM | 有证据支持的“如何／为何发生”的因果解释；只有相关性或推断连接不够。 |
| RESOURCE | 交付供复用的数据集、图谱、数据库、目录、基准、基因组组装或整理性综合。 |
| THEORY | 新提出的形式原理、定律或概念／数学框架；做了理论计算不自动构成 THEORY。 |

图中仅保留分类表；下方两个提示词框及底部灰色注释已按要求删除。来源说明保留于本文档与提示词附件。
`prompts.json` 还保留历史 `SYSTEM_PROMPT`、完整 `PROMPT_TEMPLATE`，
以及全文 `CONSOLIDATOR_SYSTEM`、`VERIFIER_SYSTEM`。
动态论文内容不在此静态模板附件中；全文 miner 的 user 输入是
`paper_id` 和带 `span_id/section/page/text` 的 spans，响应 schema 的
`claim_type` 取值来自 `InnovationClaimType` 五类枚举。

两条路径须区分：历史图从编号摘要句抽取 1–3 条；当前全文路径依次挖掘、
合并和核验（当前 consolidator 提示词与 schema 为 1–8 条）。
全文 miner 并未包含历史摘要提示词里的详细类型定义，因此不能把该表
描述为两条路径共享的完整 system prompt。图中呈现的是当前源码，
不声称所有历史产物均由这个版本的提示词生成。

类别表示贡献角色，不是创新等级、学科分区、图社区，也不能单独证明因果或新颖性。
脚本核验文本重叠与画布边界，记录来源文件 SHA-256 以便追溯；这不构成科学正确性验证。

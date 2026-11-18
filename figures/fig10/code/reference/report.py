"""Package the generated report with an explicit evidence-reviewed supplement."""

from __future__ import annotations
from figure_pipeline.paths import PROJECT_ROOT

import json
from pathlib import Path

ROOT = PROJECT_ROOT
OUT = ROOT / "outputs/fig10_reference"
LABELS = [
    "统一生物分子预测",
    "Pairformer 与原子坐标扩散",
    "噪声尺度与结构学习",
    "训练与交叉蒸馏",
    "跨任务基准表现",
    "共价修饰与磷酸化",
    "置信度与推理采样",
    "服务器与算法开放",
]
STANCE = {
    "recognized": "范围内认可",
    "unresolved": "未决",
    "incremental_or_limited": "有限增量",
    "challenged": "质疑",
}


def main() -> None:
    rows = []
    for i, label in enumerate(LABELS, 1):
        values = [
            json.loads((OUT / f"analysis/{b}/{i:02}/assessment.json").read_text())[
                "overall_stance"
            ]
            for b in ("gear", "graph", "fusion")
        ]
        rows.append(
            f"| C{i} | {label} | " + " | ".join(STANCE[v] for v in values) + " |"
        )
    intro = """# AlphaFold 3：ASPR 完整创新性分析与 Fig10 正式案例

目标论文：[Abramson et al., Nature (2024)](https://doi.org/10.1038/s41586-024-07487-w)。历史截止日：2024-05-08；案例执行及审阅日期：2026-10-31。

## 核心结论

本次分析支持 AF3 在统一生物分子预测、Pairformer 与原子坐标扩散架构、限定基准表现以及共价修饰评估方面的具体贡献。RFAA、AF2、核酸与配体专用预测器等构成历史基础，因而不能将“统一建模”或“使用扩散”本身写成全球首次。八条贡献的融合结果为四条范围内认可、四条未决；这不是创新分数，未决也不表示缺乏创新。

共享主张、独立 GEAR 与 Graph、联合 Graph、融合均已完成。下表直接读取最终 assessment；详细科学理由及证据见后文与原始分析。C5 的原文支持为部分支持，性能判断保留原评测条件。

| 主张 | 内容 | GEAR | Graph | 融合 |
|---|---|---|---|---|
"""
    supplement = """

## 完整性审阅补充：C8 与联合 Graph

以下为根据已保存证据与分析补充的正式案例说明。自动整篇报告正文覆盖 C1–C7，但未单独展开 C8，因此保留其原文，并在此显式补齐，不改写分支判断。

**C8 资源开放。** 论文方法部分声明计划提供仅限非商业用途的服务器，对配体和共价修饰输入设限，补充材料给出算法伪代码而不提供实现代码。这是所分析文本中的发表时期资源安排，不是对今天代码或服务可用性的调查。原文依据为 `gear:P:S-961dd96e647acf4ad603`，可在 `analysis/gear/08/` 的 EvidenceStore 查验；该段位于结构化正文字符区间 60119–60393。已有网页建模服务及 AlphaFold 数据库为一般交付方式提供历史背景，但现有检索不足以确认这一限制与披露组合的首创性。Graph 的主题邻近也不能补足政策历史比较，故 GEAR、Graph、融合均为未决。详见 [C8 融合原始结果](analysis/fusion/08/assessment.json)。

**联合邻域。** 八条目标主张合计连接 48 条不同的历史主张，其中 13 条被至少两条目标主张共享。80 条插入边来自每条最多十个、cosine 严格大于 0.5 的合格邻居。临时插入后，历史邻域的连通分量由 15 个变为 1 个，新增可连通历史节点对为 711。它们直接来自 [joint facts](analysis/graph/joint/facts.json)，不是单条主张指标相加；没有人为建立目标主张之间的边。这些量反映局部连接变化，不能推出真实知识推导、因果、跨学科首创或全球影响。

**分支分歧。** C3 的 Graph 判断认可局部知识关系，GEAR 与融合仍未解决历史新颖性。原始 joint 解读中一处问题将噪声尺度误称为 C2；正式案例按共享主张身份将该内容归于 C3，原始输出留存不改。Graph 认可与 GEAR 未决是证据维度不同，并非通过多数表决解决。

## Fig10 与图注

![Fig10](Fig10_outlined.png)

**Fig. 10 | AlphaFold 3 的有证据依据的创新性分析。** a，共享全文贡献与 GEAR、Graph、融合判断；认可表示给定证据范围内的认可，不等于全球首创。b，三项可追溯历史比较，分别展示统一建模的已有基础、具体架构变化和修饰相关实证增量。c，从实际联合邻域绘制的临时插入图，48 个历史节点与 8 个编号目标节点，实线表示历史连接、虚线表示插入连接；空间位置仅为布局。d，证据综合与边界，包含 C3 的分支分歧。Graph 语料日期为 2023-01-07 至 2024-05-07；更早先行工作由 GEAR 检索处理。所有图谱指标均限定于该语料与邻域。

图形沿用仓库 Fig1–8 的 SVG 绘制与导出方式、衬线字体及蓝／橙／紫分面配色。[PDF](Fig10_outlined.pdf) · [可编辑 SVG](Fig10.svg) · [逐项来源数据](figure_source_data.json)。

## 方法与证据边界

"""
    methods = (OUT / "case_methods.md").read_text().split("\n", 2)[2]
    generated = (OUT / "innovation_report.md").read_text()
    (OUT / "case_report.md").write_text(
        intro
        + "\n".join(rows)
        + supplement
        + methods
        + "\n\n## 自动生成的整篇分析与原始引文附录\n\n"
        + generated
    )


if __name__ == "__main__":
    main()

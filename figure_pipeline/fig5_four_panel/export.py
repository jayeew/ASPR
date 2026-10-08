from __future__ import annotations

import zipfile
from pathlib import Path

from figure_pipeline.fig3_revision.storage import read

from .prepare import OUT


def main() -> None:
    summary = read(OUT / 'summary.json')
    protocol = read(OUT / 'protocol.json')
    background = read(OUT / 'paper_background.json')
    a = read(OUT / 'panel_a_summary.json')
    b = read(OUT / 'panel_b_summary.json')
    c = read(OUT / 'panel_c_summary.json')
    d = read(OUT / 'panel_d_summary.json')
    groups = read(OUT / 'frozen_history_groups.json')
    notes = [
        '# Fig.5 四 panel：数据审查、纠错评价与绘图', '',
        '本次仅复用原始 `outputs/fig5_robustness` 的 100 篇观察样本、20 篇压力样本及其中 5 篇重复。'
        '未纳入后续扩展实验，未重新生成分支分析或报告，未修改 Fig.1–4 或旧 Fig.5 结果。', '',
        f'新增评价请求 **{summary["new_calls"]}** 次（包含拆批及失败），硬上限 30 次；'
        f'原批次 {protocol["prior_requests"]} 次，本批合计 **{protocol["prior_requests"] + summary["new_calls"]}** 次。'
        '评价模型为 gpt-6.1-sol/high；每篇论文合并所有科学方面及匿名候选。容量不足时按候选拆批。', '',
        '## 数据审查与处理', '',
        '- 原计划 85 份报告，实际 84 份。s41467-026-69411-2 的 E50 报告缺失；保留缺失，不补生成。',
        '- 100 篇来源表中发现两处同一原文块对应多个 DOI：'
        's43246-026-01100-5 的 anie/ange 版本，及 s42003-026-09556-4 的预印本 v2/v3。'
        '按同一原文合并计数；原始身份、原文块与溯源元数据保存在 source_identity_merges.json。',
        '- 压力样本中的双 DOI 均在 E50 保留集合内，未发现删除来源经别名进入实际分支或 writer 输入。'
        '这是计数修正；实际来源比例见 e50_input_audit.csv。',
        '- 来源分组按去重后的全体 100 篇输入确定：≤16 与 >16。'
        f'可读来源范围为 {min(r["readable_sources"] for r in background)}–{max(r["readable_sources"] for r in background)}，'
        f'全文来源比例最低 {100*min(r["fulltext_fraction"] for r in background):.1f}%。',
        '- 在读取本轮报告评价结果前，复核了 19 个 E50 条件参考：原理由明确指出必需原文缺失、'
        '科学等价替代不足，却把材料不足标作 unresolved。统一改为 unanswerable；'
        '原标签、理由和固定要点身份保存在 frozen_history_groups.csv，未按候选表现改组。',
        '- 纠错评价接入 Fig.4 的 J/S 细化标准与 JOINT_REVIEW；范围限定不自动算弃答，'
        '遗漏的空 scope 不自动算科学未决；有正确片段但另有实质冲突的要点不计完成。', '',
        '## a：观察分层', '',
        '复用 Fig.4 原有完整系统评价，不混入本轮纠错评价。能力轮廓中的点为论文等权的有据正确覆盖率，'
        '图例列出有效论文数及引文接触任务的有效论文数；上下两幅轮廓是同一批论文的两种分组。含 5 篇开发样本，'
        '不应称为独立留出领域泛化。该批覆盖范围不支持“无历史资料仍可靠”的结论。', '',
        '| 分组 | 历史核对 | 知识位置 | 联合结构 | 结构分辨率 | 引文接触 |', '|---|---|---|---|---|---|',
    ]
    for group in ('life', 'physical_engineering', 'medicine', 'earth_environment', 'lower', 'higher'):
        cells = [r for r in a if r['group'] == group]
        notes.append('| ' + group + ' | ' + ' | '.join(f'{100*r["mean"]:.1f}% (n={r["n"]})' if r['mean'] is not None else 'NA' for r in cells) + ' |')
    notes += ['', '## b：配对变化', '',
              '每篇论文的固定适用要点分母保持不变；K5 按其实际图事实评价。'
              '细线连接同一论文在邻居上限调整与模型替换下的配对差值，菱形为平均差；95% 区间来自 10,000 次论文配对 bootstrap（种子 20261003）。'
              '区间反映样本差值不确定性，不是评委准确率或等效性检验。', '',
              '| 条件 | 方面 | n | 平均差（百分点） | 95% 区间 |', '|---|---|---:|---:|---|']
    for r in b:
        if r['condition'] != 'F_REPEAT' and r['mean'] is not None:
            notes.append(f'| {r["condition"]} − F | {r["aspect"]} | {r["n"]} | {r["mean"]:+.2f} | [{r["low"]:.2f}, {r["high"]:.2f}] |')
    notes += ['', '5 篇 F_REPEAT 的本轮评价结果保留在 panel_b_paired / panel_b_summary 表，'
              '仅作附表描述，不用于定义普遍噪声带。', '', '## c：证据条件化行为', '',
              f'原 20 篇共有 66 个历史方面要点。依据原候选不可见的条件参考，'
              f'{sum(r["eligible"] for r in groups)} 个满足 F 可答且依赖独立历史原文。'
              '3 个主要由稿件自身支持的要点未纳入主图；E50 缺失论文与技术无效配对另行排除。', '',
              '| 证据组 | 条件 | n / N | 有据作答 | 明确保留 | 无据断言 | 未完成/遗漏 | 评委未决 |',
              '|---|---|---|---|---|---|---|---|']
    for group in ('retained', 'lost'):
        for condition in ('F', 'E50'):
            rr = [r for r in c if r['group'] == group and r['condition'] == condition]
            notes.append(f'| {group} | {condition} | {rr[0]["n"]} / {rr[0]["N"]} | ' +
                         ' | '.join(f'{r["percent"]:.1f}% ({r["count"]})' if r['percent'] is not None else 'NA' for r in rr) + ' |')
    notes += ['', '流向图连接同一固定要点在完整历史材料与约半数来源条件下的实际状态；带宽按组内要点比例绘制。明确保留与合理保留不同：'
              '仍可答组中的保留通常是不必要的；依据丢失组也必须有针对该要点的成立理由，'
              '且无同时出现的无据断言，才能计为合理保留。未完成/遗漏包括明确漏答、'
              '不足以完成构念的部分回答以及其他未达标实质回答；这些细类保留在逐要点标签中。'
              '仅绘制实际出现的状态与流向；该历史要点子集中没有已确认无据断言或评委未决，不能外推为整体错误率为零。', '',
              '## d：质量—成本', '',
              '四条件共同有效且成本完整的论文集合；有据作答产出率按每篇全部原固定适用要点计算，再对论文等权平均。'
              'E50 丢失依据的要点保留在分母。横轴为每篇相对 F 的累计调用时间比的中位数；'
              '四个配置用不同形状标记，箭头表示相对完整系统的质量—耗时位移；右侧卡片完整列出配置、作答率、合理保留率、无据断言率和耗时比。', '',
              '| 配置 | 有效论文数 | 调用耗时比中位数 | 有据作答产出率 | 合理保留率 | 已确认无据断言率 | 评价未决率 |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in d:
        if r['n']:
            notes.append(f'| {r["condition"]} | {r["n"]} | {r["time_ratio_median"]:.3f} | ' +
                         ' | '.join(f'{100*r[k]:.2f}%' for k in ('grounded_answer_rate', 'reasonable_abstention_rate', 'confirmed_unsupported_rate', 'scientific_unresolved_rate')) + ' |')
    notes += ['', '归属成本计入 K5 复用的 GEAR 分析；新支出与复用归属成本在 costs.csv 分列。'
              '不包括检索、嵌入、建库，累计调用耗时不是用户墙钟等待。无完整货币费用记录，未推算美元成本。'
              '仅绘制四个已观测配置，不拟合连续最优前沿。', '',
              '## 评价有效性与未决边界', '',
              f'逐要点状态：`{summary["part_states"]}`。技术缺失不作科学零分；'
              '只有技术完整的论文/方面进入对应配对均值。科学未决留在固定分母中但不计已确认作答，'
              '因此 Q/覆盖率是已确认有效产出率，U 是已确认错误比例，不能把 U 的补数称为可靠率。',
              f'参考可答性与候选保留合理性存在 {summary["reference_abstention_disagreements"]} 条标记分歧；'
              '详见 answer_parts 的 reference_abstention_disagreement 字段。条件参考与纠错评价均为模型辅助结果，'
              '不构成人工金标准。', '', '## 四张底表与复现', '',
              '- paper_background.csv：论文背景、开发样本标记与去重来源分层。',
              '- conditions_materials.csv：论文×条件的可用性、模型、K 和来源保留比例。',
              '- answer_parts.csv / .json：逐固定要点评价、来源及原报告片段定位。',
              '- costs.csv：调用、token、归属时间、新支出及复用时间。', '',
              '原始报告、原文、日志保持在原目录；本轮原响应、调用日志、匿名映射及实际发送材料保存于本输出目录。'
              'ZIP 包含四张底表、panel 数据、审查记录、原始评价 JSON、匿名映射和新代码；'
              '不复制全量旧论文原文或事件日志。', '',
              '```bash', '/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.aggregate',
              '/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.render',
              '/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig5_four_panel.export', '```', '']
    (OUT / '数据审查与结果说明.md').write_text('\n'.join(notes))
    caption = 'Fig. 5 | Robustness, evidence boundaries and computational cost.\nThe full system uses GPT-6.1 Sol, all available historical-source passages and at most ten eligible semantic neighbors. The reduced-neighbor condition retains at most five neighbors; the reduced-evidence condition retains approximately half of the deduplicated historical sources; the model-replacement condition uses GPT-5.6 Luna. Original reports and corrected evaluations are reused without new model calls for this visual revision.\n(a) Capability profiles of the existing full-system evaluation across research domains and readable independent historical-source strata (at most 16 versus more than 16, determined by the 100-paper median). Points are paper-mean grounded correct coverage; categorical connecting lines aid profile comparison and do not represent temporal change. Both profile plots use the displayed 85–102% vertical scale. Legend counts are valid papers; citation-contact task counts are shown separately. These are overlapping partitions of the same cohort, including five development papers. One paper has technically incomplete evaluation for the other four tasks. These observations do not establish held-out-domain generalization.\n(b) One thin path per paper connects the five-neighbor and GPT-5.6 Luna conditions, both expressed as grounded-coverage changes relative to the full system. Diamonds and whiskers are paper-mean paired differences and 95% percentile intervals from 10,000 paper-paired bootstrap samples. Tasks and fixed applicability are unchanged; reduced-neighbor answers are judged against their actual graph. No equivalence threshold is assumed.\n(c) Observed paired answer-part transitions when approximately half the historical sources are retained, separated by evidence sufficiency determined before inspecting corrective evaluations. Ribbon widths represent answer-part proportions within each group, with the same part identities at both endpoints. Among 38 still-answerable parts from 18 papers, 35 stay grounded, one changes from grounded to incomplete, and two remain incomplete. Among 21 parts with lost support from 12 papers, 18 change from grounded answers to explicit abstention and three to incomplete responses. No confirmed unsupported assertions or genuine evaluator-unresolved states occur in this displayed historical subset; they are retained in the source table and assessed separately across all tasks. Explicit abstention is not automatically reasonable. The unavailable reduced-evidence report is excluded without replacement. Material-insufficiency labels originally called unresolved were normalized from candidate-free reference reasons, retaining the originals.\n(d) Four observed configurations on a common 19-paper cohort. Grounded answer yield is the paper-mean proportion of all originally applicable parts answered correctly with adequate evidence and scope; lost-evidence parts stay in the denominator. Horizontal positions are medians of within-paper cumulative invocation-time ratios to the full system. Arrows show configuration displacements from the full system, not a fitted frontier. Configuration cards state answer yield, reasonable targeted abstention, confirmed unsupported assertions and time ratios in full. Costs cover fixed-input branch analysis and report writing, including reused GEAR analysis for the five-neighbor condition; retrieval, embedding and graph construction are excluded. Invocation time is not wall-clock latency.\nPanels b–d use corrected model-assisted evaluation aligned with the accepted joint-structure and structural-resolution criteria. Technical missingness is not scored as scientific failure, and scientific unresolved labels remain explicit. The references and evaluations are not human gold labels. Five repeated runs remain supplementary data. Visual styling follows Fig.1–4 reference artwork, with serif headings, Arial labels and light-blue panel cards; methodological footer notes are provided here rather than below the plots.\n'
    (OUT / 'caption.txt').write_text(caption)
    target = OUT / 'Fig5_data_and_figure.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix in ('.csv', '.json', '.md', '.txt', '.png', '.svg', '.pdf'):
                archive.write(path, 'Fig5/' + path.name)
        for directory in ('evaluation', 'aligned', 'mapping'):
            for path in (OUT / directory).rglob('*.json'):
                archive.write(path, 'Fig5/' + str(path.relative_to(OUT)))
        for path in Path(__file__).parent.glob('*.py'):
            archive.write(path, 'code/' + path.name)
    print(target)


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from figure_pipeline.fig3_revision.storage import read
from figure_pipeline.fig4_rerun.config import CONDITIONS, NAMES

from .models import ASPECTS, Config


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def coverage_table(summary: list[dict[str, str]], report_set: str, field: str) -> list[str]:
    index = {(r['condition'], r['aspect']): r for r in summary if r['report_set'] == report_set}
    lines = ['|配置|' + '|'.join(ASPECTS.values()) + '|', '|---|' + '---:|' * len(ASPECTS)]
    for condition in CONDITIONS:
        values = []
        for aspect in ASPECTS:
            row = index.get((condition, aspect), {})
            value = row.get(field)
            values.append(f'{float(value):.1%}（n={row["complete_papers"]}）' if value else 'NA')
        lines.append('|' + NAMES[condition] + '|' + '|'.join(values) + '|')
    return lines


def readout(config: Config) -> None:
    summary = csv_rows(config.output / 'configuration_summary.csv')
    effects = csv_rows(config.output / 'paired_effects_summary.csv')
    state = read(config.output / 'run_summary.json')
    lengths = csv_rows(config.output / 'report_lengths.csv')
    report_counts = Counter(r['report_set'] for r in lengths)
    lines = ['# 五篇七配置探索：结果总表', '',
             '本表由逐要点评价本地汇总。选定科学问题的覆盖率不是整篇报告总质量；不合成总分。', '',
             '先按论文计算，再论文等权平均。n为该方面评价完整的论文数；原始分子、分母、未决及技术缺失见CSV。', '',
             '## 新解释与共同writer：内容覆盖率', '',
             *coverage_table(summary, 'new', 'content_coverage'), '',
             '## 新解释与共同writer：外部证据落实率', '',
             '要求内容正确、范围正确，且报告确实正确使用独立原文、原生图事实或基于可见图的推导。', '',
             *coverage_table(summary, 'new', 'grounded_coverage'), '',
             '## 原有35份报告：按同一最终参考重新评价', '',
             *coverage_table(summary, 'existing', 'content_coverage'), '',
             '两轮的模型与解释流程均有变化；新一轮还存在TDP-43完整Graph生成受限造成的缺失。不能用两张表的差直接声称单一因素的因果提升。', '',
             '## 新一轮完整配对：加入分支与三个图组件', '',
             '|比较|评价方面|内容差值（百分点）|证据落实差值（百分点）|配对n|提高／持平／降低论文数|',
             '|---|---|---:|---:|---:|---|']
    for row in effects:
        if row['report_set'] != 'new':
            continue
        def percentage(value: str) -> str:
            return f'{100 * float(value):+.1f}' if value else 'NA'
        counts = '/'.join(row[k] for k in ('content_improved_papers', 'content_tied_papers', 'content_reduced_papers'))
        lines.append(f'|{row["contrast"]}|{row["aspect_name"]}|{percentage(row["content_delta"])}|'
                     f'{percentage(row["grounded_delta"])}|{row["paired_papers"]}|{counts}|')
    lines += ['', '## 完成与缺失', '',
              f'- 依据原文和图筛选：{state["screened_papers"]}篇；保留{len(state["included_papers"])}篇，图无用排除{len(state["excluded_papers"])}篇。',
              f'- 可用旧报告：{report_counts["existing"]}份；新增完成报告：{report_counts["new"]}份。',
              f'- 新增请求启动记录：{state["calls"]}次；{state["calls_with_unknown_usage"]}次未返回用量，未将其当成零消耗。',
              '- TDP-43新增完整Graph解释受到服务端内容限制，未通过其他模型或通道绕过；新增仅Graph、完整系统两份报告缺失。其原有七报告仍纳入旧报告评价。',
              '- TDP-43和THz没有正向引用接触，路径方面不适用；不会把它们填成零。',
              '- 丝素H1与保留的H2重复，去重后不入分母；Cx46/50 H2b需未提供的历史原文，在首次评价前排除。',
              '- 参考开发中的修正、失败和恢复均保留。没有独立人类专家复标，不能宣称盲测基准或人类认可。',
              '- 没有启动95篇，没有绘图，没有编写或运行回归测试。', '',
              '完整流程和变更理由见《实验口径.md》；具体原句见answer_parts.json与paired_changes.json。']
    probe_path = config.output / 'joint_probe.json'
    if probe_path.exists():
        probe = read(probe_path)
        lines += ['', '## THz定点诊断：与主比较分开', '',
                  '用各自原有实际输入，向完整系统和移除联合图提出相同的两个明确问题；未提供参考答案。', '',
                  '|配置|正确且范围正确的答案要点|', '|---|---:|']
        for candidate in probe['evaluation']['candidates']:
            parts = [p for q in candidate['questions'] for p in q['parts']]
            correct = sum(p['status'] == 'correct' and p['scope_correct'] is True for p in parts)
            condition = probe['mapping'][candidate['report_id']]
            lines.append(f'|{NAMES[condition]}|{correct}/{len(parts)}|')
        lines += ['', '所问跨邻域边也出现在第五条贡献的单贡献图卡中，移除联合图仍可重建。'
                  '两份普通报告在这些预选要点上均为0/4；明确提问后均为4/4。'
                  '这揭示任务指向与输入冗余，不能解释为联合图没有其他作用。',
                  '本诊断增加两次作答、一次匿名评价，未计入上面的七配置主结果。']
    (config.output / '结果总表.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

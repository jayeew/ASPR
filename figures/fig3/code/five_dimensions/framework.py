"""Source-bound reviewer rationale for the five-dimension framework.

Reads pooled coding and writes only the new framework directory. No model calls.
"""

from __future__ import annotations
from figure_pipeline.paths import PROJECT_ROOT

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = PROJECT_ROOT
SOURCE = ROOT / 'outputs/reviewer_open_coding_pilot_20261116/open_discovery'
DEFAULT_OUTPUT = ROOT / 'outputs/innovation_five_dimensions/framework'
THEMES = ('T01', 'T02', 'T03', 'T04', 'T06', 'T05', 'T46', 'T12')
DIMENSIONS = [
    ('historical_originality', '历史原创性', ['T01'], '直接历史比较依据'),
    ('understanding_capability', '认识与能力推进', ['T02', 'T03', 'T04'], '不同贡献形式的概念归并'),
    ('knowledge_organisation', '知识组织与整合', ['T06'], '审稿中的整合关注＋本文Graph表征扩展'),
    ('evidential_basis', '证据基础', ['T46', 'T12'], '从研究质量关注纳入的创新判断证成要求'),
    ('scientific_value', '科学价值', ['T05'], '问题重要性与实际贡献价值的联系'),
]
EXAMPLES = [
    ('T01', 'P002', None,
     'The paper seemingly overclaims novelty: SAT was not invented by the authors, but rather was previously reported in references [42] and [50].',
     '技术本身已有前作，需要确定具体新增对象；不能把已有方法整体说成首次。'),
    ('T02', 'P005', None,
     'These important and very novel data go some way to clarify previous and conflicting evidence',
     '句中连续片段：审稿者评价数据有助于澄清此前冲突，支持将认识推进作为分析内容。'),
    ('T03', 'P190', None,
     'Supplementary Table 4 compares reconfigurable devices in the literature and shows that only this work offers all the functions in a single device.',
     '评价对象是新增功能组合，而不只是单个性能指标；本分析记录该评审判断，不独立确证首次性。'),
    ('T04', 'P781', None,
     'Its significance lies not in conceptual novelty but in the generation of a high-quality, multi-tissue dataset for a non-model species.',
     '资源填补覆盖空白可以有贡献，不要求所有论文都以概念突破为形式。'),
    ('T06', 'P002', None,
     'I appreciate the insight that the authors establish a connection between sharp minima in the loss landscape and the poor robustness observed in physical systems.',
     '有具体科学内容的联系受到肯定；不能从这句直接推出图指标能够衡量创新。'),
    ('T05', 'P005', None,
     'Pulmonary microvascular thrombosis is a well-known complication of sickle cell disease (SCD) and is associated with significant morbidity and mortality but it is not clear why only a sub-set of patients develop pulmonary thrombosis evidenced by autopsy and CT studies.',
     '提出问题的重要性和未解决之处；还需结合论文实际贡献，不能直接由疾病严重推出成果重大。'),
    ('T46', 'P003', None,
     'might be somewhat too definitive. Two points could warrant consideration:',
     '连续片段：审稿者指出关于GTP状态抑制剂缺乏有意义差异的结论过于绝对，随后引用作者数据及其他情形说明理由。'),
    ('T12', 'P003', None,
     'However, it is also possible that growth factors drive drug resistance through other pathways including the PI3K/Akt pathway.',
     '竞争解释意味着现有证据尚不能专一归因；这里提供的是证据基础的核验要求，不是另一项新颖性标准。'),
]
BOUNDARY_REPLACEMENTS = [
    ('T03', 'P002', 'P002_R3_3',
     'Absence of full neural network functionality: Despite being framed as work on “physical neural networks,” the experimental demonstrations are limited to simple linear functions. No physical implementation of nonlinear activation or full inference capability is presented.',
     '原选择片段主要询问实验控制可行性；同篇评审者3明确讨论所声称完整能力与实际展示的差距，因此保留论文级T03命中，改用此处解释其主题依据。'),
    ('T06', 'P007', 'P007_R3_3',
     'While we appreciate the authors’ attempt to link physical disturbances to biogeochemical processes, we recommend revising this section to emphasise the need for future studies to better constrain the nature and extent of such effects in sandy environments.',
     '原选择片段主要赞许多方法形成有力证据；同篇评审者3直接讨论物理扰动与生物地球化学过程之间的联系及其不足。主题计数保留，不能将其解释为整合创新已获认可。'),
    ('T06', 'P085', 'P085_R3_2',
     'Based on this relationship, they show that one network can predict general trends in the other. Finally, they devise a means to identify trends in connectivity across genetic interaction and protein-protein interaction networks.',
     '原选择片段主要质疑网络比较有效性；同篇评审者3明确解释跨网络联系所增加的预测认识，故以此补充主题证据，论文计数不变。'),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _locate(source: str, quote: str) -> tuple[int, int]:
    """Allow whitespace differences only; copy actual source characters."""
    pattern = r'\s+'.join(re.escape(word) for word in quote.split())
    match = re.search(pattern, source)
    if match is None:
        raise ValueError(f'Quote not in source: {quote[:80]}')
    return match.start(), match.end()


def _example(record: tuple, evidence: dict, reports: dict) -> dict[str, Any]:
    theme, paper, block_id, quote, meaning = record
    original = evidence[(paper, theme)]
    block_id = block_id or original['source_block_id']
    report = reports[paper]
    block = next(b for b in report['blocks'] if b['block_id'] == block_id)
    source_path = Path(original['source_path'])
    source_text = source_path.read_text(encoding='utf-8')
    start, end = _locate(source_text, quote)
    block_start, block_end = _locate(block['text'], quote)
    context_start = max(0, start - 350)
    context_end = min(len(source_text), end + 500)
    return {
        'theme_id': theme, 'pilot_id': paper, 'paper_id': report['paper_id'],
        'journal': report['journal'], 'field': report['field'],
        'reviewer_id': block['reviewer_id'], 'source_block_id': block_id,
        'displayed_quote': quote, 'exact_source_quote': source_text[start:end],
        'source_path': str(source_path), 'source_start_char': start, 'source_end_char': end,
        'block_start_char': block_start, 'block_end_char': block_end,
        'context': source_text[context_start:context_end],
        'context_start_char': context_start, 'context_end_char': context_end,
        'interpretation_zh': meaning,
        'verified_source_and_eligible_review_block': True,
        'normalisation': 'Whitespace only; no silent word or punctuation changes.',
        'original_record_id': original['record_id'],
    }


def _statistics() -> list[dict[str, Any]]:
    matrix = _read_csv(SOURCE / 'paper_theme_matrix.csv')
    stats = {r['theme_id']: r for r in _read_csv(SOURCE / 'theme_statistics.csv')}
    assert len(matrix) == len({r['pilot_id'] for r in matrix}) == 994
    rows = []
    for theme in THEMES:
        count = sum(int(row[theme]) for row in matrix)
        assert count == int(stats[theme]['present_papers'])
        dimension = next(d for d in DIMENSIONS if theme in d[2])
        rows.append({
            'theme_id': theme, 'theme_name_zh': stats[theme]['name_zh'],
            'eligible_papers': 994, 'present_papers': count,
            'percentage': 100 * count / 994, 'count_adjustment': 0,
            'dimension': dimension[0], 'dimension_name_zh': dimension[1],
            'statistic_meaning': '论文级相关关注点出现率，包括肯定、质疑、批评与要求；非维度认可率',
        })
    return rows


def _render(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    for path in ('/mnt/c/Windows/Fonts/arial.ttf', '/mnt/c/Windows/Fonts/arialbd.ttf'):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'historical_originality': '#2767B0', 'understanding_capability': '#2C8B82',
              'knowledge_organisation': '#7755A1', 'evidential_basis': '#BE783D',
              'scientific_value': '#475C76'}
    labels = ['Substantive difference from prior work', 'Advance in explanation',
              'New observable or operational capability', 'Reusable resources and coverage gaps',
              'Understanding through integration', 'Important problems and unmet needs',
              'Calibration of claims to evidence', 'Confounding and specific attribution']
    fig = plt.figure(figsize=(15.4, 8.6), facecolor='white')
    fig.text(.035, .955, 'Extended Data Fig. 6 | Reviewer concerns informing the five-dimensional framework',
             fontsize=17, weight='bold', va='top', color='#10213B')
    ax = fig.add_axes([.295, .215, .245, .60])
    ax.barh(range(8), [r['percentage'] for r in rows], height=.60,
            color=[colors[r['dimension']] for r in rows])
    ax.set_yticks(range(8), [f"{r['theme_id']}  {label}" for r, label in zip(rows, labels)])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Papers mentioning the concern (%)', labelpad=10)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.grid(True, color='#E4E9EF', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=8, labelsize=10)
    for i, row in enumerate(rows):
        ax.text(row['percentage'] + 1.2, i, f"{row['present_papers']}  ({row['percentage']:.1f}%)",
                va='center', fontsize=9, color='#22334C')
    fig.text(.035, .865, 'a', fontsize=18, weight='bold')
    fig.text(.058, .865, 'Related concerns in 994 papers', fontsize=13, weight='bold')
    bx = fig.add_axes([.615, .18, .36, .67])
    bx.set_axis_off()
    cards = [
        ('historical_originality', 'Historical originality', 'T01  ·  What is new relative to prior work?', 'Review-derived core question'),
        ('understanding_capability', 'Advances in understanding and capability', 'T02 + T03 + T04  ·  What has advanced?', 'Conceptual grouping of contribution forms'),
        ('knowledge_organisation', 'Knowledge organisation and integration', 'T06  ·  What does connecting knowledge add?', 'Integration concerns + Graph extension'),
        ('evidential_basis', 'Evidential basis', 'T46 + T12  ·  What supports the contribution?', 'Justification requirement from quality concerns'),
        ('scientific_value', 'Scientific value', 'T05  ·  Why does the contribution matter?', 'Review-derived value question'),
    ]
    for i, (key, title, line1, line2) in enumerate(cards):
        y = .83 - i * .19
        color = colors[key]
        patch = FancyBboxPatch((.015, y), .965, .15, boxstyle='round,pad=0.01,rounding_size=0.012',
                              ec=color, fc=matplotlib.colors.to_rgba(color, .065), lw=1)
        bx.add_patch(patch)
        bx.text(.045, y + .117, title, weight='bold', fontsize=12, color=color, va='center')
        bx.text(.045, y + .074, line1, fontsize=10.2, color='#24354C', va='center')
        bx.text(.045, y + .031, line2, fontsize=9.5, color='#526479', va='center')
    fig.text(.615, .865, 'b', fontsize=18, weight='bold')
    fig.text(.638, .865, 'From concerns to analysis dimensions', fontsize=13, weight='bold')
    fig.text(.035, .10,
             'Counts are paper-level, non-exclusive theme occurrences, including praise, criticism and requests; they are not dimension-endorsement rates.',
             fontsize=10, color='#46556B')
    fig.text(.035, .071,
             'Graph perturbations are this study\'s methodological extension. The mapping does not establish independent factors or a summed novelty score.',
             fontsize=10, color='#46556B')
    fig.text(.035, .042,
             'Source: pooled substantive reviewer material from six Nature Portfolio journals; no Nature main-journal papers. Selected themes are not an exhaustive ranking.',
             fontsize=9.4, color='#65748A')
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(output / f'ExtendedDataFig06.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


def _write_documents(rows: list[dict], examples: list[dict], audit: list[dict], output: Path) -> None:
    table = '| 编码 | 相关关注点 | 篇数 | 出现率 | 对应维度 |\n|---|---|---:|---:|---|\n'
    table += '\n'.join(f"| {r['theme_id']} | {r['theme_name_zh']} | {r['present_papers']} | {r['percentage']:.1f}% | {r['dimension_name_zh']} |" for r in rows)
    texts = []
    for item in examples:
        texts.append(f"### {item['theme_id']} · {item['pilot_id']} · 评审者{item['reviewer_id']}\n\n"
                     f"> {item['displayed_quote']}\n\n{item['interpretation_zh']}\n\n"
                     f"[原文]({item['source_path']})；原文块 `{item['source_block_id']}`；字符位置 {item['source_start_char']}–{item['source_end_char']}。")
    report = f'''# 五维创新分析框架的真人审稿依据

## 定位与来源

本规范为已确定的五维创新分析框架提供可追溯的经验依据。统一使用994篇具有实质性真人审稿材料的论文，不按材料收集阶段分组。每篇论文对每个主题最多计一次；主题可以共现，比例不相加。来源为Nature Portfolio六本期刊，不含Nature主刊。

五维分析回答目标论文的创新是什么；后续0–3级评分回答某种方法把这些问题分析得是否准确、完整、有据。两者不是同一个评分对象，不生成论文创新总分。

## 五维来源与分析问题

| 维度 | 核心问题 | 来源与扩展 |
|---|---|---|
| 历史原创性 | 相对已有研究，哪些内容是新的？ | T01提供直接历史比较依据 |
| 认识与能力推进 | 多理解了什么，或能做什么此前难以做到的事情？ | T02、T03、T04分别体现认识、能力和资源贡献；这是不同贡献形式的概念归并 |
| 知识组织与整合 | 联系、修订或重组了哪些知识，组合增加了什么？ | T06支持科学整合的关注；用知识图谱扰动表征它是本文的方法扩展，不是原评审共同使用的指标 |
| 证据基础 | 哪些证据支持贡献，支持到什么程度和范围？ | T46、T12来自研究质量要求，作为创新判断能否成立的证成维度纳入 |
| 科学价值 | 推进了什么重要问题，为什么值得重视？ | T05提供问题重要性和未满足需求的依据；仍需说明论文实际贡献如何推进该问题 |

证据基础在五维报告中评价论文的证据；独立核验则检查报告对这些证据及其他维度的判断是否正确。证据较多或较强不能直接推出创新程度较高。

## 选定主题的统一统计

{table}

八项按与五维问题的直接关联选取，兼顾不同贡献形式，不是全部100项的频率前八名。低频资源主题保留是为了避免将概念或机制突破当作唯一创新形式。频数包含肯定、质疑、批评和补充要求，表示相关问题被讨论，而非创新获认可或维度获验证。

本次针对已有边界问题核查了P002/T03、P007/T06、P085/T06。同篇其他合格审稿段落仍可支持这些相关主题，故更新派生证据指向而不扣减论文计数；八项频数与原统一矩阵一致。修正只写入本目录的审计账本，原始编码和旧统计未变。

## 典型原文

{chr(10).join(texts)}

## 框架的适用表述

可以表述为：从真人审稿材料的主要关注点提炼创新评价问题，并结合Graph对知识联系与组织方式的表征，形成五维创新分析框架。不能表述为：五个相互独立的评价因子已由994篇真人评审验证，或图谱扰动强度已被证明等同于创新程度。

本次不重做全语料开放编码，也不将图的科学整合维度与实验泛化范围混为一谈。原五角度规范保留为历史方案，当前五维规范与正式报告/核验接口对齐。

## 文件

- `theme_statistics.csv`：八项论文级频数与五维对应。
- `dimension_mapping.json`：五维问题的来源属性。
- `reviewer_examples.json`：原文、来源位置及解释。
- `boundary_audit.json`：三个边界案例及替代证据。
- `validation.json`：计数、引文及原始输入未变核验。
- `ExtendedDataFig06.svg/pdf/png`：框架来源图。
- `figure_legend_zh.md`：可用于稿件的中文图注。
'''
    (output / 'framework_basis_zh.md').write_text(report, encoding='utf-8')
    (output / 'figure_legend_zh.md').write_text(
        '**Extended Data Fig. 6 | 真人审稿关注点与五维创新分析框架的来源。**\n\n'
        '**a，**994篇具有实质性真人审稿材料的论文中，八项选定关注点的出现篇数与比例。'
        '每篇论文对每项主题最多计一次；不同主题可共现，分母均为994。'
        '主题包含肯定、质疑、批评和补充要求，其频率不是维度认可率或创新得分。'
        '**b，**关注点与历史原创性、认识与能力推进、知识组织与整合、证据基础和科学价值的对应。'
        '知识组织与整合以评审中的科学整合关注为基础，Graph扰动表征为本文的方法扩展；'
        '证据基础承接研究质量中的证据要求，考察论文所声称贡献的实验支持及成立范围。'
        '图中主题按框架关联选择，不代表全部关注点的频率排名，也不构成五个独立因子的统计验证。'
        '样本来自Nature Portfolio六本期刊，不含Nature主刊。\n', encoding='utf-8')
    lines = ['# 边界核查记录', '', '本轮仅修正派生证据定位，八项论文级频数不变。', '']
    for item in audit:
        lines.extend([f"## {item['pilot_id']} / {item['theme_id']}", '', item['reason_zh'], '',
                      '> ' + item['replacement_evidence']['displayed_quote'], ''])
    (output / 'boundary_audit_zh.md').write_text('\n'.join(lines), encoding='utf-8')


def build_framework(output_dir: Path | None = None) -> dict[str, Any]:
    output = output_dir or DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    input_names = ['theme_statistics.csv', 'paper_theme_matrix.csv', 'theme_evidence.jsonl', 'raw_reports.jsonl']
    hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in input_names}
    evidence = {}
    for line in (SOURCE / 'theme_evidence.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('included') and row['kind'] == 'match' and row['theme_id'] in THEMES:
            evidence[(row['pilot_id'], row['theme_id'])] = row
    reports = {r['pilot_id']: r for r in map(json.loads, (SOURCE / 'raw_reports.jsonl').read_text().splitlines())}
    rows = _statistics()
    examples = [_example(r, evidence, reports) for r in EXAMPLES]
    audit = []
    for replacement in BOUNDARY_REPLACEMENTS:
        item = _example(replacement, evidence, reports)
        original = evidence[(replacement[1], replacement[0])]
        audit.append({'pilot_id': replacement[1], 'theme_id': replacement[0], 'action': 'replace_illustrative_evidence_only',
                      'reason_zh': replacement[4], 'count_before': 1, 'count_after': 1,
                      'original_evidence': original, 'replacement_evidence': item})
    with (output / 'theme_statistics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / 'dimension_mapping.json', {
        'eligible_papers': 994, 'unit': 'paper-level non-exclusive concern occurrence',
        'dimensions': [{'dimension': d[0], 'name_zh': d[1], 'themes': d[2], 'basis_zh': d[3]} for d in DIMENSIONS],
        'graph_status': 'Methodological extension; not a directly observed reviewer metric.',
        'not_claimed': ['factor independence', 'dimension endorsement rates', 'summed innovation scores'],
    })
    _write_json(output / 'reviewer_examples.json', examples)
    _write_json(output / 'boundary_audit.json', audit)
    _write_documents(rows, examples, audit, output)
    _render(rows, output)
    assert hashes == {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in input_names}
    validation = {'status': 'passed', 'eligible_papers': 994, 'selected_themes': 8,
                  'counts_match_original_paper_matrix': True, 'count_changes': 0,
                  'verified_primary_examples': len(examples), 'verified_boundary_replacements': len(audit),
                  'source_and_eligible_review_block_checks': True, 'original_inputs_unchanged': hashes,
                  'model_calls': 0, 'external_searches': 0}
    _write_json(output / 'validation.json', validation)
    return validation


if __name__ == '__main__':
    print(json.dumps(build_framework(), ensure_ascii=False))

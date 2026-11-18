"""Conceptual consolidation and pooled paper-level reporting; no model calls."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def category_stat(category: dict, rows: list[dict]) -> dict:
    n = sum(any(row[t] == '1' for t in category['theme_ids']) for row in rows)
    return dict(category_id=category['category_id'], name_zh=category['name_zh'],
                role=category['role'], question=category['question'],
                constituent_themes=len(category['theme_ids']), present_papers=n,
                denominator=len(rows), percentage=100 * n / len(rows))


def write_mapping(book: dict, themes: dict) -> None:
    rows = []
    text = ['# 七类理论归并与具体关注点映射', '',
            '统计总体为994篇有效论文。每个具体关注点指定一个主要归属，以保持汇总规则可追溯；跨类关系保留在解释栏，不自动重复计入创新评价。', '',
            '这里的七类是按评价对象和判断功能进行的概念归并，不是统计聚类得到的独立因子。', '']
    for c in book['categories']:
        text += [f"## {c['category_id']} {c['name_zh']}", '', c['question'], '',
                 c['scope'], '', '**边界：**' + c['boundary'], '',
                 '| 具体关注点 | 原始问题 | 跨类解释条件 |', '|---|---|---|']
        for tid in c['theme_ids']:
            t = themes[tid]
            bridge = book['conditional_links'].get(tid, '')
            rows.append(dict(theme_id=tid, name_zh=t['name_zh'], category_id=c['category_id'],
                             category_name=c['name_zh'], role=c['role'], question=t['question'],
                             conditional_link=bridge))
            text.append(f"| {tid} {t['name_zh']} | {t['question']} | {bridge or '按主要评价对象归类'} |")
        text.append('')
    write_csv(HERE / 'theme_to_theoretical_category.csv', rows)
    (HERE / 'theoretical_framework_zh.md').write_text('\n'.join(text))


def write_report(book: dict, stats: list[dict], role_stats: list[dict]) -> None:
    n = stats[0]['denominator']
    lines = ['# 真人审稿关注点：创新评价、研究质量与评价可靠性', '',
             f'以{n}篇具有实质性真人审稿材料的论文为统一分析总体，将100个具体关注点归并为7个理论大类。全文和统计表均使用同一总体。', '',
             '**核心区分：原创性与意义评价“贡献是什么、为什么值得”；研究质量评价“结论是否成立、在何处成立、能否核查”；研究责任评价“研究是否正当”；评价可靠性关注“谁依据什么条件作出判断”。**', '',
             '## 七个大类及总体分布', '',
             '| 大类 | 核心问题 | 主要性质 | 涉及论文 | 占比 |', '|---|---|---|---:|---:|']
    for c, s in zip(book['categories'], stats, strict=True):
        lines.append(f"| {c['name_zh']} | {c['question']} | {c['role']} | {s['present_papers']} | {s['percentage']:.1f}% |")
    lines += ['', '同篇论文的多个审稿人、多个具体关注点涉及同一大类，只计1篇；不同大类可以共现，比例不相加为100%。肯定、质疑及补证要求均表示该问题被讨论，不表示研究通过该项评价。大类范围不同，频率不能直接作为重要性权重。', '',
              '## 这七类为什么具有不同的理论功能', '']
    for c in book['categories']:
        lines += [f"### {c['category_id']} {c['name_zh']}", '', c['scope'], '', c['boundary'], '']
    lines += ['## 哪些真正属于创新评价', '',
              '**前两类构成创新评价的核心：原创性回答“新在哪里、以何种形式推进认识”；贡献意义回答“这种推进为什么值得认可”。** 二者需要同时讨论，却不能相互替代。研究可以新而意义有限，也可以通过确认、资源建设或综合形成重要贡献，而不以颠覆既有预期为必要条件。', '',
              '结论有效性、稳健性与适用性、透明性与可核查性，首先是研究质量要求。它们决定一项贡献能否成立、可信到何种程度，并不直接等于创新程度。研究责任另列，是因为伦理、福利、隐私和诚信既不能折算成新颖程度，也不能只被当作实验技术质量。', '',
              '部分具体问题会连接研究质量与创新评价，需要看审稿人实际在判断什么：', '',
              '| 具体问题 | 仅属于质量要求时 | 同时参与创新评价时 |', '|---|---|---|',
              '| 机制或理论解释 | 解释是否合理、证据是否足够 | 是否比已有解释推进认识、改变概念理解 |',
              '| 性能、效应与组件增益 | 结果准确吗、性能够用吗、新组件有作用吗 | 相对已有方案是否产生实质新能力或值得的新增收益 |',
              '| 数据或工具资源 | 能否下载、运行、复现 | 是否填补知识覆盖空白并使新的研究成为可能 |',
              '| 文献与历史比较 | 引用是否准确、完整、公平 | 所声称发现的先后关系、原创归属和新增内容是否成立 |',
              '| 基础理解与应用价值 | 某种应用尚不能实施 | 基础认识是否已经构成独立且重要的贡献 |', '',
              '当前统计按每个具体关注点的主要含义归类，跨类解释只作为阅读规则；不把所有性能、机制或透明性意见自动增加到创新评价的篇数。', '',
              '## 研究可靠性与评价可靠性不能混淆', '',
              '研究可靠性关注被评价对象：测量是否可靠、统计是否适当、结论是否可复现。评价可靠性关注评价行为：评价者是否具备相关专长、实际检查了什么、标准是否适合研究类型、是否将声望或期刊偏好混入判断。', '',
              '例如，“训练标签是否正确”首先是研究验证的质量问题；“评审是否有能力核验该模型”才是评价者判断的依据问题。“作者按要求修改了某段文字”也不意味着其创新性已经得到证实。', '',
              '第七类的频数表示评审文本涉及这些评价条件，不是已测得评价可靠率。真正测量评价可靠性，还需评价者一致性、判断稳定性或对证据变化的响应等专门检验；本次归并不将关注点出现率替代这些指标。', '',
              '## 典型原文及其归属', '']
    examples = json.loads((HERE / 'report_examples.json').read_text())
    owners = {t: c for c in book['categories'] for t in c['theme_ids']}
    comments = {
        'P781': '资源型贡献可以独立于概念新颖性被评价；原文明确将研究价值放在高质量、可复用的数据资源上。',
        'P085': '确认预期内关系仍可能有贡献，因此“意外”不应成为认可研究价值的唯一标准。',
        'P591': '审稿人识别出作者忽视的负结果价值，这是贡献意义的判断，不等于要求更多阳性证据。',
        'P766': '检查模型解释是否对应实际特征使用，首先验证解释可信性，不能直接证明科学创新。',
        'P661': '镇痛和福利要求属于研究责任，不能作为创新高低的维度。',
        'P940': '这是审稿人对发表激励影响证据呈现的反思，属于评价环境；记录该观点不等于已经验证这种因果机制。',
    }
    for example in examples:
        ev = example['evidence']; pid = example['pilot_id']; owner = owners[example['theme_id']]
        norm = lambda value: re.sub(r'\s+', ' ', value).strip()
        assert norm(example['displayed_quote']) in norm(ev['quote'])
        lines += [f"**{owner['name_zh']} · {pid} / 评审者{ev['reviewer_id']}**", '',
                  '> ' + example['displayed_quote'], '', comments[pid] +
                  f" [原文]({ev['source_path']})。", '']
    lines += ['## 对论文论证的建议', '',
              '论文可以以“原创性—贡献意义”为创新评价的核心，将有效性、适用性和可核查性作为支持判断的证据条件；研究责任以及评价者的标准、能力和偏差分别交代。这样既能讨论什么是有价值的创新，也能解释为什么一个创新判断值得相信。', '',
              '这一框架没有把现有五个解释角度直接当成已获验证的五维量表。科学内容可以作为贡献描述，历史比较帮助识别原创性，证据基础支持判断可信度，科学意义对应贡献价值；图谱结构形式则可作为本文识别贡献的具体表征方式，不能仅凭这些审稿主题的频数宣称它已被验证为独立评价维度。', '',
              '## 数据与核查入口', '',
              '- [七类总体统计](open_discovery/theoretical_category_statistics.csv)',
              '- [100个关注点到七类的完整映射与边界](open_discovery/theoretical_framework_zh.md)',
              '- [具体关注点的总体统计](open_discovery/theme_statistics.csv)',
              '- [七类的期刊与领域分布](open_discovery/theoretical_category_by_group.csv)',
              '- [原文证据与定位](open_discovery/theme_evidence.jsonl)',
              '- [统计方法](open_discovery/protocol.md)', '',
              '样本名册为1,000篇，其中994篇纳入分析；3篇仅有结果性意见，3篇独立来源不可用。材料来自Nature Portfolio六本期刊，不含Nature主刊。归并是对真人审稿文本的模型辅助编码结果进行概念整理，不是统计因子分析或经过验证的独立维度量表。论文正文未修改。', '']
    report = '\n'.join(lines)
    assert not re.search(r'原98|追加896|original_98|extension_896|两次统计|原100篇|追加900', report)
    (BASE / 'report_zh.md').write_text(report)
    (HERE / 'findings_zh.md').write_text(report.split('## 这七类为什么', 1)[0] + '\n完整解释见[报告](../report_zh.md)。\n')
    (HERE / 'innovation_interpretation.md').write_text(report.split('## 哪些真正属于创新评价', 1)[1].split('## 典型原文及其归属', 1)[0].strip() + '\n')


def main() -> None:
    book = json.loads((HERE / 'theoretical_codebook.json').read_text())
    themes = {t['theme_id']: t for t in json.loads((HERE / 'complete_screening/theme_codebook.json').read_text())}
    matrix = read_csv(HERE / 'paper_theme_matrix.csv')
    overall = read_csv(HERE / 'theme_statistics.csv')
    assert len(matrix) == len({r['pilot_id'] for r in matrix}) == 994
    assert len(overall) == 100 and all('cohort' not in r for r in matrix + overall)
    assert all(r['status'] == 'completed' for r in matrix)
    assigned = [t for c in book['categories'] for t in c['theme_ids']]
    assert Counter(assigned) == Counter({t: 1 for t in themes})
    categories = book['categories']
    stats = [category_stat(c, matrix) for c in categories]
    write_csv(HERE / 'theoretical_category_statistics.csv', stats)
    write_csv(HERE / 'navigation_group_statistics.csv', stats)
    category_matrix = []
    for row in matrix:
        output = {k: row[k] for k in ['pilot_id', 'paper_id', 'journal', 'field']}
        output.update({c['category_id']: int(any(row[t] == '1' for t in c['theme_ids'])) for c in categories})
        category_matrix.append(output)
    write_csv(HERE / 'paper_theoretical_category_matrix.csv', category_matrix)
    role_stats = []
    for role in dict.fromkeys(c['role'] for c in categories):
        ids = [c['category_id'] for c in categories if c['role'] == role]
        n = sum(any(row[i] == 1 for i in ids) for row in category_matrix)
        role_stats.append(dict(role=role, present_papers=n, denominator=len(matrix), percentage=100*n/len(matrix)))
    write_csv(HERE / 'evaluation_role_statistics.csv', role_stats)
    grouped = []
    for dimension in ['journal', 'field']:
        for value in sorted({r[dimension] for r in matrix}):
            part = [r for r in matrix if r[dimension] == value]
            grouped.extend(dict(group_dimension=dimension, group_value=value, **category_stat(c, part)) for c in categories)
    write_csv(HERE / 'theoretical_category_by_group.csv', grouped)
    write_json(HERE / 'theme_navigation.json', {'groups': [dict(group_id=c['category_id'], name_zh=c['name_zh'],
               navigation_scope=c['scope'], theme_ids=c['theme_ids']) for c in categories]})
    write_json(HERE / 'distribution_summary.json', dict(papers=len(matrix), observed_themes=100,
               theoretical_categories=len(categories), reporting='pooled_only', category_statistics=stats))
    write_mapping(book, themes)
    write_report(book, stats, role_stats)
    # Recount from active, source-bound evidence rather than the matrix aggregation.
    owner = {tid: c['category_id'] for c in categories for tid in c['theme_ids']}
    evidence_sets = {c['category_id']: set() for c in categories}
    for line in (HERE / 'theme_evidence.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record['included'] and record['kind'] == 'match':
            evidence_sets[owner[record['theme_id']]].add(record['pilot_id'])
    for c, s in zip(categories, stats, strict=True):
        assert len(evidence_sets[c['category_id']]) == s['present_papers']
        assert evidence_sets[c['category_id']] == {r['pilot_id'] for r in category_matrix if r[c['category_id']] == 1}
    for dimension in ['journal', 'field']:
        for c in categories:
            parts = [r for r in grouped if r['group_dimension'] == dimension and r['category_id'] == c['category_id']]
            assert sum(r['denominator'] for r in parts) == 994
            assert sum(r['present_papers'] for r in parts) == len(evidence_sets[c['category_id']])
    write_json(HERE / 'theoretical_aggregation_audit.json', dict(status='passed', papers=994, themes=100, categories=7,
               counts_recomputed_from_active_source_evidence=True, mapping_exhaustive_and_unique=True,
               conditional_links_add_no_counts=True, group_partitions_verified=True,
               codebook_sha256=hashlib.sha256((HERE / 'theoretical_codebook.json').read_bytes()).hexdigest(),
               matrix_sha256=hashlib.sha256((HERE / 'paper_theme_matrix.csv').read_bytes()).hexdigest(),
               evidence_sha256=hashlib.sha256((HERE / 'theme_evidence.jsonl').read_bytes()).hexdigest()))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()

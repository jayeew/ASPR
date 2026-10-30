"""Validate the locked Nature roster and write the coverage report and repair queue."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq

from .nature_types import OUT
from .stage0 import write_csv, write_json


def read_rows(name: str) -> list[dict[str, Any]]:
    return pq.read_table(OUT / (name + '.parquet')).to_pylist()


def make_queue(rows: list[dict[str, Any]]) -> None:
    output = []
    for row in rows:
        if not row['missing_reasons']:
            continue
        authors = json.loads(row['authorships_json'])
        missing_authors = [{'author_id': a['author'].get('id'), 'author_name': a['author'].get('display_name'),
            'raw_affiliation_strings': a.get('raw_affiliation_strings', [])} for a in authors if not a.get('institutions')]
        country_gaps = {i['id']: i.get('display_name') for a in authors for i in a.get('institutions', [])
                        if not i.get('country_code')}
        output.append({'doi': row['doi'], 'work_id': row['work_id'], 'title': row['title'],
            'year': row['publisher_year'], 'article_url': row['article_url'],
            'needs_abstract': not row['has_abstract'], 'no_linked_institution': not row['has_institution'],
            'needs_author_institution_mapping': not row['all_observed_authors_have_institution'],
            'needs_institution_country': not row['complete_institution_country'],
            'missing_author_institutions': missing_authors, 'linked_institutions_without_country': country_gaps,
            'missing_reasons': row['missing_reasons'],
            'next_action': '核对出版社摘要和作者机构；保留原始署名与来源，再映射机构ID/类型/国家，不猜测高校归属'})
    write_csv(OUT / 'metadata_repair_queue.csv', output)


def type_tables(catalogue: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    counts = Counter((r['publisher_year'], r['publisher_type']) for r in catalogue)
    write_csv(OUT / 'publisher_types_by_year.csv', [{'publisher_type': kind,
        **{str(year): counts[(year, kind)] for year in [2023, 2024, 2025]},
        'total': sum(counts[(year, kind)] for year in [2023, 2024, 2025])}
        for kind in sorted({r['publisher_type'] for r in catalogue})])
    included = {r['doi'] for r in rows}
    write_csv(OUT / 'publisher_type_decisions.csv', [{**r, 'decision':
        'primary_article' if r['doi'] in included else
        'excluded_retracted_article' if r['publisher_type'] == 'Article' else
        'analysis_sensitivity' if r['publisher_type'] == 'Analysis' else
        'excluded_nonprimary_type'} for r in catalogue])


def validate(rows: list[dict[str, Any]], catalogue: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads((OUT / 'catalogue_first_pages.json').read_text())
    for year in [2023, 2024, 2025]:
        selected = [r for r in catalogue if r['publisher_year'] == year]
        assert len(selected) == expected[str(year)]['expected_count']
    assert len({r['doi'] for r in catalogue}) == len(catalogue)
    assert len({r['doi'] for r in rows}) == len(rows)
    assert len({r['work_id'] for r in rows}) == len(rows)
    assert all(r['publisher_type'] == 'Article' and r['include_primary'] for r in rows)
    assert all(r['metadata_match_status'] == 'unique_nature_doi_match' for r in rows)
    assert not any(r['is_retracted_openalex'] or r['publisher_retracted'] for r in rows)
    assert len(rows) == audit['locked_research_n']
    for key, count in audit['missing_list_sizes'].items():
        subset = read_rows(key)
        assert len(subset) == count
        assert {r['doi'] for r in subset}.issubset({r['doi'] for r in rows})
    assert len(read_rows('missing_abstract')) == len(rows) - sum(r['has_abstract'] for r in rows)
    assert len(read_rows('missing_all_institutions')) == len(rows) - sum(r['has_institution'] for r in rows)
    assert sum(audit['prior_reconciliation']['decisions'].values()) == 11089
    assert sum(r['research_n'] for r in audit['coverage'][1:]) == len(rows)
    assert len(read_rows('prior_candidates_unresolved')) == 0
    return {'passed': True, 'unique_dois': len(rows), 'unique_work_ids': len(rows),
            'catalogue_rows': len(catalogue), 'unresolved_prior_candidates': 0,
            'checked': ['annual catalogue counts', 'DOI and Work ID uniqueness', 'type and retraction eligibility',
                        'missing-list membership and counts', 'year and prior-candidate denominators']}


def coverage_table(coverage: list[dict[str, Any]]) -> str:
    lines = ['| 在线发表年份 | 主分析论文 | 有摘要 | 有至少一个机构ID | 每位已记录作者有机构ID |',
             '|---|---:|---:|---:|---:|']
    for row in coverage[1:] + coverage[:1]:
        values = [str(row['year'] or '合计'), str(row['research_n'])]
        for count, rate in [('abstract_n', 'abstract_coverage'), ('institution_n', 'institution_coverage'),
                            ('all_authors_institution_n', 'all_authors_institution_coverage')]:
            values.append(f"{row[count]:,}（{row[rate]:.2%}）")
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def analysis_review() -> None:
    evidence = json.loads((OUT / 'boundary_article_evidence.json').read_text())
    rows = [{'doi': r['doi'], 'title': r['title'], 'year': r['publisher_year'],
             'publisher_type': r['publisher_type'], 'decision': 'separate_analysis_sensitivity_cohort',
             'reason': '出版社 Analysis 类，摘要含新的定量分析/建模/证据综合结果；按类型独立保留，不归为新闻评论',
             'evidence_url': r['article_url'], 'publisher_abstract': r['publisher_abstract']}
            for r in evidence if r['publisher_type'] == 'Analysis']
    assert len(rows) == 13 and all(r['publisher_abstract'] for r in rows)
    write_csv(OUT / 'analysis_type_review.csv', rows)


def report(audit: dict[str, Any]) -> None:
    text = f'''# Nature 主刊 2023–2025 研究论文类型核查与补齐清单

核查批次：2026-10-21。按出版社在线发表日期纳入，按本次核查时点排除已撤稿研究。摘要和机构覆盖均为本次 OpenAlex API 元数据状态，不是本地 snapshot 的覆盖率，也不是出版社页面可获取率。

## 结论与固定分母

已逐页核对 Nature 三年的全部 664 页目录：2023 年 4,320 条、2024 年 4,400 条、2025 年 4,552 条，共 13,272 条；年度数与目录标示数一致，无重复 DOI、跨年重复或目录日期年份冲突。

主分析采用出版社 **Article** 类型：2023/2024/2025 分别为 1,150/1,239/1,302 篇，共 **3,691 篇**。其中 8 篇的撤稿状态由出版社撤稿通知与 OpenAlex 双重确认，分别排除 3/3/2 篇，最终主分析清单 **3,683 篇**。三年没有出版社 Letter 类型记录。News、News & Views、Comment、Editorial、Correspondence、Research Briefing、综述、更正、撤稿通知等不进入 Article 主分析。

额外 **13 篇 Analysis（1/6/6）** 全部保留为独立敏感性分析组。它们不是普通新闻评论，出版社定义允许对既有数据做新的分析；已逐篇检查摘要中的分析结果。Article + Analysis 扩展口径共 **3,696 篇**，不能把 3,683 误称为“所有含原创结果的类型总数”。Matters Arising 等针对既有论文的讨论、答复按本次“独立原创研究文章”口径单列排除，不判断其科学贡献为零。

类型依据：[Nature 研究文章格式说明](https://www.nature.com/nature/for-authors/formatting-guide)、[其他投稿类型说明](https://www.nature.com/nature/for-authors/other-subs)。在线日期来自[2023](https://www.nature.com/nature/articles?year=2023)、[2024](https://www.nature.com/nature/articles?year=2024)、[2025](https://www.nature.com/nature/articles?year=2025)年度目录；不按 DOI 中的年份或纸本卷期归年。

## 重新计算的覆盖率

{coverage_table(audit['coverage'])}

全部 3,691 篇 Article 都以标准 DOI 唯一匹配到 Nature Source `S137773608` 的 OpenAlex 记录，无缺失/多义 DOI 连接；主分析 3,683 篇的 DOI 和 Work ID 均唯一。这里“有机构”指至少一位作者具有至少一个 OpenAlex institution ID；“每位已记录作者有机构”也只表示每人至少匹配一个机构，**不等于已恢复全部多重署名**，更不等于都属于高校。

摘要与至少一个机构同时可用 **2,071 篇（56.23%）**；摘要与每位已记录作者至少一个机构同时可用 **1,803 篇（48.95%）**。每位已记录作者都有机构且所有已关联机构有 country_code 的论文 **2,562 篇（69.56%）**。本批未检出显式作者截断标记/计数差异；这不证明原作者名单绝无漏项。

Analysis 的 13 篇均有摘要和至少一个机构；扩展口径摘要覆盖为 2,086/3,696（56.44%），机构存在覆盖为 3,693/3,696（99.92%）。单独结果见 `article_plus_analysis_coverage_by_year.csv`。

## 需要补齐哪些论文

所有逐篇清单均含标题、DOI、Work ID、在线日期、出版社链接及缺失标志；CSV 为带 BOM 的 UTF-8，Parquet 还保存现有摘要和逐作者机构 JSON。

| 任务 | 论文数 | 逐篇文件 | 含义 |
|---|---:|---|---|
| 补摘要 | 1,610 | [missing_abstract.csv](missing_abstract.csv) | 2023/2024/2025 分别为 558/547/505；指 OpenAlex 无摘要 |
| 补全论文机构ID | 3 | [missing_all_institutions.csv](missing_all_institutions.csv) | 全体作者均无已关联机构ID，不代表出版社无署名 |
| 补逐作者机构映射 | 438 | [incomplete_author_institutions.csv](incomplete_author_institutions.csv) | 包括上述 3 篇和另外 435 篇部分作者未匹配 |
| 补机构/国家链路 | 1,121 | [incomplete_institution_country.csv](incomplete_institution_country.csv) | 包括 438 篇机构不完整，以及另外 683 篇已关联机构中存在国家缺失 |
| 任一缺失的去重总清单 | 2,237 | [metadata_repair_queue.csv](metadata_repair_queue.csv) | 逐作者未匹配姓名/原始署名、缺国家的机构ID均展开在列中 |

这些集合重叠，**不能相加**。若仅要求“有摘要 + 至少一个机构”，需补齐的去重论文数为 **1,612**；若用于严格的作者、国家、高校分数计数，应按 2,237 篇总队列核验。全部主分析论文均已有 primary topic。

完全未匹配机构ID的三篇已查到出版社署名，都是企业：

| 年份 | 论文 | DOI | 出版社署名机构 |
|---|---|---|---|
| 2023 | Illuminating protein space with a programmable generative model | [10.1038/s41586-023-06728-8](https://www.nature.com/articles/s41586-023-06728-8) | Generate Biomedicines, Somerville, MA, USA |
| 2024 | Room temperature operation of germanium–silicon single-photon avalanche diode | [10.1038/s41586-024-07076-x](https://www.nature.com/articles/s41586-024-07076-x) | Artilux Inc., Zhubei |
| 2024 | Optical clocks at sea | [10.1038/s41586-024-07225-2](https://www.nature.com/articles/s41586-024-07225-2) | Vector Atomic, Inc., Pleasanton, CA, USA |

原始署名字符串、作者对应关系和摘要已另存 `metadata_gap_spotcheck.json`，尚未写回 OpenAlex 覆盖率统计。补齐应先保留出版社机构名，再核实标准机构ID/类型/国家；找不到公共机构ID时保留本地来源可追溯的实体和 unknown 状态，不强行归入高校。抽查三篇均存在出版社摘要，这不代表其余 1,607 篇已完成摘要抓取。

## 旧 11,089 条候选如何变化

旧 API 候选与新目录逐条核对后：3,678 条进入主分析，7,391 条为不纳入的其他出版社类型，13 条 Analysis 单列；另有 5 条仓储版本与正式论文重复、1 条实际在线发表于 2026 年、1 条补充数据记录被误标为期刊文章。7 个 DOI/日期例外已用出版社与 DataCite 显式关系核实，无待判定记录。

再补回旧筛选漏掉的 5 篇出版社 Article，得到 **3,678 + 5 = 3,683**。漏项原因是 OpenAlex 的三条语言误标（nl/fr/lt）和两条类型误标（review/conference-abstract）；出版社页面均为英语 Article。详见 `research_added_beyond_prior_candidates.csv` 与 `metadata_gap_spotcheck.json`。例外中有一篇 Nature 2026 正式论文与 2025 会议摘要混合，说明仅凭 OpenAlex 年份与 Source 仍可能出错。

旧 11,089 候选的 23.64% 摘要/53.71% 机构覆盖是混合文献类型统计，不能继续作为 Fig.8a 的研究论文缺失率。本报告仅替换 Nature 主刊母体与覆盖口径，不更改三领域期刊名录和其他历史结果。

## 可复用数据与下一步工作量

主分析复用 3,683 个唯一 Work ID、DOI、主题/领域与现有元数据，现有 2,073 份 OpenAlex 摘要可作为后续文本质检输入；当前只检查非空，并未完成摘要内容真实性、版本和完整性逐篇验证。补齐目标为 1,610 份摘要、438 篇的作者机构映射、另 683 篇的机构国家映射。补齐需按 DOI 对齐出版社正文，不用标题猜配或模型生成摘要。

这一步没有调用语言模型或 embedding，也没有启动全文/贡献分析。后续每篇一次的基础分析预算应改按 3,683 篇（扩展口径 3,696 篇）计，不能继续用 11,089 条混合候选计价；模型 token、重复次数和运行耗时仍需在固定样本上实测。本次没有执行 1,610 篇摘要批量补全。

## 交付与证据边界

- [主分析清单](research_roster_locked.csv) / `research_roster_locked.parquet`：3,683 篇。
- [8 篇撤稿研究及通知链接](excluded_retracted_research.csv)：包括 2026 年发布的撤稿通知，故不是仅截至 2025 年的历史撤稿状态。
- [13 篇 Analysis 类型审查](analysis_type_review.csv) / `analysis_sensitivity_roster.parquet`。
- [全部 13,272 条的纳排决定](publisher_type_decisions.csv)、[各类型年度数量](publisher_types_by_year.csv)、[旧候选逐条决定](prior_candidate_decisions.csv)。
- `publisher_raw/` 保存目录/文章 HTML 及请求地址、抓取时间、SHA256；`openalex/api_raw/` 保存无密钥请求与完整 API 响应；`datacite_raw/` 保存仓储关联证据；`publisher_notices.json` 保存 35 条通知与目标论文的显式关系。
- `validation.json` 核验目录分母、唯一键、纳排及缺失清单；`artifact_manifest.json` 记录本次交付数据文件 SHA256，方便追踪此批次文件，不代表完成科学有效性验证。

这是已核对出版社目录的类型清单，不是逐篇科学贡献人工审稿；地方机构解析和原始多重署名仍需核验。覆盖统计属于当前 API，不能写成 snapshot 的重跑结果。历史 `data/claim_graph` 与旧 Fig.8 数据未改写。
'''
    (OUT / 'Nature研究论文清单与缺失统计.md').write_text(text, encoding='utf-8')


def main() -> None:
    rows, catalogue = read_rows('research_roster_locked'), read_rows('publisher_catalogue')
    audit = json.loads((OUT / 'research_audit.json').read_text())
    write_json(OUT / 'validation.json', validate(rows, catalogue, audit))
    make_queue(rows)
    type_tables(catalogue, rows)
    analysis_review()
    report(audit)
    paths = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv', '.parquet', '.json', '.md']
             and p.name != 'artifact_manifest.json']
    write_json(OUT / 'artifact_manifest.json', {'generated_at': datetime.now(timezone.utc).isoformat(),
        'files': [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]})
    print('Validated roster and generated report, type decisions, repair queue and manifest')


if __name__ == '__main__':
    main()

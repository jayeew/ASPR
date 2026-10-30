"""Report measured metadata gains without treating local affiliations as standard IDs."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .nature_backfill_fetch import BASE, OUT
from .stage0 import write_csv, write_json


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for year in [2023, 2024, 2025, 0]:
        group = [r for r in rows if not year or r['publisher_year'] == year]
        record: dict[str, Any] = {'year': year, 'paper_n': len(group)}
        fields = ['has_abstract', 'has_institution', 'all_observed_authors_have_institution',
                  'complete_institution_country', 'has_abstract_enriched', 'has_standard_institution_enriched',
                  'all_authors_standard_institution_enriched', 'all_linked_countries_enriched',
                  'all_authors_at_least_one_country_enriched']
        for field in fields:
            record[field + '_n'] = sum(bool(r[field]) for r in group)
            record[field + '_rate'] = record[field + '_n'] / len(group) if group else None
        record['all_authors_affiliation_text_or_standard_id_n'] = sum(all(a['institutions'] or a['backfill_affiliations']
            for a in json.loads(r['authorships_enriched_json'])) for r in group)
        result.append(record)
    write_csv(OUT / 'coverage_before_after_by_year.csv', result)
    return result


def export_papers(rows: list[dict[str, Any]]) -> None:
    summary, added, unresolved = [], [], []
    for row in rows:
        fields = ['doi', 'work_id', 'title', 'publisher_year', 'article_url', 'abstract_source',
                  'abstract_source_receipt', 'has_abstract_enriched', 'has_standard_institution_enriched',
                  'all_authors_standard_institution_enriched', 'all_linked_countries_enriched',
                  'all_authors_at_least_one_country_enriched', 'publisher_author_n', 'observed_author_n']
        item = {key: row[key] for key in fields}
        item['abstract_length'] = len(row['abstract_text_enriched'])
        summary.append(item)
        if row['abstract_source'] == 'nature_publisher':
            added.append({**item, 'abstract_text': row['abstract_text_enriched']})
        missing = []
        for key, reason in [('has_abstract_enriched', 'missing_abstract'),
                            ('all_authors_standard_institution_enriched', 'incomplete_standard_institution_mapping'),
                            ('all_linked_countries_enriched', 'incomplete_institution_country_chain'),
                            ('all_authors_at_least_one_country_enriched', 'author_country_unknown')]:
            if not row[key]:
                missing.append(reason)
        if missing:
            unresolved.append({**item, 'remaining_reasons': missing})
    write_csv(OUT / 'research_roster_enriched.csv', summary)
    write_csv(OUT / 'abstracts_added.csv', added)
    write_csv(OUT / 'unresolved_papers.csv', unresolved)


def local_affiliations() -> dict[str, int]:
    audit = json.loads((OUT / 'author_mapping_audit.json').read_text())
    links, local = [], {}
    usage: dict[str, set[str]] = defaultdict(set)
    for row in audit:
        for affiliation in row['affiliation_evidence']:
            links.append({'doi': row['doi'], 'author_position': row['author_position'],
                          'author_name': row['author_name'], **affiliation})
            if not affiliation['resolved_institution_ids']:
                key = affiliation['local_affiliation_id']
                local[key] = {'local_affiliation_id': key, 'raw_affiliation': affiliation['address'],
                              'country_code': affiliation['country_code'],
                              'entity_kind': 'affiliation_unit_not_standard_institution',
                              'standard_id_status': 'unresolved', 'example_doi': row['doi'],
                              'source': affiliation['address_source'],
                              'source_receipt': affiliation.get('publisher_source_receipt')}
                usage[key].add(row['doi'])
    write_csv(OUT / 'author_affiliation_links.csv', links)
    write_csv(OUT / 'local_affiliation_units.csv', [{**r, 'paper_n': len(usage[key])} for key, r in local.items()])
    return {'missing_author_n_before': len(audit), 'standard_author_mappings_added_n': sum(a['resolved'] for a in audit),
            'unresolved_author_n': sum(not a['resolved'] for a in audit),
            'author_affiliation_link_n': len(links), 'unresolved_local_affiliation_unit_n': len(local),
            'authors_without_address_n': sum(not a['affiliation_evidence'] for a in audit),
            'publisher_sourced_author_n': sum(any(e['address_source'].startswith('publisher') for e in a['affiliation_evidence']) for a in audit),
            'openalex_raw_fallback_author_n': sum(any(e['address_source'] == 'openalex_raw_affiliation' for e in a['affiliation_evidence']) for a in audit)}


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = BASE / 'research_roster_locked.parquet'
    previous = json.loads((BASE / 'artifact_manifest.json').read_text())['files']
    expected = next(r['sha256'] for r in previous if r['path'] == source.name)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    baseline = pq.read_table(source).to_pylist()
    before = {r['doi']: r for r in baseline}
    assert len(rows) == 3683 and {r['doi'] for r in rows} == set(before)
    assert len({r['work_id'] for r in rows}) == len(rows)
    receipts = set()
    for row in rows:
        prior = before[row['doi']]
        assert row['abstract_text'] == prior['abstract_text']
        assert row['authorships_json'] == prior['authorships_json']
        authors = json.loads(row['authorships_enriched_json'])
        assert len(authors) == len(json.loads(prior['authorships_json']))
        assert all(a['author'] == b['author'] for a, b in zip(authors, json.loads(prior['authorships_json'])))
        if prior['has_abstract']:
            assert row['abstract_text_enriched'] == prior['abstract_text']
        if row['abstract_source'] == 'nature_publisher':
            receipt = row['abstract_source_receipt']
            receipts.add(receipt)
            assert row['has_abstract_enriched'] and len(row['abstract_text_enriched']) >= 100
        for author in authors:
            for inst in author['institutions']:
                assert inst.get('id') or str(inst.get('ror', '')).startswith('https://ror.org/')
                if inst.get('country_code'):
                    assert len(inst['country_code']) == 2 and inst['country_code'].isupper()
    attempted = json.loads((OUT / 'abstracts_fetch_summary.json').read_text())
    assert len(attempted) == 1610 and len({r['doi'] for r in attempted}) == 1610
    for receipt in receipts:
        path = Path(receipt)
        raw = json.loads(path.read_text())
        assert hashlib.sha256(gzip.decompress(path.with_suffix('.gz').read_bytes())).hexdigest() == raw['sha256']
    return {'passed': True, 'baseline_sha256_unchanged': expected, 'cohort_n': len(rows),
            'abstract_target_attempted_n': len(attempted), 'added_abstract_source_receipts_n': len(receipts),
            'checks': ['baseline file hash unchanged', 'fixed DOI/Work ID denominator', 'original fields preserved',
                       'original author identities and counts preserved', 'source-backed abstract additions',
                       'standard institution ID namespaces', 'country code shape']}


def notebook() -> None:
    source = f'''from pathlib import Path
import json
import pyarrow.parquet as pq
base = Path({str(BASE)!r})
out = Path({str(OUT)!r})
before = pq.read_table(base / 'research_roster_locked.parquet').to_pylist()
after = pq.read_table(out / 'research_roster_enriched.parquet').to_pylist()
assert {{r['doi'] for r in before}} == {{r['doi'] for r in after}}
assert len(after) == 3683
for year in [2023, 2024, 2025, 0]:
    group = [r for r in after if not year or r['publisher_year'] == year]
    print(year or 'all', len(group),
          'abstract', sum(r['has_abstract'] for r in group), sum(r['has_abstract_enriched'] for r in group),
          'all authors standard institutions', sum(r['all_observed_authors_have_institution'] for r in group),
          sum(r['all_authors_standard_institution_enriched'] for r in group),
          'institution-country chain', sum(r['complete_institution_country'] for r in group),
          sum(r['all_linked_countries_enriched'] for r in group))
'''
    write_json(OUT / 'coverage_audit.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': 'markdown', 'metadata': {}, 'id': 'scope', 'source': [
            ('# Nature metadata backfill audit\nFixed 3,683-paper cohort. Counts use the original observed author set. '
             'Local affiliation keys do not count as standard institution IDs. Sources and unresolved mappings remain separate.')]},
            {'cell_type': 'code', 'metadata': {}, 'id': 'audit', 'execution_count': None, 'outputs': [],
             'source': source.splitlines(keepends=True)}]})


def report(coverage: list[dict[str, Any]], stats: dict[str, int], repairs: list[dict[str, Any]]) -> None:
    total = coverage[-1]
    lines = ['| 年份 | 论文数 | 摘要：补齐前→后 | 每位已记录作者有标准机构：前→后 | 机构—国家链路：前→后 |',
             '|---|---:|---:|---:|---:|']
    pairs = [('has_abstract', 'has_abstract_enriched'), ('all_observed_authors_have_institution', 'all_authors_standard_institution_enriched'),
             ('complete_institution_country', 'all_linked_countries_enriched')]
    for row in coverage:
        cells = [str(row['year'] or '合计'), str(row['paper_n'])]
        for old, new in pairs:
            cells.append(f'{row[old + "_n"]:,}（{row[old + "_rate"]:.2%}）→ {row[new + "_n"]:,}（{row[new + "_rate"]:.2%}）')
        lines.append('| ' + ' | '.join(cells) + ' |')
    added = total['has_abstract_enriched_n'] - total['has_abstract_n']
    text = f'''# Nature 2023–2025 摘要、作者机构和国家补齐结果

批次：2026-10-21。固定上一阶段的 **3,683 篇 Article 主分析清单**，未重新筛选论文、未纳入 13 篇 Analysis 敏感性组。所有百分比分母均为固定论文数；原始 OpenAlex 字段与补齐字段分别保存。补充元数据来自当前出版社页面和机构登记库，不是本地 snapshot 的新扫描结果。

## 实际完成情况

1. 对缺摘要的 1,610 篇逐篇请求 Nature 页面，以 DOI、Nature 期刊名和 Article 类型核验身份；实际新增 **{added:,} 篇**出版社摘要，剩余 **{3683 - total['has_abstract_enriched_n']:,} 篇**缺摘要。仅提取摘要正文，移除文献引用标记，保留科学上下标为 `^{{…}}`、`_{{…}}`；未用标题、编辑导语或模型生成内容替代摘要。
2. 原来 {stats['missing_author_n_before']:,} 条作者记录无标准机构ID，补上 **{stats['standard_author_mappings_added_n']:,} 条**；仍有 **{stats['unresolved_author_n']:,} 条**未可靠匹配标准机构。已尽可能从出版社按同篇作者姓名和显式机构链接恢复署名，再按已有精确署名关联或 ROR 匹配规则处理。未匹配项保留原始署名和本地 affiliation 单元键，**本地键不算标准机构ID**。
3. 原机构国家字段为空的 **{len(repairs)} 个机构**均用已有 ROR ID 查回唯一国家代码，保留登记库原记录和时间。该修复针对机构登记地点，不代表作者国籍，不把子机构自动并入母高校。

上述缺机构作者中，{stats['publisher_sourced_author_n']} 条采用出版社显式署名关系，{stats['openalex_raw_fallback_author_n']} 条因未能唯一对应出版社作者而保留 OpenAlex 原始署名作回退；后者不计作新取得的出版社信息。缺作者地址的剩余数为 {stats['authors_without_address_n']}。

## 前后覆盖率

{chr(10).join(lines)}

“每位作者有标准机构”表示每位原 OpenAlex 已记录作者至少关联一个 OpenAlex 或 ROR 机构，不表示多重署名全部恢复。标准机构ID不同于已知的原始机构署名。“机构—国家链路”要求每位原已记录作者有标准机构，且所有已关联机构都有国家代码。两项仍可能受既有 OpenAlex 署名错误影响，本次修复不等于逐作者全面人工核验。

此外，有机构原始署名或标准机构的作者覆盖完整论文数为 **{total['all_authors_affiliation_text_or_standard_id_n']:,}/{total['paper_n']:,}**；每位作者至少有一个可识别国家/地区的论文数为 **{total['all_authors_at_least_one_country_enriched_n']:,}/{total['paper_n']:,}（{total['all_authors_at_least_one_country_enriched_rate']:.2%}）**。后者允许国家来自署名中明确的末尾国家字段，即便标准机构 ID 尚未解决；它不能替代标准高校归属。

国家覆盖基于机构登记地点或明确署名地址，不代表逐人核实实际工作地点。跨城市校区、跨国企业分部及旧 OpenAlex 机构归属仍需在国家/高校分数计数前审查；不会把当前机构登记状态描述成发表当年的已核实历史状态。

## 补齐数据与未解决项

- [合并后的论文清单](research_roster_enriched.csv)：每篇的摘要来源及补齐覆盖标志；完整摘要、原始与补齐后的作者机构 JSON 在 `research_roster_enriched.parquet`。
- [新增出版社摘要](abstracts_added.csv)：新增摘要正文、DOI、标题、来源页面及抓取回执路径。
- [逐作者匹配审计](author_mapping_audit.csv)：只针对原来缺机构的作者，记录新增机构、匹配依据和状态。
- [作者—署名链接](author_affiliation_links.csv) 与 [本地署名单元](local_affiliation_units.csv)：保留未标准化的单位名称、地点和来源，避免信息丢失；本地单元不代表已统一的机构实体。
- [仍缺标准机构的作者](unresolved_author_institutions.csv)、[仍未匹配的署名地址](unresolved_affiliations.csv)、[仍有缺失的论文](unresolved_papers.csv)。即使某作者已匹配一个机构，其其他未解决署名也继续保留。
- [机构国家修复表](institution_country_repairs.csv)、[年度前后覆盖](coverage_before_after_by_year.csv)、[可运行核验 notebook](coverage_audit.ipynb)。

仍有 {stats['authors_without_address_n']} 条原作者记录没有取得可用署名地址，保留 unresolved；不从同名作者其他论文或共同作者机构推测。出版社当前署名与 OpenAlex 作者数不一致的论文另存 `publisher_author_count_differences.csv`，未将新增/团体作者强行匹配到既有作者ID，也未更改本报告的原作者分母。

三篇此前全体作者均无机构ID的论文，其出版社企业署名已保留（Generate Biomedicines、Artilux、Vector Atomic）；对标准数据库的搜索未得到可靠实体时不伪造 OpenAlex/ROR ID。国家按明确地址规范为 ISO 国家/地区代码，机构类型未知时不归入高校。

## 匹配与证据规则

出版社作者采用同篇规范化姓名唯一匹配和显式 Aff 链接；现工作地址与研究署名分开，现地址不用于新增研究机构归属。缺标准ID时先尝试既有精确署名—机构关联；其次使用 ROR 的 chosen 候选，并要求名称/别名出现在署名的机构部分、明确国家一致、登记城市对应。同名异地医院、地名误当机构名等冲突保留待审。该规则偏保守，跨城市校区/企业分部可能需要人工合并。[ROR 官方匹配说明](https://ror.readme.io/docs/api-affiliation)也明确说明自动匹配可能出错。

`publisher_raw/`、`publisher_records/`、`ror_raw/`、`openalex/api_raw/` 保存原始响应、解析记录和无密钥请求；先前缓存页面的来源回执仍指向上一阶段目录。`fetch_attempts.jsonl` 追加保存成功与失败尝试，后续成功不擦除之前失败。`iso_3166-1.json` 保留国家代码参考表；别名规则见代码。`artifact_manifest.json` 保存最终数据文件摘要，`validation.json` 保存固定分母、原文件未变、字段保留等校验结果。

没有调用语言模型或 embedding。摘要补齐后的文本尚未启动科学贡献提取；机构标准化未解决项应在高校比较前复核，不能将它们当作“无机构”或“非高校”的已确认结论。
'''
    (OUT / '补齐结果与剩余清单.md').write_text(text, encoding='utf-8')


def main() -> None:
    rows = pq.read_table(OUT / 'research_roster_enriched.parquet').to_pylist()
    write_json(OUT / 'validation.json', validate(rows))
    coverage = summarize(rows)
    export_papers(rows)
    stats = local_affiliations()
    repairs = json.loads((OUT / 'institution_country_repairs.json').read_text())
    write_csv(OUT / 'institution_country_repairs.csv', repairs)
    differences = [{'doi': r['doi'], 'title': r['title'], 'openalex_observed_author_n': r['observed_author_n'],
                    'publisher_author_n': r['publisher_author_n'], 'article_url': r['article_url']}
                   for r in rows if r['publisher_author_n'] is not None and r['publisher_author_n'] != r['observed_author_n']]
    write_csv(OUT / 'publisher_author_count_differences.csv', differences)
    write_json(OUT / 'backfill_summary.json', {'coverage': coverage, 'author_repairs': stats,
        'country_repair_statuses': dict(Counter(r['status'] for r in repairs)),
        'publisher_author_count_difference_papers_n': len(differences)})
    notebook()
    report(coverage, stats, repairs)
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != 'artifact_manifest.json']
    write_json(OUT / 'artifact_manifest.json', {'generated_at': datetime.now(timezone.utc).isoformat(),
        'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(paths)]})
    print(json.dumps({'coverage_total': coverage[-1], 'author_repairs': stats}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

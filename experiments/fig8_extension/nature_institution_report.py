"""Validate the second review and report conditional Fig.8 metadata sensitivity."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq

from .nature_institution_apply import MANUAL, address_country, country_names
from .nature_institution_review import OUT, PREV
from .stage0 import write_csv, write_json


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def conflicts(before: list[dict[str, Any]]) -> dict[str, Any]:
    hits = []
    partial_authors, partial_papers = 0, set()
    for row in before:
        for pos, author in enumerate(json.loads(row['authorships_enriched_json'])):
            ids = {i['id']: i for i in author['institutions'] if i.get('id')}
            if ids and not author.get('backfill_affiliations') and any(not x['institution_ids'] for x in author.get('affiliations', [])):
                partial_authors += 1
                partial_papers.add(row['doi'])
            for aff in author.get('affiliations', []):
                country = address_country(aff['raw_affiliation_string'], country_names())
                linked = [ids[i] for i in aff['institution_ids'] if i in ids]
                if country and len(linked) == 1 and linked[0].get('country_code') and linked[0]['country_code'] != country:
                    hits.append({'doi': row['doi'], 'author_position': pos, 'institution_id': linked[0]['id'],
                        'name': linked[0]['display_name'], 'raw_address': aff['raw_affiliation_string'],
                        'address_country': country, 'registry_country': linked[0]['country_code'],
                        'cn_hk_mo_granularity': {country, linked[0]['country_code']} <= {'CN', 'HK', 'MO'}})
    write_csv(OUT / 'country_address_conflicts.csv', hits)
    return {'conflict_link_n': len(hits), 'conflict_paper_n': len({x['doi'] for x in hits}),
            'cn_hk_mo_granularity_n': sum(x['cn_hk_mo_granularity'] for x in hits),
            'other_conflict_link_n': sum(not x['cn_hk_mo_granularity'] for x in hits),
            'other_conflict_paper_n': len({x['doi'] for x in hits if not x['cn_hk_mo_granularity']}),
            'already_mapped_authors_with_unparsed_fragments': partial_authors,
            'already_mapped_papers_with_unparsed_fragments': len(partial_papers)}


def entity_summary(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Any] = {}
    for row in audit:
        for entity in row['entities']:
            if entity['identity_status'] == 'identity_unresolved':
                continue
            key = entity['review_entity_id']
            item = grouped.setdefault(key, {'entity_id': key, 'name': entity['display_name'],
                'entity_type': entity['review_entity_type'], 'identity_status': entity['identity_status'],
                'papers': set(), 'authors': set()})
            item['papers'].add(row['doi'])
            item['authors'].add((row['doi'], row['author_position']))
    output = [{k: v for k, v in r.items() if k not in ['papers', 'authors']} |
              {'paper_n': len(r['papers']), 'author_n': len(r['authors']), 'dois': sorted(r['papers'])} for r in grouped.values()]
    output.sort(key=lambda r: (-r['paper_n'], -r['author_n']))
    write_csv(OUT / 'resolved_entities_by_frequency.csv', output)
    write_csv(OUT / 'verified_entities_pending_registry_id.csv', [r for r in output if r['identity_status'] != 'standard_registry'])
    return output


def sensitivity_tables() -> dict[str, Any]:
    totals = json.loads((OUT / 'fractional_summary.json').read_text())
    country_rows = read_csv('country_fractional_scenarios.csv')
    university_rows = read_csv('education_fractional_scenarios.csv')
    result = {}
    for label, rows, key, top_n in [('country', country_rows, 'country_code', 12), ('education', university_rows, 'institution_id', 15)]:
        scenarios: dict[str, dict[str, float]] = defaultdict(dict)
        names = {}
        for row in rows:
            scenarios[row['scenario']][row[key]] = float(row['fractional_n'])
            names[row[key]] = row.get('name', row[key])
        main = scenarios['reviewed']
        ordered = sorted([k for k in main if k != 'unknown'], key=lambda k: -main[k])
        ranking = {s: {k: i + 1 for i, k in enumerate(sorted([k for k in data if k != 'unknown'], key=lambda k: -data[k]))}
                   for s, data in scenarios.items()}
        table = []
        for ident in ordered:
            item = {key: ident, 'name': names[ident]}
            for scenario, values in scenarios.items():
                item[scenario + '_fractional_n'] = values.get(ident, 0)
                item[scenario + '_rank'] = ranking[scenario].get(ident)
                item[scenario + '_share_of_papers_pct'] = values.get(ident, 0) / totals[scenario]['paper_n'] * 100
            item['complete_case_loss_pct'] = (1 - scenarios['complete_case'].get(ident, 0) / main[ident]) * 100
            table.append(item)
        write_csv(OUT / (label + '_comparison_sensitivity.csv'), table)
        result[label] = {'top_n': top_n, 'main_top_ids': ordered[:top_n],
            'top_order_unchanged_from_before': all(ranking['before'].get(k) == i + 1 for i, k in enumerate(ordered[:top_n])),
            'top_order_unchanged_after_nonpersonal_exclusion': all(ranking['exclude_headings_and_collectives'].get(k) == i + 1 for i, k in enumerate(ordered[:top_n])),
            'boundary_gap_fractional_n': main[ordered[top_n - 1]] - main[ordered[top_n]],
            'largest_raw_country_sensitivity_change': max(abs(scenarios['raw_country_sensitivity'].get(k, 0) - v) for k, v in main.items()),
            'largest_nonpersonal_exclusion_change': max(abs(scenarios['exclude_headings_and_collectives'].get(k, 0) - v) for k, v in main.items())}
    return result


def validate(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = json.loads((PREV / 'artifact_manifest.json').read_text())
    source = PREV / 'research_roster_enriched.parquet'
    expected = next(r['sha256'] for r in manifest['files'] if r['path'] == source.name)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == expected
    assert len(before) == len(after) == 3683
    assert len({r['doi'] for r in after}) == 3683
    assert all(r['has_abstract_enriched'] and r['abstract_text_enriched'] for r in after)
    for old, new in zip(before, after, strict=True):
        assert all(new[k] == v for k, v in old.items())
        a_old, a_new = json.loads(old['authorships_enriched_json']), json.loads(new['authorships_reviewed_json'])
        assert len(a_old) == len(a_new)
        for x, y in zip(a_old, a_new, strict=True):
            assert x['author'] == y['author']
            assert y['institutions'][:len(x['institutions'])] == x['institutions']
            assert all(not i.get('review_entity_id', '').startswith('reviewed-local:') for i in y['institutions'])
    totals = json.loads((OUT / 'fractional_summary.json').read_text())
    assert all(abs(v['country_weight_sum'] - v['paper_n']) < 1e-7 for v in totals.values())
    result = {'cohort_unchanged': True, 'old_fields_unchanged': True, 'author_identity_order_unchanged': True,
              'abstracts_present': 3683, 'source_sha256': digest, 'source_matches_previous_manifest': True,
              'local_entities_not_counted_as_standard': True, 'fractional_weight_conservation': True}
    write_json(OUT / 'validation.json', result)
    return result


def coverage_detail(after: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in after:
        topic = json.loads(row['primary_topic_json'] or '{}') or {}
        field = topic.get('field', {}).get('display_name', 'unknown')
        groups[(field, row['publisher_year'])].append(row)
    write_csv(OUT / 'coverage_by_field_year.csv', [{'primary_field': field, 'year': year, 'n': len(rows),
        'all_authors_standard_n': sum(r['all_authors_standard_institution_reviewed'] for r in rows),
        'all_authors_verified_entity_n': sum(r['all_authors_verified_entity_reviewed'] for r in rows),
        'all_slots_country_n': sum(r['all_affiliation_slots_country_resolved'] for r in rows)}
        for (field, year), rows in sorted(groups.items())])


def report() -> None:
    before = pq.read_table(PREV / 'research_roster_enriched.parquet').to_pylist()
    after = pq.read_table(OUT / 'research_roster_reviewed.parquet').to_pylist()
    summary = json.loads((OUT / 'review_summary.json').read_text())
    audit = json.loads((OUT / 'author_review_audit.json').read_text())
    decisions = json.loads((OUT / 'affiliation_decisions.json').read_text())
    conflicts_summary, sensitivity = conflicts(before), sensitivity_tables()
    entities = entity_summary(audit)
    validate(before, after)
    coverage_detail(after)
    write_csv(OUT / 'remaining_identity_units.csv', [r for r in decisions if r['status'] == 'identity_unresolved'])
    totals = json.loads((OUT / 'fractional_summary.json').read_text())
    summary.update(country_conflict_audit=conflicts_summary, sensitivity=sensitivity)
    summary['all_authors_at_least_one_country_n'] = sum(all(any(i.get('country_code') for i in a['institution_review_entities']) for a in json.loads(r['authorships_reviewed_json'])) for r in after)
    summary['raw_address_n'] = sum(len(r['address_aliases']) for r in decisions)
    summary['normalized_address_n'] = len(decisions)
    summary['resolved_authorship_types'] = dict(Counter(i['review_entity_type'] for r in audit for i in r['entities'] if i['identity_status'] != 'identity_unresolved'))
    write_json(OUT / 'review_summary.json', summary)
    n, main = len(after), totals['reviewed']
    lines = ['# Nature 2023–2025 机构映射复核及 Fig.8 影响评估', '',
        '核查日期：2026-10-22。固定母体为已按出版社类型核验的 3,683 篇 Nature Article（剔除撤稿）；不含 13 篇 Analysis。输入是上一轮补齐版，未改写其文件或原字段。', '',
        '## 本轮结论', '',
        "对原剩余 422 篇、1,850 条缺标准机构的作者记录执行复核。新增 **65 条标准机构映射**，另 **489 条有来源支持的本地实体映射**；后者没有伪造 ROR/OpenAlex ID。严格标准 ID 缺口为 **415 篇／1,785 条作者记录**；允许已核实本地实体后，仍有 **393 篇／1,296 条记录**未确认实体。这里的作者记录包含团体和分组标题，不是 1,296 位自然人。", '',
        '对 737 个原始地址（规范化后 734 个地址单元）进行候选复核，执行：658 条普通查询、968 条引号/组织片段查询，均有缓存且无请求失败。只以完整组织名称、国家及城市相符接受唯一自动候选；其他标准映射必须有人工审核的官网/出版社证据。34 条人工规则见 `manual_institution_decisions.json`，每项保留理由及来源。当前 13 个地址单元取得标准映射，29 个取得本地实体映射；其余 692 个保留身份未确认。未查询到记录不等于注册库必然没有该机构。', '',
        '## 覆盖率与分母', '',
        '| 指标 | 上轮 | 本轮 |', '|---|---:|---:|',
        '| 摘要 | 3,683 / 3,683（100%） | 3,683 / 3,683（100%） |',
        f"| 每个原始作者条目至少一个可识别国家/地区 | 3,664（99.48%） | {summary['all_authors_at_least_one_country_n']:,}（99.48%） |",
        f"| 每个原始作者条目至少一个标准机构 | 3,261（88.54%） | 3,268（{3268/n*100:.2f}%） |",
        f"| 每个作者条目至少一个已确认实体，含本地映射 | 3,261 | 3,290（{3290/n*100:.2f}%） |",
        f"| 全部显式机构片段均可解析实体（更严格新口径） | 本轮重建 | 2,891（{2891/n*100:.2f}%） |",
        f"| 全部显式机构片段都有国家/地区代码（更严格新口径） | 本轮重建 | 3,639（{3639/n*100:.2f}%） |", '',
        f"另检出 {conflicts_summary['already_mapped_authors_with_unparsed_fragments']:,} 条已有机构 ID 的作者记录仍有空 `institution_ids` 的原始地址片段，涉及 {conflicts_summary['already_mapped_papers_with_unparsed_fragments']} 篇。每作者至少一个 ID 与全部机构份额完整必须区分；本轮分数计数将这些片段保留为 unknown，不把份额转给已知高校。", '',
        '## 高频单位与类型处理', '',
        '| 核实实体 | 类型 | 涉及论文 | 作者记录 | ID 状态 |', '|---|---|---:|---:|---|']
    source_urls = {i['review_entity_id']: r['evidence'].get('source_url', i.get('ror')) for r in decisions for i in r['entities']}
    for item in entities[:16]:
        item['linked_name'] = f"[{item['name']}]({source_urls[item['entity_id']]})"
        lines.append(f"| {item['linked_name']} | {item['entity_type']} | {item['paper_n']} | {item['author_n']} | {item['identity_status']} |")
    lines.extend(['', '新增映射按作者–实体记录计：企业 465，医疗机构 31，博物馆 16，研究项目 15，高校 13，研究机构 11，非营利组织 3。企业研发部门归企业；研究所、医院、博物馆与高校分别保留。Eyewire 的项目身份已确认，但未据当前官网描述把历史署名自动合并到 Princeton。仅凭 Inc./Ltd./LLC 等明确法律形式、museum 或 independent researcher 标签得到的类型标记不计作实体映射完成。', '',
        '## 团体作者与分组标题', '',
        '48 条原本没有地址的记录现分为：17 条出版社分组标题、8 条明确团体作者、4 条被 OpenAlex 收作作者但在出版社致谢中列为数据联盟的条目、19 条尚不能确认机构的个人/缺失署名记录。四个数据联盟是 cVEDA、Harvard Aging Brain Study、IMAGEN、POND；不能把联盟总部或项目主持人的机构分给全体成员。', '',
        '原始作者顺序、数量和标识均保留。另列两种敏感性条件：仅剔除 17 条标题，以及再剔除 8 条团体和 4 条致谢联盟。只对这 48 条进行过此轮署名粒度审核，不声称已清理全母体的团体作者。', '',
        '## 对 Fig.8 的影响', '',
        '权重为作者等权、作者内去重机构与显式未解析片段等分；已知国家的未解析机构仍可贡献国家权重。无机构证据的作者保留 unknown；高校份额不重新归一化。合并键仅使用确认的标准 ID 或审核过的本地实体 ID，未解析地址用不同的临时键。此处的敏感性界限以这个固定片段表示为条件，未来实体拆分、重复项合并或原始错误关系修正可能再改变权重。', '',
        f"**国家/地区：** 本轮身份补齐没有改变国家分数产出，因为新增实体的国家此前已在署名地址中识别。仍未知的国家权重为 **{main['country_unknown_weight']:.4f} 篇等价量／{main['country_unknown_weight']/n*100:.4f}%**，不是缺失论文比例。剔除已核实非个人条目后为 {totals['exclude_headings_and_collectives']['country_unknown_weight']:.4f}。前 12 个国家/地区的顺序在补齐前后及非个人条目剔除中一致。", '',
        f"但第 12 位 IL 与第 13 位 DK 的分数产出差仅 **{sensitivity['country']['boundary_gap_fractional_n']:.4f}**，小于全局未知国家权重；因此不能声称前 12 名单在最坏缺失分配下已锁定。另有 448 条地址–登记国家冲突，涉及 117 篇：366 条属于 CN/HK/MO 编码粒度差异；其余 82 条涉及 39 篇，需区分海外分支、地址串合并和错误机构链接。它们单列在 `country_address_conflicts.csv`；保留既有地区代码，不自动合并。", '',
        f"`raw_country_sensitivity` 只在单机构 ID、唯一显式地址国家时用原始地址国家替换登记国家作诊断，排除 CN/HK/MO 粒度替换；不是已人工确认的纠错版。单一国家变化的最大绝对值为 {sensitivity['country']['largest_raw_country_sensitivity_change']:.4f} 篇等价量。覆盖率不能替代归属正确性审核。", '',
        f"**高校：** 新增高校份额为 {main['education_weight']-totals['before']['education_weight']:.4f} 篇等价量，头部 15 所候选高校的名单和顺序未变。机构身份未知权重从 {totals['before']['unknown_identity_weight']:.4f} 降至 **{main['unknown_identity_weight']:.4f}（{main['unknown_identity_weight']/n*100:.2f}%）**；在剔除已确认非高校类型后，类型仍未知的权重为 **{main['university_unknown_upper_weight']:.4f}**。这只是固定片段模型下可分配给高校的宽松上界，不是这些份额确属高校的估计，也不是缺失随机的假设。第 15/16 位候选差距 {sensitivity['education']['boundary_gap_fractional_n']:.4f}，现有证据不能证明高校名单对所有缺失分配稳健。", '',
        '高校候选表沿用原 OpenAlex `education` 分类，并纳入本轮明确核实高校；原库中的学校/医学院类型与大学合并层级并未全部人工复核。因此教育机构表只用于诊断，不直接宣告最终高校排行。医院、企业、独立研究所的原有份额不因高校图过滤而被重分配。', '',
        '**不建议完全案例删除。** 若只保留全部机构片段解析完成的论文，母体会降至 2,891 篇，移除 792 篇（21.50%）。完整案例中国分数产出减少约 31.6%，美国减少约 16.8%，并改变多国顺序；这是选择性缺失，不是质量差异。主分析应保留全母体及 unknown，完整案例仅作为敏感性条件。', '',
        '**可执行决定：** 国家描述性分数产出可继续计算，但在最终锁定 12 个展示主体前处理国家/地区编码口径及 39 篇非粒度冲突；高校图保留未知份额并暂不冻结 15 所名单，优先处理剩余高权重、可能属于高校或含大学子机构的条目。对于无法确认的公司、研究项目、独立研究者和团体署名，保留明确类别或未解决状态，不能靠名称相似度补上大学 ID。', '',
        '本次只覆盖 Nature 主刊 2023–2025。Fig.8c 最终仍应复用 b 的三个领域跨期刊母体；这里的缺失率和名单稳定性不能直接推广到跨期刊数据。本轮没有运行贡献分类、知识结构或领域年份调整，不能据此判断国家/高校的贡献构成或创新性差异。', '',
        '## 文件、来源与复现', '',
        '- `research_roster_reviewed.parquet`：原字段及完整新作者实体结构。',
        '- `remaining_papers.csv`：严格标准映射仍有缺口的 415 篇；含本地实体口径标记。',
        '- `remaining_identity_authors.csv`：仍缺确认实体的 1,296 条记录；`remaining_standard_mapping_authors.csv` 另含已确认但无标准 ID 的记录。',
        '- `remaining_identity_units.csv`：692 个地址单元，含涉及论文、作者数、类型和来源；`verified_entities_pending_registry_id.csv` 为仅缺注册 ID 的实体。',
        '- `affiliation_decisions.json/csv`、`author_review_audit.json/csv`、`author_grain_review.json/csv`：逐项证据及处理结论。',
        '- `country_comparison_sensitivity.csv`、`education_comparison_sensitivity.csv`、`institution_type_fractional_scenarios.csv`：6 个计数条件。',
        '- `coverage_by_year.csv`、`coverage_by_field_year.csv`、`paper_missing_weight.csv`：年度、领域和论文缺失分布。',
        '- `primary_sources/`、`ror_queries/` 与上一轮出版社收据构成来源链。43 个补充来源中 35 个保存了成功响应；8 个官网直接下载失败，保留失败收据，证据来自本轮浏览核查的对应官网页面及人工规则中的记录，不伪称全源文件缓存成功。',
        '- `validation.json`：输入哈希与上轮清单一致、3,683 篇母体不变、摘要完整、作者身份/顺序不变、本地 ID 不冒充标准 ID、所有分数条件权重守恒。', '',
        '```bash', 'python3 -m experiments.fig8_extension.nature_institution_review collect',
        'python3 -m experiments.fig8_extension.nature_institution_review quoted',
        'python3 -m experiments.fig8_extension.nature_institution_review propose',
        'python3 -m experiments.fig8_extension.nature_institution_apply fetch',
        'python3 -m experiments.fig8_extension.nature_institution_apply apply',
        'python3 -m experiments.fig8_extension.nature_institution_report', 'python3 -m pytest -q tests/fig8_extension', '```', ''])
    (OUT / '机构复核与Fig8影响评估.md').write_text('\n'.join(lines))
    write_json(OUT / 'manual_institution_decisions.json', json.loads(MANUAL.read_text()))
    manifest = [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json']
    write_json(OUT / 'artifact_manifest.json', {'generated_at': datetime.now(timezone.utc).isoformat(), 'files': manifest})
    print('Validation and impact report complete')


if __name__ == '__main__':
    report()

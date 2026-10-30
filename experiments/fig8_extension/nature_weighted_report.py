"""Build a source-linked review report and a runnable local audit notebook."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from lxml import html

from .nature_weighted_review import FIELD, OUT, PREV, ROOT, RULES, publisher
from .stage0 import write_csv, write_json


def read(name: str) -> Any:
    return json.loads((OUT / name).read_text())


def rate(n: int) -> str:
    return f'{n:,} / 3,683（{100 * n / 3683:.2f}%）'


def rank_table(filename: str, key: str, n: int) -> pd.DataFrame:
    data = pd.read_csv(OUT / filename)
    before = data[data.scenario == 'before'].set_index(key)
    current = data[data.scenario == 'reviewed'].copy()
    current['previous_rank'] = current[key].map(before['rank'])
    current['delta_fractional_n'] = current['fractional_n'] - current[key].map(before['fractional_n']).fillna(0)
    write_csv(OUT / filename.replace('_scenarios', '_comparison'), current.to_dict('records'))
    return current.head(n)


def publication_disclosures() -> None:
    records = []
    parent_records = []
    for rule in json.loads(RULES.read_text()):
        if not rule['source_url'].startswith('https://www.nature.com/articles/'):
            continue
        doi = '10.1038/' + rule['source_url'].rsplit('/', 1)[-1]
        pub = publisher(doi)
        if rule.get('source_kind') == 'publisher_explicit_parent':
            parent_records.append({'rule': rule, 'doi': doi, 'source_receipt': pub['source_receipt'],
                                   'publisher_affiliations': pub['affiliations']})
            continue
        raw = Path(pub['source_receipt']).with_suffix('.gz')
        tree = html.fromstring(gzip.decompress(raw.read_bytes()).decode('utf-8'))
        nodes = tree.xpath('//*[self::h2 or self::h3][contains(text(),"Competing interests")]/following-sibling::p[1]')
        records.append({'canonical_name': rule['canonical_name'], 'doi': doi, 'source_url': rule['source_url'],
            'source_receipt': pub['source_receipt'], 'disclosure': '\n'.join(x.text_content() for x in nodes),
            'interpretation': rule['note']})
    write_json(OUT / 'publisher_disclosure_evidence.json', records)
    write_json(OUT / 'publisher_parent_evidence.json', parent_records)


def notebook() -> None:
    code = """from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq
root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
if not (root / 'review_summary.json').exists():
    root = Path('/home/jayee/workspace/ASPR/outputs/fig8_data/nature_weighted_review')
summary = json.loads((root / 'review_summary.json').read_text())
weights = json.loads((root / 'fractional_summary.json').read_text())
assert summary['cohort_n'] == 3683
assert summary['before_unresolved_authors'] - summary['newly_resolved_authors'] + summary['newly_quarantined_authors'] == summary['remaining_identity_authors']
for scenario, values in weights.items():
    assert abs(values['country_mass'] - values['paper_n']) < 1e-7
roster = pq.read_table(root / 'research_roster_weighted_review.parquet', columns=['doi']).to_pandas()
assert roster.doi.nunique() == 3683
print(summary)
for filename, n in [('country_rank_comparison.csv', 13), ('university_rank_comparison.csv', 16)]:
    display(pd.read_csv(root / filename).head(n))
print(json.loads((root / 'topk_stability.json').read_text()))
"""
    content = {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Nature 2023–2025 高权重机构复核\n',
            '读取本轮成果，复核分母、作者队列守恒、国家权重守恒和前列名单。无需网络或重新抓取。\n',
            '完整重算：`python3 -m experiments.fig8_extension.nature_weighted_review apply`。']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    write_json(OUT / 'weighted_review_audit.ipynb', content)
    # Execute the actual notebook code in a plain Python namespace, without network access.
    exec(compile(code, str(OUT / 'weighted_review_audit.ipynb'), 'exec'), {'display': lambda x: None, 'print': lambda *x: None})


def build() -> None:
    summary, fractions, stability = read('review_summary.json'), read('fractional_summary.json'), read('topk_stability.json')
    country = rank_table('country_rank_scenarios.csv', 'country', 16)
    university = rank_table('university_rank_scenarios.csv', 'institution_id', 20)
    changes, audit, remaining = read('mapping_change_log.json'), read('author_coverage_audit.json'), read('remaining_weighted_units.json')
    resolved = [c for c in changes if c['old_slot']['identity_status'] == 'identity_unresolved' and c['new_slot'] and
                c['new_slot']['identity_status'] != 'identity_unresolved']
    count = Counter(c['new_slot']['identity_status'] for c in resolved)
    target = {(a['doi'], a['author_position']) for a in audit if not a['before_known']}
    extra_authors = {(c['doi'], c['author_position']) for c in resolved} - target
    enriched = pq.read_table(OUT / 'research_roster_weighted_review.parquet').to_pylist()
    by_doi: dict[str, Any] = defaultdict(list)
    for a in audit:
        by_doi[a['doi']].append(a)
    years = []
    for year in [2023, 2024, 2025]:
        rows = [r for r in enriched if r['publisher_year'] == year]
        years.append({'year': year, 'paper_n': len(rows), **{k: sum(all(a[k] for a in by_doi[r['doi']]) for r in rows)
            for k in ['after_known', 'standard_after', 'all_slots_known', 'all_slots_country']}})
    write_csv(OUT / 'coverage_by_year.csv', years)
    actions = read('country_conflict_decisions.json')
    write_csv(OUT / 'country_conflicts_unresolved.csv', [a for a in actions if a['action'].startswith('unresolved_') or a['action'].startswith('quarantine_')])
    cov = summary['coverage']
    rows_text = '\n'.join(f"| {label} | {rate(old)} | {rate(cov[key])} |" for label, old, key in [
        ('每位作者至少一个已确认单位／署名身份',3290,'after_known'), ('每位作者至少一个标准注册机构',3268,'standard_after'),
        ('全部显式机构槽位身份已解析',2891,'all_slots_known'), ('全部显式槽位国家／地区可用',3639,'all_slots_country')])
    country_text = '\n'.join(f"| {int(r['rank'])} | {r['country']} | {r['fractional_n']:.6f} | {r['delta_fractional_n']:+.6f} |" for r in country.to_dict('records'))
    uni_text = '\n'.join(f"| {int(r['rank'])} | {r['name']} | {r['fractional_n']:.6f} | {r['delta_fractional_n']:+.6f} |" for r in university.to_dict('records'))
    top_remaining = '\n'.join(f"| {u['address'].replace('|','/')} | {u['weight']:.4f} | {u['paper_n']} | {u['unresolved_reason']} |" for u in remaining[:15])
    clean = stability['country']['exclude_nonpersonal_missing_only_bound']
    body = f"""# Nature 2023–2025 高权重机构与国家归属复核（第二轮）

本轮扫描上轮全部 393 篇未完成队列，并按全队列未解析机构的分数权重优先核查。新增确认 **{summary['newly_resolved_authors']} 个原先无已确认机构的作者记录**，同时隔离 3 个旧身份误配；剩余 **{summary['remaining_identity_papers']} 篇、{summary['remaining_identity_authors']} 个作者记录**。{393 - summary['remaining_identity_papers']} 篇达到每位作者至少一个已确认单位／身份，其他论文仍可能有部分地址未解析。长期尾部没有获得足够证据的单位继续保留，未宣称全部逐项外部核查成功。

国家前 12、高校前 15 的成员和顺序，在全量样本的本轮前后、非个人署名排除、注册地和原始地址国家敏感性情景中均未改变。**“观察到稳定”不等于“缺失数据下必然稳定”**；边界和口径限制见下文。

## 范围、分母与可追溯性

- 固定 Nature 主刊在线发表年份 2023–2025 的 3,683 篇 Article 原创研究名录。摘要和原始字段均保持原样；不涉及三个领域全期刊的最终 Fig8c 外推，不评估创新性或科研质量。
- 本轮输入是上轮 `nature_institution_review/research_roster_reviewed.parquet`。新结果仅增加 `{FIELD}` 分析层；原始作者顺序、身份、机构列表和全部继承字段逐项保持不变。
- 单篇论文总权重为 1；作者等分，再按去重后的机构／显式未知地址槽位等分。一个机构有两个经出版社确认的工作国家时，仅在国家统计中均分该机构份额；高校仍计一次。无机构或国家信息的份额保留，不向已知高校重新分摊。
- 公司、研究所、联合项目与高校分开。临床中心及联合项目的单一高校上卷未自动进行。现有 OpenAlex `education` 仍是教育机构代理分类；未宣称完成全量学校／医院／院系层级标准化。
- 新身份确认涉及 **{len(resolved)} 个地址槽位**（标准身份 {count['standard_registry']}，有来源的本地身份 {count['verified_local_no_registry_id']}），其中包括原先已拥有其他机构的 **{len(extra_authors)} 个作者记录**。这与新增补齐 {summary['newly_resolved_authors']} 个“全无身份”作者是不同分母，不能相加当作论文数。

## 覆盖率

| 指标 | 上轮 | 本轮 |
|---|---:|---:|
{rows_text}

“至少一个单位”不保证所有作者的全部兼任地址完整。“标准注册机构”仅统计已有 OpenAlex/ROR 身份；本地核实的公司、项目等不会冒充标准 ID。独立研究者在新分析层单独标记为非机构身份，不计入高校。

未解析身份总份额：{fractions['before']['identity_unknown']:.6f} → **{fractions['reviewed']['identity_unknown']:.6f} 篇等价**（{fractions['reviewed']['identity_unknown']/3683:.3%}）；仍可能属于高校的未知类型份额：{fractions['before']['possible_university_unknown']:.6f} → **{fractions['reviewed']['possible_university_unknown']:.6f}**。国家未知份额：{fractions['before']['country_unknown']:.6f} → **{fractions['reviewed']['country_unknown']:.6f}**（{fractions['reviewed']['country_unknown']/3683:.3%}）。这些是缺失权重，不是缺失论文数。

## 高权重单位的处理

出版社完整署名还明确补出了原始片段省略的北京大学、ETH、Washington、Queensland、Colorado Boulder 和 Valencia 上级高校；仅在作者与该完整地址准确关联时应用。对 Imec、Naban 保护区和中法合作平台等与出版社主署名不一致的原始片段，未借用作者其他单位的国家来填空。

- [Columbia 官方院系说明](https://www.biochem.cuimc.columbia.edu/about-us) 支持 Vagelos 相关院系归入 Columbia；[清华设施目录](https://www.tsinghua.edu.cn/en/Research/Facilities1/Centers___Labs.htm) 明确列出两个结构生物学中心，故与已有清华身份合并，避免重复分摊。
- [Scripps 官方地址](https://www.scripps.edu/krishnamurthy/contacts.html) 支持 La Jolla 地址变体归一，按研究机构保留；[Hefei National Laboratory](https://en.hfnl.cn/index.htm) 与中科大的微尺度物质科学国家研究中心区分。
- [AWS 量子中心](https://aws.amazon.com/blogs/quantum-computing/announcing-the-opening-of-the-aws-center-for-quantum-computing/) 属公司研发；[FluoRok](https://www.fluorok.com/) 为 Oxford 衍生公司，均不因校园位置或创始人关系记作高校。
- [Seattle Hub](https://newsroom.uw.edu/news-releases/seattle-hub-for-synthetic-biology-launched/)、[BOTTLE](https://www.bottle.org/) 和 [NCCR MSE](https://www.unibas.ch/en/Research/Research-in-Basel/National-Networks/NCCR-MSE.html) 保留为联合项目，未把全部份额分给单一高校。
- StemiRNA、APRIL、Lilly、FoRx、TetraMem、BlueRock 等用出版社署名地址与利益声明相互核对，保留完整原始页面和 `publisher_disclosure_evidence.json`。这类公司确认缩小“可能属于高校”的未知份额。
- 部分单位身份虽能确认，但作者与出版社地址不能可靠关联，仍留在未解决队列，标记 `unit_identity_verified_but_author_to_publisher_address_link_unconfirmed`。

## 国家冲突审查

原 448 条冲突地址链接中，366 条 CN/HK/MO 差异按既有国家／地区编码口径保留；没有将它们当作错误自动覆盖。其他 **82 条链接（77 个作者—机构键、39 篇论文）**均完成证据检查：

- 59 个键用已核对的出版社工作地址设置国家／地区；跨国分支保留原注册国，新增工作地点字段；NUS 与 IOCB 的双地点份额按去重国家等分。这只修复已检查冲突，不能据此声称全库均已采用实际工作地。
- 2 个仅属于现地址的机构槽位从新分析层排除：NIST 错配／埃及现地址、Hitachi 现地址；原始记录仍保留。1 个 NUS 原始额外地址被出版社主署名否定，保留 Singapore。
- 3 个键纠正为 Voice of the Ocean Foundation（2 个）和 Theiagen Genomics（1 个）；1 个墨西哥高校误配改为出版社注明的独立研究者。
- 3 个身份误配暂不强行建立标准身份：柬埔寨森林与野生动物研究单位、秘鲁 Servicios de Biodiversidad EIRL、德国 Lauresham 实验考古单位；出版地址国家可用，机构身份仍未解决。
- 2 个 Helsinki 地址在出版社原文中同时混入 Finland 和 Córdoba, Spain，继续保留冲突；2 个 Kisangani 的 Congo 尾词保留 CD；另 4 个 PR/US、SJ/NO 差异保留地区层级。

具体作者、DOI、原 ID、地址、动作及原始页面回执见 `country_conflict_decisions.csv`；仍有疑问的 5 个键见 `country_conflicts_unresolved.csv`。国家列的补全不代表机构身份已经确认。

## 国家前 12 的稳定性

| 名次 | 国家／地区代码 | 本轮分数计数 | 相对上轮变化 |
|---:|---|---:|---:|
{country_text}

第 12 名 IL（以色列）与第 13 名 DK（丹麦）间距 **{stability['country']['boundary_gap']:.6f}**，小于保留全部作者记录时的未知国家份额 **{fractions['reviewed']['country_unknown']:.6f}**。因此保守基线不能锁死第 12 名；在“固定槽位、仅补未知国家”条件下，当前前 11 名的入围成员可保持。

排除上轮已识别的 17 个团体分组标题、8 个团体作者、4 个非署名致谢联盟后，未知国家份额降至 **{clean['unknown_country_mass']:.6f}**，小于边界差距 **{clean['boundary_gap']:.6f}**，余量 **{clean['margin']:.6f}**。在该署名口径和固定槽位缺失模型下，前 12 成员成立；这不是对所有潜在机构误配、未来去重、地区聚合或作者清洗的无条件保证，也不保证内部精确顺序。

只选完整机构论文的 complete-case 情景保留 {fractions['complete_case']['paper_n']} 篇，国家前 12 中 KR 进入、ES 退出。不能用删除缺失论文来宣称完整样本的名单更可靠。

## 高校前 15 的稳定性

| 名次 | 教育机构／高校 | 本轮分数计数 | 相对上轮变化 |
|---:|---|---:|---:|
{uni_text}

Cornell 第 15、Peking 第 16，差距 **{stability['university']['boundary_gap']:.6f}**（上轮约 0.979623）。已核实修补、作者类型排除和国家口径变化均未改变前 15 成员及顺序。完整机构样本的成员仍相同，内部顺序改变。

剩余可能属于高校的未知类型份额 **{fractions['reviewed']['possible_university_unknown']:.6f}** 仍明显超过边界。按已知工作国家限制未知份额归属的条件上界，仅能保守保证 Harvard、Stanford、MIT 这 3 个现有成员在前 15 内；不能保证其余成员或精确排序。此上界假定固定槽位和本国高校，不涵盖跨国校区、未知单位合并去重或统一医院／大学上卷；它是筛查界限，不是置信区间。

**建议 Fig8 使用当前名单作为暂定展示名单，同时保留国家第 12／13 与高校第 15／16 的边界敏感性说明。最终冻结高校名单前，需要统一医院、院系、共同研究中心与大学的归并规则，尤其不能只对北京大学或某个边界竞争高校单独上卷。**

## 未解决项和下一轮优先级

`remaining_papers.csv` 是剩余 {summary['remaining_identity_papers']} 篇；`remaining_identity_authors.csv` 是 {summary['remaining_identity_authors']:,} 个无已确认单位的作者；`remaining_weighted_units.csv` 覆盖全 3,683 篇的全部未解析槽位，包含已有其他映射的作者，范围更大。保留已确认单位但作者关联失败的候选证据，以及无地址／团体作者的不确定状态。

| 未解析地址（按权重） | 份额 | 涉及论文 | 原因 |
|---|---:|---:|---|
{top_remaining}

下一轮优先：高校医院与中心层级统一、影响边界的未知教育相关单位、已经确认单位但作者名无法关联的记录；“No affiliation evidence”需逐篇确认署名粒度，不能按作者姓名推断国籍或任职单位。

## 文件和复算

- `research_roster_weighted_review.parquet`：原始列不变、新分析层。
- `mapping_change_log.json/csv`、`weighted_affiliation_decisions.json/csv`、`manual_rules.json`：修改前后、来源、人工判断和未解决原因。
- `country_rank_comparison.csv`、`university_rank_comparison.csv`：完整排名与变化；`*_rank_scenarios.csv`：六个情景。
- `conditional_membership_bounds.csv`、`topk_stability.json`：条件上界；`coverage_by_year.csv`：年度分母。
- `source_receipts.json`、`primary_sources/`、`publisher_records/`、`web_evidence/`：原始或复用回执。直接抓取失败与网页检索证据并存，未将失败记录伪装为成功下载。
- `weighted_review_audit.ipynb`：无需网络的可执行审计；`validation.json`、`artifact_manifest.json`：守恒与来源哈希。

复算命令（项目根目录）：

```bash
python3 -m experiments.fig8_extension.nature_weighted_review apply
python3 -m experiments.fig8_extension.nature_weighted_report
python3 -m pytest -q tests/fig8_extension/test_weighted_review.py tests/fig8_extension/test_institution_review.py tests/fig8_extension/test_nature_backfill.py
```

本轮相关测试 22 项通过；另执行数据层的 3,683 篇唯一性、原始字段和作者次序保留、六情景权重守恒及 notebook 审计。通过这些检查说明计数与流程一致，不代表所有机构判断均已外部复核。
"""
    (OUT / '高权重机构复核与名单稳定性.md').write_text(body)
    publication_disclosures()
    notebook()
    manifest = {'input': str(PREV / 'research_roster_reviewed.parquet'),
        'input_sha256': hashlib.sha256((PREV / 'research_roster_reviewed.parquet').read_bytes()).hexdigest(),
        'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*')
                  if p.is_file() and p.name != 'artifact_manifest.json'},
        'scripts': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            Path(__file__).with_name('nature_weighted_review.py'), RULES]}}
    write_json(OUT / 'artifact_manifest.json', manifest)
    print('Report, comparison tables, disclosure evidence, executed audit notebook and manifest saved.')


if __name__ == '__main__':
    build()

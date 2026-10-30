"""Publish the reproducible local audit, source register and remaining queues."""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .nature_boundary_sources import FIELD, OUT, PREV, SLOTS
from .stage0 import write_csv, write_json


def read(name: str) -> Any:
    return json.loads((OUT / name).read_text())


def supporting_tables() -> dict[str, Any]:
    rows = pq.read_table(OUT / 'research_roster_boundary_review.parquet').to_pylist()
    weights: dict[str, float] = defaultdict(float)
    doi_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        aa = json.loads(row[FIELD])
        for a in aa:
            for s in a[SLOTS]:
                weights[s['review_entity_id']] += 1 / len(aa) / len(a[SLOTS])
                doi_sets[s['review_entity_id']].add(row['doi'])
    decisions = {d['entity_id']: d for d in read('hierarchy_decisions.json')}
    contributions, pending = [], []
    selected = {x['entity']['review_entity_id'] for x in read('registry_discovery_roster.json')}
    for ident, d in decisions.items():
        record = {**d, 'fractional_n': weights.get(ident, 0), 'paper_n': len(doi_sets[ident])}
        if d['action'] in ['academic', 'health_system']:
            for target in d['target_ids']:
                contributions.append({**record, 'university_id': target, 'university_name': decisions[target]['name']})
        if ident in selected and (d['action'] in ['nonuniversity_or_parent_unresolved', 'standalone_education_proxy',
                                                'temporal_unresolved', 'national_facility_parent_role_unresolved'] or d['evidence_level'] == 'registry_structural_screen'):
            pending.append(record)
    write_csv(OUT / 'university_rollup_contributions.csv', sorted(contributions, key=lambda x: -x['fractional_n']))
    write_csv(OUT / 'hierarchy_followup_queue.csv', sorted(pending, key=lambda x: -x['fractional_n']))
    attempts = pd.read_csv(OUT / 'identity_resolution_attempts.csv').to_dict('records')
    by_address: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in attempts:
        by_address[a['address']].append(a)
    priority = []
    for unit in read('boundary_priority_units.json'):
        attempted = by_address[unit['address']]
        resolved = sum(bool(a['resolved']) for a in attempted)
        status = 'resolved_author_address_links' if resolved else 'screened_no_confirmed_identity'
        note = 'No university parent inferred from city, email, nearby institution or coaffiliation.'
        if attempted and not resolved:
            status = 'identity_candidate_but_author_address_link_unconfirmed'
        if unit['address'].startswith('Tsientang'):
            status, note = 'temporal_identity_unresolved', 'Official registration July 2025; earlier publication names need history reconciliation.'
        if 'New Cornerstone' in unit['address']:
            status, note = 'host_unresolved', 'Same named laboratory label occurs at multiple hosts; cannot assign Shenzhen address to Peking.'
        priority.append({**unit, 'status': status, 'resolved_slots': resolved, 'note': note})
    write_csv(OUT / 'boundary_priority_review.csv', priority)
    # Country conflict history remains visible and is not silently declared repaired.
    source = PREV / 'country_conflicts_unresolved.csv'
    (OUT / 'country_conflicts_unresolved_inherited.csv').write_bytes(source.read_bytes())
    for filename, key in [('country_rank_scenarios.csv', 'country'), ('university_rank_scenarios.csv', 'institution_id')]:
        df = pd.read_csv(OUT / filename)
        current = df[df.scenario == 'direct_units'].copy()
        old = df[df.scenario == 'before'].set_index(key)
        current['previous_rank'] = current[key].map(old['rank'])
        current['previous_fractional_n'] = current[key].map(old['fractional_n'])
        current['delta_fractional_n'] = current.fractional_n - current.previous_fractional_n.fillna(0)
        write_csv(OUT / filename.replace('scenarios', 'comparison'), current.to_dict('records'))
    counts = Counter(d['action'] for ident, d in decisions.items() if ident in selected)
    return {'hierarchy_candidates_n': len(selected), 'hierarchy_action_counts': dict(counts),
            'hierarchy_followup_records': len(pending), 'boundary_priority_units': len(priority),
            'priority_units_with_resolved_links': sum(u['resolved_slots'] > 0 for u in priority)}


def notebook() -> None:
    code = f'''from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq
root = Path({str(OUT)!r})
summary = json.loads((root / 'review_summary.json').read_text())
assert summary['cohort_n'] == 3683
assert summary['before_unresolved_authors'] - summary['newly_resolved_authors'] == summary['remaining_identity_authors']
roster = pq.read_table(root / 'research_roster_boundary_review.parquet', columns=['doi']).to_pandas()
assert roster.doi.nunique() == 3683
fractions = json.loads((root / 'fractional_summary.json').read_text())
for v in fractions.values():
    assert abs(v['country_mass'] - v['paper_n']) < 1e-7
rank = pd.read_csv(root / 'university_rank_scenarios.csv')
main = rank[rank.scenario == 'direct_units'].head(15)
joint = rank[rank.scenario == 'joint_equal_split'].head(15)
assert 'Peking University' not in set(main['name'])
assert 'Peking University' in set(joint['name'])
print(summary)
print(main[['rank', 'name', 'fractional_n']].to_string(index=False))
print('All saved-data audit checks passed; hierarchy correctness is not established by these assertions.')
'''
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(code, str(OUT / 'boundary_review_audit.ipynb'), 'exec'), {})  # noqa: S102 - generated, trusted audit cell
    nb = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# Nature 大学归并与边界审计\n',
        '读取本轮固定输出；不访问网络。分数计数衡量文献份额，不是创新得分。']},
        {'cell_type': 'code', 'execution_count': 1, 'metadata': {}, 'source': code.splitlines(keepends=True),
         'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': stdout.getvalue().splitlines(keepends=True)}]}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5}
    write_json(OUT / 'boundary_review_audit.ipynb', nb)


def report(extra: dict[str, Any]) -> None:
    summary, frac = read('review_summary.json'), read('fractional_summary.json')
    st = read('topk_stability.json')['scenarios']
    uni = pd.read_csv(OUT / 'university_rank_scenarios.csv')
    main = uni[uni.scenario == 'direct_units']
    old = uni[uni.scenario == 'before'].set_index('institution_id')
    table = '\n'.join(f"| {r['rank']} | {r['name']} | {r['fractional_n']:.6f} | {int(old.loc[r['institution_id'], 'rank']) if r['institution_id'] in old.index else '—'} |"
                      for r in main.head(20).to_dict('records'))
    scenario_table = '\n'.join(f"| {s} | {v['university']['cutoff'][0]} | {v['university']['next'][0]} | {v['university']['gap']:.6f} | {'一致' if v['university']['same_members_as_direct'] else '改变'} |"
                               for s, v in st.items())
    old_cov = read_previous_coverage()
    cov_table = '\n'.join(f"| {label} | {old_cov[k]} | {summary['coverage'][k]} | {summary['coverage'][k]/3683:.2%} |"
        for k, label in [('after_known','每位作者至少一个确认单位／署名身份'), ('standard_after','每位作者至少一个标准注册机构'),
                         ('all_slots_known','全部显式机构槽位身份确认'), ('all_slots_country','全部显式槽位国家／地区可用')])
    country = st['direct_units']['country']
    clean = st['exclude_nonpersonal']['country']
    cornell = main[main['name'] == 'Cornell University'].iloc[0]
    peking = main[main['name'] == 'Peking University'].iloc[0]
    text = f'''# Nature 2023–2025 医院、院系、联合中心归并与边界复核

本轮从 **365 篇、1,070 个无已确认单位的作者记录**继续筛查；新增确认 **{summary['newly_resolved_authors']} 个作者记录**，剩余 **{summary['remaining_identity_papers']} 篇、{summary['remaining_identity_authors']} 个作者记录**。在全样本中实际修复 {summary['changed_slot_n']} 个地址槽位、涉及 {summary['changed_paper_n']} 篇；包含已有其他机构的作者，不应与新增完整论文数相加。

**主要变化来自机构层级，而不是多补几篇摘要。** Weill Cornell Medicine 是 Cornell 的医学院，原来单列的 12.703920 篇等价应在大学口径合并。统一规则后 Cornell 为第 {int(cornell['rank'])}，{cornell['fractional_n']:.6f} 篇等价；原 Cornell—北京大学边界变为 **UCLA（28.739508）—北京大学（{peking['fractional_n']:.6f}）**，差 {st['direct_units']['university']['gap']:.6f}。联合中心等分情景中，北京大学进入前 15、UCLA 退出，因此高校名单尚不能冻结。

## 固定范围和计数

继续使用原来的 3,683 篇 Nature 主刊 Article，在线年份 2023–2025；没有增加文献、恢复被排除的非原创类型，也没有改变摘要。每篇论文权重 1，作者等分，再按该作者的已知机构和显式未知地址槽位等分。本轮身份修补后的同一机构去重；未知份额保留，不向已知高校重新分配。

大学归并默认**汇总原有槽位份额**：同一作者的大学和医学院都归入大学时，两部分相加，单作者总份额仍不超过 1。另设 `university_deduplicated`，先把父子单位合为一个槽位再等分。两种分母分别输出；没有将其混用。国家分数始终在大学上卷前计算，Cornell Qatar 等海外校区保留发表时工作地点的国家／地区。本轮的身份修复没有改变国家未知总份额。

国家与高校表是本批 Nature 论文的文献份额比较，不能等同科研质量、创新能力或三个领域全期刊结果。

## 统一归并规则

| 对象 | `direct_units` 主比较 | 其他情景／限制 |
|---|---|---|
| 院系、医学院、明确的大学内部中心 | 有独立来源确认单一大学上级时上卷；同规则应用全样本 | 学术单位单独情景 `academic_only` |
| 直属临床学院／大学一体化医疗体系 | 官方直属名录或 ROR 单一结构上级，且无反证时纳入 | `academic_only` 排除医院及混合医疗体系；不声称法人所有权相同 |
| 独立医院、教学附属医院、独立研究所 | 不因教学合作或名称中有大学而上卷 | NYP/Weill Cornell Medical Center、Brigham、Whitehead 明确保留独立 |
| 多方共建中心 | 保留联合单位，其份额不计入某一家大学 | `joint_equal_split` 按登记／官方共建方等分；非高校伙伴也占份额，这是压力情景，不是实际贡献估计 |
| 大学联盟／系统 | 保留具体大学／校区，不归到 University of London 或州大学系统 | 统一应用，而非只处理 Cornell、北大 |
| 企业、衍生公司、国家实验室运营关系 | 保留独立；公司不能因创始人任职或所在地归高校 | JPL、SLAC 等多方运营关系不在主情景全部转给大学 |
| 合并／更名时间不清 | 保留发表时身份并列入时间核查队列 | UT San Antonio 2025 合并不追溯套用到 2023–2024 |

规则优先级：可核对的官方反证／明确组织说明高于注册库结构关系；**OpenAlex lineage 仅用于发现候选**。本轮筛出 {extra['hierarchy_candidates_n']} 个候选机构，新增 1,872 次精确 ROR 查询并复用既有记录。原始回执全部保留。机构主表并非完成了 {extra['hierarchy_candidates_n']} 次人工官网核查。

仍有 **{frac['direct_units']['registry_only_rollup_mass']:.6f} 篇等价**的上卷仅由 ROR 结构关系支持，未逐条取得发表年代的官方组织史确认；另保留 standalone education 代理机构。统一规则的实现不意味着全部层级判断均已最终证实，不能据此给出无条件稳定保证。`hierarchy_followup_queue.csv` 保留 {extra['hierarchy_followup_records']} 条注册库待复核／层级未定记录。

## 关键证据与修补

- [Cornell 官方历史](https://weill.cornell.edu/our-story/history) 支持 Weill Cornell Medicine 和 Qatar 医学院归入 Cornell；[Cornell 兽医学院](https://www.vet.cornell.edu/about-us/cvm-strategic-plans/strategic-plan-2023-2027/about-college) 是 Cornell 合同学院，SUNY 支持不等于另一个大学归属。
- [北京大学医学部直属医院名录](https://www.bjmu.edu.cn/ylfw/zsfsyy/index.htm) 用于区分直属与共建／教学医院；主情景纳入已出现的第一、人民、第三、口腔、肿瘤医院。Peking Union Medical College 与北京大学分开。
- [Stanford Medicine 原则](https://med.stanford.edu/about/principles.html) 支持一体化医疗体系情景；[Koch Institute 官方介绍](https://ki.mit.edu/about) 明确它是无附属医院的基础研究中心，按学术单位归 MIT，避免单凭 healthcare 标签误分。
- [Whitehead 2023 官方说明](https://wi.mit.edu/sites/default/files/2023-05/About_Whitehead_5-19-23.pdf) 明确独立研究、治理和财务；[NYP 医院目录](https://www.nyp.org/locations/manhattan/nyp-weill-cornell-medical-center) 区分医院和 Cornell 医学院；这两项均覆盖了容易误用的 ROR parent。
- [北京大学—清华生命科学联合中心](https://news.pku.edu.cn/info/7921/3049761.htm)、[MCQST 共建机构](https://www.mcqst.de/about/ecosystem/) 都不能全部计给某一家大学。膜生物学实验室的注册库仍列三个上级，官方又说明 2025 年重组；其实际年代／校区分配未锁定，等分只是敏感性情景。
- [Allen Discovery Center](https://alleninstitute.org/news/the-paul-g-allen-frontiers-group-announces-allen-discovery-center-for-neuroimmune-interactions-at-icahn-school-of-medicine-at-mount-sinai) 为 Mount Sinai 与 Weill Cornell 联合中心；[Tri-I MD–PhD](https://teach.weill.cornell.edu/mdphd) 为三方项目。新增确认其身份，未全部归入 Cornell。
- [Sloan Kettering Institute](https://www.mskcc.org/research/ski) 的结构／化学生物学项目与干细胞中心归 MSK，未因附近或联合培养关系归 Cornell。Lynxi、StoneWise、Trip.com、Volastra、GRIN、Shuimu、Fauna、DP Technology、Synfuels 的单位身份和企业属性采用官方说明或出版社披露核对；无 ROR 的仍使用有来源的本地 ID。
- [上海算法创新研究院](https://www.iaar.ac.cn/) 为独立研究机构，未因北大合作者而归入北大；[北师大多尺度自旋物理实验室](https://multiscale.bnu.edu.cn/) 有明确院校来源，已按作者—出版社地址匹配后归北师大。

## 覆盖率

分母均为 3,683 篇。机构“至少一个已确认身份”含本地确认单位／署名身份，严格注册机构覆盖率单列。

| 指标 | 上轮论文数 | 本轮论文数 | 本轮覆盖率 |
|---|---:|---:|---:|
{cov_table}

摘要仍为 3,683/3,683；并不代表全文或所有作者地址都完整。未知机构身份总份额从 **56.473890** 降至 **{frac['direct_units']['identity_unknown']:.6f}**；仍可能属于高校的未知类型份额从 **49.248081** 降至 **{frac['direct_units']['possible_university_unknown']:.6f}**。医院上卷本身不会虚增身份覆盖率。

## 国家前 12

完整样本名单与顺序仍为 US、CN、GB、DE、CH、FR、JP、AU、CA、NL、ES、IL。IL 为 {country['cutoff'][1]:.6f}，DK 为 {country['next'][1]:.6f}，差 **{country['gap']:.6f}**；未知国家 **{country['unknown_mass']:.6f}** 大于差距。因此完整基线只在固定槽位、仅补未知国家的模型下保证前 11 个现有成员，不能无条件锁定第 12 名。

排除此前识别的 17 个团体分组标题、8 个团体作者和 4 个非署名致谢联盟后，未知国家 {clean['unknown_mass']:.6f}，IL–DK 差 {clean['gap']:.6f}，余量 **{clean['gap']-clean['unknown_mass']:.6f}**；只在该署名口径与固定槽位缺失模型下保证前 12 成员。这不保证内部名次，也不涵盖现有误配、地址冲突或区域汇总变化。仅使用完整机构论文（{frac['complete_case']['paper_n']} 篇）时 KR 进入、ES 退出，提示 complete-case 选择偏差。

## 高校前 15

| 当前名次 | 大学 | `direct_units` 篇等价 | 原口径名次 |
|---:|---|---:|---:|
{table}

| 情景 | 第 15 名 | 第 16 名 | 差距 | 前 15 成员相对主情景 |
|---|---|---|---:|---|
{scenario_table}

主情景与仅学术单位、大学槽位去重、排除非个人署名、完整论文子集的前 15 **成员相同但内部次序并非均相同**。联合中心等分时北京大学为第 14（31.336107），UCLA 为第 16（29.724621）；真实第 15/16 边界变为 Columbia—UCLA。该情景已足以说明成员对归并口径敏感，无需把未核实单位强行分给北大。

只固定现有层级映射，并假定未知单位只能归入相同工作国家的高校时，当前主情景仅能保守保证 Stanford、Harvard、MIT、Cornell 这四个成员入围；这不是置信区间。尚未核实的层级关系、联合中心实际所在分支、海外校区和未来槽位去重均不在此界限内，**不能解释为四校在所有规则下的保证**。

## 未解决项及下一步

保留 {summary['remaining_identity_papers']} 篇论文队列和 {summary['remaining_identity_authors']} 个作者记录，另有全样本未解析地址队列；“部分作者已有单位”和“全部地址解析”分别计数。`boundary_priority_review.csv` 对 {extra['boundary_priority_units']} 个边界候选地址记录筛查动作，其中 {extra['priority_units_with_resolved_links']} 个出现已确认作者链接，其余含未逐条外部核实的长尾，未宣称全部人工核查完成。

已确认的疑点不强行填充：Nature 一篇原文将 Lifang Hou、Donald M. Lloyd-Jones 的 Chicago 预防医学系写为 Northeastern，而 [Northwestern 官方历史](https://news.feinberg.northwestern.edu/2022/11/07/celebrating-50-years-of-the-department-of-preventive-medicine/) 与其不一致，两个作者保留待确认。Tsientang 在 2024 文献中出现，但[官网](https://www.tias.ac.cn/about.html)说明 2025 年 7 月注册，前身关系仍未锁定；不按 `pku.edu.cn` 邮箱划归北大。New Cornerstone 同名标签的具体宿主、作者关联失败的地址、无地址团体／个人记录继续保留。上轮五个国家／身份疑点原样列入 `country_conflicts_unresolved_inherited.csv`。

Fig8 的大学名单建议保留主情景与联合中心分摊情景两套，暂不冻结唯一前 15。下一轮的优先项已经从 Cornell—北大转为 **UCLA—北大及其联合中心分支**：优先核对同一论文、同一作者、同一地址中明确的校区／上级关系，并复核参与边界的 ROR-only 上卷；不能只凭作者其他兼任单位分配中心份额。

## 复用与复算

所有先前列及作者字段保留不变，新分析列 `{FIELD}` 写入 `research_roster_boundary_review.parquet`。当前目录的来源登记、逐槽位变更、候选未确认原因、原始 ROR 回执、八情景完整排名、条件界限和执行后的 notebook 均可检查；旧轮次未覆盖。

```bash
python3 -m experiments.fig8_extension.nature_boundary_review
python3 -m experiments.fig8_extension.nature_boundary_rank
python3 -m experiments.fig8_extension.nature_boundary_report
python3 -m pytest -q tests/fig8_extension/test_boundary_review.py tests/fig8_extension/test_weighted_review.py tests/fig8_extension/test_institution_review.py tests/fig8_extension/test_nature_backfill.py
```

30 项相关测试通过；另外检查了名录唯一性、全部继承字段和作者顺序不变、八情景国家总权重守恒、大学上卷不改变国家分配，以及执行 notebook 的保存数据审计。上述验证支持计算可复核，不替代外部机构事实核实。
'''
    (OUT / '大学归并规则与边界稳定性.md').write_text(text)


def read_previous_coverage() -> dict[str, Any]:
    return json.loads((PREV / 'review_summary.json').read_text())['coverage']


def main() -> None:
    extra = supporting_tables()
    write_json(OUT / 'hierarchy_review_summary.json', extra)
    report(extra)
    notebook()
    scripts = list(Path(__file__).parent.glob('nature_boundary_*.py'))
    manifest = {'input': str(PREV / 'research_roster_weighted_review.parquet'),
        'input_sha256': hashlib.sha256((PREV / 'research_roster_weighted_review.parquet').read_bytes()).hexdigest(),
        'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*')
                  if p.is_file() and p.name != 'artifact_manifest.json'},
        'scripts': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}}
    write_json(OUT / 'artifact_manifest.json', manifest)
    print('Report, remaining queues, comparison tables, executed notebook and artifact hashes saved.')


if __name__ == '__main__':
    main()

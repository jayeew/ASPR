"""Write auditable coverage, unresolved queues and branch sensitivity readout."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq

from .nature_joint_branch_review import FIELD, INPUT, OUT, PREV, SLOTS
from .stage0 import write_csv, write_json


def main() -> None:
    rows = pq.read_table(OUT / 'research_roster_joint_review.parquet').to_pylist()
    summary = json.loads((OUT / 'review_summary.json').read_text())
    stability = json.loads((OUT / 'topk_stability.json').read_text())['scenarios']
    fractions = json.loads((OUT / 'fractional_summary.json').read_text())
    audit = json.loads((OUT / 'branch_address_audit.json').read_text())
    queue = list(csv.DictReader((PREV / 'remaining_papers.csv').open(encoding='utf-8-sig')))
    remaining = {r['doi'] for r in queue}
    counts: Counter[str] = Counter()
    priority = []
    for row in rows:
        aa = json.loads(row[FIELD])
        old_authors = json.loads(row['authorships_boundary_review_json'])
        known = [any(s['identity_status'] != 'identity_unresolved' for s in a[SLOTS]) for a in aa]
        counts['all_authors_known'] += all(known)
        counts['unresolved_authors'] += sum(not k for k in known)
        counts['all_slots_known'] += all(s['identity_status'] != 'identity_unresolved' for a in aa for s in a[SLOTS])
        counts['all_countries_known'] += all(s.get('site_country_codes') or s.get('country_code') for a in aa for s in a[SLOTS])
        names = {s['display_name'] for a in aa for s in a[SLOTS]}
        affected = sorted(names & {'Peking University', 'University of California, Los Angeles',
                                  'Columbia University', 'University of California, Berkeley'})
        if row['doi'] in remaining:
            units = [s['display_name'] for a in aa for s in a[SLOTS] if s['identity_status'] == 'identity_unresolved']
            priority.append({'doi': row['doi'], 'boundary_coaffiliations_discovery_only': affected,
                             'unknown_units': sorted(set(units)), 'action': 'retain_unresolved_no_new_identity_evidence'})
        for i, a in enumerate(aa):
            assert all(a[k] == value for k, value in old_authors[i].items())
            for old, new in zip(a['institution_boundary_review_entities'], a[SLOTS], strict=True):
                assert all(new[k] == value for k, value in old.items())
    assert counts['all_authors_known'] == 3326 and counts['unresolved_authors'] == 1051
    assert counts['all_slots_known'] == 2959 and counts['all_countries_known'] == 3656
    write_json(OUT / 'recomputed_coverage.json', dict(counts))
    priority.sort(key=lambda r: (-bool(r['boundary_coaffiliations_discovery_only']), r['doi']))
    write_csv(OUT / 'remaining_357_screen.csv', priority)
    write_csv(OUT / 'unresolved_branches.csv', [a for a in audit if a['status'] != 'confirmed_branch'])
    by_entity: dict[str, Any] = defaultdict(lambda: {'confirmed_n': 0, 'unresolved_n': 0, 'confirmed_mass': 0.0, 'unresolved_mass': 0.0})
    for a in audit:
        kind = 'confirmed' if a['status'] == 'confirmed_branch' else 'unresolved'
        by_entity[a['entity']][kind+'_n'] += 1
        by_entity[a['entity']][kind+'_mass'] += a['weight']
    ranks = list(csv.DictReader((OUT / 'university/rank_scenarios.csv').open(encoding='utf-8-sig')))
    overlap = len(remaining & {a['doi'] for a in audit})
    summary.update(reviewed_branch_papers_in_remaining357=overlap,
        remaining357_boundary_coaffiliation_papers=sum(bool(p['boundary_coaffiliations_discovery_only']) for p in priority),
        unresolved_branch_slots=sum(a['status']=='unresolved_branch' for a in audit),
        unresolved_branch_mass=sum(a['weight'] for a in audit if a['status']=='unresolved_branch'),
        ror_only_rollup_mass_before=fractions['previous_main']['registry_only_rollup_mass'],
        ror_only_rollup_mass_after=fractions['direct_units']['registry_only_rollup_mass'])
    write_json(OUT / 'review_summary.json', summary)
    lines = ['# Fig8 联合中心分支与高校边界复核', '',
        '锁定 Nature 主刊 2023–2025 年 3,683 篇原创 Article；前序数据不覆盖。本轮是作者署名与高校归并审计，不是创新质量或影响力结果。', '',
        '## 完成范围与覆盖率', '',
        (f'筛查保留队列全部 357 篇，其中 {summary["remaining357_boundary_coaffiliation_papers"]} 篇涉及 UCLA、北大、Columbia 或 Berkeley 的共同署名，仅用于排查优先级。'
        f'复用/解析 115 篇出版社页面，对 72 篇的 376 个联合中心作者机构槽位逐条核对；其中 {overlap} 篇属于 357 篇队列，其余属于全队列的排名归并审查。'), '',
        '确认 55 篇论文的 285 个槽位分支。未新增完全缺失机构的身份映射；357 篇/1,051 条作者记录仍保留未解决，不能把分支核实数当作补齐论文数。', '',
        '|覆盖指标|篇数 / 3,683|比例|', '|---|---:|---:|',
        '|摘要|3,683|100%|', '|所有作者至少一个已确认机构|3,326|90.31%|',
        '|所有作者至少一个标准机构 ID|3,270|88.79%|', '|全部署名槽位机构已知|2,959|80.34%|',
        '|全部署名槽位国家已知|3,656|99.27%|', '',
        '## 归并规则与来源', '',
        '沿用每篇→作者→机构槽位分数计数。仅在同一条出版社作者地址明确列出大学主体时赋予联合中心分支，保留原机构 ID、国家和全部旧字段。共同中心名称本身、另一条高校署名、邮箱、城市均不能单独确认分支。医院继续按上一轮的直属临床体系/独立教学医院规则处理。', '',
        'CNSI 的 UCLA 与 UCSB 分支分别保留；生命科学联合中心和膜生物学实验室按署名明确的北大/清华分支处理。未确认分支在主口径保留为独立槽位，不重新分配其份额。', '',
        '|中心|已确认槽位|未确认槽位|已确认份额|未确认份额|', '|---|---:|---:|---:|---:|']
    for name, v in by_entity.items():
        lines.append(f'|{name}|{v["confirmed_n"]}|{v["unresolved_n"]}|{v["confirmed_mass"]:.6f}|{v["unresolved_mass"]:.6f}|')
    lines += ['', '七条原仅 ROR 支持的学术子单位关系取得官方依据：北大蛋白质与植物基因、人工微结构与介观物理、核物理与核技术三个重点实验室；Columbia 的 Lamont；Oxford 的 Kennedy 和 Centre for Human Genetics；Berkeley 的 Space Sciences Laboratory。关系证据升级不等于所有作者的旧 OpenAlex 匹配都已独立证明，出版社地址审计单列保存。', '',
        f'全队列仅 ROR 支持的归并份额由 {summary["ror_only_rollup_mass_before"]:.6f} 降至 {summary["ror_only_rollup_mass_after"]:.6f}；其余关系继续列入核查表。', '',
        '官方时间信息需要保留：量子物质中心的 2023 年介绍记载历史“两校一所”，同时预告未来由两校承担建设，不能据此确定整个 2023–2025 年的单一伙伴分母。膜生物学实验室记载 2024 年 12 月重组，历史三分室与新两校组织不能无条件等同。主分析只采用明确分支；未明确分支另做三方/两方分配敏感性，均不是历史归属真值。', '',
        '## 国家前 12', '',
        '全样本国家分配完全未变，前 12 仍为 US、CN、GB、DE、CH、FR、JP、AU、CA、NL、ES、IL。第 12 位 IL=40.464973，第 13 位 DK=39.507276，差 0.957698，小于未知国家份额 1.412272，不能给出无条件稳定保证。固定现有映射、只分配缺失国家时，当前 11 个成员得到下界保证。', '',
        '排除 29 条明确团体/标题记录的敏感性分析，未知国家份额降至 0.938649，边界差 0.957799，只有约 0.019150 裕量；这依赖排除规则。完整案例筛选仍由 KR 替换 ES，提示缺失选择偏差。', '',
        '## 高校前 15', '',
        '主口径北大由第 16 升至第 12，UCLA 保持第 15；Columbia 由第 14 退至第 16。北大新增分支份额 3.843073，UCLA 新增 1.938974。Berkeley 与 UCLA 仅差 0.001029，UCLA 与 Columbia 仅差 0.046536。', '',
        '|排名|高校|分数计数|', '|---:|---|---:|']
    for r in ranks:
        if r['scenario']=='direct_units' and int(r['rank'])<=17:
            lines.append(f'|{r["rank"]}|{r["name"]}|{float(r["fractional_n"]):.6f}|')
    lines += ['', '|敏感性口径|第15位|第16位|相对主口径成员变化|', '|---|---|---|---|']
    for key, v in stability.items():
        u = v['university']
        changes = '无' if u['same_members_as_direct'] else '进入 '+', '.join(u['entered_vs_direct'])+'；退出 '+', '.join(u['left_vs_direct'])
        lines.append(f'|{key}|{u["cutoff"][0]} ({u["cutoff"][1]:.6f})|{u["next"][0]} ({u["next"][1]:.6f})|{changes}|')
    lines += ['', '主口径、未确定中心均分的两种情景、大学去重、排除团体以及去除剩余仅 ROR 归并的情景，前 15 成员相同，但内部顺序不完全相同。只计学术单位、排除临床医院体系时，北大退出而 Columbia 进入。完整案例也会改变成员。因此 Fig8 高校前 15 仍依赖归并定义，不能称为稳定最终名单。', '',
        f'仍有 {summary["unresolved_branch_slots"]} 个分支槽位、{summary["unresolved_branch_mass"]:.6f} 份额无法确定。机构身份未知份额 54.574597，其中可能高校 47.671565；这与分支不确定性是两种不同缺失，不能直接混称或当作置信区间。固定映射且按工作国家约束的缺失分配检验仍仅保证 Stanford、Harvard、MIT、Cornell 四个当前前 15 成员；不覆盖海外校区、父子关系和分母调整。', '',
        '## 复现与未解决项', '',
        '`research_roster_joint_review.parquet` 保存新增审计层；`branch_address_audit.csv`、`unresolved_branches.csv` 保存作者—地址证据与未确认记录；`remaining_357_screen.csv` 是全队列筛查记录，`remaining_papers.csv` 和 `remaining_identity_authors.csv` 保留原缺失项。`remaining_ror_only_rollups.csv` 保留未完成官方验证的高校归并关系。', '',
        '独立医院不因教学关系归入高校；New Cornerstone 共用标签、Tsientang 时间冲突和 Northeastern/Northwestern 署名冲突等继续未解决。不把“未发现问题”当作独立核实。', '',
        '复现：`python3 -m experiments.fig8_extension.nature_joint_branch_review` → `python3 -m experiments.fig8_extension.nature_joint_branch_rank` → `python3 -m experiments.fig8_extension.nature_joint_branch_report`。本轮源页面请求中 4 个直接请求返回 403，另存官方网页的检索/浏览回执作为补充，未伪报为直接抓取成功。', '',
        '## 官方证据索引', '']
    rules = json.loads((OUT / 'official_rollup_rules.json').read_text())
    for name, (_, url) in rules.items():
        lines.append(f'- [{name}]({url})')
    lines += ['- [CNSI UCLA](https://cnsi.ucla.edu/about-us/)',
              '- [CNSI UCSB](https://www.cnsi.ucsb.edu/about)',
              '- [生命科学联合中心](https://cls.tsinghua.edu.cn/Introduction/)',
              '- [膜生物学实验室历史与重组](https://www.biomembrane.tsinghua.edu.cn/p/shiyanshijianjie.html)',
              '- [量子物质中心 2023 年介绍](https://110.phy.pku.edu.cn/info/1076/1222.htm)']
    (OUT / '联合中心分支与名单稳定性.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    code = ("from pathlib import Path\nimport json\np = Path("+repr(str(OUT))+ ")\n"
            "s = json.loads((p/'review_summary.json').read_text())\n"
            "assert s['cohort_n'] == 3683 and s['remaining_identity_papers'] == 357\n"
            "print(json.dumps(s, ensure_ascii=False, indent=2))\nprint((p/'topk_stability.json').read_text())")
    executed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    write_json(OUT / 'joint_branch_audit.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': 'code', 'id': 'actual-result-check', 'metadata': {}, 'source': code.splitlines(True),
                   'execution_count': 1, 'outputs': [{'output_type': 'stream', 'name': 'stdout',
                                                   'text': executed.stdout.splitlines(True)}]}]})
    validation = json.loads((OUT / 'validation.json').read_text())
    validation.update(coverage_recomputed=True, inherited_author_and_slot_fields_unchanged=True)
    write_json(OUT / 'validation.json', validation)
    write_json(OUT / 'artifact_manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'input': str(INPUT), 'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

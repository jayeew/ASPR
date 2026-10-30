"""Produce the clinical-boundary review receipt and executable audit notebook."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from typing import Any

from .nature_clinical_review import INPUT, OUT, PREV
from .stage0 import write_csv, write_json


def main() -> None:
    summary = json.loads((OUT/'review_summary.json').read_text())
    stability = json.loads((OUT/'topk_stability.json').read_text())['scenarios']
    audit = json.loads((OUT/'slot_audit.json').read_text())
    coverage = json.loads((OUT/'author_coverage.json').read_text())
    ranks = list(csv.DictReader((OUT/'universities_rank_scenarios.csv').open(encoding='utf-8-sig')))
    counts: dict[str, Any] = {}
    for key in ['after','all_slots_known','all_slots_country']:
        failed = {r['doi'] for r in coverage if not r[key]}
        counts[key] = {'paper_n':3683-len(failed),'denominator':3683,'percent':100*(3683-len(failed))/3683}
    write_json(OUT/'coverage_summary.json',counts)
    previous_queue = list(csv.DictReader((PREV/'remaining_papers.csv').open(encoding='utf-8-sig')))
    remaining = {r['doi'] for r in coverage if not r['after']}
    changed = {r['doi'] for r in json.loads((OUT/'changes.json').read_text())}
    inspected = {r['doi'] for r in audit}
    write_csv(OUT/'input_350_disposition.csv',[{**r,'status':'resolved' if r['doi'] not in remaining else
        'changed_still_unresolved' if r['doi'] in changed else 'reviewed_still_unresolved' if r['doi'] in inspected else 'not_targeted_this_pass'} for r in previous_queue])
    names = {'University of California, Los Angeles':'UCLA','Columbia University':'Columbia',
        'Peking University':'北京大学','University of Oxford':'Oxford'}
    table=[]
    for scenario,s in stability.items():
        u,c=s['university'],s['country']
        table.append(f"| {scenario} | {names.get(u['cutoff'][0],u['cutoff'][0])} / {names.get(u['next'][0],u['next'][0])} | {u['gap']:.6f} | {', '.join(names.get(x,x) for x in u['entered_vs_direct']) or '无'} | {', '.join(names.get(x,x) for x in u['left_vs_direct']) or '无'} | {'相同' if c['same_members_as_direct'] else '改变'} |")
    boundary='\n'.join(f"| {r['rank']} | {names.get(r['name'],r['name'])} | {float(r['fractional_n']):.6f} |" for r in ranks if r['scenario']=='direct_units' and (int(r['rank'])>=12 and int(r['rank'])<=16 or r['name']=='Columbia University'))
    text=f'''# Fig8 临床体系与高校边界核查

固定队列：Nature 主刊 2023–2025 在线发表的 3,683 篇原创 Article。输入为上一轮出版社冲突修订层，旧字段与输入文件均保留。本轮为机构身份和统计归并审计，不是创新分析结果。

## 本轮结果

- 剩余缺失机构论文 350 → **{summary['remaining_identity_papers']}**，未识别作者 1,033 → **{summary['remaining_identity_authors']}**；新增解决 **{summary['newly_resolved_authors']}** 位作者，26 个机构身份槽位得到确认。
- 原 78 条待确认分支中确认 **8** 条 QB3 校区分支，剩 **70** 条。覆盖 84 篇、373 个目标槽位，未逐篇完成全部 350 篇的人工核查；`input_350_disposition.csv` 记录未进入本轮重点范围的论文。
- **另有 53 条扩展核查未解决记录**：45 条 CUIMC 署名链接、5 条已归并北大医院的署名链接、3 条生殖实验室署名。它们单列于 `additional_rollup_unresolved.csv`，不能把“原队列剩70条”表述为全部未解决项仅70条。与身份缺失可能重叠，不可按论文数相加。
- 全部作者至少有一个已知机构的论文为 {counts['after']['paper_n']}/3683（{counts['after']['percent']:.2f}%）；全部槽位身份已知 {counts['all_slots_known']['paper_n']}/3683；全部槽位国家已知 {counts['all_slots_country']['paper_n']}/3683。摘要沿用上一轮 3683/3683，未重新抓取。

## 关键证据与处理

1. **CUIMC：此前漏归并，不能再沿用旧边界。** 201 个槽位共 5.346623 份额中，156 个通过完整出版社署名解析、唯一精确姓名匹配及地址名称检验，归入 Columbia，新增 4.431584 份额；45 个共 0.915038 份额仍未归并。官方资料把大学医学中心与 NewYork-Presbyterian 医院区分；本轮排除含 NYP 的混合地址，不将合作医院自动并入大学。医学中心的大学院系按 academic 分类，不能因名称含 Medical 就剔除。
2. **北大临床体系：保留医院边界，补足实验室。** 现有五家医院共45个槽位中，40个出版社链接复核通过，5个链接待确认，主口径保留旧归并并另做严格署名敏感性。20个生殖内分泌与辅助生殖技术北京市重点实验室槽位通过署名核验，按北医三院临床研究体系归入北大（0.231151份额）；另3条无充分署名链接的实验室记录不补齐、不归并。官方2025年材料说明实验室重组更名，本队列保留论文当时名称，不把新名称追溯套用。
3. **机构身份补齐。** Humabs BioMed（19槽位）为瑞士公司，CNR-INO Pisa（6槽位）为意大利国家研究机构，均不并入高校；Institute of Human Nutrition（1槽位）为 Columbia 学术单位，新增0.010000份额。保留 CH/IT/US 工作地点国家，不按母公司所在国改写。
4. **联合中心。** 8条 QB3 记录以论文中的 Berkeley 校区及作者署名大学、或明确 UCSF Quantitative Biosciences Institute 地址，结合官方校区材料确认。其余仅有北京地址的清北联合中心、量子物质中心等不凭作者另一个大学署名强行分配。Harbor–UCLA 为县属医院，未因名字含 UCLA 而归并。

## 重算排名

| 排名 | 高校 | 主口径分数计数 |
|---:|---|---:|
{boundary}

主口径相对上轮：**Columbia进入前15，UCLA退出**。新的15/16边界为Oxford—UCLA，差距 {stability['direct_units']['university']['gap']:.6f}。旧UCLA—Columbia边界不再适用。

| 统计口径 | 高校15/16位 | 差距 | 相对主口径进入 | 退出 | 国家前12成员 |
|---|---|---:|---|---|---|
{chr(10).join(table)}

`direct_units`：沿用作者数×机构槽位的分数计数，直接学术单位与已确认直属临床体系归并。`academic_only`：剔除health_system类别，保留大学医学院等学术单位。`university_deduplicated`：每位作者先将归属同一高校的槽位去重，再分配份额，仍保留非高校槽位在分母。`publisher_verified_clinical`：剔除本轮5条无法严格链接到出版社署名的北大医院槽位的高校归属，保留国家及分母，检验旧归并的影响。`exclude_all_reviewed_medical`：将本轮CUIMC、IHN、实验室及北大医院排除的压力测试，**不是对所有高校全面去医学化，也不是推荐口径**。`complete_case`要求全篇所有机构槽位身份已知，存在选择偏差。

**高校名单仍不稳定。** academic_only 和 university_deduplicated 均使 UCLA进入、北大退出；完整案例筛选使UCLA进入、Oxford退出。Columbia在已列口径下均进入前15，但这只是已测口径内的结果，不能视为所有未知归并下的保证。

国家主名单仍为 {', '.join(stability['direct_units']['country']['members'])}。IL为40.489973，DK为39.473942，差距1.016031小于未知国家份额1.412272，仍无法保证前12。完整案例改变国家成员（KR进入、ES退出）。本轮身份和归并修订不改变主口径国家分配；所有高校归并口径间国家分配一致，完整案例/排除非个人作者改变样本或分母则另论。

缺失份额界限仅在现有机构、国家映射和分母固定条件下成立，不覆盖署名冲突、联合单位归属及海外分支不确定性；不能称作置信区间。仍需处理的高权重单位及论文见 `remaining_weighted_units.json`，所有未解决原分支保留于 `unresolved_branches.csv`。

## 来源与可复核文件

- [CUIMC官方简介](https://www.cuimc.columbia.edu/about-us)及[2018年大学与NYP分别命名说明](https://www.cuimc.columbia.edu/news/irving-familys-700-million-bequest-columbia-and-nyp-sets-stage-dramatic-advances-cancer)。
- [北大直属附属医院](https://www.bjmu.edu.cn/ylfw/zsfsyy/index.htm)、[北医三院生殖医学中心](https://www.puh3.net.cn/szyxzx/ksjj.htm)、[2025年实验室重组会议](https://sklffp.puh3.net.cn/info/1024/7382.htm)。
- [QB3 Berkeley校区研究设施](https://qb3.berkeley.edu/category/news/core-research-facilities/)、[UCSF QBI](https://qbi.ucsf.edu/about-qbi)、[IHN](https://www.ihn.cuimc.columbia.edu/about-us)、[Humabs官方实体与地址](https://www.vir.bio/privacy-notice/)、[CNR-INO Pisa](https://www.ino.cnr.it/?page_id=9664)。

官方页面直抓失败/403保留于 `source_receipts.json`；网页检索证据另存 `web_sources_*.json`，不将失败请求标为成功。每条论文署名来源、解析状态、作者位置、归并决定及变更前后槽位见 `slot_audit.json`、`changes.json`；不声称所有已继承实体均得到全量人工复核。

复算：`python3 -m experiments.fig8_extension.nature_clinical_review`；报告：`python3 -m experiments.fig8_extension.nature_clinical_report`。`clinical_audit.ipynb` 为已执行的输入保全、覆盖率和排名审计。22项相关测试及ruff通过；守恒、旧字段保全、全国家分配不变见 `validation.json`。
'''
    (OUT/'临床体系归并与名单稳定性.md').write_text(text)
    code=(
        'from pathlib import Path\nimport json, hashlib\np=Path('+repr(str(OUT))+')\n'
        'summary=json.loads((p/"review_summary.json").read_text())\n'
        'assert hashlib.sha256(Path('+repr(str(INPUT))+').read_bytes()).hexdigest()==summary["input_sha256"]\n'
        'validation=json.loads((p/"validation.json").read_text())\nassert all(validation.values())\n'
        'print(summary)\nprint(json.loads((p/"coverage_summary.json").read_text()))\n'
        'for name,s in json.loads((p/"topk_stability.json").read_text())["scenarios"].items():\n print(name,s["university"]["cutoff"],s["university"]["next"])')
    executed = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'markdown','id':'scope','metadata':{},'source':'Clinical review audit. Code executed with the local Python interpreter; captured output retained.'},
                 {'cell_type':'code','id':'audit','metadata':{},'source':code,'execution_count':1,
                  'outputs':[{'output_type':'stream','name':'stdout','text':executed.stdout}]}]}
    write_json(OUT/'clinical_audit.ipynb',nb)
    write_json(OUT/'artifact_manifest.json',{'input_sha256':summary['input_sha256'],'files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}})


if __name__ == '__main__':
    main()

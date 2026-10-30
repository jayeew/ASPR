"""Package source receipts, completeness report and an executed release audit."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .nature_release_prepare import INPUT, OUT
from .stage0 import write_csv, write_json


def discover_paths(value: Any, found: set[Path]) -> None:
    if isinstance(value,dict):
        for v in value.values():
            discover_paths(v,found)
    elif isinstance(value,list):
        for v in value:
            discover_paths(v,found)
    elif isinstance(value,str) and len(value)<1000 and value.startswith(('/home/jayee/workspace/ASPR/','outputs/fig8_data/')):
        path=Path(value)
        if path.suffix in ['.json','.gz','.pdf','.html']:
            found.add(path.resolve())


def bundle_sources() -> list[dict[str,Any]]:
    paths: set[Path]=set()
    for f in ['slot_dispositions.json','registry_query_log.json','new_official_sources.json','high_frequency_official_sources.json']:
        discover_paths(json.loads((OUT/f).read_text()),paths)
    for f in (OUT/'publisher_records').glob('*.json'):
        discover_paths(json.loads(f.read_text()),paths)
    for line in (OUT/'papers.jsonl').open():
        discover_paths(json.loads(line),paths)
    extra=set()
    for p in paths:
        for extension in ['.gz','.raw.gz']:
            side=p.with_suffix(extension)
            if side.exists():
                extra.add(side)
    paths |= extra
    inventory=[]
    folder=OUT/'evidence';folder.mkdir(exist_ok=True)
    for p in sorted(paths):
        if not p.is_file():
            inventory.append({'original_path':str(p),'status':'referenced_file_unavailable'})
            continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        destination=folder/(digest+p.suffix)
        if not destination.exists():
            if p.is_relative_to(OUT.parent.resolve()):
                destination.symlink_to(os.path.relpath(p,destination.parent.resolve()))
            else:
                shutil.copyfile(p,destination)
        inventory.append({'original_path':str(p),'status':'bundled','sha256':digest,'portable_path':str(destination.relative_to(OUT))})
    write_json(OUT/'source_inventory.json',inventory)
    return inventory


def main() -> None:
    p=OUT
    summary=json.loads((p/'completeness_summary.json').read_text())
    stability=json.loads((p/'stability.json').read_text())
    data=json.loads((p/'expected_results.json').read_text())
    names=json.loads((p/'entity_names.json').read_text())
    queue=json.loads((p/'priority123_dispositions.json').read_text())
    observed=summary['observed'];personal=summary['personal'];n=summary['cohort_n']
    countries=list(csv.DictReader((p/'country27_dispositions.csv').open(encoding='utf-8-sig')))
    missing=list(csv.DictReader((p/'missing348_dispositions.csv').open(encoding='utf-8-sig')))
    remaining=[r for r in missing if r['status']!='resolved']
    write_csv(p/'remaining_papers.csv',remaining)
    write_csv(p/'remaining_country_papers.csv',[r for r in countries if r['status']!='resolved'])
    write_csv(p/'remaining_priority_records.csv',[{**r,**r['disposition']} for r in queue if r['disposition']['final_status'].startswith('unresolved')])
    counts=[]
    for mode,d in [('原始观察记录',observed),('固定主口径：个人作者',personal)]:
        for key,label in [('all_authors_one_identity','每位作者至少一个已知机构'),('all_slots_identity','所有机构槽位身份已知'),('all_slots_country','所有机构槽位国家已知')]:
            counts.append(f"| {mode} | {label} | {d[key]}/{n} | {d[key]/n*100:.2f}% |")
    scenarios=[]
    for scenario,result in stability.items():
        u,c=result['universities'],result['countries']
        scenarios.append(f"| {scenario} | {data[scenario]['paper_n']} | {names[u['cutoff'][0]]} / {names[u['next'][0]]} | {u['gap']:.6f} | {', '.join(names[i] for i in u['entered_vs_main']) or '无'} | {', '.join(names[i] for i in u['left_vs_main']) or '无'} | {'相同' if c['same_members'] else '变化'} |")
    sources=bundle_sources()
    country=stability['main_personal_direct']['countries'];unknown=data['main_personal_direct']['countries'].get('unknown',0)
    oldunknown=data['legacy_observed_direct']['countries'].get('unknown',0)
    outside=[r for r in csv.DictReader((p/'unresolved_slots.csv').open(encoding='utf-8-sig')) if r['in_closure_scope']=='False']
    report=f'''# Nature 数据准备收尾与完整性报告

## 交付结论

**已完成约定队列的系统处置并形成可复算的当前数据表；“处置完成”不等于缺失清零。**
当前主表为Nature主刊2023–2025在线发表的{n:,}篇原创Article。以下348/27/48篇是既有收尾核查范围；新增记录另行留痕。348篇机构缺失、27篇国家缺口、48篇重点归属论文合并为400篇，已全部读取出版社记录、逐作者检查现有身份，并为每个待处理槽位保留来源与最终状态。本轮不扩展到Fig8三个领域全期刊样本，也不重新定义论文创新指标。

- 348篇初始缺失论文：5篇达到全部观察作者至少一个已知机构；**343篇保留明确未知项**，没有“尚未检查”的初始队列论文。
- 原1,012条无已知机构的作者记录补齐81条，剩931条。99个身份槽位获得补齐（94个官网确认单位、4个IAAI区域分支、1个既有确证单位的精确名称复用）；另修正14个不受出版社研究署名支持的槽位。身份补齐与错误删除分别统计。
- 国家缺口27篇：8篇解决，**19篇仍有未知国家**。个人作者主口径排除29条已标注非个人记录后，国家缺口涉及10篇；这部分下降来自口径变化，不能计作新增补齐。
- 重点归属123条涉及48篇：47条确认或修正，**76条保留未解决**，包括70条联合单位分支、4条署名身份冲突和2条作者链接问题。原70条联合分支不能凭同一作者的另一个高校署名强制归属。
- 对未识别名称执行786个机构注册库查询词，复用576个已有查询，新增210个；请求失败0。查无精确匹配只是“证据不足”，不是机构不存在。

## 完整性口径

摘要：{summary['abstract_n']:,}/{n:,}。原始观察作者记录{summary['observed_author_records']:,}条；主口径的个人作者记录{summary['personal_author_records']:,}条。记录数不是去重自然人数。

| 分母口径 | 完整性定义 | 完整论文数 | 覆盖率 |
|---|---|---:|---:|
{chr(10).join(counts)}

每位作者至少一个机构、所有机构槽位完整、国家信息完整，是三个不同条件。仅保留一项已知机构不能宣称整篇机构信息完整。全部{n:,}篇均保留，不以缺失作为主分析排除标准。

本轮按用户指定队列处理400篇；其余论文继承既有核查记录。全队列仍有{len({r['doi'] for r in outside})}篇包含未落入本轮范围的部分未知槽位，状态标为 `inherited_outside_requested_closure_scope`，没有冒充本轮已复核。这些论文此前每位作者至少一个机构，但并非所有槽位已知。

## 主要修订及证据边界

高频单位补齐包括Lockheed Martin Space、Leidos SEMO、NBACC、Sony AI Zurich、Seer、CISTIM和Zetta AI，按论文地点保留国家。CISTIM由KU Leuven创办不能直接推成校内单位；本版不自动归并。Zetta部分论文写NJ而美国联邦企业记录写NY，US国家一致，原始州名冲突保留。

国家修复包括Great Britain→GB、Turkey→TR、St Helena Island→SH，均要求与该作者出版社地址相连。Deep Light SA结合论文署名和公司官网补CH；IAAI Sulawesi分支结合组织自身历史资料补ID；CRA依据出版社明确Slovenia补SI。没有用作者国籍、另一机构地点或单位网址国家后缀猜填。

重点署名修订覆盖CUIMC、Vagelos、Columbia直接署名及明确误分配；未把NewYork-Presbyterian整体并入大学。两项国家缺口来自错误机构槽位，删除错误槽位而不是为错误记录填国家。仅对明确检查的作者实施删除；其他疑似冲突保留未解决。

匹配采用唯一精确作者姓名或出版社明确的机构作者列表；不做模糊姓名指派。允许个别作者的完整链接在全页其他团体无机构时成立，避免把全页解析不完整误当作所有作者都不可用。注册库候选必须匹配组织名称、国家及城市，地名不能充当组织名；所有新官网单位另保留官方URL与页面回执。

## 固定统计协议

主口径为 `main_personal_direct`：每篇权重1，按个人作者记录均分，再按其观察机构槽位均分；未知槽位仍占分母。排除已标注的团体作者、分组标题及仅致谢出现的团体，共29条，原数据不删除。大学直接学术单位及直属临床体系按已存证规则归并；联合中心没有确定分支则不向高校分配。

这是从旧观察记录口径到个人作者口径的显式调整，理由是避免把“Primary authors”等标题当作个人。**未依据排名选择口径**。`legacy_observed_direct`完整保留旧分母。主口径保留已观察机构槽位；`university_deduplicated`先按高校归属去重，是独立敏感性，不偷偷改变主分母。

国家使用工作地点ISO国家/地区代码，HK、TW等沿用独立代码；跨地点槽位平分；学校归并不改写工作地点。跨国机构、医院归属及未知分支仍是局限。输入为OpenAlex观察到的作者记录，未全面重建出版社完整作者清单，也未声称模糊重名/重复作者已全部消除。

## 口径敏感性

| 口径 | 论文数 | 高校15/16位 | 差距 | 相对主口径进入 | 退出 | 国家前12成员 |
|---|---:|---|---:|---|---|---|
{chr(10).join(scenarios)}

主口径国家前12为 {', '.join(country['members'])}。第12/13位差距{country['gap']:.6f}大于未知国家份额{unknown:.6f}，**只在现有映射和分母固定、仅分配缺失国家的条件下，前12成员可保持**。旧观察口径未知份额{oldunknown:.6f}，不能沿用个人口径保证；完整案例仍会使KR进入、ES退出。不是无条件稳定性保证或置信区间。

高校仍受临床纳入和高校去重口径影响，不能为了固定前15而丢弃未知项或调整归并规则。`complete_case`是有选择偏差的敏感性样本，不代替全队列主分析。

## 文件及离线复算

- `papers.jsonl`：{n:,}篇的可移植分析数据，含摘要、观察作者、机构身份、国家、各口径高校映射、来源字段和最终状态。
- `research_roster_release.parquet`：当前含采集字段的工作表；papers.jsonl为其分析投影。
- `protocol.json`：锁定分母、国家代码、归并及缺失处理规则。
- `missing348_dispositions.csv`、`country27_dispositions.csv`、`priority123_dispositions.csv`：初始队列逐条最终处置；`paper_dispositions.csv`为全部400篇。
- `slot_dispositions.json/csv`、`author_dispositions.json/csv`、`changes.json/csv`：每项依据、注册库查询回执、作者链接、修改前后值。
- `remaining_papers.csv`、`remaining_country_papers.csv`、`remaining_priority_records.csv`、`unresolved_slots.csv`：明确保留的未知项。不同队列有重叠，不可相加为论文总数。
- `coverage_by_paper.csv`、`completeness_summary.json`、`rankings.csv`、`stability.json`：完整性及八种口径结果。
- `evidence/`、`source_inventory.json`、`publisher_records/`、`web_close_sources_*.json`：本地来源副本与原路径映射。回执仅有HTTP200不等于页面正文可用；例如纳板河官网直抓空白，仍保留未知。当前引用源文件{sum(r['status']=='bundled' for r in sources)}项已打包，{sum(r['status']!='bundled' for r in sources)}项引用文件不可用并显式登记。

在 nature 目录执行：

```bash
python3 recompute.py . --output recomputed_results.json
```

仅需Python标准库，不联网、不需要OpenAlex或ROR凭证。脚本重新计算全部8种口径，与独立使用现有分析引擎生成的 `expected_results.json` 逐实体核对（容差1e-9）。这是对当前数据与当前归并映射的复算，不是重新抓取整个OpenAlex历史snapshot。

既有收尾核查已验证输入字段保全、唯一DOI、全部初始队列已处置、各口径国家份额守恒、独立离线复算通过。检查回执见 `validation.json` 与 `reproduction_receipt.json`。文件校验清单见 `manifest.json`；校验和保证文件可追踪，不等于机构归属或科学结论全部无误。
'''
    (p/'完整性报告.md').write_text(report)
    (p/'README.md').write_text('# Nature data\n\nSee 完整性报告.md for scope and limitations.\n\nOffline: `python3 recompute.py . --output recomputed_results.json`\n\nPrimary protocol: main_personal_direct. Missing data is explicit, not imputed.\n')
    code='from pathlib import Path\nimport json,subprocess,sys\np=Path('+repr(str(OUT))+')\nprint(json.loads((p/"completeness_summary.json").read_text()))\nr=subprocess.run([sys.executable,str(p/"recompute.py"),str(p)],check=True,capture_output=True,text=True)\nprint(r.stdout)\n'
    executed=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    write_json(p/'release_audit.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'scope','metadata':{},'source':'Release audit executed using local Python; stdout captured below.'},
        {'cell_type':'code','id':'audit','metadata':{},'execution_count':1,'source':code,'outputs':[{'output_type':'stream','name':'stdout','text':executed.stdout}]}]})
    write_json(p/'manifest.json',{'release_id':'nature','input_path':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'files':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.name!='manifest.json'}})
    print(p,flush=True)


if __name__=='__main__':
    main()

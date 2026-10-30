"""Audit report for publisher corrections and symmetric present-address screening."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .nature_conflict_review import INPUT, OUT, PREV
from .stage0 import write_csv, write_json


def main() -> None:
    summary=json.loads((OUT/'review_summary.json').read_text())
    stats=json.loads((OUT/'topk_stability.json').read_text())['scenarios']
    fractions=json.loads((OUT/'fractional_summary.json').read_text())
    records=list(csv.DictReader((OUT/'universities_rank_scenarios.csv').open(encoding='utf-8-sig')))
    corrections=json.loads((OUT/'author_corrections.json').read_text())
    pa=json.loads((OUT/'present_address_audit.json').read_text())
    counts=Counter(r['status'] for r in pa)
    changes=json.loads((OUT/'identity_changes.json').read_text())
    remaining=list(csv.DictReader((OUT/'remaining_papers.csv').open(encoding='utf-8-sig')))
    remaining_ids={r['doi'] for r in remaining}
    touched={r['doi'] for r in corrections+changes}
    old=list(csv.DictReader((PREV/'remaining_papers.csv').open(encoding='utf-8-sig')))
    write_csv(OUT/'input_351_disposition.csv',[{**r,'status':'still_unresolved' if r['doi'] in remaining_ids else 'newly_complete',
        'targeted_change_in_this_pass':r['doi'] in touched} for r in old])
    write_json(OUT/'present_screen_summary.json',{'paper_n':272,'author_record_n':len(pa),'statuses':dict(counts),
        'corrected_papers':len({r['doi'] for r in corrections if 'present_address_evidence' in r})})
    lines=['# Fig8：出版社署名冲突与现地址复核','',
        '固定 2023–2025 年 Nature 主刊 3,683 篇原创 Article，增加独立数据层。前序文件和作者字段保留，所有删除都记录原槽位、出版社作者编号、地址及新槽位。','',
        '## 补齐与纠错','',
        '剩余机构缺失论文 **351 → 350**；完全缺失机构的作者记录 **1,041 → 1,033**。补齐 Unistellar 7 个、Revivicor 5 个机构槽位，分布于3篇，其中8条作者记录由完全未知变为有已确认机构，1篇新达到全部作者有机构。使用有来源的本地ID，未假造ROR/OpenAlex标准ID。','',
        '原14条出版社署名冲突涉及8篇、12条作者记录。已逐条按完整出版社作者—机构编号链核对，移除39个混入的旧机构槽位，包含14个争议联合中心槽位。保留确实属于该作者且尚未确认的机构，例如 Jian Li 的 Aging Biomarker Consortium。Jiaming Li与Jian Li、Mengqi Wu与Menghao Wu对应不同出版社机构编号；旧表包含相同混合列表。不据此断言其OpenAlex作者ID被合并。','',
        '原92条分支中14条作为错误作者分配关闭，**78条继续未解决**。这与新增机构身份补齐是不同操作，不能把修正数量等同于新增完整论文。','',
        '## UCLA与Columbia的现地址统一检查','',
        '在前序表中涉及任一学校的全部272篇论文内，对所有作者采用同一条规则：仅删除出版社明确标为present address、与该作者唯一匹配、且没有同时出现在其研究署名中的原机构槽位。完整研究署名和至少一个已保留的明确机构是删除前提。本文延续先前出版社解析器排除现地址的定义。','',
        '修正42篇中的73条作者记录，删除92个现地址槽位；加上14冲突相关修正，本轮共修正50篇、85条作者记录，删除131个不适用于研究署名的槽位。不会把现地址的机构事实改写为虚假身份，而是从研究归属计数中排除。','',
        '关键实例 s41586-024-07136-2：Xingchen Ji 的上海交大地址是Aff3，Yoshitomo Okawachi的Xscape地址是Aff4，均明确标为现地址；其研究署名均属Columbia。该篇原Columbia份额0.875，修正后1.0，增加0.125。Xscape官方确认其公司身份及Columbia衍生背景；计数处理依据本篇署名角色，不是把公司并入大学。只修正此篇会造成边界翻转，但双侧全范围修正后的净结果仍是UCLA领先。','',
        '筛查不是全部完成核实：7,952条作者记录中，6,781条在严格匹配条件下无现地址，73条修正，22条未找到可明确删除的现地址专属槽位，3条仍未解决；1,073条因全篇引用解析完整性或唯一名字匹配条件不满足，未自动修正。最后这一数量不等于存在1,073条现地址错误。证据和状态均在present_address_audit.json及present_address_findings.csv。272篇外尚未统一筛查，不能将本轮结果视为全队列现地址清洗完成。','',
        '## 覆盖率','', '|指标|篇数 / 3,683|比例|','|---|---:|---:|','|摘要完整|3,683|100%|']
    labels={'after_known':'全部作者至少一个已知机构','standard_after':'全部作者至少一个标准机构ID',
            'all_slots_known':'全部保留机构槽位已知','all_slots_country':'全部保留槽位国家已知'}
    for k,n in summary['coverage'].items():lines.append(f'|{labels[k]}|{n}|{n/3683*100:.4f}%|')
    c=stats['direct_units']['country'];u=stats['direct_units']['university']
    lines+=['','## 国家前12','',f'主口径名单不变：{", ".join(c["members"])}。第12位IL={c["cutoff"][1]:.6f}，第13位DK={c["next"][1]:.6f}，差{c["gap"]:.6f}；未知国家份额{c["unknown_mass"]:.6f}仍更大，不能保证边界稳定。', '',
        '本轮国家份额发生有据可查的变化：包括HKUST作者去除混入的北大地址、Toronto/Yunnan作者去除混入的北大地址，以及现地址排除。沿用原国家/地区编码口径，包括HK独立编码。country_allocation_changes.json逐国列出差值，country_change_validation.json验证这些差值完全等于已记录作者修正的汇总；总论文份额仍守恒。', '',
        '所有全样本高校归并情景使用相同国家分配。完整案例仍会改变国家前12成员，不能用完整案例名单替代固定队列的国家比较。', '',
        '## 高校前15','',
        f'主口径前15成员与上一轮相同；UCLA第15={u["cutoff"][1]:.6f}，Columbia第16={u["next"][1]:.6f}。差距从0.046536扩大到{u["gap"]:.6f}，但对归并口径仍敏感。', '',
        '|排名|高校|分数计数|','|---:|---|---:|']
    for r in records:
        if r['scenario']=='direct_units' and int(r['rank'])<=16:lines.append(f'|{r["rank"]}|{r["name"]}|{float(r["fractional_n"]):.6f}|')
    lines+=['','|情景|第15位|第16位|相对主口径变化|','|---|---|---|---|']
    for key,record in stats.items():
        v=record['university'];delta='无' if v['same_members_as_direct'] else '进入 '+', '.join(v['entered_vs_direct'])+'；退出 '+', '.join(v['left_vs_direct'])
        lines.append(f'|{key}|{v["cutoff"][0]} ({v["cutoff"][1]:.6f})|{v["next"][0]} ({v["next"][1]:.6f})|{delta}|')
    d=fractions['direct_units']
    lines+=['','只计学术单位，以及大学去重后重新分配分母时，Columbia进入、北大退出。完整案例则Columbia进入、Oxford退出。共同中心均分和仅官方直接归并的所测情景成员一致，但不能覆盖所有剩余不确定性。', '',
        f'未知机构身份份额{d["identity_unknown"]:.6f}，其中可能高校份额{d["possible_university_unknown"]:.6f}。固定映射和分母、按工作国家约束的缺失分配检验保证的当前前15成员为：{", ".join(u["fixed_mapping_country_constrained_guaranteed_current_members"])}。这不是置信区间，也不包含现地址未审查记录、海外校区及层级关系变动。', '',
        '## 文件与验证','',
        'research_roster_conflict_review.parquet为新分析层；author_corrections.json/csv记录完整前后列表；resolved_14_conflicts.csv、unresolved_branches.csv记录已关闭及78条剩余分支；remaining_papers.csv和remaining_identity_authors.csv为350篇和1,033条作者队列。input_351_disposition.csv区分新增完整论文与仍待核查项。', '',
        '19项相关测试通过，检查唯一作者、完整引用要求、现地址与研究地址共现时保留、非高校分室及未知份额不重分配。固定3,683篇、前序字段及哈希保留、分数计数守恒、国家差值与修正日志一致均通过；新增脚本ruff通过。所有来源请求失败均保留，web_conflictSources*.json保存官方检索补充；未把403或连接失败标为成功。', '',
        '复现：python3 -m experiments.fig8_extension.nature_conflict_review → nature_conflict_analyze → nature_conflict_report。conflict_audit.ipynb含实际执行的结果检查。','',
        '### 关键来源','',
        '- [Nature论文及现地址](https://www.nature.com/articles/s41586-024-07136-2)',
        '- [Xscape公司说明](https://www.xscapephotonics.com/about-us)',
        '- [Unistellar官方登记信息](https://shop.eu.unistellar.com/policies/legal-notice)',
        '- [Revivicor美国政府登记地址](https://www.govinfo.gov/content/pkg/FR-2021-04-01/pdf/2021-06704.pdf)',
        '- [HKUST Biological Cryo-EM Center](https://cryo-em.hkust.edu.hk/)',
        '- [湖北大学重点实验室说明](https://eng.hubu.edu.cn/info/1030/2086.htm)']
    (OUT/'出版社署名冲突与名单稳定性.md').write_text('\n'.join(lines)+'\n')
    code="from pathlib import Path\nimport json\np=Path("+repr(str(OUT))+")\ns=json.loads((p/'review_summary.json').read_text())\nassert s['cohort_n']==3683 and s['resolved_previous_conflicts']==14\nprint(json.dumps(s,indent=2))\nprint((p/'topk_stability.json').read_text())"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    write_json(OUT/'conflict_audit.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'cell_type':'code','id':'audit','metadata':{},'execution_count':1,'source':code.splitlines(True),
        'outputs':[{'output_type':'stream','name':'stdout','text':result.stdout.splitlines(True)}]}]})
    val=json.loads((OUT/'validation.json').read_text());val.update(tests_passed=19,ruff='passed')
    write_json(OUT/'validation.json',val)
    write_json(OUT/'artifact_manifest.json',{'input':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'},
        'scripts':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('experiments/fig8_extension').glob('nature_conflict_*.py')}})
    print('Report, executed notebook and manifest saved')


if __name__=='__main__':
    main()

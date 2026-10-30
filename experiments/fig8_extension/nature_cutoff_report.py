"""Source-linked review report, reproducible audit notebook and manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .nature_cutoff_review import INPUT, OUT, PREV, RULES
from .stage0 import write_csv, write_json


def main() -> None:
    st=json.loads((OUT/'topk_stability.json').read_text())['scenarios']
    fs=json.loads((OUT/'fractional_summary.json').read_text())
    ranks=list(csv.DictReader((OUT/'universities_rank_scenarios.csv').open(encoding='utf-8-sig')))
    changes=json.loads((OUT/'identity_changes.json').read_text())
    branches=json.loads((OUT/'branch_audit.json').read_text())
    before=list(csv.DictReader((PREV/'remaining_papers.csv').open(encoding='utf-8-sig')))
    remain=list(csv.DictReader((OUT/'remaining_papers.csv').open(encoding='utf-8-sig')))
    ids={r['doi'] for r in remain}
    touched={r['doi'] for r in changes+branches}
    write_csv(OUT/'input_357_disposition.csv',[{**r,'status':'still_unresolved' if r['doi'] in ids else 'newly_complete',
        'targeted_in_this_pass':r['doi'] in touched} for r in before])
    lines=['# Fig8：UCLA—Columbia 边界与剩余机构复核','',
        '固定 Nature 主刊 2023–2025 年 3,683 篇原创 Article。本轮新增独立审计层，旧表、旧作者字段和原机构槽位字段保留；并非创新能力或影响力得分。','',
        '## 本轮补齐','',
        '未解决论文 357 → **351**；完全缺失机构的作者记录 1,051 → **1,041**。修改 8 篇中的 19 个机构槽位，10 条作者记录由完全未知变为至少一个已确认机构；6 篇新达到全部作者有机构。其余修改改善部分槽位完整度，不能再算作新补齐论文。','',
        '|指标|本轮|上一轮|','|---|---:|---:|',
        '|摘要完整|3,683 / 3,683（100%）|3,683|',
        f'|全部作者至少一个已知机构|3,332 / 3,683（{3332/3683*100:.4f}%）|3,326|',
        '|全部作者至少一个标准机构 ID|3,270 / 3,683（88.7863%）|3,270|',
        f'|全部机构槽位已知|2,966 / 3,683（{2966/3683*100:.4f}%）|2,959|',
        '|全部槽位国家已知|3,656 / 3,683（99.2669%）|3,656|','',
        '新单位使用有出处的本地标识，不冒充 OpenAlex/ROR 标准 ID。站点国家取论文署名，未用总部国家覆盖。','',
        '|确认单位|补齐槽位数|分类及边界处理|','|---|---:|---|']
    counts=Counter(c['new_entity']['display_name'] for c in changes)
    notes={'Columbia Equine Hospital':'兽医机构，不能因 Columbia 同名归入 Columbia University',
           'Meta FAIR Paris':'企业研发单位，保留 FR 工作地点','Psiphon Inc.':'公司，大学孵化来源不等于高校子单位',
           'Proteros biostructures GmbH':'公司','Speech Graphics Ltd':'公司，大学衍生企业不并入高校',
           'Autonomy Institute':'独立研究机构/智库','Hamburg Observatory':'大学物理系天文台，按官方关系归入 Universität Hamburg'}
    for name,n in counts.items():lines.append(f'|{name}|{n}|{notes[name]}|')
    lines+=['','## 分支与仅 ROR 关系','',
        '原 91 条逐条复核：11 条确认属于中科院动物所膜生物学分室，计为非高校；80 条继续未解决，其中 14 条作者—中心旧匹配在出版社该作者地址中找不到对应中心。14 条保留为署名冲突，在高校均分敏感性中隔离，不删除原身份、国家或分母。其余 66 条没有足够的分支证据，未用同一作者的另一条高校地址代替。', '',
        '扩展到邻近高校的联合中心：确认 QB3 39 条、IGI 30 条同地址明确的 Berkeley 分支，分别贡献 1.539894、1.015818。另确认 PGEC 2 条明确署名 USDA-ARS 的非高校分支。新增范围仍有 12 条未确认，故本次未解决分支总表为 **92 条 = 原队列80 + 新范围12**。', '',
        'QB3、PGEC、Biohub New York 的伙伴关系取得官方依据，仍视为共同中心，不能把伙伴关系解释为大学独占所有。IGI 官方当前列出 Berkeley、UCSF、Davis、UCSD 四方，2023 年来源列出两校及附属伙伴；跨年份伙伴分母仍待核实。本轮主计数只采纳论文明确的分支，均分情景延续历史候选分母，作为有局限的敏感性，不声称已统一历史组织结构。', '',
        f'全队列仍仅 ROR 支持的直接学术/临床归并份额为 {fs["direct_units"]["registry_only_rollup_mass"]:.6f}，与上一轮相同。本轮升级的是上述共同中心关系及署名分支；不能把它们说成减少了这部分直接归并份额。', '',
        '## 国家前 12','',
        '全部全样本口径下国家分配与上一轮完全一致：US、CN、GB、DE、CH、FR、JP、AU、CA、NL、ES、IL。IL=40.464973、DK=39.507276，差 0.957698；未知国家份额 1.412272 更大，边界仍可能翻转。固定已知映射且仅分配缺失份额时，保证当前前 11 位成员。', '',
        '排除明确非个人记录后，未知国家份额 0.938649，边界差 0.957799，条件裕量约 0.019150；不能转换为无条件保证。完整案例 2,966 篇的前12仍是 KR 进入、ES 退出，提示样本筛选敏感性。', '',
        '## 高校前 15','',
        '主口径前15成员与上一轮一致，内部顺序变化：Berkeley 增至 33.235224，升至第12；北大第13、Oxford第14，UCLA仍第15。Columbia第16。UCLA—Columbia仍仅差 **0.046535983**。', '',
        '|排名|高校|分数计数|','|---:|---|---:|']
    for r in ranks:
        if r['scenario']=='direct_units' and int(r['rank'])<=17:lines.append(f'|{r["rank"]}|{r["name"]}|{float(r["fractional_n"]):.6f}|')
    lines+=['','|口径|第15位|第16位|相对主口径成员变化|','|---|---|---|---|']
    for scenario,v in st.items():
        u=v['university'];delta='无' if u['same_members_as_direct'] else '进入 '+', '.join(u['entered_vs_direct'])+'；退出 '+', '.join(u['left_vs_direct'])
        lines.append(f'|{scenario}|{u["cutoff"][0]} ({u["cutoff"][1]:.6f})|{u["next"][0]} ({u["next"][1]:.6f})|{delta}|')
    lines+=['','医院口径仍影响成员：仅学术单位时北大退出、Columbia进入；完整案例时 Oxford退出、Columbia进入。共同中心均分、大学去重、排除明确非个人记录和去掉剩余ROR直接归并情景的前15成员与主口径一致；这只说明所测情景一致，不是全局稳定证明。', '',
        f'未知机构身份份额由 {fs["previous_main"]["identity_unknown"]:.6f} 降至 {fs["direct_units"]["identity_unknown"]:.6f}；可能高校未知份额由 {fs["previous_main"]["possible_university_unknown"]:.6f} 降至 {fs["direct_units"]["possible_university_unknown"]:.6f}。', '',
        '**更正上一轮叙述：** 上轮 Markdown 沿用了更早“保证四个成员”的文字，但上轮机器可读结果已经只保证 Stanford、Harvard、MIT 三个。本轮重新计算也是这三个。该下界仅适用于固定映射、固定分母、大学国家与工作地相符的缺失分配，不含海外校区、联合中心关系和署名冲突。旧报告不覆盖，此处保留勘误。', '',
        '## 证据与复现','',
        '`identity_changes.json/csv` 保存每次修补的原记录、新记录、出版社地址与官方URL；`branch_audit.csv` 保存174条本轮分支判断。`unresolved_branches.csv` 保存92条未解决分支，`publisher_assignment_conflicts.csv` 单列14条冲突。`remaining_papers.csv`、`remaining_identity_authors.csv` 是351篇和1,041条作者剩余队列；`input_357_disposition.csv` 区分本轮定向核查与仍待后续核查，不能把全量筛查宣称为逐条完成官方验证。', '',
        '验证：固定队列唯一性、旧字段与上轮parquet哈希不变、国家分配不变、分数计数守恒通过；15项相关测试和新增脚本ruff检查通过。部分官网直接请求失败或403；`source_receipts.json` 保留失败，`web_sources*.json` 保存官方搜索/浏览补充证据。', '',
        '执行：`python3 -m experiments.fig8_extension.nature_cutoff_review` → `python3 -m experiments.fig8_extension.nature_cutoff_analyze` → `python3 -m experiments.fig8_extension.nature_cutoff_report`。', '',
        '### 官方来源','']
    for _,name,_,_,url in RULES:lines.append(f'- [{name}]({url})')
    lines += ['- [中科院动物所分室与实验室历史](https://www.biomembrane.tsinghua.edu.cn/p/shiyanshijianjie.html)',
        '- [QB3 组织说明](https://qb3.org/about/)', '- [USDA PGEC](https://www.ars.usda.gov/pacific-west-area/albany-ca/pgec/docs/main/)',
        '- [IGI 组织说明](https://innovativegenomics.org/about-us/)',
        '- [Biohub New York 2023年成立公告](https://chanzuckerberg.com/newsroom/biohub-new-york-technologies-detect-disease/)']
    (OUT/'UCLA-Columbia边界复核.md').write_text('\n'.join(lines)+'\n')
    code="from pathlib import Path\nimport json\np=Path("+repr(str(OUT))+")\ns=json.loads((p/'review_summary.json').read_text())\nassert s['cohort_n']==3683 and s['remaining_identity_papers']==351\nprint(json.dumps(s,indent=2))\nprint((p/'topk_stability.json').read_text())"
    run=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    write_json(OUT/'cutoff_audit.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'cell_type':'code','id':'audit','metadata':{},'execution_count':1,'source':code.splitlines(True),
                  'outputs':[{'output_type':'stream','name':'stdout','text':run.stdout.splitlines(True)}]}]})
    validation=json.loads((OUT/'validation.json').read_text());validation.update(tests_passed=15,ruff='passed')
    write_json(OUT/'validation.json',validation)
    write_json(OUT/'artifact_manifest.json',{'input':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'files':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'},
        'scripts':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('experiments/fig8_extension').glob('nature_cutoff_*.py')}})
    print('Report, audit notebook and manifest saved')


if __name__=='__main__':
    main()

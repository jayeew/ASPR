"""Refresh the current Fig8b completeness report and derived metadata queues."""
from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .fig8b_qualification_decide import NONPRIMARY, SPECIAL
from .fig8b_qualification_recompute import gzip_text, iter_papers, metrics

BASE=Path('outputs/fig8_data/fields')
OUT=BASE / 'qualification'


def metadata_gaps() -> dict:
    counts=Counter(); denominator=0
    columns=['work_id','doi','field','missing_abstract','missing_institution','missing_country','paper_shard','metadata_root','abstract_source','author_source_receipt']
    with gzip_text(OUT/'research_metadata_gaps.csv.gz') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for path, row, decision in iter_papers(OUT, research_only=True):
            denominator += 1
            values=metrics(row)
            gaps={'missing_abstract':not values['abstract'],'missing_institution':not values['every_author_institution'],'missing_country':not values['every_author_country']}
            counts.update(gaps)
            if any(gaps.values()):
                counts['any_metadata_gap']+=1
                writer.writerow({'work_id':row['work_id'],'doi':row['doi'],'field':row['field'],**gaps,
                    'paper_shard':'../papers/'+path.name,'metadata_root':'..',
                    'abstract_source':row['abstract_source'],'author_source_receipt':row.get('authorships_source_receipt') or row['raw_receipt']})
    result={**dict.fromkeys(['missing_abstract','missing_institution','missing_country','any_metadata_gap'],0),**counts,'qualified_denominator':denominator};(OUT/'research_metadata_gaps_summary.json').write_text(json.dumps(result,indent=2));return result


def main() -> None:
    check=json.loads((OUT/'recomputed_results.json').read_text())
    if not check['coverage_equal'] or not check['eligibility_equal'] or check['identity_errors']:
        raise ValueError('Independent recomputation failed')
    gaps=metadata_gaps()
    with (OUT/'coverage.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    stats={(r['field'],r['population']):r for r in rows if r['year']=='0'}
    locked=check['verified_research_count'];excluded=sum(n for k,n in check['statuses'].items() if k.startswith('excluded'))
    names={'biochemistry':'生化、遗传与分子生物学','materials':'材料科学','physics':'物理与天文学'}
    coverage_table=[];selection=[]
    for field,label in names.items():
        r=stats[(field,'verified_research')];n=int(r['papers'])
        rate=lambda k, r=r, n=n:f"{int(r[k]):,}（{100*int(r[k])/n:.2f}%）" if n else '0'
        coverage_table.append(f"| {label} | {n:,} | {rate('abstract')} | {rate('every_author_institution')} | {rate('every_author_country')} |")
        pending=int(stats[(field,'pending')]['papers']);ex=int(stats[(field,'excluded')]['papers']);total=n+pending+ex
        selection.append(f'| {label} | {total:,} | {n:,}（{100*n/total:.2f}%） | {ex:,} | {pending:,} |')
    jats=Counter();jats_http=Counter()
    for p in (OUT/'jats_parsed').glob('*.json'):
        r=json.loads(p.read_text());jats[r['status']]+=1;jats_http[str(r.get('http_status'))]+=1
    with (OUT/'catalogue_completeness.csv').open(encoding='utf-8-sig') as stream:catalogues=list(csv.DictReader(stream))
    aps=json.loads((OUT/'aps_all_status.json').read_text())
    scope=json.loads((OUT/'secondary_scope.json').read_text());secondary=json.loads((OUT/'secondary_status.json').read_text())
    with gzip.open(OUT/'doi_query_attempts.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['doi','query_status','receipt','phase'])
        for dbpath,table_name,prefix,phase in [(BASE/'epmc_queue.sqlite','queue','../','metadata_repair'),(OUT/'secondary_attempts.sqlite','attempts','','qualification')]:
            db=sqlite3.connect(dbpath)
            for doi,status,receipt in db.execute(f'SELECT doi,status,receipt FROM {table_name}'):
                writer.writerow([doi,status,prefix+(receipt or ''),phase])
            db.close()
    audit={'catalogue_source_years':len(catalogues),'catalogue_complete_source_years':sum(r['complete']=='True' for r in catalogues),
        'jats_attempt_status':dict(jats),'jats_http_status':dict(jats_http),'aps_months':len(aps),
        'aps_complete_months':sum(r['status']=='complete' for r in aps),'aps_records':sum(r['records'] for r in aps),
        'nature_single_lookup_targets':len(json.loads((OUT/'nature_missing_targets.json').read_text())),
        'aps_single_lookup_targets':len(json.loads((OUT/'aps_missing_targets.json').read_text())),
        'additional_secondary_doi_scope':scope['additional_dois'],'secondary_attempt_status':secondary}
    (OUT/'retrieval_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    protocol={'dataset':'fig8/fields','updated_at':datetime.now(timezone.utc).isoformat(),
        'metadata_table':'../papers','update_policy':'Append, overwrite or merge the current tables; regenerate qualification and coverage after changes',
        'candidate_grain':'One OpenAlex Work; current candidate universe, 2023-2025 primary-field English article/letter journal candidates',
        'research_grain':'Unique DOI after explicit journal-identity duplicate resolution',
        'included':'Matched DOI/title/journal primary evidence, explicit research subtype and consistent electronic publication year 2023-2025; no unresolved source-type conflict',
        'research_types':['research-article','Original Research','Nature research-journal Article','APS article / letter except RMP and Physics commentary journals'],
        'nonprimary_types':sorted(NONPRIMARY),'special_subtypes_pending':sorted(SPECIAL),
        'date_rule':'JATS epub/epub-ppub or electronic pub only; Nature publisher catalogue online dates; APS electronic-first publication date; explicit publisher-deposited Crossref published-online. Print, collection, received, accepted, indexed and generic untyped dates do not substitute.',
        'secondary_rule':'Europe PMC electronicPublicationDate and pubTypeList are separate supporting evidence. Journal Article and Crossref journal-article never imply original research.',
        'identity_rule':'Exact DOI, punctuation/case-normalized title, and exact ISSN intersection or journal/source identity. Reject unmatched evidence.',
        'coverage_denominator':'Per verified paper: every observed author has >=1 institution/country; full publisher byline and all affiliation slots are not asserted.',
        'table_join':'Join ../papers and decisions on work_id within matching shards. eligibility_status, online_year and publisher_types are authoritative qualification fields; no duplicated research-paper table.',
        'metadata_changes':'Qualification and metadata are separate tables; coverage is recalculated from current abstracts/authors.',
        'scope_limit':'Verified subset, not an exhaustive final research census. Original window/field/language prefilters remain; papers outside original OpenAlex date window not comprehensively recovered.',
        'institution_rule':'Inherit Nature / Fig8b evidence rules; do not infer university rollup from ROR parents or education label.',
        'nature_label_rule':'Rejoin official annual Nature Portfolio imprint by verified online year; Springer Nature group is distinct.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    text=f'''# Fig8b 类型、在线年份与去重核查报告

**当前已确认 {locked:,} 篇证据确认的研究论文子集；全候选集的最终研究论文总清单尚未锁定。** 已确认排除 {excluded:,} 条，仍有 {check['unresolved_count']:,} 条待核。未核实记录没有被当作原创研究，也没有被静默删除。

论文主表与资格判定表统一维护，可追加、覆盖和合并。目录整理不增加核实论文或填补缺失；覆盖率从当前论文主表重新计算。

## 分母与进度

| 领域 | 候选 | 已核实研究子集 | 明确排除 | 待核 |
|---|---:|---:|---:|---:|
{chr(10).join(selection)}

## 已核实研究子集覆盖率

| 领域 | 研究论文 | 有摘要 | 每位观测作者至少一个机构 | 每位观测作者至少一个国家 |
|---|---:|---:|---:|---:|
{chr(10).join(coverage_table)}

`coverage.csv` 中，已核实研究子集按确认的在线年份分年；排除和待核组保留OpenAlex候选年，以便核对原始分母，不替日期未知记录推定在线年份。`research_journal_coverage.csv` 按领域×期刊，`research_publisher_coverage.csv` 按在线年份重新匹配 Nature Portfolio 年度标签。机构/国家有值不代表所有分支已核实；不自动归并高校。已核实研究子集中仍缺摘要 {gaps['missing_abstract']:,} 篇，至少一位观测作者缺机构 {gaps['missing_institution']:,} 篇、缺国家 {gaps['missing_country']:,} 篇；三类去重后 {gaps['any_metadata_gap']:,} 篇。逐篇补齐队列见`research_metadata_gaps.csv.gz`，与资格待核队列分开。子集中仍有 {check['verified_author_cap_remaining']} 篇作者记录达到接口上限，保留其标记。

## 核查与来源

1. Nature Portfolio：在已识别期刊范围内抓取2023–2025年度官方目录，逐篇核对 DOI、标题和期刊来源；保留全部页、抓取时间、目录分母与分页完整性。无年度结果、访问失败或计数不齐均未作完整名录声明。Nature与Nature Neuroscience的2024年目录各比页面标示少1条，重新抓取对应短页后差异仍存在，证据保存在`catalogue_followup/`；已读文章逐条证据仍可使用，但这两个年度目录不标为完整。对未出现在年度目录且无既有类型/在线日期证据的8个候选Nature DOI另查出版社页面，保留结果于`nature_single/`。
2. 其他出版社：对已有缓存中与候选 DOI、标题匹配且可开放读取的31,906个PMC目标，读取出版物XML前置信息。保存精确的原始XML前缀及哈希，读取 `article-type` 与明确电子出版日期；正文不用于类型判定，不将数据库收录时间用作在线日期。XML可能经PMC规范化，因此单列为归档出版物证据。
3. APS：通过官方公开元数据接口，按月遍历2023–2025全部公开返回的目录，并跟随服务器分页链接。另对未出现在年度目录中的1,761个候选DOI逐篇查询，以发现在线年越界或未解决的访问/身份问题。区分 `articleType` 与恒为article的对象类型；Letter按APS研究类型政策处理，RMP及Physics评论类内容单列排除。
4. Europe PMC：复用初次已查询DOI，再查询剩余 {scope['additional_dois']:,} 个可查询DOI；两阶段查询合计覆盖1,508,997个唯一非空DOI；另有10,564条候选缺DOI。查询complete仅表示接口完整返回，不表示命中有效证据。电子日期和文献类型只作二级证据。仅有 `Journal Article` 不纳入研究清单；仅有Review等二级标签保留待主源确认。本轮新增二级结果中的PMC全文线索没有全部进一步抓取，列入后续主源核查范围。
5. 复用已保存的Crossref出版商提交记录及本轮重复DOI查询中的`published-online`，只补充明确在线日期证据；其宽泛类型不用于认定原创研究。日期与其他主源冲突则保留待核。
6. DOI重复：3组均通过出版社提交的题名与ISSN确定正式期刊对应Work，排除3条错挂到另一期刊的记录；保留被排除Work→正式Work映射、来源和日期。没有合并不明作者信息。

检索完成状态与命中有效证据分开统计：
```json
{json.dumps(audit,ensure_ascii=False,indent=2)}
```

## 固定纳入规则

- 证据须匹配 DOI、标题和期刊；标点/大小写规范化允许，模糊标题匹配不用于纳入。
- 只纳入明确研究类型且在线年在2023–2025的记录。新闻、评论、综述、社论、更正、撤稿通知等按明确主源类型排除。
- 分析、短报告、协议、病例、数据论文等需要内容或期刊特定规则的记录保留待核；不会把所有Article字样都认作研究。
- 研究类型与综述/完整性标记冲突、不同主源在线年冲突均暂停纳入。完整性关注标记不等同于无效论文结论。
- 在线年优先明确电子发表字段；印刷年、合集年、接收年、录入年不替代。在线日期精度保留在证据中。子集中 {check['verified_online_year_shifts']} 篇在线年与OpenAlex年不同。
- 同一DOI只保留经期刊身份核实的一个Work。Nature Portfolio按在线年份的官方imprint重新匹配，不用Springer Nature集团标签替代。

逐状态计数：
```json
{json.dumps(check['statuses'],ensure_ascii=False,indent=2)}
```

## 未解决项与适用范围

`unresolved.csv.gz` 逐篇保留类型、主源/二级日期、证据ID与未解决原因；`decision_evidence.jsonl.gz` 保存接受和拒绝的主源匹配，`secondary_evidence.jsonl.gz` 保留二级线索。HTTP失败或未命中不意味着论文不是原创研究。

本轮主源覆盖偏向Nature Portfolio、APS和可开放读取PMC XML的期刊。**已核实子集不能代表三个领域全部期刊的无偏样本，也不能直接用于Nature与其他出版商的正式比较。** 正式全期刊实验仍需补查其他出版社类型/在线年、特殊类型和冲突；还需检查OpenAlex窗口外但在线年落在目标窗口的漏收文献。当前表只回答候选中哪些记录已具有足够纳入证据。

## 统一数据与复算

- `../papers/`：唯一的三个领域论文主表，保存摘要、作者机构及来源；`decisions/`保存资格、类型、在线年份和未解决状态，以work_id关联。
- `research_roster.csv`、覆盖率表及缺口队列均为可重新生成的视图，不再保存重复研究论文数据。
- `duplicate_resolution.csv`保留重复DOI判定；证据原文和查询记录保留。
- `protocol.json`、`decide.py`、`recompute.py`定义当前口径和离线复算；`audit.ipynb`为入口。
- 数据可直接更新，无父版本依赖或版本冻结要求。新论文或身份/资格变化须重建判定；仅摘要/机构变化可使用`--refresh`刷新覆盖率。完整性报告随后刷新。

运行：
```bash
python3 recompute.py .
```
需Python与pyarrow。复算逐篇重新执行纳入决策，并从摘要及作者原记录重算三个状态分母的覆盖率。已核对 {check['candidate_count']:,} 条候选、{locked:,} 个唯一研究DOI；决策与覆盖率一致，身份重复错误为0。代码检查结果见 `code_validation.json`。

## 方法来源

- [Nature原创研究类型](https://www.nature.com/nature/for-authors/formatting-guide)、[Nature其他类型](https://www.nature.com/nature/for-authors/other-subs)
- [APS Harvest字段与分页](https://harvest.aps.org/docs/harvest-api)、[APS电子优先出版](https://journals.aps.org/article.html)、[APS Letter定义](https://journals.aps.org/prl/authors)
- [RMP内容范围](https://journals.aps.org/rmp/about)、[Physics不发表原始研究](https://physics.aps.org/faq)
- [JATS日期语义](https://jats.nlm.nih.gov/publishing/tag-library/1.1d3/element/pub-date.html)、[Europe PMC接口](https://europepmc.org/RestfulWebService)
- [Crossref期刊条目也包括非研究内容](https://www.crossref.org/documentation/principles-practices/journals/)
'''
    (OUT/'完整性报告.md').write_text(text)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'markdown','id':'scope','metadata':{},'source':['# Fig8b 离线复算\n从统一论文主表和资格表复算，不访问网络。']},
        {'cell_type':'code','id':'recompute','metadata':{},'execution_count':None,'outputs':[],'source':['from pathlib import Path\nimport subprocess, sys\nsubprocess.run([sys.executable, "recompute.py", "."], check=True)']},
        {'cell_type':'code','id':'results','metadata':{},'execution_count':None,'outputs':[],'source':['import json\njson.loads(Path("recomputed_results.json").read_text())']}]}
    (OUT/'audit.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
    print('Updated current completeness report',flush=True)


if __name__=='__main__':main()

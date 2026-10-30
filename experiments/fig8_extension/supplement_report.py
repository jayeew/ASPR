"""Source capability, live budgets, grouping diagnostics and panel readiness."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .fig8b_qualification_build import match
from .fig8b_qualification_decide import decide, electronic_date, type_class
from .fig8b_qualification_recompute import iter_papers, metrics
from .supplement_allocation import (
    SCENARIOS,
    allocate,
    known_universities,
    official_relationships,
)
from .supplement_queue import ROOT, connect
from .supplement_sources import publisher_url


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def pilot_results(root: Path) -> list[dict]:
    db = connect(root)
    db.row_factory = sqlite3.Row
    q = root / 'fields/qualification'
    with (root / 'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        sources = {r['source_id']: r for r in csv.DictReader(stream)}
    counts = defaultdict(Counter)
    for row in db.execute('SELECT * FROM pilot'):
        row = dict(row)
        url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{row["pmcid"]}/fullTextXML' if row['arm'] == 'pmc' else publisher_url(row['doi'], row['route'])
        path = q / 'supplement_parsed' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        values = counts[row['arm']]
        values['targets'] += 1
        if not path.exists():
            values['not_attempted'] += 1
            continue
        r = json.loads(path.read_text())
        values['attempted'] += 1
        values[r['status']] += 1
        values['request_attempts'] += len(r.get('attempts', []))
        values['retained_bytes'] += r.get('retained_bytes', 0)
        values['summed_request_seconds'] += r.get('elapsed_seconds', 0)
        if r['status'] != 'parsed':
            continue
        p = r['parsed']
        if row['arm'] == 'pmc':
            online, precision = electronic_date(p['dates'])
            item = {'family': 'archived_jats', 'title': ' '.join(p['title']), 'issns': p['issn'],
                    'journal_titles': p['journal_titles'], 'kind': p['article_type'], 'online_date': online,
                    'date_precision': precision, 'categories': p['categories']}
        else:
            item = {'family': 'publisher_' + row['route'], 'title': (p['title'] or [''])[0], 'issns': p['issns'],
                    'journal_titles': p['journal_titles'], 'kind': (p['article_types'] or ['unspecified'])[0],
                    'online_date': p['online_date'], 'categories': []}
        paper = db.execute('SELECT title,journal,source_id FROM papers WHERE work_id=?', (row['work_id'],)).fetchone()
        candidate = {'work_id': row['work_id'], 'title': paper['title'], 'source_id': paper['source_id'], 'journal_name': paper['journal']}
        m = match(candidate, item, sources[paper['source_id']])
        doi_ok = row['doi'] in [d.lower() for d in p.get('doi', [])]
        valid = m['identity'] == 'matched' and doi_ok
        values['identity_matched'] += valid
        values['explicit_type_online_year_confirmed'] += valid and decide([m], [])['status'].startswith(('included_', 'excluded_'))
        values['research_type_confirmed'] += valid and type_class(item['kind'], item['family'], item['categories']) == 'research'
        values['abstract_available_on_matched_source'] += valid and bool(p.get('abstract'))
    output = []
    for arm, count in sorted(counts.items()):
        pending = db.execute("SELECT count(*) FROM papers WHERE status LIKE 'pending%' AND " + ('pmc_unread=1' if arm == 'pmc' else 'route=?'), () if arm == 'pmc' else (arm,)).fetchone()[0]
        attempted = count['attempted']
        effective = count['explicit_type_online_year_confirmed']
        output.append({'source': arm,
                       'measurement_scope': 'historical_stratified_pilot_endpoint_only',
                       'pilot_endpoint': 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML' if arm == 'pmc' else publisher_url('{doi}', arm),
                       'current_source_review': 'supplement/official_bulk/source_path_review.json',
                       'budget_scope': 'Historical endpoint extrapolation; not current collector ETA or a recommendation to retry restricted hosts',
                       **{k: count[k] for k in ['targets', 'attempted', 'not_attempted', 'parsed', 'access_restricted', 'not_available', 'retry_exhausted', 'retryable_error', 'publication_metadata_missing', 'request_attempts', 'identity_matched', 'explicit_type_online_year_confirmed', 'research_type_confirmed', 'abstract_available_on_matched_source', 'retained_bytes']},
                       'identity_match_rate': count['identity_matched'] / attempted if attempted else 0,
                       'qualification_yield': effective / attempted if attempted else 0,
                       'remaining_candidate_tasks': pending,
                       'serial_request_hours_estimate': round(pending * count['summed_request_seconds'] / attempted / 3600, 2) if attempted else None,
                       'minimum_pacing_hours_before_retries': round(pending * (0.34 if arm == 'pmc' else 1.05) / 3600, 2),
                       'paced_hours_with_pilot_attempt_ratio': round(pending * count['request_attempts'] / attempted * (0.34 if arm == 'pmc' else 1.05) / 3600, 2) if attempted else None,
                       'retained_storage_gb_estimate': round(pending * count['retained_bytes'] / attempted / 1e9, 3) if attempted else None,
                       'budget_limitations': 'Stratified pilot; request-seconds are serial work, not parallel wall time; failures and source changes remain possible. No costs purchased.'})
    write_csv(root / 'supplement/source_capabilities.csv', output)
    db.close()
    return output


def boundaries(root: Path) -> dict:
    db = connect(root)
    known = {r[0] for r in db.execute("SELECT doi FROM papers WHERE doi<>''")}
    q = root / 'fields/qualification'
    missing = {}
    for path in (q / 'catalogue_parsed').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('status') != 'parsed':
            continue
        for row in r['rows']:
            if row['doi'] in known or type_class(row['publisher_type'], 'nature_catalogue') != 'research':
                continue
            missing[row['doi']] = {'doi': row['doi'], 'title': row['title'], 'online_date': row['publisher_date'],
                'source_id': r['source_id'], 'receipt': str(path.relative_to(root)),
                'status': 'field_language_and_original_candidate_filter_verification_required',
                'reason': 'Official research catalogue record outside current candidate roster; may belong to another field or filter. Not automatically appended.'}
    for path in (q / 'aps_all_raw').glob('*.gz'):
        with gzip.open(path, 'rt') as stream:
            r = json.load(stream)
        if r.get('http_status') != 200:
            continue
        for p in r['response']['data']:
            doi = p.get('identifiers', {}).get('doi', '').lower()
            if not doi or doi in known or p.get('articleType') not in ['article', 'letter'] or p.get('journal', {}).get('id') in ['RMP', 'PHYSICS']:
                continue
            missing[doi] = {'doi': doi, 'title': p.get('title', {}).get('value', ''), 'online_date': p.get('date', ''),
                'source_id': '', 'receipt': str(path.relative_to(root)),
                'status': 'field_language_and_original_candidate_filter_verification_required',
                'reason': 'APS online-year directory outside current candidate roster; primary field is not inferred from journal.'}
    write_csv(root / 'supplement/boundary_candidates.csv', list(missing.values()))
    db.close()
    return {'cached_official_directory_candidates_not_in_roster': len(missing), 'appended_without_field_proof': 0,
            'scope': 'Cached Nature Portfolio and APS directories only; all-publisher boundary census incomplete'}


def rank_bounds(values: Counter, unknown: float, n: int) -> dict:
    ranked = sorted(values.items(), key=lambda x: (-x[1], x[0]))
    top = ranked[:n]
    gap = top[-1][1] - ranked[n][1] if len(top) == n and len(ranked) > n else 0
    return {'top': [{'id': k, 'fractional_papers': v} for k, v in top], 'unknown_fractional_papers': unknown,
            'cutoff_gap': gap, 'stable_under_unrestricted_unknown_mass': bool(len(top) == n and gap > unknown),
            'bound_assumption': 'All unresolved author/slot mass could support any group. This is conservative and does not certify observed institution links.'}


def readiness(root: Path) -> dict:
    q = root / 'fields/qualification'
    groups = defaultdict(Counter)
    allocations={name:{'countries':Counter(),'universities':Counter(),'stats':Counter()} for name in SCENARIOS}
    diagnostics=Counter()
    known=known_universities(root)
    relationships=official_relationships(root)
    memberships = defaultdict(lambda: {'labels': set(), 'evidence': set(), 'papers': 0})
    for _, paper, decision in iter_papers(q):
        label = decision['nature_portfolio_at_verified_online_year']
        year = decision['online_year'] if decision['eligibility_status'] == 'included_verified_research' else paper['year_openalex']
        group = groups[(paper['field'], year, paper['source_id'], label)]
        group['candidates'] += 1
        group[decision['eligibility_status'].split('_')[0]] += 1
        if decision['eligibility_status'] != 'included_verified_research':
            continue
        values = metrics(paper)
        group.update({k: v for k, v in values.items() if k != 'papers'})
        key = (paper['source_id'], decision['online_year'])
        memberships[key]['labels'].add(label)
        memberships[key]['evidence'].update(json.loads(decision.get('portfolio_evidence_ids', '[]')))
        memberships[key]['papers'] += 1
        diagnostics['research_papers'] += 1
        diagnostics['portfolio_unknown'] += label not in ['True', 'False']
        parsed_authors = json.loads(paper['authors_json'])
        for scenario in SCENARIOS:
            c,u,stats=allocate(paper,scenario,known,relationships,parsed_authors=parsed_authors)
            allocations[scenario]['countries'].update(c);allocations[scenario]['universities'].update(u);allocations[scenario]['stats'].update(stats)
    write_csv(root / 'supplement/coverage_by_group.csv', [{'field': k[0], 'year': k[1], 'source_id': k[2], 'portfolio': k[3], **v} for k, v in sorted(groups.items())])
    write_csv(root / 'supplement/annual_portfolio_evidence.csv', [{'source_id': k[0], 'year': k[1], 'observed_labels': json.dumps(sorted(v['labels'])),
        'matched_papers': v['papers'], 'evidence_ids': json.dumps(sorted(v['evidence'])),
        'scope': 'Observed paper-specific dated evidence; do not extrapolate across transfers or to unobserved papers'} for k, v in sorted(memberships.items())])
    sensitivities={}
    for scenario,values in allocations.items():
        c,u=values['countries'],values['universities']
        sensitivities[scenario]={'country_top12':rank_bounds(Counter({k:v for k,v in c.items() if k!='unknown'}),c['unknown'],12),
            'university_top15':rank_bounds(Counter({k:v for k,v in u.items() if k!='unknown'}),u['unknown'],15),'coverage':dict(values['stats'])}
    country=sensitivities['main_personal_direct']['country_top12'];university=sensitivities['main_personal_direct']['university_top15']
    diagnostics.update(allocations['main_personal_direct']['stats'])
    result = {'a': {'metadata_status': 'ready_with_documented_unknowns', 'scope': 'Existing Nature verified Article cohort; claims/graph outside this task'},
              'b': {'metadata_status': 'blocked', 'reason': 'All-journal eligibility, dated publisher membership and missingness bias are not yet resolved'},
              'c': {'metadata_status': 'blocked', 'reason': 'Uses b cohort; publisher byline/affiliation structure, ownership and unresolved mass remain incomplete'},
              'diagnostics': dict(diagnostics), 'country_top12': country, 'university_top15': university,'sensitivity_scenarios':sensitivities,
              'interpretation': 'Conservative slot-based diagnostics, not final national/university scientific rankings. No institution ownership inferred from education labels.'}
    reopened = root / 'supplement/official_entities/nature_official_reference_reopen.json'
    if reopened.exists():
        review = json.loads(reopened.read_text())
        if review.get('records'):
            result['c']['reopened_nature_source_reviews'] = {
                'units': len(review['records']), 'papers': review.get('nature_dois'),
                'receipt': str(reopened.relative_to(root)),
                'limitation': 'Invalid official-source references require review of institutional assignments; prior ready status is not proof of these assignments.'}
            result['a']['institution_source_limitation'] = str(reopened.relative_to(root))
    (root / 'supplement/panel_readiness.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def institution_priorities(db: sqlite3.Connection) -> list[dict]:
    """Route by aggregated slot requirements, not a unit name alone."""
    columns = ['task_id', 'unit', 'author_records', 'affected_papers', 'weight',
               'missing_identity', 'missing_country']
    rows = [dict(zip(columns, row, strict=True)) for row in db.execute('''
       SELECT t.task_id,json_extract(t.payload,'$.name'),count(*),
              count(DISTINCT p.work_id),count(*)*count(DISTINCT p.work_id) weight,
              json_extract(t.payload,'$.missing_identity'),
              json_extract(t.payload,'$.missing_country')
       FROM tasks t JOIN task_papers p USING(task_id)
       WHERE t.kind='institution' AND t.active=1 AND t.status='pending'
       GROUP BY t.task_id ORDER BY weight DESC''')]
    for row in rows:
        if row['missing_identity'] == 1:
            row['next_action'] = ('identity_and_signed_country' if row['missing_country'] == 1
                                  else 'institution_identity')
        elif row['missing_country'] == 1:
            row['next_action'] = 'signed_workplace_country_only'
        else:
            row['next_action'] = 'refresh_slot_requirements_before_acquisition'
    return rows


def write_institution_priorities(root: Path, rows: list[dict]) -> None:
    """Keep the combined queue and expose overlapping field-specific queues."""
    write_csv(root / 'supplement/institution_priority.csv', rows)
    write_csv(root / 'supplement/institution_identity_priority.csv',
              [row for row in rows if row['missing_identity'] == 1])
    write_csv(root / 'supplement/institution_country_priority.csv',
              [row for row in rows if row['missing_country'] == 1])


def report(root: Path = ROOT) -> dict:
    capabilities = pilot_results(root)
    boundary = boundaries(root)
    panels = readiness(root)
    db = connect(root)
    tasks = list(db.execute('SELECT kind,status,count(*) FROM tasks WHERE active=1 GROUP BY kind,status'))
    top = institution_priorities(db)
    write_institution_priorities(root, top)
    ownership_pending = [dict(zip(
        ['task_id','entity_id','entity_name','example_signature','author_records','affected_papers','weight','status'], row, strict=True))
        for row in db.execute('''
        SELECT t.task_id,t.entity_key,coalesce(i.name,t.entity_key),json_extract(t.payload,'$.name'),
               count(*),count(DISTINCT p.work_id),count(*)*count(DISTINCT p.work_id),t.status
        FROM tasks t JOIN task_papers p USING(task_id)
        LEFT JOIN identities i ON i.entity_id=t.entity_key
        WHERE t.kind='university_ownership' AND t.active=1 AND t.status<>'resolved'
        GROUP BY t.task_id ORDER BY count(*)*count(DISTINCT p.work_id) DESC,t.entity_key''')]
    write_csv(root / 'supplement/university_ownership_priority.csv', ownership_pending)
    journal_status=[dict(zip(['source_id','journal','candidate_n','qualified_n','pending_n','attempted_or_inherited_confirmed_n'],row,strict=True)) for row in db.execute('''
        SELECT p.source_id,min(p.journal),count(*),sum(p.status='included_verified_research'),sum(p.status LIKE 'pending%'),
        sum(p.status NOT LIKE 'pending%' OR EXISTS(SELECT 1 FROM tasks t WHERE t.kind='qualification' AND t.entity_key=p.doi AND t.attempt_n>0))
        FROM papers p GROUP BY p.source_id''')]
    for row in journal_status:
        row['processing_status']='qualification_remaining' if row['pending_n'] else 'current_candidates_classified'
    write_csv(root/'supplement/journal_processing_status.csv',journal_status)
    type_rules=[{'source_family':'archived_jats_or_literal_publisher_type','raw_type':'Research Article / Original Research / Original Research Article','decision':'research_unless_subtype_or_integrity_conflict','evidence_url':'https://jats.nlm.nih.gov/publishing/tag-library/1.4/attribute/article-type.html','scope':'Matched DOI,title,journal; explicit electronic year required'},
        {'source_family':'nature_catalogue','raw_type':'Article','decision':'research_for_research_journals_with_existing_exceptions','evidence_url':'https://www.nature.com/nature/for-authors/formatting-guide','scope':'Publisher directory and journal-specific nonresearch exclusions'},
        {'source_family':'aps','raw_type':'article / letter','decision':'research_except_RMP_and_Physics','evidence_url':'https://journals.aps.org/prl/authors','scope':'APS own metadata; commentary journals explicitly excluded'},
        {'source_family':'crossref_or_openalex','raw_type':'journal-article / article','decision':'not_sufficient_to_confirm_research','evidence_url':'https://www.crossref.org/documentation/principles-practices/journals/','scope':'Generic object class only; no research inference'}]
    type_rules.append({'source_family':'publisher_springer','raw_type':'OriginalPaper / Original Article / Research Paper; ReviewPaper','decision':'original aliases research; ReviewPaper nonprimary; subtype and conflict holds retained','evidence_url':'https://www.springernature.com/gp/open-science/oa-agreements/usa/nist','scope':'Exact matched official Springer page; explicit online year and all existing conflicts required; retained rule receipt supplement/springer_type_aliases/rule.json'})
    type_rules.append({'source_family':'publisher_springer','raw_type':'ASO Visual Abstract / Abstract with ASO Visual Abstract category',
        'decision':'nonprimary companion visual abstract',
        'evidence_url':'https://surgonc.org/wp-content/uploads/2021/02/ASOInstructions_FEB9.2021_web_links.pdf',
        'scope':'Annals of Surgical Oncology with matched journal ISSN, native type and ASO Visual Abstract title prefix; conflicts retained; supplement/aso_visual_abstract/review.json'})
    type_rules.extend([
        {'source_family':'archived_jats','raw_type':'discussion / in-brief / reviewer-report / editor-report / community-comment',
         'decision':'nonprimary commentary, issue summary or peer-review document',
         'evidence_url':'https://jats.nlm.nih.gov/publishing/tag-library/1.4/attribute/article-type.html',
         'scope':'Article-level type only; matched publication identity, holds retained; supplement/native_nonresearch_formats/review.json'},
        {'source_family':'nature_page','raw_type':'ReviewPaper / Focus Review','decision':'nonprimary review',
         'evidence_url':'https://www.nature.com/documents/PJ_GudeToAuthors.pdf',
         'scope':'Polymer Journal only; exact journal title and ISSN, source identity matched; native_nonresearch_formats/review.json'}])
    type_rules.extend([
        {'source_family':'publisher_mdpi_oai','raw_type':'Opinion','decision':'nonprimary viewpoint',
         'evidence_url':'https://www.mdpi.com/about/article_types',
         'scope':'Matched native DOI/title/journal and explicit Opinion type; other formats unaffected; supplement/publisher_opinion_policy/review.json'},
        {'source_family':'publisher_frontiers','raw_type':'Opinion','decision':'nonprimary viewpoint',
         'evidence_url':'https://www.frontiersin.org/for-authors/where-to-publish/article-types',
         'scope':'Matched native DOI/title/journal and explicit Opinion type; conflicts retained; supplement/publisher_opinion_policy/review.json'},
        {'source_family':'nature_page / nature_catalogue',
         'raw_type':'Basic Science Article / Clinical Research Article / Population Study Article',
         'decision':'research subject to date, subtype and integrity checks',
         'evidence_url':'https://www.nature.com/documents/pr_instructions_to_authors.pdf',
         'scope':'Pediatric Research only, exact journal title and ISSN; OriginalPaper alone insufficient; historical and 2025 policies retained in supplement/pediatric_research_types/review.json'}])
    write_csv(root/'supplement/source_type_rules.csv',type_rules)
    ownership={}
    for line in (root/'nature/papers.jsonl').open():
        paper=json.loads(line)
        for author in paper['authors']:
            for slot in author['slots']:
                for scenario,targets in slot['university_targets'].items():
                    for ident,fraction in targets.items():
                        ownership[(slot['entity_id'],ident,scenario,fraction)]={'entity_id':slot['entity_id'],'university_id':ident,'scenario':scenario,'fraction':fraction,
                            'scope':'Inherited Nature current study rule; new hospital/joint branches require matching branch evidence',
                            'evidence_receipt':'nature/papers.jsonl#'+paper['doi'],'publisher_url':paper['publisher_url'],
                            'source_fields':json.dumps(slot.get('source_fields',{}),ensure_ascii=False),'review_status':'inherited_current_scenario_not_newly_revalidated'}
    write_csv(root/'supplement/institution_ownership_rules.csv',list(ownership.values()))
    summary = {'updated_at': datetime.now(timezone.utc).isoformat(), 'task_statuses': tasks, 'pilot': capabilities,
               'boundaries': boundary, 'panel_status': {k: panels[k] for k in ['a', 'b', 'c']},
               'institution_units': len(top), 'university_ownership_units':len(ownership_pending),
               'cost_policy': 'Public sources only; no purchased data or paid model calls',
               'sampling_note': '4000-DOI fixed stratified measurement cohort; other cached/acquired evidence is reused for general supplementation, not counted as pilot observations'}
    (root / 'supplement/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    table = '\n'.join(f"| {r['source']} | {r['attempted']}/{r['targets']} | {r['identity_matched']} | {r['explicit_type_online_year_confirmed']} | {r['access_restricted']} |" for r in capabilities)
    check = json.loads((root / 'fields/qualification/recomputed_results.json').read_text())
    gaps = json.loads((root / 'fields/qualification/research_metadata_gaps_summary.json').read_text())
    if gaps.get('qualified_denominator') != check['verified_research_count']:
        raise ValueError('Stale metadata gap denominator; run the current qualification recomputation first')
    content = f'''# Fig8 统一数据补充进展与完整性报告

更新：{summary['updated_at']}。全部修改写入同一份数据，不创建版本副本。阶段一工具、试测及后续批处理共用任务账本；任务已入账不等于已核实。

当前候选 {check['candidate_count']:,} 条，已核实研究 {check['verified_research_count']:,} 篇，资格待核 {check['unresolved_count']:,} 条。
已纳入论文缺摘要 {gaps['missing_abstract']:,} 篇、至少一位作者缺机构 {gaps['missing_institution']:,} 篇、缺国家 {gaps['missing_country']:,} 篇；去重 {gaps['any_metadata_gap']:,} 篇。这些是字段覆盖，不是全部署名槽位已确认。

## 执行状态与中断恢复

本报告只描述已经应用并复算的检查点；新材料应用资格证据并完成复算后才进入本报告分母。复算状态见 `derived_refresh_status.json`，尚未纳入报告的更新见 `derived_refresh_pending.json`（如存在）。当前来源队列见 `springer_remaining/status.json`、`mdpi_oai/targeted/status.json`、`pmc_journal_batches/status.json`；固定日期批次及后续入库见 `mdpi_oai/remaining_date_deposits/status.json`、`metadata_status.json`、`phase_finish_status.json`、`new_qualified_metadata_status.json`（后三项位于同一目录）。这些队列状态不能代替资格结论。宿主机重启会停止普通后台进程，恢复前须确认进程实际存活和队列锁，不凭旧 PID 或状态文件推断仍在运行。

宿主机中断恢复检查见 `recovery_receipt.json`；受损来源隔离留痕，缺失原文重新获取，不构造替代 XML。当前应用证据是否受影响由该检查单独记录。

## 固定试测集合与来源能力

按领域×年份轮转、期刊优先轮转、DOI固定seed=20261021选择，跨组去重。初次选择器的排序偏差已在解释结果前修正，记录见 `pilot_sampling_correction.json`；此前已取得材料保留为一般补充证据。

| 来源 | 已尝试/目标 | 身份匹配 | 类型和年份可明确处置 | 访问受限 |
|---|---:|---:|---:|---:|
{table}

身份匹配同时要求DOI、标题、期刊一致。可明确处置含纳入与排除；不是预计新增研究论文数。网页下载成功但无匹配身份或明确字段，不计为确认。早期试测及其端点外推预算见 `source_capabilities.csv`；这不是当前采集器的预计完成时间，也不能把网页限制推广到同出版社的 OAI、归档或云端来源。当前路径见 `official_bulk/source_path_review.json`。序列请求小时数不是并发完成时间；失败、限流和期刊间差异会改变估计。其中 Elsevier 试测返回的页面缺少可提取的论文身份字段，500 条均未确认，不能按成功补齐计算。最低限速耗时未包含重试、排队、解析和复算；PMC 官方 OAI 批量回退还受允许时段约束。未采购数据，未调用付费语言模型。

## 来源与更新规则

`tasks.sqlite` 保存候选索引、按DOI/机构去重的任务、逐次尝试和逐字段修改前后值。`pilot.json`固定测量集合。`fields/qualification/supplement_raw`保存公开原文，`supplement_parsed`保存解析和HTTP状态；既有JATS原文也被复用。

只按精确论文身份和唯一同名作者建立链接。机构标准ID必须来自明确ROR声明，或署名中的机构全名与地点对应唯一注册实体。不会凭机构名称模糊匹配、总部国家或ROR父子关系推断归属。高校归并继续使用Nature已核查口径，新的临床/联合分支仍须官方证据。机构候选和正式高校归并规则分表；注册表education标签不作为高校所有权证明。

年度出版社归属由既有官方年目录及同论文、同在线年的主源出版社署名补充；空目录项保持unknown。`annual_portfolio_evidence.csv`只汇总已观察论文，不能外推为整年所有文章的归属。新的类型名称使用明确原创研究字样；特殊类型保持待核。

## 时间边界与机构任务

在已缓存Nature Portfolio和APS官方目录中发现 {boundary['cached_official_directory_candidates_not_in_roster']:,} 个当前名录外的研究型DOI，见 `boundary_candidates.csv`。这些可能属于其他领域、语言或原检索过滤条件；没有字段证据时不追加。当前边界检查不是全出版社普查。

机构单位按作者记录数×受影响论文数排序，共 {len(top):,} 个当前任务，见 `institution_priority.csv`。无法确定机构或分支的任务继续保留；不将没有检索过官方网站的任务称为人工复核完成。

## 绘图就绪状态

- a：Nature既有元数据可用于后续分析，缺失仍按原口径报告；claim/图指标未在本轮计算。
- b：尚未就绪。全期刊资格分母及分组缺失偏差仍待解决，不能将已核实子集直接称为全期刊正式比较。
- c：尚未就绪。继承b的分母限制，且仍缺出版物署名结构及高校所有权证据。

国家前12的未知质量最坏情形稳定性：{panels['country_top12']['stable_under_unrestricted_unknown_mass']}；高校前15：{panels['university_top15']['stable_under_unrestricted_unknown_mass']}。完整分配和假设见 `panel_readiness.json`。未知作者/槽位可支持任意组的上界很保守；没有合理候选范围时不宣称稳定。

## 复算

从仓库根目录运行 `python3 -m experiments.fig8_extension.supplement_report` 重新生成本报告与分组诊断。资格和覆盖率先由统一数据的qualification复算脚本更新。`audit.ipynb`提供可执行入口；没有伪造执行记录。
'''
    (root / 'supplement/完整性报告.md').write_text(content)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'cell_type': 'markdown', 'id': 'scope', 'metadata': {}, 'source': ['# Fig8 数据补充复算\n在仓库根目录运行下面命令；读取当前数据与来源，刷新派生报告。']},
                  {'cell_type': 'code', 'id': 'report', 'metadata': {}, 'execution_count': None, 'outputs': [],
                   'source': ['import subprocess, sys\nsubprocess.run([sys.executable, "-m", "experiments.fig8_extension.supplement_report"], check=True)']}]}
    (root / 'supplement/audit.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2))
    db.close()
    return summary


def render_goal_scope(root: Path = ROOT) -> None:
    """Render the changed human goal from saved results, without recomputing data."""
    report_path=root/'supplement/完整性报告.md'
    scope_path=root/'supplement/active_goal_scope.json'
    if not report_path.exists()or not scope_path.exists():return
    scope=json.loads(scope_path.read_text())
    result=json.loads((root/'fields/qualification/recomputed_results.json').read_text())
    target=scope.get('target_included_unique_research',500000);count=result['verified_research_count']
    note=(f'<!-- active-goal -->\n当前有界目标：至少 {target:,} 篇具备身份、原创研究类型与 2023–2025 在线年份证据的去重论文；'
          f'本次已复算 {count:,} 篇。'
          +('扩量采集已停止；其余候选与字段未知项继续留在统一数据中。'if count>=target and scope.get('expansion_stopped')else'尚未达到扩量停止条件。')
          +'达到该元数据数量目标不代表全期刊比较、国家/高校比较或创新指标分析已经完成。收尾依据见 `target_500k/completion.json`。\n<!-- /active-goal -->\n')
    content=report_path.read_text()
    if '<!-- active-goal -->'in content:
        before,tail=content.split('<!-- active-goal -->',1);_,after=tail.split('<!-- /active-goal -->',1);content=before+note+after
    else:
        first,rest=content.split('\n',1);content=first+'\n\n'+note+rest
    content=content.replace('身份匹配同时要求DOI、标题、期刊一致。',
        '身份匹配核对DOI、期刊及题名。题名按留痕规则标准化；额外题名变体仅通过官方同DOI双语目录、原始完整作者序列或完整摘要等严格证据建立对应关系，并保留原始来源。')
    content=content.replace('新的类型名称使用明确原创研究字样；特殊类型保持待核。',
        '原创类型依据明确主源标签或有官方依据的期刊专属格式规则确认；无法确认的特殊类型继续待核。')
    if count>=target and scope.get('expansion_stopped'):
        content=content.replace('当前来源队列见','历史来源队列留痕见').replace('当前路径见 `official_bulk/source_path_review.json`。','历史路径评估见 `official_bulk/source_path_review.json`，本次补足来源与未解决项见 `target_500k/completion.json`。')
    report_path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = report(args.root)
    render_goal_scope(args.root)
    print(json.dumps({'task_statuses': result['task_statuses'], 'institution_units': result['institution_units']}))


if __name__ == '__main__':
    main()

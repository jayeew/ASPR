"""Aggregate the complete historical Nature source corpus, never export raw rows."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from summarize_claim_graph import numeric, counts

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/claim_graph'
OUT = ROOT / 'outputs/GRAPH_STATISTICS'


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(SOURCE / f'{name}.parquet')


def complete(values: pd.Series) -> dict:
    missing = values.isna() | values.fillna('').astype(str).str.strip().eq('')
    return {'total':len(values),'present':int((~missing).sum()),'missing_or_empty':int(missing.sum()),
            'distinct_present':int(values[~missing].nunique()),'duplicate_present_rows_after_first':int(values[~missing].duplicated().sum())}


def distribution_by(data: pd.DataFrame, group: str, metric: str) -> dict:
    return {str(k):numeric(g[metric]) for k,g in data.groupby(group,dropna=False)}


def cross(data: pd.DataFrame, columns: list[str]) -> list[dict]:
    return data.groupby(columns,dropna=False).size().rename('count').reset_index().fillna('MISSING').to_dict('records')


def local_access(paths: pd.Series) -> dict:
    present=paths.notna() & paths.fillna('').astype(str).str.strip().ne('')
    parents = {str(Path(str(p)).parent) for p in paths[present].unique()}
    accessible = set()
    for parent in parents:
        if not Path(parent).is_dir():
            continue
        with os.scandir(parent) as entries:
            accessible.update(str(Path(parent) / entry.name) for entry in entries if entry.is_file())
    cache={str(p):str(p) in accessible for p in paths[present].unique()}
    roots={}
    for p in cache:
        pieces=Path(p).parts
        root='/'.join(pieces[:3]).replace('//','/') if len(pieces)>2 else str(Path(p).parent)
        roots[root] = roots.get(root,0)+1
    return {'rows':len(paths),'path_present_rows':int(present.sum()),'file_accessible_rows':int(paths[present].astype(str).map(cache).sum()),
            'file_inaccessible_or_absent_rows':int(len(paths)-paths[present].astype(str).map(cache).sum()),'path_mount_counts':roots,
            'method':'Directory scan with DirEntry.is_file at audit time; inaccessible does not prove source never existed; no manuscript contents exported.'}


def normalize_doi(value: object) -> str:
    if pd.isna(value): return ''
    return re.sub(r'^(https?://(dx\.)?doi\.org/|doi:\s*)','',str(value).strip().lower())


def main() -> None:
    roster=read('nature_targets'); works=read('canonical_target_works'); abstracts=read('abstracts'); sentences=read('abstract_sentences'); claims=read('claim_nodes'); embeddings=read('claim_embedding_index')
    assert roster.article_id.is_unique and works.nature_article_id.is_unique
    d=roster.merge(works,left_on='article_id',right_on='nature_article_id',how='left',validate='one_to_one',suffixes=('_roster','_work'))
    d['claim_count']=d.article_id.map(claims.groupby('parent_paper_id').size()).fillna(0).astype(int)
    d['sentence_count']=d.article_id.map(sentences.groupby('article_id').size()).fillna(0).astype(int)
    d['abstract_chars']=d.article_id.map(abstracts.set_index('article_id').abstract_text.fillna('').str.len())
    d['abstract_whitespace_tokens']=d.article_id.map(abstracts.set_index('article_id').abstract_text.fillna('').str.split().str.len())
    d['has_claims']=d.claim_count.gt(0)
    d['has_abstract']=d.abstract_chars.gt(0)
    c=claims.merge(d[['article_id','year','source_name','field_name']],left_on='parent_paper_id',right_on='article_id',how='left',validate='many_to_one')
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'All 24,919 historical Nature-family target papers (2023–2025), distinct from 200/1000 live target-paper study.',
             'paper_count':len(d),'unique_article_keys':d.article_id.nunique(),'canonical_join_missing':int(d.work_id.isna().sum()),
             'paper_distributions':{col:counts(d[col]) for col in ['year','journal_id','source_name','field_name','work_type','hop_min']},
             'year_by_journal':cross(d,['year','source_name']),'year_by_field':cross(d,['year','field_name']),
             'metadata_completeness':{col:complete(d[col]) for col in ['doi_roster','doi_work','title_roster','title_work','publication_date','publication_year','source_id','source_name','primary_topic_id','primary_topic_name','field_id','field_name','received_date_text','accepted_date_text']},
             'normalized_doi':complete(d.doi_roster.map(normalize_doi)),
             'normalized_title':complete(d.title_roster.fillna('').str.casefold().str.replace(r'\s+',' ',regex=True).str.strip()),
             'reference_count':numeric(d.referenced_works_count),
             'reference_count_by_year':distribution_by(d,'year','referenced_works_count'),
             'reference_count_by_field':distribution_by(d,'field_name','referenced_works_count'),
             'local_fulltext_access':local_access(d.paper_markdown_path), 'local_reviewer_access':local_access(d.peer_review_markdown_path),
             'abstracts':{'rows':len(abstracts),'unique_article_keys':abstracts.article_id.nunique(),'papers_with_nonempty_abstract':int(d.has_abstract.sum()),'character_length':numeric(d.abstract_chars),'whitespace_token_count':numeric(d.abstract_whitespace_tokens),
                          'token_note':'Whitespace-separated token proxy, not a model tokenizer; no raw text exported.', 'extraction_method_counts':counts(abstracts.extraction_method),'characters_by_year':distribution_by(d,'year','abstract_chars'),'whitespace_tokens_by_year':distribution_by(d,'year','abstract_whitespace_tokens')},
             'sentences':{'rows':len(sentences),'unique_sentence_keys':sentences.sentence_id.nunique(),'papers_with_sentences':sentences.article_id.nunique(), 'per_paper_including_zero':numeric(d.sentence_count),'per_paper_histogram':counts(d.sentence_count),'character_length':numeric(sentences.sentence_text.fillna('').str.len()),'whitespace_token_count':numeric(sentences.sentence_text.fillna('').str.split().str.len()),'nonnegative_start_offset_rows':int(sentences.markdown_start_char.ge(0).sum()),'nonnegative_end_offset_rows':int(sentences.markdown_end_char.ge(0).sum())},
             'claims':{'rows':len(claims),'unique_claim_keys':claims.claim_id.nunique(),'papers_with_claims':int(d.has_claims.sum()),'papers_without_claims':int((~d.has_claims).sum()),'per_paper_all_24919':numeric(d.claim_count),'per_paper_histogram_all_24919':counts(d.claim_count),'roles':counts(claims.claim_type),'role_type_count':claims.claim_type.nunique(),'by_year_role':cross(c,['year','claim_type']),'by_journal_role':cross(c,['source_name','claim_type']),'by_field_role':cross(c,['field_name','claim_type']),'paper_coverage_by_year':cross(d,['year','has_claims']),'paper_coverage_by_field':cross(d,['field_name','has_claims']),'paper_coverage_by_work_type':cross(d,['work_type','has_claims'])},
             'embedding_coverage':{'indexed_claim_rows':len(embeddings),'indexed_unique_claims':embeddings.claim_id.nunique(),'historical_claims_missing_embedding':int((~claims.claim_id.isin(embeddings.claim_id)).sum()),'papers_with_at_least_one_indexed_claim':int(claims.loc[claims.claim_id.isin(embeddings.claim_id),'parent_paper_id'].nunique())}}
    dates={}
    for col in ['received_date_text','accepted_date_text','publication_date']:
        parsed=pd.to_datetime(d[col],format='mixed',errors='coerce')
        dates[col]={'parseable':int(parsed.notna().sum()),'unparseable_or_missing':int(parsed.isna().sum()),'earliest':str(parsed.min().date()) if parsed.notna().any() else None,'latest':str(parsed.max().date()) if parsed.notna().any() else None}
        d[col+'_parsed']=parsed
    dates['received_to_accepted_days']=numeric((d.accepted_date_text_parsed-d.received_date_text_parsed).dt.days)
    dates['negative_received_to_accepted_intervals']=int(((d.accepted_date_text_parsed-d.received_date_text_parsed).dt.days<0).sum())
    summary['date_parsing']=dates
    missing=set(d.loc[~d.has_claims,'article_id'])
    failures=pd.read_csv(SOURCE/'claim_failures.csv')
    matched=failures[failures.article_id.isin(missing)]
    summary['claim_absence_audit']={'missing_papers':len(missing),'matching_saved_failure_rows':len(matched),'saved_failure_reason_counts':counts(matched.reason),'missing_papers_without_saved_failure_reason':len(missing-set(matched.article_id)),'interpretation':'No saved failure row explains these missing claim inventories; do not infer extraction failure, scientific absence or rejection.','missing_claim_papers_by_year':counts(d.loc[~d.has_claims,'year']),'missing_claim_papers_by_work_type':counts(d.loc[~d.has_claims,'work_type']),'missing_claim_papers_by_journal':counts(d.loc[~d.has_claims,'source_name']),'missing_claim_papers_with_abstract':int(d.loc[~d.has_claims,'has_abstract'].sum())}
    summary['source_inventory']=[{'path':str((SOURCE/f'{name}.parquet').relative_to(ROOT)),'rows':len(frame),'bytes':(SOURCE/f'{name}.parquet').stat().st_size,'mtime_utc':datetime.fromtimestamp((SOURCE/f'{name}.parquet').stat().st_mtime,timezone.utc).isoformat()} for name,frame in [('nature_targets',roster),('canonical_target_works',works),('abstracts',abstracts),('abstract_sentences',sentences),('claim_nodes',claims),('claim_embedding_index',embeddings)]]
    (OUT/'nature_corpus_statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 全部 24,919 篇历史 Nature 语料统计','',f'统计于 {summary["generated_at_utc"]}。此分母为 2023–2025 历史语料，**不是** 200/1,000 篇新论文实验队列。仅统计，不含论文标题、DOI、原文、ID、逐论文表或向量。','',f'共 **{len(d):,} 篇**，均完成 canonical metadata 对齐；**{int(d.has_claims.sum()):,} 篇有 claims**，共 **{len(claims):,} 条**，另 **{int((~d.has_claims).sum())} 篇没有保存 claims**。摘要 **{len(abstracts):,} 条**，摘要句子 **{len(sentences):,} 条**。','', '|年份|论文数|','|---|---:|']
    lines += [f'|{k}|{v:,}|' for k,v in summary['paper_distributions']['year'].items()]
    lines += ['', '|期刊|论文数|','|---|---:|']+[f'|{k}|{v:,}|' for k,v in summary['paper_distributions']['source_name'].items()]
    lines += ['', '|领域|论文数|','|---|---:|']+[f'|{k}|{v:,}|' for k,v in summary['paper_distributions']['field_name'].items()]
    lines += ['', '## 摘要、引用和抽取规模','', '|指标|有效数|均值|中位数|P05|P95|','|---|---:|---:|---:|---:|---:|']
    for name,s in [('每篇引用数',summary['reference_count']),('每篇摘要字符',summary['abstracts']['character_length']),('每篇摘要空白分词数',summary['abstracts']['whitespace_token_count']),('每篇摘要句数',summary['sentences']['per_paper_including_zero']),('每篇主张数（含零）',summary['claims']['per_paper_all_24919'])]:
        lines.append(f'|{name}|{s["valid"]:,}|{s["mean"]:.4g}|{s["median"]:.4g}|{s["p05"]:.4g}|{s["p95"]:.4g}|')
    lines += ['', '## 覆盖与解释边界','', f'- 70,034 条历史 claims 均具有 embedding index；至少一条 embedding 的论文数为 {summary["embedding_coverage"]["papers_with_at_least_one_indexed_claim"]:,}。', '- 缺少 claims 的 5 篇都有摘要，但 claim_failures.csv 中没有对应失败记录；原因未知，不能改写成“无贡献”或推断为执行失败。',f'- 当前环境可访问的全文文件：{summary["local_fulltext_access"]["file_accessible_rows"]:,}/{len(d):,}；审稿文件：{summary["local_reviewer_access"]["file_accessible_rows"]:,}/{len(d):,}。此项只检查路径可访问性，不代表重新全文核验。', '- 空白分词数只是长度代理，不是模型 tokenizer 长度；学科来自保存的 canonical metadata，没有新增人工验证。', '- JSON 还提供年度×期刊/领域、贡献角色×年度/期刊/领域、年度/领域/work type 主张覆盖、元数据完整度/去重、收稿录用日期及间隔、全部来源文件统计。']
    (OUT/'nature_corpus_summary.md').write_text('\n'.join(lines)+'\n')
    assert len(d)==24919 and int(d.has_claims.sum())==24914 and len(claims)==70034
    print('Nature corpus aggregate report complete:',len(d),'papers')


if __name__=='__main__':
    main()

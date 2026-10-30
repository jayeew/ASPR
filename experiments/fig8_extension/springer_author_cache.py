"""Recover explicit publisher HTML/XML byline-to-address links from existing receipts."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from lxml import html

from .aps_authors import jobs as aps_jobs, source_record as aps_source_record
from .arrow_row_patch import patch_rows, selected_rows
from .frontiers_authors import parse_authors as parse_frontiers_authors
from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import country_names
from .primary_country_resolution import index_authors, resolve
from .rrpharm_authors import parse_authors as parse_rrpharm_authors
from .supplement_apply import abstract_ok, complete_byline, enrich_authors, signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv


def parse_authors(raw: bytes | str) -> dict:
    tree=html.fromstring(raw)
    lists=tree.xpath('//ul[@data-test="authors-list"]')
    if len(lists)!=1:return {'authors':[], 'byline_has_etal':True}
    affiliations={}
    for node in tree.xpath('//li[@id][p[@class="c-article-author-affiliation__address"]]'):
        parts=node.xpath('./p[@class="c-article-author-affiliation__address"]')
        affiliations[node.get('id')]={'text':' '.join(parts[0].text_content().split()),'countries':[],'identifiers':[],'names':[]}
    authors=[]
    for node in lists[0].xpath('./li'):
        names=node.xpath('./a[@data-test="author-name"]')
        if len(names)!=1:continue
        refs=node.xpath('./sup/a[starts-with(@href,"#Aff")]/@href')
        refs=list(dict.fromkeys(ref[1:] for ref in refs))
        authors.append({'name':' '.join(names[0].text_content().split()),'nonpersonal':False,
            'orcid':next(iter(node.xpath('.//a[contains(@class,"js-orcid")]/@href')),''),
            'affiliations':[affiliations[r] for r in refs if r in affiliations],
            'unresolved_affiliation_refs':[r for r in refs if r not in affiliations]})
    meta=tree.xpath('//meta[@name="citation_author"]/@content')
    normalized_meta=[key(' '.join(reversed(v.split(', ',1)))) if ', ' in v else key(v) for v in meta]
    complete=bool(authors) and [key(v['name']) for v in authors]==normalized_meta
    return {'authors':authors,'byline_has_etal':not complete,
            'byline_check':'DOM byline order agrees with complete citation-author metadata' if complete else 'Completeness not confirmed'}


def parse_country_authors(raw: str) -> dict:
    """Certify individual explicit links without certifying a truncated/discordant byline."""
    data = parse_authors(raw)
    tree = html.fromstring(raw)
    nodes = defaultdict(list)
    present_by_id = defaultdict(list)
    for note in tree.xpath('//*[starts-with(@id,"nAff")]/p[contains(concat(" ",normalize-space(@class)," ")," js-present-address ")]'):
        present_by_id['#' + note.getparent().get('id')].append(' '.join(note.text_content().split()))
    for node in tree.xpath('//ul[@data-test="authors-list"]/li'):
        names = node.xpath('./a[@data-test="author-name"]')
        if len(names) == 1:
            nodes[key(' '.join(names[0].text_content().split()))].append(node)
    for author in data['authors']:
        matches = nodes[key(author['name'])]
        certified = False
        if len(matches) == 1:
            node = matches[0]
            refs = node.xpath('./sup/a/@href')
            group = node.xpath('./a[@data-test="author-name" and starts-with(@data-author-popup,"group-")]')
            present = []
            for ref in refs:
                notes = present_by_id.get(ref, [])
                if ref.startswith('#nAff') and len(notes) == 1:
                    present.append({'reference': ref, 'text': notes[0],
                                    'rule': 'Existing Nature research-worksite scope; present address retained separately'})
            excluded = {n['reference'] for n in present}
            certified = bool(refs) and not group and all(r.startswith('#Aff') or r in excluded for r in refs)
            certified = certified and bool(author['affiliations']) and not author['unresolved_affiliation_refs']
            if present:
                author['excluded_present_addresses'] = present
        author['country_link_certified'] = bool(certified)
    data['country_certification_scope'] = 'explicit_individual_links_only'
    data['native_country_resolution_certified'] = any(a['country_link_certified'] for a in data['authors'])
    return data


def source_digest(record: dict, raw_hash: str) -> str:
    full = 'Proceedings of the National Academy of Sciences of the United States of America'
    suffix = ':pnas-journal-alias-1' if record.get('route')=='pmc_archive_priority' and full in record.get('parsed',{}).get('journal_titles',[]) else ''
    if record.get('source')=='pmc' and not record.get('parsed',{}).get('authors') and not record.get('parse_authors_from_raw'):
        suffix+=':jats-explicit-author-group-1'
    if record.get('raw_text_encoding'):
        suffix += ':html-encoding-' + record['raw_text_encoding']
    return 'address-country-review-2:' + raw_hash + suffix


def already_processed(db: sqlite3.Connection, root: Path, wid: str, receipt: str) -> bool:
    """Skip immutable, already-applied receipt hashes before reading a master shard."""
    record = json.loads((root / receipt).read_text())
    digest = record.get('raw_sha256')
    if not digest:
        return False
    return db.execute(
        'SELECT 1 FROM springer_author_processed WHERE work_id=? AND receipt=? AND digest=?',
        (wid, receipt, source_digest(record,digest))).fetchone() is not None


def prepare_source(root: Path, paper: dict, receipt: str, issns: set[str],
                   aps_only: bool = False) -> tuple[dict, str, dict]:
    """Read and parse immutable cache evidence; never write shared data."""
    if aps_only:
        record, raw = aps_source_record(root, receipt, paper['doi'])
    else:
        record = json.loads((root / receipt).read_text())
        raw = gzip.decompress((root / 'fields/qualification' / record['raw_receipt']).read_bytes())
    raw_hash = hashlib.sha256(raw).hexdigest()
    digest = 'aps-native-byline-1:' + raw_hash if aps_only else source_digest(record, raw_hash)
    if not signature_match(paper, record['parsed'], issns):
        return record, digest, {}
    data = (record['parsed'] if record.get('route') in {'copernicus', 'aps', 'elife_api'}
            or record.get('source') in {'pmc', 'publisher_jats', 'publisher_native_xml'} else parse_authors(raw.decode(record['raw_text_encoding']) if record.get('raw_text_encoding') else raw))
    if record.get('source') in {'pmc', 'publisher_jats'} and not data.get('authors'):
        data = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
    if record.get('route') == 'frontiers':
        data = parse_frontiers_authors(raw, paper['doi'])
    if record.get('route') == 'sciopen':
        from .sciopen_metadata import parse_authors as parse_sciopen_authors
        data = parse_sciopen_authors(raw)
    if record.get('route') in {'rrpharmacology_html_byline', 'ojs_html_byline'}:
        data = parse_rrpharm_authors(raw, paper['doi'])
    if not data.get('abstract') and record['parsed'].get('abstract'):
        data = {**data, 'abstract': record['parsed']['abstract']}
    return record, digest, data


def prepared_sources(root: Path, papers: list[dict], targets: dict[str, str],
                     journals: dict[str, set[str]], aps_only: bool,
                     workers: int) -> Iterator[tuple[dict, tuple[dict, str, dict]]]:
    """Bound prefetch to twice the workers, retaining deterministic write order."""
    selected = [paper for paper in papers if paper['work_id'] in targets]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(selected), workers * 2):
            batch = selected[start:start + workers * 2]
            futures = [pool.submit(prepare_source, root, paper, targets[paper['work_id']],
                                   journals[paper['source_id']], aps_only) for paper in batch]
            for paper, future in zip(batch, futures):
                yield paper, future.result()


def run(root: Path = ROOT, aps_only: bool = False,
        selected_jobs: list[tuple[str, str, str]] | None = None,
        workers: int = 4, fill_abstract: bool = False, reprocess_selected: bool = False) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError('Cache workers must be between 1 and 8')
    if reprocess_selected and selected_jobs is None:
        raise ValueError('Reprocessing requires explicit selected jobs and a scoped source adapter')
    stats=Counter();out=root/'supplement/springer_author_cache';out.mkdir(exist_ok=True)
    with (root/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(root)
        db.execute('CREATE TABLE IF NOT EXISTS springer_author_processed(work_id TEXT,receipt TEXT,digest TEXT,status TEXT,PRIMARY KEY(work_id,receipt,digest))')
        jobs=defaultdict(list)
        candidates = selected_jobs if selected_jobs is not None else db.execute("SELECT p.work_id,p.shard,j.receipt FROM papers p JOIN publisher_remaining_jobs j USING(doi) WHERE p.status='included_verified_research' AND j.route IN ('springer','copernicus','pmc_archive_priority','frontiers') AND j.status='parsed' AND j.applied=1")
        for wid,shard,receipt in candidates:
            if not aps_only:
                if not reprocess_selected and already_processed(db,root,wid,receipt):stats['unchanged_receipts_skipped_before_shard_read']+=1
                else:jobs[shard].append((wid,receipt))
        if aps_only:
            if selected_jobs is None:
                jobs=aps_jobs(root,db)
            else:
                for wid,shard,receipt in selected_jobs:
                    jobs[shard].append((wid,receipt))
        registry=defaultdict(list);by_name=defaultdict(list)
        for r in json.loads((root/'fields/institution_registry.json').read_text()):
            if r.get('ror'):registry[r['ror'].rstrip('/')].append(r['id'])
            by_name[key(r['display_name'])].append(r)
        names=country_names()
        for value in ['korea','georgia','congo']:names.pop(normalized(value),None)
        journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(root/'fields/journal_registry.csv')}
        for shard,selected in jobs.items():
            targets=dict(selected);path=root/'fields/papers'/shard;table=pq.read_table(path)
            rows=selected_rows(table,set(targets));changed=False;patches={}
            for paper, (record, digest, author_data) in prepared_sources(
                    root, rows, targets, journals, aps_only, workers):
                wid=paper['work_id']
                receipt=targets[wid]
                if not reprocess_selected and db.execute('SELECT 1 FROM springer_author_processed WHERE work_id=? AND receipt=? AND digest=?',(wid,receipt,digest)).fetchone():continue
                def finish(status: str) -> None:
                    db.execute('INSERT OR IGNORE INTO springer_author_processed VALUES(?,?,?,?)',(wid,receipt,digest,status));stats[status]+=1
                if not signature_match(paper,record['parsed'],journals[paper['source_id']]):finish('identity_unmatched');continue
                source={**author_data,'evidence_receipt':'fields/qualification/'+record['raw_receipt']}
                if not source['authors']:finish('no_explicit_byline');continue
                current=json.loads(paper['authors_json'])
                if any(a.get('nature_v1_slots') for a in current):finish('nature_review_preserved');continue
                existing_primary=any(a.get('publisher_affiliation_slots') for a in current)
                same_source=all(not a.get('publisher_affiliation_slots') or a.get('publisher_affiliation_source')==source['evidence_receipt'] for a in current)
                if existing_primary and not same_source:finish('existing_primary_requires_source_comparison');continue
                before=dict(paper)
                if fill_abstract and not paper['abstract'] and abstract_ok(author_data.get('abstract', '')):
                    paper['abstract'] = author_data['abstract']
                    paper['abstract_source'] = 'matched_publisher:' + source['evidence_receipt']
                    stats['abstracts_filled'] += 1
                if existing_primary:
                    authors=current;complete=False
                    if reprocess_selected:
                        authors,delta=enrich_authors(current,source,registry,by_name,names);stats.update(delta)
                else:
                    extended,complete=complete_byline(current,source)
                    authors,delta=enrich_authors(extended,source,registry,by_name,names);stats.update(delta)
                other_sources=[json.loads(v[0]) for v in db.execute("SELECT payload FROM source_metadata WHERE doi=? AND kind='jats'",(paper['doi'],))]
                other_sources=[v for v in other_sources if signature_match(paper,v,journals[paper['source_id']])]
                country_sources=[index_authors(v) for v in [source,*other_sources]]
                reviewed=[]
                for author in authors:
                    updated,evidence=resolve(author,country_sources,names);reviewed.append(updated)
                    if evidence:
                        stats['lower_authority_country_conflicts_resolved']+=1
                        entity=wid+':'+str(author['position'])
                        db.execute("UPDATE tasks SET status='resolved',receipt=?,reason=? WHERE kind='country_conflict' AND entity_key IN (?,?)",(json.dumps(evidence),'Explicit agreeing primary addresses supersede lower-authority country; before values retained',entity,entity+':secondary'))
                paper['authors_json']=json.dumps(reviewed,ensure_ascii=False)
                if complete:
                    paper['author_list_may_be_truncated']=False
                    paper['authorships_source_receipt']=source['evidence_receipt']
                    paper['full_authorships_status']='publisher_byline_reconstructed_unique_exact_links'
                values=metrics(paper)
                for metric,value in values.items():
                    if 'after_'+metric in paper:paper['after_'+metric]=value
                paper['metadata_final_status']='gaps_remain' if not all(values[k] for k in ['abstract','every_author_institution','every_author_country']) else 'observed_fields_present_not_publisher_verified'
                fields=[field for field in paper if paper[field]!=before[field]]
                for field in fields:
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                        (wid,field,json.dumps(before[field]),json.dumps(paper[field]),source['evidence_receipt'],'Exact paper identity and unique author identity; explicit publisher affiliation links; no headquarters or university-parent inference'))
                if fields:
                    changed=True;stats['papers_changed']+=1
                    patches[wid]={field:paper[field] for field in fields}
                db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',(not values['abstract'],not values['every_author_institution'],not values['every_author_country'],wid))
                finish('cache_processed')
            if changed:
                temporary=path.with_suffix('.parquet.tmp');pq.write_table(patch_rows(table,patches),temporary,compression='zstd');temporary.replace(path)
            db.commit()
        db.close()
    result={**stats,'network_requests':0,'cache_workers':workers,'write_workers':1}
    with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    atomic_json(out/'status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()

"""Public-source adapters with per-host pacing, explicit receipts and resumable tasks."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from copy import deepcopy
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
from itertools import zip_longest
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
from lxml import etree, html

from .fig8b_qualification_evidence import parse_front
from .supplement_queue import ROOT, attempt, connect, select_pilot, task

LOCAL = threading.local()
LOCK = threading.Lock()
NEXT: dict[str, float] = {}


def atomic_bytes(path: Path, data: bytes) -> None:
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def atomic_json(path: Path, record: dict) -> None:
    atomic_bytes(path,json.dumps(record,ensure_ascii=False).encode())


def paced(url: str) -> None:
    host = urlparse(url).netloc
    interval = 0.34 if 'ebi.ac.uk' in host else 1.05
    with LOCK:
        now = time.monotonic()
        start = max(now, NEXT.get(host, now))
        NEXT[host] = start + interval
    time.sleep(max(0, start - time.monotonic()))


def text(node: etree._Element) -> str:
    return ' '.join(''.join(node.itertext()).split())


def affiliation_text(node: etree._Element) -> str:
    """Exclude structured labels/IDs, preserving all declared address text."""
    clean = deepcopy(node)
    for child in clean.xpath('./label | .//institution-id'):
        parent = child.getparent()
        previous = child.getprevious()
        if child.tail:
            if previous is not None:
                previous.tail = (previous.tail or '') + child.tail
            else:
                parent.text = (parent.text or '') + child.tail
        parent.remove(child)
    return text(clean)


def separate_affiliation_notes(author: dict) -> dict:
    """Retain exact contribution/contact notes separately from worksite addresses."""
    addresses = []; notes = list(author.get('non_affiliation_notes', []))
    for affiliation in author.get('affiliations', []):
        label = ''.join(c for c in affiliation.get('text', '').casefold() if c.isalnum()).lstrip('0123456789')
        if (label in {'theseauthorscontributedequally', 'authorscontributedequally', 'leadcontact'}
                and not any(affiliation.get(k) for k in ['countries', 'identifiers', 'names'])):
            if affiliation not in notes: notes.append(affiliation)
        else: addresses.append(affiliation)
    return {**author, 'affiliations': addresses, **({'non_affiliation_notes': notes} if notes else {})}


def shared_group_affiliation(author: etree._Element) -> dict | None:
    """Read a single group-wide affiliation only in an unlinked contributor group."""
    group = author.getparent()
    if group is None or group.tag != 'contrib-group':
        return None
    shared = group.findall('./aff')
    if len(shared) != 1 or group.xpath('./contrib/xref[@ref-type="aff"] | ./contrib/aff'):
        return None
    aff = shared[0]
    if not affiliation_text(aff):
        return None
    return {'text': affiliation_text(aff),
            'countries': [{'text': text(c), 'code': c.get('country', '')} for c in aff.findall('.//country')],
            'identifiers': [{'type': i.get('institution-id-type', ''), 'value': text(i)} for i in aff.findall('.//institution-id')],
            'names': [text(n) for n in aff.findall('.//institution')],
            'link_basis': 'single_unreferenced_affiliation_for_contributor_group'}


def jats_metadata(data: bytes) -> dict:
    result = parse_front(data)
    article = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))
    meta = article.find('./front/article-meta')
    language = article.get('{http://www.w3.org/XML/1998/namespace}lang', '').casefold()
    abstracts = meta.findall('./abstract')
    abstracts = [a for a in abstracts if a.get('{http://www.w3.org/XML/1998/namespace}lang', language).casefold() in ['', 'en']]
    auxiliary_types = {'graphical', 'toc', 'teaser', 'executive-summary', 'significance',
                       'author-summary', 'plain-language-summary'}
    result['auxiliary_abstracts'] = [
        {'type': a.get('abstract-type'), 'text': text(a)}
        for a in abstracts if a.get('abstract-type', '').casefold() in auxiliary_types]
    abstracts = [a for a in abstracts if a.get('abstract-type', '').casefold() not in auxiliary_types]
    result['abstract'] = '\n\n'.join(text(p) for a in abstracts for p in (a.xpath('./p|./sec') or [a]))
    result['abstract_language'] = language or 'not_explicit_candidate_language_required'
    affiliations = {}
    for aff in meta.xpath('.//aff'):
        affiliations[aff.get('id', '')] = {'text': affiliation_text(aff),
            'countries': [{'text': text(c), 'code': c.get('country', '')} for c in aff.findall('.//country')],
            'identifiers': [{'type': i.get('institution-id-type', ''), 'value': text(i)} for i in aff.findall('.//institution-id')],
            'names': [text(n) for n in aff.findall('.//institution')]}
    authors = []
    for author in meta.xpath('./contrib-group/contrib[@contrib-type="author"] | ./contrib-group[@content-type="author"]/contrib[not(@contrib-type)]'):
        name = author.find('./name')
        full_name = ' '.join(filter(None, [name.findtext('given-names'), name.findtext('surname')])) if name is not None else ''
        if not full_name:
            names = author.findall('./string-name') + author.findall('./collab')
            full_name = text(names[0]) if names else ''
        refs = [r for node in author.xpath('./xref[@ref-type="aff"]') for r in (node.get('rid') or '').split()]
        addresses = [affiliations[r] for r in refs if r in affiliations]
        addresses += [{'text': affiliation_text(a), 'countries': [{'text': text(c), 'code': c.get('country', '')} for c in a.findall('.//country')], 'identifiers': [], 'names': [text(n) for n in a.findall('.//institution')]} for a in author.findall('./aff')]
        if not refs and not addresses:
            shared = shared_group_affiliation(author)
            if shared is not None:
                addresses.append(shared)
        authors.append({'name': full_name, 'nonpersonal': author.find('./collab') is not None,
                        'orcid': author.xpath('string(./contrib-id[@contrib-id-type="orcid"])'), 'affiliations': addresses,
                        'unresolved_affiliation_refs': [r for r in refs if r not in affiliations]})
    result['authors'] = [separate_affiliation_notes(author) for author in authors]
    result['byline_has_etal'] = bool(meta.xpath('./contrib-group//etal'))
    return result


def publisher_url(doi: str, source: str) -> str:
    if doi.strip().casefold().startswith('10.1063/'):
        return 'https://doi.org/' + quote(doi.strip(), safe='/():')
    if doi.strip().casefold().startswith('10.1002/btpr.'):
        return 'https://aiche.onlinelibrary.wiley.com/doi/' + quote(doi.strip(), safe='/():')
    if source == 'sciopen':
        if not doi.lower().startswith('10.26599/'):
            raise ValueError('SciOpen adapter requires the reviewed 10.26599 scope')
        return 'https://www.sciopen.com/article/' + quote(doi.upper(), safe='/():')
    if source == 'nature':
        if not doi.lower().startswith('10.1038/'):
            raise ValueError('Nature page route requires a 10.1038 DOI')
        return 'https://www.nature.com/articles/' + quote(doi.split('/', 1)[1], safe='():')
    if source=='imr':
        part=re.fullmatch(r'10\.31083/j\.(fbe|fbs|fbl)(\d{2})(\d{2})\d+',doi)
        if not part:raise ValueError('Unvalidated IMR DOI path')
        journal,volume,issue=part.groups()
        return f'https://www.imrpress.com/journal/{journal.upper()}/{int(volume)}/{int(issue)}/{doi}'
    suffix = quote(doi, safe='/():')
    prefixes = {'elsevier': 'https://linkinghub.elsevier.com/retrieve/doi/',
                'wiley': 'https://onlinelibrary.wiley.com/doi/', 'springer': 'https://link.springer.com/article/',
                'acs': 'https://pubs.acs.org/doi/', 'iop': 'https://iopscience.iop.org/article/',
                'rsc': 'https://pubs.rsc.org/en/content/articlelanding/',
                'frontiers': 'https://www.frontiersin.org/articles/'}
    return prefixes.get(source, 'https://doi.org/') + suffix


def native_html_text(data: bytes, route: str) -> bytes | str:
    """Decode known UTF-8 native pages before lxml's legacy byte fallback."""
    if route in {'nature', 'springer'}:
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return data
    return data


def parse_publisher(data: bytes | str) -> dict:
    tree = html.fromstring(data)
    meta: dict[str, list] = {}
    for element in tree.xpath('//meta[@content]'):
        name = element.get('name') or element.get('property') or ''
        meta.setdefault(name.lower(), []).append(element.get('content'))
    doi = meta.get('citation_doi') or meta.get('dc.identifier') or []
    doi = [re.sub(r'^https?://(?:dx\.)?doi\.org/|^doi:', '', d, flags=re.IGNORECASE).strip().lower() for d in doi if '10.' in d]
    title = meta.get('citation_title') or meta.get('dc.title') or []
    if not any(t.strip() for t in title) and doi and meta.get('citation_journal_title'):
        headings = tree.xpath('//h1[@data-test="article-title"]')
        if len(headings) == 1:
            title = [' '.join(headings[0].text_content().split())]
    kind = list(dict.fromkeys((meta.get('citation_article_type') or []) + (meta.get('dc.type') or [])))
    type_review = {}
    if meta.get('citation_publisher') == ['IMR Press']:
        genres = set()
        for node in tree.xpath('//script[@type="application/ld+json"]/text()'):
            try:
                article = json.loads(node)
            except json.JSONDecodeError:
                continue
            if not isinstance(article,dict) or article.get('@type')!='ScholarlyArticle':continue
            if not any(article.get('@id','').endswith('/'+d) for d in doi):continue
            genre=article.get('genre')
            if isinstance(genre,str) and genre:genres.add(genre)
        type_review={'citation_types':kind,'matched_article_genres':sorted(genres)}
        if len(genres)==1:kind=sorted(genres)
    dates = meta.get('citation_online_date') or []
    explicit_dates = tree.xpath('//time[@itemprop="datePublished"]/@datetime')
    # Unqualified publication dates and JSON-LD datePublished are retained, never substituted for electronic dates.
    online, precision = ('', 'missing_explicit_electronic_date')
    if dates:
        normalized = [d.replace('/', '-')[:10] for d in dates]
        if len({d[:4] for d in normalized}) == 1 and all(re.match(r'^\d{4}-\d{2}-\d{2}$', d) for d in normalized):
            try:
                date.fromisoformat(min(normalized))
                online, precision = min(normalized), 'day'
            except ValueError:
                online, precision = '', 'invalid_explicit_electronic_date'
    date_review = {}
    if any(d.startswith('10.3389/') for d in doi):
        history = {}
        for label in ['Published', 'Accepted']:
            values = tree.xpath('//p[normalize-space(text())="'+label+'"]/following-sibling::p[1]/text()')
            parsed_dates = set()
            for value in values:
                try:
                    parsed_dates.add(datetime.strptime(value.strip(), '%d %B %Y').date().isoformat())
                except ValueError:
                    continue
            history[label] = sorted(parsed_dates)
        date_review = {'citation_online_date': dates, 'explicit_article_history': history}
        if len(history['Published']) == 1:
            online, precision = history['Published'][0], 'day'
            date_review['basis'] = 'Frontiers explicit Published article-history label; citation_online_date can duplicate Accepted'
        else:
            online, precision = '', 'frontiers_publication_history_missing_or_conflicting'
    nodes = tree.xpath('//*[@id="Abs1-content"]//p|//section[@data-title="Abstract"]//p|//*[@id="abstract"]//p|//*[contains(concat(" ",normalize-space(@class)," ")," abstractSection ")]//p')
    abstract = '\n\n'.join(dict.fromkeys(text(n) for n in nodes if text(n))) or '\n\n'.join(meta.get('citation_abstract', []))
    return {'doi': doi, 'title': title, 'issns': meta.get('citation_issn', []),
            'journal_titles': meta.get('citation_journal_title', []), 'article_types': kind, 'article_type_review':type_review,
            'online_date': online, 'date_precision': precision, 'date_field_review': date_review, 'abstract': abstract,
            'other_dates_not_used': {'citation_publication_date': meta.get('citation_publication_date', []), 'web_datePublished': explicit_dates},
            'publisher': meta.get('citation_publisher', [])}


def request(item: dict, root: Path, source: str) -> dict:
    qualification = root / 'fields/qualification'
    pmcid = item.get('pmcid', '')
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML' if source == 'pmc' else publisher_url(item['doi'], item['route'])
    token = hashlib.sha256(url.encode()).hexdigest()
    folder = qualification / 'supplement_parsed'
    folder.mkdir(exist_ok=True)
    dest = folder / (token + '.json')
    previous_attempts=[]
    if dest.exists():
        existing = json.loads(dest.read_text())
        if existing.get('status') in ['retryable_error','request_error'] and len(existing.get('attempts',[]))>=3:
            existing['status']='retry_exhausted'
            atomic_json(dest,existing)
        if existing.get('status') not in ['retryable_error', 'request_error']:
            return {**existing, 'reused_receipt': True}
        previous_attempts=existing.get('attempts',[])
    session = getattr(LOCAL, 'session', None)
    if session is None:
        LOCAL.session = session = requests.Session()
        session.trust_env = False
        session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (public metadata; paced requests)'
    record = {'doi': item['doi'], 'pmcid': pmcid, 'source': source, 'route': item['route'],
              'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': list(previous_attempts)}
    started = time.monotonic()
    raw = b''
    for n in range(len(previous_attempts),3):
        paced(url)
        record['attempts'].append({'try':n+1,'at':datetime.now(timezone.utc).isoformat()})
        try:
            with session.get(url, timeout=(10, 35), stream=True) as response:
                code = response.status_code
                record.update(http_status=code, final_url=response.url)
                record['http_request_count']=record.get('http_request_count',0)+1+len(getattr(response,'history',[]))
                record['attempts'][-1]['http_status']=code
                if code in [401, 403, 404, 410]:
                    record['status'] = 'access_restricted' if code in [401, 403] else 'not_available'
                    raw = response.content[:500000]
                    break
                if code==429:
                    record.update(status='retry_deferred',retry_after=response.headers.get('Retry-After',''))
                    break
                if code in [500, 502, 503, 504]:
                    record['status'] = 'retryable_error'
                    if n < 2:
                        delay = response.headers.get('Retry-After', '')
                        if delay.isdigit() and int(delay) > 60:
                            record['defer_retry_after_seconds'] = int(delay)
                            break
                        time.sleep(max(2 ** n, int(delay) if delay.isdigit() else 0))
                    continue
                if not response.ok:
                    record['status'] = 'http_error'
                    break
                for chunk in response.iter_content(65536):
                    raw += chunk
                    if source == 'pmc' and b'</front>' in raw:
                        raw = raw[:raw.index(b'</front>') + len(b'</front>')]
                        break
                    if len(raw) > 8_000_000:
                        raise ValueError('Metadata capture exceeds 8 MB')
                if source == 'pmc':
                    if not raw.endswith(b'</front>'):
                        raise ValueError('No complete JATS front matter')
                    record['parsed'] = jats_metadata(raw + b'</article>')
                else:
                    page = native_html_text(raw, item['route'])
                    record['parsed'] = parse_publisher(page)
                    if isinstance(page, str):
                        record['raw_text_encoding'] = 'utf-8'
                    if item['route'] == 'sciopen':
                        from .sciopen_metadata import enrich_metadata
                        record['parsed'] = enrich_metadata(raw, parse_publisher(raw.decode('utf-8')))
                        record['raw_text_encoding'] = 'utf-8'
                record['status'] = 'parsed'
                break
        except (requests.RequestException, ValueError, etree.XMLSyntaxError, etree.ParserError) as error:
            record['attempts'][-1]['error']=type(error).__name__
            record.update(status='request_error', error=type(error).__name__)
            if raw:
                failed_folder=qualification/'supplement_raw';failed_folder.mkdir(exist_ok=True)
                failed_path=failed_folder/(token+f'.attempt_{n+1}.failed.gz')
                atomic_bytes(failed_path,gzip.compress(raw,mtime=0))
                record['attempts'][-1]['raw_failure_receipt']=str(failed_path.relative_to(qualification))
            raw = b''
            if n < 2:
                time.sleep(2 ** n)
    if record.get('status') in ['retryable_error','request_error'] and len(record['attempts'])>=3 and not record.get('defer_retry_after_seconds'):
        record['status']='retry_exhausted'
    raw_folder = qualification / 'supplement_raw'
    raw_folder.mkdir(exist_ok=True)
    raw_path = raw_folder / (token + '.xml.gz' if source == 'pmc' else token + '.html.gz')
    if raw:
        atomic_bytes(raw_path,gzip.compress(raw,mtime=0))
        record['raw_receipt'] = str(raw_path.relative_to(qualification))
        record['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    record.update(elapsed_seconds=time.monotonic() - started, retained_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
                  parsed_receipt=str(dest.relative_to(qualification)), capture_scope='JATS exact front prefix' if source == 'pmc' else 'Publisher HTML metadata page')
    atomic_json(dest,record)
    if source == 'pmc' and record['status'] == 'parsed':
        # The existing qualification builder reads this directory; keep its receipt shape.
        native_raw = qualification / 'jats_raw' / (pmcid + '.front.xml.gz')
        if not native_raw.exists():
            native_raw.symlink_to('../' + str(raw_path.relative_to(qualification)))
        native = {**record, 'raw_receipt': str(native_raw.relative_to(qualification))}
        atomic_json(qualification/'jats_parsed'/(pmcid+'.json'),native)
    return record


def bounded_results(jobs: list[tuple[dict, str]], root: Path, workers: int, deadline: float | None = None) -> Iterator[tuple[dict, str, dict]]:
    """Keep only live requests queued; a stop marker drains those requests safely."""
    pending = iter(jobs)
    stop = root / 'supplement/stop_acquisition'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        def refill() -> None:
            while len(futures) < workers and not stop.exists() and (deadline is None or time.monotonic() < deadline):
                pair = next(pending, None)
                if pair is None:
                    break
                item, source = pair
                futures[pool.submit(request, item, root, source)] = pair
        refill()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item, source = futures.pop(future)
                yield item, source, future.result()
            refill()


def run(root: Path = ROOT, mode: str = 'pilot', workers: int = 8, checkpoint_every: int = 0) -> dict:
    from .supplement_acquisition import plan

    db = connect(root)
    if mode == 'pilot':
        groups = {}
        for item in select_pilot(root):
            source = 'pmc' if item['arm'] == 'pmc' else 'publisher'
            groups.setdefault(item['arm'], []).append((item, source))
        jobs = [entry for batch in zip_longest(*groups.values()) for entry in batch if entry is not None]
    else:
        jobs = plan(root)
    started = time.monotonic()
    initial_targets = len(jobs)
    counts: dict[str, int] = {}
    processed = set()
    completed = 0
    def progress(remaining: int) -> dict:
        state = {'mode': mode, 'completed': completed, 'targets_at_start': initial_targets,
                 'remaining_after_replanning': remaining, 'statuses': counts,
                 'wall_seconds': time.monotonic() - started, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'status': 'stopped_after_inflight' if (root / 'supplement/stop_acquisition').exists() else 'running' if remaining else 'complete'}
        atomic_json(root / 'supplement' / (mode + '_retrieval_status.json'), state)
        return state
    progress(len(jobs))
    while jobs and not (root / 'supplement/stop_acquisition').exists():
        batch = jobs[:checkpoint_every] if mode == 'pmc' and checkpoint_every else jobs
        completed_before_batch = completed
        for item, source, result in bounded_results(batch, root, workers):
            processed.add(item.get('pmcid') or item['doi'])
            entity = item['doi'] or item['work_id']
            ident = hashlib.sha256(('qualification:' + entity).encode()).hexdigest()[:32]
            if not db.execute('SELECT 1 FROM tasks WHERE task_id=?', (ident,)).fetchone():
                ident = task(db, 'qualification', entity, item['route'], {'doi': item['doi'], 'pmcids': [item.get('pmcid', '')]})
            if not result.get('reused_receipt'):
                attempt(db, ident, source, 'evidence_ready' if result['status'] == 'parsed' else result['status'],
                        'fields/qualification/' + result['parsed_receipt'], {'http_status': result.get('http_status'),
                        'reason': result.get('error', result['status']), 'elapsed_seconds': result.get('elapsed_seconds'), 'reused': False})
            counts[result['status']] = counts.get(result['status'], 0) + 1
            completed += 1
            if completed % 100 == 0:
                print(json.dumps(progress(max(0, len(jobs) - (completed - completed_before_batch)))), flush=True)
        if mode == 'pilot' or not checkpoint_every or (root / 'supplement/stop_acquisition').exists():
            break
        subprocess.run([sys.executable, '-m', 'experiments.fig8_extension.supplement_run', 'checkpoint'], check=True)
        jobs = [(item, source) for item, source in plan(root) if item['pmcid'] not in processed]
        progress(len(jobs))
    final = progress(0 if not (root / 'supplement/stop_acquisition').exists() else max(0, initial_targets - completed))
    db.close()
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pilot', 'pmc'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--checkpoint-every',type=int,default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.mode, args.workers,args.checkpoint_every)))


if __name__ == '__main__':
    main()

"""Bounded lexical/Qwen/OpenAlex recall and OpenScholar ranking, lazy and shared."""
from __future__ import annotations

import calendar
import html
import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .checkpoints import Checkpoints
from .config import Config
from .progress import log
from .storage import MissingInput

_LOCAL_LOCK = threading.RLock()
_LOCAL: dict[str, Any] = {}
_REMOTE_LOCK = threading.Lock()
_OPENALEX_NEXT = 0.0
_CROSSREF_LOCK = threading.Lock()


def doi(value: Any) -> str:
    return re.sub(r'^https?://(dx\.)?doi.org/', '', str(value or '').lower()).strip()


def retrieval_needs_retry(result: dict[str, Any]) -> bool:
    failures = result.get('failures', [])
    if any('HTTP Error 400' in str(row.get('error', '')) and
           str(row.get('step', '')).startswith(('openalex_', 'citation_')) for row in failures):
        if any(re.search(r'[?*]', query) for query in result.get('queries', [])):
            return True
    # A denied or absent publisher page is not repaired by reranking unchanged sources.
    return any(not re.search(r'HTTP Error (?:400|401|403|404|405|410)\b', str(row.get('error', '')))
               for row in failures)


def eligible(source: dict[str, Any], target: dict[str, Any]) -> bool:
    if doi(source.get('doi')) and doi(source.get('doi')) == doi(target.get('doi')):
        return False
    target_doi = doi(target.get('doi'))
    front = re.sub(r'\s+', ' ', source.get('passage', '')[:6000]).casefold()
    if target_doi and any(target_doi in match[0] for match in re.finditer(
            r'(?:version of record|a version of this preprint was published|'
            r'published version (?:is available |at )|now published in).{0,500}', front)):
        return False
    title = lambda s: re.sub(r'\W+', '', str(s or '').casefold())
    if title(source.get('title')) and title(source.get('title')) == title(target.get('title')):
        return False
    published = source.get('publication_date')
    if not published and source.get('year'):
        published = f"{int(source['year']):04d}-12-31"
    try:
        return date.fromisoformat(str(published)[:10]) < date.fromisoformat(target['cutoff'])
    except (TypeError, ValueError):
        return False


def abstract(index: dict[str, list[int]] | None) -> str:
    positions = {pos: word for word, entries in (index or {}).items() for pos in entries}
    return ' '.join(positions[i] for i in sorted(positions))


class Retriever:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.network_calls = 0
        self.checkpoints: Checkpoints | None = None
        self.date_cache: dict[str, dict[str, Any] | None] = {}

    def step(self, name: str, action: Callable[[], Any]) -> Any:
        if self.checkpoints is None:
            return action()
        return self.checkpoints.run(name, action)

    def request(self, url: str, binary: bool = False) -> Any:
        global _OPENALEX_NEXT
        for attempt in range(min(2, self.config.network_retries)+1):
            if 'api.openalex.org' in url:
                with _REMOTE_LOCK:
                    wait = max(0.0, _OPENALEX_NEXT-time.monotonic())
                    if wait <= 60:
                        _OPENALEX_NEXT = max(time.monotonic(), _OPENALEX_NEXT)+1.0
                if wait > 60:
                    raise URLError(f'OpenAlex retry deferred for {wait:.0f}s; local retrieval remains available')
                if wait:
                    time.sleep(wait)
            self.network_calls += 1
            log(self.config, '网络请求', f'发起文献网络请求，当前请求尝试={attempt+1}。',
                **(self.checkpoints.context if self.checkpoints else {}), attempt=attempt+1)
            try:
                request = Request(url, headers={'User-Agent': 'ASPR scientific research'})
                with urlopen(request, timeout=self.config.network_timeout_seconds) as response:
                    raw = response.read(self.config.download_max_bytes+1)
                if len(raw) > self.config.download_max_bytes:
                    raise ValueError('Download exceeds 25 MB budget')
                return raw if binary else json.loads(raw)
            except (HTTPError, URLError, TimeoutError) as exc:
                transient = not isinstance(exc, HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
                if not transient or attempt == min(2, self.config.network_retries):
                    raise
                delay = 2**attempt
                if isinstance(exc, HTTPError) and exc.code == 429:
                    value = exc.headers.get('Retry-After', '')
                    delay = float(value) if value.replace('.', '', 1).isdigit() else 30*(attempt+1)
                    if 'api.openalex.org' in url:
                        with _REMOTE_LOCK:
                            _OPENALEX_NEXT = max(_OPENALEX_NEXT, time.monotonic()+delay)
                log(self.config, '网络重试', f'瞬态网络错误，第{attempt+1}/{min(2, self.config.network_retries)}次重试，服务要求等待{delay:g}秒。',
                    **(self.checkpoints.context if self.checkpoints else {}), error_type=type(exc).__name__)
                if delay > 60:
                    raise
                time.sleep(delay)
        raise RuntimeError('Request attempts exhausted')

    def openalex(self, query: str, cutoff: str, direct: bool = False) -> list[dict[str, Any]]:
        # Generated questions use punctuation literally, not OpenAlex wildcard syntax.
        search_text = re.sub(r'[?*]', ' ', query).strip()
        params: dict[str, Any] = {} if direct else {'search': search_text, 'per-page': self.config.openalex_candidates,
                                                   'filter': f'to_publication_date:{cutoff}'}
        if os.environ.get('OPENALEX_API_KEY'):
            params['api_key'] = os.environ['OPENALEX_API_KEY']
        base = 'https://api.openalex.org/works'
        if direct:
            base += '/'+quote(query, safe=':/')
        try:
            payload = self.request(base+'?'+urlencode(params))
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if direct and not doi(query).startswith('10.'):
                raise
            return self.crossref(query, cutoff, direct)
        works = [payload] if direct else payload['results']
        return [{'work_id': w['id'].rsplit('/', 1)[-1], 'title': w.get('title', ''),
            'doi': w.get('doi'), 'publication_date': w.get('publication_date'),
            'source_type': 'abstract' if w.get('abstract_inverted_index') else 'metadata_only',
            'passage': abstract(w.get('abstract_inverted_index')), 'url': w['id'],
            'pdf_url': (w.get('best_oa_location') or {}).get('pdf_url'), 'query': query}
            for w in works]

    def crossref(self, query: str, cutoff: str, direct: bool = False) -> list[dict[str, Any]]:
        params: dict[str, Any] = {} if direct else {
            'query.bibliographic': re.sub(r'[?*]', ' ', query).strip(),
            'rows': self.config.openalex_candidates}
        if cutoff and not direct:
            params['filter'] = 'until-pub-date:'+cutoff
        url = 'https://api.crossref.org/works'+('/'+quote(doi(query), safe='') if direct else '')
        # The public Crossref pool permits only one concurrent request.
        with _CROSSREF_LOCK:
            payload = self.request(url+('?' + urlencode(params) if params else ''))
            time.sleep(.25)
        works = [payload['message']] if direct else payload['message']['items']
        result = []
        for work in works:
            parts = next((work[k]['date-parts'][0] for k in ('published', 'published-online', 'published-print', 'issued')
                          if work.get(k, {}).get('date-parts')), [])
            published = None
            if parts:
                year, month = parts[0], parts[1] if len(parts) > 1 else 12
                day = parts[2] if len(parts) > 2 else calendar.monthrange(year, month)[1]
                published = f'{year:04d}-{month:02d}-{day:02d}'
            clean = lambda text: html.unescape(re.sub(r'<[^>]*>', ' ', text)).strip()
            passage = clean(work.get('abstract', ''))
            links = [v['URL'] for v in work.get('link', []) if v.get('content-type') == 'application/pdf' and v.get('URL')]
            ident = doi(work.get('DOI'))
            result.append({'work_id': 'CROSSREF:'+ident, 'doi': ident,
                'title': clean((work.get('title') or [''])[0]), 'publication_date': published,
                'source_type': 'abstract' if passage else 'metadata_only', 'passage': passage,
                'url': 'https://doi.org/'+ident, 'pdf_url': links[0] if links else None,
                'query': query, 'retrieval_provider': 'crossref',
                'retrieval_limitation': 'OpenAlex unavailable; publisher-deposited Crossref metadata used.'})
        return result

    def local(self) -> dict[str, Any]:
        key = str(self.config.graph_assets)
        with _LOCAL_LOCK:
            if key in _LOCAL:
                return _LOCAL[key]
            import pyarrow.parquet as pq
            root = self.config.graph_assets
            rows = pq.read_table(root/'abstracts.parquet').to_pylist()
            metadata = pq.read_table(root/'canonical_target_works.parquet').to_pylist()
            by_article = {r.get('nature_article_id'): r for r in metadata if r.get('nature_article_id')}
            corpus = []
            fts = sqlite3.connect(':memory:', check_same_thread=False)
            fts.execute('CREATE VIRTUAL TABLE docs USING fts5(title, passage)')
            for row in rows:
                meta = by_article.get(row['article_id'], {})
                corpus.append({'work_id': meta.get('work_id', row['article_id']), 'article_id': row['article_id'],
                    'title': row['title'], 'doi': row['doi'], 'year': row['year'],
                    'publication_date': str(meta.get('publication_date') or ''),
                    'source_type': 'abstract', 'passage': row['abstract_text'],
                    'local_fulltext': row['paper_markdown_path']})
                fts.execute('INSERT INTO docs VALUES (?, ?)', (row['title'], row['abstract_text']))
            state = {'corpus': corpus, 'fts': fts, 'by_work': {r['work_id']: r for r in corpus},
                     'by_article': {r['article_id']: r for r in corpus}, 'by_doi': {doi(r['doi']): r for r in corpus}}
            _LOCAL[key] = state
            return state

    def recall(self, query: str, target: dict[str, Any], lexical_only: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        local = self.local()
        with _LOCAL_LOCK:
            tokens = re.findall(r'\w+', query)
            expression = ' OR '.join('"'+t+'"' for t in tokens)
            hits = local['fts'].execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT 200',
                                        (expression,)).fetchall() if expression else []
            lexical = [local['corpus'][i-1] for (i,) in hits if eligible(local['corpus'][i-1], target)][:20]
            if lexical_only:
                return lexical[:self.config.lexical_candidates], []
            if 'encoder' not in local:
                import faiss
                import pyarrow.parquet as pq
                import torch
                from sentence_transformers import SentenceTransformer
                local['encoder'] = SentenceTransformer(str(self.config.embedding_model), device=self.config.device,
                    model_kwargs={'torch_dtype': torch.bfloat16} if self.config.device.startswith('cuda') else {})
                local['index'] = faiss.read_index(str(self.config.graph_assets/'claim_semantic_index.faiss'))
                local['claim_rows'] = pq.read_table(self.config.graph_assets/'claim_embedding_index.parquet').to_pylist()
                local['parents'] = {r['claim_id']: r['parent_paper_id'] for r in
                    pq.read_table(self.config.graph_assets/'claim_nodes.parquet', columns=['claim_id', 'parent_paper_id']).to_pylist()}
            vector = local['encoder'].encode([query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
            _, indices = local['index'].search(vector, min(2000, local['index'].ntotal))
            semantic, seen = [], set()
            for index in indices[0]:
                if index < 0:
                    continue
                claim = local['claim_rows'][int(index)]
                parent = local['parents'][claim['claim_id']]
                source = local['by_work'].get(parent) or local['by_article'].get(parent)
                if source and parent not in seen and eligible(source, target):
                    semantic.append(source)
                    seen.add(parent)
                if len(semantic) >= self.config.semantic_candidates:
                    break
            return lexical[:self.config.lexical_candidates], semantic

    def rank(self, sources: list[dict[str, Any]], target: dict[str, Any], queries: list[str], dual: bool) -> list[dict[str, Any]]:
        if not sources:
            return []
        local = self.local()
        with _LOCAL_LOCK:
            if 'reranker' not in local:
                from sentence_transformers import CrossEncoder
                local['reranker'] = CrossEncoder(str(self.config.reranker_model), device=self.config.device)
            views = [target['title']+'\n'+target['manuscript'][:6000]]
            if dual:
                views.append('Historical comparison: '+'; '.join(queries))
            chosen: dict[int, dict[str, Any]] = {}
            combined: dict[int, float] = {}
            for view_index, view in enumerate(views):
                scores = local['reranker'].predict([[view, (s.get('title') or '')+'\n'+(s.get('passage') or '')[:6000]]
                    for s in sources[:self.config.rerank_limit]], batch_size=8)
                for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True), 1):
                    chosen.setdefault(index, {**sources[index], 'ranking': {}})['ranking'][str(view_index)] = float(scores[index])
                    combined[index] = combined.get(index, 0.0)+1/(60+rank)
            # Both reading views must affect the prefix selected by the caller.
            # Concatenating their top tens silently discarded the query view for GEAR.
            return [chosen[index] for index in sorted(chosen, key=lambda i: combined[i], reverse=True)[:20]]

    def fulltext(self, source: dict[str, Any]) -> dict[str, Any]:
        source = dict(source)
        local = self.local()['by_doi'].get(doi(source.get('doi')), {})
        path = source.get('local_fulltext') or local.get('local_fulltext')
        if path and Path(path).is_file():
            text = Path(path).read_text(encoding='utf-8')
            pages = re.split(r'(?=<!-- GEAR_PAGE:)', text)
            text = ''.join(pages[:self.config.fulltext_max_pages])
            source.update(source_type='fulltext', passage=text[:self.config.fulltext_max_chars],
                fulltext_origin='local', original_path=str(path), char_range=[0, min(len(text), self.config.fulltext_max_chars)],
                pages_read=min(len(pages), self.config.fulltext_max_pages))
        elif self.config.download_fulltext:
            if not source.get('pdf_url') and source.get('doi'):
                if source.get('retrieval_provider') == 'crossref' and _OPENALEX_NEXT-time.monotonic() > 60:
                    # The Crossref search already supplied this work's complete link list.
                    # Repeating its DOI lookup during an OpenAlex cooldown adds no PDF location.
                    source['fulltext_status'] = 'crossref_has_no_pdf_location_openalex_deferred'
                    return source
                found = self.openalex('https://doi.org/'+doi(source['doi']), '', direct=True)
                source['pdf_url'] = found[0].get('pdf_url') if found else None
            if not source.get('pdf_url'):
                source['fulltext_status'] = 'no_public_pdf_location'
                return source
            import pymupdf as fitz
            content = self.request(source['pdf_url'], binary=True)
            if b'%PDF-' not in content[:1024]:
                source['fulltext_status'] = 'downloaded_response_is_not_pdf'
                return source
            with fitz.open(stream=content, filetype='pdf') as document:
                pages_read = min(len(document), self.config.fulltext_max_pages)
                text = '\n'.join(document[i].get_text() for i in range(pages_read))
            if len(text) < 1000 and re.search(r'JavaScript is disabled|enable JavaScript to proceed', text, re.I):
                source['fulltext_status'] = 'downloaded_document_is_browser_notice'
                return source
            source.update(source_type='fulltext', passage=text[:self.config.fulltext_max_chars],
                fulltext_origin='public', char_range=[0, min(len(text), self.config.fulltext_max_chars)],
                pages_read=pages_read, downloaded_bytes=len(content))
        return source

    def search(self, queries: list[str], target: dict[str, Any], prefix: str,
               citations: list[str] | None = None, method: str = 'eacl') -> dict[str, Any]:
        candidates, failures, ranked = [], [], []
        start_calls = self.network_calls
        for query_index, query in enumerate(queries[:self.config.max_queries]):
            try:
                lexical, semantic = self.step(f'{prefix}recall_{query_index:03d}', lambda query=query: self.recall(query, target))
                candidates.extend({**s, 'channel': channel, 'rank': i+1, 'query': query}
                    for channel, rows in [('lexical', lexical), ('semantic', semantic)] for i, s in enumerate(rows))
            except (InterruptedError, MissingInput):
                raise
            except (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error) as exc:
                failures.append({'channel': 'local_recall', 'query': query, 'error': str(exc)})
            try:
                candidates.extend({**s, 'channel': 'openalex', 'rank': i+1}
                                  for i, s in enumerate(self.step(f'{prefix}openalex_{query_index:03d}',
                                      lambda query=query: self.openalex(query, target['cutoff']))))
            except (InterruptedError, MissingInput):
                raise
            except (OSError, ValueError, KeyError) as exc:
                failures.append({'channel': 'openalex', 'query': query, 'error_type': type(exc).__name__})
        for citation_index, citation in enumerate(citations or []):
            try:
                direct = citation.startswith(('10.', 'https://doi.org/', 'https://openalex.org/'))
                rows = self.step(f'{prefix}citation_{citation_index:04d}',
                                 lambda citation=citation, direct=direct: self.openalex(citation, target['cutoff'], direct=direct))
                candidates.extend({**s, 'channel': 'citation', 'rank': i+1} for i, s in enumerate(rows[:1]))
            except (InterruptedError, MissingInput):
                raise
            except (OSError, ValueError, KeyError) as exc:
                failures.append({'channel': 'citation', 'citation': citation, 'error_type': type(exc).__name__})
        merged: dict[str, dict[str, Any]] = {}
        for source in candidates:
            if not eligible(source, target):
                continue
            key = doi(source.get('doi')) or source.get('work_id') or source.get('title')
            item = merged.setdefault(key, {**source, 'reciprocal_rank': 0.0})
            item['reciprocal_rank'] += 1/(60+source['rank'])
        pool = sorted(merged.values(), key=lambda s: s['reciprocal_rank'], reverse=True)[:self.config.candidate_limit]
        try:
            ranked = self.step(prefix+'ranking', lambda: self.rank(pool, target, queries, dual=method != 'reviewgrounder'))
        except (InterruptedError, MissingInput):
            raise
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            failures.append({'channel': 'ranking', 'error': str(exc)})
        sources = []
        for i, source in enumerate(ranked[:10 if method == 'reviewgrounder' or method.startswith('gear_') else 20]):
            source = {**source, 'source_id': f'{prefix}{i+1:04d}'}
            try:
                source = self.step(f'{prefix}fulltext_{i:03d}', lambda source=source: self.fulltext(source))
            except (InterruptedError, MissingInput):
                raise
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                failures.append({'channel': 'fulltext', 'source_id': source['source_id'], 'error_type': type(exc).__name__})
            if eligible(source, target):
                sources.append(source)
        return {'sources': sources, 'candidates': candidates, 'rank_pool': pool, 'failures': failures,
                'queries': queries[:4], 'network_calls': self.network_calls-start_calls,
                'fulltext_count': sum(s['source_type'] == 'fulltext' for s in sources)}

    def enrich_dates(self, sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        result, failures = [], []
        for source_index, original in enumerate(sources):
            source = dict(original)
            if not source.get('publication_date') and source.get('source_type') not in {'manuscript', 'review', 'graph_observation'}:
                if source.get('doi'):
                    try:
                        key = doi(source['doi'])
                        if key not in self.date_cache:
                            self.date_cache[key] = None
                            self.date_cache[key] = self.step(f'date_{source_index:05d}',
                                lambda key=key: self.openalex('https://doi.org/'+key, '', direct=True)[0])
                        found = self.date_cache[key]
                        if found is not None:
                            source['publication_date'] = found['publication_date']
                            source['date_origin'] = found['url']
                    except (InterruptedError, MissingInput):
                        raise
                    except (OSError, ValueError, KeyError) as exc:
                        failures.append({'source_id': source['source_id'], 'error_type': type(exc).__name__})
                if not source.get('publication_date') and source.get('year'):
                    source['publication_date'] = f"{int(source['year']):04d}-12-31"
                    source['date_origin'] = 'year_only_conservative_end'
            result.append(source)
        return result, failures

    async def search_async(self, engine: Any, target: dict[str, Any], queries: list[str],
                           citations: list[str], prefix: str, method: str, directory: Path) -> dict[str, Any]:
        from .storage import read, write
        failures: list[dict[str, Any]] = []

        async def saved(name: str, action: Callable[[], Any], gpu: bool = False) -> Any:
            path = directory/(name+'.json')
            if path.exists():
                return read(path)
            try:
                value = await engine.io(action, gpu=gpu, download=name.startswith('fulltext_'))
                write(path, value)
                return value
            except (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error) as exc:
                failures.append({'step': name, 'error': str(exc)})
                return []

        import asyncio
        async def local_query(index: int, query: str) -> Any:
            old = directory/f'local_{index}.json'
            if old.exists():
                return read(old)
            lexical = await saved(f'lexical_{index}', lambda: self.recall(query, target, True))
            semantic = await saved(f'semantic_{index}', lambda: self.recall(query, target), True)
            return [lexical[0] if lexical else [], semantic[1] if semantic else []]

        local_jobs = [local_query(i, q) for i, q in enumerate(queries[:self.config.max_queries])]
        remote_jobs = [saved(f'openalex_{i}', lambda q=q: self.openalex(q, target['cutoff'])) for i, q in enumerate(queries[:self.config.max_queries])]
        citations = list(dict.fromkeys(citations))
        citation_jobs = [saved(f'citation_{i}', lambda q=q: self.openalex(q, target['cutoff'],
                            q.startswith(('10.', 'https://doi.org/', 'https://openalex.org/')))) for i, q in enumerate(citations)]
        local, remote, cited = await asyncio.gather(asyncio.gather(*local_jobs), asyncio.gather(*remote_jobs), asyncio.gather(*citation_jobs))
        candidates = []
        for pair in local:
            for channel, rows in zip(('lexical', 'semantic'), pair):
                candidates.extend({**s, 'channel': channel, 'rank': i+1} for i, s in enumerate(rows))
        for rows in remote:
            candidates.extend({**s, 'channel': s.get('retrieval_provider', 'openalex'), 'rank': i+1} for i, s in enumerate(rows))
        for rows in cited:
            candidates.extend({**s, 'channel': 'citation', 'rank': 1} for s in rows[:1])
        merged: dict[str, dict[str, Any]] = {}
        for source in candidates:
            if eligible(source, target):
                key = doi(source.get('doi')) or source.get('work_id') or source['title']
                item = merged.setdefault(key, {**source, 'reciprocal_rank': 0.0})
                item['reciprocal_rank'] += 1/(60+source['rank'])
        pool = sorted(merged.values(), key=lambda s: s['reciprocal_rank'], reverse=True)[:self.config.candidate_limit]
        ranked = await saved('ranking', lambda: self.rank(pool, target, queries, method != 'reviewgrounder'), True)
        async def download(i: int, source: dict[str, Any]) -> dict[str, Any]:
            original = {**source, 'source_id': f'{prefix}{i+1:04d}'}
            key = doi(source.get('doi')) or source.get('work_id') or source.get('pdf_url') or prefix+str(i)
            if key not in engine.downloads:
                engine.downloads[key] = asyncio.create_task(saved(f'fulltext_{i}', lambda: self.fulltext(original)))
            value = await engine.downloads[key]
            if not isinstance(value, dict):
                return {**original, 'fulltext_status': 'failed'}
            fields = {k: value[k] for k in ('passage', 'source_type', 'fulltext_origin', 'original_path',
                      'char_range', 'pages_read', 'downloaded_bytes', 'fulltext_status') if k in value}
            result = {**original, **fields}
            write(directory/f'fulltext_{i}.json', result)
            return result
        sources = await asyncio.gather(*(download(i, s) for i, s in enumerate(ranked[:10 if method == 'reviewgrounder' or method.startswith('gear_') else 20])))
        sources = [source for source in sources if eligible(source, target)]
        return {'sources': sources, 'queries': queries[:self.config.max_queries], 'failures': failures, 'candidate_count': len(candidates),
                'network_calls': self.network_calls, 'fulltext_count': sum(s['source_type'] == 'fulltext' for s in sources)}

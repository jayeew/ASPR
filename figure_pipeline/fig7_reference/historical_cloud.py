"""Observed scientific phrase frequencies across all historical claim nodes."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .data import ASSETS, read, write
from .predict import csv_write
from .unified import BASE

STOP = set(ENGLISH_STOP_WORDS) | set('''study studies showed shows show demonstrated demonstrate
 demonstrates identified identify identifies developed develop develops using used use uses based
 associated increased increases increase reduced reduce reduces compared significant significantly high
 higher low lower new novel results result analysis analyses data approach approaches strategy strategies
 method methods observed including respectively approximately different experimental experiments
 conditions model models male female mice human humans vitro vivo ex ma cm van der s previously
 reported theoretical calculations calculation achieved wide range positively correlated versus mah
 g v patients patient'''.split())
SINGULAR = dict(zip(
    'cells networks factors structures simulations acids responses fields sites groups frameworks'.split(),
    'cell network factor structure simulation acid response field site group framework'.split()))
TOKEN = re.compile(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]')


def mentions(text: str) -> dict[str,str]:
    """Contiguous, source-preserving 3–6-token candidates; never bridge punctuation."""
    tokens = list(TOKEN.finditer(text.lower()))
    found = {}
    for length in range(3,7):
        for index in range(len(tokens)-length+1):
            span = tokens[index:index+length]
            terms = [t.group() for t in span]
            if any(t in STOP or not t[0].isalpha() for t in terms):
                continue
            phrase = ' '.join(SINGULAR.get(t,t) for t in terms)
            found[phrase] = text[span[0].start():span[-1].end()]
    return found


def categorize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vocabulary = read(Path(__file__).with_name('historical_topics.json'))
    by_phrase = {r['phrase']:r for r in vocabulary}
    for row in rows:
        topic = by_phrase[row['phrase']]
        row['category'] = topic['category']
        row['category_rationale'] = topic['category_rationale']
        row['color'] = BASE[row['category']]
    return rows


def prepare(destination: Path) -> list[dict[str, Any]]:
    from collections import Counter

    nodes = pd.read_parquet(ASSETS/'claim_nodes.parquet')
    years = pd.to_datetime(nodes.publication_date).dt.year
    assert years.isin([2023,2024,2025]).all() and nodes.claim_id.is_unique
    vocabulary = read(Path(__file__).with_name('historical_topics.json'))
    labels = {r['phrase']:r['display_label'] for r in vocabulary}
    assert len(labels)==len(vocabulary)
    papers: dict[str,set[str]] = defaultdict(set)
    claim_counts: dict[str,int] = defaultdict(int)
    candidates: Counter[str] = Counter()
    sources = []
    for paper_id,group in nodes.groupby('parent_paper_id',sort=False):
        paper_phrases = set()
        for node in group.itertuples():
            found = mentions(node.claim_text)
            paper_phrases.update(found)
            for topic in sorted(found.keys() & labels.keys()):
                papers[topic].add(paper_id)
                claim_counts[topic] += 1
                sources.append({'phrase':topic,'claim_id':node.claim_id,'paper_id':paper_id,
                                'matched_text':found[topic],'claim_text':node.claim_text})
        candidates.update(paper_phrases)
    assert set(papers)==set(labels), 'Reviewed topic lacks source matches'
    maximum = max(len(ids) for ids in papers.values())
    rows = []
    for phrase in sorted(labels,key=lambda p:(-len(papers[p]),p)):
        count = len(papers[phrase])
        weight = (count/maximum)**.65
        rows.append({'category':'Historical','phrase':phrase,'display_label':labels[phrase],
                     'frequency':count,'claim_count':claim_counts[phrase],
                     'source_paper_count':count,'source_paper_ids':sorted(papers[phrase]),
                     'prominence':weight,'font_size_pt':round(18+18*weight),
                     'bold':count>=maximum*.5,'color':'#124F93','displayed':True})
    rows = categorize(rows)
    candidate_rows = [{'phrase':p,'paper_count':v} for p,v in candidates.most_common() if v>=5]
    write(destination/'data/historical_topic_vocabulary.json',vocabulary)
    write(destination/'data/historical_wordcloud.json',rows)
    write(destination/'data/historical_summary.json',{
        'claims':len(nodes),'papers':nodes.parent_paper_id.nunique(),'years':[2023,2024,2025],
        'annual_claims':years.value_counts().sort_index().to_dict(),
        'displayed_terms':len(rows),'count_unit':'distinct papers explicitly mentioning a reviewed research topic in claims',
        'source':'data/claim_graph/claim_nodes.parquet','stop_words':sorted(STOP),
        'singular_normalization':SINGULAR,'candidate_min_papers':5,'candidate_count':len(candidate_rows),
        'selection':'Source-reviewed scientific topics with object, process, mechanism or qualified technical route; not the global top 68 n-grams.',
        'coverage_limit':'All nodes scanned; counts cover explicit normalized phrase mentions, not inferred semantic topic membership.'})
    csv_write(destination/'tables/historical_topic_candidates.csv',candidate_rows)
    csv_write(destination/'tables/historical_wordcloud.csv',rows)
    csv_write(destination/'tables/historical_phrase_sources.csv',sources)
    print(f"Historical topics: {len(rows)}; all {len(nodes)} claims scanned; {len(candidate_rows)} recurring candidates.")
    return rows

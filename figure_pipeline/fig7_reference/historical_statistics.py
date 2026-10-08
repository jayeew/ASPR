"""Corpus-ranked historical noun-phrase statistics; no hand-selected topic vocabulary."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .data import ASSETS, read, write
from .predict import csv_write
from .unified import BASE

TOKEN = re.compile(r'[^\W_]+(?:[-–][^\W_]+)*|[^\w\s]',re.UNICODE)
SINGULAR = dict(zip(
    'cells networks factors structures simulations acids responses fields sites groups frameworks'.split(),
    'cell network factor structure simulation acid response field site group framework'.split()))
PLURALS = {**SINGULAR, 'batteries':'battery','regions':'region','proteins':'protein',
           'membranes':'membrane','receptors':'receptor','reactions':'reaction','scores':'score',
           'assays':'assay','inhibitors':'inhibitor','waves':'wave','liquids':'liquid','algorithms':'algorithm'}
# Generic linguistic/measurement heads, independent of scientific field and observed rank.
GENERIC_HEADS = set('study result effect level number rate value increase decrease change measurement experiment analysis calculation data performance efficiency accuracy year time condition sample patient mouse participant group role evidence strategy approach scope concentration temperature density yield survival'.split())
GENERIC_MODIFIERS = set('new novel observed respective corresponding previous proposed reported'.split())
BAD_TOKENS = set('study result show showed demonstrate demonstrated identify identified reveal revealed suggest suggested using used including respectively approximately compared versus'.split())
MIN_PAPERS, MAX_TERMS = 5, 68


def words(text: str) -> list[re.Match[str]]:
    return list(TOKEN.finditer(text))


def canonical(text: str) -> str:
    return ' '.join(PLURALS.get(m.group().lower(),m.group().lower()) for m in words(text))


def candidates_from_doc(doc: Any) -> list[tuple[str,str]]:
    found = []
    for chunk in doc.noun_chunks:
        start = chunk.start
        while start<chunk.end and (doc[start].pos_ in {'DET','PRON'} or doc[start].lower_ in GENERIC_MODIFIERS):
            start += 1
        if start==chunk.end or chunk.root.lemma_.lower() in GENERIC_HEADS:
            continue
        span = doc[start:chunk.end]
        tokens = words(span.text)
        if not 3<=len(tokens)<=6:
            continue
        if any(not t.group()[0].isalpha() or t.group().lower() in BAD_TOKENS for t in tokens):
            continue
        # A bare surname construction lacks a scientific head (e.g. "Van der Waals").
        if tokens[-2].group().lower() in {'der','van','von','de','del'}:
            continue
        # Require a complete nominal expression without a clausal verb/conjunction.
        if any(t.pos_ in {'VERB','AUX','CCONJ','SCONJ','PRON'} for t in span):
            continue
        found.append((canonical(span.text),span.text))
    return found


def extract(nodes: pd.DataFrame, destination: Path) -> dict[str,str]:
    import spacy
    nlp = spacy.load('en_core_web_sm',disable=['ner'])
    surface: dict[str,Counter[str]] = defaultdict(Counter)
    for index,doc in enumerate(nlp.pipe(nodes.claim_text.tolist(),batch_size=128,n_process=2),1):
        for phrase,label in set(candidates_from_doc(doc)):
            surface[phrase][label] += 1
        if index%5000==0:
            print(f'Parsed {index}/{len(nodes)} historical claims',flush=True)
    assert index==len(nodes)
    labels = {p:sorted(c,key=lambda s:(-c[s],s))[0] for p,c in surface.items()}
    write(destination/'data/historical_candidate_vocabulary.json',labels)
    write(destination/'data/historical_parser.json',{'spacy':spacy.__version__,
          'model':'en_core_web_sm','model_version':nlp.meta['version'],'parsed_claims':len(nodes)})
    return labels


def matching_mentions(text: str, vocabulary: set[str]) -> dict[str,str]:
    tokens = words(text)
    normalized = [PLURALS.get(t.group().lower(),t.group().lower()) for t in tokens]
    found = {}
    for size in range(3,7):
        for index in range(len(tokens)-size+1):
            phrase = ' '.join(normalized[index:index+size])
            if phrase in vocabulary:
                found[phrase] = text[tokens[index].start():tokens[index+size-1].end()]
    return found


def category_for(phrase: str) -> tuple[str,str]:
    if (re.search(r'\b(tandem|hybrid|composite|bilayer|heterostructure|multimodal)\b',phrase)
        or 'coupled' in phrase or 'framework membrane' in phrase or phrase=='car t cell'
        or 'perovskite solar cell' in phrase):
        return 'Combination','Explicit coupled process, layered/hybrid system or integrated architecture.'
    if re.search(r'\b(simulation|theory|microscopy|sequencing|spectroscopy|network|learning|imaging|assay|editing|editor|blockade|deposition|scanning|cytometry|diffraction|model|stimulation|spectrometry)\b',phrase):
        return 'Method','Explicit analytical, computational, measurement or intervention route.'
    return 'Problem','Research object, process or phenomenon; no explicit method/combination cue.'


def rank_rows(papers: dict[str,set[str]], counts: dict[str,int], labels: dict[str,str]) -> list[dict[str,Any]]:
    ordered = sorted((p for p,ids in papers.items() if len(ids)>=MIN_PAPERS),key=lambda p:(-len(papers[p]),p))
    fuller = {p:sorted((q for q in ordered if p!=q and ' '+p+' ' in ' '+q+' '
                       and len(papers[q])>=.8*len(papers[p])),key=lambda q:(-len(papers[q]),q))
              for p in ordered}
    maximum = len(papers[ordered[0]])
    rows = []
    display_rank = 0
    for rank,phrase in enumerate(ordered,1):
        if not fuller[phrase]:
            display_rank += 1
        count = len(papers[phrase])
        weight = (count/maximum)**.65
        category,reason = category_for(phrase)
        label = labels[phrase]
        label = label[0].upper()+label[1:]
        rows.append({'rank':rank,'display_rank':display_rank if not fuller[phrase] else None,
            'suppressed_by_fuller_phrase':fuller[phrase],
            'category':category,'category_rationale':reason,
            'phrase':phrase,'display_label':label,'frequency':count,'claim_count':counts[phrase],
            'source_paper_count':count,'source_paper_ids':sorted(papers[phrase]),'prominence':weight,
            'font_size_pt':round(18+18*weight),'bold':count>=maximum*.5,'color':BASE[category],
            'displayed':not fuller[phrase] and display_rank<=MAX_TERMS})
    return rows


def prepare(destination: Path, reuse_parsed: bool = False) -> list[dict[str,Any]]:
    nodes = pd.read_parquet(ASSETS/'claim_nodes.parquet')
    years = pd.to_datetime(nodes.publication_date).dt.year
    assert nodes.claim_id.is_unique and years.isin([2023,2024,2025]).all()
    labels = (read(destination/'data/historical_candidate_vocabulary.json') if reuse_parsed
              else extract(nodes,destination))
    generic_endings = GENERIC_HEADS | {h+'s' for h in GENERIC_HEADS} | {'datum'}
    labels = {p:v for p,v in labels.items() if p.split()[-1] not in generic_endings}
    vocabulary = set(labels)
    papers: dict[str,set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    sources = []
    for node in nodes.itertuples():
        for phrase,span in matching_mentions(node.claim_text,vocabulary).items():
            papers[phrase].add(node.parent_paper_id)
            counts[phrase] += 1
            sources.append({'phrase':phrase,'claim_id':node.claim_id,'paper_id':node.parent_paper_id,
                            'matched_text':span,'claim_text':node.claim_text})
    rows = rank_rows(papers,counts,labels)
    eligible = {r['phrase'] for r in rows}
    write(destination/'data/historical_wordcloud.json',rows)
    write(destination/'data/historical_topic_vocabulary.json',[{'phrase':r['phrase'],
          'display_label':r['display_label'],'category':r['category'],'category_rationale':r['category_rationale']}
          for r in rows if r['displayed']])
    write(destination/'data/historical_summary.json',{'claims':len(nodes),'papers':nodes.parent_paper_id.nunique(),
          'years':[2023,2024,2025],'source':'data/claim_graph/claim_nodes.parquet',
          'source_sha256':hashlib.sha256((ASSETS/'claim_nodes.parquet').read_bytes()).hexdigest(),
          'candidate_count':len(labels),'eligible_count':len(rows),'displayed_terms':sum(r['displayed'] for r in rows),
          'minimum_papers':MIN_PAPERS,'token_length':[3,6],'generic_heads':sorted(GENERIC_HEADS),
          'incomplete_name_head_particles':['de','del','der','van','von'],
          'generic_leading_modifiers':sorted(GENERIC_MODIFIERS),'excluded_tokens':sorted(BAD_TOKENS),
          'plural_normalization':PLURALS,'selection':'Global distinct-paper frequency descending, canonical phrase ascending for ties; suppress a contained shorter phrase when a fuller candidate reaches at least 80% of its paper count. Display the top 68 remaining; no manual whitelist, category quotas or frequency weighting by category.',
          'count_unit':'Distinct papers with an explicit normalized phrase occurrence in historical claims.',
          'coverage_limit':'All claims scanned; automatic noun-phrase statistics, not exhaustive semantic topic classification.'})
    csv_write(destination/'tables/historical_wordcloud.csv',rows)
    csv_write(destination/'tables/historical_topic_candidates.csv',rows)
    csv_write(destination/'tables/historical_phrase_sources.csv',[r for r in sources if r['phrase'] in eligible])
    print('Historical ranking:',[(r['display_label'],r['frequency']) for r in rows[:10]],flush=True)
    return rows

"""Independently check historical paper counts, source spans and global ranking."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import json,re
import pandas as pd
from figure_pipeline.fig7_reference.historical_statistics import PLURALS
from figure_pipeline.fig7_reference.data import ROOT, write


def audit(destination: Path) -> dict[str, object]:
    root = destination
    rows=json.loads((root/'data/historical_wordcloud.json').read_text())
    summary=json.loads((root/'data/historical_summary.json').read_text())
    parser=json.loads((root/'data/historical_parser.json').read_text())
    nodes=pd.read_parquet('data/claim_graph/claim_nodes.parquet')
    sources=pd.read_csv(root/'tables/historical_phrase_sources.csv')
    assert len(nodes)==summary['claims']==parser['parsed_claims']==70034
    assert nodes.parent_paper_id.nunique()==summary['papers']==24914
    assert not sources[['phrase','claim_id']].duplicated().any()
    reference=nodes.set_index('claim_id')
    assert (sources.claim_text==sources.claim_id.map(reference.claim_text)).all()
    assert (sources.paper_id==sources.claim_id.map(reference.parent_paper_id)).all()
    assert all(x.matched_text in x.claim_text for x in sources.itertuples())
    counts=sources.groupby('phrase').agg(papers=('paper_id','nunique'),claims=('claim_id','count'))
    for row in rows:
     assert row['frequency']==counts.loc[row['phrase'],'papers']
     assert row['claim_count']==counts.loc[row['phrase'],'claims']
     assert row['frequency']==len(set(row['source_paper_ids']))
    assert rows==sorted(rows,key=lambda r:(-r['frequency'],r['phrase']))
    shown=[r for r in rows if r['displayed']]
    assert shown==[r for r in rows if not r['suppressed_by_fuller_phrase']][:68]
    for row in shown:
     parts=[]
     for token in row['phrase'].split():
      variants={token}|{k for k,v in PLURALS.items() if v==token}
      parts.append('(?:'+'|'.join(re.escape(v) for v in sorted(variants))+')')
     pattern=r'(?<!\w)(?<!\w-)(?<!\w–)'+r'\s+'.join(parts)+r'(?!\w|[-–]\w)'
     count=nodes.loc[nodes.claim_text.str.contains(pattern,case=False,regex=True),'parent_paper_id'].nunique()
     assert count==row['frequency'],(row['phrase'],count,row['frequency'])
    result={'claims_scanned':len(nodes),'papers':nodes.parent_paper_id.nunique(),
     'eligible_candidates':len(rows),'displayed':len(shown),'all_eligible_source_counts_verified':True,
     'all_68_displayed_counts_independently_recomputed_by_regex':True,'global_top_68_verified':True,
     'historical_categories':dict(Counter(r['category'] for r in shown)),
     'top_five':[{k:r[k] for k in ('display_label','frequency','rank')} for r in shown[:5]],
     'battery_related_candidates':[{k:r[k] for k in ('display_label','frequency','rank','displayed')} for r in rows if 'batter' in r['phrase']],
     'scope':'Automatic complete noun-phrase statistics under saved rules; not validated semantic topic coverage.'}
    write(root/'qa/historical_statistical_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == "__main__":
    audit(ROOT/"outputs/fig7_reference/two_panel")

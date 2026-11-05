"""Prepare Fig8 citation percentiles and institutional topic breadth from the fixed sample."""
from concurrent.futures import ProcessPoolExecutor
import gzip
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

BASE = Path('outputs/fig8_data')
TABLES = BASE / 'analysis/tables'


def read_counts(path: Path) -> list:
    values = []
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            work = orjson.loads(line)['work']
            count = work.get('cited_by_count')
            if count is not None:
                values.append((work['id'], count))
    return values


def main() -> None:
    sample = pd.read_parquet(BASE/'analysis/cohort/selection.parquet', columns=['included','work_id','source_id','field','online_year','paper_shard']).query('included').copy()
    files = sorted((BASE/'supplement/snapshot_location_recovery/files').glob('*.jsonl.gz'))
    with ProcessPoolExecutor(8) as pool:
        counts = dict(pair for chunk in pool.map(read_counts, files, chunksize=8) for pair in chunk)
    sample['citations'] = sample.work_id.map(counts)
    groups = ['field','online_year']
    coverage = sample.groupby(groups).citations.agg(['size','count'])
    print(coverage, flush=True)
    coverage.to_csv(TABLES/'citation_coverage.csv')
    score = sample.citations.copy()
    for _, group in sample.dropna(subset=['citations']).groupby(groups):
        v = group.citations
        cutoff = v.quantile(.9, interpolation='higher')
        weight = (.1*len(v)-(v>cutoff).sum())/(v==cutoff).sum()
        score.loc[group.index] = np.where(v>cutoff,1,np.where(v==cutoff,weight,0))
    sample['credit'] = score
    topics = pd.concat([pd.read_parquet(BASE/'fields/papers'/f,columns=['work_id','topic_id']) for f in sample.paper_shard.unique()]).drop_duplicates('work_id')
    sample = sample.merge(topics,on='work_id',how='left')
    sample[['work_id','citations','credit','topic_id']].to_parquet(TABLES/'impact_inputs.parquet',index=False)
    def aggregate(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
        stats = frame.groupby(keys).credit.agg(rate='mean',known='count',total='size')
        stats.loc[stats.known/stats.total < .9,'rate'] = np.nan
        return stats
    aggregate(sample,['source_id','field']).to_csv(TABLES/'journal_impact.csv')
    inst = pd.read_parquet(TABLES/'paper_institutions.parquet',columns=['work_id','institution_id']).merge(sample[['work_id','credit','topic_id']],on='work_id')
    stats = aggregate(inst,['institution_id'])
    sizes = inst.groupby(['institution_id','topic_id']).size()
    probs = sizes/sizes.groupby(level=0).transform('sum')
    stats['effective_topics'] = np.exp(-(probs*np.log(probs)).groupby(level=0).sum())
    stats.to_csv(TABLES/'institution_impact.csv',index_label='entity_id')


if __name__ == '__main__':
    main()

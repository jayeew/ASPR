"""Isolated RFdiffusion2 case using existing scientific runtime stages."""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup

from experiments.innovation_200.common import configure_limits, experiment_config
from experiments.innovation_200.reporting import generate_report, report_markdown
from gear.artifacts import read_model, write_model
from gear.innovation.graph_batch import analyze_task, finalize_graph, model_tasks, prepare_graph
from gear.innovation.pipeline import run_branch
from gear.innovation.shared import prepare_shared
from gear.innovation.usage import progress_logging
from gear.review_contracts import InnovationPaperInput

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'outputs/fig6_rf2_study'
PID = 's41592-025-02975-x'
CASE = STUDY / 'papers' / PID


def setup() -> InnovationPaperInput:
    soup = BeautifulSoup((STUDY/'source/article.html').read_text(), 'html.parser')
    article = soup.find('article')
    sections = article.select('div.c-article-section')
    chunks = []
    for sec in sections:
        head = sec.find(['h2','h3'])
        if head and head.get_text(strip=True) in ['References','Acknowledgements','Author information','Ethics declarations','Additional information','Rights and permissions','About this article']: continue
        chunks.append(sec.get_text('\n',strip=True))
    abstract = soup.select_one('#Abs1-content').get_text(' ',strip=True)
    path=STUDY/'source/manuscript.md'
    path.write_text('# Atom-level enzyme active site scaffolding using RFdiffusion2\n\n'+ '\n\n'.join(chunks))
    item=InnovationPaperInput(paper_id=PID,paper_path=path,title='Atom-level enzyme active site scaffolding using RFdiffusion2',doi='10.1038/'+PID,venue='Nature Methods',publication_date='2025-12-03',cutoff_date='2025-04-09',abstract_text=abstract,abstract_source='Publisher HTML',authors=['Woody Ahern','Jason Yim','David Baker'])
    write_model(STUDY/'input.json',item)
    (STUDY/'temporal_scope.json').write_text(json.dumps({'target_version':'Published article retrieved 2026-10-03','online_publication':'2025-12-03','preprint_first_public':'2025-04-10','preprint_doi':'10.1101/2025.04.09.648075','cutoff':'2025-04-09','interpretation':'Retrospective analysis of published claims against a conservative pre-preprint historical set. Not a submission-time evaluation; later target additions are not assigned an earliest-disclosure claim. Self versions must not count as independent prior art.'},indent=2))
    return item


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['shared','graph','gear','reports']);args=parser.parse_args()
    STUDY.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(STUDY/f'{args.stage}.log')])
    logger=logging.getLogger('rf2')
    configure_limits(4,2)
    config=experiment_config()
    item=read_model(STUDY/'input.json',InnovationPaperInput) if (STUDY/'input.json').exists() else setup()
    with progress_logging(logger,'[RFdiffusion2]'):
        paper,shared=prepare_shared(item,CASE,config)
        if args.stage=='shared': return
        if args.stage=='graph':
            failures=prepare_graph([(item,CASE)],config,ROOT/'data/claim_graph',ROOT/'data/models/Qwen3-Embedding-4B',8)
            if failures: raise RuntimeError(str(failures))
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda task: analyze_task(task,config), model_tasks([(item,CASE)])))
            finalize_graph(item,CASE,config)
        elif args.stage=='gear':
            run_branch(item,CASE,config,paper,shared,'gear',ROOT/'data/claim_graph',ROOT/'data/models/Qwen3-Embedding-4B')
        else:
            for system in ['fusion','fusion_no_joint']:
                result=generate_report(PID,system,CASE)
                dest=STUDY/'reports'/system/f'{PID}.json';dest.parent.mkdir(parents=True,exist_ok=True)
                write_model(dest,result);dest.with_suffix('.md').write_text(report_markdown(result))


if __name__=='__main__':main()

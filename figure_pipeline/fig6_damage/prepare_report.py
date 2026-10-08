"""Regenerate a complete report from preserved DAMAGE branch artifacts."""
from __future__ import annotations

import json
import logging
import re
import experiments.innovation_200.reporting as reporting
import shutil
from pathlib import Path

from experiments.innovation_200.common import configure_limits
from experiments.innovation_200.reporting import generate_report, report_markdown
from gear.artifacts import write_model
from gear.innovation.usage import progress_logging

ROOT=Path(__file__).resolve().parents[2]
PID='s41467-026-69179-5'
OLD=ROOT/'outputs/innovation_200_20260907'
NEW=ROOT/'outputs/fig6_damage_study'


def main() -> None:
    newcase=NEW/'papers'/PID
    if not newcase.exists():shutil.copytree(OLD/'papers'/PID,newcase)
    archived=NEW/'original_report';archived.mkdir(parents=True,exist_ok=True)
    for suffix in ['json','md']:shutil.copy2(OLD/f'reports/fusion/{PID}.{suffix}',archived/f'{PID}.{suffix}')
    NEW.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(NEW/'report.log')])
    configure_limits(2,2)
    reporting.REPORT_PROMPT += '\n正文（含引用标记）控制在2500—3200字以内，确保所有段落和句子完整结束；合并重复信息。'
    with progress_logging(logging.getLogger('damage'),'[DAMAGE report]'):
        result=generate_report(PID,'fusion',newcase)
    if not re.sub(r'\[[^\]\n]+\]', '', result.body).rstrip().endswith(('。','！','？','.')):raise ValueError('Report ends mid-sentence')
    path=NEW/f'reports/fusion/{PID}.json';path.parent.mkdir(parents=True,exist_ok=True)
    write_model(path,result);path.with_suffix('.md').write_text(report_markdown(result))
    (NEW/'provenance.json').write_text(json.dumps({'branches':'Unchanged copies from innovation_200_20260907','report':'New report from existing branches; separate generation, not original uninterrupted run','reason':'Original saved report terminates mid-sentence at exactly 4000 characters','reviewer_material_in_writer_input':False,'generation_change':'Added neutral length instruction: 2500–3200 characters including aliases, complete sentences; no scientific stance instruction'},indent=2))


if __name__=='__main__':main()

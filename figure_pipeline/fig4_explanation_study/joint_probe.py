from __future__ import annotations

import asyncio

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_rerun.config import Report

from .evaluation import PROMPT as EVALUATION_PROMPT, availability, segments
from .generation import report_material
from .models import Config, Judgments
from .runtime import Runner

PAPER = 's41467-026-68290-x'
QUESTIONS = [
    '针对图中系统架构贡献CLAIM::01与像素吞吐贡献CLAIM::06，逐一比较可见历史邻居身份：是否有共享的历史claim、来自哪篇父论文？解释共同历史内容与这两个目标贡献的关系，并说明不能由共享身份推出的科学结论。',
    '继续仅针对CLAIM::01与CLAIM::06：可见材料是否提供跨越这两个历史邻域的既有历史边？如有，明确给出两端节点，解释科学联系与原文核验范围；如未提供，区分缺少记录与不存在。不要补画边，不要把目标插入路径当作既有历史边。',
]
PROMPT = '''Answer the two supplied focused questions in Chinese using only the actual input materials.
Do not regenerate a whole-paper report. Cite actual graph identities and original passages, distinguish
recorded facts from reconstruction, and state genuine missingness. No reference answers are supplied.
Do not infer support, causation or priority from graph edges. This is a separate diagnostic of a specific
unanswered comparison; do not compare experimental condition names or optimize for any ranking.
Return the answer body and actual cited_source_ids.'''


async def run_probe(config: Config) -> None:
    runner = Runner(config)

    async def answer(condition: str) -> dict:
        material = report_material(config, PAPER, condition)
        material['focused_questions'] = QUESTIONS
        return await runner.call(PAPER, 'joint_probe_reports', condition, PROMPT, material, Report,
                                 model='gpt-6.1-sol', effort='xhigh')

    conditions = ['F', 'F_noJ']
    answers = await asyncio.gather(*(answer(c) for c in conditions))
    original = read(config.source / 'inputs/evaluation' / f'{PAPER}.json')
    reference = read(config.output / 'reference/papers' / f'{PAPER}.json')
    candidates = [{'report_id': f'R{i:02d}', 'report_segments': segments(result['body']),
                   'material_availability': availability(config, PAPER, condition)}
                  for i, (condition, result) in enumerate(zip(conditions, answers), 1)]
    payload = {k: original[k] for k in ('manuscript', 'evidence_blocks', 'native_graph')}
    payload.update(reference_questions=[q for q in reference['questions'] if q['aspect'] == 'joint_contribution'],
                   candidates=candidates)
    result = await runner.call(PAPER, 'joint_probe_evaluation', 'joint_contribution', EVALUATION_PROMPT,
                               payload, Judgments)
    for candidate in result['candidates']:
        index = {s['segment_id']: s['text'] for c in candidates if c['report_id'] == candidate['report_id']
                 for s in c['report_segments']}
        for question in candidate['questions']:
            for part in question['parts']:
                part['report_quotes'] = [index[s] for s in part['report_segment_ids'] if s in index]
                part['technical_state'] = 'completed' if all(s in index for s in part['report_segment_ids']) else 'invalid_segment_id'
    write(config.output / 'joint_probe.json', {'paper_id': PAPER, 'questions_asked': QUESTIONS,
          'mapping': {'R01': 'F', 'R02': 'F_noJ'}, 'answers': dict(zip(conditions, answers)), 'evaluation': result,
          'scope': 'one-case two-condition focused-task diagnostic, separate from main seven-condition reports',
          'reason': 'All seven new reports omitted the preselected joint comparison; test availability versus task focus.',
          'included_in_main_metrics': False})

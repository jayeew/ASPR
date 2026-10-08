from __future__ import annotations

import csv
import random
import zipfile

from figure_pipeline.fig3_revision.storage import read, write

from .diagnostic import CONDITIONS, OLD
from .diagnostic import OUT as DIAGNOSTIC

OUT = DIAGNOSTIC.parent / 'human_review'


def main() -> None:
    papers = read(OLD / 'cohort.json')['repeated']
    blinded = OUT / 'blinded_materials'
    rows, keys = [], []
    for paper in papers:
        material = read(OLD / 'report_inputs/LUNA' / f'{paper}.json')
        visible = {k: v for k, v in material.items()
                   if k not in ('scientific_evidence_analysis', 'knowledge_graph_analysis')}
        write(blinded / paper / 'evidence_and_public_questions.json', visible)
        conditions = list(CONDITIONS)
        random.Random(f'20261004:{paper}:human').shuffle(conditions)
        for i, condition in enumerate(conditions, 1):
            label = f'report_{i}'
            text = read(DIAGNOSTIC / 'reports' / condition / f'{paper}.json')['body']
            (blinded / paper / f'{label}.md').write_text(text)
            keys.append({'paper_id': paper, 'anonymous_report': label, 'condition': condition})
            for question in material['public_tasks']['questions']:
                rows.append({'paper_id': paper, 'anonymous_report': label,
                             'question_id': question['question_id'], 'public_question': question['question'],
                             'reviewer_code': '', 'review_date': '', 'answer_completeness': '',
                             'content_correctness': '', 'scope_appropriate': '', 'evidence_sufficient': '',
                             'unsupported_assertion_present': '', 'judgment_explicitly_withheld': '',
                             'withholding_justified': '', 'report_quote': '', 'evidence_location': '',
                             'reason': '', 'reviewer_uncertainty': ''})
    with (blinded / 'review_form.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (blinded / '审阅说明.txt').write_text('''此包尚无人工评审结果；所有评审栏均为空。
请先阅读公开问题与证据，再独立审阅匿名报告。不要寻找预设的配置排名。
完整性填写 complete / partial / omitted / withheld / unresolved。
正确性填写 correct / partly_correct / incorrect / unresolved；其余判断填写 yes / no / unresolved。
保留判断必须针对所问内容，泛泛的局限说明不等于弃答。遗漏不等于事实错误。
引用报告原句及对应原始证据的位置，并说明判断理由。领域知识不足时请标明未决，不必猜测。
历史图中的生成文本不等于原始文献验证；论文层级引用路径不证明主张间因果或支持。
只使用本包实际证据评价材料内可支持性；如另外查阅文献，请明确标记其出处及外部证据身份。
模型身份映射和机器评分均未放入此盲审包。请填写实际评审日期与匿名评审者编号。
''', encoding='utf-8')
    write(OUT / 'unblinding_key_keep_separate.json', keys)
    with zipfile.ZipFile(OUT / 'Fig5_blinded_human_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in blinded.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(blinded))
    write(OUT / 'status.json', {'papers': len(papers), 'anonymous_reports': len(keys),
                               'question_review_rows': len(rows), 'human_reviews_received': 0,
                               'machine_scores_in_blinded_package': False})
    print(f'Prepared {len(keys)} anonymous reports; no human judgments fabricated.')


if __name__ == '__main__':
    main()

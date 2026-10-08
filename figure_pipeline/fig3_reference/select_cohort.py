"""Copy a 100-paper cohort without recorded model refusals; run no models."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import re
import shutil
import subprocess

from .settings import METHODS, OUTPUT, ROOT, STUDY, read, roster

DESTINATION = ROOT / 'outputs/fig3_reference/dataset'
PATTERN = r'limited access to this content|model_content_rejection|content_policy_violation|content_filter|blocked.{0,35}(safety|policy)|safety.{0,20}(blocked|refusal)'


def rejection_records() -> dict[str, list[str]]:
    """Use error metadata and task prefixes, never cited-paper IDs in error bodies."""
    ids = {p['paper_id'] for p in roster()}
    found: dict[str, list[str]] = defaultdict(list)
    result = subprocess.run(['rg', '-l', '-i', PATTERN, str(STUDY), str(OUTPUT/'data'), '--glob', '*.json', '--glob', '*.jsonl', '--glob', '*.log'], capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    for filename in result.stdout.splitlines():
        path = Path(filename)
        path_ids = [i for i in ids if i in str(path)]
        if path_ids:
            for ident in path_ids:
                found[ident].append(str(path.relative_to(ROOT)))
            continue
        if path.suffix == '.json':
            value = read(path)
            for error in value.get('errors', []) if isinstance(value, dict) else []:
                if re.search(PATTERN, str(error), re.I):
                    task = str(error.get('task', error.get('paper_id', '')))
                    for ident in ids:
                        if ident in task:
                            found[ident].append(str(path.relative_to(ROOT)))
        else:
            for line in path.read_text(errors='replace').splitlines():
                if re.search(PATTERN, line, re.I):
                    prefix = re.split(r'[\"\x27]error[\"\x27]\s*:', line, maxsplit=1)[0]
                    for ident in ids:
                        if ident in prefix:
                            found[ident].append(str(path.relative_to(ROOT)))
    return {i: sorted(set(paths)) for i, paths in found.items()}


def select(papers: list[dict], excluded: dict) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        if p['paper_id'] not in excluded:
            groups[p.get('field_name') or 'Unknown'].append(p)
    total = sum(map(len, groups.values()))
    if total < 100:
        raise ValueError(f'Only {total} eligible papers')
    quotas = {f: len(rows)*100//total for f, rows in groups.items()}
    ranked = sorted(groups, key=lambda f: (-(len(groups[f])*100 % total), f))
    for f in ranked[:100-sum(quotas.values())]:
        quotas[f] += 1
    rng = random.Random(20260922)
    selected = set()
    for field in sorted(groups):
        rows = sorted(groups[field], key=lambda p: p['paper_id'])
        rng.shuffle(rows)
        selected.update(p['paper_id'] for p in rows[:quotas[field]])
    return [p for p in papers if p['paper_id'] in selected]


def copy_existing(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    papers = roster()
    excluded = rejection_records()
    selected = select(papers, excluded)
    if DESTINATION.exists():
        raise FileExistsError(f'Dataset already exists: {DESTINATION}')
    DESTINATION.mkdir(parents=True)
    selected_ids = {p['paper_id'] for p in selected}
    missing = []
    copied_rows = []
    for p in selected:
        ident = p['paper_id']
        copy_existing(STUDY/'papers'/ident, DESTINATION/'study/papers'/ident)
        for folder in ('human_refs', 'reviewer_consistency'):
            copy_existing(STUDY/folder/f'{ident}.json', DESTINATION/'study'/folder/f'{ident}.json')
        for folder in ('reports', 'human_evaluation', 'evaluations'):
            for src in (STUDY/folder).rglob(f'{ident}.json'):
                copy_existing(src, DESTINATION/'study'/src.relative_to(STUDY))
        row = dict(p)
        for key, folder in (('paper_path', 'manuscripts'), ('review_path', 'peer_reviews')):
            src = Path(p[key]); dst = DESTINATION/'sources'/folder/src.name
            copy_existing(src, dst)
            row[key] = str(dst)
            if not dst.exists():
                missing.append((ident, 'source', key))
        copied_rows.append(row)
        paths = [('inputs', Path('inputs')/f'{ident}.json'), ('reviews', Path('reviews')/ident/'linked.json'), ('clusters', Path('insight_clusters')/f'{ident}.json')]
        for stage in ('reports', 'units', 'quality', 'support'):
            paths.extend((stage, Path(stage)/m/f'{ident}.json') for m in METHODS)
        paths.extend(('preference', Path('preferences')/m/f'{ident}_{o}.json') for m in METHODS[:-1] for o in ('AB', 'BA'))
        for stage, rel in paths:
            src = OUTPUT/'data'/rel
            copy_existing(src, DESTINATION/'data'/rel)
            if not src.exists():
                missing.append((ident, stage, str(rel)))
        copy_existing(OUTPUT/'data/reviews'/ident, DESTINATION/'data/reviews'/ident)
        for src in (OUTPUT/'data/reports').glob(f'*/{ident}.md'):
            copy_existing(src, DESTINATION/'data'/src.relative_to(OUTPUT/'data'))
    (DESTINATION/'papers.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in copied_rows))
    with (DESTINATION/'selection.csv').open('w') as file:
        writer = csv.writer(file); writer.writerow(['paper_id', 'field', 'selected', 'recorded_refusal', 'record_paths'])
        for p in papers:
            i = p['paper_id']; writer.writerow([i, p.get('field_name'), i in selected_ids, i in excluded, ';'.join(excluded.get(i, []))])
    with (DESTINATION/'missing.csv').open('w') as file:
        writer = csv.writer(file); writer.writerow(['paper_id', 'stage', 'missing_file']); writer.writerows(missing)
    note = f'''# 新版 Fig.3 的 100 篇评测子集\n\n从原 200 篇中排除 {len(excluded)} 篇有明确模型拦截记录的论文，其余按 field_name 比例分层，以最大余数分配名额，随机种子 20260922，抽取 100 篇。不以评分、偏好、聚类完成度或普通技术失败筛选。\n\n搜索范围覆盖原研究目录与 Fig.3 数据目录中的 JSON、JSONL、日志，包含逐 claim 的 gear_attempts 和 evidence_trace。任一 claim 或任一阶段有明确拦截即排除整篇。结论仅为现有留存记录中未发现模型安全拦截，不保证遗失日志或未来运行没有拦截。\n\n数据缺失允许保留，详见 missing.csv；selection.csv 记录全部 200 篇的纳入／排除情况和明确拦截来源。原始 limited 状态、证据不足和评测失败不改为成功。未启动任何模型调用。\n\n目录：papers.jsonl 为新名单；sources 为正文及审稿原文副本；study 为选中论文的既有运行数据及历史报告；data 为当前 Fig.3 的报告、抽取、核验、评分、偏好和聚类。旧 human_evaluation 仅对应旧报告，不作为新报告评价。嵌套历史文件保留原始来源路径，主名单的正文路径已指向副本。\n\n本子集不包含原 200 篇的汇总表或图，后续必须按新名单重新汇总；旧绘图入口仍固定为 200 篇，未切换。选择产生的学科／主题限制须在论文中披露，不能把此子集当成未经筛选的原队列。\n\n原 200 篇、Fig.3 图件和对应代码已存档至 ../archives/fig3_200_20260922.tar.gz；原目录保留。\n'''
    (DESTINATION/'README.md').write_text(note)
    print(json.dumps({'selected': len(selected), 'excluded': list(excluded), 'missing_entries': len(missing), 'destination': str(DESTINATION)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Stage whole figures and publish an already reviewed, result-filled manuscript.

No model calls, result editing, or compilation occurs here. Missing new figures
are never substituted with their previous versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .config import DAMAGE_ID, OUTPUT, ROOT
from .io import read, write

MANUSCRIPT = ROOT / 'manuscript/nature_innovation'
FIGURES = (
    ('Fig01', 'Fig1', 'retained', 'Fig01.pdf'),
    ('Fig02', 'Fig2', 'updated', 'figures/Fig2.pdf'),
    ('Fig03', 'Fig3', 'updated', 'figures/Fig3.pdf'),
    ('Fig04', 'Fig4', 'retained', 'Fig04.pdf'),
    ('Fig05', 'Fig5', 'retained', 'Fig05.pdf'),
    ('Fig06', 'Fig8', 'retained', 'Fig06.pdf'),
    ('ExtendedDataFig01', 'Fig6', 'updated', 'figures/Fig6.pdf'),
    ('ExtendedDataFig02', 'Fig7', 'retained', 'ExtendedDataFig02.pdf'),
    ('ExtendedDataFig03', 'Fig10', 'retained', 'ExtendedDataFig03.pdf'),
    ('ExtendedDataFig04', 'Fig9', 'retained', 'ExtendedDataFig04.pdf'),
    ('ExtendedDataFig05', 'Fig11', 'retained', 'ExtendedDataFig05.pdf'),
    ('ExtendedDataFig06', 'new reviewer framework', 'updated', 'framework/ExtendedDataFig06.pdf'),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_figures(output: Path = OUTPUT, manuscript: Path = MANUSCRIPT) -> dict[str, Any]:
    destination = manuscript / 'figures_v9'
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, original, kind, relative in FIGURES:
        source = (manuscript / 'figures' if kind == 'retained' else output) / relative
        target_path = destination / f'{target}.pdf'
        row = {'manuscript_id': target, 'original_id': original, 'kind': kind,
               'source': str(source), 'destination': str(target_path)}
        if not source.is_file():
            row['status'] = 'missing_new_source'
            row['stale_destination_exists'] = target_path.exists()
        else:
            source_hash = sha256(source)
            if not target_path.is_file() or sha256(target_path) != source_hash:
                shutil.copy2(source, target_path)
            row.update(status='ready', sha256=source_hash)
            assert sha256(target_path) == source_hash
        rows.append(row)
    result = {'figures': rows, 'ready': sum(r['status'] == 'ready' for r in rows),
              'expected': len(rows), 'old_figures_directory_untouched': True}
    write(output / 'delivery/figure_manifest.json', result)
    return result


def readiness(source: Path, output: Path = OUTPUT, manuscript: Path = MANUSCRIPT,
              allow_incomplete: bool = False) -> dict[str, Any]:
    from .aggregate import load_result
    from .damage_figure import verification_details

    failures = []
    required = ['summary.json', 'tables/dimension_scores.csv', 'tables/summary.csv',
                'tables/paired_effects.csv', 'tables/support_summary.csv', 'tables/missing.csv',
                f'reports/full/{DAMAGE_ID}.json',
                f'evaluations/full/{DAMAGE_ID}.json', 'figures/DAMAGE_editorial_en.json',
                'figures/Fig6_unchanged_regions.json', 'figures/fig3_replacement_audit.json',
                'figures/Fig2_text_changes.json']
    failures.extend(f'Missing required result: {name}' for name in required if not (output / name).is_file())
    damage_verification = None
    damage_result, damage_error = load_result(output, DAMAGE_ID, 'full')
    if damage_error is not None or damage_result is None:
        failures.append(f'DAMAGE current verification unavailable: {damage_error}')
    else:
        try:
            damage_verification = verification_details(damage_result)
        except ValueError as exc:
            failures.append(str(exc))
        if damage_verification and damage_verification['technical_failure_dimensions']:
            if not (output / 'tables/technical_failures.csv').is_file():
                failures.append('DAMAGE partial verification requires the explicit technical-failure table.')
    summary_path = output / 'summary.json'
    if summary_path.is_file():
        summary = read(summary_path)
        counts = summary['completion']
        if counts['main_papers'] != 100 or counts['main_expected_evaluations'] != 600:
            failures.append('Summary does not cover the 100-paper six-method design.')
        if counts['main_completed_evaluations'] < 600 and not allow_incomplete:
            failures.append('Main comparison incomplete; explicitly document missing results before allowing partial completion.')
        if not any(row['n'] > 0 for row in summary['paired_effects']):
            failures.append('No real paired five-dimensional results.')
        if counts.get('damage_available_evaluations', counts['damage_completed_evaluations']) != 1:
            failures.append('Summary does not include the available full or partial DAMAGE verification.')
    if not source.is_file():
        failures.append(f'Missing reviewed manuscript: {source}')
    else:
        text = source.read_text(encoding='utf-8')
        if 'PENDING' in text:
            failures.append('Manuscript still contains PENDING markers; no final manuscript will be created.')
        if re.search(r'\\(?:title|author|maketitle)\b', text):
            failures.append('Manuscript contains title/author commands.')
        if re.search(r'\\section\*?\{[Ll]imitation', text):
            failures.append('Limitation must remain in a separate document.')
        expected = [f'figures_v9/{row[0]}.pdf' for row in FIGURES]
        actual = re.findall(r'\\includepdf\[[^\n]+?\]\{([^}]+)\}', text)
        if actual != expected:
            failures.append('Manuscript whole-figure order or figures_v9 paths differ from the delivery mapping.')
        for relative in expected:
            if not (manuscript / relative).is_file():
                failures.append(f'Missing staged figure: {relative}')
    result = {'ready': not failures, 'failures': failures, 'source': str(source),
              'allow_incomplete': allow_incomplete, 'compilation_performed': False,
              'damage_verification': damage_verification}
    write(output / 'delivery/readiness.json', result)
    return result


def publish(source: Path, output: Path = OUTPUT, manuscript: Path = MANUSCRIPT,
            allow_incomplete: bool = False) -> dict[str, Any]:
    stage_figures(output, manuscript)
    result = readiness(source, output, manuscript, allow_incomplete)
    if not result['ready']:
        return result
    target = manuscript / 'manuscript_zh_v9.tex'
    if target.is_file() and sha256(target) != sha256(source):
        raise FileExistsError(f'Existing final source differs; inspect it before replacing: {target}')
    if not target.is_file():
        shutil.copy2(source, target)
    result.update(published_source=str(target), sha256=sha256(target))
    write(output / 'delivery/readiness.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'check', 'publish'])
    parser.add_argument('--source', type=Path, default=OUTPUT / 'draft/manuscript_zh_v9_pending.tex')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Use only after final missing results and actual denominators are documented.')
    arguments = parser.parse_args()
    if arguments.action == 'stage':
        result = stage_figures()
    elif arguments.action == 'check':
        result = readiness(arguments.source, allow_incomplete=arguments.allow_incomplete)
    else:
        result = publish(arguments.source, allow_incomplete=arguments.allow_incomplete)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('ready') is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

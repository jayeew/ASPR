"""Source-bound Fig8 feasibility pilot; never substitute old graph metrics or novelty scores."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .supplement_allocation import (
    SCENARIOS,
    allocate,
    known_universities,
    official_relationships,
)
from .supplement_queue import ROOT

OUT = ROOT / 'analysis/preview'
TYPES = ['METHOD', 'FINDING', 'MECHANISM', 'RESOURCE', 'THEORY']
SEED = '20261021'


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def norm(value: str) -> str:
    return ' '.join(value.split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sentences(abstract: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract) if s.strip()]


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    native = {p['doi']: p for p in map(json.loads, (ROOT / 'nature/papers.jsonl').open())}
    nature_meta = pq.read_table(ROOT / 'nature/research_roster_release.parquet', columns=['doi', 'primary_topic_json', 'publisher_date']).to_pylist()
    nature_by_field = defaultdict(list)
    for row in nature_meta:
        if row['doi'] not in native:
            continue
        topic = json.loads(row['primary_topic_json'] or '{}') or {}
        paper = native[row['doi']]
        field = topic.get('field', {}).get('display_name', 'Unknown')
        nature_by_field[field].append({**paper, 'field': field, 'online_date': row['publisher_date'], 'journal': 'Nature', 'portfolio': 'True', 'panel': 'a', 'source_receipt': 'nature/papers.jsonl#' + row['doi']})
    top_fields = sorted(nature_by_field, key=lambda f: (-len(nature_by_field[f]), f))[:6]
    selected = []
    for field in top_fields:
        candidates = sorted(nature_by_field[field], key=lambda p: digest(SEED + p['doi']))
        # Two calendar years per field when possible; selection precedes claim coding.
        picked = [candidates[0]]
        alternatives = [p for p in candidates if p['year'] != picked[0]['year']]
        if alternatives:
            picked.append(alternatives[0])
        elif len(candidates) > 1:
            picked.append(candidates[1])
        selected.extend(picked)
    a_dois = {p['doi'] for p in selected}
    canonical = pq.read_table('data/claim_graph/canonical_target_works.parquet').to_pylist()
    old_dois = {r['doi'].lower(): r for r in canonical}
    abstracts = {r['doi'].lower(): r for r in pq.read_table('data/claim_graph/abstracts.parquet').to_pylist()}
    claims = pq.read_table('data/claim_graph/claim_nodes.parquet').to_pylist()
    old_claims = defaultdict(list)
    for row in claims:
        old_claims[row['parent_paper_id']].append(row)
    groups, counts, audit = defaultdict(list), Counter(), []
    files = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        check = json.loads((ROOT / 'fields/qualification/recomputed_results.json').read_text())
        for path in sorted((ROOT / 'fields/papers').glob('*.parquet')):
            decision_path = ROOT / 'fields/qualification/decisions' / path.name
            decisions = {r['work_id']: r for r in pq.read_table(decision_path).to_pylist()}
            files.append({'paper_shard': str(path), 'paper_mtime_ns': path.stat().st_mtime_ns, 'decision_shard': str(decision_path), 'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest()})
            for batch in pq.ParquetFile(path).iter_batches(batch_size=500, columns=['work_id', 'doi', 'field', 'title', 'journal_name', 'abstract', 'abstract_source', 'authors_json', 'author_list_may_be_truncated', 'topic_id']):
                for row in batch.to_pylist():
                    decision = decisions[row['work_id']]
                    doi = row['doi']
                    if doi in old_dois:
                        old = abstracts.get(doi, {})
                        old_rows = old_claims.get(old_dois[doi]['nature_article_id'], [])
                        audit.append({'doi': doi, 'current_eligibility': decision['eligibility_status'], 'field': row['field'],
                            'old_claim_n': len(old_rows), 'abstract_normalized_exact': norm(old.get('abstract_text', '')) == norm(row['abstract'] or ''),
                            'support_fragments_present': bool(old_rows) and all(norm(fragment) in norm(row['abstract'] or '') for c in old_rows for fragment in c['source_fragments']),
                            'fulltext_available': Path(old.get('paper_markdown_path', '/nonexistent')).is_file(),
                            'formal_reuse_status': 'candidate_only_extraction_provenance_not_certified'})
                    if decision['eligibility_status'] != 'included_verified_research':
                        continue
                    counts['qualified'] += 1
                    label = decision['nature_portfolio_at_verified_online_year']
                    if label not in ['True', 'False'] or len(row['abstract'] or '') < 80 or row['author_list_may_be_truncated']:
                        continue
                    authors = json.loads(row['authors_json'])
                    personal = [a for a in authors if not a.get('nonpersonal_verified')]
                    if doi in a_dois or not personal or not any(a.get('publisher_affiliation_slots') or a.get('nature_v1_slots') for a in personal):
                        continue
                    group = (row['field'], decision['online_year'], label)
                    counts['pilot_pool_' + '|'.join(map(str, group))] += 1
                    candidate = {'doi': doi, 'work_id': row['work_id'], 'title': row['title'], 'year': decision['online_year'], 'field': row['field'],
                        'online_date': decision['online_date'], 'journal': row['journal_name'], 'portfolio': label, 'panel': 'bc',
                        'abstract': row['abstract'], 'abstract_source': row['abstract_source'], 'authors': authors, 'topic_id': row['topic_id'],
                        'source_receipt': str(path.relative_to(ROOT)) + '#' + row['work_id'], 'qualification_evidence_ids': decision['evidence_ids']}
                    def selection_rank(paper: dict) -> tuple[int, str]:
                        if paper['portfolio'] == 'False':
                            return 0, digest(SEED + paper['doi'])
                        target = {2023: 'Nature', 2024: 'Nature Communications', 2025: 'other'}[paper['year']]
                        actual = paper['journal'] if paper['journal'] in ['Nature', 'Nature Communications'] else 'other'
                        return int(actual != target), digest(SEED + paper['doi'])
                    if not groups[group] or selection_rank(candidate) < selection_rank(groups[group][0]):
                        groups[group] = [candidate]
        fcntl.flock(lock, fcntl.LOCK_UN)
    for group in sorted(groups):
        selected.extend(groups[group])
    # Avoid outcome-based selection; the sample is explicitly a source-availability pilot.
    selected.sort(key=lambda p: (p['panel'], p['field'], p['year'], p['portfolio'], p['doi']))
    for i, paper in enumerate(selected, 1):
        paper['pilot_id'] = f'P{i:02d}'
        paper['abstract_sha256'] = digest(paper['abstract'])
        paper['sentences'] = [{'id': f'S{j:02d}', 'text': s} for j, s in enumerate(sentences(paper['abstract']), 1)]
    write_json(OUT / 'pilot_inputs.json', selected)
    (OUT / 'coding_inputs.txt').write_text('\n\n'.join(p['pilot_id'] + '\n' + '\n'.join(s['id'] + ' ' + s['text'] for s in p['sentences']) for p in selected))
    write_csv(OUT / 'reuse_candidates.csv', audit)
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'seed': SEED, 'sample_n': len(selected), 'a_n': sum(p['panel'] == 'a' for p in selected),
        'bc_n': sum(p['panel'] == 'bc' for p in selected), 'nature_field_counts': {k: len(v) for k, v in nature_by_field.items()},
        'a_top_fields': top_fields, 'pilot_pool_counts': dict(counts), 'completed_checkpoint': check,
        'old_canonical_n': len(canonical), 'old_claim_n': len(claims), 'old_overlap_candidates_n': len(audit),
        'old_overlap_qualified_n': sum(r['current_eligibility'] == 'included_verified_research' for r in audit),
        'old_overlap_qualified_exact_abstract_n': sum(r['current_eligibility'] == 'included_verified_research' and r['abstract_normalized_exact'] for r in audit),
        'old_overlap_qualified_supported_fragments_n': sum(r['current_eligibility'] == 'included_verified_research' and r['support_fragments_present'] for r in audit),
        'old_nature_main_n': sum(r['source_name'] == 'Nature' for r in canonical),
        'study_years': dict(Counter(json.loads(line)['publication_date'][:4] for line in Path('outputs/innovation_200_20260907/papers.jsonl').read_text().splitlines())),
        'reference_2018_2022_claim_graph_ready': False,
        'sampling_scope': 'Technical source-availability pilot, not a probability sample of all journals. A: two papers from distinct years where possible in each of six largest metadata fields. B/C: one per field/year/known-portfolio stratum with at least one publisher affiliation link, excluding a DOIs. For portfolio coverage, prefer Nature in 2023, Nature Communications in 2024, other Portfolio journals in 2025 when available. Does not certify byline or mapping completeness.',
        'source_files': files}
    write_json(OUT / 'reuse_audit.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['source_files', 'completed_checkpoint', 'pilot_pool_counts', 'nature_field_counts']}, ensure_ascii=False))


def diversify_portfolio() -> None:
    """Guarantee publisher-family coverage without filtering on institution completeness."""
    import sqlite3
    papers = json.loads((OUT / 'pilot_inputs.json').read_text())
    groups = defaultdict(list)
    with (ROOT / 'fields/qualification/research_roster.csv').open(encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            if r['nature_portfolio_at_verified_online_year'] != 'True':
                continue
            year = int(r['online_year'])
            journal = r['journal_name'].casefold()
            if (year == 2024 and journal == 'nature communications') or (year == 2025 and journal not in ['nature', 'nature communications']):
                groups[(r['field'], year)].append(r)
    db = sqlite3.connect(ROOT / 'supplement/tasks.sqlite')
    used = {p['doi'] for p in papers}
    replaced = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        for i, old in enumerate(papers):
            if old['panel'] != 'bc' or old['portfolio'] != 'True' or old['year'] == 2023:
                continue
            candidates = sorted(groups[(old['field'], old['year'])], key=lambda r: digest(SEED + r['doi']))
            for r in candidates:
                if r['doi'] in used and r['doi'] != old['doi']:
                    continue
                record = db.execute('SELECT shard FROM papers WHERE work_id=?', (r['work_id'],)).fetchone()
                if not record:
                    continue
                path = ROOT / 'fields/papers' / record[0]
                row = pq.read_table(path, filters=[('work_id', '=', r['work_id'])]).to_pylist()[0]
                if len(row['abstract'] or '') < 80 or row['author_list_may_be_truncated']:
                    continue
                new = {**old, 'doi': r['doi'], 'work_id': r['work_id'], 'title': r['title'], 'journal': r['journal_name'],
                       'online_date': r['online_date'], 'abstract': row['abstract'], 'abstract_source': row['abstract_source'],
                       'authors': json.loads(row['authors_json']), 'topic_id': row['topic_id'], 'source_receipt': str(path.relative_to(ROOT)) + '#' + r['work_id'],
                       'qualification_evidence_ids': r['evidence_ids'], 'abstract_sha256': digest(row['abstract'])}
                new['sentences'] = [{'id': f'S{j:02d}', 'text': t} for j, t in enumerate(sentences(new['abstract']), 1)]
                used.discard(old['doi']);used.add(new['doi']);papers[i] = new
                replaced.append({'pilot_id': old['pilot_id'], 'old_doi': old['doi'], 'doi': new['doi'], 'journal': new['journal'], 'reason': 'Predefined portfolio-family coverage; no institution-completeness requirement or claim outcome selection'})
                break
    db.close()
    write_json(OUT / 'pilot_inputs.json', papers)
    write_json(OUT / 'portfolio_sampling_adjustment.json', replaced)
    (OUT / 'coding_inputs.txt').write_text('\n\n'.join(p['pilot_id'] + '\n' + '\n'.join(s['id'] + ' ' + s['text'] for s in p['sentences']) for p in papers))
    print(json.dumps(replaced))


def structural_metrics(neighborhoods: list[set[str]], communities: dict[str, str]) -> dict:
    nonempty = [n for n in neighborhoods if n]
    union = set().union(*nonempty) if nonempty else set()
    labels = Counter(communities[n] for n in union if n in communities)
    total = sum(labels.values())
    diversity = float(np.exp(-sum((n / total) * np.log(n / total) for n in labels.values()))) if total else None
    pairs = [len(a & b) / len(a | b) for i, a in enumerate(nonempty) for b in nonempty[i + 1:]]
    shared = sum(sum(node in n for n in nonempty) >= 2 for node in union)
    return {'D': diversity, 'J': float(np.mean(pairs)) if pairs else None,
            'S': shared / len(union) if len(nonempty) >= 2 else None}


def checked_claims(paper: dict, annotations: dict) -> list[dict]:
    claims = annotations[paper['pilot_id']]
    if not 0 <= len(claims) <= 3:
        raise ValueError('Pilot permits zero to three supported claims per selected paper')
    by_id = {s['id']: s['text'] for s in paper['sentences']}
    output = []
    for i, item in enumerate(claims, 1):
        if item['type'] not in TYPES or not item['sentences'] or not item['claim'].strip():
            raise ValueError('Invalid claim annotation')
        quotes = [by_id[ident] for ident in item['sentences']]
        if not all(q in paper['abstract'] for q in quotes):
            raise ValueError('Support text not in source abstract')
        output.append({'claim_id': paper['pilot_id'] + f'::C{i:02d}', 'doi': paper['doi'], 'type': item['type'],
                       'text': item['claim'], 'sentence_ids': item['sentences'], 'support_quotes': quotes,
                       'abstract_sha256': paper['abstract_sha256'], 'source_receipt': paper['source_receipt'],
                       'coding_method': 'assistant_coded_pilot_in_this_chat; no separate inference API; not human annotation or production extractor output'})
    return output


def group_composition(papers: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for paper in papers:
        groups[paper[key]].append(paper)
    return [{'group': name, 'paper_n': sum(r['claim_n'] > 0 for r in rows), 'eligible_n': len(rows),
             **{t: float(np.mean([r[t] for r in rows if r['claim_n'] > 0])) if any(r['claim_n'] > 0 for r in rows) else None for t in TYPES}} for name, rows in groups.items()]


def analyze() -> None:
    papers = json.loads((OUT / 'pilot_inputs.json').read_text())
    annotations = json.loads((OUT / 'pilot_annotations.json').read_text())
    annotated_hashes = json.loads((OUT / 'annotation_input_hashes.json').read_text())
    if annotated_hashes != {p['pilot_id']: digest(p['abstract']) for p in papers}:
        raise ValueError('Annotations belong to different input abstracts; recode before analysis')
    if set(annotations) != {p['pilot_id'] for p in papers}:
        raise ValueError('Annotation roster differs from pilot roster')
    quality = json.loads((OUT / 'pilot_input_quality.json').read_text())
    metrics, claims = [], []
    known, relationships = known_universities(ROOT), official_relationships(ROOT)
    weights = []
    for paper in papers:
        if digest(paper['abstract']) != paper['abstract_sha256']:
            raise ValueError('Pilot abstract changed after selection')
        coded = checked_claims(paper, annotations)
        claims.extend(coded)
        counts = Counter(c['type'] for c in coded)
        metrics.append({k: paper[k] for k in ['pilot_id', 'doi', 'panel', 'field', 'year', 'portfolio', 'journal']} | {
            'claim_n': len(coded), 'claim_status': quality.get(paper['pilot_id'], {}).get('status', 'supported_claims' if coded else 'no_supported_claims'), **{t: counts[t] / len(coded) if coded else None for t in TYPES}, 'D': None, 'J': None, 'S': None,
            'structure_status': 'missing_common_2018_2022_reference_graph', 'adjusted_status': 'not_estimable_pilot_strata_below_100',
            'fulltext_required_for_these_metrics': False})
        if paper['panel'] != 'bc':
            continue
        for scenario in SCENARIOS:
            countries, universities, _ = allocate({'authors_json': json.dumps(paper['authors'])}, scenario, known, relationships)
            for dimension, values in [('country', countries), ('university', universities)]:
                for entity, weight in values.items():
                    weights.append({'pilot_id': paper['pilot_id'], 'doi': paper['doi'], 'scenario': scenario, 'dimension': dimension, 'entity': entity, 'weight': weight})
                if dimension == 'university' and countries:
                    weights.append({'pilot_id': paper['pilot_id'], 'doi': paper['doi'], 'scenario': scenario, 'dimension': dimension,
                                    'entity': 'non_university', 'weight': max(0, 1 - sum(values.values()))})
    by_id = {p['pilot_id']: p for p in metrics}
    aggregates = []
    names = json.loads((ROOT / 'nature/entity_names.json').read_text())
    for scenario in SCENARIOS:
        for dimension in ['country', 'university']:
            entities = defaultdict(list)
            for w in weights:
                if w['scenario'] == scenario and w['dimension'] == dimension and w['weight'] > 0:
                    entities[w['entity']].append(w)
            for entity, rows in entities.items():
                total = sum(r['weight'] for r in rows)
                aggregates.append({'scenario': scenario, 'dimension': dimension, 'entity': entity, 'name': names.get(entity, entity),
                    'paper_n': len(rows), 'fractional_papers': total, 'effective_n': total ** 2 / sum(r['weight'] ** 2 for r in rows),
                    **{t: sum(r['weight'] * by_id[r['pilot_id']][t] for r in rows) / total for t in TYPES},
                    'formal_display_status': 'not_estimable_pilot_below_prespecified_thresholds'})
    write_json(OUT / 'pilot_claims.json', claims)
    write_csv(OUT / 'paper_metrics.csv', metrics)
    write_csv(OUT / 'allocation_weights.csv', weights)
    write_csv(OUT / 'country_university_metrics.csv', aggregates)
    a = group_composition([p for p in metrics if p['panel'] == 'a'], 'field')
    b = group_composition([{**p, 'group': p['field'] + '|' + p['portfolio']} for p in metrics if p['panel'] == 'bc'], 'group')
    write_csv(OUT / 'panel_a_composition.csv', a)
    write_csv(OUT / 'panel_b_composition.csv', b)
    # Confirm conservation separately from scientific/entity correctness.
    main = [w for w in weights if w['scenario'] == 'main_personal_direct']
    for p in [p for p in metrics if p['panel'] == 'bc']:
        for dimension in ['country', 'university']:
            total = sum(w['weight'] for w in main if w['pilot_id'] == p['pilot_id'] and w['dimension'] == dimension)
            if abs(total - 1) > 1e-8:
                raise ValueError('Fractional mass not conserved')
    summary = {'paper_n': len(metrics), 'claim_n': len(claims), 'a_paper_n': sum(p['panel'] == 'a' for p in metrics),
        'bc_paper_n': sum(p['panel'] == 'bc' for p in metrics), 'claimed_paper_n': sum(p['claim_n'] > 0 for p in metrics), 'input_quality_hold_n': len(quality), 'all_claims_source_spans_valid': True,
        'fractional_mass_conserved': True, 'D_available_n': 0, 'J_available_n': 0, 'adjusted_comparisons_available_n': 0,
        'country_unknown_fractional_papers': sum(w['weight'] for w in main if w['dimension'] == 'country' and w['entity'] == 'unknown'),
        'university_unknown_fractional_papers': sum(w['weight'] for w in main if w['dimension'] == 'university' and w['entity'] == 'unknown'),
        'semantic_review': 'Assistant source-bound pilot; independent human adjudication remains pending',
        'external_model_calls': 0, 'new_embeddings': 0, 'formal_comparison_ready': False}
    write_json(OUT / 'pilot_validation.json', summary)
    render(a, b, aggregates, summary)
    print(json.dumps(summary))


def render(a: list[dict], b: list[dict], entities: list[dict], summary: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1.0, 1.1], hspace=.6, wspace=.62)
    cmap = LinearSegmentedColormap.from_list('share', ['#f5f7f8', '#b2cbd7', '#235b79'])
    def heatmap(ax, rows: list[dict], labels: list[str], title: str) -> None:
        values = np.array([[r[t] * 100 for t in TYPES] for r in rows])
        ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(5), ['Method', 'Finding', 'Mechanism', 'Resource', 'Theory'], rotation=25, ha='right')
        ax.set_yticks(range(len(rows)), labels)
        for i in range(len(rows)):
            for j in range(5):
                ax.text(j, i, f'{values[i,j]:.0f}', ha='center', va='center', color='white' if values[i,j] > 65 else '#202b33', fontsize=9)
        ax.set_title(title, loc='left', fontsize=12, pad=14)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    short = {'Biochemistry, Genetics and Molecular Biology': 'Biochemistry / genetics', 'Physics and Astronomy': 'Physics / astronomy',
             'Agricultural and Biological Sciences': 'Agricultural / biological sciences', 'Earth and Planetary Sciences': 'Earth / planetary sciences',
             'Immunology and Microbiology': 'Immunology / microbiology'}
    heatmap(fig.add_subplot(grid[0, 0]), a, [short.get(r['group'], r['group']) + f" (n={r['paper_n']}/{r['eligible_n']})" for r in a], 'a  Nature: paper-weighted claim roles (%)')
    ax = fig.add_subplot(grid[0, 1]); ax.axis('off')
    ax.text(0, 1, 'PREVIEW / technical pilot only', fontsize=15, weight='bold', va='top', color='#875f16')
    lines = [f"{summary['paper_n']} eligible papers | {summary['claimed_paper_n']} claim-bearing | {summary['claim_n']} claims", '',
             'Assistant-coded abstracts; no independent human validation.',
             'Source-availability sample; not representative of all journals.', '',
             'D / J: not computed — common 2018–2022 graph absent.',
             'Adjusted effects / intervals: not estimable at pilot n.',
             'Full text: optional for this abstract-profile experiment.', '',
             'All displayed compositions are raw descriptive pilot values.',
             'No innovation scores or country/university rankings.']
    ax.text(0, .84, '\n'.join(lines), va='top', linespacing=1.7, fontsize=10)
    ax = fig.add_subplot(grid[1, :])
    labels = []
    colors = ['#285f83', '#a9bfd0', '#c89036', '#9b9d60', '#b7859a']
    left = np.zeros(len(b))
    for i, t in enumerate(TYPES):
        values = np.array([r[t] * 100 for r in b])
        ax.barh(range(len(b)), values, left=left, color=colors[i], label=t.title(), height=.64, edgecolor='white', linewidth=.6)
        left += values
    for i, r in enumerate(b):
        field, brand = r['group'].split('|')
        labels.append(f"{field.title()} · {'Nature Portfolio' if brand == 'True' else 'Other publishers'}  (n={r['paper_n']})")
    ax.set_yticks(range(len(b)), labels);ax.invert_yaxis();ax.set_xlim(0, 100)
    for label, r in zip(ax.get_yticklabels(), b, strict=True):
        if r['group'].endswith('|True'):
            label.set_color('#875f16');label.set_weight('bold')
    ax.set_xlabel('Paper-weighted share of extracted claims (%)')
    ax.set_title('b  Three fields: raw publisher-group composition (pilot substitute for pending structural comparison)', loc='left', fontsize=12, pad=30)
    ax.legend(ncol=5, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.01), fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    import textwrap
    for col, dimension in enumerate(['country', 'university']):
        rows = sorted([r for r in entities if r['scenario'] == 'main_personal_direct' and r['dimension'] == dimension and r['entity'] not in ['unknown', 'non_university']], key=lambda r: (-r['fractional_papers'], r['entity']))[:5]
        labels = [textwrap.fill({'Università degli Studi del Piemonte Orientale “Amedeo Avogadro”': 'University of Eastern Piedmont'}.get(r['name'], r['name']), 26) + f"\n(w={r['fractional_papers']:.2f}; n={r['paper_n']})" for r in rows]
        heatmap(fig.add_subplot(grid[2, col]), rows, labels, 'c  ' + ('Countries' if col == 0 else 'Universities') + ': raw weighted composition (%)')
    fig.suptitle('Fig8 — source-bound contribution-profile pilot', x=.05, ha='left', fontsize=18, weight='bold', y=.985)
    fig.text(.05, .025, f"Panel c uses the same {summary['bc_paper_n']} papers as b. Unknown country mass: {summary['country_unknown_fractional_papers']:.2f}; university mass: {summary['university_unknown_fractional_papers']:.2f} paper-equivalents.\nTop five shown only for workflow inspection; formal top-12 / top-15 selections and thresholds are not claimed. Source records and recomputation tables accompany the figure.", fontsize=9, color='#4b555d')
    fig.subplots_adjust(left=.23, right=.97, top=.92, bottom=.14)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / ('Fig8_preview.' + ext), dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'diversify', 'analyze'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
        diversify_portfolio()
    elif args.action == 'diversify':
        diversify_portfolio()
    else:
        analyze()


if __name__ == '__main__':
    main()

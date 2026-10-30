"""Offline Fig8 preview on the completed cohort; never launch acquisition or models."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .fig8_preview import TYPES, checked_claims, digest, norm, sentences, write_csv, write_json
from .journal_readiness import abstract_screen
from .supplement_allocation import SCENARIOS, allocate, known_universities, official_relationships
from .supplement_queue import ROOT

OUT = ROOT / 'analysis/preview'
FIELDS = ['biochemistry', 'materials', 'physics']
LABELS = {'biochemistry': 'Biochemistry / genetics', 'materials': 'Materials science',
          'physics': 'Physics / astronomy'}
COMPARATORS = {'https://openalex.org/S3880285': 'Science',
               'https://openalex.org/S110447773': 'Cell',
               'https://openalex.org/S125754415': 'PNAS'}
NATURE = 'https://openalex.org/S137773608'


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def brand_status(counts: Counter) -> str:
    """A journal with unknown years must not silently become a non-Portfolio journal."""
    keys = {k for k, v in counts.items() if v}
    if keys == {'True'}:
        return 'Nature Portfolio'
    if keys == {'False'}:
        return 'Other publishers'
    return 'Mixed / unconfirmed'


def build_inventory() -> tuple[dict, list[dict]]:
    completion = json.loads((ROOT / 'supplement/target_500k/completion.json').read_text())
    decisions, brands, annual = {}, defaultdict(Counter), Counter()
    for path in sorted((ROOT / 'fields/qualification/decisions').glob('*.parquet')):
        for row in pq.read_table(path, filters=[('eligibility_status', '=', 'included_verified_research')]).to_pylist():
            if row['doi'] in decisions:
                raise ValueError('Qualified DOI is not unique')
            row['shard'] = path.name
            decisions[row['doi']] = row
            label = row['nature_portfolio_at_verified_online_year']
            for field in [row['field'], 'all_three_fields']:
                brands[(row['source_id'], field)][label] += 1
            annual[(row['field'], row['online_year'])] += 1
    assert len(decisions) == completion['included'] == 500002
    rows = csv_rows(ROOT / 'analysis/journal_readiness/all_journals.csv')
    for row in rows:
        counts = brands[(row['source_id'], row['field'])]
        assert sum(counts.values()) == int(row['verified_n'])
        row['portfolio_group'] = brand_status(counts)
        row['portfolio_confirmed_n'] = counts['True']
        row['nonportfolio_confirmed_n'] = counts['False']
        row['brand_unconfirmed_n'] = sum(counts.values()) - counts['True'] - counts['False']
        row['candidate_verified_fraction'] = int(row['verified_n']) / int(row['candidate_n'])
        row['formal_ready'] = False
    assert sum(int(r['verified_n']) for r in rows if r['field'] == 'all_three_fields') == 500002
    write_csv(OUT / 'journal_plot_data.csv', rows)
    write_csv(OUT / 'cohort_field_year.csv', [{'field': f, 'online_year': y, 'included_n': n}
              for (f, y), n in sorted(annual.items())])
    return decisions, rows


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    decisions, journal_rows = build_inventory()
    original = json.loads((OUT / 'pilot_inputs.json').read_text())
    selected = {r['doi']: r for r in original if r['doi'] in decisions}
    reuse = [{'pilot_id': r['pilot_id'], 'doi': r['doi'], 'status':
              'current_cohort_member' if r['doi'] in decisions else 'outside_fixed_cohort_retained_only_as_old_evidence'}
             for r in original]
    new_ids = {}
    for source, name in COMPARATORS.items():
        pool = sorted([d for d in decisions.values() if d['source_id'] == source],
                      key=lambda d: digest('20261021' + d['doi']))
        for index, row in enumerate(pool[:2], 1):
            new_ids[row['doi']] = f'{name}{index}'
    targets = set(selected) | set(new_ids)
    canonical = {r['doi'].lower(): r for r in pq.read_table('data/claim_graph/canonical_target_works.parquet').to_pylist() if r['doi']}
    old_abstracts = {r['doi'].lower(): r for r in pq.read_table('data/claim_graph/abstracts.parquet').to_pylist()}
    old_claims = defaultdict(list)
    for row in pq.read_table('data/claim_graph/claim_nodes.parquet').to_pylist():
        old_claims[row['parent_paper_id']].append(row)
    inputs, audit = [], []
    for path in sorted((ROOT / 'fields/papers').glob('*.parquet')):
        wanted = [doi for doi in targets | (set(canonical) & set(decisions)) if decisions[doi]['shard'] == path.name]
        if not wanted:
            continue
        cols = ['work_id', 'doi', 'title', 'field', 'source_id', 'journal_name', 'abstract', 'abstract_source', 'authors_json', 'author_list_may_be_truncated']
        for row in pq.read_table(path, columns=cols, filters=[('doi', 'in', wanted)]).to_pylist():
            doi = row['doi']
            if doi not in decisions or decisions[doi]['work_id'] != row['work_id']:
                continue
            d = decisions[doi]
            if doi in canonical:
                claims = old_claims[canonical[doi]['nature_article_id']]
                exact = norm(old_abstracts.get(doi, {}).get('abstract_text', '')) == norm(row['abstract'] or '')
                support = bool(claims) and all(c['source_fragments'] and all(norm(f) in norm(row['abstract'] or '') for f in c['source_fragments']) for c in claims)
                audit.append({'doi': doi, 'work_id': row['work_id'], 'journal': row['journal_name'],
                              'claim_n': len(claims), 'exact_abstract': exact, 'all_support_fragments_present': support,
                              'formal_reuse': 'pending_extraction_protocol_compatibility',
                              'legacy_graph_reuse': 'incompatible_2023_2025_target_graph'})
            if doi not in targets:
                continue
            old = selected.get(doi)
            if old and norm(old['abstract']) != norm(row['abstract'] or ''):
                reuse.append({'pilot_id': old['pilot_id'], 'doi': doi, 'status': 'abstract_changed_annotation_not_reused'})
                continue
            p = {'pilot_id': old['pilot_id'] if old else new_ids[doi], 'doi': doi, 'work_id': row['work_id'],
                 'title': row['title'], 'field': row['field'], 'journal': row['journal_name'], 'source_id': row['source_id'],
                 'year': d['online_year'], 'online_date': d['online_date'],
                 'portfolio': d['nature_portfolio_at_verified_online_year'],
                 'abstract': row['abstract'], 'abstract_sha256': digest(row['abstract'] or ''),
                 'abstract_source': row['abstract_source'], 'authors': json.loads(row['authors_json']),
                 'author_list_may_be_truncated': row['author_list_may_be_truncated'],
                 'source_receipt': str(path.relative_to(ROOT)) + '#' + row['work_id'],
                 'qualification_evidence_ids': d['evidence_ids'],
                 'sentences': old['sentences'] if old else [{'id': f'S{i:02d}', 'text': s} for i, s in enumerate(sentences(row['abstract'] or ''), 1)],
                 'annotation_origin': 'reused_pilot' if old else 'new_comparator_pilot',
                 'abstract_screen': abstract_screen(row['abstract'], doi)}
            inputs.append(p)
    write_json(OUT / 'current_pilot_inputs.json', sorted(inputs, key=lambda p: p['pilot_id']))
    write_csv(OUT / 'pilot_reuse_status.csv', reuse)
    write_csv(OUT / 'current_legacy_reuse.csv', audit)
    counts = {'fixed_cohort_n': len(decisions), 'journals_with_included_papers': sum(int(r['verified_n']) > 0 for r in journal_rows if r['field'] == 'all_three_fields'),
              'nature_in_fixed_cohort_n': sum(d['source_id'] == NATURE for d in decisions.values()),
              'legacy_qualified_overlap_n': len(audit),
              'legacy_exact_abstract_n': sum(r['exact_abstract'] for r in audit),
              'legacy_supported_fragments_n': sum(r['all_support_fragments_present'] for r in audit),
              'legacy_formal_protocol_certified_n': 0,
              'reused_pilot_n': sum(p['annotation_origin'] == 'reused_pilot' for p in inputs),
              'new_comparator_pilot_n': sum(p['annotation_origin'] != 'reused_pilot' for p in inputs),
              'new_network_requests': 0, 'new_model_api_calls': 0,
              'metadata_sources': 'current master and completed journal_readiness; no qualification or report refresh',
              'analysis_scope': 'Fixed observed qualified cohort; not all journal output; a restricted to Nature intersection',
              'created_at': datetime.now(timezone.utc).isoformat()}
    write_json(OUT / 'current_reuse_summary.json', counts)
    for paper in inputs:
        if paper['annotation_origin'] == 'new_comparator_pilot':
            print(paper['pilot_id'], paper['doi'], paper['title'], paper['abstract_screen'])
            for s in paper['sentences']:
                print(s['id'], s['text'])
    print(json.dumps(counts))


def compose(rows: list[dict], keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for values, papers in sorted(groups.items()):
        valid = [p for p in papers if p['claim_n'] > 0]
        output.append(dict(zip(keys, values)) | {'eligible_n': len(papers), 'metric_n': len(valid),
            **{t: float(np.mean([p[t] for p in valid])) if valid else None for t in TYPES}})
    return output


def summarize_entities(weights: list[dict], metrics: list[dict], names: dict) -> list[dict]:
    by_id = {p['pilot_id']: p for p in metrics}
    groups = defaultdict(list)
    for row in weights:
        if row['weight'] > 0:
            groups[(row['scenario'], row['dimension'], row['entity'])].append(row)
    result = []
    for (scenario, dimension, entity), rows in sorted(groups.items()):
        valid = [r for r in rows if by_id[r['pilot_id']]['claim_n'] > 0]
        total = sum(r['weight'] for r in rows)
        metric_total = sum(r['weight'] for r in valid)
        result.append({'scenario': scenario, 'dimension': dimension, 'entity': entity,
            'name': names.get(entity, entity), 'paper_n': len(rows), 'metric_n': len(valid),
            'fractional_papers': total, 'metric_fractional_papers': metric_total,
            'effective_n': metric_total ** 2 / sum(r['weight'] ** 2 for r in valid) if valid else 0,
            **{t: sum(r['weight'] * by_id[r['pilot_id']][t] for r in valid) / metric_total if metric_total else None for t in TYPES},
            'formal_ready': False})
    return result


def analyze() -> None:
    inputs = json.loads((OUT / 'current_pilot_inputs.json').read_text())
    annotations = json.loads((OUT / 'pilot_annotations.json').read_text())
    additions = json.loads((OUT / 'comparator_annotations.json').read_text())
    for row in additions:
        paper = next(p for p in inputs if p['pilot_id'] == row['pilot_id'])
        assert paper['abstract_sha256'] == row['abstract_sha256']
        annotations[row['pilot_id']] = row['claims']
    old_hashes = json.loads((OUT / 'annotation_input_hashes.json').read_text())
    metrics, claims, weights = [], [], []
    known, relationships = known_universities(ROOT), official_relationships(ROOT)
    for p in inputs:
        assert digest(p['abstract']) == p['abstract_sha256']
        if p['annotation_origin'] == 'reused_pilot':
            assert old_hashes[p['pilot_id']] == p['abstract_sha256']
        if p['abstract_screen'] in ['quality_review', 'missing']:
            coded = []
        else:
            coded = checked_claims(p, annotations)
        claims.extend(coded)
        counts = Counter(c['type'] for c in coded)
        metrics.append({k: p[k] for k in ['pilot_id', 'doi', 'work_id', 'field', 'year', 'journal', 'source_id', 'portfolio']} | {
            'claim_n': len(coded), 'claim_status': 'pilot_supported' if coded else 'no_usable_claims',
            **{t: counts[t] / len(coded) if coded else None for t in TYPES},
            'D': None, 'J': None, 'S': None, 'adjusted_log_D': None, 'adjusted_J': None,
            'metric_limitation': 'assistant_pilot_not_production;common_historical_graph_missing'})
        for scenario in SCENARIOS:
            if p['author_list_may_be_truncated']:
                countries = universities = {'unknown': 1.0}
            else:
                countries, universities, _ = allocate({'authors_json': json.dumps(p['authors'])}, scenario, known, relationships)
            for dimension, values in [('country', countries), ('university', universities)]:
                values = dict(values)
                if dimension == 'university' and countries:
                    values['non_university'] = max(0.0, 1 - sum(values.values()))
                for entity, weight in values.items():
                    weights.append({'pilot_id': p['pilot_id'], 'doi': p['doi'], 'scenario': scenario,
                                    'dimension': dimension, 'entity': entity, 'weight': weight})
                if scenario != 'complete_case' or countries:
                    assert abs(sum(values.values()) - 1.0) < 1e-8
    names = json.loads((ROOT / 'nature/entity_names.json').read_text())
    entities = summarize_entities(weights, metrics, names)
    a = compose([p for p in metrics if p['source_id'] == NATURE], ['field'])
    b = compose(metrics, ['field', 'source_id', 'journal', 'portfolio'])
    write_json(OUT / 'pilot_claims.json', claims)
    write_csv(OUT / 'paper_metrics.csv', metrics)
    write_csv(OUT / 'panel_a_composition.csv', a)
    write_csv(OUT / 'panel_b_composition.csv', b)
    write_csv(OUT / 'allocation_weights.csv', weights)
    write_csv(OUT / 'country_university_metrics.csv', entities)
    main = [w for w in weights if w['scenario'] == 'main_personal_direct']
    summary = json.loads((OUT / 'current_reuse_summary.json').read_text()) | {
        'pilot_n': len(inputs), 'pilot_claimed_n': sum(r['claim_n'] > 0 for r in metrics), 'pilot_claim_n': len(claims),
        'pilot_journal_n': len({p['source_id'] for p in inputs}), 'pilot_journal_field_points': len(b),
        'a_pilot_n': sum(r['source_id'] == NATURE for r in metrics),
        'country_unknown_weight': sum(w['weight'] for w in main if w['dimension'] == 'country' and w['entity'] == 'unknown'),
        'university_unknown_weight': sum(w['weight'] for w in main if w['dimension'] == 'university' and w['entity'] == 'unknown'),
        'D_available_n': 0, 'J_available_n': 0, 'formal_ready': False,
        'fractional_mass_conserved': True, 'claim_source_fragments_present': True,
        'semantic_validation': 'assistant source-bound coding only; not independent human adjudication',
        'same_papers_in_b_and_c': True, 'bootstrap_not_run_reason': 'technical source-availability sample; no population inference',
        'collection_expansion': False}
    write_json(OUT / 'pilot_validation.json', summary)
    render()
    write_readme(summary)
    print(json.dumps(summary, ensure_ascii=False))


def plot_style() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})


def save_figure(fig: object, name: str) -> None:
    import matplotlib.pyplot as plt
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=190, facecolor='white')
    plt.close(fig)


def heatmap(ax: object, rows: list[dict], labels: list[str], title: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    values = np.array([[float(r[t]) * 100 if r[t] not in [None, ''] else np.nan for t in TYPES] for r in rows])
    cmap = LinearSegmentedColormap.from_list('share', ['#f6f7f8', '#b5ceda', '#245b78'])
    cmap.set_bad('#dddddd')
    ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(5), [t.title() for t in TYPES], rotation=20, ha='right')
    ax.set_yticks(range(len(rows)), labels)
    for i in range(len(rows)):
        for j in range(5):
            value = values[i, j]
            ax.text(j, i, 'NA' if np.isnan(value) else f'{value:.0f}', ha='center', va='center',
                    color='white' if value > 65 else '#26323a', fontsize=10)
    ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=15)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)


def render_pilot(summary: dict) -> None:
    import matplotlib.pyplot as plt
    import textwrap
    a = csv_rows(OUT / 'panel_a_composition.csv')
    b = csv_rows(OUT / 'panel_b_composition.csv')
    entities = csv_rows(OUT / 'country_university_metrics.csv')
    fig = plt.figure(figsize=(19, 14), facecolor='white')
    grid = fig.add_gridspec(3, 6, height_ratios=[.85, 1.65, 1.05], hspace=.65, wspace=2.8)
    heatmap(fig.add_subplot(grid[0, :3]), a,
            [LABELS[r['field']] + f" (n={r['metric_n']})" for r in a],
            'a  Nature within the fixed cohort: claim-role composition (%)')
    ax = fig.add_subplot(grid[0, 3:]); ax.axis('off')
    ax.text(0, 1, 'TECHNICAL PREVIEW — NO FORMAL RANKING', color='#936416', weight='bold', fontsize=13, va='top')
    ax.text(0, .78, f"Fixed metadata cohort: {summary['fixed_cohort_n']:,} papers\n"
            f"Analyzed pilot: {summary['pilot_n']} papers / {summary['pilot_journal_n']} journals / {summary['pilot_claim_n']} claims\n"
            f"Reused {summary['reused_pilot_n']} pilot papers; added six cached comparator abstracts.\n"
            'Assistant-coded, source-availability sample; not representative.\n'
            'D / J unavailable: common historical reference graph absent.\n'
            'No full-text download, new embeddings or acquisition.', va='top', linespacing=1.7)
    short = {'Proceedings of the National Academy of Sciences': 'PNAS',
             'International Journal of Molecular Sciences': 'Int. J. Molecular Sciences',
             'Physical and Engineering Sciences in Medicine': 'Phys. Eng. Sci. Medicine',
             'Cell Communication and Signaling': 'Cell Communication & Signaling'}
    for i, field in enumerate(FIELDS):
        ax = fig.add_subplot(grid[1, i * 2:(i + 1) * 2])
        rows = sorted([r for r in b if r['field'] == field], key=lambda r: (r['portfolio'] != 'True', r['journal']))
        for y, r in enumerate(rows):
            brand = r['portfolio'] == 'True'
            ax.scatter(float(r['METHOD']) * 100, y, s=65, marker='D' if brand else 'o',
                       color='#ad701f' if brand else '#39728b', zorder=3)
        labels = [textwrap.fill(short.get(r['journal'], r['journal']), 25) + f"  [n={r['metric_n']}]" for r in rows]
        ax.set_yticks(range(len(rows)), labels, fontsize=9)
        for label, r in zip(ax.get_yticklabels(), rows, strict=True):
            if r['source_id'] in COMPARATORS or r['portfolio'] == 'True':
                label.set_weight('bold')
            if r['portfolio'] == 'True':
                label.set_color('#936416')
        ax.set_ylim(len(rows) - .3, -.7); ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis='x', color='#e5e8ea', zorder=0)
        ax.set_xlabel('Mean within-paper METHOD share (%)')
        ax.set_title(('b  ' if i == 0 else '') + LABELS[field], loc='left', fontsize=12, weight='bold', pad=16)
    for i, dimension in enumerate(['country', 'university']):
        rows = sorted([r for r in entities if r['scenario'] == 'main_personal_direct' and r['dimension'] == dimension
                       and r['entity'] not in ['unknown', 'non_university']], key=lambda r: -float(r['fractional_papers']))[:5]
        labels = [textwrap.fill(r['name'], 27) + f"\nw={float(r['fractional_papers']):.2f}; n={r['paper_n']}" for r in rows]
        heatmap(fig.add_subplot(grid[2, i * 3:(i + 1) * 3]), rows, labels,
                'c  ' + ('Countries' if dimension == 'country' else 'Universities') + ': pilot-weighted roles (%)')
    fig.suptitle('Fig8 | Fixed-cohort contribution preview', x=.035, ha='left', y=.98, fontsize=20, weight='bold')
    fig.text(.035, .94, 'Gold diamonds: Nature Portfolio. Blue circles: other publishers. Each point in b is one journal within one field.', fontsize=11)
    fig.text(.035, .035, f"Panel b shows descriptive METHOD share, not the planned adjusted ln(D)–J comparison. All pilot journal counts are below the formal n >= 30 threshold.\n"
             f"Panel c uses all {summary['pilot_n']} b papers. Unknown mass: country {summary['country_unknown_weight']:.2f}, university {summary['university_unknown_weight']:.2f} paper-equivalents; unknown is not redistributed.\n"
             'Five pilot entities shown for workflow inspection only; national top-12 / university top-15 stability is not established.', fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=.18, right=.97, top=.88, bottom=.14)
    save_figure(fig, 'Fig8_preview')


def render_journals(summary: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = csv_rows(OUT / 'journal_plot_data.csv')
    colors = {'Nature Portfolio': '#ad701f', 'Other publishers': '#477c94', 'Mixed / unconfirmed': '#9ba3a8'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2))
    for ax, field in zip(axes, FIELDS, strict=True):
        field_rows = [r for r in rows if r['field'] == field and int(r['verified_n']) > 0]
        for group in ['Other publishers', 'Mixed / unconfirmed', 'Nature Portfolio']:
            subset = [r for r in field_rows if r['portfolio_group'] == group]
            ax.scatter([float(r['candidate_verified_fraction']) * 100 for r in subset], [int(r['verified_n']) for r in subset],
                       s=25 if group != 'Nature Portfolio' else 50, marker='D' if group == 'Nature Portfolio' else 'o',
                       facecolors='none' if group == 'Mixed / unconfirmed' else colors[group], edgecolors=colors[group],
                       linewidths=.7, alpha=.7 if group != 'Nature Portfolio' else 1, zorder=3 if group == 'Nature Portfolio' else 2)
        required = [r for r in field_rows if r['source_id'] in COMPARATORS or r['source_id'] == NATURE]
        for j, r in enumerate(sorted(required, key=lambda r: -int(r['verified_n']))):
            name = COMPARATORS.get(r['source_id'], 'Nature')
            ax.annotate(f"{name}  n={int(r['verified_n']):,}", xy=(float(r['candidate_verified_fraction']) * 100, int(r['verified_n'])),
                        xytext=(.04, .96 - j * .075), textcoords='axes fraction', fontsize=9, weight='bold',
                        arrowprops={'arrowstyle': '-', 'color': '#777777', 'linewidth': .7}, va='top',
                        bbox={'boxstyle': 'square,pad=.15', 'fc': 'white', 'ec': 'none', 'alpha': .9})
        missing = [name for source, name in COMPARATORS.items() if not any(r['source_id'] == source for r in field_rows)]
        if missing:
            ax.text(.04, .03, ', '.join(missing) + ': no included paper in this field', transform=ax.transAxes, fontsize=9)
        ax.set_yscale('log'); ax.set_xlim(-2, 103); ax.set_ylim(.8, 100000)
        ax.set_xlabel('Verified research / current candidates (%)')
        ax.set_title(f"{LABELS[field]}\n{len(field_rows):,} journals; {sum(int(r['verified_n']) for r in field_rows):,} papers", fontsize=12, loc='left', pad=12)
        ax.grid(axis='y', color='#e7e9eb', zorder=0)
    axes[0].set_ylabel('Included papers per journal (log scale)')
    legend = [Line2D([], [], marker='D' if g == 'Nature Portfolio' else 'o', linestyle='',
                     markerfacecolor='none' if g == 'Mixed / unconfirmed' else c, markeredgecolor=c, label=g)
              for g, c in colors.items()]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.045, .89), ncol=3, frameon=False)
    fig.suptitle(f"Fig8b | Journal coverage in the fixed {summary['fixed_cohort_n']:,}-paper cohort", x=.05, ha='left', fontsize=19, weight='bold')
    fig.text(.05, .025, 'One point per journal per field. This is a metadata diagnostic, not an innovation or D/J result.\n'
             'Candidate share mixes unresolved records and confirmed exclusions; it is not a journal completeness rate. Nonempty abstract coverage is supplied in journal_plot_data.csv.\n'
             'Nature Portfolio is based on verified publication-year evidence; mixed or unknown annual labels remain separate. Scope is the observed fixed cohort, not all journal output.', fontsize=10)
    fig.subplots_adjust(left=.065, right=.98, top=.76, bottom=.22, wspace=.18)
    save_figure(fig, 'Fig8b_journal_coverage')


def render() -> None:
    plot_style()
    summary = json.loads((OUT / 'pilot_validation.json').read_text())
    render_pilot(summary)
    render_journals(summary)


def write_readme(summary: dict) -> None:
    # Filled from reviewed current results, without triggering metadata recomputation.
    rows = csv_rows(OUT / 'journal_plot_data.csv')
    selected = [r for r in rows if r['source_id'] in COMPARATORS and r['field'] == 'all_three_fields']
    comparator_table = '\n'.join(f"| {COMPARATORS[r['source_id']]} | {int(r['candidate_n']):,} | {int(r['verified_n']):,} | {int(r['pending_n']):,} | {float(r['abstract_nonempty_rate']):.2%} | {int(r['abstract_quality_review_n'])} |" for r in selected)
    text = f'''# Fig8 固定样本重绘预览

本次仅使用现有 **500,002 篇合格去重论文**，未扩大采集、未请求外部数据、未调用付费模型；主表和资格判定不变。本目录是同一数据集的分析产物，不是另一版数据。

## 图和实际分析范围

- `Fig8_preview.png/pdf/svg`：a/b/c 摘要贡献技术预览，**{summary['pilot_n']} 篇、{summary['pilot_journal_n']} 刊、{summary['pilot_claim_n']} 条贡献**。复用 {summary['reused_pilot_n']} 篇旧试编码，新增 Science、Cell、PNAS 各两篇缓存摘要的同类试编码。所有论文都属于固定合格集合。
- `Fig8b_journal_coverage.png/pdf/svg`：以全体固定样本汇总，每领域每刊一个点。Nature Portfolio 用金色菱形，其他出版社用蓝色圆点，年度归属不明或混合单列。标注 Science、Cell、PNAS 和 Nature；无纳入论文的期刊明确注明而不补零点。
- 主图 b 当前横轴为论文等权的 METHOD 贡献比例，完整五类构成保存在 CSV；**这不是原定 ln(D)–J 结构比较**。逐刊数不足 30，不计算正式排名、调整效果或推断区间。
- c 与 b 使用相同的 {summary['pilot_n']} 篇论文，沿用八种归属口径。主口径国家未知 {summary['country_unknown_weight']:.3f}、高校未知 {summary['university_unknown_weight']:.3f} 个论文权重，另保留非高校权重；不把未知分摊给已知机构。

## 各 panel 尚缺的分析

| Panel | 固定样本和指标 | 当前已完成 | 正式结果还缺 |
|---|---|---|---|
| a | 固定集合中的 Nature 主刊，共 {summary['nature_in_fixed_cohort_n']:,} 篇；按三领域展示 T，以及计划中的 D/J | 复用摘要贡献，绘制论文等权 T 热图 | 其余论文统一贡献抽取及质量复核；D/J 历史背景。固定集合只有三个领域，不能冒充原方案 Nature 全领域全主刊分析 |
| b | 三领域，逐刊一点；调整后 ln(D)、J，T 为配套画像 | 全样本逐刊元数据表与散点；小样本逐刊贡献点图，突出 Portfolio，标注对照刊 | 统一贡献、共同参考图、合法邻域、领域/年份调整、缺失集中与共同支持分析、论文级区间；Science/Cell 等是有选择的已核实子集，不能推广全刊 |
| c | 与 b 同集合；作者等权、作者内单位等分；T 与 D/J，国家前12/高校前15按元数据分数产出选名单 | 当前试样按八种口径重新分配，贡献加权汇总及守恒检查 | 固定全样本分数权重与高校主归并、作者/槽位缺失影响、名单敏感性、ESS、统一贡献与结构指标；预览前五主体不是正式名单 |

T 为每篇各类型 claim 数/总 claim 数，先在论文内归一化后按论文汇总。D 为去重历史邻域社区分布的有效社区数，J 为至少两个非空 claim 邻域之间的平均 Jaccard；没有合格邻居时记缺失。目标仅查询历史背景，不能用同年或后来的论文补邻居，旧百分位不复用。这些是结构画像，不是创新质量评分。

## 结果复用和边界

旧 claim graph 与当前合格论文重叠 **{summary['legacy_qualified_overlap_n']:,} 篇**；其中当前摘要与旧摘要规范化一致 {summary['legacy_exact_abstract_n']:,} 篇，所有旧支持片段仍存在 {summary['legacy_supported_fragments_n']:,} 篇（两数重叠，不能相加）。逐篇见 `current_legacy_reuse.csv`。旧抽取协议的兼容性尚未认证，因此没有把旧大批量贡献与试编码混成同一生产结果。

旧 2023–2025 社区不能替代原方案 2018–2022 的共同背景。当前指定 Fig8 资产中没有已完成的这一背景；本轮遵守不扩采，未启动补建。下一步可先在固定集合上推进统一摘要贡献画像；若坚持完整 D/J 主图，还需要确认可用的既有历史资产及建立参考图，或另行决定是否调整结构实验设计，不能偷偷用目标集代替历史集。

主实验使用英文摘要，**不要求先下载 50 万篇全文**。全文 GEAR 结果与摘要贡献协议不同，不能直接拼接。元数据剩余摘要缺口 812、机构缺口 112,591、国家缺口 30,247，互有重叠；本轮未把缺失当已修复，也未重启下载。非空摘要不等于完整可用，纳入数也不等于贡献完成数。

固定样本有 499,190 篇非空摘要，是统一抽取的输入上限；质量筛查后可能减少。当前 47 条仅属技术试编码，不能宣称这些论文已完成生产分析。

| 对照刊 | 既有候选量 | 固定集合纳入 | 既有待核量（不继续采集） | 纳入篇非空摘要率 | 摘要筛查风险篇数 |
|---|---:|---:|---:|---:|---:|
{comparator_table}

候选不是已知全部研究论文，含排除项和未决项；纳入/候选比不能叫“真实召回率”。尤其 Science、Cell 的已核实子集具有来源可得性选择，正式陈述须限定于固定观察样本。PNAS 的较高纳入比例也不能替代贡献抽取、质量核验和分析可比性。

旧试样中不属于固定集合的记录仅留作证据，本轮不使用，详见 `pilot_reuse_status.csv`。Science、Cell、PNAS 新试样按预先固定种子20261021的DOI哈希排序各取两篇，不按贡献结果筛选；其余试样沿用旧可用性分层样本。**整体并非概率代表样本**。编码由当前助手基于原文进行，不是独立人工金标准；片段/哈希和权重检查只验证来源和算术，不证明科学分类完全正确。

## 可复算文件与下一步

`journal_plot_data.csv`：期刊×领域全样本分母、摘要和年度品牌；`cohort_field_year.csv`：固定集合按在线年份分布。
`current_pilot_inputs.json`：当前试样、摘要、署名和资格证据；`pilot_annotations.json` / `comparator_annotations.json`：复用/新增试编码及来源哈希。
`paper_metrics.csv`、`panel_a_composition.csv`、`panel_b_composition.csv`、`allocation_weights.csv`、`country_university_metrics.csv`：实际绘图输入。
`pilot_validation.json`：核查结果与明确缺失；`current_reuse_summary.json`：复用数量。

后续优先顺序：①固定集合摘要质量筛查并确定唯一生产抽取协议；②复用已认证贡献或执行统一抽取，先出逐刊T画像；③复用现有署名做全样本国家/高校分数汇总及缺失敏感性；④有合格历史资产时才计算D/J；⑤最后一次统一调整、区间和正式绘图。不会因指标缺失而恢复全库采集。

从项目根目录离线重绘：
```bash
python3 -m experiments.fig8_extension.fig8_redraw render
```
用已保存试样重算贡献与机构分配并绘图：
```bash
python3 -m experiments.fig8_extension.fig8_redraw analyze
```
`prepare` 只读取现有主表并重建小样本输入与复用核查，不采集；只有主表或样本选择发生变化时才需要运行。
'''
    (OUT / 'README.md').write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'analyze', 'render'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'analyze':
        analyze()
    else:
        render()


if __name__ == '__main__':
    main()

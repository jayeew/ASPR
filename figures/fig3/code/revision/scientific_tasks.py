"""Source-preserving tasks for the current GEAR/Graph/Full study."""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from pydantic import Field, create_model

from . import models as m
from .materials import batches, blocks, catalog, encoded, packet, terms


def effective_increment(reference: dict[str, Any], prediction: dict[str, Any], verdict: dict[str, Any]) -> bool | None:
    if reference['state'] == 'insufficient_material':
        return None
    return bool(reference['state'] in {'supported_difference', 'bounded_increment'}
                and prediction['asserts_increment'] and verdict['difference_correct'] is True
                and verdict['scope_correct'] is True and verdict['material_overclaim'] is False)


def original_quote(quote: str, text: str) -> str:
    if quote and quote in text:
        return quote
    if '\\n' in quote:
        try:
            return original_quote(quote.replace('\\n', '\n'), text)
        except ValueError:
            pass
    unwrapped = re.sub(r'([A-Za-z]{3,})-[ \t]*\r?\n(?=[a-z])', r'\1', quote)
    if unwrapped != quote:
        try:
            return original_quote(unwrapped, text)
        except ValueError:
            pass
    sentence_start = re.sub(r'^(The|This|These|Those|We|Our)\b', lambda match: match[0].lower(), quote)
    if sentence_start != quote:
        try:
            return original_quote(sentence_start, text)
        except ValueError:
            pass
    pieces = [p.strip() for p in re.split(r'…+|\.{3,}', quote) if p.strip()]
    if len(pieces) > 1 and all(len(p) >= 2 for p in pieces):
        start, end = None, 0
        for piece in pieces:
            try:
                found = original_quote(piece, text[end:])
            except ValueError:
                break
            position = text.index(found, end)
            start = position if start is None else start
            end = position+len(found)
        else:
            return text[start:end]
    words = quote.split()
    match = re.search(r'\s+'.join(re.escape(word) for word in words), text) if words else None
    if match:
        return match.group()
    # PDF ligatures and whitespace are typography; return the actual source span.
    typography = str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"',
                               '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '−': '-'})
    compact, positions = [], []
    for index, char in enumerate(text):
        for expanded in unicodedata.normalize('NFKC', char).translate(typography):
            if not expanded.isspace():
                compact.append(expanded)
                positions.append(index)
    needle = ''.join(c for c in unicodedata.normalize('NFKC', quote).translate(typography) if not c.isspace())
    start = ''.join(compact).find(needle) if needle else -1
    if start >= 0:
        return text[positions[start]:positions[start+len(needle)-1]+1]
    # Quotation delimiters can be omitted by extraction; scientific text must still match.
    stripped = [(char, pos) for char, pos in zip(compact, positions) if char != '"']
    needle = needle.replace('"', '')
    start = ''.join(c for c, _ in stripped).find(needle) if needle else -1
    if start >= 0:
        return text[stripped[start][1]:stripped[start+len(needle)-1][1]+1]
    line_hyphens = {match.start()+len(match[1]) for match in
                    re.finditer(r'([A-Za-z]{3,})-[ \t]*\r?\n(?=[a-z])', text)}
    wrapped = [(char, pos) for char, pos in stripped if pos not in line_hyphens]
    start = ''.join(char for char, _ in wrapped).find(needle) if needle else -1
    if start >= 0:
        return text[wrapped[start][1]:wrapped[start+len(needle)-1][1]+1]
    figure_reference = r'\((?:Supplementary\s+)?Figs?\.[^()]{0,120}\)'
    omitted = {pos for match in re.finditer(figure_reference, text)
               for pos in range(match.start(), match.end())}
    without_refs = [(char, pos) for char, pos in wrapped if pos not in omitted]
    quote_without_refs = re.sub(figure_reference, '', quote)
    ref_needle = ''.join(c for c in unicodedata.normalize('NFKC', quote_without_refs).translate(typography)
                         if not c.isspace() and c != '"')
    start = ''.join(char for char, _ in without_refs).find(ref_needle) if ref_needle else -1
    if start >= 0:
        return text[without_refs[start][1]:without_refs[start+len(ref_needle)-1][1]+1]
    trimmed = quote.rstrip('。.;；,，')
    if trimmed and trimmed != quote:
        return original_quote(trimmed, text)
    raise ValueError('Selected quotation is not a passage of the supplied original text')


async def select_core(stages: Any) -> dict[str, Any]:
    async def candidates(group: list[Any], index: int) -> dict[str, Any]:
        text = '\n'.join(part['text'] for part in group)
        sentences = {f'S{i:04d}': sentence for i, sentence in enumerate(
            (s for s in re.split(r'(?<=[.!?。])\s+(?=[A-Z\u4e00-\u9fff])', text) if s.strip()), 1)}
        value = await stages.ask('core', {'original_sentences': sentences}, m.CoreSentences, f'sentences_{index:04d}',
            'From these original manuscript sentences choose at most two central contribution candidates. '
            'Return their supplied sentence_id plus a concise contribution description and selection reason. '
            'Ignore background, cited prior work and future work. An empty candidate list is allowed. '
            'The source sentence will be copied by the program; do not rewrite it.')
        if any(row['sentence_id'] not in sentences for row in value['items']):
            raise ValueError('Unknown manuscript sentence ID')
        return {'items': [{'core_id': '', 'description': row['description'],
                           'selection_reason': row['selection_reason'],
                           'manuscript_quote': sentences[row['sentence_id']]} for row in value['items']]}

    values = await stages.map('原文核心候选', batches(stages.data['manuscript_blocks'], 2, 6500), candidates)
    pool = {f'K{i:04d}': row for i, row in enumerate((r for v in values for r in v['items']), 1)}
    current = [{'candidate_id': key, **row} for key, row in pool.items()]
    level = 0
    while len(current) > 2:
        groups = batches(current, 10, 7500)

        async def choose(group: list[Any], index: int, level: int = level) -> list[Any]:
            result = await stages.ask('core', {'candidates': group}, m.CoreChoice,
                f'choose_{level}_{index}', 'Select up to TWO most central, distinct scientific contributions '
                'among these manuscript-only candidates. Return their supplied candidate_ids. '
                'Do not select by novelty confidence or availability of historical evidence.')
            ids = set(result['candidate_ids'])
            if not ids <= {r['candidate_id'] for r in group}:
                raise ValueError('Unknown core candidate')
            return [r for r in group if r['candidate_id'] in ids]

        selected = await stages.map(f'原文核心选择_{level}', groups, choose)
        current = [r for group in selected for r in group]
        level += 1
    items = [{**pool[r['candidate_id']], 'core_id': f'C{i:02d}'} for i, r in enumerate(current, 1)]
    selected = {'status': 'selected' if items else 'no_selectable_core', 'items': items,
                'reason': 'Central contributions selected from original manuscript passages.'}
    mapping = await stages.ask('core', {'core': selected, 'claims': stages.get('claims')['claims']},
        m.CoreMapping, 'mapping', 'Map each core to supplied shared claim IDs without changing its definition.') if items else {'items': []}
    return {**selected, 'mapping': mapping['items']}


def claim_sources(stages: Any, claim: dict[str, Any], sources: list[Any]) -> dict[str, Any]:
    graph_blocks = []
    for source in sources:
        if source.get('source_id') == 'GRAPH:'+claim['claim_id']:
            passage = source['passage']
            fact = json.loads(passage)
            intervals = [(0, len(passage))]
            if isinstance(fact.get('claim'), dict):
                target = encoded(fact['claim'])
                begin = passage.index(target)
                intervals = [(0, begin), (begin+len(target), len(passage))]
            # Keep the complete observed neighborhood and its ORIGINAL offsets.
            # The target claim is context, not independent manuscript evidence.
            for begin, end in intervals:
                for part in blocks(passage[begin:end], size=1200):
                    graph_blocks.append({'text': part['text'], 'provenance': [{
                        'source_id': source['source_id'], 'source_type': 'graph_observation',
                        'start': begin+part['start'], 'end': begin+part['end']}]})
    history_blocks = []
    if any(s['source_id'].startswith('GEAR_') for s in sources):
        index = next(i for i, c in enumerate(stages.get('claims')['claims']) if c['claim_id'] == claim['claim_id'])
        for source in sources:
            if source['source_id'].startswith(f'GEAR_{index:02d}_'):
                # Reproduce the original branch's per-work reading, rather than
                # selecting a smaller cross-claim subset at fusion/report time.
                history_blocks.extend(packet(catalog([source], 1500), claim, 3500)['evidence_blocks'])
    manuscript = stages.manuscript_material(claim, 6500)
    evidence = manuscript['evidence_blocks']+history_blocks+graph_blocks
    for block in evidence:
        origin = block['provenance'][0]
        block['block_id'] = f'{origin["source_id"]}@{origin["start"]}:{origin["end"]}'
    return {'evidence_blocks': evidence}


async def reference_core(stages: Any, core: dict[str, Any], index: int) -> dict[str, Any]:
    pool = stages.get('evidence_pool')
    chunks = {b['block_id']: b for b in pool['blocks']}
    query = terms(encoded(core))
    ranked, seen = [], set()
    ordered_sources = sorted(pool['sources'], key=lambda s: s['source_type'] != 'fulltext')
    for source in ordered_sources:
        if source['source_type'] in {'manuscript', 'graph_observation', 'review'}:
            continue
        identity = source.get('doi') or source.get('work_id') or source.get('title') or source['source_id']
        if identity in seen:
            continue
        passages = sorted((chunks[b['block_id']] for b in source['block_ids']),
                          key=lambda b: -len(query & terms(b['text'])))
        if not passages:
            continue  # A title-only retrieval is a lead, not historical scientific content.
        seen.add(identity)
        score = len(query & terms(source.get('title', '')))*3+max(
            (len(query & terms(b['text'])) for b in passages), default=0)
        ranked.append((score, source, passages[:2]))
    ranked.sort(key=lambda r: -r[0])

    async def compare(row: Any, number: int) -> dict[str, Any]:
        _, source, passages = row
        value = await stages.ask('reference', {'core': [core], 'historical_source':
            {k: source.get(k) for k in ('source_id', 'source_type', 'title', 'doi', 'publication_date')},
            'original_historical_passages': passages, **stages.manuscript_material(core, 2500)},
            m.HistoricalComparison, f'compare_{index}_{number}',
            'Compare this manuscript core with this single independent historical work. Determine actual relevance, '
            'common ground and a concrete difference at matching scope. Quote at most two short ORIGINAL source '
            'passages. Label evidence.source_id MANUSCRIPT or HISTORICAL; the program assigns the actual historical '
            'source key. If irrelevant say so. Missing details cannot establish firstness. '
            'Keep the comparison compact, about 500 Chinese characters excluding quotations.')
        value['source_id'] = source['source_id']
        original = '\n'.join(p['text'] for p in passages)
        for evidence_index, evidence in enumerate(value['evidence']):
            if evidence['source_id'] in {'MANUSCRIPT', core['core_id']} or evidence['source_type'].lower() in {
                    'manuscript', 'manuscript_core'}:
                try:
                    evidence['quote'] = original_quote(evidence['quote'], stages.data['manuscript'])
                except ValueError:
                    supplied = core['manuscript_quote']+'\n'+'\n'.join(
                        block['text'] for block in stages.manuscript_material(core, 2500)['evidence_blocks'])
                    located = await locate_quotes(stages, supplied, {'quote': evidence['quote']},
                        f'compare_manuscript_location_{index}_{number}_{evidence_index}')
                    if located['quote'] is None:
                        raise ValueError('Historical comparison cited a manuscript assertion absent from its supplied source')
                    evidence['quote'] = original_quote(located['quote'], stages.data['manuscript'])
                evidence.update(source_id='MANUSCRIPT', source_type='manuscript')
        located = await locate_quotes(stages, original, {str(i): e['quote'] for i, e in enumerate(value['evidence'])
            if e['source_id'] != 'MANUSCRIPT'}, f'compare_locations_{index}_{number}')
        for i, evidence in enumerate(value['evidence']):
            if evidence['source_id'] != 'MANUSCRIPT':
                if located[str(i)] is None:
                    raise ValueError('Historical comparison quotation is absent from its original source')
                evidence['quote'] = located[str(i)]
                evidence['source_id'] = source['source_id']
                evidence['source_type'] = source['source_type']
        return value

    comparisons = await stages.map(f'逐来源历史比较_{index}', ranked[:10], compare)
    relevant = [r for r in comparisons if r['relevant']]
    async def supplement() -> None:
        retrieved = await stages.search(f'core_{core["core_id"]}', {'core': core,
            'instruction': 'Find direct historical scientific comparators for this exact core, including prior work named in its original manuscript context.'})
        extra = catalog(retrieved['sources'], 1500)
        extra_chunks = {b['block_id']: b for b in extra['blocks']}
        readable = [s for s in extra['sources'] if s['block_ids']]
        rows = [(0, s, sorted((extra_chunks[b['block_id']] for b in s['block_ids']),
                    key=lambda b: -len(query & terms(b['text'])))[:2]) for s in readable[:10]]
        # Keep distinct call names for additional, genuinely different source material.
        additions = await stages.map(f'补充历史比较_{index}', rows, lambda row, i: compare(row, i+10))
        comparisons.extend(additions)

    async def judge(step: str) -> dict[str, Any]:
        value = await stages.ask('reference', {'core': core, 'historical_comparisons': comparisons,
            **stages.manuscript_material(core, 3000)}, m.ReferenceItem, step,
            'Return ONE consolidated historical reference for this core across ALL supplied comparisons, '
            'not one judgment per historical work. Require relevant independent historical content '
            'for supported_difference, bounded_increment or substantially_covered. The target manuscript alone cannot '
            'establish a historical difference. Irrelevant comparisons or absent comparator details mean insufficient_material. '
            'Distinguish reported target facts from established prior facts. Use original quotations and source IDs from '
            'the individual comparisons, never quote a model comparison as source text. No firstness from absence.')
        if value['core_id'] != core['core_id']:
            raise ValueError('Historical reference did not return the supplied core')
        historical_ids = {r['source_id'] for r in comparisons if r['relevant'] and any(
            e['source_id'] != 'MANUSCRIPT' for e in r['evidence'])}
        if not any(e['source_id'] in historical_ids for e in value['evidence']):
            value['state'] = 'insufficient_material'
            value['limitations'].append('No relevant independent historical quotation supports a determinate comparison.')
        return {'items': [value]}

    if not relevant:
        await supplement()
    result = await judge(f'reference_{index:05d}')
    if relevant and result['items'][0]['state'] == 'insufficient_material':
        await supplement()
        result = await judge(f'reference_completed_{index:05d}')
    return result


async def fuse_claims(stages: Any, gear: dict[str, Any], graph: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    from gear.innovation.analysis import COMMON, FUSION
    from gear.innovation.contracts import Assessment

    gear_rows = {r['claim_id']: r for r in gear['analyses']}
    graph_rows = {r['claim_id']: r for r in graph['analyses']}

    async def one(claim: dict[str, Any], index: int) -> dict[str, Any]:
        ident = claim['claim_id']
        branches = {'gear': gear_rows[ident], 'graph': graph_rows[ident]}
        findings = [{'finding_key': f'{name}:{i}', **finding} for name, row in branches.items()
                    for i, finding in enumerate(row['findings'])]
        evidence = claim_sources(stages, claim, gear['sources']+graph['sources'])
        decisions = []
        for number, group in enumerate(batches(findings, 4, 4000)):
            result = await stages.ask('baseline', {'claim_id': ident, 'claim_text': claim['normalized_claim_text'],
                'findings': group, **evidence}, m.FusionDecisions, f'finding_decisions_{index}_{number}',
                FUSION+' For EACH supplied finding_key choose retain/merge/correct/omit/unresolved. '
                'Use original evidence where available; unavailable verification is unresolved, not refutation. '
                'Do not drop a distinct relevant supported finding to save space. Explain decisions with source keys.')
            expected = {r['finding_key'] for r in group}
            by_key = {r['finding_key']: r for r in result['decisions'] if r['finding_key'] in expected}
            if set(by_key) != expected:
                raise ValueError('Fusion decision omitted a supplied finding')
            decisions.extend(by_key.values())
        result = await stages.ask('baseline', {'claim_id': ident, 'claim_text': claim['normalized_claim_text'],
            'gear': branches['gear'], 'graph': branches['graph'], 'decisions': decisions, **evidence},
            Assessment, f'fuse_{index:05d}', COMMON+'\n'+FUSION, True)
        result.update(claim_id=ident, claim_text=claim['normalized_claim_text'])
        retained = {d['finding_key'] for d in decisions if d['action'] == 'retain'}
        present = {encoded(f) for f in result['findings']}
        for finding in findings:
            original = {k: v for k, v in finding.items() if k != 'finding_key'}
            if finding['finding_key'] in retained and encoded(original) not in present:
                result['findings'].append(original)
                present.add(encoded(original))
        return {'assessment': result, 'claim_id': ident, 'decisions': decisions}

    values = await stages.map('贡献融合', stages.get('claims')['claims'], one)
    return [v['assessment'] for v in values], [{'claim_id': v['claim_id'], 'decisions': v['decisions']} for v in values]


async def write_scientific_report(stages: Any, method: str, analysis: dict[str, Any], sources: list[Any]) -> dict[str, Any]:
    claims = {r['claim_id']: r for r in stages.get('claims')['claims']}
    full = method == 'full'
    instruction = ('Write a Chinese scientific analysis section for this contribution using the supplied judgment '
        'and original evidence. Preserve concrete findings, historical differences, scope and usable source IDs. '
        'Every substantive factual/comparative statement needs its actual source locator: use the supplied '
        'source_id@start:end, never a local T-number copied from an intermediate assessment. No new scientific claims, '
        'no system name, no novelty score. Generic caution must not replace specific analysis. ')
    instruction += ('There is NO word or character target. Include all distinct supported findings and necessary '
                    'qualifications; remove only repetition.' if full else
                    f'Keep this section approximately {max(130, 1450//max(1, len(analysis["claims"])))} Chinese characters.')

    async def section(row: dict[str, Any], index: int) -> dict[str, Any]:
        claim = claims[row['claim_id']]
        payload = {'claim_id': row['claim_id'], 'claim_text': claim['normalized_claim_text'],
                   'assessment': row, **claim_sources(stages, claim, sources)}
        prompt = instruction
        if full:
            findings = [{'finding_key': f'{method}:{i}', **finding} for method in ('gear', 'graph')
                        for branch in stages.report(method)['analyses'] if branch['claim_id'] == row['claim_id']
                        for i, finding in enumerate(branch['findings'])]
            payload.update(findings=findings, decisions=next(v['decisions'] for v in analysis['fusion_decisions']
                                                            if v['claim_id'] == row['claim_id']))
            prompt += (' For EACH original finding_key also record its final report outcome, a short verbatim '
                       'report_quote, and the scientific reason. Merged findings can share a quote. If omitted, '
                       'leave the quote empty and explain the specific reason; length is not a reason. '
                       'This mapping is metadata, not prose to add to the report.')
        result = await stages.ask('baseline', payload, m.FullReport if full else m.Report,
                                  f'writer_claim_{index:03d}', prompt, True)
        if full:
            expected = {v['finding_key'] for v in findings}
            if {v['finding_key'] for v in result['finding_outcomes']} != expected:
                raise ValueError('Final report omitted a branch-finding outcome')
            located = await locate_quotes(stages, result['body'], {
                row['finding_key']: row['report_quote'] for row in result['finding_outcomes']
                if row['report_quote']}, f'writer_locations_{index:03d}')
            missing = {key for key, quote in located.items() if quote is None}
            repaired = await repair_report_outcomes(stages, result['body'],
                [f for f in findings if f['finding_key'] in missing], index) if missing else {}
            for outcome in result['finding_outcomes']:
                if outcome['finding_key'] in repaired:
                    outcome.update(repaired[outcome['finding_key']])
                    continue
                if outcome['report_quote']:
                    outcome['report_quote'] = located[outcome['finding_key']]
        return result

    sections = await stages.map('贡献报告原文', analysis['claims'], section)
    if analysis.get('joint_graph'):
        joint = await stages.ask('baseline', {'contribution_sections': sections, 'joint_graph': analysis['joint_graph']},
            m.Report, 'writer_joint', 'Write a Chinese whole-paper synthesis using the supplied contribution sections '
            'and joint graph observations. Explain evidenced complementarity, shared limitations and unresolved '
            'conflicts. Do not invent target-to-target relationships or repeat the contribution sections. '
            'Preserve source locators. '+('No length target; retain all necessary joint insights.' if full else
                                         'Aim for approximately 300 Chinese characters.'), True)
        sections.append(joint)
    result = {'body': '\n\n'.join(row['body'] for row in sections),
              'cited_source_ids': list(dict.fromkeys(k for row in sections for k in row['cited_source_ids']))}
    if full:
        result['finding_retention'] = [{'claim_id': row['claim_id'], 'outcomes': section['finding_outcomes']}
                                       for row, section in zip(analysis['claims'], sections)]
    return result


async def repair_report_outcomes(stages: Any, report: str, findings: list[Any], index: int) -> dict[str, Any]:
    clauses = {f'S{i:04d}': match for i, match in enumerate(
        re.finditer(r'[^。；\n!?]+[。；\n!?]*', report), 1)}
    result = await stages.ask('extract', {'original_report_clauses': {k: v.group() for k, v in clauses.items()},
        'original_branch_findings': findings}, m.ReportFindingLocations, f'writer_outcomes_{index:03d}',
        'original_report_clauses contains the actual final report, split into verbatim numbered clauses. '
        'Trace EACH supplied original branch finding to that report. Earlier writer metadata '
        'could not be located and is not evidence of retention. Return retained, merged, corrected '
        '(explicitly narrowed or corrected), unresolved (report explicitly leaves it unresolved), or omitted. '
        'Select first_clause_id and last_clause_id enclosing the actual report discussion, including its scope. '
        'A span may encompass several intervening clauses when discussion is distributed. Omitted requires '
        'empty clause IDs and an honest reason: if the report gives none, say no omission reason is stated. '
        'Do not quote original evidence or branch text as report content. Program copies report text itself.')
    rows = {row['finding_key']: row for row in result['outcomes']}
    if set(rows) != {f['finding_key'] for f in findings}:
        raise ValueError('Report outcome mapping omitted a supplied finding')
    output = {}
    for key, row in rows.items():
        quote = ''
        if row['action'] != 'omitted':
            if row['first_clause_id'] not in clauses or row['last_clause_id'] not in clauses:
                raise ValueError('Report outcome mapping lacks the actual report clause for its stated action')
            first, last = clauses[row['first_clause_id']], clauses[row['last_clause_id']]
            if first.start() > last.start():
                raise ValueError('Report outcome mapping reversed its source span')
            quote = report[first.start():last.end()]
        output[key] = {'finding_key': key, 'action': row['action'], 'report_quote': quote, 'reason': row['reason']}
    return output


async def extract_report(stages: Any, method: str) -> dict[str, Any]:
    report = stages.report(method)
    claims = [{k: row[k] for k in ('claim_id', 'normalized_claim_text')} for row in stages.get('claims')['claims']]
    core = stages.get('core')['items']

    async def part(row: dict[str, Any], index: int) -> dict[str, Any]:
        value = await stages.ask('extract', {'report': row['text'], 'original_references': report['cited_source_ids'],
            'claims': claims, 'core': [{k: c[k] for k in ('core_id', 'description')} for c in core]},
            m.Units, f'extract_part_{index}',
            'Extract atomic scientific assertions VERBATIM from this original report passage. Include concrete '
            'limitations/scope corrections as substantive assertions needing verification, as well as numerical, '
            'mechanistic, historical and structural facts. Do not emit transitions or generic empty caution as '
            'scientific units. Every emitted scientific assertion needs verification, including manuscript paraphrases. '
            'Preserve usable original locators. Paraphrase means merely repeating manuscript facts '
            'without analysis. Map supplied claim IDs only. For each core record what THIS PASSAGE explicitly says '
            'about increment, preserving conflicting quotes; do not infer from scientific background or reference labels.')
        located = await locate_quotes(stages, row['text'], {str(i): u['quote'] for i, u in enumerate(value['units'])},
                                      f'extract_locations_{index}')
        value['unlocated_candidates'] = sum(quote is None for quote in located.values())
        value['units'] = [{**u, 'quote': located[str(i)]} for i, u in enumerate(value['units']) if located[str(i)] is not None]
        for unit in value['units']:
            unit['needs_verification'] = True
        return value

    values = await stages.map('报告原句抽取', blocks(report['body'], size=6000), part)
    units, seen = [], set()
    for value in values:
        for unit in value['units']:
            if unit['quote'] not in seen:
                seen.add(unit['quote'])
                units.append({**unit, 'unit_id': f'U{len(units)+1:04d}'})

    async def prediction(item: dict[str, Any], index: int) -> dict[str, Any]:
        subject = {key: item[key] for key in ('core_id', 'description')}
        result = await stages.ask('extract', {'core_topic_only': subject, 'report': report['body']}, m.Predictions,
            f'prediction_{index}', 'Return exactly one prediction for the supplied core using the ORIGINAL report. '
            'positive_increment/limited_increment require an actually stated concrete increment (qualified is allowed). '
            'Explicitly withholding a historical judgment is explicit_abstention, not limited_increment. '
            'asserts_increment is true for an explicitly claimed concrete increment or unqualified firstness; '
            'false for mere potential, author-claim repetition or abstention. The core description identifies '
            'the topic ONLY: it is not evidence of what the report says. Distinguish the report endorsing a '
            'historical difference from attributing a performance number, firstness or system description to '
            'the manuscript authors. Author attribution plus an unresolved historical comparison does not '
            'endorse that comparison. All quotes must occur in report, never in core_topic_only. '
            'Read the complete report and preserve conflicting original quotes.')
        found = [r for r in result['predictions'] if r['core_id'] == item['core_id']]
        if len(found) != 1:
            raise ValueError('Core prediction missing')
        requested = {f'{field}/{i}': q for field in ('quotes', 'conflicting_quotes')
                     for i, q in enumerate(found[0][field])}
        located = await locate_quotes(stages, report['body'], requested, f'prediction_locations_{index}')
        if any(quote is None for quote in located.values()):
            result = await stages.ask('extract', {'core_topic_only': subject, 'report': report['body'],
                'unsupported_previous_prediction': found[0]}, m.Predictions, f'prediction_source_repair_{index}',
                'Reclassify this core from the original report: previous prediction quoted content not present '
                'in report. Do not preserve its label by default. Core topic is not report evidence. '
                'Mere author attribution is not endorsement; historical judgment withheld is explicit_abstention. '
                'Return one prediction with only exact report quotes, or no quotes when absent. '
                'Set asserts_increment true only when the report itself endorses a concrete historical increment.')
            found = [r for r in result['predictions'] if r['core_id'] == item['core_id']]
            if len(found) != 1:
                raise ValueError('Repaired core prediction missing')
            requested = {f'{field}/{i}': q for field in ('quotes', 'conflicting_quotes')
                         for i, q in enumerate(found[0][field])}
            located = await locate_quotes(stages, report['body'], requested,
                                          f'prediction_repaired_locations_{index}')
            if any(quote is None for quote in located.values()):
                raise ValueError('Repaired core prediction cites content absent from the report')
            for field in ('quotes', 'conflicting_quotes'):
                found[0][field] = [located[f'{field}/{i}'] for i in range(len(found[0][field]))]
            return found[0]
        for field in ('quotes', 'conflicting_quotes'):
            found[0][field] = [located[f'{field}/{i}'] for i in range(len(found[0][field]))
                               if located[f'{field}/{i}'] is not None]
        return found[0]

    predictions = await stages.map('报告核心预测', core, prediction)
    return {'units': units, 'predictions': predictions,
            'unlocated_candidates': sum(v.get('unlocated_candidates', 0) for v in values)}


async def locate_quotes(stages: Any, text: str, quotes: dict[str, str], step: str) -> dict[str, str | None]:
    located, missing = {}, {}
    for key, quote in quotes.items():
        try:
            located[key] = original_quote(quote, text)
        except ValueError:
            missing[key] = quote
    if not missing:
        return located
    matches = list(re.finditer(r'[^。；，,\n!?]+[。；，,\n!?]*', text))
    clauses = {f'S{i:04d}': match for i, match in enumerate(matches, 1)}
    answer = await stages.ask('extract', {'original_source_clauses': {k: v.group() for k, v in clauses.items()},
        'candidate_quotes': missing}, m.QuoteSpans, step,
        'Locate EACH candidate quote in the ORIGINAL source. A candidate may be an atomic paraphrase, '
        'contain omissions or have transcription mistakes. Select first_clause_id and last_clause_id for the '
        'smallest contiguous source passage expressing that SAME complete assertion, including necessary scope. '
        'The program copies the source verbatim. If the source does not express it, set present_in_source=false '
        'and leave both clause IDs empty. Do not invent source content or judge scientific correctness. '
        'Return each supplied quote_key exactly once; keep the reason short.')
    found = {r['quote_key']: r for r in answer['spans'] if r['quote_key'] in missing and (
        not r['present_in_source'] or r['first_clause_id'] in clauses and r['last_clause_id'] in clauses)}
    for index, key in enumerate(sorted(set(missing)-set(found))):
        item = create_model('RequestedQuoteSpan', __base__=m.QuoteSpan,
                            quote_key=(str, Field(json_schema_extra={'enum': [key]})),
                            first_clause_id=(str, Field(json_schema_extra={'enum': ['', *clauses]})),
                            last_clause_id=(str, Field(json_schema_extra={'enum': ['', *clauses]})))
        schema = create_model('OneQuoteSpan', __base__=m.QuoteSpans,
                              spans=(list[item], Field(min_length=1, max_length=1)))
        extra = await stages.ask('extract', {
            'original_source_clauses': {k: v.group() for k, v in clauses.items()},
            'candidate_quotes': {key: missing[key]}}, schema, f'{step}_missing_{index}',
            'Locate this one omitted object in the original source. Select the smallest contiguous '
            'clause range expressing the same complete assertion and necessary scope. The program '
            'copies the source verbatim. If it is absent, set present_in_source=false and leave '
            'both clause IDs empty. Do not invent content or infer scientific correctness.')
        if extra['spans'][0]['quote_key'] != key:
            raise ValueError('Source quote recovery returned a different object')
        found[key] = extra['spans'][0]
    for key, row in found.items():
        if not row['present_in_source']:
            located[key] = None
            continue
        first, last = clauses[row['first_clause_id']], clauses[row['last_clause_id']]
        if last.end() < first.start():
            raise ValueError('Source quote lookup reversed its source span')
        located[key] = text[first.start():last.end()]
    return located


async def quality_report(stages: Any, method: str) -> dict[str, Any]:
    report = stages.report(method)['body']
    checks = stages.get('quality_checklist')
    value = await stages.ask('quality', {'report': report, 'checklist': checks,
        **stages.material(stages.get('core'), 6000)}, m.Quality, 'quality_original',
        'Score exactly the five supplied dimensions using the ORIGINAL report, not checklist/background wording. '
        '0=absent/materially incorrect; 1=generic or major substantive gaps; 2=specific and largely correct '
        'but bounded gaps; 3=specific, complete for applicable core issues and source-grounded. Generic caution '
        'does not earn scope_uncertainty=3: necessary contribution-specific limitations must be stated. '
        'Every report_quote must be copied verbatim from report. Cite original evidence separately. '
        'Explain concrete strengths and deficiencies, not fluency or report length.', True)
    located = await locate_quotes(stages, report, {
        f'{i}/{j}': quote for i, dimension in enumerate(value['dimensions'])
        for j, quote in enumerate(dimension['report_quotes'])}, 'quality_report_locations')
    for i, dimension in enumerate(value['dimensions']):
        quotes = [located[f'{i}/{j}'] for j in range(len(dimension['report_quotes']))]
        if any(quote is None for quote in quotes):
            schema = create_model('CorrectedQualityScore', __base__=m.QualityScore,
                dimension=(str, Field(json_schema_extra={'enum': [dimension['dimension']]})))
            corrected = await stages.ask('quality', {'report': report,
                'checklist': [check for check in checks['dimensions']
                              if check['dimension'] == dimension['dimension']],
                'previous_assessment': dimension,
                'absent_report_quotes': [q for q, found in zip(dimension['report_quotes'], quotes) if found is None],
                **stages.material(stages.get('core'), 6000)}, schema, f'quality_dimension_source_repair_{i}',
                'Reassess ONLY this dimension from the ORIGINAL report. The previous assessment cited '
                'content absent from that report; do not attribute checklist or background content to it. '
                'Reconsider the score and reason, not just the quotation. Score 0=absent/materially incorrect; '
                '1=generic or major gaps; 2=specific and largely correct with bounded gaps; '
                '3=specific, complete for applicable core issues and source-grounded. '
                'Copy short report quotations exactly; use an empty list when the feature is absent. '
                'Keep external evidence separate. Do not favor any method.', True)
            recovered = await locate_quotes(stages, report,
                {str(j): q for j, q in enumerate(corrected['report_quotes'])}, f'quality_corrected_locations_{i}')
            quotes = [recovered[str(j)] for j in range(len(corrected['report_quotes']))]
            if any(quote is None for quote in quotes):
                raise ValueError('Corrected quality assessment still cited content absent from the report')
            dimension.update(corrected)
        dimension['report_quotes'] = quotes
    return value


async def review_dynamics(stages: Any) -> dict[str, Any]:
    from .recheck import unique_rows
    sections = stages.get('review_sections')['sections']
    concerns = stages.get('checklist')['concerns']
    completed, jobs = {}, []
    for concern in concerns:
        original = {'concern_id': concern['concern_id'], 'reviewer_id': concern['reviewer_id'],
                    'earlier_round': concern['round_number'], 'earlier_stance': concern['stance']}
        later = [s for s in sections if s['role'] == 'reviewer' and s['identity_explicit']
                 and s['reviewer_id'] == concern['reviewer_id'] and s['round_number'] > concern['round_number'] > 0]
        if not later:
            completed[concern['concern_id']] = {**original, 'later_round': None, 'later_stance': '',
                'author_response_quote': '', 'later_reviewer_quote': '', 'same_reviewer_explicit': False,
                'resolution': 'no_explicit_followup', 'reason': 'No explicit later round by the same identified reviewer.'}
            continue
        query = terms(concern['object_description']+' '+concern['quote'])
        ranked = sorted(later, key=lambda s: (-len(query & terms(s['quote'])), s['round_number']))[:2]
        replies = sorted((s for s in sections if s['role'] == 'author'),
                         key=lambda s: -len(query & terms(s['quote'])))[:1]
        jobs.append({'concern': concern, 'original': original, 'later': ranked, 'replies': replies})

    prompt = ('Trace EACH supplied concern using its permitted original later reviewer sections. '
              'Explicit resolution requires a later same-reviewer statement about this specific issue. '
              'Author self-claims, general thanks and overall satisfaction cannot establish resolution. '
              'Return exactly one change per supplied concern_id; no followup is valid when these passages '
              'do not address that issue. Author quotations are context, not proof of resolution or timing.')

    def payload(group: list[dict[str, Any]]) -> dict[str, Any]:
        source = {s['section_id']: s for job in group for s in job['later']+job['replies']}
        return {'concerns': [job['concern'] for job in group], 'sections': list(source.values()),
                'allowed_followup_sections': {job['concern']['concern_id']: [s['section_id'] for s in job['later']]
                                               for job in group}}

    async def batch(group: list[dict[str, Any]], index: int) -> None:
        result = await stages.ask('review_dynamics', payload(group), m.ReviewDynamics,
                                  f'followup_batch_{index:04d}', prompt)
        expected = {job['concern']['concern_id'] for job in group}
        found = unique_rows(result['changes'], 'concern_id', expected)
        for job in group:
            ident = job['concern']['concern_id']
            if ident not in found:
                result = await stages.ask('review_dynamics', payload([job]), m.ReviewDynamics,
                                          'followup_missing_'+ident, prompt)
                found.update(unique_rows(result['changes'], 'concern_id', {ident}))
            if ident not in found:
                raise ValueError('Reviewer followup omitted supplied concern: '+ident)
            row = {**found[ident], **job['original']}
            matched = []
            for section in job['later']:
                try:
                    quote = original_quote(row['later_reviewer_quote'], section['quote'])
                except ValueError:
                    continue
                matched.append((section, quote))
            rounds = {section['round_number'] for section, _ in matched}
            if len(rounds) == 1:
                row.update(later_round=next(iter(rounds)), later_reviewer_quote=matched[0][1],
                           same_reviewer_explicit=True)
            else:
                row.update(later_round=None, later_stance='', later_reviewer_quote='',
                           resolution='no_explicit_followup', same_reviewer_explicit=False)
            if row['author_response_quote']:
                try:
                    row['author_response_quote'] = original_quote(row['author_response_quote'],
                        '\n'.join(s['quote'] for s in job['replies']))
                except ValueError:
                    row['author_response_quote'] = ''
            completed[ident] = row
    await stages.map('审稿问题随访', batches(jobs, 3, 45000), batch)
    return {'changes': [completed[c['concern_id']] for c in concerns]}


def single_error_schema(key: str) -> Any:
    row = create_model('RequestedBranchError', __base__=m.ErrorFlow,
                       branch_unit_key=(str, Field(json_schema_extra={'enum': [key]})))
    return create_model('SingleBranchErrorJudgment', __base__=m.FusionJudgment,
                        errors=(list[row], Field(min_length=1, max_length=1)))


async def reassess_fusion_error(stages: Any, key: str, unit: dict[str, Any], support: dict[str, Any],
                               group: list[Any], previous: dict[str, Any], index: int, number: int) -> dict[str, Any]:
    answer = await stages.ask('fusion', {'errors': [{'unit_key': key, 'unit': unit, 'support': support}],
        'full_original_units': group, 'previous_judgment': previous}, single_error_schema(key),
        f'error_source_repair_{index}_{number}',
        'Reassess ONLY the supplied erroneous branch assertion against the original Full units. '
        'The previous Full quotation was absent from the supplied report. Reassess the status and reason, '
        'not just the quotation. A paraphrase of the branch error is not an explicit correction. '
        'Explicit correction requires an actual Full statement correcting that specific error; silence '
        'is not correction. Propagation also requires an actual Full assertion carrying the same specific '
        'error: failure to correct or mention it does not mean propagation. Use not_propagated for absence '
        'in these supplied passages, or unknown when correspondence cannot be determined. '
        'Quote one short contiguous original Full passage, or leave the quote empty '
        'when no matching statement exists. Return exactly one errors record for the supplied branch key. '
        'Use only supplied fusion/unit keys. newly_introduced_error_keys must be empty.')
    matched = [r for r in answer['errors'] if r['branch_unit_key'] == key]
    if len(matched) != 1:
        raise ValueError('Reassessed fusion judgment omitted its input object')
    value = matched[0]
    allowed = {u['unit_key'] for u in group}
    value['full_unit_keys'] = [k for k in value['full_unit_keys'] if k in allowed]
    if value['status'] in {'propagated', 'explicitly_corrected'} and not value['full_quote']:
        raise ValueError('Reassessed fusion judgment lacks its required Full statement')
    if value['full_quote']:
        located = await locate_quotes(stages, '\n'.join(u['quote'] for u in group),
            {'quote': value['full_quote']}, f'error_repaired_locations_{index}_{number}')
        if located['quote'] is None:
            raise ValueError('Reassessed fusion quotation is absent from the Full report')
        value['full_quote'] = located['quote']
    return value


async def fusion_errors(stages: Any) -> dict[str, Any]:
    extracted = {method: {u['unit_id']: u for u in stages.get('extract', method)['units']}
                 for method in ('gear', 'graph', 'fusion')}
    errors = {method: [r for r in stages.get('support', method)['units'] if r['errors'] or r['support'] == 'contradicted']
              for method in extracted}
    full_units = [{'unit_key': 'fusion/'+u['unit_id'], 'quote': u['quote']} for u in extracted['fusion'].values()]

    async def branch_error(item: dict[str, Any], index: int) -> dict[str, Any]:
        row, method = item['support'], item['method']
        key = method+'/'+row['unit_id']
        unit = extracted[method][row['unit_id']]
        candidates = []
        for number, group in enumerate(batches(full_units, 1000, 40000)):
            answer = await stages.ask('fusion', {'errors': [{'unit_key': key, 'unit': unit, 'support': row}],
                'full_original_units': group}, single_error_schema(key), f'error_{index}_{number}',
                'Evaluate ONLY the one supplied erroneous branch assertion against these original Full units. '
                'Return one errors record for that branch_unit_key. Explicit correction needs a quoted Full statement '
                'actually correcting the specific error; silence is not correction. This is a partial report: absence '
                'means not_propagated within these passages only. Propagation requires an actual Full assertion '
                'carrying the same specific error, not merely failure to correct it. '
                'Do not reverse direction or return additional errors. '
                'Use only supplied fusion/unit keys. newly_introduced_error_keys must be empty here.')
            matched = list({encoded(r): r for r in answer['errors'] if r['branch_unit_key'] == key}.values())
            if len(matched) != 1:
                raise ValueError('Fusion error judgment omitted its input object')
            value = matched[0]
            allowed = {u['unit_key'] for u in group}
            value['full_unit_keys'] = [k for k in value['full_unit_keys'] if k in allowed]
            if value['full_quote']:
                original = '\n'.join(u['quote'] for u in group)
                located = await locate_quotes(stages, original, {'quote': value['full_quote']},
                                              f'error_locations_{index}_{number}')
                if located['quote'] is None:
                    value = await reassess_fusion_error(stages, key, unit, row, group, value, index, number)
                else:
                    value['full_quote'] = located['quote']
            elif value['status'] in {'propagated', 'explicitly_corrected'}:
                value = await reassess_fusion_error(stages, key, unit, row, group, value, index, number)
            candidates.append(value)
        # Propagation anywhere remains propagation, even when another passage corrects it.
        priority = {'propagated': 0, 'explicitly_corrected': 1, 'unknown': 2, 'not_propagated': 3}
        return min(candidates, key=lambda r: priority[r['status']]) if candidates else {
            'branch_unit_key': key, 'status': 'unknown', 'full_unit_keys': [], 'full_quote': '',
            'evidence': [], 'reason': 'Full has no extracted scientific assertions.'}

    work = [{'method': method, 'support': row} for method in ('gear', 'graph') for row in errors[method]]
    judged = await stages.map('分支错误去向', work, branch_error)

    async def new_error(row: dict[str, Any], index: int) -> str | None:
        unit = extracted['fusion'][row['unit_id']]
        branch_units = [{'unit_key': method+'/'+u['unit_id'], 'quote': u['quote']}
                        for method in ('gear', 'graph') for u in extracted[method].values()]
        results = []
        for number, group in enumerate(batches(branch_units, 1000, 40000)):
            value = await stages.ask('fusion', {'errors': [{'unit': unit, 'support': row}],
                'branch_original_units': group}, m.NewError, f'new_error_{index}_{number}',
                'Does this Full error already occur in these original GEAR/Graph passages? '
                'newly_introduced=false if the same scientific error is present; quote its actual branch text. '
                'Compare scientific meaning and scope, not wording. Assess only the supplied Full error.')
            results.append(value['newly_introduced'])
        return 'fusion/'+row['unit_id'] if results and all(results) else None

    added = await stages.map('Full新增错误', errors['fusion'], new_error)
    return {'errors': judged, 'newly_introduced_error_keys': [k for k in added if k],
            'reason': 'One outcome per existing branch error; Full errors evaluated separately.'}

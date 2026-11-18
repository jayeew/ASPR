from __future__ import annotations

import asyncio
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_capabilities.materials import payload as previous_payload
from figure_pipeline.fig4_rerun.config import CONDITIONS, NAMES, Report

from .models import Analysis, Config
from .runtime import Runner

GEAR_PROMPT = '''Perform the GEAR scientific-evidence interpretation for this paper, independently of any
graph interpretation. You have the manuscript, neutral contribution identities and existing retrieved
original historical passages. Work in Chinese. Check manuscript support, concrete prior-art overlap,
residual scientific difference and necessary scope. Use actual original source statements, not merely
titles or the target author's characterization of the literature. Explicitly distinguish same object and
result, partial overlap, parallel mechanisms, different experimental settings and wrong source attribution.
Preserve counterevidence and unresolved limits. Missing prior art is not proof of firstness. Do not label
the target's bibliography as independent full-text verification. Source IDs can have GEAR/HISTORY/EACL
aliases; their original text and provenance matter. A source that does not support one comparison need
not be globally irrelevant. Cover the principal claims without inventing extra retrieval or controls.
Return evidence-grounded findings, with observed source content separated from your interpretation.
Use historical_verification or knowledge_position as appropriate. Do not invent graph observations,
joint topology, community/structure values or citation paths; none are provided to this branch.'''

GRAPH_PROMPT = '''Interpret the supplied historical Claim Graph for this paper, independently of GEAR.
Work in Chinese. Connect actual scientific contribution content to concrete relevant historical neighbor
content, separating related object/result, similar methods and parallel mechanisms. Explain useful
patterns of knowledge position, complementary/shared historical bases across contributions, quantitative
neighborhood structure, and citation contact types WHEN supported by the supplied facts. In each finding
separate recorded observation from bounded scientific interpretation, cite exact graph fields/identities
and original passages where scientific content is asserted, and retain important limitations.
Do not just list metric values or repeat generic cautions. Explain how the observed pattern changes a
scientist's understanding of the contribution and its historical setting. Communities are clusters, not
verified disciplines; local connectivity is not impact; citation paths are not support, derivation or
priority; lack of recorded paths is not proof of global absence. Distinguish chemical pathways from paths
between parent papers. Do not infer scientific novelty from a graph metric.
Use only this actual view. Some facts may be absent. Do not fill a missing module with an extra text
analysis or imagined facts. Correct deductions from visible identities/edges are allowed if their basis
and derivation are explicit; do not assert omitted numerical values without a valid derivation. A common
source visible in two claim cards can be discussed, but is not automatically the complete joint topology.
Do not sum single-insertion changes to claim a joint effect. Separate observed and inferred quantities.
Return evidence-grounded findings for knowledge_position, joint_contribution, structural_resolution or
citation_contact. No GEAR judgments, old graph summaries, gold answers or candidate reports are supplied.'''

GRAPH_DEFINITIONS = '''Metric meanings in the existing implementation: effective community count is
1/sum(p_i^2), where p_i is the normalized sum of nonnegative neighbor cosine weights in each known
community; unknown communities are excluded from that normalization. Neighbor density is existing
edges divided by n choose 2. In a single insertion, component_merge_count is the number of historical
components minus one; newly connected pairs are n choose 2 minus the sum of within-component pairs.
Joint values use the union graph and actual insertion edges, not sums of single-card values. Citation
direct/two-hop/shared-reference fields describe parent-paper contacts. Shared references mean a common
bibliography entry, not necessarily a common citing paper. A count without witness IDs cannot identify
the intermediate paper. These are definitions only; compute or interpret values only from visible facts.'''

WRITER_PROMPT = '''Write a Chinese scientific innovation analysis using only your actual supplied materials.
Use these common sections: 主要科学贡献; 历史证据与剩余增量; 知识关系与多贡献组合; 结构和引用联系提供的解释; 结论与适用范围.
No length target. Preserve concrete supported findings and meaningful limits rather than generic praise.
Address the manuscript's principal contributions and explain which existing scientific results overlap,
what differs, and why the evidence supports that distinction. Distinguish the target manuscript's account
of prior work from independent original historical evidence actually available. Cite source/block IDs or
manuscript reference numbers/page locations precisely, and state when only the manuscript is available.
Explain the knowledge relationships, multi-contribution connections, quantitative differences in the
historical knowledge setting and specific citation-contact distinctions that the materials support.
Scientific interpretation matters: bare metric recitation or saying 'it is not causal' is insufficient.
Keep observations separate from interpretation and preserve the scope of each comparison. The conclusion
should bring the scientific increment and knowledge-position explanation together where evidence permits.
Analyst findings are fallible. Original passages and native facts take precedence; handle real conflicts
instead of blindly combining claims. Correct reasoning from visible raw facts is allowed if its basis is
clear. Never invent missing numbers, references or a missing module. A brief accurate limitation is useful
when evidence is absent. Do not compare the system configurations or mention experiment scores.
Graph proximity/community/connectivity/path facts are not proof of novelty, support, priority or causality.
Return the final body and cited_source_ids, with no hidden intermediary task or score.'''


def eligible(config: Config) -> list[str]:
    ids = read(config.source / 'pilot.json')['paper_ids']
    if config.paper_ids is not None:
        ids = [p for p in ids if p in config.paper_ids]
    result = []
    for paper in ids:
        path = config.output / 'reference/papers' / f'{paper}.json'
        if not path.exists():
            path = config.output / 'reference_parts/history_knowledge' / f'{paper}.json'
        if read(path)['graph_useful']:
            result.append(paper)
    return result


def view(config: Config, paper: str, condition: str) -> dict[str, Any]:
    # Existing masks operate on their config.output input directory. Point only that read at the source.
    return previous_payload(config.model_copy(update={'output': config.source}), paper, condition)


async def generate_analyses(config: Config) -> None:
    runner = Runner(config)

    async def gear(paper: str) -> None:
        base = read(config.source / 'inputs/papers' / f'{paper}.json')
        material = {'manuscript': base['manuscript'],
                    'neutral_claims': [c['claim'] for c in base['graph']['cards']],
                    'original_historical_passages': base['gear_evidence']['evidence_blocks']}
        await runner.call(paper, 'analysis', 'gear', GEAR_PROMPT, material, Analysis,
                          model='gpt-6.1-sol', effort='xhigh')

    async def graph(paper: str, condition: str) -> None:
        base = read(config.source / 'inputs/papers' / f'{paper}.json')
        material = {'manuscript': base['manuscript'], 'native_graph': view(config, paper, condition)['graph'],
                    'graph_metric_definitions': GRAPH_DEFINITIONS,
                    'original_historical_passages': base['graph_evidence']['evidence_blocks']}
        await runner.call(paper, 'analysis', 'graph_' + condition, GRAPH_PROMPT, material, Analysis,
                          model='gpt-6.1-sol', effort='xhigh')

    jobs = [gear(p) for p in eligible(config)]
    jobs += [graph(p, c) for p in eligible(config) for c in ('F', 'F_noJ', 'F_noM', 'F_noP')]
    await asyncio.gather(*jobs)


def report_material(config: Config, paper: str, condition: str) -> dict[str, Any]:
    material = view(config, paper, condition)
    material.pop('gear', None)  # Old GEAR generated judgments never enter this round.
    if condition not in ('T', 'G'):
        material['scientific_evidence_analysis'] = read(config.output / 'analysis/gear' / f'{paper}.json')
    if condition not in ('T', 'E'):
        material['graph_metric_definitions'] = GRAPH_DEFINITIONS
        graph_condition = 'F' if condition in ('F', 'G') else condition
        material['knowledge_graph_analysis'] = read(config.output / 'analysis' / ('graph_' + graph_condition) / f'{paper}.json')
    return material


async def generate_reports(config: Config) -> None:
    runner = Runner(config)

    async def one(paper: str, condition: str) -> None:
        material = report_material(config, paper, condition)
        result = await runner.call(paper, 'reports', condition, WRITER_PROMPT, material, Report,
                                   model='gpt-6.1-sol', effort='xhigh')
        path = config.output / 'reports' / condition / f'{paper}.md'
        path.write_text(result['body'] + '\n', encoding='utf-8')
        write(config.output / 'report_inputs' / condition / f'{paper}.json', {
            'paper_id': paper, 'condition': condition, 'configuration': NAMES[condition],
            'material_fields': list(material), 'input': material,
            'report_chars': len(result['body']), 'model': 'gpt-6.1-sol', 'effort': 'xhigh'})

    conditions = config.conditions or CONDITIONS
    await asyncio.gather(*(one(p, c) for p in eligible(config) for c in conditions))

from __future__ import annotations

from figure_pipeline.fig4_rerun.run import EVALUATOR as CORE_EVALUATOR

WRITER = '''Write one complete Chinese scientific innovation report directly from the supplied materials.
Use the same four sections: 主要科学贡献; 与已有研究的具体区别; 科学关系与证据; 适用范围与限制.
There is no length target. Address the principal contributions, explain supported prior-work overlap,
the specific remaining increment and its necessary scope. Cite usable original source IDs/locations.
Explain what kinds of prior knowledge the contributions connect, how these scientific relationships
differ, and what combining the contributions adds. Use concrete contributions and historical sources,
not generic novelty or usefulness language. The manuscript's own citations remain usable evidence.
Where supplied graph facts permit it, explain the actual observed pattern and its scientific relevance:
* Joint: shared historical neighbors and connections when multiple contributions are considered together,
  including how this differs from describing one contribution alone. Identify the involved contributions.
* Structure: concentration/diversity of neighborhoods, observed local connectivity and component changes;
  link concrete values or identities to a useful, bounded description of the historical knowledge setting.
* Citation contacts: distinguish recorded direct citation, two-hop, shared-reference and semantic-only links
  for specific historical sources and contributions. Chemical/biological pathways are not citation paths.
Do not merely reproduce a table of numbers or list generic graph cautions. Explain supported positive
observations as well as their limits. If the supplied facts do not establish an aspect, briefly state the
limitation; do not reconstruct removed fields, infer unprovided values or invent additional evidence.
Original passages outrank supplied analytical judgments when they conflict. Similarity/communities/local
connectivity do not establish novelty, disciplines, support or global impact. Parent-paper citation paths
do not establish claim-level derivation, priority or causality; zero snapshot counts do not establish
global absence of citations. Preserve actual manuscript context and experimental qualifications.
Do not mention input configurations, system names, evaluator dimensions, scores or a novelty score.
Return body and cited_source_ids only. This is the final report, with no intermediate analysis task.'''

RUBRIC = '''Evaluate five separate aspects of this anonymous report. No total score or condition ranking.
Apply exactly the same content standard, fixed applicability and evidence to every report of a paper.
Source names and graph vocabulary alone earn no credit; equivalent correct explanations count regardless
of which materials produced them. Judge what is actually written, never supply an absent explanation.
Original manuscript/historical passages establish scientific correctness; native graph facts establish
only recorded graph observations. Generated branch judgments and old reports are not evaluation answers.
The Fig3 criteria are reusable themes, not answers: old excerpt-only limitations do not override the full
manuscript supplied here. Verify source identity, scientific object, relationship and scope. Do not demand
universal novelty/firstness or agreement with a generated reference when original evidence contradicts it.
Use scores 0,1,2,3 with these anchors:
scientific_increment: 0=no usable or materially wrong comparison; 1=generic/fragmentary comparison;
2=concrete, supported overlap and residual difference for a principal contribution with correct scope;
3=coherent, specific comparisons for the principal contributions, distinguishing shared prior knowledge,
remaining differences and meaningful limits. A source list or generic abstention is not explanation.
knowledge_relations: 0=no usable or materially wrong relationship; 1=one isolated correct relation;
2=multiple concrete, supported relations between contributions and historical knowledge, distinguishing
their scientific roles; 3=an organized account of distinct knowledge bases and how the paper connects
them, with concrete contributions/sources and correct relation types. Neither graph jargon nor numerical
metrics are required for this dimension. Manuscript-supported relations may also qualify.
joint_explanation: 0=no useful joint explanation or wrong topology; 1=only generic synthesis or shared-base
mention; 2=a specific verified shared-neighbor/union-connectivity pattern involving at least two target
contributions and a bounded explanation; 3=also explains the difference from single-contribution views
and its scientific meaning. Generic biological/chemical mechanisms or whole-paper summaries alone do
not earn >=2. Do not require private derived counts that the report did not receive.
structure_explanation: 0=absent or materially wrong; 1=only numerical repetition or general caution;
2=a concrete correct concentration/diversity/connectivity pattern with a useful bounded interpretation;
3=relates distinct contribution neighborhoods or structural changes to an evidenced knowledge-setting
explanation. Scientific atomic structures are not historical-graph structure. Community labels are not
verified disciplines; local connectivity is not novelty or global impact.
citation_explanation: 0=no correct specific citation-contact explanation; 1=only general correct caution
or an isolated contact label; 2=a source/contribution-specific recorded contact-type distinction with
scientific relevance and correct limits; 3=an organized comparison of concrete recorded contacts that
distinguishes citation-mediated contacts from semantic-only ones and explains what this changes in
historical interpretation. No need for direct paths when only co-reference contacts are recorded.
The fixed opportunities apply to all seven reports of a paper. Missing a module is not not_applicable.
For citation_explanation, a paper with no recorded positive citation/two-hop/co-reference contacts has
no positive-contact opportunity: return not_applicable and null score for EVERY report of that paper.
You may describe its zero-snapshot caveat in the reason; do not grade it as a global citation-absence fact.
For applicable dimensions, absent content has assessed score=0. Correct abstention is not an error but
does not earn substantive-explanation credit. If evidence cannot resolve substantive correctness, return
unresolved with null score and a specific reason, rather than a confirmed score. A major contradictory
overclaim cannot be cancelled by a correct sentence elsewhere; discuss all material contradictions.
Return exactly one dimension judgment for each of the five dimension IDs. Cite 1-2 SHORT contiguous
verbatim report passages where present, exact original source passages and/or explicit graph field paths
WITH their observed values. Never concatenate noncontiguous text, paraphrase a quote or move citations.
Score 0 for absence may have no report quote; assessed score>0 must have a matching report quote.
not_applicable/unresolved must have null score. Keep original evidence, reason and unresolved_reason.
Also return the two fixed core judgments in items using the uniform core rubric below; this is the same
single evaluation call, not a separate task. Do not extract all assertions, cluster information, track
fusion, count relations or evaluate risks.
'''

EVALUATOR = RUBRIC + '\nCORE RUBRIC:\n' + CORE_EVALUATOR.replace(
    'Evaluate only these cores. No full-report assertion extraction, information counts or relation statistics.',
    'These core judgments accompany the five dimension judgments in the same response.')

from figure_pipeline.fig4_explanation_100 import prompts as previous

GRAPH_DEFINITIONS = previous.GRAPH_DEFINITIONS + '\n' + previous.STRUCTURE_DEFINITIONS
GEAR = previous.GEAR_PROMPT + previous.TASK_ANALYSIS + previous.REFINED_ANALYSIS
GRAPH = previous.GRAPH_PROMPT + previous.TASK_ANALYSIS + previous.REFINED_ANALYSIS
WRITER = previous.WRITER + previous.REFINED_WRITER

REFERENCE = '''Prepare a PRIVATE conditional evidence reference in Chinese. No candidate reports are supplied.
Reuse every originally applicable question/part identity EXACTLY; do not invent questions, split parts,
or change public tasks or original applicability. Return views F, E50, K5, each containing every applicable
part exactly once. Full manuscript/original blocks are provided once. Each view specifies the exact visible
historical blocks/source aliases and native graph. F and E50 share the original graph; K5 has its own graph.
Never count a historical generated claim_text, target bibliography or the target's historical account as
independently verified original literature. Explicitly account for other visible scientifically equivalent
sources, not just whether the old gold exemplar survived. A missing source does not always remove answerability.

For each view/part state answerable/unanswerable/unresolved, supported expected content, acceptable scientific
alternatives and exact evidence/graph locations. Preserve the scientific object and construct. Historical
scientific truths do not change when evidence is removed, but native structural observations CAN change with
K5. Rebuild K5-specific exemplars from its actual graph, not the old K10 numeric answer or pathway. If a
previous relationship disappears, a correct scoped account of the new snapshot can answer the question.
When materials genuinely cannot answer the task, retain the part and explain the limitation, do not remove
it from the denominator or award generic caution as substantive correct coverage. Correct abstention is
evaluated separately. If source/reference conflicts cannot be resolved, say unresolved. References are
model-curated, not expert truth. Do not prescribe winners or expose these answers to the writer.'''

EVALUATE = previous.EVALUATION + previous.JOINT_REVIEW + '''
FIG5 EXTENSION: Each anonymous candidate points to its own visible material packet and conditional reference.
Use ONLY that candidate's visible originals and native graph for grounding/answerability, although all raw
materials are supplied to you for scientific checking. Other candidates' materials must not be credited to it.
The reference is fallible: verify it against supplied originals and preserve unresolved conflicts.
Native graph changes legitimately change graph observations; do not grade a reduced graph against old values.

For each fixed part also classify explicitly_abstains (true/false/null), abstention_appropriateness
(reasonable/unnecessary/unresolved/not_applicable), and unsupported_definitive (true/false/null).
An explicit, part-specific statement that the available evidence cannot decide it is abstention. Silence,
omission, a generic caveat, and YOUR inability to grade are not system abstention. Give its segment IDs.
Reasonable means the visible packet genuinely cannot support a substantive answer; unnecessary means a
supported equivalent answer was available; unresolved means this cannot be established. If no explicit
abstention, use not_applicable (or unresolved when the abstention flag itself is unknown).
unsupported_definitive means the report makes a definite material assertion for this part without adequate
visible evidence, even if it happens to agree with the full reference. Give the exact assertion segment IDs.
Use null for genuinely unresolvable judgments; never convert an unknown error into false. A correct part
cannot cancel a contradictory or unsupported assertion elsewhere. Record scope, evidence IDs and reasons.
Every true abstention/error needs report segment evidence. Return all supplied report/question/part IDs
exactly once. Do not give total scores, model preferences, or guess source model/experimental condition.'''

from figure_pipeline.fig4_explanation_study.generation import (
    GEAR_PROMPT, GRAPH_DEFINITIONS, GRAPH_PROMPT, WRITER_PROMPT,
)
from figure_pipeline.fig4_explanation_study.evaluation import PROMPT as PREVIOUS_EVALUATION

REFERENCE = '''Create source-grounded scientific tasks and a PRIVATE evaluation reference in Chinese.
No candidate report, old branch judgment or condition score is provided. First screen the native graph:
retain this paper if at least one scientifically meaningful, source-checkable knowledge relationship
beyond repeating its manuscript is available. Neighbor count is not usefulness. Missing positive citation
contacts alone is not a reason to exclude a paper. Give concrete source/claim examples for the decision.

Produce exactly these task slots: H1,H2 historical_verification; K1,K2 knowledge_position;
J1,J2 joint_contribution; S1,S2 structural_resolution; P1,P2 citation_contact.
Each question field is a PUBLIC task given unchanged to all seven configurations, including paper-only.
It MUST use only target-manuscript objects, methods, scientific concepts and contribution descriptions.
Never reveal historical author/paper names, dates, source IDs, graph node IDs, community IDs, graph
values, a historical answer, or that a relation is known to exist. Do not name an external source even
if the target manuscript cites it. Use target-paper scientific concepts to specify the comparison.
Each manuscript_anchor is a short exact manuscript passage grounding the public question's target.
Public questions ask for bounded analysis of evidence if available, not a list of hidden reference answers.
Do NOT reveal private applicability, claim_ids, answers or anchors inside the public question text.

H: prior-work overlap and residual difference with independent originals; K: scientifically meaningful
distinctions among related objects, methods or mechanisms; J: specific shared or distinct historical
bases/connectivity of at least two target contributions; S: a useful quantitative neighborhood comparison
and its bounded meaning, not copying arbitrary numbers; P: concrete observed parent-paper citation
contacts versus semantic proximity, their scientific context and limitations.
Focus the task enough to be answerable without revealing the historical answer. The two tasks within an
aspect must ask different things. All originals available here are exactly those available to the full writer.
Do not require material from a broader evaluation archive. Fig3 criteria/cores guide relevance, but their
generated judgments are not scientific evidence. The supplied originals/native facts take precedence.

For each applicable task create ONE OR TWO independent answer_parts with IDs <question_id>a and b.
expected_content specifies the scientific criterion and a concrete supported exemplar. acceptable_variants
must explicitly accept other original-supported, scientifically equivalent answers to the PUBLIC task;
the exemplar is not an exclusive paper/node/number keyword. Require object/relation/scope correctness,
not incidental wording. A missing migration direction is a failure only when the task actually asks it.
Provide exact original quotes/locations or native graph field/values. Prefer one part for the concrete
comparison and one for scientific interpretation/limits. Do not count the same fact twice across tasks.
If the supplied evidence cannot resolve a useful task, mark applicable=false with no answer_parts;
still write a neutral manuscript-grounded PUBLIC question, without announcing the hidden inapplicability.
When no positive direct/two-hop/shared-reference contacts exist, BOTH P tasks are inapplicable.

Graph cards can permit reconstructing deleted summaries or joint facts. Correct reconstruction is valid.
Communities are not verified disciplines; connectivity is not impact/novelty; paths are not claim support,
antecedence or causality. An unrecorded path does not prove global absence. Native graph observations and
scientific original-source interpretations must remain distinct. Preserve abstract-only limitations.
The manuscript may already contain a correct comparison: mark this honestly; independent verification is
a separate benefit. Missing evidence cannot establish firstness. These are model-curated development
references, not expert labels. Do not force a configuration ranking.'''

TASK_ANALYSIS = '''\nThe common public scientific tasks are supplied. Address the tasks supported by this
branch's own materials, using concrete original facts and bounded interpretation. No private reference
or other branch judgment is available. Do not invent a replacement for a missing branch/component.'''

WRITER = WRITER_PROMPT + '''\nThe public_tasks list defines the common scientific assignment. Explicitly
address every task, identifying its question_id in the prose, within a coherent complete report using the
common sections. Where actual materials cannot answer it, say what is missing briefly and accurately.
Do not infer hidden answers or applicability. A correct equivalent comparison is useful; do not hunt for
an assumed gold source. Cite actual original blocks/native facts and distinguish the manuscript's own
historical account from independent verification. No length target; no scoring or configuration names.'''

EVALUATION = PREVIOUS_EVALUATION + '''\nThis round evaluates explicit PUBLIC tasks, not whether a report
mentions a curator's favorite example. For each private answer part, apply its scientific criterion to the
actual public question. A different source/relationship/quantitative contrast that directly answers that
question, is scientifically equivalent and is supported by the supplied originals/native facts earns
correct even when it differs from the reference exemplar. Explain the equivalence with the actual evidence.
Do not penalize a missing incidental number, source title, qualifier or mechanistic direction unless it
is necessary for the public task's scientific correctness. Do not reward generic answers, bare numbers,
or changing the target scientific object. Materially contradictory claims remain errors. References are
fallible: a genuine conflict with originals is unresolved, not a forced match. Some reports may be
technically unavailable: return exactly the report IDs supplied, never fabricate the missing candidates.
No global report ranking, no requested winner. Quotes must be recovered by actual segment IDs.'''

STRUCTURE_DEFINITIONS = '''Additional implementation definitions (no paper-specific answer):
Rao–Stirling = sum over community pairs a<b of 2*p_a*p_b*clip(1-dot(centroid_a,centroid_b),0,1).
p uses normalized nonnegative neighbor cosine weights over known communities. This includes semantic
centroid separation, unlike label richness or effective community count. Centroid vectors are not
provided in these packets. A missing stored Rao value cannot be recovered from IDs alone.
first_observed_recent_nature_pair_share is the fraction of distinct known-community pairs absent from
the fixed historical community_pair_history table. It is corpus-limited non-observation, not firstness.
community_pair_mean_surprisal averages -log((connector_count+0.5)*historical_claim_count/
(community_a_claim_count*community_b_claim_count)) across distinct known-community pairs; if any pair
lacks usable counts it is null. Negative values are possible: this is normalized commonness, not
-log of a bounded probability. Raw historical counts are not supplied; no percentiles or universal rarity.
Joint connectivity uses the historical neighborhood union and actual target-to-neighbor insertion
edges. Keep historical-only paths distinct from paths that pass through inserted target nodes. A shared
neighbor, an indirect historical connection, and a parent-paper citation contact are different relations.
'''

REFINED_REFERENCE = REFERENCE + '''\nFOCUSED DEVELOPMENT REVISION: return ONLY J1,J2,S1,S2 in that order.
The old H,K,P tasks are frozen externally; do not revise or duplicate them. No candidate reports/scores
are supplied. The factual opportunity packet is private derived bookkeeping from the same Full graph,
not additional evidence to be leaked to writers. Determine applicability from scientific evidence, not
from whether a configuration could win. Preserve valid reconstruction; no forced separation.

J1: Ask a manuscript-grounded whole-paper question requiring a concrete partition/relationship among
all major contributions on the union historical neighborhood, distinguishing common nodes, indirect
historical edges and target-mediated connectivity. Require a checkable connective example and bounded
scientific interpretation, not just a shared-neighbor count. Where joint-only historical edges alter
connectivity or introduce a distinct connection, use a scientifically interpretable such example;
if none is supported, explicitly document lack of unique-information opportunity privately. Do not
assume each extra edge changes connectivity or is a verified scientific support relation.
J2: Complement J1 with a scientifically meaningful cross-neighborhood connection or overlapping versus
complementary contribution bases that is not answered by repeating J1. Require the actual relational
structure and what original-source science permits interpreting. Union integration is not the sum of
single-claim insertions. Do not force this task applicable when the second substantive question is absent.

S1: Ask whether the distribution across knowledge groups and the semantic separation BETWEEN those
groups tell the same story for concrete manuscript contributions. Require both constructs, including
weighted group distribution and distance-aware diversity with necessary scope. Label count alone cannot
establish weighted diversity or centroid separation. A nonexclusive scientifically equivalent method
must actually answer the SAME constructs; do not insist on exact incidental decimals or one source.
S2: Ask whether local neighborhood connectivity/concentration and the historically observed prevalence
of its group combinations are different for concrete manuscript contributions. Require both local
structure and a corpus-bounded historical combination observation with scientific interpretation.
Density alone is not historical prevalence. Undefined statistics remain unresolved, not zero. A useful
observed absence statistic can coexist with null surprisal; do not invent its missing value.

For each applicable question use two parts: (a) requested structural distinction established with
correct source/derivation, (b) its specific scientific interpretation and necessary scope. Generic
cautions without the specific structural observation cannot satisfy (b). Explain in acceptable_variants
which alternatives preserve the scientific construct and which merely change the question.
Public questions must state these distinctions in understandable scientific language WITHOUT private
source/node names, observed counts, expected direction, answer groups or preselected winners. Scientific
objects and methods must come from the target manuscript. Private evidence uses exact fields/originals.
'''

REFINED_ANALYSIS = '''\nAddress the public joint and structural tasks at their stated resolution. Distinguish
whole-paper union organization from repeating pairwise shared-neighbor lists. Distinguish weight
concentration, between-group semantic separation, local connectivity and historical combination
prevalence; these are not interchangeable. Show actual derivation when reconstructing absent summaries.
If the required evidence is absent, preserve that limit. Keep source-supported scientific interpretation
separate from unverified graph-node text. Neither branch may invent facts to meet a task.'''

REFINED_WRITER = '''\nBefore finalizing, check every public task against the available branch findings and
original facts. Preserve the concrete observation, the scientific interpretation it supports, and the
necessary comparison conditions; do not drop an evidence-qualified historical comparison when also
presenting graph findings. For J/S tasks use the exact requested constructs, not an easier proxy.
Scientific alternatives are welcome if they answer those constructs. Missing centroid/frequency data
cannot be replaced by label counts. Correct reconstructions from remaining facts remain valid.
This is internal organization within this single response, not an extra fusion/output stage.'''

REFINED_EVALUATION = '''\nFocused J/S revision: equivalence preserves the requested scientific construct.
A shared-neighbor intersection does not alone answer a requested whole-union partition or indirect
connection. Label richness, effective weighted count, centroid-distance diversity, local density and
historical combination prevalence are distinct. A correct proxy can be a useful partial observation
without completing a part requiring another construct. Apply this identically to all anonymous reports.
Do not demand the exemplar's exact node/number if another supported answer establishes the same scope.
Correct reconstruction counts. An interpretation part requires its concrete structural/scientific
premise; generic caution alone is not coverage. Freeze H/K/P criteria unchanged. No preferred winner.'''

JOINT_REVIEW = '''\nDOCUMENTED CRITERION REVIEW APPLIED TO EVERY AVAILABLE CONFIGURATION IN THIS PAPER:
Judge the requirements actually communicated in the PUBLIC question. Private exemplars cannot silently
add another required experiment/counterfactual. For example, if a public question requests a complete
union partition, a checkable cross-neighborhood connection and explanation of what individual summaries
miss, a correct complete partition plus a traced connection absent from single-card induced edges can
establish that information increment WITHOUT necessarily reporting an explicit three-to-two component
counterfactual. Conversely, when the public question explicitly asks WHETHER adding an edge changes
connectivity, merely saying the edge existed before target insertion does not answer that question.
Scientific interpretation must be specific and grounded, but exact exemplar method names are not
mandatory when an equivalent object/relation explanation actually answers the question. Generic
cautions still do not count. Give each anonymous report equal treatment; it can improve, remain the
same, or decline. Do not see or infer a requested score, winner or prior ranking. If the private part
bundles unsupported extra requirements, evaluate only the part's substantive construct that the PUBLIC
task actually asks, and document the distinction in reason. Keep genuine omissions and errors.
'''

# Apply the accepted J/S constructs while creating all ten reference slots once.
EXPANDED_REFERENCE = REFERENCE + '''\nACCEPTED DEVELOPMENT PROTOCOL FOR THE REMAINING PAPERS:
Return all ten task slots H1,H2,K1,K2,J1,J2,S1,S2,P1,P2. The H/K/P criteria above remain
unchanged. Apply the following more specific J/S criteria. Derived factual opportunities come
from the same Full input and remain private. Screen before generating any candidate reports;
do not select papers or references to obtain a preferred configuration ranking.
J1:''' + REFINED_REFERENCE.split('\nJ1:', 1)[1].replace(
    'For each applicable question use two parts:', 'For each applicable J/S question use two parts:')

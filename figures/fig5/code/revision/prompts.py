from figure_pipeline.fig4_explanation_100 import prompts as fig4
from figure_pipeline.fig5_robustness import prompts as old

GEAR, GRAPH, WRITER = old.GEAR, old.GRAPH, old.WRITER
DEFINITIONS = old.GRAPH_DEFINITIONS
REFERENCE = '''Prepare a PRIVATE condition-specific reference in Chinese BEFORE any candidate is generated or evaluated.
Return every originally applicable question_id/part_id exactly once; fixed applicability and identities must not change.
The supplied visible_packet is the ONLY evidence available under this condition. fixed_reference is a fallible
full-condition exemplar, NOT an additional source of evidence. Evaluate scientific equivalents and alternative support.
Label each part answerable, unanswerable or unresolved.
ANSWERABILITY IS ABOUT THIS PACKET, NOT WHETHER THE SCIENTIFIC TRUTH IS RESOLVED:
- answerable: sufficient visible material supports completing the requested part.
- unanswerable: inspection establishes that required support is absent/inadequate, even if some subfacts can be stated.
  In particular, a missing required historical original makes independent verification unanswerable, NOT unresolved.
  Do not replace a missing independent comparison with a manuscript-only observation. Nor demand private exemplar extras.
- unresolved: the reference evaluator cannot determine sufficiency because of ambiguous task scope, conflicting materials,
  illegible evidence or an unresolvable evidence interpretation. State the specific ambiguity/conflict, not mere absence.
A scientific conclusion may remain unknown precisely because the task is known to be unanswerable with current evidence. The independent_work_aliases map identifies duplicate versions; aliases are one work, not independent corroborations. Evidence.source_id must use an actual source/block ID, not a canonical work label. Cite actual visible block/source IDs and exact quotes or
native graph locations/values. Graph observations must be recalculated from the supplied snapshot, not copied from
full-condition exemplars. Generated historical claim_text and target bibliography are NOT independent original verification.
Missing originals do not imply absence of prior art. A new scoped graph observation can still answer a graph task.
Unresolved means the reference cannot establish sufficiency, not that the candidate omitted an answer (no candidate supplied).
Keep expected_content and acceptable_variants specific to the public task. These are model references, not human truth.
''' + fig4.REFINED_EVALUATION + fig4.JOINT_REVIEW
SUPPORT = '''Using ONLY the supplied full visible material and frozen private/public tasks, identify at most ONE
eligible historical-evidence-dependent target part (prefer H, otherwise N) for a critical deletion experiment.
Do not inspect candidate reports. Return no targets with an explicit ineligible_reason if no defensible target exists.
For the target, enumerate ALL materially plausible complete support groups as lists of canonical independent work IDs
from source_aliases: each group is independently sufficient; all its members are jointly needed. Audit alternative
support in both evidence branches, manuscript, graph facts and generated historical claim texts. List supporting block IDs.
critical_remove must hit EVERY sufficient support group, so the requested judgment cannot remain supported after deletion.
noncritical_remove must contain exactly as many OTHER independent works, retain a complete support group, and be
irrelevant/redundant for the same target. Do not force eligibility when equivalent support persists in native graph or
manuscript, when necessity is unclear, or matched noncritical deletion is impossible. Explain necessity specifically.
EVIDENCE STANDARD MUST MATCH THE CONDITIONAL REFERENCE: generated historical claim_text, target bibliography,
and the target manuscript's historical narrative are NOT independent verification of earlier work. If the frozen part
requires independent historical verification, these derived/restated claims do not constitute a sufficient alternative
support group after originals are deleted. Do NOT reject a target merely because a historical assertion remains in
such a derived source. Conversely, native graph facts may still answer graph-position tasks, and manuscript experiments
may answer target-specific scope parts; assess the actual public construct, not the aspect label alone. Alternative
support must fully satisfy that construct under the same evidence standard, not merely mention the same subject.
Deletion operates on independent works and every original-text alias in both branches, while native graph stays fixed.
Do not choose targets according to model outputs or presumed favorable experimental results. Return exact existing IDs.
'''
EVALUATE = old.EVALUATE + fig4.REFINED_EVALUATION + '''
CORRECTED RESPONSE TAXONOMY (takes precedence over earlier wording): response_type is exactly one of
substantive_answer, target_abstention, omission, unresolved. A factual answer plus a limitation on causality or
scope remains substantive_answer. It is NOT target_abstention. Explicitly withholding the requested target judgment
is target_abstention; absent discussion is omission. Uncertainty of the evaluator is unresolved, not omission.
explicitly_abstains must be true only for target_abstention, false for substantive_answer/omission and null for unresolved.
For omission set status=not_written, not unresolved. For a pure abstention status=not_written; assess its appropriateness
against conditional answerability separately. Set conflicting_material_error for any substantive contradictory factual
error concerning this same part anywhere in the report. A correct quote cannot cancel another material error.
Grounding, scope, unsupported assertions and all error claims need exact report segments and source evidence.
Do not count a boundary caution alongside a correct answer as abstention. Do not count silence as reasonable abstention.
Use only visible_packet as candidate evidence. The conditional reference is fallible; keep scientific conflicts unresolved.
'''
DW = '''Write a Chinese scientific innovation report answering all supplied public tasks using the manuscript,
original_evidence and native graph directly. Preserve evidence/source IDs, scientific scope, and limitations.
No branch interpretation or private answers are supplied. Do not infer firstness from missing sources or causality
from graph paths. Explicitly withhold only judgments that the available material cannot support.
''' + fig4.REFINED_WRITER

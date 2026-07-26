# Fig.4 — Human innovation judgments and transparent peer review

**Question.** Does the current v6.1 score align with blinded human judgments
of novelty, significance, and difference from prior art, and with real
review dimensions?

**Design.** Cohort A is a newly generated current-score sample of 30 papers
(low/middle/high, 10 each) crossed with three labelers. Cohort B is the
existing 50-paper Nature transparent-peer-review diagnostic cohort.

**Current state.** The code emits the blinded packet and 90 blank label rows.
Scores and targets are absent from the labeler packet.

**Gate.** Until all 90 records are independently completed, the figure is
`DRAFT_LABELS`; no agreement or external-validity claim is calculated.

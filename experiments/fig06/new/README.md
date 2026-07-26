# Fig.6 — Robustness and failure boundaries

**Question.** How stable are the registered signals under missing references,
and where do data sparsity and metadata loss make them unreliable?

**Sample.** Up to 200 papers per each of the 12 domains, stratified by era
and reference volume. The frozen sample contains only 159 eligible
mathematics/statistics papers, so the auditable total is 2,359 rather than an
invented 2,400.

**Experiments.** Domain OOF intervals; the previously registered 80%
resampling check shown separately; and a common stratified sample with true
feature recomputation at 100%, 75%, 50%, 25%, and 10% reference retention
(20 repetitions for each non-full dose). The figure also includes
horizon/fold stability, a model specification curve, and the
reference-count/metadata-coverage reliability boundary.

**Boundary.** Unrelated-reference contamination and mapping-deletion panels
are withheld until exact recomputation implementations exist. Legacy proxy
doses are not promoted into the new figure.

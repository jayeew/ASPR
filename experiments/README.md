# Experiment index

The current research focus is **Claim Graph + GEAR innovation analysis**.
Historical scoring and Fig.1–Fig.10 workflows remain available for their own
questions. Directory names such as `new` describe earlier figure revisions,
not the current runtime or a newly approved protocol.

## Current workflows

- [Staged innovation study](innovation_200/README.md): sampling, reviewer
  reconstruction, shared claims, GEAR, Graph, whole-paper reports, reference
  comparison, AI preferences and summaries. Initially 200 papers; an extension
  supports claims/reference preparation for 1,000 papers.
- [Runtime analysis and controls](../gear/innovation/): current analysis,
  shared-claim direct/RAG controls, reference evaluation and diagnostics.
  `scripts/innovation_experiment.py` is a separate screen/freeze/development/test
  workflow; its freeze policy does not automatically apply to `innovation_200`.
- [Study tests](../tests/innovation_200/) and [runtime tests](../tests/innovation_v2/).

The roster, completed stage outputs and evaluation denominators are distinct.
Expanding the corpus or finishing claims does not complete system comparisons.
Models, retrieval policy and resume settings are described in the study README.

## Historical Fig.1–Fig.10

| Figure | Retained question | Relation to current research |
| --- | --- | --- |
| [1](fig01/new/README.md) | Landmark transitions and multivariate displacement | Motivation; different unit from Claim Graph |
| [2](fig02/new/README.md) | Evidence-derived indicators and nested sets | Measurement provenance; not validation of new graph metrics |
| [3](fig03/new/README.md) | HGB multi-horizon diffusion prediction | Predictive association, not novelty accuracy |
| [4](fig04/new/README.md) | Earlier layered Graph/GEAR validation | Historical protocols/results requiring compatibility review |
| [5](fig05/new/README.md) | Retrospective frontier ranking | Historical prediction application |
| [6](fig06/new/README.md) | Reference loss and scoring robustness | Perturbation methods, not current runtime evidence |
| [7](fig07/new/README.md) | Venue profiles and later diffusion | Descriptive portfolio study |
| [8](fig08/new/README.md) | Earlier framework illustration | Historical architecture |
| [9](fig09/new/README.md) | Locked TLR3 case | Prior outputs with missing-model limitations |
| [10](fig10/new/README.md) | Earlier module ablation | Comparability audit, not a validated current ablation |

Read each figure's README for its retained sources and runner. Some depend on
retired contracts or local assets: existing renders and historical audits do not
prove compatibility with today's runtime. Do not rewrite generated outputs to
make them appear current.

`common/new/base/` and `common/new/adapters/` contain older builders/renderers.
`gear/evaluation/` and `gear/review_reconstruction/` contain earlier evaluation
and five-part review reconstruction workflows.

## New figure planning status

The confirmed direction prioritizes the current innovation-analysis system and
retains historical scoring where relevant. New Fig.1–Fig.10 allocation,
subexperiments and chart designs remain under discussion. Old filenames,
implemented controls and existing figure numbers are not a frozen new protocol
or completed scientific conclusions.

See [architecture](../docs/module_architecture.md) and
[implementation scope](../docs/innovation_v2_implementation.md).

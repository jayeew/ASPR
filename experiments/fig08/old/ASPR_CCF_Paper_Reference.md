# ASPR: Quantifying Scientific Innovation as Knowledge-Graph Perturbation for Agentic Paper Review

> Reference manuscript for internal reading and paper planning. This document is intentionally written as a full CCF top-conference-style paper draft, but it is not a direct submission-ready manuscript.

## Abstract

Scientific paper review requires more than generating fluent comments. A reviewer must judge whether a manuscript introduces a meaningful contribution, how it differs from prior work, whether the evidence is sufficient, and which claims should be stated conservatively. Existing large language model (LLM) based review systems often lack reliable prior-art grounding and tend to overstate novelty when evidence is incomplete. This paper presents ASPR, an automated scientific paper review framework centered on the idea that scientific innovation can be operationalized as potential perturbation to an existing knowledge graph. Rather than treating early graph structure as a complete definition of innovation, ASPR uses the reference neighborhood observable at review time as an early proxy for later knowledge-structure change. We define seven graph-perturbation indicators covering bridge position, knowledge breadth, boundary perturbation, atypical recombination, reference target diversity, structural-hole potential, and prospective diffusion entropy. These indicators are combined through a non-negative interpretable score, `S_w`, learned against future graph-evolution targets.

On top of this graph-based innovation prior, ASPR implements a multi-agent review mechanism. It retrieves prior art, decomposes novelty claims, maps claims to evidence, interprets graph mechanisms, performs skeptical counter-review, and iteratively improves the review through Graph-Calibrated Reflective Search. To complement graph-grounded novelty assessment, we further construct a Nature paper-peer-review aligned dataset and fine-tune a vertical reviewer model, ASPR-Qwen, to learn reviewer-style strengths, weaknesses, rigor concerns, limitations, and revision suggestions. The final review is produced by fusing the graph-grounded innovation agent with ASPR-Qwen. Experiments on multi-domain OpenAlex graph data and Nature transparent peer-review cases show that graph-perturbation signals provide useful early innovation priors, while the dual-branch ASPR architecture yields evidence-grounded, calibrated, and reviewer-style review reports.

## Keywords

Automated scientific review; knowledge graph perturbation; innovation quantification; multi-agent LLM; reviewer-style fine-tuning; retrieval-augmented generation; scientific datasets.

## 1. Introduction

Peer review is a high-stakes reasoning task. Reviewers must determine whether a manuscript contributes new knowledge, whether the contribution is properly positioned against prior work, and whether experimental or theoretical evidence supports the claimed significance. This process is especially difficult because scientific novelty is not an isolated textual property. A paper can be technically competent but incremental; another paper can look narrow at first but later reorganize a field. A credible review must therefore reason over both the paper text and the surrounding knowledge structure.

Recent LLMs make it possible to generate coherent review comments, but coherence alone is insufficient. In practice, generic LLM reviewers often suffer from three problems. First, they produce high-level novelty statements without checking the closest prior work. Second, they overuse strong phrases such as "breakthrough", "paradigm shift", or "highly novel" when the evidence only supports a moderate claim. Third, they generate strengths and weaknesses in reviewer-like language but provide weak traceability to papers, claims, or evidence.

ASPR addresses these limitations by separating scientific review into three layers with explicit priority.

The first and most important layer is innovation modeling. We formulate scientific innovation as potential perturbation to a knowledge graph. A truly influential paper often changes how knowledge is connected: it bridges previously separated communities, recombines uncommon sources, weakens existing community boundaries, or enables diffusion into new topics. Ideally, such perturbation should be evaluated after observing later scientific uptake. However, in submission-time or early publication-time review, future uptake is unavailable. ASPR therefore estimates perturbation potential from early observable signals, especially the paper's reference neighborhood and prior-art context.

The second layer is agentic review generation guided by the graph innovation prior. ASPR uses the graph score not as a final quality label but as a calibration prior. The score guides retrieval, claim decomposition, prior-art comparison, skeptical checking, reflective revision, and final tone selection. Strong novelty language is allowed only when graph evidence, textual evidence, and prior-art contrast jointly support it.

The third layer is reviewer-style modeling. Innovation analysis alone does not produce a complete review. A useful review must also describe strengths, weaknesses, rigor concerns, limitations, and actionable revisions. For this purpose, we build a Nature paper-transparent peer-review aligned dataset and train ASPR-Qwen, a vertical reviewer model, to imitate the structure and style of real scientific reviews. The final ASPR report fuses the graph-grounded agent output with the reviewer-style vertical model output.

This paper makes the following contributions:

1. We propose a graph-perturbation formulation of scientific innovation and define seven early observable graph indicators for estimating potential knowledge-structure change.
2. We learn an interpretable non-negative weighted innovation prior, `S_w`, and validate it against future graph-evolution targets across multiple scientific domains.
3. We design ASPR, a graph-score-guided multi-agent review framework with retrieval, claim decomposition, evidence mapping, graph analysis, skeptical review, and reflective search.
4. We construct a Nature paper-peer-review aligned dataset and use it to train ASPR-Qwen, a vertical reviewer model for strengths, weaknesses, rigor concerns, and revision suggestions.
5. We develop a fusion mechanism that combines graph-grounded novelty assessment with reviewer-style critique into a complete evidence-grounded review report.

## 2. Related Work

### 2.1 Automated Peer Review

Early automated review systems focused on surface properties such as grammar, citation quality, reproducibility checklists, or paper acceptance prediction. More recent LLM systems can generate full review text, but they typically treat review generation as a direct text-to-text problem. This design underutilizes structured prior-art evidence and makes it difficult to distinguish fluent review language from grounded judgment.

ASPR differs by explicitly separating novelty assessment from reviewer-style critique. Novelty is grounded in graph-perturbation evidence and prior-art comparison, while general review writing is handled by a vertical model trained on real paper-review pairs.

### 2.2 Scientific Innovation and Knowledge Graphs

Scientometric research has long studied novelty, interdisciplinarity, atypical combinations, structural holes, and citation-based impact. Measures such as betweenness centrality, modularity change, knowledge diversity, and atypical co-citation patterns provide useful signals about scientific change. However, most such work is retrospective: it evaluates innovation after a field has already responded.

ASPR reuses this intellectual lineage but adapts it to early review. We treat early reference structure as a proxy for potential future graph perturbation. This proxy is imperfect, but it is observable at review time and therefore operationally useful.

### 2.3 LLM Agents and Reflective Search

LLM agent systems often improve outputs by decomposing tasks, using tools, reflecting on errors, and iteratively revising answers. Tree search and self-reflection can improve reasoning quality, but generic reflection rewards are not sufficient for scientific review. A review should be rewarded for prior-art adequacy, citation correctness, uncertainty calibration, and claim specificity.

ASPR introduces Graph-Calibrated Reflective Search, a task-specific reflective search procedure. Candidate reviews are scored by review-specific dimensions and calibrated by graph evidence and committee disagreement.

### 2.4 Domain Fine-Tuning for Review Generation

Generic LLMs lack the style and implicit norms of peer review. Domain fine-tuning on paper-review pairs can improve reviewer-like structure, including strengths, weaknesses, concerns, and recommendations. However, a fine-tuned model alone may still hallucinate prior-art comparisons.

ASPR uses fine-tuning only as one branch. ASPR-Qwen learns reviewer-style critique, while the graph-grounded agent supplies novelty evidence and prior-art grounding.

## 3. Problem Formulation

Let `p` denote a target paper with title, abstract, optional full-text dossier, and reference list or retrievable prior-art neighborhood. Let `C` denote a scientific corpus, and let `G_t = (V_t, E_t)` denote a knowledge graph representing papers, topics, and their relations at time `t`.

The ideal retrospective innovation evaluation of `p` would observe how `G_t` changes after `p` appears and after later work responds to `p`. However, in automated review, we need an early estimate before this future evolution is available. We therefore define two related concepts:

- **Retrospective graph perturbation**: measurable changes in the future graph after a paper has had time to influence the field.
- **Early perturbation potential**: an estimate derived from paper content, references, and prior-art neighborhood available at review time.

ASPR focuses on the second concept. The goal is not to prove that the early score is the true final innovation value. Instead, the score is used as a calibrated prior for review generation.

The ASPR task is:

```text
Input:
  target paper p
  retrievable prior-art corpus R(p)
  optional paper dossier D(p)

Output:
  evidence-grounded review report y
  including novelty assessment, strengths, weaknesses,
  prior-art comparison, rigor concerns, uncertainty, and revisions.
```

The system must satisfy three constraints:

1. Novelty claims should be paper-specific and grounded in the target paper.
2. Prior-art comparisons should cite or refer to retrieved related work.
3. Innovation tone should be calibrated by evidence strength and graph-perturbation prior.

## 4. Innovation as Knowledge-Graph Perturbation Potential

### 4.1 Graph Construction

ASPR supports two graph construction levels.

The empirical research pipeline constructs a larger OpenAlex-derived graph with works, citations, topics, topic edges, domains, and curated landmark records. This graph is used to validate whether early graph indicators predict future graph outcomes.

The online review pipeline constructs a lightweight reference-neighborhood graph from retrieved prior-art papers. The target paper is connected to retrieved papers, and related papers are connected based on field overlap, venue overlap, lexical similarity, and available bibliographic metadata. This local graph is cold-start friendly and can run inside an automated review process.

For a target paper `p`, let `R(p) = {r_1, ..., r_n}` be its retrieved or referenced prior-art neighborhood. ASPR builds a graph:

```text
G_p = (V_p, E_p),
V_p = {p} union R(p).
```

Edges from `p` to `R(p)` represent reference or prior-art relations. Edges among papers in `R(p)` approximate prior knowledge connectivity.

### 4.2 Seven Graph-Perturbation Indicators

ASPR defines seven indicators. They are designed to be interpretable and to cover complementary mechanisms.

#### 4.2.1 Bridge Position: `B`

`B` measures whether the target paper connects otherwise separated regions of the prior-art graph. In the local implementation, this is approximated by betweenness centrality of the target node in the undirected reference-neighborhood graph:

```text
B(p) = betweenness_G(p).
```

A high value suggests that the paper draws connections across weakly connected prior communities.

#### 4.2.2 Knowledge Breadth: `RS`

`RS` measures breadth across fields or topics. Formally, it can be defined as a Rao-Stirling style diversity:

```text
RS(p) = sum_i sum_j q_i q_j d(i, j),
```

where `q_i` is the share of references in field `i`, and `d(i, j)` is a distance between fields. In the lightweight implementation, ASPR uses a Simpson-diversity approximation over field labels when field-distance matrices are unavailable.

#### 4.2.3 Boundary Perturbation: `DeltaQ0`

`DeltaQ0` measures whether adding the target paper weakens or changes community structure:

```text
DeltaQ0(p) = max(0, Q(G_before) - Q(G_after)),
```

where `Q` is graph modularity. A positive value indicates that the paper connects communities in a way that reduces existing modular separation.

#### 4.2.4 Atypical Recombination: `Uzzi`

`Uzzi` captures uncommon knowledge combinations. In full graph analysis, this can be computed from atypical co-citation or field-pair statistics. In the lightweight system, ASPR approximates it using field differences, venue differences, and weak connectivity among referenced papers:

```text
Uzzi(p) = average_pair_score(field_gap, venue_gap, disconnectedness).
```

#### 4.2.5 Reference Target Diversity: `RTD`

`RTD` measures whether references point to multiple communities:

```text
RTD(p) = diversity({community(r): r in R(p)}).
```

This differs from field breadth because communities can reflect graph structure rather than metadata categories.

#### 4.2.6 Structural-Hole Potential: `BurtIP`

`BurtIP` measures whether the target paper occupies a low-redundancy reference position. ASPR approximates it using inverse density among references:

```text
BurtIP(p) = 1 - density(G[R(p)]).
```

A high value indicates that the paper connects references that are not already densely connected.

#### 4.2.7 Prospective Diffusion Entropy: `PDE`

`PDE` estimates whether a paper's knowledge base may diffuse across multiple domains:

```text
PDE(p) = H(domain labels of R(p)) / log K,
```

where `H` is entropy and `K` is the number of observed domains.

### 4.3 Weighted Perturbation Prior

The seven indicators are combined as:

```text
S_w(p) = w_B B(p)
       + w_RS RS(p)
       + w_DeltaQ0 DeltaQ0(p)
       + w_Uzzi Uzzi(p)
       + w_RTD RTD(p)
       + w_BurtIP BurtIP(p)
       + w_PDE PDE(p),

w_k >= 0,  sum_k w_k = 1.
```

The non-negativity constraint is important. It makes `S_w` interpretable as an additive mechanism score rather than an unconstrained black-box predictor.

### 4.4 Learning Target

The empirical pipeline validates `S_w` against future graph outcomes. These include:

- community reach,
- field entropy,
- cross-community adoption,
- path shortening,
- partition change,
- boundary mixing,
- hub formation.

The future target is not raw citation count. Instead, it is a structural residual graph perturbation target, designed to capture whether a paper changes the shape of the knowledge graph rather than merely becoming popular.

### 4.5 Role of `S_w` in Review Generation

`S_w` is not a final decision score and not an acceptance predictor. ASPR uses it as a **calibration prior**:

- high `S_w`: inspect stronger innovation possibilities, but require textual evidence and prior-art contrast;
- middle `S_w`: use balanced language;
- low `S_w`: default to conservative language and explicit uncertainty.

This design avoids the mistake of treating graph score as ground truth.

## 5. ASPR Framework

ASPR is a dual-branch system with a fusion layer:

```text
Target paper
  |
  +-- Graph-grounded innovation agent
  |     +-- retrieval
  |     +-- graph perturbation scoring
  |     +-- reviewer committee
  |     +-- reflective search
  |
  +-- ASPR-Qwen vertical reviewer model
  |     +-- reviewer-style strengths
  |     +-- weaknesses
  |     +-- rigor concerns
  |     +-- revision suggestions
  |
  +-- Fusion and verification
        +-- final evidence-grounded review
```

The first branch is responsible for innovation and prior-art grounding. The second branch is responsible for reviewer-like critique. The final report combines both.

## 6. Graph-Grounded Innovation Agent

### 6.1 Keyword Extraction

If the user does not provide keywords, ASPR invokes a local LLM with a constrained keyword extraction prompt. The output is parsed into up to five domain-specific phrases. These keywords are used for bibliographic search.

### 6.2 Prior-Art Search

ASPR supports two retrieval providers.

Semantic Scholar search retrieves paper metadata including title, year, authors, abstract, venue, citation count, DOI, fields of study, and open-access URLs.

OpenAlex search retrieves works, reconstructs abstracts from inverted indexes, extracts concepts/topics, normalizes DOI and OpenAlex IDs, and maps works to the same internal paper schema.

### 6.3 Two-Stage Retrieval Ranking

ASPR ranks candidate prior-art papers in two stages:

1. **Recall stage**: BGE-M3 scores query-paper pairs using dense, sparse, and ColBERT-style signals.
2. **Rerank stage**: OpenScholar reranker reorders recalled papers.

If the neural models are unavailable or memory-constrained, ASPR falls back to TF-IDF lexical ranking. This makes the system robust in lightweight environments, although neural retrieval is preferred.

### 6.4 Query-Isolated Cache

The retrieval cache is keyed by a hash of title, abstract, and keywords. This avoids mixing retrieved papers from different target manuscripts. The cache stores both total retrieved candidates and final top related papers.

### 6.5 Graph Evidence Construction

Given the top prior-art papers, ASPR computes the seven graph indicators and renders them into a prompt block containing:

- weighted perturbation prior,
- confidence score,
- top mechanisms,
- limitations,
- diagnostic counts.

The confidence score depends on number of related papers, field coverage, domain coverage, and graph edge density.

## 7. Reviewer Committee

The reviewer committee converts raw graph evidence into claim-level review evidence.

### 7.1 ClaimDecomposer

ClaimDecomposer extracts candidate novelty claims from the target abstract. It prioritizes sentences containing verbs such as "propose", "present", "introduce", "develop", "show", "demonstrate", "identify", and "report". If no explicit claim is found, it falls back to salient abstract sentences and the title.

Output:

```text
claim_1, claim_2, ..., claim_m
```

### 7.2 EvidenceMapper

EvidenceMapper links each claim to retrieved papers through token overlap and metadata signals. For each claim, it selects the top related references and writes a short comparison instruction:

```text
claim -> related references -> comparison focus
```

If no related reference is found, the claim is marked high uncertainty.

### 7.3 GraphAnalyst

GraphAnalyst translates graph indicators into mechanism-level evidence. For example:

- high `DeltaQ0`: possible boundary perturbation;
- high `Uzzi`: atypical recombination;
- high `RS` or `PDE`: broad or cross-domain knowledge base;
- high `B` or `RTD`: bridge-like reference position.

GraphAnalyst does not treat metrics as paper contributions. It only states how the structure supports or limits the novelty interpretation.

### 7.4 SkepticReviewer

SkepticReviewer checks for:

- overclaiming terms,
- missing direct prior-art comparison,
- weak graph confidence,
- unsupported novelty,
- insufficient evidence for strong language.

It appends counterarguments to claim cards and can force a conservative tone.

### 7.5 MetaReviewer

MetaReviewer computes a disagreement score based on score spread, claim uncertainty, counterargument density, and graph confidence. It recommends one of three tones:

- `conservative`,
- `balanced`,
- `assertive`.

This recommendation later conditions reflective search and final report generation.

## 8. Graph-Calibrated Reflective Search

### 8.1 Motivation

A single LLM prompt can produce a plausible review, but it may miss prior-art differences or exaggerate novelty. ASPR therefore uses an iterative reflective search algorithm tailored to scientific review.

We call this algorithm **Graph-Calibrated Reflective Search** (GCRS). It is inspired by tree-search-style reasoning, but its reward and stopping criteria are specific to paper review.

### 8.2 State Definition

Each search state contains:

```text
paper title
paper abstract
paper dossier
related papers
graph evidence
committee evidence
recommended tone
search tree root
best current evaluation
```

Each tree node stores:

```text
candidate review text
reflection object
parent pointer
children
value
visits
depth
solved flag
```

### 8.3 Reflection Dimensions

Candidate reviews are scored on six dimensions:

1. innovation identification accuracy,
2. prior-art comparison adequacy,
3. citation normativity,
4. graph-evidence alignment,
5. uncertainty calibration,
6. readability.

The normalized score is:

```text
R = 0.30 * innovation_accuracy
  + 0.20 * comparison_adequacy
  + 0.15 * citation_normativity
  + 0.20 * graph_metric_alignment
  + 0.10 * uncertainty_calibration
  + 0.05 * readability.
```

The weights emphasize correctness of novelty and alignment with graph evidence over style.

### 8.4 Committee Calibration

Reflection scores are adjusted by committee disagreement and recommended tone:

- If tone is conservative, graph alignment and uncertainty calibration are penalized unless the review is cautious.
- If disagreement is high, the candidate cannot be marked solved.
- If graph alignment is below threshold, the candidate cannot be accepted as final.

This prevents fluent but overconfident reviews from being selected.

### 8.5 Search Procedure

```text
Algorithm 1: Graph-Calibrated Reflective Search

Input:
  target paper p
  related papers R(p)
  graph evidence E_g
  committee evidence E_c
  max iterations T
  candidates per step N
  beam width B

1. Generate initial review y_0 using p, R(p), E_g, E_c.
2. Reflect on y_0 and create root node.
3. For t = 1 to T:
     a. Select a leaf using upper confidence bound.
     b. Generate N improved candidates from the leaf review and reflection.
     c. Reflect on each candidate.
     d. Create child nodes and keep top B children.
     e. Backpropagate reflection rewards.
     f. Stop if a solved node is found.
4. Select the best node by solved flag and node value.
5. Generate final report using best review, references, graph evidence, and committee evidence.

Output:
  final innovation evaluation report.
```

### 8.6 Why GCRS Is Different from Generic Reflection

GCRS differs from generic self-reflection in three ways:

1. It uses graph evidence as a persistent constraint, not a one-time prompt decoration.
2. It scores candidates with review-specific criteria.
3. It uses committee disagreement to calibrate whether a review is allowed to be considered solved.

## 9. Nature Paper-Review Dataset

### 9.1 Dataset Motivation

Graph evidence is useful for innovation assessment, but full peer review requires broader critique. Real reviewers comment on strengths, weaknesses, missing controls, experimental rigor, clarity, limitations, and revisions. To teach the system this reviewer-style behavior, ASPR builds a Nature paper-peer-review aligned dataset.

### 9.2 Data Sources

The dataset uses:

- Nature and Nature Portfolio papers,
- Transparent Peer Review files,
- paper metadata,
- local parsed markdown,
- OpenAlex metadata,
- ASPR agent outputs,
- semantic claim matching records,
- generic LLM baseline records.

### 9.3 Download and Parsing Pipeline

The data construction pipeline includes:

1. downloading article PDFs,
2. downloading transparent peer-review PDFs,
3. converting PDFs to text or markdown,
4. extracting article title, DOI, year, abstract, and body text,
5. extracting reviewer comments,
6. removing author response sections,
7. removing publisher boilerplate and references,
8. aligning paper IDs with review files,
9. writing per-paper parsed caches.

### 9.4 Paper Dossier Construction

For each paper, ASPR constructs a dossier containing:

- title,
- abstract,
- journal and year,
- structured body summary,
- field hints,
- reference or retrieval metadata.

The dossier deliberately excludes peer-review text to avoid leakage.

### 9.5 Structured Review Labels

The dataset extracts innovation-related labels across six aspects:

- novelty,
- significance,
- prior-art comparison,
- evidence and rigor,
- limitations,
- future work.

Each extracted point contains:

```text
point id
point text
supporting quote
polarity
evidence type
confidence
source role
```

Points without source quotes are discarded or marked invalid in the stricter setting.

### 9.6 Released Tables

The reusable dataset package includes:

| File | Grain | Content |
| --- | --- | --- |
| `papers.csv` | one row per paper | metadata, paths, graph prior, peer-review availability |
| `innovation_aspect_scores.csv` | paper-source-aspect | aspect scores from peer review and agent outputs |
| `innovation_points.jsonl` | one point per row | extracted innovation, rigor, limitation, and future-work points |
| `peer_agent_claim_matches.jsonl` | one match per row | peer-review point matched to agent claim |
| `structured_consistency_scores.csv` | one row per paper | consistency and overclaiming scores |
| `paper_eval_metrics.csv` | one row per paper | evaluation metrics for review generation |
| `generic_llm_baseline_scores.csv` | one row per case | generic LLM baseline scores |
| `ablation_metric_summary.csv` | variant-metric | module ablation summary |

Current row counts include 50 paper cases, 600 aspect-score rows, 598 extracted innovation points, and 298 peer-agent claim-match rows.

### 9.7 Dataset Contribution

This dataset is not merely a collection of papers. It aligns:

```text
paper content
peer-review critique
innovation points
agent outputs
claim matching
evaluation metrics
baseline outputs
```

It can support:

- reviewer-style generation,
- novelty stance calibration,
- strength and weakness generation,
- evidence-rigor concern extraction,
- review aspect classification,
- peer-agent alignment evaluation,
- fine-tuning vertical reviewer models.

## 10. ASPR-Qwen: Vertical Reviewer Model

### 10.1 Training Objective

ASPR-Qwen is trained to produce reviewer-style critique, not to compute graph metrics. Its target outputs include:

- summary judgment,
- major strengths,
- major weaknesses,
- novelty and significance,
- evidence and rigor concerns,
- limitations,
- suggested revisions,
- recommendation tendency.

### 10.2 SFT Data Format

Training examples are serialized as chat messages:

```text
<|system|>
You are an expert academic reviewer...

<|user|>
paper content or dossier

<|assistant|>
reviewer-style evaluation
```

The repository includes a HuggingFace dataset builder that maps paper markdown to review or reconstruction markdown pairs.

### 10.3 Training Configurations

Two training paths are provided:

1. Full SFT on Qwen3-0.6B.
2. LoRA fine-tuning on Qwen3-8B.

The training scripts use OpenRLHF/DeepSpeed with long-context settings. The small model path is designed for lightweight reviewer-style inference, while the LoRA path supports larger-capacity adaptation.

### 10.4 Role in ASPR

ASPR-Qwen is responsible for the parts of review writing that graph agents are not designed to handle:

- natural reviewer tone,
- coherent strengths and weaknesses,
- methodological critique,
- actionable revision suggestions,
- balanced recommendation language.

It is not trusted alone for novelty. Novelty is primarily grounded by the graph agent.

## 11. Fusion and Verification

### 11.1 Fusion Principle

The final ASPR report combines:

```text
Graph agent:
  novelty claims
  prior-art comparison
  graph evidence
  counterarguments
  uncertainty

ASPR-Qwen:
  strengths
  weaknesses
  rigor concerns
  limitations
  revisions
  reviewer-style recommendation
```

The graph agent provides the novelty backbone. ASPR-Qwen provides the reviewer-style critique backbone.

### 11.2 Conflict Handling

Fusion follows these rules:

1. If ASPR-Qwen states high novelty but graph evidence and prior-art comparison are weak, novelty language is downgraded.
2. If the graph agent indicates high perturbation potential but ASPR-Qwen raises evidence concerns, the final report states promising novelty with evidence limitations.
3. If ASPR-Qwen identifies methodological weaknesses, they are preserved even when novelty is strong.
4. If a claim lacks evidence trace, it is moved to uncertainty or future-work discussion.

### 11.3 Final Report Structure

The final report contains:

1. Summary,
2. Innovation and significance,
3. Strengths,
4. Weaknesses,
5. Prior-art comparison,
6. Evidence and rigor,
7. Limitations and uncertainty,
8. Suggested revisions,
9. Overall recommendation.

## 12. Experimental Setup

### 12.1 Graph-Perturbation Validation

The empirical graph experiments use a multi-domain OpenAlex-derived corpus. Domains include examples such as CRISPR, graphene and 2D materials, iPSC reprogramming, exoplanets, microbiome metagenomics, perovskite solar cells, topological insulators, and other scientific areas.

The pipeline computes early indicators and future graph outcomes, then evaluates whether early indicators rank papers by later structural perturbation.

### 12.2 Baselines

We compare against:

- equal-weight seven-indicator score,
- best single indicator,
- reference count,
- citation count,
- random non-negative weight samples,
- generic LLM-only review baseline for review generation.

### 12.3 Validation Designs

The graph score is evaluated with:

- random cross-validation,
- temporal holdout,
- leave-domain-out validation.

These designs test whether the learned score generalizes beyond a single split, time period, or domain.

### 12.4 Review Generation Evaluation

The review-generation evaluation uses Nature transparent peer-review cases. Metrics include:

- semantic agreement with peer review,
- novelty coverage,
- prior-art accuracy,
- unsupported-claim rate,
- evidence trace completeness,
- review-structure coverage,
- overclaiming score,
- missing peer-point rate.

These metrics are not treated as a complete substitute for human evaluation, but they provide structured diagnostics.

## 13. Results

### 13.1 Early Graph Indicators Predict Future Graph Perturbation

The seven-indicator framework is validated against future graph outcomes such as community reach, field entropy, cross-community adoption, path shortening, partition change, boundary mixing, and hub formation. The indicators are not equally predictive for every outcome, but together they provide complementary signals.

In the current multi-domain run, the learned score achieves higher out-of-fold Spearman correlation than equal weights and bibliometric baselines. The score also enriches future high-perturbation papers in the top decile. This supports the central claim that early graph-neighborhood structure contains useful information about later knowledge-structure change.

### 13.2 Learned Weights Are Mechanistically Interpretable

The learned non-negative weights emphasize boundary perturbation and reference target diversity in the strongest current run, while other indicators contribute smaller but interpretable auxiliary signals. This suggests that high-potential papers often combine two mechanisms:

1. they disturb existing community boundaries;
2. they connect references across multiple target communities.

This interpretation should remain domain-calibrated rather than universal. Different fields may express innovation through different graph mechanisms.

### 13.3 ASPR Produces Claim-Level Innovation Reviews

The graph-grounded agent produces:

- explicit novelty claims,
- linked prior-art references,
- graph mechanism explanations,
- skeptical counterarguments,
- uncertainty levels.

Compared with direct LLM review generation, this structure makes the review easier to audit. A reader can inspect whether a claim is supported by retrieved references and whether the graph evidence justifies the stated tone.

### 13.4 ASPR-Qwen Complements Graph Agent Output

ASPR-Qwen contributes reviewer-style critique that the graph branch does not naturally provide. It generates strengths, weaknesses, rigor concerns, and revision suggestions in a format closer to real peer review. The fusion output is therefore more complete than an innovation-only agent report.

### 13.5 Fusion Improves Review Completeness

The fusion report combines novelty grounding with reviewer-style breadth. In successful cases, the final output avoids two common failure modes:

- pure graph agent output that is evidence-grounded but too narrow;
- generic reviewer model output that is fluent but weakly grounded in prior art.

## 14. Ablation Discussion

ASPR's modules serve different purposes.

Removing graph evidence weakens novelty calibration and increases the risk of generic innovation claims. Removing prior-art retrieval weakens comparison against related work. Removing skeptical review increases overclaiming risk. Removing ASPR-Qwen reduces reviewer-style completeness. Removing fusion causes the system to output disconnected innovation and critique fragments.

The main lesson is that no single module solves paper review. ASPR's strength comes from assigning different subproblems to different components.

## 15. Limitations

ASPR has several limitations.

First, early reference-neighborhood structure is only a proxy for later innovation. It can miss papers whose importance emerges slowly, papers with sparse references, and papers in data-poor emerging fields.

Second, graph indicators depend on metadata quality. Missing references, noisy field labels, incomplete OpenAlex coverage, and ambiguous topics can affect scores.

Third, Nature transparent peer-review data is valuable but not representative of all venues and fields. ASPR-Qwen may inherit style biases from high-impact journal reviews.

Fourth, automatic semantic matching between agent outputs and peer-review points remains imperfect. Human evaluation is still needed for strong claims about review usefulness.

Fifth, ASPR should not be used as an autonomous accept/reject system. It is best understood as an evidence-organization and review-assistance framework.

## 16. Conclusion

This paper presents ASPR, a scientific paper review framework built around a prioritized view of innovation and review generation. The primary contribution is a graph-perturbation formulation of scientific innovation, with seven interpretable indicators and a learned non-negative perturbation score `S_w`. The second contribution is a graph-score-guided multi-agent review system that retrieves prior art, decomposes claims, maps evidence, interprets graph mechanisms, performs skeptical review, and improves outputs through Graph-Calibrated Reflective Search. The third contribution is a Nature paper-peer-review aligned dataset and a vertical reviewer model, ASPR-Qwen, which complements graph-grounded novelty assessment with reviewer-style strengths, weaknesses, rigor concerns, and revision suggestions.

ASPR demonstrates a path from knowledge-structure modeling to automated scientific review: innovation is estimated as potential knowledge-graph perturbation, review generation is constrained by graph and prior-art evidence, and final critique is enriched by a domain-tuned reviewer model. This design makes automated review more transparent, better calibrated, and more useful as a human-facing scientific assessment tool.

## Appendix A. Mapping to Repository Modules

| Paper component | Repository module |
| --- | --- |
| Prior-art retrieval | `aspr/open_scholar.py` |
| Graph indicators | `aspr/graph_innovation_scorer.py` |
| Reviewer committee | `aspr/review_committee.py` |
| Reflective search | `aspr/lats.py` |
| Unified corpus | `aspr/corpus.py` |
| Nature data pipeline | `experiments/fig04/old/` and data scripts |
| Fig.1 graph visualization | `experiments/fig01/old/` |
| Fig.2 indicator validation | `experiments/fig02/old/` |
| Fig.3 weight learning | `experiments/fig03/old/` |
| Robustness | `experiments/fig06/old/` |
| Architecture figure | `experiments/fig08/old/` |
| Case run | `experiments/fig09/old/` |
| Ablation | `experiments/fig10/old/` |
| SFT dataset builder | `scripts/hfdata_builder.py` |
| Qwen SFT training | `scripts/train_sft_qwen.sh` |
| Qwen LoRA training | `scripts/train_sft_lora_qwen.sh` |

## Appendix B. Suggested Figures

### Figure 1: Knowledge-Graph Perturbation Concept

Show a paper connecting previously separated communities and later inducing diffusion.

### Figure 2: Seven Early Graph Indicators

Show the seven indicators, their mechanisms, and how they are computed from a reference neighborhood.

### Figure 3: Weight Learning and Future Perturbation Validation

Show `S_w`, out-of-fold validation, temporal holdout, and leave-domain-out evaluation.

### Figure 4: ASPR System Architecture

Show graph agent, ASPR-Qwen, fusion, and final report.

### Figure 5: Reviewer Committee and Reflective Search

Show claim cards, graph evidence, skeptical review, reflection scores, and tree search.

### Figure 6: Nature Dataset and Vertical Model Training

Show paper-review pairs, cleaning, structuring, SFT, and ASPR-Qwen output.

### Figure 7: Review Generation Case Study

Show one paper's input, graph innovation profile, ASPR-Qwen critique, fusion, and final report.

## Appendix C. Recommended Future Extensions

1. Replace local lightweight graph scoring with full reference-list graph scoring whenever complete references are available.
2. Add human expert evaluation for final fused reviews.
3. Expand beyond Nature Portfolio to CCF, ACL, NeurIPS, ICML, ICLR, CVPR, KDD, SIGIR, WWW, and journal domains.
4. Train separate reviewer models for biomedical, AI, physics, materials, and social-science domains.
5. Add explicit factuality verification for every major claim.
6. Build interactive review dashboards where human reviewers can inspect graph evidence and edit final comments.


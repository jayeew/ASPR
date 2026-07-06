# Data Sources And Sharing Boundaries

This project uses public scholarly metadata, locally cached derived tables, and local manuscript/peer-review text caches to build the Fig.1-Fig.10 evidence package.

## Public Metadata Sources

- **OpenAlex**: works, source/venue metadata, concepts/topics, citation and reference metadata where available.
- **Semantic Scholar**: prior-art search and citation metadata when API access is configured.
- **Crossref**: DOI and publication metadata checks when available.
- **Local derived corpus tables**: `data/knowledge_corpus/...` views such as works, citations, topics, topic edges, domains, and figure-specific views.

## Restricted Or Third-party Content

- Local Nature manuscript markdown, peer-review markdown, and PDFs under paths such as `/mnt/d/aspr_nature_markdown` and `/mnt/d/dataset/...` are used for local analysis and review.
- These files may contain third-party copyrighted full text, peer-review files, or publisher PDFs. The project must not redistribute copyrighted full text, publisher PDFs, or peer-review PDFs unless an explicit license permits redistribution.
- Publication packages should share only derived metadata, feature tables, quality reports, source tables, figure manifests, and reproducibility scripts.

## Recommended Public Release Package

- Frozen figure source tables and derived metadata needed to reproduce the reported metrics.
- Quality gate reports, run manifests, claim ledger, and Nature-readiness summaries.
- Scripts, configuration, environment files, and tests.
- A data availability statement that explains which original sources are public, which local text/PDF sources are restricted, and how reviewers can request access-compatible reproductions.

## Current Nature-ready Gap

The current main graph-perturbation claim is gated separately from ASPR application claims. Fig.6 now includes construction-matched full-rerun robustness artifacts, and Fig.9 has a checkpoint-generated single-case ASPR-Qwen output with saved metadata. Remaining strict external-evidence gaps are Fig.4 blinded low/middle/high novelty-significance labels and Fig.10 blinded human preference ratings. Fig.10 true module reruns are present, but the ASPR performance claim remains Extended Data until the human-preference gate passes.

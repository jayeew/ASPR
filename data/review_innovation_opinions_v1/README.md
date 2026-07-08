# Review Innovation Opinions v1

Reusable ASPR data distilled from `outputs/kg_perturbation_fig4_full50`, plus compact fig10/fig5 benchmark tables.

## Main Tables

- `papers.csv`: 50 paper cases with metadata, source paths, graph-prior scores, and peer-review availability.
- `innovation_aspect_scores.csv`: 600 paper/source/aspect score rows from peer-review and agent-derived innovation judgements.
- `innovation_points.jsonl`: 598 extracted critique points across novelty, significance, prior-art comparison, evidence rigor, limitations, and future work.
- `peer_agent_claim_matches.jsonl`: 298 peer-to-agent claim alignment rows.
- `structured_consistency_scores.csv`: 50 paper-level semantic consistency and overclaiming scores.
- `paper_eval_metrics.csv`: 50 paper-level evaluation metrics used by fig4/fig10.
- `innovation_judgements_raw.jsonl`: compact preserved copy of the nested source judgement records.
- `source_inventory.csv`: source path, copied raw path, row count, and checksum for each source artifact.

## Caveats

These are silver labels derived from transparent peer-review artifacts and ASPR experiment outputs, not manually adjudicated gold labels. The two failed judgement rows are retained for auditability. Full fig10 review texts are not copied into this v1 package; `manifest.json` records their source directories and counts.

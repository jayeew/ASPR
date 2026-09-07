# Nature 2026 paired peer-review test set

This local test set contains 1,000 validated paper/peer-review pairs formally
published in 2026. Every accepted article also has a `-026-` Nature DOI segment,
matching the year convention used by the existing 2023--2025 local corpus.

## Composition

| Journal | Journal ID | Pairs |
| --- | ---: | ---: |
| Nature Communications | 41467 | 750 |
| Communications Biology | 42003 | 50 |
| Communications Chemistry | 42004 | 50 |
| Communications Physics | 42005 | 50 |
| Communications Materials | 43246 | 50 |
| Communications Earth & Environment | 43247 | 50 |

The included publication dates range from 2026-01-10 through 2026-05-08.

## Layout

- `paper/`: 1,000 article Markdown files.
- `peer_review/`: 1,000 peer-review Markdown files.
- `manifest.jsonl`: one source- and hash-traceable record per accepted pair.
- `rejected.jsonl`: candidate failures, retained for auditability.
- `summary.json`: counts, paths, validation policy, and manifest hash.
- `/mnt/d/dataset/nature_2026_testset/paper/`: original article PDFs.
- `/mnt/d/dataset/nature_2026_testset/peer_review/`: original review PDFs.

## Acceptance checks

A pair is accepted only when both publisher files pass. Article PDFs must have a
PDF signature, parse successfully, contain at least two pages and 5,000
extractable characters, and contain the expected Nature article ID. Review PDFs
must come from the publisher's transparent-peer-review link, parse successfully,
contain at least 1,000 extractable characters, and contain both the peer-review
file label and reviewer/report structure. Failed candidates are replaced.

## Rebuild or resume

```bash
python3 scripts/build_nature_2026_testset.py \
  --target 1000 \
  --workers 6 \
  --candidates-per-journal 1000
```

The command resumes from `manifest.jsonl`. Add `--fresh` only when intentionally
replacing this generated test set. Publisher-specific article and peer-review
licenses remain applicable; this corpus is stored locally for research testing.

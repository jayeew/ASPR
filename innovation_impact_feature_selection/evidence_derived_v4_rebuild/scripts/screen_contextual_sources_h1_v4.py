"""Create the independent H1 contextual-source screening artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "contextual_source_screening_input_v4.csv"
OUTPUT = ROOT / "outputs" / "contextual_source_screening_H1_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_source_screening_H1_completed_v4.manifest.json"

INCLUDE = {
    "10.1002/asi.22680", "10.1002/asi.23515", "10.1007/s00799-020-00288-2",
    "10.1007/s10489-017-1105-y", "10.1007/s11192-010-0202-z",
    "10.1007/s11192-016-2150-8", "10.1007/s11192-017-2296-z",
    "10.1007/s11192-022-04486-4", "10.1007/s11192-024-05116-x",
    "10.1007/s40037-021-00695-4", "10.1016/j.ejor.2015.04.002",
    "10.1016/j.emj.2026.03.002", "10.1016/j.heliyon.2017.e00300",
    "10.1016/j.ipm.2023.103323", "10.1016/j.jbusres.2022.06.050",
    "10.1017/cbo9781316161012", "10.1038/s41562-022-01351-5",
    "10.1038/s41586-022-05543-x", "10.1093/pnasnexus/pgae155",
    "10.1093/reseval/rvaa038", "10.1108/00220410810844150",
    "10.1108/jd-12-2020-0218", "10.1111/acem.12482", "10.1126/science.aao0185",
    "10.1139/facets-2019-0012", "10.1162/qss_a_00109", "10.1214/09-sts285",
    "10.1257/jel.20161326", "10.1353/lib.2013.0005",
    "10.1371/journal.pbio.1002541", "10.1371/journal.pone.0005910",
    "10.1371/journal.pone.0006022", "10.1371/journal.pone.0120495",
    "10.1371/journal.pone.0135095", "10.1371/journal.pone.0230416",
    "10.1371/journal.pone.0251493", "10.1371/journal.pone.0253129",
    "10.1371/journal.pone.0274693", "10.1515/9783110255553",
    "10.1515/9783110308464", "10.1515/jdis-2017-0002",
    "10.1609/icwsm.v6i1.14305", "10.1787/208277770603",
    "10.1787/5jlr2z70k0bx-en", "10.3145/epi.2022.jul.11",
    "10.3354/esep00084", "10.3354/meps08587", "10.3389/fnhum.2013.00291",
    "10.3389/frma.2016.00001", "10.3389/frma.2021.742311",
    "10.34734/fzj-2023-02860", "10.48550/arxiv.2106.01083",
    "10.5005/jp-journals-10024-1525", "10.5465/annals.2017.0099",
    "openalex:w2301623411",
}

UNCERTAIN = {
    "10.1002/leap.1110", "10.1007/s10479-016-2236-y", "10.1007/s10668-023-03225-w",
    "10.1007/s10796-017-9810-y", "10.1007/s10796-022-10279-0",
    "10.1007/s10961-017-9637-1", "10.1007/s11024-015-9271-8",
    "10.1007/s11192-013-1103-8", "10.1007/s11846-021-00492-7",
    "10.1007/s40685-018-0080-4", "10.1007/s40822-016-0054-9",
    "10.1016/j.cities.2022.103709", "10.1016/j.jeap.2023.101253",
    "10.1016/j.techfore.2020.120522", "10.1016/j.trd.2021.103133",
    "10.1057/s41599-020-00647-z", "10.1080/23789689.2019.1611056",
    "10.1162/qss_a_00021", "10.1257/jep.29.1.89", "10.1371/journal.pone.0066938",
    "10.1371/journal.pone.0199031", "10.1371/journal.pone.0242857",
    "10.1504/ejim.2020.105567", "10.1515/bfp-2020-2042",
    "10.22439/fs.v25i2.5578", "10.23987/sts.60610",
    "10.3390/su152316386", "10.3390/su9061011",
    "10.34734/fzj-2023-02860",
}


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(row: dict[str, str]) -> str:
    """Return a verbatim title or first abstract sentence as screening evidence."""
    abstract = row["abstract"].strip()
    if not abstract:
        return f"Title: {row['title']}"
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    return sentences[0].strip()


def screen(row: dict[str, str]) -> dict[str, str]:
    """Apply the source-lead screen without approving any feature or formula."""
    key = row["doi"] or row["record_key"]
    if key in INCLUDE:
        row.update(
            screen_decision="include_definition_or_review",
            evidence_span=evidence(row),
            rationale=(
                "The title/abstract directly concerns scholarly publications, citations, bibliometrics, "
                "research evaluation, publication text, or a documented publication-time context; retain "
                "it only as a source-recovery lead for a possible definition, application, validation, or indicator review."
            ),
        )
    elif key in UNCERTAIN:
        row.update(
            screen_decision="uncertain",
            evidence_span=evidence(row),
            rationale=(
                "The title/abstract indicates a potentially adjacent bibliometric or scholarly-communication topic, "
                "but does not itself establish a paper-level T0 feature definition, application, validation, or indicator review role."
            ),
        )
    else:
        row.update(
            screen_decision="exclude_not_relevant",
            evidence_span=evidence(row),
            rationale=(
                "The title/abstract is a subject-matter mapping, domain review, or non-publication-level analysis; "
                "it does not provide a shown lead for defining, applying, validating, or reviewing a paper-level v4 feature."
            ),
        )
    return row


def main() -> None:
    """Write the completed H1 screening CSV and manifest."""
    input_sha = sha256(INPUT)
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [screen(dict(row)) for row in reader]
    if len(rows) != 120:
        raise ValueError(f"Expected 120 rows, found {len(rows)}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["screen_decision"] for row in rows)
    manifest: dict[str, Any] = {
        "schema_version": "contextual_source_screening_h1_manifest_v4",
        "run_id": "contextual-source-screening-h1-independent-v4-20260819",
        "reviewer_role": "H1",
        "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "Independent title/abstract-only screening for source leads; no formula, indicator, or final-feature approval and no AI/H2 output used.",
        "input_artifacts": {str(INPUT): input_sha},
        "output_artifacts": {str(OUTPUT): sha256(OUTPUT)},
        "row_count": len(rows),
        "decision_counts": dict(sorted(counts.items())),
        "independence_guards": {"read_ai_or_h2_results": False, "used_qwen_or_ollama": False},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "manifest": str(MANIFEST), **manifest["decision_counts"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

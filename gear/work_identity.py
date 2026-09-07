"""Conservative exclusion of the target work and suspected target versions."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from gear.contracts import PaperMetadata, RetrievedWork


def normalize_identifier(value: str | None) -> str:
    return (
        re.sub(
            r"^(?:https?://(?:doi.org|openalex.org)/|doi:)",
            "",
            value or "",
            flags=re.IGNORECASE,
        )
        .strip()
        .casefold()
    )


def version_identity(work: RetrievedWork, target: PaperMetadata) -> str | None:
    for left, right in ((work.doi, target.doi), (work.work_id, target.openalex_id)):
        if left and right and normalize_identifier(left) == normalize_identifier(right):
            return "same_identifier"

    def title(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold()))

    if target.title and title(work.title) == title(target.title):
        # A title alone is a suspicion, never proof of identical research.
        return "suspected_target_version"
    overlap = {title(x) for x in work.authors} & {title(x) for x in target.authors}
    if (
        overlap
        and target.title
        and SequenceMatcher(None, title(work.title), title(target.title)).ratio() >= 0.9
    ):
        return "suspected_target_version_title_and_authors"
    return None

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
OUTPUT_DIR = ROOT / "outputs"
DATABASE_PATH = OUTPUT_DIR / "evidence_derived_v4.sqlite3"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def iter_csv(path: Path) -> Iterable[Dict[str, str]]:
    """Stream a UTF-8 CSV as dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_csv_iter(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> int:
    """Stream deterministic UTF-8 CSV rows and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping deterministically."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def local_environment_value(name: str) -> str:
    """Read one variable from the process or a non-executable .env parser."""
    process_value = os.environ.get(name, "").strip()
    if process_value:
        return process_value
    env_path = WORKSPACE_ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_name, raw_value = line.split("=", maxsplit=1)
        variable_name = raw_name.strip().removeprefix("export ").strip()
        if variable_name != name:
            continue
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        return value.strip()
    return ""


def normalize_doi(value: Any) -> str:
    """Normalize a DOI or DOI URL."""
    doi = str(value or "").strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


def normalize_text(value: Any) -> str:
    """Normalize Unicode text for identity and comparison."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_term(value: Any) -> str:
    """Normalize an English term without changing its construct meaning."""
    return normalize_text(value)


def term_match_key(value: Any) -> str:
    """Create a conservative English singular/plural matching key."""
    tokens: List[str] = []
    for token in normalize_term(value).split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif (
            len(token) > 4
            and token.endswith("s")
            and not token.endswith(("ss", "us", "is"))
        ):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def quote_search_term(term: str) -> str:
    """Quote a phrase or punctuation-bearing OpenAlex search term."""
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'"{escaped}"'
        if " " in escaped or any(not char.isalnum() for char in escaped)
        else escaped
    )


def or_block(terms: Iterable[str]) -> str:
    """Build a deterministic Boolean OR block without mutating evidence terms."""
    by_identity: Dict[str, str] = {}
    for term in terms:
        display = " ".join(
            unicodedata.normalize("NFC", str(term or "")).strip().split()
        )
        if display:
            by_identity.setdefault(display.casefold(), display)
    values = sorted(by_identity.values(), key=lambda value: (normalize_text(value), value))
    if not values:
        raise ValueError("A Boolean block cannot be empty")
    return "(" + " OR ".join(quote_search_term(value) for value in values) + ")"


def parse_bool(value: Any, field: str = "") -> bool:
    """Parse a strict CSV/JSON boolean."""
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field or 'value'} must be true or false: {value!r}")


def deterministic_ten_percent(identity: str) -> bool:
    """Select a stable ten-percent audit sample from an identity."""
    prefix = sha256_bytes(identity.encode("utf-8"))[:8]
    return int(prefix, 16) % 100 < 10


def raw_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    """Return raw categorical agreement."""
    if len(left) != len(right) or not left:
        raise ValueError("Agreement inputs must be nonempty and aligned")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Return unweighted Cohen's kappa."""
    observed = raw_agreement(left, right)
    categories = sorted(set(left) | set(right))
    size = len(left)
    expected = sum(
        (left.count(category) / size) * (right.count(category) / size)
        for category in categories
    )
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def gwet_ac1(left: Sequence[str], right: Sequence[str]) -> float:
    """Return nominal Gwet's AC1 agreement coefficient."""
    observed = raw_agreement(left, right)
    categories = sorted(set(left) | set(right))
    size = len(left)
    average_probabilities = [
        (left.count(category) + right.count(category)) / (2 * size)
        for category in categories
    ]
    category_count = len(categories)
    if category_count <= 1:
        return 1.0
    expected = sum(
        probability * (1 - probability)
        for probability in average_probabilities
    ) / (category_count - 1)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_from_pair_counts(
    pair_counts: Mapping[Tuple[str, str], int],
) -> Dict[str, float | int]:
    """Calculate agreement coefficients without materializing all labels."""
    size = sum(pair_counts.values())
    if size <= 0:
        raise ValueError("Agreement pair counts must be nonempty")
    categories = sorted(
        {left for left, _ in pair_counts}
        | {right for _, right in pair_counts}
    )
    left_counts = {
        category: sum(
            count
            for (left, _), count in pair_counts.items()
            if left == category
        )
        for category in categories
    }
    right_counts = {
        category: sum(
            count
            for (_, right), count in pair_counts.items()
            if right == category
        )
        for category in categories
    }
    observed = sum(
        count
        for (left, right), count in pair_counts.items()
        if left == right
    ) / size
    kappa_expected = sum(
        (left_counts[category] / size)
        * (right_counts[category] / size)
        for category in categories
    )
    if kappa_expected == 1:
        kappa = 1.0 if observed == 1 else 0.0
    else:
        kappa = (observed - kappa_expected) / (1 - kappa_expected)
    if len(categories) <= 1:
        ac1 = 1.0
    else:
        probabilities = [
            (left_counts[category] + right_counts[category]) / (2 * size)
            for category in categories
        ]
        ac1_expected = sum(
            probability * (1 - probability)
            for probability in probabilities
        ) / (len(categories) - 1)
        if ac1_expected == 1:
            ac1 = 1.0 if observed == 1 else 0.0
        else:
            ac1 = (observed - ac1_expected) / (1 - ac1_expected)
    return {
        "n": size,
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
    }

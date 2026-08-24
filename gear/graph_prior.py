"""Single-result adapter around cached or computed Graph scores."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from .calibration import CalibrationService, normalize_openalex_id
from .config import GearConfig, load_config
from .contracts import PaperIR
from .graph_prior_contracts import GraphResultV3, GraphResultV4


class GraphUnavailableError(RuntimeError):
    """Raised when the Graph module cannot return a complete V3 result."""


class GraphScorer(Protocol):
    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphResultV4: ...


class GraphResultTable:
    """Resolve existing Graph output without exposing its storage provenance."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    def lookup(self, paper_ir: PaperIR) -> GraphResultV4 | None:
        if not self.path.is_file():
            return None
        if self.path.suffix.casefold() == ".jsonl":
            return self._lookup_jsonl(paper_ir)
        return self._lookup_parquet(paper_ir)

    def _lookup_jsonl(self, paper_ir: PaperIR) -> GraphResultV4 | None:
        identifiers = _paper_identifiers(paper_ir)
        matches: list[GraphResultV4] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    result = graph_result_v4(payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid Graph result record at line {line_number}"
                    ) from exc
                if _normalize_identifier(result.paper_id) in identifiers:
                    matches.append(result)
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("Graph result file has ambiguous paper identity")
        result = matches[0]
        if result.paper_id != paper_ir.paper_id:
            return result.model_copy(update={"paper_id": paper_ir.paper_id})
        return result

    def _lookup_parquet(self, paper_ir: PaperIR) -> GraphResultV4 | None:
        import pandas as pd

        frame = pd.read_parquet(self.path)
        required = {
            "aspr_score",
            "p_uptake",
            "conditional_diffusion",
            "feature_coverage",
        }
        if not required.issubset(frame.columns):
            raise ValueError("Graph result table lacks required V3 fields")
        identifiers = _paper_identifiers(paper_ir)
        matches = frame.loc[
            frame.apply(lambda row: bool(identifiers & _row_identifiers(row)), axis=1)
        ]
        if matches.empty:
            return None
        if len(matches) != 1:
            raise ValueError("Graph result table has ambiguous paper identity")
        row = matches.iloc[0]
        return GraphResultV4(
            paper_id=paper_ir.paper_id,
            score_0_100=float(row["aspr_score"]),
            p_uptake=float(row["p_uptake"]),
            conditional_diffusion=float(row["conditional_diffusion"]),
            feature_coverage=float(row["feature_coverage"]),
        )


class GraphService:
    """Return one V3 result regardless of how the Graph module produced it."""

    def __init__(
        self,
        config: GearConfig | None = None,
        *,
        calibration_factory: Callable[[], Any] | None = None,
        result_table: GraphResultTable | None = None,
    ) -> None:
        self.config = config or load_config()
        self._calibration_factory = calibration_factory
        configured_path = self.config.graph_results_path
        self._result_table = result_table or (
            GraphResultTable(self.config.resolve_path(configured_path))
            if configured_path is not None
            else None
        )
        self._calibration: Any | None = None
        self.last_failure: str | None = None
        self.last_packet: Any | None = None

    def _service(self) -> Any:
        if self._calibration is None:
            self._calibration = (
                self._calibration_factory()
                if self._calibration_factory is not None
                else CalibrationService(self.config)
            )
        return self._calibration

    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphResultV4:
        self.last_failure = None
        self.last_packet = None
        try:
            if self._result_table is not None:
                stored = self._result_table.lookup(paper_ir)
                if stored is not None:
                    return stored
            packet = self._service().build_packet(paper_ir, cutoff_date=cutoff_date)
            self.last_packet = packet
            result = graph_result_from_calibration(packet)
            if result.paper_id != paper_ir.paper_id:
                raise ValueError("Graph result paper_id does not match PaperIR")
            return result
        except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as exc:
            self.last_failure = f"{type(exc).__name__}:{exc}"
            raise GraphUnavailableError(self.last_failure) from exc


def graph_result_from_calibration(packet: Any) -> GraphResultV4:
    """Project any complete Graph packet into the source-agnostic V4 contract."""
    forecast = packet.forecast
    values = {
        "score_0_100": forecast.aspr_score_0_100,
        "p_uptake": forecast.p_uptake,
        "conditional_diffusion": forecast.conditional_diffusion,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise GraphUnavailableError(
            f"Graph module returned an incomplete result: {sorted(missing)}"
        )
    return GraphResultV4(
        paper_id=packet.paper_id,
        score_0_100=float(values["score_0_100"]),
        p_uptake=float(values["p_uptake"]),
        conditional_diffusion=float(values["conditional_diffusion"]),
        feature_coverage=float(packet.reliability.feature_coverage),
    )


def graph_result_v4(
    value: Mapping[str, Any] | GraphResultV3 | GraphResultV4,
) -> GraphResultV4:
    """Read V4 directly or migrate a complete V3 result with empty hints."""
    if isinstance(value, GraphResultV4):
        return value
    if isinstance(value, GraphResultV3):
        payload = value.model_dump(exclude={"contract"})
    else:
        payload = dict(value)
        if payload.get("contract") == "aspr_graph_result_v4":
            return GraphResultV4.model_validate(payload)
        payload.pop("contract", None)
    return GraphResultV4(**payload, seed_work_ids=[], search_terms=[])


def build_graph_search_hints(target: Any, context: Any) -> tuple[list[str], list[str]]:
    """Derive stable, cutoff-safe work seeds and lexical hints without a model call."""
    target_id = str(getattr(target, "work_id", getattr(target, "paper_id", "")))
    references = list(getattr(target, "references", ()) or ())
    target_year = int(
        getattr(target, "publication_year", getattr(target, "year", 0)) or 0
    )
    coupling = getattr(context, "bibliographic_coupling_index", {}) or {}
    counts: Counter[str] = Counter()
    fields: list[str] = []
    for reference in references:
        year = int(
            getattr(reference, "publication_year", getattr(reference, "year", 0)) or 0
        )
        if target_year and year and year >= target_year:
            continue
        reference_id = str(
            getattr(reference, "reference_id", getattr(reference, "work_id", "")) or ""
        )
        coupling_ids = coupling.get(reference_id, ())
        if not coupling_ids and re.fullmatch(r"W\d+", reference_id, re.IGNORECASE):
            coupling_ids = coupling.get(
                f"https://openalex.org/{reference_id.upper()}", ()
            )
        for work_id in coupling_ids:
            work_id = str(work_id)
            if work_id and _normalize_identifier(work_id) != _normalize_identifier(
                target_id
            ):
                counts[work_id] += 1
        field = str(getattr(reference, "field_id", "") or "").strip()
        if field:
            fields.append(field)
    seeds = [
        work_id
        for work_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :8
        ]
    ]
    title = str(getattr(target, "title", "") or "")
    words = re.findall(r"[A-Za-z0-9]+", title.casefold())
    seen = {
        " ".join(item) if isinstance(item, tuple) else str(item)
        for item in (getattr(context, "seen_title_bigrams", set()) or set())
    }
    bigrams = [f"{left} {right}" for left, right in pairwise(words)]
    terms: list[str] = []
    for term in [*(term for term in bigrams if term not in seen), *fields]:
        if term and term not in terms:
            terms.append(term)
        if len(terms) == 8:
            break
    return seeds, terms


def _paper_identifiers(paper_ir: PaperIR) -> set[str]:
    values = {
        paper_ir.paper_id,
        paper_ir.metadata.openalex_id,
        paper_ir.metadata.doi,
    }
    return {_normalize_identifier(value) for value in values if value}


def _row_identifiers(row: Any) -> set[str]:
    values = [row.get(name) for name in ("paper_id", "openalex_id", "doi")]
    return {
        _normalize_identifier(value)
        for value in values
        if value is not None and str(value).strip()
    }


def _normalize_identifier(value: Any) -> str:
    text = str(value).strip()
    openalex = normalize_openalex_id(text)
    if openalex is not None:
        return openalex.casefold()
    lowered = text.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            return lowered[len(prefix) :]
    return lowered


# Compatibility names remain importable, but current runtime semantics are V3.
GraphPriorService = GraphService
graph_prior_from_calibration = graph_result_from_calibration


__all__ = [
    "GraphPriorService",
    "GraphResultTable",
    "GraphScorer",
    "GraphService",
    "GraphUnavailableError",
    "build_graph_search_hints",
    "graph_prior_from_calibration",
    "graph_result_from_calibration",
    "graph_result_v4",
]

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
from .graph_prior_contracts import (
    GraphResultV3,
    GraphResultV4,
    GraphRuntimePacketV1,
    GraphTopologySeedV1,
)


class GraphUnavailableError(RuntimeError):
    """Raised when the Graph module cannot return a legal runtime packet."""


class GraphScorer(Protocol):
    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphRuntimePacketV1: ...


class GraphResultTable:
    """Resolve existing Graph output without exposing its storage provenance."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    def lookup(self, paper_ir: PaperIR) -> GraphRuntimePacketV1 | None:
        if not self.path.is_file():
            return None
        if self.path.suffix.casefold() == ".jsonl":
            return self._lookup_jsonl(paper_ir)
        packet_path = self.path.parent / "graph_runtime_packets.jsonl"
        if packet_path.is_file():
            return GraphResultTable(packet_path)._lookup_jsonl(paper_ir)
        return self._lookup_parquet(paper_ir)

    def _lookup_jsonl(self, paper_ir: PaperIR) -> GraphRuntimePacketV1 | None:
        identifiers = _paper_identifiers(paper_ir)
        matches: list[GraphRuntimePacketV1] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    result = graph_runtime_packet(payload)
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

    def _lookup_parquet(self, paper_ir: PaperIR) -> GraphRuntimePacketV1 | None:
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
        feature_columns = [
            str(name) for name in frame.columns if re.fullmatch(r"EF\d{4}", str(name))
        ]
        feature_path = self.path.parent / "score_features.parquet"
        if not feature_columns and feature_path.is_file():
            feature_frame = pd.read_parquet(feature_path)
            feature_matches = feature_frame.loc[
                feature_frame.apply(
                    lambda item: bool(identifiers & _row_identifiers(item)), axis=1
                )
            ]
            if len(feature_matches) == 1:
                feature_row = feature_matches.iloc[0]
                feature_columns = [
                    str(name)
                    for name in feature_frame.columns
                    if re.fullmatch(r"EF\d{4}", str(name))
                ]
                row = row.copy()
                for name in feature_columns:
                    row[name] = feature_row[name]
        feature_values = {
            name: None if pd.isna(row[name]) else float(row[name])
            for name in feature_columns
        }
        missing = [name for name, value in feature_values.items() if value is None]
        raw_expected = row.get(
            "raw_expected_diffusion", row.get("raw_prediction_score")
        )
        if raw_expected is None or pd.isna(raw_expected):
            raw_expected = float(row["p_uptake"]) * float(row["conditional_diffusion"])
        return GraphRuntimePacketV1(
            paper_id=paper_ir.paper_id,
            score_0_100=float(row["aspr_score"]),
            p_uptake=float(row["p_uptake"]),
            conditional_diffusion=float(row["conditional_diffusion"]),
            raw_expected_diffusion=float(raw_expected),
            feature_values=feature_values,
            missing_feature_ids=missing,
        )


class GraphService:
    """Return one runtime packet regardless of how Graph produced it."""

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

    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphRuntimePacketV1:
        self.last_failure = None
        self.last_packet = None
        try:
            if self._result_table is not None:
                stored = self._result_table.lookup(paper_ir)
                if stored is not None:
                    return stored
            packet = self._service().build_packet(paper_ir, cutoff_date=cutoff_date)
            self.last_packet = packet
            result = graph_runtime_packet_from_calibration(packet)
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


def graph_runtime_packet_from_calibration(packet: Any) -> GraphRuntimePacketV1:
    """Preserve the score decomposition and full measured feature profile."""
    forecast = packet.forecast
    required = {
        "score_0_100": forecast.aspr_score_0_100,
        "p_uptake": forecast.p_uptake,
        "conditional_diffusion": forecast.conditional_diffusion,
    }
    missing_required = [name for name, value in required.items() if value is None]
    if missing_required:
        raise GraphUnavailableError(
            f"Graph module returned an incomplete result: {sorted(missing_required)}"
        )
    measurement = packet.measurement
    feature_values: dict[str, Any] = {}
    for group in (
        measurement.substantive_innovation,
        measurement.t0_potential,
        measurement.opportunity,
        measurement.context_control,
    ):
        feature_values.update(group)
    if isinstance(feature_values.get("EF0197"), str):
        feature_values["EF0197"] = None
    missing_feature_ids = sorted(
        {
            *getattr(packet.reliability, "missing_features", []),
            *(name for name, value in feature_values.items() if value is None),
        }
    )
    return GraphRuntimePacketV1(
        paper_id=packet.paper_id,
        score_0_100=float(required["score_0_100"]),
        raw_expected_diffusion=(
            float(forecast.raw_expected_diffusion)
            if forecast.raw_expected_diffusion is not None
            else float(required["p_uptake"]) * float(required["conditional_diffusion"])
        ),
        p_uptake=float(required["p_uptake"]),
        conditional_diffusion=float(required["conditional_diffusion"]),
        feature_values=feature_values,
        historical_bands=dict(measurement.historical_bands),
        missing_feature_ids=missing_feature_ids,
        diagnostic_flags=list(
            dict.fromkeys(
                [
                    *getattr(packet.reliability, "drift_flags", []),
                    *getattr(packet.reliability, "quality_flags", []),
                ]
            )
        ),
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


def graph_runtime_packet(
    value: Mapping[str, Any] | GraphResultV3 | GraphResultV4 | GraphRuntimePacketV1,
) -> GraphRuntimePacketV1:
    """Read the runtime packet or migrate V3/V4 without activating search terms."""
    if isinstance(value, GraphRuntimePacketV1):
        return value
    if isinstance(value, (GraphResultV3, GraphResultV4)):
        payload = value.model_dump(mode="json")
    else:
        payload = dict(value)
        if payload.get("contract") == "aspr_graph_runtime_packet_v1":
            return GraphRuntimePacketV1.model_validate(payload)
    payload.pop("contract", None)
    seeds = [
        GraphTopologySeedV1(work_id=work_id)
        for work_id in payload.pop("seed_work_ids", [])
        if str(work_id).strip()
    ]
    payload.pop("search_terms", None)
    payload.pop("feature_coverage", None)
    return GraphRuntimePacketV1(
        **payload,
        raw_expected_diffusion=float(payload["p_uptake"])
        * float(payload["conditional_diffusion"]),
        topology_seeds=seeds,
    )


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


def build_graph_topology_seeds(
    target: Any, context: Any, *, maximum: int = 8
) -> list[GraphTopologySeedV1]:
    """Build structured coupling anchors without lexical term generation."""
    target_id = str(getattr(target, "work_id", getattr(target, "paper_id", "")))
    coupling = getattr(context, "bibliographic_coupling_index", {}) or {}
    metadata = getattr(context, "reference_metadata", {}) or {}
    shared: dict[str, set[str]] = {}
    fields: dict[str, set[str]] = {}
    for reference in list(getattr(target, "references", ()) or ()):
        reference_id = str(getattr(reference, "reference_id", "") or "")
        candidates = coupling.get(reference_id, ())
        if not candidates and re.fullmatch(r"W\d+", reference_id, re.IGNORECASE):
            candidates = coupling.get(
                f"https://openalex.org/{reference_id.upper()}", ()
            )
        for work_id_value in candidates:
            work_id = str(work_id_value)
            if not work_id or _normalize_identifier(work_id) == _normalize_identifier(
                target_id
            ):
                continue
            shared.setdefault(work_id, set()).add(reference_id)
            field_id = str(getattr(reference, "field_id", "") or "")
            if field_id:
                fields.setdefault(work_id, set()).add(field_id)
    ranked = sorted(shared, key=lambda work_id: (-len(shared[work_id]), work_id))[
        :maximum
    ]
    return [
        GraphTopologySeedV1(
            work_id=work_id,
            publication_year=getattr(metadata.get(work_id), "publication_year", None),
            shared_reference_count=len(shared[work_id]),
            shared_reference_ids=sorted(shared[work_id]),
            anchor_field_ids=sorted(fields.get(work_id, set())),
        )
        for work_id in ranked
    ]


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


# Compatibility names remain importable; current runtime uses the packet contract.
GraphPriorService = GraphService
graph_prior_from_calibration = graph_result_from_calibration


__all__ = [
    "GraphPriorService",
    "GraphResultTable",
    "GraphScorer",
    "GraphService",
    "GraphUnavailableError",
    "build_graph_search_hints",
    "build_graph_topology_seeds",
    "graph_prior_from_calibration",
    "graph_result_from_calibration",
    "graph_result_v4",
    "graph_runtime_packet",
    "graph_runtime_packet_from_calibration",
]

"""Dual-score adapter shared by online scoring and all figure views."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import FeatureSpec, ScorePacket
from .models import DomainYearCalibrator, FittedCandidateModel, empirical_cdf
from .release import load_release


DEFAULT_CLAIM_SCOPE = (
    "42 Nature Portfolio sources; pre-publication-year graph; "
    "validated conditionally among papers with at least 10 future citers"
)


def _optional_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _quality_flags(value: Any) -> Tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(";") if item.strip())
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def build_score_packets(
    frame: pd.DataFrame,
    *,
    horizon: Optional[int] = None,
    model_version: str,
    feature_version: str = "nature-multihorizon-feature-v1",
    paper_id_col: str = "paper_id",
    mechanism_prefix: str = "mechanism__",
    claim_scope: str = DEFAULT_CLAIM_SCOPE,
) -> List[ScorePacket]:
    """Validate a score table at the stable ``ScorePacket`` boundary."""

    if paper_id_col not in frame:
        raise ValueError(f"Missing paper id column: {paper_id_col}")
    channel_columns = [column for column in frame if column.startswith(mechanism_prefix)]
    packets: List[ScorePacket] = []
    for _, row in frame.iterrows():
        row_horizon = int(horizon if horizon is not None else row.get("horizon"))
        channels = {
            column.removeprefix(mechanism_prefix): float(row[column])
            for column in channel_columns
            if _optional_float(row[column]) is not None
        }
        packet = ScorePacket(
            paper_id=str(row[paper_id_col]),
            horizon=row_horizon,
            mechanism_channels=channels,
            score_mechanism=_optional_float(row.get("score_mechanism")),
            score_performance_raw=_optional_float(row.get("score_performance_raw")),
            score_performance_calibrated=_optional_float(row.get("score_performance_calibrated")),
            score_performance_percentile=_optional_float(row.get("score_performance_percentile")),
            model_version=str(row.get("model_version", model_version)),
            feature_version=str(row.get("feature_version", feature_version)),
            quality_flags=_quality_flags(row.get("quality_flags")),
            claim_scope=str(row.get("claim_scope", claim_scope)),
        )
        packets.append(packet)
    return packets


def packets_to_frame(packets: Sequence[ScorePacket]) -> pd.DataFrame:
    """Flatten packets without losing version or claim-scope fields."""

    rows = []
    for packet in packets:
        row = {
            "paper_id": packet.paper_id,
            "horizon": packet.horizon,
            "score_mechanism": packet.score_mechanism,
            "score_performance_raw": packet.score_performance_raw,
            "score_performance_calibrated": packet.score_performance_calibrated,
            "score_performance_percentile": packet.score_performance_percentile,
            "model_version": packet.model_version,
            "feature_version": packet.feature_version,
            "quality_flags": ";".join(packet.quality_flags),
            "claim_scope": packet.claim_scope,
        }
        row.update({f"mechanism__{name}": value for name, value in packet.mechanism_channels.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def score_frame(
    frame: pd.DataFrame,
    *,
    horizon: int,
    mechanism_model: FittedCandidateModel,
    performance_model: FittedCandidateModel,
    calibrator: DomainYearCalibrator,
    performance_percentile_reference: Optional[Sequence[float]] = None,
    model_version: str,
    feature_version: str = "nature-multihorizon-feature-v1",
    quality_flags_col: str = "quality_flags",
    claim_scope: str = DEFAULT_CLAIM_SCOPE,
) -> tuple[pd.DataFrame, List[ScorePacket]]:
    """Score a publication-time feature frame with the frozen dual models."""

    output = frame[[column for column in ("paper_id", "domain12", "publication_year", quality_flags_col) if column in frame]].copy()
    raw = performance_model.predict(frame)
    calibrated = calibrator.predict(frame, raw)
    reference = np.asarray(
        performance_percentile_reference
        if performance_percentile_reference is not None
        else calibrator.calibrated_reference_,
        dtype=float,
    )
    output["horizon"] = int(horizon)
    output["score_mechanism"] = mechanism_model.predict(frame)
    output["score_performance_raw"] = raw
    output["score_performance_calibrated"] = calibrated
    # Percentiles use the calibrated training distribution when explicitly
    # supplied; otherwise the fold-local raw-score reference is the safe
    # fallback and remains monotone in the performance score.
    output["score_performance_percentile"] = np.clip(empirical_cdf(calibrated, reference), 0.0, 1.0)
    output["model_version"] = model_version
    output["feature_version"] = feature_version
    output["claim_scope"] = claim_scope
    channels = mechanism_model.mechanism_channels(frame)
    for channel in channels:
        output[f"mechanism__{channel}"] = channels[channel].to_numpy(float)
    if quality_flags_col in output and quality_flags_col != "quality_flags":
        output = output.rename(columns={quality_flags_col: "quality_flags"})
    packets = build_score_packets(
        output,
        horizon=horizon,
        model_version=model_version,
        feature_version=feature_version,
        claim_scope=claim_scope,
    )
    return output, packets


class ScorePacketAdapter:
    """Small compatibility facade for release and online callers."""

    @staticmethod
    def from_frame(frame: pd.DataFrame, **kwargs: Any) -> List[ScorePacket]:
        return build_score_packets(frame, **kwargs)

    @staticmethod
    def to_frame(packets: Sequence[ScorePacket]) -> pd.DataFrame:
        return packets_to_frame(packets)


class FrozenReleaseScorer:
    """Read-only lookup and inference boundary for a frozen evidence release."""

    def __init__(self, release_path: Path, *, horizon: int = 5) -> None:
        import joblib

        self.release = load_release(Path(release_path), require_frozen=True)
        self.horizon = int(horizon)
        score_path = self.release.artifact("paper_scores")
        self.scores = pd.read_parquet(score_path)
        self.scores = self.scores[self.scores["horizon"].eq(self.horizon)].copy()
        if self.scores.empty:
            raise ValueError(f"Frozen release has no paper scores for horizon={self.horizon}")
        self.bundle = joblib.load(self.release.artifact(f"model_bundle_tau{self.horizon}"))
        if int(self.bundle.get("horizon", -1)) != self.horizon:
            raise ValueError("Frozen model bundle horizon does not match scorer")

    @staticmethod
    def _normalize_doi(value: str) -> str:
        return str(value or "").lower().replace("https://doi.org/", "").strip()

    def lookup(
        self,
        *,
        paper_id: Optional[str] = None,
        doi: Optional[str] = None,
    ) -> ScorePacket:
        """Return one validated ScorePacket by immutable paper ID or DOI."""
        selected = self.scores
        if paper_id:
            selected = selected[selected["paper_id"].astype(str).eq(str(paper_id))]
        elif doi and "doi" in selected:
            normalized = self._normalize_doi(doi)
            selected = selected[
                selected["doi"].map(self._normalize_doi).eq(normalized)
            ]
        else:
            raise ValueError("paper_id or doi is required")
        if len(selected) != 1:
            raise KeyError(
                f"Expected one frozen score row, found {len(selected)} for paper_id={paper_id!r}, doi={doi!r}"
            )
        return build_score_packets(
            selected,
            horizon=self.horizon,
            model_version=str(selected.iloc[0]["model_version"]),
        )[0]

    def lookup_frame(self, paper_ids: Sequence[str]) -> pd.DataFrame:
        """Return versioned score rows without silently dropping unknown IDs."""
        requested = pd.DataFrame({"paper_id": list(paper_ids)})
        result = requested.merge(
            self.scores,
            on="paper_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if not result["_merge"].eq("both").all():
            missing = result.loc[result["_merge"].ne("both"), "paper_id"].astype(str).tolist()
            raise KeyError(f"Frozen release has no score for paper IDs: {missing}")
        return result.drop(columns=["_merge"])

    def score_features(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[ScorePacket]]:
        """Score unseen papers from the locked 18 publication-prior features."""
        if "source_max_year" not in frame or "publication_year" not in frame:
            raise ValueError("Online scoring requires source_max_year and publication_year provenance")
        source_year = pd.to_numeric(frame["source_max_year"], errors="coerce")
        publication_year = pd.to_numeric(frame["publication_year"], errors="coerce")
        if not (source_year.notna() & publication_year.notna() & source_year.lt(publication_year)).all():
            raise ValueError("Online features violate source_max_year < publication_year")
        scoring_frame = frame.copy()
        if "quality_flags" not in scoring_frame:
            scoring_frame["quality_flags"] = ""
        scoring_frame["quality_flags"] = scoring_frame["quality_flags"].fillna("").map(
            lambda value: ";".join(
                item
                for item in (
                    str(value).strip(";"),
                    "out_of_cohort_extrapolation",
                    "future_citer_eligibility_unknown",
                )
                if item
            )
        )
        return score_frame(
            scoring_frame,
            horizon=self.horizon,
            mechanism_model=self.bundle["mechanism_model"],
            performance_model=self.bundle["performance_model"],
            calibrator=self.bundle["calibrator"],
            performance_percentile_reference=self.bundle[
                "performance_percentile_reference"
            ],
            model_version=str(self.bundle["model_version"]),
            feature_version=str(self.bundle["feature_version"]),
            claim_scope=(
                "42 Nature Portfolio sources; pre-publication-year graph; "
                "out-of-cohort extrapolation; future-citer eligibility unknown"
            ),
        )


def build_paper_scores(
    model_frame: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    model_ledger: Optional[pd.DataFrame] = None,
    *,
    feature_version: str = "nature-multihorizon-feature-v1",
    claim_scope: str = DEFAULT_CLAIM_SCOPE,
) -> pd.DataFrame:
    """Materialize one release score row per paper/horizon from selected OOF.

    The function intentionally does not refit a model.  It is the normalized
    bridge used by the evaluate stage after nested training has completed.
    """

    del model_ledger  # the selected model and fold are already explicit in OOF
    if "is_selected" not in oof_predictions:
        raise ValueError("Long OOF table is missing is_selected")
    selected = oof_predictions.loc[
        oof_predictions["is_selected"].fillna(False).astype(bool)
    ].copy()
    if selected.duplicated(["paper_id", "horizon"]).any():
        raise ValueError("Selected OOF contains duplicate paper/horizon rows")
    selected = selected.rename(
        columns={
            "prediction_raw": "score_performance_raw",
            "prediction_calibrated": "score_performance_calibrated",
            "prediction_percentile": "score_performance_percentile",
            "model_id": "model_version",
        }
    )
    metadata_columns = [
        column
        for column in ("paper_id", "horizon", "quality_flags")
        if column in model_frame
    ]
    if "quality_flags" in metadata_columns and "quality_flags" not in selected:
        selected = selected.merge(
            model_frame[metadata_columns].drop_duplicates(["paper_id", "horizon"]),
            on=["paper_id", "horizon"],
            how="left",
            validate="one_to_one",
        )
    selected["feature_version"] = feature_version
    selected["claim_scope"] = claim_scope
    if "quality_flags" not in selected:
        selected["quality_flags"] = ""
    required = [
        "paper_id",
        "horizon",
        "score_mechanism",
        "score_performance_raw",
        "score_performance_calibrated",
        "score_performance_percentile",
        "model_version",
        "feature_version",
        "quality_flags",
        "claim_scope",
    ]
    mechanisms = sorted(column for column in selected if column.startswith("mechanism__"))
    return selected[required + mechanisms].reset_index(drop=True)

"""Fail-closed publication eligibility, separate from available draft panels."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PanelSpec:
    figure: int
    panel: str
    question: str
    unit: str
    cohort_id: str
    source_hashes: dict[str, str]
    metric_version: str
    required_artifacts: tuple[str, ...]
    evidence_status: str
    claims_allowed: tuple[str, ...]
    claims_not_allowed: tuple[str, ...]
    renderer: str
    display_tokens: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIREMENTS = {
    1: ('clean_atlas', 'frozen_local_cases'),
    2: ('mechanism_source', 'terminal_source_trace'),
    3: ('structural_diagnostics', 'independent_relation_audit'),
    4: ('paired_benchmark', 'identity_blind_reevaluation'),
    5: ('full_fusion_reports', 'independent_fusion_change_audit'),
    6: ('uniform_path_variants', 'paired_ablation_outcomes', 'same_text_joint_control'),
    7: ('execution_evidence_separation', 'independent_sufficiency_audit', 'perturbation_outcomes'),
    8: ('deduplicated_phrase_baselines', 'claim_direction_predictions', 'historical_future_adoption_labels'),
    9: ('field_provenance', 'paper_normalized_roles'),
    10: ('traceable_case', 'main_reviewer_reference'),
}


def publication_missing(figure: int, artifacts: dict[str, bool]) -> list[str]:
    """Presence must be explicitly attested; filenames alone cannot certify science."""
    return [name for name in REQUIREMENTS[figure] if artifacts.get(name) is not True]


def require_publication(figures: list[int], artifacts: dict[str, bool]) -> None:
    missing = {str(f): publication_missing(f, artifacts) for f in figures if publication_missing(f, artifacts)}
    if missing:
        raise ValueError(f'Publication blocked: required evidence is missing: {missing}')

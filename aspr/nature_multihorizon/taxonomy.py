"""Deterministic mapping of paper-level OpenAlex topics to twelve domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class DomainDefinition:
    domain_id: str
    label: str
    positive_terms: Tuple[str, ...]


# Order is intentional: narrow domains are evaluated before broad parents.
DOMAIN_DEFINITIONS: Tuple[DomainDefinition, ...] = (
    DomainDefinition(
        "neuroscience",
        "Neuroscience",
        (
            "neurosc",
            "brain",
            "neural circuit",
            "neural activity",
            "cognitive neuroscience",
            "neurobiology",
        ),
    ),
    DomainDefinition(
        "ecology_evolution_microbiology",
        "Ecology, evolution and microbiology",
        (
            "ecolog",
            "evolution",
            "biodiversity",
            "microbiol",
            "microbiome",
            "zoolog",
            "botany",
            "marine biology",
        ),
    ),
    DomainDefinition(
        "clinical_health",
        "Clinical and health sciences",
        (
            "medicine",
            "clinical",
            "health",
            "disease",
            "oncology",
            "cardiology",
            "epidemiol",
            "surgery",
            "pharmacolog",
            "nursing",
        ),
    ),
    DomainDefinition(
        "astronomy_space",
        "Astronomy and space sciences",
        ("astronom", "astrophys", "cosmolog", "space science", "planetary science"),
    ),
    DomainDefinition(
        "earth_climate_environment",
        "Earth, climate and environmental sciences",
        (
            "earth",
            "geolog",
            "geophys",
            "geoscience",
            "climate",
            "environment",
            "oceanograph",
            "atmospher",
            "hydrolog",
        ),
    ),
    DomainDefinition(
        "materials_nanoscience",
        "Materials and nanoscience",
        ("materials", "nanosc", "nanotech", "metallurg", "ceramic", "polymer science"),
    ),
    DomainDefinition(
        "computer_science_ai",
        "Computer science and AI",
        (
            "computer science",
            "artificial intelligence",
            "machine learning",
            "data science",
            "software",
            "information systems",
            "computer vision",
            "natural language processing",
        ),
    ),
    DomainDefinition(
        "mathematics_statistics",
        "Mathematics and statistics",
        ("mathemat", "statistic", "probability", "operations research", "optimization theory"),
    ),
    DomainDefinition(
        "engineering_energy",
        "Engineering and energy",
        (
            "engineering",
            "energy",
            "robotic",
            "electrical",
            "mechanical",
            "civil engineering",
            "renewable",
        ),
    ),
    DomainDefinition(
        "chemistry",
        "Chemistry",
        (
            "chemistry",
            "chemical",
            "catalysis",
            "spectroscopy",
            "organic synthesis",
        ),
    ),
    DomainDefinition(
        "physics",
        "Physics",
        (
            "physics",
            "quantum",
            "condensed matter",
            "particle",
            "optics",
            "thermodynamic",
        ),
    ),
    DomainDefinition(
        "life_molecular",
        "Life and molecular sciences",
        (
            "life sciences",
            "biolog",
            "biochem",
            "genetic",
            "genomic",
            "molecular",
            "cell biology",
            "cellular biology",
            "stem cell",
            "cell signaling",
            "immunolog",
            "biotechnology",
            "agricultural",
        ),
    ),
)

DOMAIN_IDS: Tuple[str, ...] = tuple(item.domain_id for item in DOMAIN_DEFINITIONS)
DOMAIN_LABELS: Dict[str, str] = {
    item.domain_id: item.label for item in DOMAIN_DEFINITIONS
}

DIRECT_FIELD_DOMAINS: Dict[str, str] = {
    "artificial intelligence": "computer_science_ai",
    "astronomy and astrophysics": "astronomy_space",
    "biology": "life_molecular",
    "biochemistry, genetics and molecular biology": "life_molecular",
    "chemistry": "chemistry",
    "chemical engineering": "chemistry",
    "computer science": "computer_science_ai",
    "dentistry": "clinical_health",
    "energy": "engineering_energy",
    "engineering": "engineering_energy",
    "environmental science": "earth_climate_environment",
    "earth sciences": "earth_climate_environment",
    "health professions": "clinical_health",
    "materials science": "materials_nanoscience",
    "mathematics": "mathematics_statistics",
    "medicine": "clinical_health",
    "microbiology": "ecology_evolution_microbiology",
    "neuroscience": "neuroscience",
    "nursing": "clinical_health",
    "pharmacology, toxicology and pharmaceutics": "clinical_health",
    "physics": "physics",
    "statistics": "mathematics_statistics",
    "veterinary": "clinical_health",
}
NONNATURAL_FIELDS = frozenset(
    {
        "arts and humanities",
        "business, management and accounting",
        "economics, econometrics and finance",
        "social sciences",
    }
)


def _topic_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return " | ".join(str(row.get(field) or "").strip().casefold() for field in fields)


def _official_field(row: Mapping[str, Any]) -> str:
    for column in ("openalex_primary_field", "primary_field"):
        value = str(row.get(column) or "").strip().casefold()
        if value:
            return value
    return ""


def _specific_fallback(specific_text: str) -> Tuple[str, str] | None:
    if any(
        term in specific_text
        for term in (
            "neural network",
            "graph neural",
            "deep learning",
            "machine learning",
            "artificial intelligence",
        )
    ):
        return "computer_science_ai", "topic_fallback:machine-learning"
    for definition in DOMAIN_DEFINITIONS:
        for term in definition.positive_terms:
            if term in specific_text:
                return definition.domain_id, f"topic_fallback:{term}"
    return None


def assign_domain12(row: Mapping[str, Any]) -> Tuple[str, str]:
    """Return ``(domain_id, reason)`` from paper-level topic metadata only."""

    specific_text = _topic_text(
        row,
        (
            "primary_topic",
            "display_topic_label",
            "openalex_primary_subfield",
            "primary_subfield",
        ),
    )
    official_field = _official_field(row)
    broad_text = _topic_text(row, ("openalex_domain",))
    text = f"{specific_text} | {broad_text}"
    if not text.replace("|", "").strip() and not official_field:
        return "unmapped", "missing_paper_topic_metadata"

    if official_field in DIRECT_FIELD_DOMAINS:
        domain = DIRECT_FIELD_DOMAINS[official_field]
        return domain, f"field_hierarchy:{official_field}"
    if official_field in NONNATURAL_FIELDS:
        return (
            "out_of_scope_nonnatural",
            f"field_hierarchy_nonnatural:{official_field}",
        )
    if official_field == "physics and astronomy":
        if any(
            token in specific_text
            for token in ("astronom", "astrophys", "cosmolog", "exoplanet", "space science")
        ):
            return "astronomy_space", "field_hierarchy:physics-and-astronomy/astronomy"
        return "physics", "field_hierarchy:physics-and-astronomy/physics"
    if official_field == "earth and planetary sciences":
        if any(
            token in specific_text
            for token in ("astronom", "astrophys", "cosmolog", "exoplanet", "space science")
        ):
            return "astronomy_space", "field_hierarchy:earth-and-planetary/space"
        return "earth_climate_environment", "field_hierarchy:earth-and-planetary/earth"
    if official_field in {
        "agricultural and biological sciences",
        "immunology and microbiology",
    }:
        if any(
            token in specific_text
            for token in (
                "ecolog",
                "evolution",
                "biodiversity",
                "microbiol",
                "microbiome",
                "zoolog",
                "botany",
                "marine biology",
            )
        ):
            return (
                "ecology_evolution_microbiology",
                f"field_hierarchy:{official_field}/ecology-microbiology",
            )
        return "life_molecular", f"field_hierarchy:{official_field}/life"
    if official_field == "psychology":
        if any(token in specific_text for token in ("neurosc", "brain", "neurobiology")):
            return "neuroscience", "field_hierarchy:psychology/neuroscience"
        return "out_of_scope_nonnatural", "field_hierarchy_nonnatural:psychology"
    if official_field == "decision sciences":
        if any(
            token in specific_text
            for token in ("operations research", "optimization", "statistics", "probability")
        ):
            return "mathematics_statistics", "field_hierarchy:decision-sciences/mathematics"
        return "out_of_scope_nonnatural", "field_hierarchy_nonnatural:decision-sciences"

    fallback = _specific_fallback(specific_text)
    if fallback is not None:
        return fallback
    nonnatural_terms = (
        "political science",
        "sociology",
        "economics",
        "business",
        "management",
        "education",
        "law",
    )
    if any(term in text for term in nonnatural_terms):
        return "out_of_scope_nonnatural", "topic_fallback_nonnatural"
    return "unmapped", "no_domain_rule_matched"


def map_domain12(papers: pd.DataFrame) -> pd.DataFrame:
    """Append stable domain columns without consulting source/venue labels."""

    output = papers.copy()
    assignments = [assign_domain12(row) for row in output.to_dict("records")]
    output["domain12"] = [item[0] for item in assignments]
    output["domain12_label"] = [
        DOMAIN_LABELS.get(
            item[0],
            "Out of scope (non-natural sciences)"
            if item[0] == "out_of_scope_nonnatural"
            else "Unmapped",
        )
        for item in assignments
    ]
    output["domain12_reason"] = [item[1] for item in assignments]
    output["natural_science_eligible"] = output["domain12"].isin(DOMAIN_IDS).astype(int)
    output["domain12_mapped"] = output["domain12"].isin(DOMAIN_IDS).astype(int)
    return output


def build_taxonomy_table(
    papers: pd.DataFrame,
    *,
    min_discussion_size: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Map papers and return a twelve-row coverage table plus audit summary."""

    mapped = map_domain12(papers)
    counts = mapped.loc[mapped["domain12_mapped"] == 1, "domain12"].value_counts()
    rows: List[Dict[str, Any]] = []
    for domain_id in DOMAIN_IDS:
        count = int(counts.get(domain_id, 0))
        rows.append(
            {
                "domain12": domain_id,
                "domain12_label": DOMAIN_LABELS[domain_id],
                "n_papers": count,
                "discussion_eligible": int(count >= int(min_discussion_size)),
            }
        )
    coverage = pd.DataFrame(rows)
    mapped_count = int(mapped["domain12_mapped"].sum())
    out_of_scope = int((mapped["domain12"] == "out_of_scope_nonnatural").sum())
    in_scope_denominator = int(len(mapped) - out_of_scope)
    audit = {
        "n_papers": int(len(mapped)),
        "n_mapped": mapped_count,
        "n_out_of_scope_nonnatural": out_of_scope,
        "n_unmapped_in_natural_scope": int(in_scope_denominator - mapped_count),
        "mapping_coverage": float(mapped_count / max(1, in_scope_denominator)),
        "n_domains_with_200": int(coverage["discussion_eligible"].sum()),
        "mapping_gate_95pct": bool(mapped_count / max(1, in_scope_denominator) >= 0.95),
        "n_field_hierarchy_assignments": int(
            mapped["domain12_reason"].astype(str).str.startswith("field_hierarchy:").sum()
        ),
        "n_topic_fallback_assignments": int(
            mapped["domain12_reason"].astype(str).str.startswith("topic_fallback:").sum()
        ),
    }
    return mapped, coverage, audit

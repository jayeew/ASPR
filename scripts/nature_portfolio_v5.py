from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from gear.corpus import (
    DEFAULT_COMPLETE_END_YEAR,
    normalize_doi,
    normalize_openalex_id,
    short_openalex_id,
    slugify,
    stable_int_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v5_nature_portfolio_full"
DEFAULT_V5_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "nature_portfolio_v5"
DEFAULT_V4_SCREEN_DOMAINS = PROJECT_ROOT / "data" / "knowledge_corpus" / "v4_screen_graph" / "domains.csv"

TARGET_WORK_TYPES = ("article", "review")
GRAPH_WORK_TYPES = ("article", "review", "preprint", "book-chapter", "book")
OPENALEX_WORK_SELECT_V5 = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "type",
        "language",
        "cited_by_count",
        "referenced_works",
        "primary_topic",
        "primary_location",
    ]
)
OPENALEX_SOURCE_SELECT_V5 = ",".join(
    [
        "id",
        "display_name",
        "issn_l",
        "issn",
        "type",
        "host_organization_name",
        "works_count",
        "cited_by_count",
        "homepage_url",
    ]
)

BROAD_CATEGORIES: List[Dict[str, str]] = [
    {
        "broad_category": "biology_life_sciences",
        "label": "Biology / Life Sciences",
        "nature_index_field": "biological sciences",
        "openalex_hints": "Biochemistry, Genetics and Molecular Biology; Agricultural and Biological Sciences; Immunology and Microbiology; Neuroscience",
    },
    {
        "broad_category": "medicine_health_sciences",
        "label": "Medicine / Health Sciences",
        "nature_index_field": "health sciences",
        "openalex_hints": "Medicine; Nursing; Pharmacology, Toxicology and Pharmaceutics; Health Professions",
    },
    {
        "broad_category": "chemistry",
        "label": "Chemistry",
        "nature_index_field": "chemistry",
        "openalex_hints": "Chemistry; Chemical Engineering",
    },
    {
        "broad_category": "physics",
        "label": "Physics",
        "nature_index_field": "physical sciences",
        "openalex_hints": "Physics and Astronomy",
    },
    {
        "broad_category": "astronomy_planetary_space_sciences",
        "label": "Astronomy / Planetary / Space Sciences",
        "nature_index_field": "physical sciences",
        "openalex_hints": "Astronomy and Astrophysics; Earth and Planetary Sciences; Space and Planetary Science",
    },
    {
        "broad_category": "earth_environment_climate_sciences",
        "label": "Earth / Environmental / Climate Sciences",
        "nature_index_field": "earth & environmental sciences",
        "openalex_hints": "Earth and Planetary Sciences; Environmental Science",
    },
    {
        "broad_category": "materials_nanoscience",
        "label": "Materials / Nanoscience",
        "nature_index_field": "applied sciences",
        "openalex_hints": "Materials Science; Nanoscience and Nanotechnology",
    },
    {
        "broad_category": "engineering_energy_applied_sciences",
        "label": "Engineering / Energy / Applied Sciences",
        "nature_index_field": "applied sciences",
        "openalex_hints": "Engineering; Energy; Computer Science Applications",
    },
    {
        "broad_category": "mathematics_statistics",
        "label": "Mathematics / Statistics",
        "nature_index_field": "physical sciences",
        "openalex_hints": "Mathematics; Statistics and Probability; Decision Sciences",
    },
    {
        "broad_category": "computer_science_ai_data_science",
        "label": "Computer Science / AI / Data Science",
        "nature_index_field": "applied sciences",
        "openalex_hints": "Computer Science; Artificial Intelligence; Information Systems",
    },
    {
        "broad_category": "psychology_social_behavioural_policy",
        "label": "Psychology / Social / Behavioural / Policy Sciences",
        "nature_index_field": "social sciences",
        "openalex_hints": "Psychology; Social Sciences; Economics, Econometrics and Finance; Business, Management and Accounting",
    },
]

FIELD_TO_BROAD_HINTS: Sequence[tuple[str, str]] = (
    ("neuroscience", "biology_life_sciences"),
    ("agricultural", "biology_life_sciences"),
    ("biochemistry", "biology_life_sciences"),
    ("genetics", "biology_life_sciences"),
    ("molecular", "biology_life_sciences"),
    ("immunology and microbiology", "biology_life_sciences"),
    ("medicine", "medicine_health_sciences"),
    ("health", "medicine_health_sciences"),
    ("pharmacology", "medicine_health_sciences"),
    ("nursing", "medicine_health_sciences"),
    ("chemistry", "chemistry"),
    ("chemical engineering", "chemistry"),
    ("astronomy", "astronomy_planetary_space_sciences"),
    ("planetary", "astronomy_planetary_space_sciences"),
    ("space", "astronomy_planetary_space_sciences"),
    ("physics", "physics"),
    ("earth", "earth_environment_climate_sciences"),
    ("environment", "earth_environment_climate_sciences"),
    ("climate", "earth_environment_climate_sciences"),
    ("materials", "materials_nanoscience"),
    ("nanoscience", "materials_nanoscience"),
    ("engineering", "engineering_energy_applied_sciences"),
    ("energy", "engineering_energy_applied_sciences"),
    ("mathematics", "mathematics_statistics"),
    ("statistics", "mathematics_statistics"),
    ("decision sciences", "mathematics_statistics"),
    ("computer science", "computer_science_ai_data_science"),
    ("artificial intelligence", "computer_science_ai_data_science"),
    ("psychology", "psychology_social_behavioural_policy"),
    ("social", "psychology_social_behavioural_policy"),
    ("economics", "psychology_social_behavioural_policy"),
    ("business", "psychology_social_behavioural_policy"),
)

EXACT_FIELD_TO_BROAD: Mapping[str, str] = {
    "earth and planetary sciences": "earth_environment_climate_sciences",
}

SUPPLEMENTAL_DOMAIN_SEEDS: Sequence[tuple[str, str, str, str]] = (
    ("biology_life_sciences", "plant_genomics", "Plant genomics", '"plant genomics" OR "plant genome"'),
    ("biology_life_sciences", "evolutionary_developmental_biology", "Evolutionary developmental biology", '"evo-devo" OR "evolutionary developmental biology"'),
    ("biology_life_sciences", "systems_biology", "Systems biology", '"systems biology" OR "gene regulatory network"'),
    ("biology_life_sciences", "ecology_biodiversity", "Ecology and biodiversity", '"biodiversity" OR "ecosystem function" OR "ecology"'),
    ("medicine_health_sciences", "clinical_trials_evidence", "Clinical trials and evidence", '"clinical trial" OR "randomized controlled trial"'),
    ("medicine_health_sciences", "infectious_disease_epidemiology", "Infectious disease epidemiology", '"infectious disease epidemiology" OR "disease outbreak"'),
    ("medicine_health_sciences", "cardiometabolic_disease", "Cardiometabolic disease", '"cardiometabolic disease" OR "diabetes" OR "cardiovascular risk"'),
    ("medicine_health_sciences", "neurodegenerative_disease", "Neurodegenerative disease", '"neurodegenerative disease" OR "Alzheimer" OR "Parkinson"'),
    ("chemistry", "organic_synthesis", "Organic synthesis", '"organic synthesis" OR "total synthesis"'),
    ("chemistry", "chemical_catalysis", "Chemical catalysis", '"chemical catalysis" OR "asymmetric catalysis"'),
    ("chemistry", "supramolecular_chemistry", "Supramolecular chemistry", '"supramolecular chemistry" OR "self-assembly"'),
    ("chemistry", "analytical_chemistry_methods", "Analytical chemistry methods", '"analytical chemistry" OR "mass spectrometry"'),
    ("physics", "quantum_information", "Quantum information", '"quantum information" OR "quantum computing"'),
    ("physics", "condensed_matter_physics", "Condensed matter physics", '"condensed matter" OR "many-body physics"'),
    ("physics", "particle_physics", "Particle physics", '"particle physics" OR "standard model"'),
    ("physics", "soft_matter_physics", "Soft matter physics", '"soft matter" OR "active matter"'),
    ("astronomy_planetary_space_sciences", "cosmology_dark_matter", "Cosmology and dark matter", '"cosmology" OR "dark matter"'),
    ("astronomy_planetary_space_sciences", "planetary_science", "Planetary science", '"planetary science" OR "solar system"'),
    ("astronomy_planetary_space_sciences", "stellar_evolution", "Stellar evolution", '"stellar evolution" OR "star formation"'),
    ("astronomy_planetary_space_sciences", "space_weather", "Space weather", '"space weather" OR "solar wind"'),
    ("earth_environment_climate_sciences", "climate_change_impacts", "Climate change impacts", '"climate change impacts" OR "global warming"'),
    ("earth_environment_climate_sciences", "oceanography", "Oceanography", '"oceanography" OR "ocean circulation"'),
    ("earth_environment_climate_sciences", "geochemistry_geophysics", "Geochemistry and geophysics", '"geochemistry" OR "geophysics"'),
    ("earth_environment_climate_sciences", "conservation_science", "Conservation science", '"conservation science" OR "species conservation"'),
    ("materials_nanoscience", "biomaterials", "Biomaterials", '"biomaterials" OR "tissue engineering scaffold"'),
    ("materials_nanoscience", "polymer_science", "Polymer science", '"polymer science" OR "macromolecular materials"'),
    ("materials_nanoscience", "semiconductor_materials", "Semiconductor materials", '"semiconductor materials" OR "optoelectronics"'),
    ("materials_nanoscience", "nanomedicine", "Nanomedicine", '"nanomedicine" OR "nanoparticle drug delivery"'),
    ("engineering_energy_applied_sciences", "renewable_energy_systems", "Renewable energy systems", '"renewable energy" OR "energy storage"'),
    ("engineering_energy_applied_sciences", "robotics_autonomous_systems", "Robotics and autonomous systems", '"robotics" OR "autonomous systems"'),
    ("engineering_energy_applied_sciences", "biomedical_engineering", "Biomedical engineering", '"biomedical engineering" OR "medical device"'),
    ("engineering_energy_applied_sciences", "environmental_engineering", "Environmental engineering", '"environmental engineering" OR "water treatment"'),
    ("mathematics_statistics", "probability_theory", "Probability theory", '"probability theory" OR "stochastic process"'),
    ("mathematics_statistics", "statistical_learning_theory", "Statistical learning theory", '"statistical learning theory" OR "high-dimensional statistics"'),
    ("mathematics_statistics", "optimization_theory", "Optimization theory", '"optimization theory" OR "convex optimization"'),
    ("mathematics_statistics", "mathematical_biology", "Mathematical biology", '"mathematical biology" OR "biomathematics"'),
    ("computer_science_ai_data_science", "machine_learning_foundations", "Machine learning foundations", '"machine learning" OR "deep learning"'),
    ("computer_science_ai_data_science", "computer_vision", "Computer vision", '"computer vision" OR "image recognition"'),
    ("computer_science_ai_data_science", "natural_language_processing", "Natural language processing", '"natural language processing" OR "language model"'),
    ("computer_science_ai_data_science", "scientific_machine_learning", "Scientific machine learning", '"scientific machine learning" OR "physics-informed neural network"'),
    ("psychology_social_behavioural_policy", "cognitive_psychology", "Cognitive psychology", '"cognitive psychology" OR "decision making"'),
    ("psychology_social_behavioural_policy", "social_psychology", "Social psychology", '"social psychology" OR "social behavior"'),
    ("psychology_social_behavioural_policy", "public_policy_science", "Public policy and science", '"public policy" OR "science policy"'),
    ("psychology_social_behavioural_policy", "economics_inequality", "Economics and inequality", '"economic inequality" OR "income inequality"'),
)

NATURE_SOURCE_SEEDS: Sequence[tuple[str, str, str, str]] = (
    ("nature_flagship", "Nature", "multidisciplinary", "https://www.nature.com/nature"),
    ("nature_research", "Nature Biotechnology", "biology_life_sciences", "https://www.nature.com/nbt"),
    ("nature_research", "Nature Cell Biology", "biology_life_sciences", "https://www.nature.com/ncb"),
    ("nature_research", "Nature Genetics", "biology_life_sciences", "https://www.nature.com/ng"),
    ("nature_research", "Nature Immunology", "biology_life_sciences", "https://www.nature.com/ni"),
    ("nature_research", "Nature Microbiology", "biology_life_sciences", "https://www.nature.com/nmicrobiol"),
    ("nature_research", "Nature Neuroscience", "biology_life_sciences", "https://www.nature.com/neuro"),
    ("nature_research", "Nature Medicine", "medicine_health_sciences", "https://www.nature.com/nm"),
    ("nature_research", "Nature Biomedical Engineering", "medicine_health_sciences", "https://www.nature.com/natbiomedeng"),
    ("nature_research", "Nature Cancer", "medicine_health_sciences", "https://www.nature.com/natcancer"),
    ("nature_research", "Nature Cardiovascular Research", "medicine_health_sciences", "https://www.nature.com/natcardiovascres"),
    ("nature_research", "Nature Chemistry", "chemistry", "https://www.nature.com/nchem"),
    ("nature_research", "Nature Chemical Biology", "chemistry", "https://www.nature.com/nchembio"),
    ("nature_research", "Nature Catalysis", "chemistry", "https://www.nature.com/natcatal"),
    ("nature_research", "Nature Physics", "physics", "https://www.nature.com/nphys"),
    ("nature_research", "Nature Astronomy", "astronomy_planetary_space_sciences", "https://www.nature.com/natastron"),
    ("nature_research", "Nature Geoscience", "earth_environment_climate_sciences", "https://www.nature.com/ngeo"),
    ("nature_research", "Nature Climate Change", "earth_environment_climate_sciences", "https://www.nature.com/nclimate"),
    ("nature_research", "Nature Ecology & Evolution", "earth_environment_climate_sciences", "https://www.nature.com/natecolevol"),
    ("nature_research", "Nature Materials", "materials_nanoscience", "https://www.nature.com/nmat"),
    ("nature_research", "Nature Nanotechnology", "materials_nanoscience", "https://www.nature.com/nnano"),
    ("nature_research", "Nature Energy", "engineering_energy_applied_sciences", "https://www.nature.com/nenergy"),
    ("nature_research", "Nature Electronics", "engineering_energy_applied_sciences", "https://www.nature.com/natelectron"),
    ("nature_research", "Nature Machine Intelligence", "computer_science_ai_data_science", "https://www.nature.com/natmachintell"),
    ("nature_research", "Nature Computational Science", "computer_science_ai_data_science", "https://www.nature.com/natcomputsci"),
    ("nature_research", "Nature Human Behaviour", "psychology_social_behavioural_policy", "https://www.nature.com/nathumbehav"),
    ("nature_multidisciplinary", "Nature Communications", "multidisciplinary", "https://www.nature.com/ncomms"),
    ("nature_multidisciplinary", "Scientific Reports", "multidisciplinary", "https://www.nature.com/srep"),
    ("communications", "Communications Biology", "biology_life_sciences", "https://www.nature.com/commsbio"),
    ("communications", "Communications Medicine", "medicine_health_sciences", "https://www.nature.com/commsmed"),
    ("communications", "Communications Chemistry", "chemistry", "https://www.nature.com/commschem"),
    ("communications", "Communications Physics", "physics", "https://www.nature.com/commsphys"),
    ("communications", "Communications Earth & Environment", "earth_environment_climate_sciences", "https://www.nature.com/commsenv"),
    ("communications", "Communications Materials", "materials_nanoscience", "https://www.nature.com/commsmat"),
    ("npj", "npj Quantum Information", "physics", "https://www.nature.com/npjqi"),
    ("npj", "npj Computational Materials", "materials_nanoscience", "https://www.nature.com/npjcompumats"),
    ("npj", "npj Climate and Atmospheric Science", "earth_environment_climate_sciences", "https://www.nature.com/npjclimatsci"),
    ("npj", "npj Digital Medicine", "medicine_health_sciences", "https://www.nature.com/npjdigitalmed"),
    ("npj", "npj Genomic Medicine", "medicine_health_sciences", "https://www.nature.com/npjgenmed"),
    ("npj", "npj Systems Biology and Applications", "biology_life_sciences", "https://www.nature.com/npjsba"),
    ("npj", "npj Clean Water", "engineering_energy_applied_sciences", "https://www.nature.com/npjcleanwater"),
    ("npj", "npj Science of Learning", "psychology_social_behavioural_policy", "https://www.nature.com/npjscilearn"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def nonempty(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def safe_numeric(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return float(default)
    return float(number)


def entropy_from_counts(counts: Sequence[float]) -> float:
    arr = np.asarray(counts, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p)).sum())


def simpson_from_counts(counts: Sequence[float]) -> float:
    arr = np.asarray(counts, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(1.0 - np.square(p).sum())


def classify_broad_category(*values: object, default: str = "multidisciplinary") -> str:
    text = " ".join(str(value or "") for value in values).lower()
    for field, category in EXACT_FIELD_TO_BROAD.items():
        if field in text:
            return category
    for hint, category in FIELD_TO_BROAD_HINTS:
        if hint in text:
            return category
    return default


def journal_family_from_name(name: object, fallback: str = "nature_portfolio") -> str:
    text = str(name or "").strip().lower()
    if text == "nature":
        return "nature_flagship"
    if text.startswith("nature reviews"):
        return "nature_reviews"
    if text.startswith("nature "):
        return "nature_research"
    if "nature communications" in text or text == "scientific reports":
        return "nature_multidisciplinary"
    if text.startswith("communications "):
        return "communications"
    if text.startswith("npj "):
        return "npj"
    return fallback


def source_short_id(value: object) -> str:
    sid = short_openalex_id(value)
    return sid if sid.startswith("S") else sid


def openalex_source_filter(source_id: object) -> str:
    sid = source_short_id(source_id)
    return f"primary_location.source.id:{sid}"


def work_source_metadata(work: Mapping[str, Any]) -> Dict[str, Any]:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return {
        "source_id": normalize_openalex_id(source.get("id")),
        "source_display_name": nonempty(source.get("display_name")),
        "source_issn_l": nonempty(source.get("issn_l")),
        "source_host_organization_name": nonempty(source.get("host_organization_name")),
    }


def topic_metadata(work: Mapping[str, Any]) -> Dict[str, Any]:
    topic = work.get("primary_topic") or {}
    field = topic.get("field") or {}
    subfield = topic.get("subfield") or {}
    return {
        "primary_topic_id": normalize_openalex_id(topic.get("id")),
        "primary_topic": nonempty(topic.get("display_name")),
        "primary_field": nonempty(field.get("display_name")),
        "primary_subfield": nonempty(subfield.get("display_name")),
    }


def fine_domain_from_work(work: Mapping[str, Any], fallback: object = "") -> str:
    meta = topic_metadata(work)
    label = meta.get("primary_topic") or meta.get("primary_subfield") or meta.get("primary_field") or fallback
    return slugify(label)


def references_from_work(work: Mapping[str, Any]) -> List[str]:
    return [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]


def target_work_row(
    work: Mapping[str, Any],
    source_row: Optional[Mapping[str, Any]] = None,
    fetched_at: Optional[str] = None,
) -> Dict[str, Any]:
    source_row = dict(source_row or {})
    topic = topic_metadata(work)
    src = work_source_metadata(work)
    refs = references_from_work(work)
    source_name = source_row.get("source_display_name") or src.get("source_display_name")
    broad = source_row.get("broad_category") or classify_broad_category(
        source_row.get("broad_category"),
        topic.get("primary_field"),
        topic.get("primary_subfield"),
        topic.get("primary_topic"),
        source_name,
    )
    domain = fine_domain_from_work(work, fallback=source_row.get("default_domain") or source_name)
    topic_label = topic.get("primary_topic") or topic.get("primary_subfield") or topic.get("primary_field") or domain
    year = int(safe_numeric(work.get("publication_year"), default=0))
    work_id = normalize_openalex_id(work.get("id"))
    source_id = source_row.get("source_id") or src.get("source_id")
    return {
        "id": work_id,
        "short_id": short_openalex_id(work_id),
        "doi": normalize_doi(work.get("doi")),
        "title": nonempty(work.get("display_name")) or work_id,
        "year": year,
        "domain": domain,
        "broad_category": broad,
        "journal_family": source_row.get("journal_family") or journal_family_from_name(source_name),
        "source_id": source_id,
        "source_display_name": source_name,
        "source_issn_l": source_row.get("issn_l") or src.get("source_issn_l"),
        "primary_field": topic.get("primary_subfield") or topic.get("primary_field") or broad,
        "openalex_primary_field": topic.get("primary_field"),
        "openalex_primary_subfield": topic.get("primary_subfield"),
        "display_community": stable_int_id(f"{domain}:{topic.get('primary_topic_id') or topic_label}"),
        "display_topic_id": topic.get("primary_topic_id"),
        "display_topic_label": topic_label,
        "primary_topic": topic_label,
        "legacy_is_landmark": 0,
        "is_landmark": 0,
        "anchor_label": "",
        "reliable_anchor_source": "",
        "anchor_policy": "venue_driven_v5",
        "document_type": nonempty(work.get("type")),
        "cited_by_count": int(safe_numeric(work.get("cited_by_count"), default=0)),
        "reference_count": int(len(refs)),
        "source_provider": "openalex",
        "source_dataset": "nature_portfolio_v5_target",
        "fetched_at": fetched_at or utc_now(),
        "referenced_works": json.dumps(refs, ensure_ascii=False),
        "partial_2026": int(year >= 2026),
        "is_target_work": 1,
    }


def reference_work_row(work: Mapping[str, Any], fetched_at: Optional[str] = None) -> Dict[str, Any]:
    topic = topic_metadata(work)
    src = work_source_metadata(work)
    topic_label = topic.get("primary_topic") or topic.get("primary_subfield") or topic.get("primary_field") or "unknown_topic"
    work_id = normalize_openalex_id(work.get("id"))
    year = int(safe_numeric(work.get("publication_year"), default=0))
    return {
        "id": work_id,
        "short_id": short_openalex_id(work_id),
        "doi": normalize_doi(work.get("doi")),
        "title": nonempty(work.get("display_name")) or work_id,
        "year": year,
        "domain": slugify(topic_label),
        "broad_category": classify_broad_category(topic.get("primary_field"), topic.get("primary_subfield"), topic_label),
        "source_id": src.get("source_id"),
        "source_display_name": src.get("source_display_name"),
        "primary_field": topic.get("primary_subfield") or topic.get("primary_field"),
        "openalex_primary_field": topic.get("primary_field"),
        "openalex_primary_subfield": topic.get("primary_subfield"),
        "display_community": stable_int_id(f"reference:{topic.get('primary_topic_id') or topic_label}"),
        "display_topic_id": topic.get("primary_topic_id"),
        "display_topic_label": topic_label,
        "document_type": nonempty(work.get("type")),
        "cited_by_count": int(safe_numeric(work.get("cited_by_count"), default=0)),
        "reference_count": len(references_from_work(work)),
        "source_provider": "openalex",
        "source_dataset": "nature_portfolio_v5_reference_closure",
        "fetched_at": fetched_at or utc_now(),
        "is_target_work": 0,
    }


def parse_referenced_works(value: object) -> List[str]:
    if isinstance(value, list):
        return [normalize_openalex_id(item) for item in value if normalize_openalex_id(item)]
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = re.split(r"[;\s,]+", text)
    if not isinstance(payload, list):
        return []
    return [normalize_openalex_id(item) for item in payload if normalize_openalex_id(item)]


def target_reference_edges(target_works: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    if target_works.empty:
        return pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])
    for row in target_works.to_dict("records"):
        source = normalize_openalex_id(row.get("id"))
        if not source:
            continue
        for ref in parse_referenced_works(row.get("referenced_works")):
            rows.append(
                {
                    "source": source,
                    "target": ref,
                    "relation": "reference",
                    "source_dataset": "nature_portfolio_v5_reference",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["source", "target"]).reset_index(drop=True) if rows else pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])


def build_time_block_folds(years: pd.Series, n_folds: int = 5) -> pd.Series:
    numeric = pd.to_numeric(years, errors="coerce")
    out = pd.Series(-1, index=years.index, dtype=int)
    valid = numeric.notna()
    unique_years = sorted(numeric[valid].astype(int).unique().tolist())
    if len(unique_years) < 2:
        out.loc[valid] = 1
        return out
    bins = np.array_split(unique_years, max(2, int(n_folds)))
    for fold_idx, year_bin in enumerate(bins, start=1):
        if len(year_bin) == 0:
            continue
        out.loc[numeric.isin([int(v) for v in year_bin])] = int(fold_idx)
    return out


def robust_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(0.5, index=values.index)
    return numeric.rank(method="average", pct=True).fillna(0.5)


def try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def write_partitioned_table(
    df: pd.DataFrame,
    root: Path,
    table_name: str,
    partition_cols: Sequence[str] = ("broad_category", "domain", "year"),
) -> Dict[str, Any]:
    table_root = root / table_name
    table_root.mkdir(parents=True, exist_ok=True)
    if df.empty:
        (table_root / "_EMPTY").write_text("", encoding="utf-8")
        return {"table": table_name, "n_rows": 0, "format": "empty", "n_partitions": 0}
    present = [col for col in partition_cols if col in df.columns]
    if not present:
        ok = try_write_parquet(df, table_root / f"{table_name}.parquet")
        if not ok:
            df.to_csv(table_root / f"{table_name}.csv", index=False)
        return {"table": table_name, "n_rows": int(len(df)), "format": "parquet" if ok else "csv", "n_partitions": 1}
    fmt = "parquet"
    n_parts = 0
    for keys, sub in df.groupby(present, dropna=False, sort=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        part_dir = table_root
        for col, key in zip(present, keys_tuple):
            safe_key = slugify(key)
            part_dir = part_dir / f"{col}={safe_key}"
        ok = try_write_parquet(sub, part_dir / "part.parquet")
        if not ok:
            fmt = "csv"
            part_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(part_dir / "part.csv", index=False)
        n_parts += 1
    return {"table": table_name, "n_rows": int(len(df)), "format": fmt, "n_partitions": int(n_parts)}


def coverage_quality_summary(
    works: pd.DataFrame,
    future: Optional[pd.DataFrame] = None,
    *,
    tau: int = 8,
    min_broad_categories: int = 10,
    min_fine_domains: int = 80,
    min_broad_eligible: int = 2000,
    min_domain_eligible: int = 200,
) -> Dict[str, Any]:
    if works.empty:
        return {"overall_pass": False, "checks": {}, "errors": ["works table is empty"]}
    frame = works.copy()
    frame["year"] = pd.to_numeric(frame.get("year", 0), errors="coerce")
    frame["reference_count"] = pd.to_numeric(frame.get("reference_count", 0), errors="coerce").fillna(0)
    frame["doi_norm"] = frame.get("doi", "").map(normalize_doi)
    frame["id_norm"] = frame.get("id", "").map(normalize_openalex_id)
    eligible = frame[frame["year"].notna() & (frame["year"].astype(int) <= DEFAULT_COMPLETE_END_YEAR - int(tau))].copy()
    if future is not None and not future.empty and "paper_id" in future.columns:
        fids = set(future["paper_id"].astype(str))
        eligible = eligible[eligible["id"].astype(str).isin(fids)].copy()
    broad_counts = eligible.get("broad_category", pd.Series("", index=eligible.index)).astype(str).value_counts()
    domain_counts = eligible.get("domain", pd.Series("", index=eligible.index)).astype(str).value_counts()
    doi_nonempty = frame[frame["doi_norm"].astype(str) != ""]
    duplicate_doi_rate = float(doi_nonempty.duplicated("doi_norm").mean()) if len(doi_nonempty) else 0.0
    duplicate_id_rate = float(frame.duplicated("id_norm").mean()) if len(frame) else 0.0
    topic_label = frame.get("display_topic_label", pd.Series("", index=frame.index)).fillna("").astype(str)
    topic_coverage = float(topic_label.str.strip().ne("").mean()) if len(frame) else 0.0
    future_coverage = float(len(eligible) / max(1, int((frame["year"] <= DEFAULT_COMPLETE_END_YEAR - int(tau)).sum())))
    checks = {
        "broad_categories_ge_min": int(broad_counts.size >= int(min_broad_categories)),
        "fine_domains_ge_min": int(domain_counts.size >= int(min_fine_domains)),
        "broad_eligible_floor": int((broad_counts >= int(min_broad_eligible)).all()) if broad_counts.size else 0,
        "domain_eligible_floor": int((domain_counts >= int(min_domain_eligible)).all()) if domain_counts.size else 0,
        "doi_duplicate_rate_lt_1pct": int(duplicate_doi_rate < 0.01),
        "openalex_id_duplicate_rate_lt_0_1pct": int(duplicate_id_rate < 0.001),
        "primary_topic_coverage_ge_95pct": int(topic_coverage >= 0.95),
        "reference_count_median_ge_15": int(float(frame["reference_count"].median()) >= 15.0),
        "future_citer_coverage_ge_80pct": int(future_coverage >= 0.80),
        "no_partial_2026": int(not (frame["year"].fillna(0).astype(int) >= 2026).any()),
    }
    return {
        "artifact_kind": "nature_portfolio_v5_data_quality_report",
        "created_at": utc_now(),
        "overall_pass": bool(checks) and all(bool(v) for v in checks.values()),
        "checks": checks,
        "tau": int(tau),
        "n_target_works": int(len(frame)),
        "n_tau_eligible_works": int(len(eligible)),
        "n_broad_categories": int(broad_counts.size),
        "n_fine_domains": int(domain_counts.size),
        "min_eligible_per_broad_category": int(broad_counts.min()) if broad_counts.size else 0,
        "min_eligible_per_fine_domain": int(domain_counts.min()) if domain_counts.size else 0,
        "duplicate_doi_rate": duplicate_doi_rate,
        "duplicate_openalex_id_rate": duplicate_id_rate,
        "primary_topic_coverage": topic_coverage,
        "reference_count_median": float(frame["reference_count"].median()) if len(frame) else 0.0,
        "future_citer_coverage": future_coverage,
        "broad_category_counts": {str(k): int(v) for k, v in broad_counts.to_dict().items()},
        "fine_domain_counts_top50": {str(k): int(v) for k, v in domain_counts.head(50).to_dict().items()},
    }

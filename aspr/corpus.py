from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from aspr.env import getenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_STRICT_CORPUS_DIR = DATA_DIR / "knowledge_corpus" / "v1_strict"
DEFAULT_BUILD_CORPUS_DIR = DATA_DIR / "knowledge_corpus" / "v1_large"
DEFAULT_CORPUS_DIR = DEFAULT_STRICT_CORPUS_DIR
DEFAULT_FIG1_ROOT = OUTPUTS_DIR / "kg_perturbation_fig1"
DEFAULT_FIG3_AUTO_ROOT = OUTPUTS_DIR / "kg_perturbation_fig3_auto"
DEFAULT_COMPLETE_END_YEAR = 2025

CORE_DOMAINS = [
    {
        "slug": "crispr",
        "display_name": "CRISPR-Cas genome editing",
        "query": '"CRISPR" OR "Cas9" OR "genome editing"',
        "field_name": "Biochemistry, Genetics and Molecular Biology",
        "subfield_name": "Genetics",
        "seed_source": "fig1_core",
    },
    {
        "slug": "graphene_2d_materials",
        "display_name": "Graphene and 2D materials",
        "query": '"graphene" OR "two-dimensional materials" OR "van der Waals heterostructure"',
        "field_name": "Materials Science",
        "subfield_name": "Condensed Matter Physics",
        "seed_source": "fig1_core",
    },
    {
        "slug": "ipsc_reprogramming",
        "display_name": "iPSC and cellular reprogramming",
        "query": '"induced pluripotent stem cell" OR "cellular reprogramming" OR "Yamanaka factors"',
        "field_name": "Biochemistry, Genetics and Molecular Biology",
        "subfield_name": "Cell Biology",
        "seed_source": "fig1_core",
    },
    {
        "slug": "transformer_foundation_models",
        "display_name": "Transformer and foundation models",
        "query": '"transformer" OR "BERT" OR "large language model" OR "foundation model"',
        "field_name": "Computer Science",
        "subfield_name": "Artificial Intelligence",
        "seed_source": "fig1_core",
    },
]

MANUAL_DOMAIN_SEEDS = [
    ("mrna_vaccines_lnp", "mRNA vaccines and lipid nanoparticles", '"mRNA vaccine" OR "lipid nanoparticle" OR "modified mRNA"', "Medicine", "Immunology"),
    ("alphafold_protein_folding", "AlphaFold and protein structure prediction", '"AlphaFold" OR "protein structure prediction" OR "protein folding"', "Biochemistry, Genetics and Molecular Biology", "Structural Biology"),
    ("single_cell_rna_seq", "Single-cell RNA sequencing", '"single-cell RNA-seq" OR "single cell transcriptomics" OR "Drop-seq"', "Biochemistry, Genetics and Molecular Biology", "Molecular Biology"),
    ("spatial_transcriptomics", "Spatial transcriptomics", '"spatial transcriptomics" OR "spatially resolved transcriptomics"', "Biochemistry, Genetics and Molecular Biology", "Molecular Biology"),
    ("car_t_cell_therapy", "CAR-T cell therapy", '"CAR-T" OR "chimeric antigen receptor T cell"', "Medicine", "Oncology"),
    ("organoids", "Organoids and stem-cell models", '"organoid" OR "cerebral organoid" OR "intestinal organoid"', "Medicine", "Cell Biology"),
    ("rna_interference", "RNA interference", '"RNA interference" OR "siRNA" OR "gene silencing"', "Biochemistry, Genetics and Molecular Biology", "Genetics"),
    ("perovskite_solar_cells", "Perovskite solar cells", '"perovskite solar cell" OR "organometal halide perovskite"', "Materials Science", "Renewable Energy"),
    ("quantum_computing", "Quantum computing", '"quantum computing" OR "quantum algorithm" OR "quantum supremacy"', "Computer Science", "Computer Science Applications"),
    ("gravitational_waves", "Gravitational waves", '"gravitational waves" OR "LIGO" OR "binary black hole"', "Physics and Astronomy", "Astronomy and Astrophysics"),
    ("exoplanets", "Exoplanets", '"exoplanet" OR "extrasolar planet" OR "transiting planet"', "Physics and Astronomy", "Astronomy and Astrophysics"),
    ("super_resolution_microscopy", "Super-resolution microscopy", '"super-resolution microscopy" OR "STED microscopy" OR "PALM microscopy" OR "STORM microscopy"', "Biochemistry, Genetics and Molecular Biology", "Biophysics"),
    ("microbiome_metagenomics", "Microbiome and metagenomics", '"microbiome" OR "metagenomics" OR "human microbiome"', "Medicine", "Microbiology"),
    ("nanopore_sequencing", "Nanopore sequencing", '"nanopore sequencing" OR "MinION" OR "single-molecule sequencing"', "Biochemistry, Genetics and Molecular Biology", "Genomics"),
    ("diffusion_generative_models", "Diffusion and generative models", '"diffusion model" OR "score-based generative model" OR "latent diffusion"', "Computer Science", "Artificial Intelligence"),
    ("lithium_ion_solid_state_batteries", "Lithium-ion and solid-state batteries", '"lithium-ion battery" OR "solid-state battery" OR "battery cathode"', "Energy", "Electrochemistry"),
    ("topological_insulators", "Topological insulators", '"topological insulator" OR "quantum spin Hall" OR "topological materials"', "Physics and Astronomy", "Condensed Matter Physics"),
    ("genome_wide_association_studies", "Genome-wide association studies", '"genome-wide association study" OR "GWAS"', "Medicine", "Genetics"),
    ("climate_attribution_models", "Climate attribution and earth-system modeling", '"climate attribution" OR "earth system model" OR "anthropogenic climate change"', "Earth and Planetary Sciences", "Atmospheric Science"),
    ("synthetic_biology", "Synthetic biology", '"synthetic biology" OR "genetic circuit" OR "minimal genome"', "Biochemistry, Genetics and Molecular Biology", "Biotechnology"),
    ("immune_checkpoint_therapy", "Immune checkpoint therapy", '"immune checkpoint inhibitor" OR "PD-1" OR "CTLA-4" OR "cancer immunotherapy"', "Medicine", "Oncology"),
    ("aav_gene_therapy", "AAV and gene therapy", '"AAV gene therapy" OR "adeno-associated virus" OR "gene therapy"', "Medicine", "Genetics"),
    ("hydrogen_electrocatalysis", "Hydrogen electrocatalysis", '"hydrogen evolution reaction" OR "oxygen evolution reaction" OR "water splitting electrocatalyst"', "Chemical Engineering", "Catalysis"),
    ("bayesian_causal_inference", "Bayesian causal inference", '"causal inference" OR "Bayesian causal" OR "potential outcomes"', "Mathematics", "Statistics and Probability"),
    ("graph_neural_networks", "Graph neural networks", '"graph neural network" OR "graph convolutional network" OR "message passing neural network"', "Computer Science", "Artificial Intelligence"),
    ("large_scale_recommendation", "Large-scale recommender systems", '"recommender system" OR "collaborative filtering" OR "neural recommendation"', "Computer Science", "Information Systems"),
    ("robot_learning", "Robot learning and reinforcement learning", '"robot learning" OR "reinforcement learning" OR "deep reinforcement learning"', "Computer Science", "Artificial Intelligence"),
    ("cancer_genomics_precision_oncology", "Cancer genomics and precision oncology", '"cancer genomics" OR "precision oncology" OR "tumor sequencing"', "Medicine", "Oncology"),
]

MANUAL_LANDMARKS = [
    ("mrna_vaccines_lnp", "Kariko 2005", "10.1016/j.immuni.2005.06.008", "Suppression of RNA recognition by Toll-like receptors", 2005),
    ("mrna_vaccines_lnp", "Polack 2020", "10.1056/NEJMoa2034577", "Safety and Efficacy of the BNT162b2 mRNA Covid-19 Vaccine", 2020),
    ("alphafold_protein_folding", "Senior 2020", "10.1038/s41586-019-1923-7", "Improved protein structure prediction using potentials from deep learning", 2020),
    ("alphafold_protein_folding", "Jumper 2021", "10.1038/s41586-021-03819-2", "Highly accurate protein structure prediction with AlphaFold", 2021),
    ("single_cell_rna_seq", "Tang 2009", "10.1038/nmeth.1315", "mRNA-Seq whole-transcriptome analysis of a single cell", 2009),
    ("single_cell_rna_seq", "Macosko 2015", "10.1016/j.cell.2015.05.002", "Highly Parallel Genome-wide Expression Profiling of Individual Cells Using Nanoliter Droplets", 2015),
    ("spatial_transcriptomics", "Stahl 2016", "10.1126/science.aaf2403", "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics", 2016),
    ("car_t_cell_therapy", "Porter 2011", "10.1056/NEJMoa1103849", "Chimeric Antigen Receptor-Modified T Cells in Chronic Lymphoid Leukemia", 2011),
    ("car_t_cell_therapy", "Maude 2014", "10.1056/NEJMoa1407222", "Chimeric Antigen Receptor T Cells for Sustained Remissions in Leukemia", 2014),
    ("organoids", "Sato 2009", "10.1038/nature07935", "Single Lgr5 stem cells build crypt-villus structures in vitro without a mesenchymal niche", 2009),
    ("organoids", "Lancaster 2013", "10.1038/nature12517", "Cerebral organoids model human brain development and microcephaly", 2013),
    ("rna_interference", "Fire 1998", "10.1038/35888", "Potent and specific genetic interference by double-stranded RNA in Caenorhabditis elegans", 1998),
    ("perovskite_solar_cells", "Kojima 2009", "10.1021/ja809598r", "Organometal Halide Perovskites as Visible-Light Sensitizers for Photovoltaic Cells", 2009),
    ("perovskite_solar_cells", "Lee 2012", "10.1126/science.1228604", "Efficient Hybrid Solar Cells Based on Meso-Superstructured Organometal Halide Perovskites", 2012),
    ("gravitational_waves", "Abbott 2016", "10.1103/PhysRevLett.116.061102", "Observation of Gravitational Waves from a Binary Black Hole Merger", 2016),
    ("exoplanets", "Mayor 1995", "10.1038/378355a0", "A Jupiter-mass companion to a solar-type star", 1995),
    ("super_resolution_microscopy", "Betzig 2006", "10.1126/science.1127344", "Imaging Intracellular Fluorescent Proteins at Nanometer Resolution", 2006),
    ("microbiome_metagenomics", "HMP 2012", "10.1038/nature11234", "Structure, function and diversity of the healthy human microbiome", 2012),
    ("nanopore_sequencing", "Kasianowicz 1996", "10.1073/pnas.93.24.13770", "Characterization of individual polynucleotide molecules using a membrane channel", 1996),
    ("diffusion_generative_models", "Ho 2020", "", "Denoising Diffusion Probabilistic Models", 2020),
    ("diffusion_generative_models", "Rombach 2022", "10.1109/CVPR52688.2022.01042", "High-Resolution Image Synthesis With Latent Diffusion Models", 2022),
    ("topological_insulators", "Kane/Mele 2005", "10.1103/PhysRevLett.95.146802", "Quantum Spin Hall Effect in Graphene", 2005),
    ("genome_wide_association_studies", "WTCCC 2007", "10.1038/nature05911", "Genome-wide association study of 14,000 cases of seven common diseases and 3,000 shared controls", 2007),
    ("synthetic_biology", "Elowitz 2000", "10.1038/35002125", "A synthetic oscillatory network of transcriptional regulators", 2000),
    ("synthetic_biology", "Gardner 2000", "10.1038/35002131", "Construction of a genetic toggle switch in Escherichia coli", 2000),
    ("immune_checkpoint_therapy", "Hodi 2010", "10.1056/NEJMoa1003466", "Improved Survival with Ipilimumab in Patients with Metastatic Melanoma", 2010),
]


@dataclass
class SourceTables:
    works: pd.DataFrame
    citations: pd.DataFrame
    topics: pd.DataFrame
    topic_edges: pd.DataFrame
    domains: pd.DataFrame
    landmarks: pd.DataFrame
    legacy_raw: Dict[str, List[Dict[str, Any]]]


def progress_log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[corpus] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: object) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").lower()).strip("_")
    return value or "domain"


def short_openalex_id(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text.rstrip("/").split("/")[-1] if text else ""


def normalize_openalex_id(value: object) -> str:
    sid = short_openalex_id(value)
    if not sid:
        return ""
    return f"https://openalex.org/{sid}"


def normalize_doi(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.I)
    return text.lower()


def normalize_title(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "null", "<na>"}:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def nonempty_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def stable_int_id(value: object, modulo: int = 100_000_000) -> int:
    text = str(value or "")
    digest = 0
    for char in text:
        digest = (digest * 131 + ord(char)) % modulo
    return int(digest)


def split_api_keys(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(split_api_keys(item))
        return list(dict.fromkeys(out))
    text = str(value or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()))


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def domain_seed_table(fig3_auto_root: Path, max_domains: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.extend(CORE_DOMAINS)
    auto = read_csv(fig3_auto_root / "domain_seeds.csv")
    if not auto.empty:
        for row in auto.to_dict("records"):
            rows.append(
                {
                    "slug": slugify(row.get("slug") or row.get("display_name")),
                    "display_name": row.get("display_name") or row.get("slug"),
                    "topic_id": normalize_openalex_id(row.get("topic_id") or row.get("id")),
                    "query": row.get("query") or row.get("display_name") or row.get("slug"),
                    "works_count": row.get("works_count", 0),
                    "field_name": row.get("field_name") or row.get("field") or "",
                    "subfield_name": row.get("subfield_name") or row.get("subfield") or "",
                    "seed_source": "fig3_auto",
                }
            )
    for slug, display_name, query, field_name, subfield_name in MANUAL_DOMAIN_SEEDS:
        rows.append(
            {
                "slug": slug,
                "display_name": display_name,
                "topic_id": "",
                "query": query,
                "works_count": 0,
                "field_name": field_name,
                "subfield_name": subfield_name,
                "seed_source": "manual_landmark",
            }
        )
    out = pd.DataFrame(rows)
    out["slug"] = out["slug"].map(slugify)
    out = out.drop_duplicates("slug", keep="first").head(int(max_domains)).reset_index(drop=True)
    for col in ["topic_id", "works_count", "field_name", "subfield_name", "seed_source"]:
        if col not in out.columns:
            out[col] = ""
    return out


def manual_landmarks() -> pd.DataFrame:
    rows = []
    for domain, label, doi, title, year in MANUAL_LANDMARKS:
        rows.append(
            {
                "domain": domain,
                "landmark_source": "manual",
                "source_id": f"{domain}:{label}",
                "label": label,
                "doi": normalize_doi(doi),
                "title": title,
                "year": int(year),
                "id": "",
                "match_confidence": 1.0 if doi else 0.75,
                "include_main": 1,
            }
        )
    return pd.DataFrame(rows)


def _refs_from_record(rec: Mapping[str, Any]) -> List[str]:
    refs = rec.get("refs") or rec.get("referenced_works") or []
    return [normalize_openalex_id(ref) for ref in refs if normalize_openalex_id(ref)]


def _selected_id_set(fig1_dir: Path) -> set[str]:
    selected = read_csv(fig1_dir / "works_selected.csv")
    if selected.empty or "id" not in selected.columns:
        return set()
    return set(selected["id"].astype(str))


def _fig1_selected_map(fig1_dir: Path) -> Dict[str, Dict[str, Any]]:
    selected = read_csv(fig1_dir / "works_selected.csv")
    if selected.empty or "id" not in selected.columns:
        return {}
    return {str(row["id"]): row for row in selected.to_dict("records")}


def import_fig1_domain(
    fig1_root: Path,
    domain: str,
    papers_per_domain: int,
    start_year: int,
    end_year: int,
    fetched_at: str,
) -> SourceTables:
    fig1_dir = fig1_root / domain
    raw_records = read_jsonl(fig1_dir / "works_raw.jsonl")
    selected_map = _fig1_selected_map(fig1_dir)
    if not raw_records:
        empty = pd.DataFrame()
        return SourceTables(empty, empty, empty, empty, empty, empty, {})

    selected_ids = _selected_id_set(fig1_dir)
    rows: List[Dict[str, Any]] = []
    legacy: List[Dict[str, Any]] = []
    for rec in raw_records:
        wid = normalize_openalex_id(rec.get("id"))
        year = pd.to_numeric(rec.get("year"), errors="coerce")
        if not wid or pd.isna(year) or int(year) < start_year or int(year) > end_year:
            continue
        selected = selected_map.get(wid, {})
        primary_topic = str(selected.get("primary_topic") or rec.get("primary_topic") or domain)
        display_label = str(selected.get("display_label") or selected.get("community_label") or primary_topic)
        community = selected.get("display_community")
        if pd.isna(community) if community is not None else True:
            community = stable_int_id(f"{domain}:{primary_topic}")
        anchor_label = str(selected.get("anchor_label") or rec.get("anchor_label") or "").strip()
        rows.append(
            {
                "id": wid,
                "short_id": short_openalex_id(wid),
                "doi": normalize_doi(selected.get("doi") or rec.get("doi")),
                "title": selected.get("title") or rec.get("title") or wid,
                "year": int(year),
                "domain": domain,
                "primary_field": primary_topic,
                "display_community": int(pd.to_numeric(community, errors="coerce") if not pd.isna(pd.to_numeric(community, errors="coerce")) else stable_int_id(f"{domain}:{primary_topic}")),
                "display_topic_id": "",
                "display_topic_label": display_label,
                "is_landmark": int(bool(anchor_label)),
                "anchor_label": anchor_label,
                "document_type": rec.get("type") or "",
                "cited_by_count": int(pd.to_numeric(rec.get("cited_by_count"), errors="coerce") if not pd.isna(pd.to_numeric(rec.get("cited_by_count"), errors="coerce")) else 0),
                "reference_count": len(_refs_from_record(rec)),
                "source_provider": "openalex",
                "source_dataset": "fig1_cache",
                "fetched_at": fetched_at,
                "referenced_works": json.dumps(_refs_from_record(rec), ensure_ascii=False),
                "partial_2026": int(int(year) >= 2026),
                "_priority": 0 if wid in selected_ids else 1,
            }
        )
        legacy.append({**rec, "id": wid, "refs": _refs_from_record(rec)})

    works = pd.DataFrame(rows)
    if works.empty:
        empty = pd.DataFrame()
        return SourceTables(empty, empty, empty, empty, empty, empty, {})
    works = works.sort_values(
        ["_priority", "is_landmark", "cited_by_count", "year"],
        ascending=[True, False, False, True],
    ).drop_duplicates("id").head(int(papers_per_domain)).drop(columns=["_priority"])
    selected_ids = set(works["id"].astype(str))
    citations = []
    legacy_by_id = {str(item.get("id")): item for item in legacy}
    for wid in selected_ids:
        rec = legacy_by_id.get(wid, {})
        for ref in _refs_from_record(rec):
            if ref in selected_ids:
                citations.append({"source": wid, "target": ref, "relation": "reference", "source_dataset": "fig1_cache"})
    citations_df = pd.DataFrame(citations).drop_duplicates() if citations else pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])
    topics_df, topic_edges_df = build_topics_and_edges(works, citations_df)
    domains = pd.DataFrame([{"slug": domain, "display_name": domain, "seed_source": "fig1_cache"}])
    landmarks = works[works["is_landmark"] == 1][["domain", "id", "anchor_label", "doi", "title", "year"]].rename(columns={"anchor_label": "label"})
    landmarks["landmark_source"] = "fig1_anchor"
    landmarks["source_id"] = landmarks["domain"].astype(str) + ":" + landmarks["label"].astype(str)
    landmarks["match_confidence"] = 1.0
    landmarks["include_main"] = 1
    legacy_selected = [legacy_by_id[wid] for wid in selected_ids if wid in legacy_by_id]
    return SourceTables(works, citations_df, topics_df, topic_edges_df, domains, landmarks, {domain: legacy_selected})


def import_fig3_auto(
    source_root: Path,
    domains: Sequence[str],
    papers_per_domain: int,
    start_year: int,
    end_year: int,
    fetched_at: str,
) -> SourceTables:
    works = read_csv(source_root / "works.csv")
    citations = read_csv(source_root / "citations.csv")
    topics = read_csv(source_root / "topics.csv")
    topic_edges = read_csv(source_root / "topic_edges.csv")
    domain_seeds = read_csv(source_root / "domain_seeds.csv")
    landmarks = read_csv(source_root / "landmark_registry.csv")
    if works.empty:
        empty = pd.DataFrame()
        return SourceTables(empty, empty, empty, empty, empty, empty, {})

    allowed = set(domains)
    works = works[works["domain"].astype(str).isin(allowed)].copy() if allowed else works.copy()
    works["year"] = pd.to_numeric(works["year"], errors="coerce")
    works = works[works["year"].between(start_year, end_year, inclusive="both")].copy()
    works["cited_by_count"] = pd.to_numeric(works.get("cited_by_count", 0), errors="coerce").fillna(0)
    works["is_landmark"] = pd.to_numeric(works.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    works = (
        works.sort_values(["domain", "is_landmark", "cited_by_count"], ascending=[True, False, False])
        .groupby("domain", as_index=False, group_keys=False)
        .head(int(papers_per_domain))
        .copy()
    )
    if works.empty:
        empty = pd.DataFrame()
        return SourceTables(empty, empty, empty, empty, empty, empty, {})
    works["id"] = works["id"].map(normalize_openalex_id)
    works["short_id"] = works["id"].map(short_openalex_id)
    works["doi"] = works.get("doi", "").map(normalize_doi)
    works["title"] = works.get("title", works["id"]).fillna(works["id"])
    works["primary_field"] = works.get("primary_field", works.get("display_topic_label", "unknown_field")).fillna("unknown_field")
    works["display_topic_id"] = works.get("display_topic_id", "")
    works["display_topic_label"] = works.get("display_topic_label", works["primary_field"]).fillna(works["primary_field"])
    works["anchor_label"] = works.get("anchor_label", "")
    works["document_type"] = works.get("document_type", works.get("type", ""))
    works["reference_count"] = pd.to_numeric(works.get("reference_count", 0), errors="coerce").fillna(0).astype(int)
    works["source_provider"] = "openalex"
    works["source_dataset"] = "fig3_auto"
    works["fetched_at"] = fetched_at
    works["referenced_works"] = "[]"
    works["partial_2026"] = (works["year"].astype(int) >= 2026).astype(int)
    keep = root_work_columns()
    works = works[[col for col in keep if col in works.columns]].copy()

    selected_ids = set(works["id"].astype(str))
    if citations.empty:
        citations = pd.DataFrame(columns=["source", "target"])
    citations = citations.copy()
    citations["source"] = citations["source"].map(normalize_openalex_id)
    citations["target"] = citations["target"].map(normalize_openalex_id)
    citations = citations[citations["source"].isin(selected_ids) & citations["target"].isin(selected_ids)].copy()
    citations["relation"] = "reference"
    citations["source_dataset"] = "fig3_auto"
    citations = citations.drop_duplicates(["source", "target"])

    communities = set(pd.to_numeric(works["display_community"], errors="coerce").dropna().astype(int))
    if not topics.empty and "community" in topics.columns:
        topics = topics[pd.to_numeric(topics["community"], errors="coerce").isin(communities)].copy()
    else:
        topics = pd.DataFrame()
    if not topic_edges.empty:
        topic_edges = normalize_topic_edges(topic_edges)
        topic_edges = topic_edges[
            topic_edges["source_community"].isin(communities)
            & topic_edges["target_community"].isin(communities)
        ].copy()
    else:
        topic_edges = pd.DataFrame()
    if topics.empty or topic_edges.empty:
        topics, topic_edges = build_topics_and_edges(works, citations)
    if not domain_seeds.empty:
        domain_seeds = domain_seeds[domain_seeds["slug"].map(slugify).isin(set(works["domain"].astype(str)))].copy()
        domain_seeds["seed_source"] = "fig3_auto"
    if not landmarks.empty:
        landmarks = normalize_landmark_registry(landmarks, "fig3_auto")
        landmarks = landmarks[landmarks["id"].isin(selected_ids) | landmarks["domain"].isin(set(works["domain"].astype(str)))].copy()
    return SourceTables(works, citations, topics, topic_edges, domain_seeds, landmarks, {})


def root_work_columns() -> List[str]:
    return [
        "id",
        "short_id",
        "doi",
        "title",
        "year",
        "domain",
        "primary_field",
        "display_community",
        "display_topic_id",
        "display_topic_label",
        "legacy_is_landmark",
        "is_landmark",
        "anchor_label",
        "reliable_anchor_source",
        "anchor_policy",
        "document_type",
        "cited_by_count",
        "reference_count",
        "source_provider",
        "source_dataset",
        "fetched_at",
        "referenced_works",
        "partial_2026",
    ]


def _ensure_work_columns(works: pd.DataFrame) -> pd.DataFrame:
    out = works.copy()
    for col in root_work_columns():
        if col not in out.columns:
            out[col] = ""
    return out[root_work_columns()].copy()


def normalize_landmark_registry(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "domain" not in out.columns:
        out["domain"] = out.get("primary_topic", "").map(slugify)
    if "label" not in out.columns:
        out["label"] = out.get("title", out.get("source_prize_id", "landmark"))
    if "source_id" not in out.columns:
        out["source_id"] = out.get("source_prize_id", out["label"])
    if "landmark_source" not in out.columns:
        out["landmark_source"] = source
    for col in ["id", "doi"]:
        if col not in out.columns:
            out[col] = ""
    out["id"] = out["id"].map(normalize_openalex_id)
    out["doi"] = out["doi"].map(normalize_doi)
    if "include_main" not in out.columns:
        out["include_main"] = 1
    out["include_main"] = pd.to_numeric(out["include_main"], errors="coerce").fillna(1).astype(int)
    if "match_confidence" not in out.columns:
        out["match_confidence"] = 1.0
    out["match_confidence"] = pd.to_numeric(out["match_confidence"], errors="coerce").fillna(1.0)
    for col in ["title", "year"]:
        if col not in out.columns:
            out[col] = ""
    return out[
        ["domain", "landmark_source", "source_id", "label", "id", "doi", "title", "year", "match_confidence", "include_main"]
    ].copy()


def prepare_strict_landmarks(landmarks: pd.DataFrame, complete_end_year: int = DEFAULT_COMPLETE_END_YEAR) -> pd.DataFrame:
    if landmarks.empty:
        return pd.DataFrame(
            columns=[
                "domain",
                "label",
                "id_norm",
                "doi_norm",
                "title_norm",
                "year",
                "landmark_source",
                "include_main",
                "has_label",
            ]
        )
    out = normalize_landmark_registry(landmarks, "strict_policy")
    out["domain"] = out["domain"].map(slugify)
    out["label"] = out["label"].map(nonempty_text)
    out["id_norm"] = out["id"].map(normalize_openalex_id)
    out["doi_norm"] = out["doi"].map(normalize_doi)
    out["title_norm"] = out["title"].map(normalize_title)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["include_main"] = pd.to_numeric(out["include_main"], errors="coerce").fillna(1).astype(int)
    out["has_label"] = out["label"].astype(str).str.strip() != ""
    out = out[out["include_main"] == 1].copy()
    if complete_end_year:
        out = out[out["year"].isna() | (out["year"] <= int(complete_end_year))].copy()
    return out.reset_index(drop=True)


def _domain_value_sets(df: pd.DataFrame, value_col: str) -> Dict[str, set[str]]:
    if df.empty or value_col not in df.columns:
        return {}
    rows = df[df[value_col].astype(str).str.strip() != ""]
    return rows.groupby("domain")[value_col].apply(lambda values: set(values.astype(str))).to_dict()


def _strict_label_lookup(landmarks: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    lookup: Dict[Tuple[str, str, str], str] = {}
    if landmarks.empty:
        return lookup
    labeled = landmarks[landmarks["has_label"].astype(bool)].copy()
    for row in labeled.to_dict("records"):
        domain = str(row.get("domain") or "")
        label = str(row.get("label") or "")
        exact_keys = [key_col for key_col in ["id_norm", "doi_norm"] if str(row.get(key_col) or "")]
        key_cols = exact_keys or ["title_norm"]
        for key_col in key_cols:
            value = str(row.get(key_col) or "")
            if value:
                lookup[(domain, key_col, value)] = label
    return lookup


def _lookup_strict_label(row: Mapping[str, Any], lookup: Mapping[Tuple[str, str, str], str]) -> str:
    domain = str(row.get("domain") or "")
    for key_col in ["id_norm", "doi_norm", "title_norm"]:
        value = str(row.get(key_col) or "")
        label = lookup.get((domain, key_col, value))
        if label:
            return label
    return ""


def apply_strict_anchor_policy(
    works: pd.DataFrame,
    landmarks: pd.DataFrame,
    complete_end_year: int = DEFAULT_COMPLETE_END_YEAR,
    noisy_ratio: float = 0.25,
) -> pd.DataFrame:
    """Recompute clean landmark flags while preserving the legacy raw flag.

    Noisy domains often came from legacy Fig. 1 caches where every selected row
    inherited `is_landmark=1`. Strict mode trusts labeled landmark registry rows,
    explicit low-density anchor labels, and non-noisy raw landmark flags.
    """
    if works.empty:
        return works.copy()

    out = works.copy()
    if "legacy_is_landmark" not in out.columns and "is_landmark" in out.columns:
        out["legacy_is_landmark"] = out["is_landmark"]
    for col in root_work_columns():
        if col not in out.columns:
            out[col] = ""
    out["domain"] = out["domain"].map(slugify)
    out["id"] = out["id"].map(normalize_openalex_id)
    out["short_id"] = out["id"].map(short_openalex_id)
    out["legacy_is_landmark"] = pd.to_numeric(
        out.get("legacy_is_landmark", out.get("is_landmark", 0)),
        errors="coerce",
    ).fillna(0).astype(int)
    raw_is_landmark = pd.to_numeric(out.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    out["doi_norm"] = out.get("doi", "").map(normalize_doi)
    out["title_norm"] = out.get("title", "").map(normalize_title)
    out["id_norm"] = out["id"].map(normalize_openalex_id)
    out["anchor_label_clean"] = out.get("anchor_label", "").map(nonempty_text)
    out["has_anchor_label"] = out["anchor_label_clean"].astype(str).str.strip() != ""

    lm = prepare_strict_landmarks(landmarks, complete_end_year=complete_end_year)
    id_sets = _domain_value_sets(lm, "id_norm")
    doi_sets = _domain_value_sets(lm, "doi_norm")
    title_fallback = lm[
        lm.get("id_norm", pd.Series("", index=lm.index)).astype(str).str.strip().eq("")
        & lm.get("doi_norm", pd.Series("", index=lm.index)).astype(str).str.strip().eq("")
    ].copy()
    title_sets = _domain_value_sets(title_fallback, "title_norm")
    label_lookup = _strict_label_lookup(lm)

    out["strict_landmark_id_match"] = [
        bool(row.id_norm and row.id_norm in id_sets.get(row.domain, set()))
        for row in out[["domain", "id_norm"]].itertuples(index=False)
    ]
    out["strict_landmark_doi_match"] = [
        bool(row.doi_norm and row.doi_norm in doi_sets.get(row.domain, set()))
        for row in out[["domain", "doi_norm"]].itertuples(index=False)
    ]
    out["strict_landmark_title_match"] = [
        bool(row.title_norm and row.title_norm in title_sets.get(row.domain, set()))
        for row in out[["domain", "title_norm"]].itertuples(index=False)
    ]
    out["strict_landmark_label"] = [_lookup_strict_label(row, label_lookup) for row in out.to_dict("records")]
    out["strict_landmark_labeled_match"] = out["strict_landmark_label"].astype(str).str.strip() != ""
    out["strict_landmark_any_match"] = (
        out["strict_landmark_id_match"].astype(bool)
        | out["strict_landmark_doi_match"].astype(bool)
        | out["strict_landmark_title_match"].astype(bool)
    )

    stats = (
        out.assign(raw_is_landmark=raw_is_landmark)
        .groupby("domain", sort=False)
        .agg(
            raw_is_landmark_rate=("raw_is_landmark", "mean"),
            anchor_label_rate=("has_anchor_label", "mean"),
            registry_match_rate=("strict_landmark_any_match", "mean"),
        )
    )
    raw_noisy = stats["raw_is_landmark_rate"] > float(noisy_ratio)
    anchor_label_noisy = stats["anchor_label_rate"] > float(noisy_ratio)
    registry_noisy = stats["registry_match_rate"] > float(noisy_ratio)
    out["domain_is_landmark_noisy"] = out["domain"].map(raw_noisy.to_dict()).fillna(False).astype(bool)
    out["domain_anchor_label_noisy"] = out["domain"].map(anchor_label_noisy.to_dict()).fillna(False).astype(bool)
    out["domain_registry_match_noisy"] = out["domain"].map(registry_noisy.to_dict()).fillna(False).astype(bool)
    out["domain_anchor_flags_noisy"] = (
        out["domain_is_landmark_noisy"]
        | out["domain_anchor_label_noisy"]
        | out["domain_registry_match_noisy"]
    )

    raw_reliable = (raw_is_landmark == 1) & (~out["domain_is_landmark_noisy"].astype(bool))
    anchor_reliable = out["has_anchor_label"].astype(bool) & (~out["domain_anchor_label_noisy"].astype(bool))
    registry_reliable = out["strict_landmark_labeled_match"].astype(bool) | (
        out["strict_landmark_any_match"].astype(bool)
        & (~out["domain_is_landmark_noisy"].astype(bool))
        & (~out["domain_registry_match_noisy"].astype(bool))
    )
    reliable = registry_reliable | anchor_reliable | raw_reliable

    out["is_landmark"] = reliable.astype(int)
    source = []
    for row, raw_ok, anchor_ok, registry_ok in zip(
        out.to_dict("records"),
        raw_reliable.tolist(),
        anchor_reliable.tolist(),
        registry_reliable.tolist(),
    ):
        if bool(row.get("strict_landmark_labeled_match")):
            source.append("landmarks_csv_labeled")
        elif registry_ok:
            source.append("landmarks_csv")
        elif anchor_ok:
            source.append("anchor_label_representative")
        elif raw_ok:
            source.append("is_landmark")
        else:
            source.append("")
    out["reliable_anchor_source"] = source
    out["anchor_policy"] = "strict"
    clean_label = out["anchor_label_clean"].where(out["is_landmark"].astype(int) == 1, "")
    fill_label = out["strict_landmark_label"].where(clean_label.astype(str).str.strip() == "", clean_label)
    out["anchor_label"] = fill_label.where(out["is_landmark"].astype(int) == 1, "").fillna("")
    return out.drop(columns=["anchor_label_clean", "has_anchor_label"], errors="ignore")


def normalize_topic_edges(topic_edges: pd.DataFrame) -> pd.DataFrame:
    out = topic_edges.copy()
    if "source" in out.columns and "source_community" not in out.columns:
        out = out.rename(columns={"source": "source_community"})
    if "target" in out.columns and "target_community" not in out.columns:
        out = out.rename(columns={"target": "target_community"})
    if "weight" not in out.columns:
        out["weight"] = 1.0
    out["source_community"] = pd.to_numeric(out["source_community"], errors="coerce").fillna(-1).astype(int)
    out["target_community"] = pd.to_numeric(out["target_community"], errors="coerce").fillna(-1).astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0)
    return out[["source_community", "target_community", "weight"]].copy()


def build_topics_and_edges(works: pd.DataFrame, citations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if works.empty:
        return pd.DataFrame(columns=["community", "label", "x", "y", "domain", "topic_id"]), pd.DataFrame(columns=["source_community", "target_community", "weight"])
    topics = (
        works[["display_community", "display_topic_label", "domain", "display_topic_id"]]
        .drop_duplicates("display_community")
        .rename(columns={"display_community": "community", "display_topic_label": "label", "display_topic_id": "topic_id"})
        .copy()
    )
    topics["community"] = pd.to_numeric(topics["community"], errors="coerce").fillna(-1).astype(int)
    n = max(1, len(topics))
    topics = topics.sort_values(["domain", "community"]).reset_index(drop=True)
    topics["x"] = [math.cos(2.0 * math.pi * idx / n) for idx in range(n)]
    topics["y"] = [math.sin(2.0 * math.pi * idx / n) for idx in range(n)]
    if citations.empty:
        edges = pd.DataFrame(columns=["source_community", "target_community", "weight"])
    else:
        comm = works.set_index("id")["display_community"].to_dict()
        tmp = citations[["source", "target"]].copy()
        tmp["source_community"] = tmp["source"].map(comm)
        tmp["target_community"] = tmp["target"].map(comm)
        tmp = tmp.dropna(subset=["source_community", "target_community"])
        tmp = tmp[tmp["source_community"] != tmp["target_community"]]
        if tmp.empty:
            edges = pd.DataFrame(columns=["source_community", "target_community", "weight"])
        else:
            edges = (
                tmp.groupby(["source_community", "target_community"], as_index=False)
                .size()
                .rename(columns={"size": "weight"})
            )
            edges["source_community"] = edges["source_community"].astype(int)
            edges["target_community"] = edges["target_community"].astype(int)
    return topics[["community", "label", "x", "y", "domain", "topic_id"]], edges


def merge_source_tables(sources: Sequence[SourceTables], seed_domains: pd.DataFrame) -> SourceTables:
    works_parts = [src.works for src in sources if not src.works.empty]
    citations_parts = [src.citations for src in sources if not src.citations.empty]
    topics_parts = [src.topics for src in sources if not src.topics.empty]
    topic_edges_parts = [src.topic_edges for src in sources if not src.topic_edges.empty]
    domain_parts = [seed_domains] + [src.domains for src in sources if not src.domains.empty]
    landmark_parts = [manual_landmarks()] + [src.landmarks for src in sources if not src.landmarks.empty]

    works = pd.concat(works_parts, ignore_index=True, sort=False) if works_parts else pd.DataFrame(columns=root_work_columns())
    if "legacy_is_landmark" not in works.columns and "is_landmark" in works.columns:
        works["legacy_is_landmark"] = works["is_landmark"]
    numeric_work_cols = {
        "year",
        "display_community",
        "legacy_is_landmark",
        "is_landmark",
        "cited_by_count",
        "reference_count",
        "partial_2026",
    }
    for col in root_work_columns():
        if col not in works.columns:
            works[col] = "" if col not in numeric_work_cols else 0
    works = works[root_work_columns()].drop_duplicates("id", keep="first").reset_index(drop=True)
    works["year"] = pd.to_numeric(works["year"], errors="coerce").fillna(0).astype(int)
    works["display_community"] = pd.to_numeric(works["display_community"], errors="coerce").fillna(-1).astype(int)
    works["legacy_is_landmark"] = pd.to_numeric(works["legacy_is_landmark"], errors="coerce").fillna(0).astype(int)
    works["is_landmark"] = pd.to_numeric(works["is_landmark"], errors="coerce").fillna(0).astype(int)
    works["cited_by_count"] = pd.to_numeric(works["cited_by_count"], errors="coerce").fillna(0).astype(int)
    works["reference_count"] = pd.to_numeric(works["reference_count"], errors="coerce").fillna(0).astype(int)
    works["partial_2026"] = pd.to_numeric(works["partial_2026"], errors="coerce").fillna(0).astype(int)

    selected_ids = set(works["id"].astype(str))
    citations = pd.concat(citations_parts, ignore_index=True, sort=False) if citations_parts else pd.DataFrame(columns=["source", "target"])
    if not citations.empty:
        citations["source"] = citations["source"].astype(str)
        citations["target"] = citations["target"].astype(str)
        citations = citations[citations["source"].isin(selected_ids) & citations["target"].isin(selected_ids)].copy()
        if "relation" not in citations.columns:
            citations["relation"] = "reference"
        if "source_dataset" not in citations.columns:
            citations["source_dataset"] = "unknown"
        citations = citations.drop_duplicates(["source", "target"]).reset_index(drop=True)

    topics = pd.concat(topics_parts, ignore_index=True, sort=False) if topics_parts else pd.DataFrame()
    if topics.empty:
        topics, topic_edges = build_topics_and_edges(works, citations)
    else:
        for col in ["domain", "topic_id"]:
            if col not in topics.columns:
                topics[col] = ""
        topics = topics.drop_duplicates(["domain", "community"], keep="first").reset_index(drop=True)
        topic_edges = pd.concat(topic_edges_parts, ignore_index=True, sort=False) if topic_edges_parts else pd.DataFrame()
        topic_edges = normalize_topic_edges(topic_edges) if not topic_edges.empty else build_topics_and_edges(works, citations)[1]

    domains = pd.concat(domain_parts, ignore_index=True, sort=False).copy()
    domains["slug"] = domains["slug"].map(slugify)
    domains = domains.drop_duplicates("slug", keep="first").reset_index(drop=True)
    if "n_works" in domains.columns:
        domains = domains.drop(columns=["n_works"])
    work_counts = works.groupby("domain").size().rename("n_works").reset_index().rename(columns={"domain": "slug"})
    domains = domains.merge(work_counts, on="slug", how="left")
    domains["n_works"] = pd.to_numeric(domains["n_works"], errors="coerce").fillna(0).astype(int)

    landmarks = pd.concat(landmark_parts, ignore_index=True, sort=False) if landmark_parts else pd.DataFrame()
    landmarks = normalize_landmark_registry(landmarks, "merged") if not landmarks.empty else pd.DataFrame()
    if not landmarks.empty:
        landmarks = landmarks.drop_duplicates(["domain", "label", "doi"], keep="first").reset_index(drop=True)

    legacy_raw: Dict[str, List[Dict[str, Any]]] = {}
    for source in sources:
        for domain, rows in source.legacy_raw.items():
            legacy_raw.setdefault(domain, []).extend(rows)
    return SourceTables(works, citations, topics, topic_edges, domains, landmarks, legacy_raw)


def write_corpus(corpus_dir: Path, tables: SourceTables, manifest: Mapping[str, Any]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables.domains, corpus_dir / "domains.csv")
    write_csv(tables.landmarks, corpus_dir / "landmarks.csv")
    write_csv(tables.works, corpus_dir / "works.csv")
    write_csv(tables.citations, corpus_dir / "citations.csv")
    write_csv(tables.topics, corpus_dir / "topics.csv")
    write_csv(tables.topic_edges, corpus_dir / "topic_edges.csv")
    legacy_dir = corpus_dir / "legacy_imports" / "fig1"
    for domain, rows in tables.legacy_raw.items():
        path = legacy_dir / domain / "works_raw.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(corpus_dir / "manifest.json", manifest)


def build_offline_corpus(args: argparse.Namespace) -> SourceTables:
    fetched_at = utc_now()
    seeds = domain_seed_table(args.fig3_auto_root, args.max_domains)
    sources: List[SourceTables] = []
    core_domains = [row["slug"] for row in CORE_DOMAINS if row["slug"] in set(seeds["slug"])]
    for domain in core_domains:
        progress_log(f"Importing Fig. 1 cache for {domain}", args.quiet)
        sources.append(
            import_fig1_domain(
                args.fig1_root,
                domain,
                papers_per_domain=args.papers_per_domain,
                start_year=args.start_year,
                end_year=args.end_year,
                fetched_at=fetched_at,
            )
        )
    remaining_auto_domains = [
        slug for slug in seeds["slug"].astype(str).tolist()
        if slug not in core_domains
    ]
    progress_log(f"Importing Fig. 3 auto corpus domains: {len(remaining_auto_domains)} requested", args.quiet)
    sources.append(
        import_fig3_auto(
            args.fig3_auto_root,
            remaining_auto_domains,
            papers_per_domain=args.papers_per_domain,
            start_year=args.start_year,
            end_year=args.end_year,
            fetched_at=fetched_at,
        )
    )
    return merge_source_tables(sources, seeds)


def _db_domain_seed(row: Mapping[str, Any]) -> Any:
    from experiments.kg_perturbation_fig3 import dataset_builder as db  # pylint: disable=import-outside-toplevel

    return db.DomainSeed(
        slug=slugify(row.get("slug")),
        display_name=str(row.get("display_name") or row.get("slug") or ""),
        topic_id=normalize_openalex_id(row.get("topic_id") or ""),
        query=str(row.get("query") or row.get("display_name") or row.get("slug") or ""),
        works_count=int(pd.to_numeric(row.get("works_count", 0), errors="coerce") if not pd.isna(pd.to_numeric(row.get("works_count", 0), errors="coerce")) else 0),
        field_name=str(row.get("field_name") or ""),
        subfield_name=str(row.get("subfield_name") or ""),
    )


def _resolved_manual_landmarks(domain: str, openalex: Any, max_candidates: int, quiet: bool = False) -> pd.DataFrame:
    from experiments.kg_perturbation_fig3 import dataset_builder as db  # pylint: disable=import-outside-toplevel

    rows: List[Dict[str, Any]] = []
    manual = manual_landmarks()
    manual = manual[manual["domain"].astype(str) == str(domain)].copy()
    records = manual.to_dict("records")
    if records:
        progress_log(f"{domain}: resolving {len(records):,} manual landmark seeds", quiet)
    for idx, row in enumerate(records, start=1):
        progress_log(f"{domain}: landmark {idx:,}/{len(records):,} {row.get('label') or row.get('title') or ''}", quiet)
        work = None
        if row.get("doi"):
            work = openalex.get_work(str(row["doi"]))
        if work is None and row.get("title"):
            try:
                candidates = openalex.list_works(
                    max_records=max_candidates,
                    search=str(row["title"]),
                    filters=[
                        "type:article|preprint|review|book-chapter|book",
                        "is_retracted:false",
                        "is_paratext:false",
                    ],
                    sort="relevance_score:desc",
                )
            except Exception:
                candidates = []
            work = candidates[0] if candidates else None
        if not work:
            rows.append(row)
            continue
        primary = work.get("primary_topic") or {}
        wid = normalize_openalex_id(work.get("id"))
        rows.append(
            {
                "domain": domain,
                "landmark_source": row.get("landmark_source", "manual"),
                "source_id": row.get("source_id") or f"{domain}:{row.get('label')}",
                "label": row.get("label") or row.get("title") or wid,
                "id": wid,
                "doi": normalize_doi(work.get("doi") or row.get("doi")),
                "title": work.get("display_name") or row.get("title") or "",
                "year": int(work.get("publication_year") or row.get("year") or 0),
                "match_confidence": row.get("match_confidence", 1.0),
                "include_main": 1,
                "primary_topic_id": normalize_openalex_id(primary.get("id")),
                "primary_topic": primary.get("display_name", ""),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["id"] = out.get("id", "").map(normalize_openalex_id)
        out["include_main"] = pd.to_numeric(out.get("include_main", 1), errors="coerce").fillna(1).astype(int)
    if records:
        progress_log(f"{domain}: resolved {len(out):,} manual landmark rows", quiet)
    return out


def fetch_online_missing_sources(args: argparse.Namespace, seeds: pd.DataFrame, current: SourceTables) -> List[SourceTables]:
    if args.offline:
        return []
    try:
        from experiments.kg_perturbation_fig3 import dataset_builder as db  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise RuntimeError(f"Cannot import Fig. 3 dataset builder for OpenAlex fetch: {exc}") from exc

    existing_domains = set(current.works["domain"].astype(str)) if not current.works.empty else set()
    missing = [row for row in seeds.to_dict("records") if str(row.get("slug")) not in existing_domains]
    if not missing:
        return []

    checkpoint_root = args.checkpoint_dir or (args.out_dir / "online_checkpoints")
    openalex = db.OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.openalex_email,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    fetched_at = utc_now()
    out: List[SourceTables] = []
    for i, row in enumerate(missing, start=1):
        seed = _db_domain_seed(row)
        checkpoint = read_source_checkpoint(checkpoint_root, seed.slug)
        if checkpoint is not None:
            progress_log(f"Using checkpoint for OpenAlex domain {i}/{len(missing)}: {seed.slug}", args.quiet)
            out.append(checkpoint)
            continue
        progress_log(f"Fetching OpenAlex domain {i}/{len(missing)}: {seed.slug}", args.quiet)
        landmarks = _resolved_manual_landmarks(seed.slug, openalex, args.max_candidates_per_landmark, quiet=args.quiet)
        try:
            works, citations, _report = db.collect_domain_works(
                seed,
                landmarks=landmarks,
                openalex=openalex,
                max_papers_per_domain=args.papers_per_domain,
                max_anchor_citers=args.max_anchor_citers,
                start_year=args.start_year,
                end_year=args.end_year,
                work_types=args.work_types,
                progress=not args.quiet,
            )
        except Exception as exc:
            progress_log(f"Skipping {seed.slug}; OpenAlex fetch failed: {exc}", args.quiet)
            continue
        if works.empty:
            continue
        works["source_provider"] = "openalex"
        works["source_dataset"] = "openalex_live"
        works["fetched_at"] = fetched_at
        works["anchor_label"] = works.get("anchor_label", "")
        if "legacy_is_landmark" not in works.columns and "is_landmark" in works.columns:
            works["legacy_is_landmark"] = works["is_landmark"]
        works["referenced_works"] = "[]"
        works["partial_2026"] = (pd.to_numeric(works["year"], errors="coerce").fillna(0).astype(int) >= 2026).astype(int)
        numeric_work_cols = {
            "year",
            "display_community",
            "legacy_is_landmark",
            "is_landmark",
            "cited_by_count",
            "reference_count",
            "partial_2026",
        }
        for col in root_work_columns():
            if col not in works.columns:
                works[col] = "" if col not in numeric_work_cols else 0
        works = works[root_work_columns()].copy()
        if citations.empty:
            citations = pd.DataFrame(columns=["source", "target"])
        citations = citations.copy()
        citations["relation"] = "reference"
        citations["source_dataset"] = "openalex_live"
        topics, topic_edges = db.build_topics_and_edges(works, citations[["source", "target"]])
        domains = pd.DataFrame(
            [
                {
                    "slug": seed.slug,
                    "display_name": seed.display_name,
                    "topic_id": seed.topic_id,
                    "query": seed.query,
                    "works_count": seed.works_count,
                    "field_name": seed.field_name,
                    "subfield_name": seed.subfield_name,
                    "seed_source": "openalex_live",
                }
            ]
        )
        source = SourceTables(
            works=works,
            citations=citations,
            topics=topics,
            topic_edges=topic_edges,
            domains=domains,
            landmarks=normalize_landmark_registry(landmarks, "manual") if not landmarks.empty else pd.DataFrame(),
            legacy_raw={},
        )
        write_source_checkpoint(checkpoint_root, seed.slug, source)
        out.append(source)
    return out


def read_source_checkpoint(checkpoint_root: Path, domain: str) -> Optional[SourceTables]:
    domain_dir = checkpoint_root / domain
    works_path = domain_dir / "works.csv"
    if not works_path.exists():
        return None
    works = read_csv(works_path)
    if works.empty:
        return None
    return SourceTables(
        works=works,
        citations=read_csv(domain_dir / "citations.csv"),
        topics=read_csv(domain_dir / "topics.csv"),
        topic_edges=read_csv(domain_dir / "topic_edges.csv"),
        domains=read_csv(domain_dir / "domains.csv"),
        landmarks=read_csv(domain_dir / "landmarks.csv"),
        legacy_raw={},
    )


def write_source_checkpoint(checkpoint_root: Path, domain: str, source: SourceTables) -> None:
    domain_dir = checkpoint_root / domain
    write_csv(source.works, domain_dir / "works.csv")
    write_csv(source.citations, domain_dir / "citations.csv")
    write_csv(source.topics, domain_dir / "topics.csv")
    write_csv(source.topic_edges, domain_dir / "topic_edges.csv")
    write_csv(source.domains, domain_dir / "domains.csv")
    write_csv(source.landmarks, domain_dir / "landmarks.csv")


def load_corpus(corpus_dir: Path) -> SourceTables:
    return SourceTables(
        works=read_csv(corpus_dir / "works.csv"),
        citations=read_csv(corpus_dir / "citations.csv"),
        topics=read_csv(corpus_dir / "topics.csv"),
        topic_edges=read_csv(corpus_dir / "topic_edges.csv"),
        domains=read_csv(corpus_dir / "domains.csv"),
        landmarks=read_csv(corpus_dir / "landmarks.csv"),
        legacy_raw={},
    )


def audit_corpus(corpus_dir: Path, min_papers_per_domain: int = 2500) -> Dict[str, Any]:
    tables = load_corpus(corpus_dir)
    works = tables.works.copy()
    citations = tables.citations.copy()
    domains = tables.domains.copy()
    landmarks = tables.landmarks.copy()
    if works.empty:
        report = {"overall_pass": False, "errors": ["works.csv has no rows"], "domains": []}
        write_json(corpus_dir / "quality_report.json", report)
        return report

    works["year"] = pd.to_numeric(works["year"], errors="coerce")
    works["is_landmark"] = pd.to_numeric(works.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    works["display_topic_label"] = works.get("display_topic_label", "").fillna("").astype(str)
    works["doi_norm"] = works.get("doi", "").map(normalize_doi)
    rows: List[Dict[str, Any]] = []
    for domain, sub in works.groupby("domain", sort=True):
        ids = set(sub["id"].astype(str))
        csub = citations[citations["source"].astype(str).isin(ids)] if not citations.empty else pd.DataFrame()
        doi_nonempty = sub[sub["doi_norm"].astype(str) != ""]
        duplicate_doi_rate = 0.0
        if len(doi_nonempty):
            duplicate_doi_rate = float(doi_nonempty.duplicated("doi_norm").mean())
        landmark_rows = int(sub["is_landmark"].sum())
        if not landmarks.empty and "domain" in landmarks.columns:
            landmark_rows = max(landmark_rows, int((landmarks["domain"].astype(str) == str(domain)).sum()))
        topic_coverage = float((sub["display_topic_label"].str.strip() != "").mean()) if len(sub) else 0.0
        row = {
            "domain": domain,
            "n_works": int(len(sub)),
            "n_landmarks_or_anchors": int(landmark_rows),
            "year_min": int(sub["year"].min()) if len(sub) else 0,
            "year_max": int(sub["year"].max()) if len(sub) else 0,
            "citation_rows": int(len(csub)),
            "citation_rows_per_work": float(len(csub) / max(1, len(sub))),
            "duplicate_doi_rate": duplicate_doi_rate,
            "topic_coverage": topic_coverage,
            "has_partial_2026": bool((sub["year"] >= 2026).any()),
        }
        row["passes"] = bool(
            row["n_works"] >= int(min_papers_per_domain)
            and row["n_landmarks_or_anchors"] >= 1
            and row["duplicate_doi_rate"] < 0.02
            and row["topic_coverage"] >= 0.90
            and not row["has_partial_2026"]
        )
        rows.append(row)
    domain_report = pd.DataFrame(rows)
    if not domains.empty:
        domain_report = domains[["slug"]].rename(columns={"slug": "domain"}).merge(domain_report, on="domain", how="left")
    domain_report = domain_report.fillna({"n_works": 0, "n_landmarks_or_anchors": 0, "passes": False})
    summary = {
        "overall_pass": bool(domain_report["passes"].fillna(False).all()) if len(domain_report) else False,
        "n_domains": int(works["domain"].nunique()),
        "n_configured_domains": int(len(domains)) if not domains.empty else int(works["domain"].nunique()),
        "n_works": int(len(works)),
        "n_citations": int(len(citations)),
        "n_topics": int(len(tables.topics)),
        "n_topic_edges": int(len(tables.topic_edges)),
        "min_papers_per_domain": int(min_papers_per_domain),
        "complete_end_year": DEFAULT_COMPLETE_END_YEAR,
        "domains": domain_report.to_dict("records"),
    }
    write_json(corpus_dir / "quality_report.json", summary)
    return summary


def _filter_complete_years(works: pd.DataFrame, include_partial_2026: bool) -> pd.DataFrame:
    out = _ensure_work_columns(works)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out[out["year"].notna()].copy()
    if not include_partial_2026:
        out = out[out["year"].astype(int) <= DEFAULT_COMPLETE_END_YEAR].copy()
    return out


def _view_works(works: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_work_columns(works)
    out["analysis_community"] = pd.to_numeric(out["display_community"], errors="coerce").fillna(-1).astype(int)
    out["community"] = out["analysis_community"]
    out["primary_topic"] = out.get("display_topic_label", out.get("primary_field", ""))
    out["community_label"] = out.get("display_topic_label", out.get("primary_field", ""))
    out["display_label"] = out.get("display_topic_label", out.get("primary_field", ""))
    out["is_closure_node"] = 0
    max_year = int(pd.to_numeric(out["year"], errors="coerce").max()) if len(out) else DEFAULT_COMPLETE_END_YEAR
    out["domain_analysis_end_year"] = min(max_year, DEFAULT_COMPLETE_END_YEAR)
    cols = [
        "id",
        "year",
        "title",
        "domain",
        "primary_field",
        "analysis_community",
        "display_community",
        "is_landmark",
        "anchor_label",
        "short_id",
        "doi",
        "cited_by_count",
        "reference_count",
        "source_dataset",
        "legacy_is_landmark",
        "reliable_anchor_source",
        "anchor_policy",
        "primary_topic",
        "community",
        "community_label",
        "display_label",
        "is_closure_node",
        "domain_analysis_end_year",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out[cols].copy()


def _filter_topics(topics: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    works = _ensure_work_columns(works)
    if topics.empty:
        return build_topics_and_edges(works, pd.DataFrame())[0]
    communities = set(pd.to_numeric(works["display_community"], errors="coerce").dropna().astype(int))
    out = topics.copy()
    out["community"] = pd.to_numeric(out["community"], errors="coerce").fillna(-1).astype(int)
    out = out[out["community"].isin(communities)].copy()
    if "domain" in out.columns and "domain" in works.columns:
        domains = set(works["domain"].dropna().astype(str))
        topic_domains = out["domain"].dropna().astype(str)
        if domains and topic_domains.ne("").any():
            out = out[out["domain"].astype(str).isin(domains)].copy()
    for col in ["label", "x", "y", "domain", "topic_id"]:
        if col not in out.columns:
            out[col] = "" if col in {"label", "domain", "topic_id"} else 0.0
    return out[["community", "label", "x", "y", "domain", "topic_id"]].copy()


def _filter_topic_edges(topic_edges: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    if topic_edges.empty:
        return pd.DataFrame(columns=["source_community", "target_community", "weight"])
    out = normalize_topic_edges(topic_edges)
    communities = set(pd.to_numeric(topics["community"], errors="coerce").dropna().astype(int))
    return out[out["source_community"].isin(communities) & out["target_community"].isin(communities)].copy()


def write_standard_view(
    view_dir: Path,
    works: pd.DataFrame,
    citations: pd.DataFrame,
    topics: pd.DataFrame,
    topic_edges: pd.DataFrame,
) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    works = _ensure_work_columns(works)
    if not {"source", "target"}.issubset(citations.columns):
        citations = pd.DataFrame(columns=["source", "target"])
    selected = set(works["id"].astype(str))
    csub = citations[citations["source"].astype(str).isin(selected) & citations["target"].astype(str).isin(selected)].copy()
    write_csv(_view_works(works), view_dir / "works.csv")
    write_csv(csub[["source", "target"]].drop_duplicates(), view_dir / "citations.csv")
    topic_sub = _filter_topics(topics, works)
    write_csv(topic_sub, view_dir / "topics.csv")
    write_csv(_filter_topic_edges(topic_edges, topic_sub), view_dir / "topic_edges.csv")


def namespace_multi_domain(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    topics: pd.DataFrame,
    topic_edges: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    works = _ensure_work_columns(works)
    if works.empty:
        return (
            pd.DataFrame(columns=root_work_columns()),
            pd.DataFrame(columns=["source", "target"]),
            pd.DataFrame(columns=["community", "label", "x", "y", "domain", "topic_id"]),
            pd.DataFrame(columns=["source_community", "target_community", "weight"]),
        )
    work_parts: List[pd.DataFrame] = []
    citation_parts: List[pd.DataFrame] = []
    topic_parts: List[pd.DataFrame] = []
    topic_edge_parts: List[pd.DataFrame] = []
    for offset_idx, domain in enumerate(sorted(works["domain"].astype(str).unique())):
        offset = offset_idx * 100_000
        sub = works[works["domain"].astype(str) == domain].copy()
        ids = set(sub["id"].astype(str))
        id_map = {wid: f"{domain}::{wid}" for wid in ids}
        comm_map = {
            int(comm): int(comm) + offset
            for comm in pd.to_numeric(sub["display_community"], errors="coerce").dropna().astype(int).unique()
        }
        sub["id"] = sub["id"].astype(str).map(id_map)
        sub["display_community"] = pd.to_numeric(sub["display_community"], errors="coerce").map(comm_map).fillna(-1).astype(int)
        work_parts.append(sub)

        csub = citations[citations["source"].astype(str).isin(ids) & citations["target"].astype(str).isin(ids)].copy()
        if not csub.empty:
            csub["source"] = csub["source"].astype(str).map(id_map)
            csub["target"] = csub["target"].astype(str).map(id_map)
            citation_parts.append(csub)

        tsub = _filter_topics(topics, works[works["domain"].astype(str) == domain]).copy()
        if not tsub.empty:
            tsub["community"] = tsub["community"].map(comm_map).fillna(-1).astype(int)
            tsub["label"] = domain + ": " + tsub["label"].astype(str)
            topic_parts.append(tsub)

        tesub = _filter_topic_edges(topic_edges, _filter_topics(topics, works[works["domain"].astype(str) == domain])).copy()
        if not tesub.empty:
            tesub["source_community"] = tesub["source_community"].map(comm_map).fillna(-1).astype(int)
            tesub["target_community"] = tesub["target_community"].map(comm_map).fillna(-1).astype(int)
            topic_edge_parts.append(tesub)
    out_works = pd.concat(work_parts, ignore_index=True, sort=False) if work_parts else pd.DataFrame(columns=root_work_columns())
    out_citations = pd.concat(citation_parts, ignore_index=True, sort=False) if citation_parts else pd.DataFrame(columns=["source", "target"])
    out_topics = pd.concat(topic_parts, ignore_index=True, sort=False) if topic_parts else pd.DataFrame()
    out_topic_edges = pd.concat(topic_edge_parts, ignore_index=True, sort=False) if topic_edge_parts else pd.DataFrame()
    return out_works, out_citations, out_topics, out_topic_edges


def write_fig1_view(
    view_dir: Path,
    works: pd.DataFrame,
    citations: pd.DataFrame,
    topics: pd.DataFrame,
    topic_edges: pd.DataFrame,
) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    w = _view_works(works)
    fig1_works = w[
        [
            "id",
            "short_id",
            "doi",
            "title",
            "year",
            "cited_by_count",
            "primary_topic",
            "community",
            "community_label",
            "display_community",
            "display_label",
            "anchor_label",
        ]
    ].copy()
    fig1_works["anchor_citer"] = 0
    fig1_works["reference_stub"] = 0
    write_csv(fig1_works.sort_values(["year", "cited_by_count"], ascending=[True, False]), view_dir / "works_selected.csv")
    csub = citations[citations["source"].isin(set(works["id"])) & citations["target"].isin(set(works["id"]))].copy()
    if csub.empty:
        edges = pd.DataFrame(columns=["source", "target", "weight", "direct", "bibliographic", "cocitation"])
    else:
        edges = csub[["source", "target"]].drop_duplicates().copy()
        edges["weight"] = 1.0
        edges["direct"] = 1
        edges["bibliographic"] = 0
        edges["cocitation"] = 0
    write_csv(edges, view_dir / "paper_edges.csv")
    topic_sub = _filter_topics(topics, works)
    counts = works.groupby("display_community").agg(
        n_papers=("id", "count"),
        cited_by_count=("cited_by_count", "sum"),
        first_year=("year", "min"),
    )
    topic_nodes = topic_sub.rename(columns={"community": "community"}).copy()
    topic_nodes = topic_nodes.merge(counts, left_on="community", right_index=True, how="left")
    topic_nodes["anchor_labels"] = ""
    write_csv(topic_nodes[["community", "label", "n_papers", "cited_by_count", "first_year", "anchor_labels", "x", "y"]], view_dir / "topic_nodes.csv")
    write_csv(_filter_topic_edges(topic_edges, topic_sub), view_dir / "topic_edges.csv")


def audit_strict_views(corpus_dir: Path) -> Dict[str, Any]:
    view_root = corpus_dir / "views" / "fig3"
    rows: List[Dict[str, Any]] = []
    if not view_root.exists():
        report = {"overall_pass": False, "errors": [f"{view_root} does not exist"], "domains": []}
        write_json(corpus_dir / "strict_view_audit.json", report)
        return report
    for domain_dir in sorted(path for path in view_root.iterdir() if path.is_dir() and path.name != "multi_domain"):
        works = read_csv(domain_dir / "works.csv")
        if works.empty:
            continue
        works["year"] = pd.to_numeric(works.get("year", 0), errors="coerce").fillna(0).astype(int)
        is_landmark = pd.to_numeric(works.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
        legacy_is_landmark = pd.to_numeric(works.get("legacy_is_landmark", is_landmark), errors="coerce").fillna(0).astype(int)
        non_landmark = works[is_landmark == 0]
        window_counts: List[int] = []
        for _, row in works[is_landmark == 1].iterrows():
            year = int(row.get("year") or 0)
            window_counts.append(int(non_landmark["year"].between(year - 5, year + 5).sum()))
        min_window_controls = min(window_counts) if window_counts else None
        landmark_rate = float(is_landmark.mean()) if len(works) else 0.0
        anchor_series = works["anchor_label"] if "anchor_label" in works.columns else pd.Series([""] * len(works))
        rows.append(
            {
                "domain": domain_dir.name,
                "n_works": int(len(works)),
                "n_landmarks": int(is_landmark.sum()),
                "is_landmark_rate": landmark_rate,
                "legacy_n_landmarks": int(legacy_is_landmark.sum()),
                "legacy_is_landmark_rate": float(legacy_is_landmark.mean()) if len(works) else 0.0,
                "n_anchor_labels": int(anchor_series.map(nonempty_text).astype(bool).sum()),
                "min_non_landmark_controls_pm5_years": min_window_controls,
                "pass_landmark_rate": int(landmark_rate <= 0.05),
                "pass_landmark_controls": int(min_window_controls is None or min_window_controls > 0),
            }
        )
    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df["passes"] = (
            report_df["pass_landmark_rate"].astype(int).eq(1)
            & report_df["pass_landmark_controls"].astype(int).eq(1)
        )
    write_csv(report_df, corpus_dir / "strict_view_audit.csv")
    report = {
        "overall_pass": bool(report_df["passes"].all()) if not report_df.empty else False,
        "n_domains": int(len(report_df)),
        "domains": report_df.to_dict("records") if not report_df.empty else [],
    }
    write_json(corpus_dir / "strict_view_audit.json", report)
    return report


def make_views(
    corpus_dir: Path,
    include_partial_2026: bool = False,
    anchor_policy: str = "legacy",
) -> None:
    tables = load_corpus(corpus_dir)
    all_works = tables.works.copy()
    if anchor_policy == "strict":
        all_works = apply_strict_anchor_policy(all_works, tables.landmarks, complete_end_year=DEFAULT_COMPLETE_END_YEAR)
    works = _filter_complete_years(all_works, include_partial_2026=include_partial_2026)
    citations = tables.citations
    domains = sorted([d for d in works["domain"].astype(str).unique() if d])
    for fig in ["fig2", "fig3"]:
        for domain in domains:
            sub = works[works["domain"].astype(str) == domain].copy()
            write_standard_view(corpus_dir / "views" / fig / domain, sub, citations, tables.topics, tables.topic_edges)
        multi = namespace_multi_domain(works, citations, tables.topics, tables.topic_edges)
        write_standard_view(corpus_dir / "views" / fig / "multi_domain", *multi)
    for domain in domains:
        sub = works[works["domain"].astype(str) == domain].copy()
        write_fig1_view(corpus_dir / "views" / "fig1" / domain, sub, citations, tables.topics, tables.topic_edges)
    # Fig. 5 consumes Fig. 3-shaped works/topics plus an external Fig. 3 score table.
    fig5_dir = corpus_dir / "views" / "fig5" / "multi_domain"
    multi = namespace_multi_domain(works, citations, tables.topics, tables.topic_edges)
    write_standard_view(fig5_dir, *multi)

    if include_partial_2026:
        partial = all_works[pd.to_numeric(all_works["year"], errors="coerce") >= 2026].copy()
        if not partial.empty:
            multi_partial = namespace_multi_domain(partial, citations, tables.topics, tables.topic_edges)
            write_standard_view(corpus_dir / "views" / "partial_2026" / "multi_domain", *multi_partial)
    write_json(
        corpus_dir / "views" / "view_manifest.json",
        {
            "materialized_at": utc_now(),
            "anchor_policy": anchor_policy,
            "include_partial_2026": bool(include_partial_2026),
            "complete_end_year": DEFAULT_COMPLETE_END_YEAR,
        },
    )
    if anchor_policy == "strict":
        audit_strict_views(corpus_dir)


def materialize_fig1_cache(
    corpus_dir: Path,
    out_dir: Path,
    domains: Sequence[str],
    anchor_policy: str = "legacy",
) -> None:
    make_views(corpus_dir, anchor_policy=anchor_policy)
    tables = load_corpus(corpus_dir)
    citations = tables.citations
    for domain in domains:
        source_view = corpus_dir / "views" / "fig1" / domain
        if not source_view.exists():
            continue
        target = out_dir / domain
        target.mkdir(parents=True, exist_ok=True)
        for name in ["works_selected.csv", "paper_edges.csv", "topic_nodes.csv", "topic_edges.csv"]:
            src = source_view / name
            if src.exists():
                target.joinpath(name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        works = tables.works[tables.works["domain"].astype(str) == domain].copy()
        ids = set(works["id"].astype(str))
        refs_by_source = citations[citations["source"].astype(str).isin(ids)].groupby("source")["target"].apply(list).to_dict()
        raw_path = target / "works_raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for row in works.to_dict("records"):
                refs = refs_by_source.get(str(row["id"]), [])
                rec = {
                    "id": row["id"],
                    "short_id": row.get("short_id") or short_openalex_id(row.get("id")),
                    "doi": row.get("doi", ""),
                    "title": row.get("title", ""),
                    "year": int(row.get("year") or 0),
                    "type": row.get("document_type", ""),
                    "language": "en",
                    "cited_by_count": int(row.get("cited_by_count") or 0),
                    "refs": refs,
                    "topics": [row.get("display_topic_label") or row.get("primary_field") or domain],
                    "primary_topic": row.get("display_topic_label") or row.get("primary_field") or domain,
                    "text": f"{row.get('title', '')} {row.get('display_topic_label', '')}",
                    "anchor_label": row.get("anchor_label", ""),
                    "anchor_year": int(row.get("year") or 0) if row.get("anchor_label") else None,
                    "anchor_citer": False,
                    "reference_stub": False,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_json(target / "cache_manifest.json", {"source": str(corpus_dir), "domain": domain, "materialized_at": utc_now()})


def build_command(args: argparse.Namespace) -> None:
    seeds = domain_seed_table(args.fig3_auto_root, args.max_domains)
    tables = build_offline_corpus(args)
    online_sources = fetch_online_missing_sources(args, seeds, tables)
    if online_sources:
        tables = merge_source_tables([tables] + online_sources, seeds)
    if args.anchor_policy == "strict":
        tables.works = apply_strict_anchor_policy(tables.works, tables.landmarks, complete_end_year=DEFAULT_COMPLETE_END_YEAR)
    manifest = {
        "profile": args.profile,
        "created_at": utc_now(),
        "corpus_dir": str(args.out_dir),
        "source_kind": "offline_import" if args.offline else "offline_import_plus_openalex_live",
        "anchor_policy": args.anchor_policy,
        "max_domains": int(args.max_domains),
        "papers_per_domain": int(args.papers_per_domain),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "complete_end_year": DEFAULT_COMPLETE_END_YEAR,
        "online_domains_fetched": len(online_sources),
        "openalex_api_key_count": len(split_api_keys([args.openalex_api_key, args.openalex_api_keys])),
        "works_rows": int(len(tables.works)),
        "citation_rows": int(len(tables.citations)),
        "topic_rows": int(len(tables.topics)),
        "topic_edge_rows": int(len(tables.topic_edges)),
        "domains_with_data": sorted(tables.works["domain"].astype(str).unique().tolist()) if not tables.works.empty else [],
        "notes": [
            "OpenAlex is the canonical graph source.",
            "Semantic Scholar caches remain legacy retrieval data and are not mixed into canonical citation graph.",
            "2026 is treated as partial and excluded from default experiment views.",
        ],
    }
    write_corpus(args.out_dir, tables, manifest)
    audit_corpus(args.out_dir, min_papers_per_domain=args.min_papers_per_domain)
    make_views(args.out_dir, include_partial_2026=args.include_partial_2026, anchor_policy=args.anchor_policy)
    progress_log(f"Wrote corpus to {args.out_dir}", args.quiet)


def audit_command(args: argparse.Namespace) -> None:
    report = audit_corpus(args.corpus_dir, min_papers_per_domain=args.min_papers_per_domain)
    progress_log(
        f"Audit overall_pass={report.get('overall_pass')} domains={report.get('n_domains')} works={report.get('n_works')}",
        args.quiet,
    )


def make_views_command(args: argparse.Namespace) -> None:
    make_views(
        args.corpus_dir,
        include_partial_2026=args.include_partial_2026,
        anchor_policy=args.anchor_policy,
    )
    progress_log(
        f"Materialized {args.anchor_policy} views under {args.corpus_dir / 'views'}",
        args.quiet,
    )


def derive_strict_command(args: argparse.Namespace) -> None:
    tables = load_corpus(args.source_dir)
    if tables.works.empty:
        raise ValueError(f"No works.csv found under {args.source_dir}")
    strict_works = apply_strict_anchor_policy(
        tables.works,
        tables.landmarks,
        complete_end_year=DEFAULT_COMPLETE_END_YEAR,
        noisy_ratio=args.noisy_landmark_ratio,
    )
    strict_tables = SourceTables(
        works=strict_works,
        citations=tables.citations,
        topics=tables.topics,
        topic_edges=tables.topic_edges,
        domains=tables.domains,
        landmarks=tables.landmarks,
        legacy_raw={},
    )
    source_manifest_path = args.source_dir / "manifest.json"
    source_manifest: Dict[str, Any] = {}
    if source_manifest_path.exists():
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_manifest = {}
    manifest = {
        "profile": args.profile,
        "created_at": utc_now(),
        "corpus_dir": str(args.out_dir),
        "source_kind": "derived_strict_from_existing_corpus",
        "source_corpus_dir": str(args.source_dir),
        "source_profile": source_manifest.get("profile", ""),
        "anchor_policy": "strict",
        "strict_noisy_landmark_ratio": float(args.noisy_landmark_ratio),
        "complete_end_year": DEFAULT_COMPLETE_END_YEAR,
        "works_rows": int(len(strict_tables.works)),
        "citation_rows": int(len(strict_tables.citations)),
        "topic_rows": int(len(strict_tables.topics)),
        "topic_edge_rows": int(len(strict_tables.topic_edges)),
        "notes": [
            "Derived from v1_large without deleting or modifying the source corpus.",
            "legacy_is_landmark preserves the original raw flag; is_landmark is recomputed by strict anchor policy.",
            "2026 is partial and excluded from default experiment views.",
        ],
    }
    write_corpus(args.out_dir, strict_tables, manifest)
    audit_corpus(args.out_dir, min_papers_per_domain=args.min_papers_per_domain)
    make_views(
        args.out_dir,
        include_partial_2026=args.include_partial_2026,
        anchor_policy="strict",
    )
    strict_report = audit_strict_views(args.out_dir)
    progress_log(
        f"Derived strict corpus at {args.out_dir}; strict_view_overall_pass={strict_report.get('overall_pass')}",
        args.quiet,
    )


def materialize_fig1_command(args: argparse.Namespace) -> None:
    materialize_fig1_cache(args.corpus_dir, args.out_dir, args.domains, anchor_policy=args.anchor_policy)
    progress_log(f"Materialized Fig. 1 cache under {args.out_dir}", args.quiet)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and manage the ASPR unified paper corpus.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build the canonical corpus and experiment views.")
    build.add_argument("--profile", default="v1_large")
    build.add_argument("--out-dir", type=Path, default=DEFAULT_BUILD_CORPUS_DIR)
    build.add_argument("--fig1-root", type=Path, default=DEFAULT_FIG1_ROOT)
    build.add_argument("--fig3-auto-root", type=Path, default=DEFAULT_FIG3_AUTO_ROOT)
    build.add_argument("--max-domains", type=int, default=48)
    build.add_argument("--papers-per-domain", type=int, default=4000)
    build.add_argument("--start-year", type=int, default=1980)
    build.add_argument("--end-year", type=int, default=DEFAULT_COMPLETE_END_YEAR)
    build.add_argument("--min-papers-per-domain", type=int, default=2500)
    build.add_argument("--offline", action="store_true", help="Use local Fig. 1/Fig. 3 caches only.")
    build.add_argument("--use-semantic-scholar", action="store_true", help="Reserved for optional S2 metadata enhancement.")
    build.add_argument("--include-partial-2026", action="store_true")
    build.add_argument("--anchor-policy", choices=["legacy", "strict"], default="legacy")
    build.add_argument("--max-anchor-citers", type=int, default=500)
    build.add_argument("--max-candidates-per-landmark", type=int, default=25)
    build.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional directory for per-domain online fetch checkpoints.")
    build.add_argument("--work-types", nargs="+", default=["article", "preprint", "review", "book-chapter", "book"])
    build.add_argument("--sleep-seconds", type=float, default=0.1)
    build.add_argument("--timeout-seconds", type=int, default=60)
    build.add_argument("--max-retries", type=int, default=5)
    build.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"))
    build.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"), help="Comma/space separated OpenAlex API keys used in round-robin order.")
    build.add_argument("--openalex-email", default=getenv("OPENALEX_EMAIL"))
    build.add_argument("--s2-api-key", default=getenv("S2_API_KEY"))
    build.add_argument("--quiet", action="store_true")
    build.set_defaults(func=build_command)

    audit = sub.add_parser("audit", help="Run quality gates for a corpus.")
    audit.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    audit.add_argument("--min-papers-per-domain", type=int, default=2500)
    audit.add_argument("--quiet", action="store_true")
    audit.set_defaults(func=audit_command)

    views = sub.add_parser("make-views", help="Materialize Fig1/Fig2/Fig3/Fig5 views.")
    views.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    views.add_argument("--include-partial-2026", action="store_true")
    views.add_argument("--anchor-policy", choices=["legacy", "strict"], default="legacy")
    views.add_argument("--quiet", action="store_true")
    views.set_defaults(func=make_views_command)

    strict = sub.add_parser("derive-strict", help="Derive a strict-anchor corpus from an existing corpus.")
    strict.add_argument("--source-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    strict.add_argument("--out-dir", type=Path, default=DEFAULT_STRICT_CORPUS_DIR)
    strict.add_argument("--profile", default="v1_strict")
    strict.add_argument("--include-partial-2026", action="store_true")
    strict.add_argument("--min-papers-per-domain", type=int, default=2500)
    strict.add_argument("--noisy-landmark-ratio", type=float, default=0.25)
    strict.add_argument("--quiet", action="store_true")
    strict.set_defaults(func=derive_strict_command)

    fig1 = sub.add_parser("materialize-fig1-cache", help="Write Fig. 1-compatible cache/export files from the corpus.")
    fig1.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    fig1.add_argument("--out-dir", type=Path, required=True)
    fig1.add_argument("--domains", nargs="+", required=True)
    fig1.add_argument("--anchor-policy", choices=["legacy", "strict"], default="legacy")
    fig1.add_argument("--quiet", action="store_true")
    fig1.set_defaults(func=materialize_fig1_command)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

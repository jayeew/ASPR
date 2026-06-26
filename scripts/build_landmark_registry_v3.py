from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from publication_corpus_v2 import fetch_openalex_work, normalize_doi, nonempty, slugify, utc_now, write_json
except ModuleNotFoundError:  # pragma: no cover
    from scripts.publication_corpus_v2 import fetch_openalex_work, normalize_doi, nonempty, slugify, utc_now, write_json

import pandas as pd


DEFAULT_TARGET_ROSTER_PATH = Path("outputs/publication_corpus_v5_fig3aware_candidate_audit/publication_target_domains.json")
DEFAULT_PERFORMANCE_CANDIDATES = ["magnetic_properties_of_thin_films"]

LANDMARK_V3_SEEDS: List[Dict[str, Any]] = [
    {
        "domain": "crispr",
        "label": "Jinek 2012",
        "doi": "10.1126/science.1225829",
        "title": "A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity",
        "year": 2012,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Canonical biochemical CRISPR-Cas9 programmability paper.",
    },
    {
        "domain": "crispr",
        "label": "Cong 2013",
        "doi": "10.1126/science.1231143",
        "title": "Multiplex genome engineering using CRISPR/Cas systems",
        "year": 2013,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Early mammalian genome engineering landmark.",
    },
    {
        "domain": "crispr",
        "label": "Mali 2013",
        "doi": "10.1126/science.1232033",
        "title": "RNA-guided human genome engineering via Cas9",
        "year": 2013,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Independent early human genome engineering landmark.",
    },
    {
        "domain": "ipsc_reprogramming",
        "label": "Takahashi/Yamanaka 2006",
        "doi": "10.1016/j.cell.2006.07.024",
        "title": "Induction of pluripotent stem cells from mouse embryonic and adult fibroblast cultures by defined factors",
        "year": 2006,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Core induced pluripotency discovery.",
    },
    {
        "domain": "ipsc_reprogramming",
        "label": "Takahashi 2007",
        "doi": "10.1016/j.cell.2007.11.019",
        "title": "Induction of pluripotent stem cells from adult human fibroblasts by defined factors",
        "year": 2007,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Human iPSC extension of the Yamanaka discovery.",
    },
    {
        "domain": "ipsc_reprogramming",
        "label": "Yu 2007",
        "doi": "10.1126/science.1151526",
        "title": "Induced pluripotent stem cell lines derived from human somatic cells",
        "year": 2007,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Independent human iPSC landmark.",
    },
    {
        "domain": "graphene_2d_materials",
        "label": "Novoselov 2004",
        "doi": "10.1126/science.1102896",
        "title": "Electric field effect in atomically thin carbon films",
        "year": 2004,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Canonical isolation and electronic graphene landmark.",
    },
    {
        "domain": "perovskite_solar_cells",
        "label": "Kojima 2009",
        "doi": "10.1021/ja809598r",
        "title": "Organometal halide perovskites as visible-light sensitizers for photovoltaic cells",
        "year": 2009,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Early photovoltaic perovskite sensitizer landmark.",
    },
    {
        "domain": "perovskite_solar_cells",
        "label": "Lee 2012",
        "doi": "10.1126/science.1228604",
        "title": "Efficient hybrid solar cells based on meso-superstructured organometal halide perovskites",
        "year": 2012,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Solid-state perovskite solar-cell efficiency landmark.",
    },
    {
        "domain": "perovskite_solar_cells",
        "label": "Burschka 2013",
        "doi": "10.1038/nature12340",
        "title": "Sequential deposition as a route to high-performance perovskite-sensitized solar cells",
        "year": 2013,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "High-performance deposition/process landmark.",
    },
    {
        "domain": "exoplanets",
        "label": "Mayor/Queloz 1995",
        "doi": "10.1038/378355a0",
        "title": "A Jupiter-mass companion to a solar-type star",
        "year": 1995,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "First widely recognized exoplanet around a Sun-like star.",
    },
    {
        "domain": "genetics_aging_and_longevity_in_model_organisms",
        "label": "Kenyon 1993",
        "doi": "10.1038/366461a0",
        "title": "A C. elegans mutant that lives twice as long as wild type",
        "year": 1993,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Canonical daf-2 longevity landmark.",
    },
    {
        "domain": "genetics_aging_and_longevity_in_model_organisms",
        "label": "Kimura 1997",
        "doi": "10.1126/science.277.5328.942",
        "title": "daf-2, an insulin receptor-like gene that regulates longevity and diapause in Caenorhabditis elegans",
        "year": 1997,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Insulin-like signaling longevity mechanism landmark.",
    },
    {
        "domain": "microbiome_metagenomics",
        "label": "Qin 2010",
        "doi": "10.1038/nature08821",
        "title": "A human gut microbial gene catalogue established by metagenomic sequencing",
        "year": 2010,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Metagenomic gut microbiome catalogue landmark.",
    },
    {
        "domain": "microbiome_metagenomics",
        "label": "HMP 2012",
        "doi": "10.1038/nature11234",
        "title": "Structure, function and diversity of the healthy human microbiome",
        "year": 2012,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Human Microbiome Project main survey landmark.",
    },
    {
        "domain": "topological_insulators",
        "label": "Kane/Mele 2005",
        "doi": "10.1103/physrevlett.95.226801",
        "title": "Quantum spin Hall effect in graphene",
        "year": 2005,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "Theoretical quantum spin Hall/topological-insulator landmark.",
    },
    {
        "domain": "topological_insulators",
        "label": "Bernevig 2006",
        "doi": "10.1126/science.1133734",
        "title": "Quantum spin Hall effect and topological phase transition in HgTe quantum wells",
        "year": 2006,
        "authority_basis": "field_consensus_manual_doi",
        "authority_note": "HgTe quantum-well topological phase prediction landmark.",
    },
    {
        "domain": "mass_spectrometry_techniques_and_applications",
        "label": "Karas/Hillenkamp 1988",
        "doi": "10.1021/ac00171a028",
        "title": "Laser desorption ionization of proteins with molecular masses exceeding 10000 daltons",
        "year": 1988,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "MALDI biomolecule mass-spectrometry landmark.",
    },
    {
        "domain": "mass_spectrometry_techniques_and_applications",
        "label": "Fenn 1989",
        "doi": "10.1126/science.2675315",
        "title": "Electrospray ionization for mass spectrometry of large biomolecules",
        "year": 1989,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Electrospray ionization biomolecule mass-spectrometry landmark.",
    },
    {
        "domain": "ubiquitin_and_proteasome_pathways",
        "label": "Ciechanover 1980",
        "doi": "10.1073/pnas.77.3.1365",
        "title": "ATP-dependent conjugation of reticulocyte proteins with the polypeptide required for protein degradation",
        "year": 1980,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Ubiquitin-mediated protein degradation discovery landmark.",
    },
    {
        "domain": "ubiquitin_and_proteasome_pathways",
        "label": "Hershko/Ciechanover 1998",
        "doi": "10.1146/annurev.biochem.67.1.425",
        "title": "The ubiquitin system",
        "year": 1998,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Authoritative synthesis used as a Nobel-related pathway landmark.",
    },
    {
        "domain": "gamma_ray_bursts_and_supernovae",
        "label": "Riess 1998",
        "doi": "10.1086/300499",
        "title": "Observational evidence from supernovae for an accelerating universe and a cosmological constant",
        "year": 1998,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "High-z supernova acceleration landmark.",
    },
    {
        "domain": "gamma_ray_bursts_and_supernovae",
        "label": "Perlmutter 1999",
        "doi": "10.1086/307221",
        "title": "Measurements of Omega and Lambda from 42 high-redshift supernovae",
        "year": 1999,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Supernova Cosmology Project acceleration landmark.",
    },
    {
        "domain": "magnetic_properties_of_thin_films",
        "label": "Baibich 1988",
        "doi": "10.1103/physrevlett.61.2472",
        "title": "Giant magnetoresistance of (001)Fe/(001)Cr magnetic superlattices",
        "year": 1988,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Giant magnetoresistance discovery landmark.",
    },
    {
        "domain": "magnetic_properties_of_thin_films",
        "label": "Binasch 1989",
        "doi": "10.1103/physrevb.39.4828",
        "title": "Enhanced magnetoresistance in layered magnetic structures with antiferromagnetic interlayer exchange",
        "year": 1989,
        "authority_basis": "nobel_and_field_consensus",
        "authority_note": "Independent GMR layered-structure landmark.",
    },
]


def title_similarity(left: object, right: object) -> float:
    return float(SequenceMatcher(None, nonempty(left).lower(), nonempty(right).lower()).ratio())


def validate_seed(row: Mapping[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    doi = normalize_doi(row.get("doi"))
    work = fetch_openalex_work(doi, timeout_seconds=timeout_seconds) if doi else None
    out = dict(row)
    out["domain"] = slugify(out.get("domain"))
    out["doi"] = doi
    out["doi_url"] = f"https://doi.org/{doi}" if doi else ""
    out["landmark_source"] = "strict_manual_v3"
    out["accepted_landmark_source"] = "strict_manual_v3"
    out["needs_manual_confirmation"] = 0
    out["include_main"] = 1
    out["openalex_id"] = nonempty((work or {}).get("id"))
    out["openalex_title"] = nonempty((work or {}).get("display_name"))
    out["openalex_year"] = int((work or {}).get("publication_year") or 0) if work else 0
    out["openalex_cited_by_count"] = int((work or {}).get("cited_by_count") or 0) if work else 0
    out["openalex_reference_count"] = len((work or {}).get("referenced_works") or []) if work else 0
    out["title_similarity"] = title_similarity(row.get("title"), out["openalex_title"]) if work else 0.0
    out["year_matches"] = int(out["openalex_year"] == int(row.get("year", 0))) if work else 0
    out["validation_status"] = "passed" if work and out["title_similarity"] >= 0.72 and out["year_matches"] else "failed"
    out["failure_reason"] = "" if out["validation_status"] == "passed" else "doi_not_found_or_metadata_mismatch"
    out["evidence_key"] = doi
    return out


def build_registry(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    passed = frame[frame["validation_status"].astype(str) == "passed"].copy()
    passed = passed.sort_values(["domain", "year", "label"]).groupby("domain", group_keys=False).head(3)
    return passed.reset_index(drop=True)


def load_roster_rows(path: Path) -> List[Dict[str, Any]]:
    """Load candidate/main domain rows from a publication target roster."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("domains", []) if isinstance(payload, Mapping) else payload
    return [dict(row) for row in rows]


def build_domain_coverage(
    registry: pd.DataFrame,
    roster_rows: Sequence[Mapping[str, Any]],
    candidate_domains: Sequence[str],
) -> pd.DataFrame:
    """Audit which candidate/main domains have strict v3 landmark coverage."""
    counts = (
        registry.groupby("domain", as_index=False)
        .agg(
            n_v3_landmarks=("doi", "count"),
            landmark_labels=("label", lambda values: "; ".join(str(v) for v in values)),
            landmark_dois=("doi", lambda values: "; ".join(str(v) for v in values)),
        )
        if not registry.empty
        else pd.DataFrame(columns=["domain", "n_v3_landmarks", "landmark_labels", "landmark_dois"])
    )
    count_lookup = {str(row["domain"]): row for row in counts.to_dict("records")}

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in roster_rows:
        domain = slugify(row.get("domain_id") or row.get("domain") or row.get("slug"))
        if not domain:
            continue
        seen.add(domain)
        found = count_lookup.get(domain, {})
        n_landmarks = int(found.get("n_v3_landmarks") or 0)
        roster_status = nonempty(row.get("status")) or "unknown"
        if n_landmarks == 0:
            v3_status = "blocked_missing_v3_landmark"
        elif n_landmarks > 3:
            v3_status = "blocked_too_many_v3_landmarks"
        elif roster_status == "main_ready":
            v3_status = "main_v3_covered"
        else:
            v3_status = "candidate_v3_covered"
        rows.append(
            {
                "domain": domain,
                "roster_status": roster_status,
                "family": nonempty(row.get("family")),
                "event_year": row.get("event_year", ""),
                "analysis_end_year": row.get("analysis_end_year", ""),
                "n_v3_landmarks": n_landmarks,
                "landmark_labels": found.get("landmark_labels", ""),
                "landmark_dois": found.get("landmark_dois", ""),
                "v3_status": v3_status,
                "eligible_for_v3_graph": int(1 <= n_landmarks <= 3),
                "eligible_for_main_roster": int(roster_status == "main_ready" and 1 <= n_landmarks <= 3),
            }
        )

    for domain in [slugify(item) for item in candidate_domains]:
        if not domain or domain in seen:
            continue
        seen.add(domain)
        found = count_lookup.get(domain, {})
        n_landmarks = int(found.get("n_v3_landmarks") or 0)
        rows.append(
            {
                "domain": domain,
                "roster_status": "performance_gated_candidate",
                "family": "",
                "event_year": "",
                "analysis_end_year": "",
                "n_v3_landmarks": n_landmarks,
                "landmark_labels": found.get("landmark_labels", ""),
                "landmark_dois": found.get("landmark_dois", ""),
                "v3_status": "candidate_v3_covered" if 1 <= n_landmarks <= 3 else "blocked_missing_v3_landmark",
                "eligible_for_v3_graph": int(1 <= n_landmarks <= 3),
                "eligible_for_main_roster": 0,
            }
        )

    for domain, found in sorted(count_lookup.items()):
        if domain in seen:
            continue
        rows.append(
            {
                "domain": domain,
                "roster_status": "registry_only_candidate",
                "family": "",
                "event_year": "",
                "analysis_end_year": "",
                "n_v3_landmarks": int(found.get("n_v3_landmarks") or 0),
                "landmark_labels": found.get("landmark_labels", ""),
                "landmark_dois": found.get("landmark_dois", ""),
                "v3_status": "v3_covered_not_current_main",
                "eligible_for_v3_graph": 1,
                "eligible_for_main_roster": 0,
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict manually curated landmark registry v3.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/landmark_registry_v3"))
    parser.add_argument("--registry-csv", type=Path, default=Path("data/knowledge_corpus/landmark_registry_v3.csv"))
    parser.add_argument("--registry-json", type=Path, default=Path("data/knowledge_corpus/landmark_registry_v3.json"))
    parser.add_argument("--target-roster-path", type=Path, default=DEFAULT_TARGET_ROSTER_PATH)
    parser.add_argument("--coverage-csv", type=Path, default=Path("outputs/landmark_registry_v3/landmark_registry_v3_domain_coverage.csv"))
    parser.add_argument("--coverage-json", type=Path, default=Path("outputs/landmark_registry_v3/landmark_registry_v3_domain_coverage.json"))
    parser.add_argument("--candidate-domain", action="append", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [validate_seed(row, timeout_seconds=args.timeout_seconds) for row in LANDMARK_V3_SEEDS]
    validation = pd.DataFrame(rows)
    validation_path = args.out_dir / "landmark_registry_v3_validation.csv"
    validation.to_csv(validation_path, index=False)

    registry = build_registry(rows)
    args.registry_csv.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.registry_csv, index=False)
    args.registry_json.write_text(
        json.dumps(registry.to_dict("records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    candidate_domains = list(args.candidate_domain or DEFAULT_PERFORMANCE_CANDIDATES)
    roster_rows = load_roster_rows(args.target_roster_path)
    coverage = build_domain_coverage(registry, roster_rows, candidate_domains)
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)
    args.coverage_json.write_text(
        json.dumps(coverage.to_dict("records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    blocked_main = (
        coverage[
            (coverage["roster_status"].astype(str) == "main_ready")
            & (coverage["eligible_for_main_roster"].astype(int) == 0)
        ]["domain"]
        .astype(str)
        .tolist()
        if not coverage.empty
        else []
    )

    manifest = {
        "artifact_kind": "landmark_registry_v3_manifest",
        "created_at": utc_now(),
        "policy": "strict_manual_or_authority_backed_doi_only_no_legacy_fig1_anchor",
        "max_landmarks_per_domain": 3,
        "n_seed_rows": int(len(validation)),
        "n_passed_rows": int(len(registry)),
        "n_domains": int(registry["domain"].nunique()) if not registry.empty else 0,
        "registry_csv": str(args.registry_csv),
        "registry_json": str(args.registry_json),
        "validation_csv": str(validation_path),
        "target_roster_path": str(args.target_roster_path),
        "coverage_csv": str(args.coverage_csv),
        "coverage_json": str(args.coverage_json),
        "performance_gated_candidates": candidate_domains,
        "blocked_main_domains_without_v3_landmarks": blocked_main,
        "failed_rows": validation[validation["validation_status"].astype(str) != "passed"][
            ["domain", "label", "doi", "failure_reason", "openalex_title", "openalex_year", "title_similarity"]
        ].to_dict("records"),
    }
    write_json(args.out_dir / "landmark_registry_v3_manifest.json", manifest)
    print(
        f"[landmark-v3] wrote {len(registry)} strict landmarks across "
        f"{registry['domain'].nunique() if not registry.empty else 0} domains",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

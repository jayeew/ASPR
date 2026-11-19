from __future__ import annotations

import argparse
import json
import sqlite3
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable

import build_saturation_alignment_protocols
import coding
import data_correspondence
import export_saturation_codebook_reference
import handoff
import indicators
import local_ai
import providers
import reporting
import retrieval
import saturation
import screening
import validate_saturation_alignment
import validate_press_review
import validate_press_revisions
from common import DATABASE_PATH, OUTPUT_DIR
from database import (
    initialize,
    register_human_review_attestation,
    register_independent_ai_review_manifest,
    supersede_source_snapshot,
    supersede_independent_ai_review_run,
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _path(value: str | None, default_name: str) -> Path:
    return Path(value).resolve() if value else OUTPUT_DIR / default_name


def _formal_record_keys(
    connection: sqlite3.Connection,
) -> Iterable[str]:
    for row in connection.execute(
        """
        SELECT record_key FROM formal_review_records ORDER BY record_key
        """
    ):
        yield str(row[0])


def _unvalidated_record_keys(
    connection: sqlite3.Connection,
    keys: Iterable[str],
) -> Iterable[str]:
    for key in keys:
        existing = connection.execute(
            """
            SELECT 1 FROM crossref_validation
            WHERE record_key = ?
              AND status IN (
                  'validated', 'validated_date_variant',
                  'resolved', 'conflict'
              )
            """,
            (key,),
        ).fetchone()
        if existing is None:
            yield key


def command_init(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return coding.initialize_project(connection)


def command_bootstrap(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    inventory = coding.mark_bootstrap_inventory(connection)
    if args.inventory_only:
        return {"inventory": inventory}
    results = []
    for query_id in coding.bootstrap_query_ids(connection):
        results.append(
            providers.retrieve_physical_query(
                connection,
                query_id,
                "bootstrap",
                max_pages=args.max_pages_per_query,
            )
        )
    return {
        "inventory": inventory,
        "retrieval": coding.mark_bootstrap_retrieval_stage(connection),
        "query_runs": results,
    }


def command_derive_discovery_frame(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return saturation.derive_discovery_queries(connection)


def command_hydrate_development_seeds(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return saturation.hydrate_development_seeds(connection)


def command_expand_development_citations(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return saturation.expand_development_seed_citations(connection)


def command_bootstrap_saturation(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    frame = saturation.derive_discovery_queries(connection)
    retrieval_result = saturation.retrieve_discovery_samples(
        connection,
        maximum_queries=args.maximum_queries,
    )
    return {"frame": frame, "retrieval": retrieval_result}


def command_assign_discovery_round(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return saturation.assign_discovery_round(connection, args.iteration)


def command_export_saturation_codebook_reference(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    term_output = _path(
        args.term_output,
        (
            "codebook_references/"
            f"rounds_01_{args.through_round:02d}_h2_term_codebook.csv"
        ),
    )
    indicator_output = _path(
        args.indicator_output,
        (
            "codebook_references/"
            f"rounds_01_{args.through_round:02d}_h2_indicator_codebook.csv"
        ),
    )
    manifest = _path(
        args.manifest,
        (
            "codebook_references/"
            f"rounds_01_{args.through_round:02d}_h2_codebooks.manifest.json"
        ),
    )
    result = export_saturation_codebook_reference.export_codebook_reference(
        connection,
        args.through_round,
        term_output,
        indicator_output,
        manifest,
    )
    prefix = f"rounds_01_{args.through_round:02d}_h2"
    for source_id, path, role in (
        (
            f"{prefix}_term_codebook",
            term_output,
            "prior_round_codebook_reference",
        ),
        (
            f"{prefix}_indicator_codebook",
            indicator_output,
            "prior_round_codebook_reference",
        ),
        (
            f"{prefix}_codebook_manifest",
            manifest,
            "prior_round_codebook_manifest",
        ),
    ):
        coding._register_snapshot(connection, source_id, path, role)
    connection.commit()
    return result


def command_build_saturation_alignment_protocols(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else OUTPUT_DIR / "alignment_protocols"
    )
    result = build_saturation_alignment_protocols.build_alignment_protocols(
        args.current_round,
        Path(args.codebook_manifest).resolve(),
        output_dir,
    )
    coding._register_snapshot(
        connection,
        f"round_{args.current_round:02d}_indicator_alignment_protocol_v3",
        Path(result["indicator_protocol"]),
        "independent_review_protocol",
    )
    coding._register_snapshot(
        connection,
        f"round_{args.current_round:02d}_term_alignment_protocol_v3",
        Path(result["term_protocol"]),
        "independent_review_protocol",
    )
    connection.commit()
    return result


def command_validate_saturation_alignment(
    _: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return validate_saturation_alignment.validate_alignment(
        args.kind,
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.protocol).resolve(),
        Path(args.manifest).resolve(),
    )


def command_validate_press_review(
    _: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return validate_press_review.validate_press_review(
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.protocol).resolve(),
        Path(args.manifest).resolve(),
    )


def command_validate_press_revisions(
    _: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return validate_press_revisions.validate_press_revisions(
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.protocol).resolve(),
        Path(args.manifest).resolve(),
    )


def command_export_discovery_screening(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(
        args.output,
        (
            f"discovery_round_{args.iteration}_screening_"
            f"{args.reviewer.casefold()}_v3.csv"
        ),
    )
    return {
        "rows": saturation.export_discovery_screening(
            connection,
            args.iteration,
            args.reviewer,
            path,
        ),
        "output": str(path),
    }


def command_finalize_discovery_screening(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return saturation.finalize_discovery_screening(
        connection,
        args.iteration,
    )


def command_export_discovery_extraction(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(
        args.output,
        f"discovery_round_{args.iteration}_extraction_v3.csv",
    )
    return {
        "rows": saturation.export_discovery_extraction(
            connection,
            args.iteration,
            path,
            extractor_role=args.extractor,
        ),
        "output": str(path),
    }


def command_import_discovery_extraction(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return saturation.import_discovery_extraction(
        connection,
        Path(args.input).resolve(),
    )


def command_export_discovery_indicator_adjudication(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(
        args.output,
        (
            f"discovery_round_{args.iteration}_indicator_"
            "adjudication_h2_v3.csv"
        ),
    )
    return {
        "rows": saturation.export_discovery_indicator_adjudication(
            connection,
            args.iteration,
            path,
        ),
        "output": str(path),
    }


def command_import_discovery_indicator_adjudication(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": saturation.import_discovery_indicator_adjudication(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_discovery_novelty_status(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return saturation.discovery_novelty_counts(
        connection,
        args.iteration,
    )


def command_record_discovery_saturation(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return saturation.record_discovery_saturation(
        connection,
        iteration=args.iteration,
        new_terms=args.new_terms,
        new_indicator_families=args.new_indicator_families,
        decision=args.decision,
        notes=args.notes,
        protocol_deviation_amendment=(
            Path(args.protocol_deviation_amendment)
            if args.protocol_deviation_amendment
            else None
        ),
    )


def command_discovery_status(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return saturation.discovery_status(connection)


def command_prepare_human_tasks(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return handoff.prepare_human_tasks(connection, force=args.force)


def command_register_human_review_attestation(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return register_human_review_attestation(
        connection,
        Path(args.input).resolve(),
    )


def command_register_independent_ai_review(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return register_independent_ai_review_manifest(
        connection,
        Path(args.input).resolve(),
    )


def command_supersede_independent_ai_review(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return supersede_independent_ai_review_run(
        connection,
        args.old_run_id,
        args.new_run_id,
        args.reason,
        allow_superset=args.allow_superset,
    )


def command_supersede_source_snapshot(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return supersede_source_snapshot(
        connection,
        args.old_source_id,
        args.new_source_id,
        Path(args.current_path).resolve(),
        Path(args.authorization).resolve(),
        args.reason,
    )


def command_ai_screen_discovery(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(
        args.output,
        f"discovery_round_{args.iteration}_screening_ai_completed_v3.csv",
    )
    return local_ai.ai_screen_discovery_round(
        connection,
        iteration=args.iteration,
        model=args.model,
        output_path=path,
        force=args.force,
    )


def command_normalize_ai_language_evidence(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else OUTPUT_DIR
    )
    return local_ai.normalize_ai_language_evidence(
        connection,
        output_dir,
    )


def command_ai_code_terms(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    output_path = _path(
        args.output,
        "term_coding_ai_completed_v3.csv",
    )
    codebook_path = _path(
        args.codebook_output,
        "term_codebook_ai_v3.json",
    )
    return local_ai.ai_code_terms(
        connection,
        model=args.model,
        output_path=output_path,
        codebook_path=codebook_path,
        force=args.force,
    )


def command_ai_code_dimensions(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    output_path = _path(
        args.output,
        "dimension_coding_ai_completed_v3.csv",
    )
    return local_ai.ai_code_dimensions(
        connection,
        model=args.model,
        output_path=output_path,
        force=args.force,
    )


def command_export_term_extraction(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "term_extraction_template_v3.csv")
    return {
        "rows": coding.export_term_extraction(connection, path),
        "output": str(path),
    }


def command_import_terms(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return coding.import_terms(connection, Path(args.input).resolve())


def command_export_term_coding(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    role = args.reviewer
    suffix = role.casefold() if role else "all_reviewers"
    path = _path(args.output, f"term_coding_{suffix}_v3.csv")
    return {
        "rows": coding.export_term_coding(
            connection,
            path,
            role,
            only_missing=args.only_missing,
        ),
        "output": str(path),
    }


def command_import_term_coding(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": coding.import_term_coding(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_code_terms(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    if args.input:
        return command_import_term_coding(connection, args)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    results: Dict[str, Any] = {}
    for role in ("AI", "H1"):
        path = output_dir / f"term_coding_{role.casefold()}_v3.csv"
        results[role] = {
            "rows": coding.export_term_coding(connection, path, role),
            "output": str(path),
        }
    return results


def command_derive_search_frame(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return coding.derive_search_frame(connection)


def command_export_press(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "press_review_template_v3.csv")
    return {
        "rows": coding.export_press(
            connection,
            path,
            only_pending=args.only_pending,
        ),
        "output": str(path),
    }


def command_import_press(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return coding.import_press(connection, Path(args.input).resolve())


def command_resolve_press_redundancy(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return coding.resolve_press_redundancy(connection)


def command_export_press_revisions(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "press_query_revisions_H2_v3.csv")
    return {
        "rows": coding.export_press_revisions(connection, path),
        "output": str(path),
    }


def command_import_press_revisions(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return coding.import_press_revisions(
        connection,
        Path(args.input).resolve(),
    )


def command_apply_press_revisions(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return coding.apply_press_revisions(connection)


def command_export_seed_template(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    del connection
    path = _path(args.output, "hidden_validation_seed_template_v3.csv")
    coding.export_seed_template(path)
    return {"output": str(path)}


def command_import_seeds(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": coding.import_hidden_seeds(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_export_hidden_seed_search_log(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    del connection
    path = _path(
        args.output,
        "hidden_validation_seed_search_log_H2.csv",
    )
    return {
        "rows": coding.export_hidden_seed_search_log_template(path),
        "output": str(path),
    }


def command_import_hidden_seed_search_log(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return coding.import_hidden_seed_search_log(
        connection,
        Path(args.input).resolve(),
    )


def command_validate_search_frame(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return coding.validate_search_frame(connection)


def command_export_seed_supplements(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "seed_supplement_template_v3.csv")
    return {
        "rows": coding.export_seed_supplement_template(connection, path),
        "output": str(path),
    }


def command_import_seed_supplements(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": coding.import_seed_supplements(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_freeze_search_frame(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return coding.freeze_search_frame(connection)


def command_reopen_search_frame(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return coding.reopen_search_frame(connection, notes=args.notes)


def command_retrieve(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return retrieval.retrieve_formal_queries(
        connection,
        max_pages_per_query=args.max_pages_per_query,
    )


def command_export_screening(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    role = args.reviewer
    suffix = role.casefold() if role else "all_reviewers"
    path = _path(args.output, f"literature_screening_{suffix}_v3.csv")
    return {
        "rows": screening.export_screening(connection, path, role),
        "output": str(path),
    }


def command_import_screening(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": screening.import_screening(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_finalize_screening(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return screening.finalize_screening(connection)


def command_screen_literature(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    if args.input:
        screening.import_screening(
            connection,
            Path(args.input).resolve(),
        )
        return screening.finalize_screening(connection)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    results: Dict[str, Any] = {}
    for role in ("AI", "H1"):
        path = output_dir / f"literature_screening_{role.casefold()}_v3.csv"
        results[role] = {
            "rows": screening.export_screening(connection, path, role),
            "output": str(path),
        }
    return results


def command_crossref_validate(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    scope_keys = list(
        screening.included_record_keys(connection)
        if args.scope == "included"
        else _formal_record_keys(connection)
    )
    not_found_reclassified = providers.reclassify_crossref_not_found(
        connection,
        scope_keys,
    )
    keys = _unvalidated_record_keys(connection, scope_keys)
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be at least one")
        keys = islice(keys, args.max_records)
    scoped_keys = list(keys)
    result = providers.crossref_validate_scope(
        connection,
        scoped_keys,
        worker_count=args.workers,
    )
    result["not_found_reclassified"] = not_found_reclassified
    result["date_variants_reclassified"] = (
        providers.reclassify_crossref_date_variants(
            connection,
            record_keys=scope_keys,
        )
    )
    return result


def command_export_crossref_conflicts(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "crossref_conflict_queue_v3.csv")
    return {
        "rows": providers.export_crossref_conflicts(
            connection,
            path,
            record_keys=_formal_record_keys(connection),
        ),
        "output": str(path),
    }


def command_import_crossref_resolutions(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": providers.import_crossref_resolutions(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_export_indicator_extraction(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "indicator_extraction_template_v3.csv")
    return {
        "rows": indicators.export_indicator_extraction(connection, path),
        "output": str(path),
    }


def command_export_indicator_adjudication(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "indicator_adjudication_H2_v3.csv")
    return {
        "rows": indicators.export_indicator_adjudication(connection, path),
        "output": str(path),
    }


def command_acquire_open_fulltexts(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else OUTPUT_DIR / "open_fulltexts"
    )
    return indicators.acquire_open_fulltexts(
        connection,
        output_dir,
        maximum_records=args.max_records,
        retry_failed=args.retry_failed,
        timeout_seconds=args.timeout_seconds,
        maximum_bytes=args.maximum_bytes,
        hydrate_locations=not args.skip_location_hydration,
    )


def command_import_indicators(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return indicators.import_indicators(
        connection,
        Path(args.input).resolve(),
    )


def command_extract_indicators(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    if args.input:
        return command_import_indicators(connection, args)
    path = _path(args.output, "indicator_extraction_template_v3.csv")
    return {
        "rows": indicators.export_indicator_extraction(connection, path),
        "output": str(path),
    }


def command_export_dimension_coding(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    role = args.reviewer
    suffix = role.casefold() if role else "all_reviewers"
    path = _path(args.output, f"dimension_coding_{suffix}_v3.csv")
    return {
        "rows": indicators.export_dimension_coding(connection, path, role),
        "output": str(path),
    }


def command_export_data_audit(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "feature_data_audit_template_v3.csv")
    return {
        "rows": indicators.export_feature_data_audit(connection, path),
        "output": str(path),
    }


def command_import_data_audit(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": indicators.import_feature_data_audit(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_build_local_input_inventory(
    _: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    path = _path(args.output, "local_t0_input_inventory_v3.json")
    return data_correspondence.build_local_t0_input_inventory(path)


def command_export_data_correspondence(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    role = str(args.reviewer).upper()
    path = _path(
        args.output,
        f"feature_data_correspondence_{role.casefold()}_v3.csv",
    )
    inventory_path = (
        Path(args.inventory).resolve()
        if args.inventory
        else data_correspondence.DEFAULT_INVENTORY_PATH
    )
    return {
        "rows": data_correspondence.export_data_correspondence(
            connection,
            path,
            role,
            inventory_path,
        ),
        "output": str(path),
    }


def command_import_data_correspondence(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return data_correspondence.import_data_correspondence(
        connection,
        Path(args.input).resolve(),
    )


def command_export_operationalization(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    role = str(args.reviewer).upper()
    path = _path(
        args.output,
        f"feature_operationalization_{role.casefold()}_v3.csv",
    )
    return {
        "rows": indicators.export_feature_operationalization(
            connection,
            path,
            role,
        ),
        "output": str(path),
    }


def command_import_operationalization(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return indicators.import_feature_operationalization(
        connection,
        Path(args.input).resolve(),
    )


def command_import_dimension_coding(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return {
        "imported": indicators.import_dimension_coding(
            connection,
            Path(args.input).resolve(),
        )
    }


def command_derive_dimensions(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    if args.input:
        indicators.import_dimension_coding(
            connection,
            Path(args.input).resolve(),
        )
        return indicators.derive_dimensions(connection)
    if args.export_templates:
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else OUTPUT_DIR
        )
        results: Dict[str, Any] = {}
        for role in ("AI", "H1", "H2"):
            path = output_dir / f"dimension_coding_{role.casefold()}_v3.csv"
            results[role] = {
                "rows": indicators.export_dimension_coding(
                    connection,
                    path,
                    role,
                ),
                "output": str(path),
            }
        return results
    return indicators.derive_dimensions(connection)


def command_select_indicators(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return indicators.select_indicators(connection)


def command_citation_track(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return retrieval.track_citations(
        connection,
        iteration=args.iteration,
        scope=args.scope,
    )


def command_record_saturation(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> Any:
    return retrieval.record_saturation_round(
        connection,
        iteration=args.iteration,
        new_records=args.new_records,
        new_terms=args.new_terms,
        new_indicator_families=args.new_indicator_families,
        decision=args.decision,
        notes=args.notes,
    )


def command_audit(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return reporting.audit(connection)


def command_status(
    connection: sqlite3.Connection,
    _: argparse.Namespace,
) -> Any:
    return {
        "counts": reporting.current_counts(connection),
        "stages": {
            str(row["stage"]): {
                "status": row["status"],
                "details": json.loads(row["details_json"]),
            }
            for row in connection.execute(
                "SELECT * FROM stage_status ORDER BY rowid"
            )
        },
    }


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone v3 command-line state machine."""
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-derived English literature, dimensions, and "
            "publication-time indicator selection v3"
        )
    )
    parser.add_argument(
        "--database",
        default=str(DATABASE_PATH),
        help="Independent v3 SQLite database path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init").set_defaults(handler=command_init)

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--inventory-only", action="store_true")
    bootstrap.add_argument("--max-pages-per-query", type=int)
    bootstrap.set_defaults(handler=command_bootstrap)

    subparsers.add_parser("derive-discovery-frame").set_defaults(
        handler=command_derive_discovery_frame
    )

    subparsers.add_parser("hydrate-development-seeds").set_defaults(
        handler=command_hydrate_development_seeds
    )

    subparsers.add_parser("expand-development-citations").set_defaults(
        handler=command_expand_development_citations
    )

    bootstrap_saturation = subparsers.add_parser("bootstrap-saturation")
    bootstrap_saturation.add_argument("--maximum-queries", type=int)
    bootstrap_saturation.set_defaults(handler=command_bootstrap_saturation)

    assign_discovery = subparsers.add_parser("assign-discovery-round")
    assign_discovery.add_argument("--iteration", type=int, required=True)
    assign_discovery.set_defaults(handler=command_assign_discovery_round)

    export_codebook = subparsers.add_parser(
        "export-saturation-codebook-reference"
    )
    export_codebook.add_argument(
        "--through-round",
        type=int,
        required=True,
    )
    export_codebook.add_argument("--term-output")
    export_codebook.add_argument("--indicator-output")
    export_codebook.add_argument("--manifest")
    export_codebook.set_defaults(
        handler=command_export_saturation_codebook_reference
    )

    build_alignment = subparsers.add_parser(
        "build-saturation-alignment-protocols"
    )
    build_alignment.add_argument(
        "--current-round",
        type=int,
        required=True,
    )
    build_alignment.add_argument("--codebook-manifest", required=True)
    build_alignment.add_argument("--output-dir")
    build_alignment.set_defaults(
        handler=command_build_saturation_alignment_protocols
    )

    validate_alignment = subparsers.add_parser(
        "validate-saturation-alignment"
    )
    validate_alignment.add_argument(
        "--kind",
        choices=("indicator", "term"),
        required=True,
    )
    validate_alignment.add_argument("--input", required=True)
    validate_alignment.add_argument("--output", required=True)
    validate_alignment.add_argument("--protocol", required=True)
    validate_alignment.add_argument("--manifest", required=True)
    validate_alignment.set_defaults(
        handler=command_validate_saturation_alignment
    )

    validate_press = subparsers.add_parser("validate-press-review")
    validate_press.add_argument("--input", required=True)
    validate_press.add_argument("--output", required=True)
    validate_press.add_argument("--protocol", required=True)
    validate_press.add_argument("--manifest", required=True)
    validate_press.set_defaults(handler=command_validate_press_review)

    validate_press_revisions_parser = subparsers.add_parser(
        "validate-press-revisions"
    )
    validate_press_revisions_parser.add_argument("--input", required=True)
    validate_press_revisions_parser.add_argument("--output", required=True)
    validate_press_revisions_parser.add_argument(
        "--protocol",
        required=True,
    )
    validate_press_revisions_parser.add_argument(
        "--manifest",
        required=True,
    )
    validate_press_revisions_parser.set_defaults(
        handler=command_validate_press_revisions
    )

    export_discovery_screen = subparsers.add_parser(
        "export-discovery-screening"
    )
    export_discovery_screen.add_argument(
        "--iteration",
        type=int,
        required=True,
    )
    export_discovery_screen.add_argument(
        "--reviewer",
        choices=("AI", "H1", "H2"),
        required=True,
    )
    _add_output_argument(export_discovery_screen)
    export_discovery_screen.set_defaults(
        handler=command_export_discovery_screening
    )

    finalize_discovery_screen = subparsers.add_parser(
        "finalize-discovery-screening"
    )
    finalize_discovery_screen.add_argument(
        "--iteration",
        type=int,
        required=True,
    )
    finalize_discovery_screen.set_defaults(
        handler=command_finalize_discovery_screening
    )

    export_discovery_extract = subparsers.add_parser(
        "export-discovery-extraction"
    )
    export_discovery_extract.add_argument(
        "--iteration",
        type=int,
        required=True,
    )
    export_discovery_extract.add_argument(
        "--extractor",
        choices=("AI", "H1"),
        default="H1",
    )
    _add_output_argument(export_discovery_extract)
    export_discovery_extract.set_defaults(
        handler=command_export_discovery_extraction
    )

    import_discovery_extract = subparsers.add_parser(
        "import-discovery-extraction"
    )
    import_discovery_extract.add_argument("--input", required=True)
    import_discovery_extract.set_defaults(
        handler=command_import_discovery_extraction
    )

    export_discovery_indicator_h2 = subparsers.add_parser(
        "export-discovery-indicator-adjudication"
    )
    export_discovery_indicator_h2.add_argument(
        "--iteration",
        type=int,
        required=True,
    )
    _add_output_argument(export_discovery_indicator_h2)
    export_discovery_indicator_h2.set_defaults(
        handler=command_export_discovery_indicator_adjudication
    )

    import_discovery_indicator_h2 = subparsers.add_parser(
        "import-discovery-indicator-adjudication"
    )
    import_discovery_indicator_h2.add_argument("--input", required=True)
    import_discovery_indicator_h2.set_defaults(
        handler=command_import_discovery_indicator_adjudication
    )

    discovery_novelty = subparsers.add_parser(
        "discovery-novelty-status"
    )
    discovery_novelty.add_argument("--iteration", type=int, required=True)
    discovery_novelty.set_defaults(
        handler=command_discovery_novelty_status
    )

    record_discovery = subparsers.add_parser(
        "record-discovery-saturation"
    )
    record_discovery.add_argument("--iteration", type=int, required=True)
    record_discovery.add_argument("--new-terms", type=int, required=True)
    record_discovery.add_argument(
        "--new-indicator-families",
        type=int,
        required=True,
    )
    record_discovery.add_argument(
        "--decision",
        choices=("continue", "freeze"),
        required=True,
    )
    record_discovery.add_argument("--notes", required=True)
    record_discovery.add_argument(
        "--protocol-deviation-amendment",
        help=(
            "Audited amendment authorizing a non-dual-zero freeze; actual "
            "novelty counts remain mandatory and are never overwritten"
        ),
    )
    record_discovery.set_defaults(
        handler=command_record_discovery_saturation
    )

    subparsers.add_parser("discovery-status").set_defaults(
        handler=command_discovery_status
    )

    prepare_human = subparsers.add_parser("prepare-human-tasks")
    prepare_human.add_argument(
        "--force",
        action="store_true",
        help="Overwrite blank worksheets; may destroy unimported edits",
    )
    prepare_human.set_defaults(handler=command_prepare_human_tasks)

    register_human = subparsers.add_parser(
        "register-human-review-attestation"
    )
    register_human.add_argument("--input", required=True)
    register_human.set_defaults(
        handler=command_register_human_review_attestation
    )

    register_independent_ai = subparsers.add_parser(
        "register-independent-ai-review"
    )
    register_independent_ai.add_argument("--input", required=True)
    register_independent_ai.set_defaults(
        handler=command_register_independent_ai_review
    )

    supersede_independent_ai = subparsers.add_parser(
        "supersede-independent-ai-review"
    )
    supersede_independent_ai.add_argument("--old-run-id", required=True)
    supersede_independent_ai.add_argument("--new-run-id", required=True)
    supersede_independent_ai.add_argument("--reason", required=True)
    supersede_independent_ai.add_argument(
        "--allow-superset",
        action="store_true",
        help=(
            "Allow a larger new artifact only after exact shared-key "
            "decision equivalence is verified."
        ),
    )
    supersede_independent_ai.set_defaults(
        handler=command_supersede_independent_ai_review
    )

    supersede_source = subparsers.add_parser(
        "supersede-source-snapshot"
    )
    supersede_source.add_argument("--old-source-id", required=True)
    supersede_source.add_argument("--new-source-id", required=True)
    supersede_source.add_argument("--current-path", required=True)
    supersede_source.add_argument("--authorization", required=True)
    supersede_source.add_argument("--reason", required=True)
    supersede_source.set_defaults(
        handler=command_supersede_source_snapshot
    )

    ai_discovery = subparsers.add_parser("ai-screen-discovery")
    ai_discovery.add_argument("--iteration", type=int, required=True)
    ai_discovery.add_argument("--model", default="qwen3:8b")
    ai_discovery.add_argument("--force", action="store_true")
    _add_output_argument(ai_discovery)
    ai_discovery.set_defaults(handler=command_ai_screen_discovery)

    normalize_ai_evidence = subparsers.add_parser(
        "normalize-ai-language-evidence"
    )
    normalize_ai_evidence.add_argument("--output-dir")
    normalize_ai_evidence.set_defaults(
        handler=command_normalize_ai_language_evidence
    )

    ai_terms = subparsers.add_parser("ai-code-terms")
    ai_terms.add_argument("--model", default="qwen3:8b")
    ai_terms.add_argument("--codebook-output")
    ai_terms.add_argument("--force", action="store_true")
    _add_output_argument(ai_terms)
    ai_terms.set_defaults(handler=command_ai_code_terms)

    ai_dimensions = subparsers.add_parser("ai-code-dimensions")
    ai_dimensions.add_argument("--model", default="qwen3:8b")
    ai_dimensions.add_argument("--force", action="store_true")
    _add_output_argument(ai_dimensions)
    ai_dimensions.set_defaults(handler=command_ai_code_dimensions)

    export_terms = subparsers.add_parser("export-term-extraction")
    _add_output_argument(export_terms)
    export_terms.set_defaults(handler=command_export_term_extraction)

    import_terms = subparsers.add_parser("import-terms")
    import_terms.add_argument("--input", required=True)
    import_terms.set_defaults(handler=command_import_terms)

    export_coding = subparsers.add_parser("export-term-coding")
    export_coding.add_argument(
        "--reviewer",
        choices=("AI", "H1", "H2"),
        required=True,
    )
    export_coding.add_argument(
        "--only-missing",
        action="store_true",
        help="Export only active terms lacking this reviewer's coding.",
    )
    _add_output_argument(export_coding)
    export_coding.set_defaults(handler=command_export_term_coding)

    import_coding = subparsers.add_parser("import-term-coding")
    import_coding.add_argument("--input", required=True)
    import_coding.set_defaults(handler=command_import_term_coding)

    code_terms = subparsers.add_parser("code-terms")
    code_terms.add_argument("--input")
    code_terms.add_argument("--output-dir")
    code_terms.set_defaults(handler=command_code_terms)

    subparsers.add_parser("derive-search-frame").set_defaults(
        handler=command_derive_search_frame
    )

    export_press = subparsers.add_parser("export-press")
    export_press.add_argument(
        "--only-pending",
        action="store_true",
        help="Export only active logical queries lacking a PRESS pass.",
    )
    _add_output_argument(export_press)
    export_press.set_defaults(handler=command_export_press)

    import_press = subparsers.add_parser("import-press")
    import_press.add_argument("--input", required=True)
    import_press.set_defaults(handler=command_import_press)

    subparsers.add_parser("resolve-press-redundancy").set_defaults(
        handler=command_resolve_press_redundancy
    )

    export_press_revisions = subparsers.add_parser(
        "export-press-revisions"
    )
    _add_output_argument(export_press_revisions)
    export_press_revisions.set_defaults(
        handler=command_export_press_revisions
    )

    import_press_revisions = subparsers.add_parser(
        "import-press-revisions"
    )
    import_press_revisions.add_argument("--input", required=True)
    import_press_revisions.set_defaults(
        handler=command_import_press_revisions
    )

    subparsers.add_parser("apply-press-revisions").set_defaults(
        handler=command_apply_press_revisions
    )

    seed_template = subparsers.add_parser("export-seed-template")
    _add_output_argument(seed_template)
    seed_template.set_defaults(handler=command_export_seed_template)

    import_seeds = subparsers.add_parser("import-seeds")
    import_seeds.add_argument("--input", required=True)
    import_seeds.set_defaults(handler=command_import_seeds)

    seed_search_log = subparsers.add_parser(
        "export-hidden-seed-search-log"
    )
    _add_output_argument(seed_search_log)
    seed_search_log.set_defaults(
        handler=command_export_hidden_seed_search_log
    )

    import_seed_search_log = subparsers.add_parser(
        "import-hidden-seed-search-log"
    )
    import_seed_search_log.add_argument("--input", required=True)
    import_seed_search_log.set_defaults(
        handler=command_import_hidden_seed_search_log
    )

    subparsers.add_parser("validate-search-frame").set_defaults(
        handler=command_validate_search_frame
    )

    export_supplements = subparsers.add_parser("export-seed-supplements")
    _add_output_argument(export_supplements)
    export_supplements.set_defaults(handler=command_export_seed_supplements)

    import_supplements = subparsers.add_parser("import-seed-supplements")
    import_supplements.add_argument("--input", required=True)
    import_supplements.set_defaults(handler=command_import_seed_supplements)

    subparsers.add_parser("freeze-search-frame").set_defaults(
        handler=command_freeze_search_frame
    )

    reopen_search = subparsers.add_parser("reopen-search-frame")
    reopen_search.add_argument("--notes", required=True)
    reopen_search.set_defaults(handler=command_reopen_search_frame)

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--max-pages-per-query", type=int)
    retrieve_parser.set_defaults(handler=command_retrieve)

    export_screen = subparsers.add_parser("export-screening")
    export_screen.add_argument(
        "--reviewer",
        choices=("AI", "H1", "H2"),
        required=True,
    )
    _add_output_argument(export_screen)
    export_screen.set_defaults(handler=command_export_screening)

    import_screen = subparsers.add_parser("import-screening")
    import_screen.add_argument("--input", required=True)
    import_screen.set_defaults(handler=command_import_screening)

    subparsers.add_parser("finalize-screening").set_defaults(
        handler=command_finalize_screening
    )

    screen_literature = subparsers.add_parser("screen-literature")
    screen_literature.add_argument("--input")
    screen_literature.add_argument("--output-dir")
    screen_literature.set_defaults(handler=command_screen_literature)

    crossref = subparsers.add_parser("crossref-validate")
    crossref.add_argument(
        "--scope",
        choices=("all", "included"),
        default="all",
    )
    crossref.add_argument(
        "--max-records",
        type=int,
        help="Process a bounded resumable batch.",
    )
    crossref.add_argument(
        "--workers",
        type=int,
        help=(
            "Crossref concurrency; defaults to the active public/polite "
            "pool limit and rejects larger values."
        ),
    )
    crossref.set_defaults(handler=command_crossref_validate)

    export_conflicts = subparsers.add_parser("export-crossref-conflicts")
    _add_output_argument(export_conflicts)
    export_conflicts.set_defaults(handler=command_export_crossref_conflicts)

    import_conflicts = subparsers.add_parser(
        "import-crossref-resolutions"
    )
    import_conflicts.add_argument("--input", required=True)
    import_conflicts.set_defaults(
        handler=command_import_crossref_resolutions
    )

    export_indicators = subparsers.add_parser(
        "export-indicator-extraction"
    )
    _add_output_argument(export_indicators)
    export_indicators.set_defaults(
        handler=command_export_indicator_extraction
    )

    adjudicate_indicators = subparsers.add_parser(
        "export-indicator-adjudication"
    )
    _add_output_argument(adjudicate_indicators)
    adjudicate_indicators.set_defaults(
        handler=command_export_indicator_adjudication
    )

    acquire_fulltexts = subparsers.add_parser(
        "acquire-open-fulltexts"
    )
    acquire_fulltexts.add_argument("--output-dir")
    acquire_fulltexts.add_argument("--max-records", type=int)
    acquire_fulltexts.add_argument(
        "--retry-failed",
        action="store_true",
    )
    acquire_fulltexts.add_argument(
        "--skip-location-hydration",
        action="store_true",
    )
    acquire_fulltexts.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
    )
    acquire_fulltexts.add_argument(
        "--maximum-bytes",
        type=int,
        default=100_000_000,
    )
    acquire_fulltexts.set_defaults(
        handler=command_acquire_open_fulltexts
    )

    import_indicator = subparsers.add_parser("import-indicators")
    import_indicator.add_argument("--input", required=True)
    import_indicator.set_defaults(handler=command_import_indicators)

    extract = subparsers.add_parser("extract-indicators")
    extract.add_argument("--input")
    _add_output_argument(extract)
    extract.set_defaults(handler=command_extract_indicators)

    export_data_audit = subparsers.add_parser("export-data-audit")
    _add_output_argument(export_data_audit)
    export_data_audit.set_defaults(handler=command_export_data_audit)

    import_data_audit = subparsers.add_parser("import-data-audit")
    import_data_audit.add_argument("--input", required=True)
    import_data_audit.set_defaults(handler=command_import_data_audit)

    build_input_inventory = subparsers.add_parser(
        "build-local-input-inventory"
    )
    _add_output_argument(build_input_inventory)
    build_input_inventory.set_defaults(
        handler=command_build_local_input_inventory
    )

    export_data_correspondence = subparsers.add_parser(
        "export-data-correspondence"
    )
    export_data_correspondence.add_argument(
        "--reviewer",
        choices=("AI", "H1", "H2"),
        required=True,
    )
    export_data_correspondence.add_argument("--inventory")
    _add_output_argument(export_data_correspondence)
    export_data_correspondence.set_defaults(
        handler=command_export_data_correspondence
    )

    import_data_correspondence = subparsers.add_parser(
        "import-data-correspondence"
    )
    import_data_correspondence.add_argument("--input", required=True)
    import_data_correspondence.set_defaults(
        handler=command_import_data_correspondence
    )

    export_operationalization = subparsers.add_parser(
        "export-operationalization"
    )
    export_operationalization.add_argument(
        "--reviewer",
        choices=("AI", "H1", "H2"),
        required=True,
    )
    _add_output_argument(export_operationalization)
    export_operationalization.set_defaults(
        handler=command_export_operationalization
    )

    import_operationalization = subparsers.add_parser(
        "import-operationalization"
    )
    import_operationalization.add_argument("--input", required=True)
    import_operationalization.set_defaults(
        handler=command_import_operationalization
    )

    export_dimensions = subparsers.add_parser("export-dimension-coding")
    export_dimensions.add_argument(
        "--reviewer",
        choices=("AI", "H1", "H2"),
        required=True,
    )
    _add_output_argument(export_dimensions)
    export_dimensions.set_defaults(handler=command_export_dimension_coding)

    import_dimensions = subparsers.add_parser("import-dimension-coding")
    import_dimensions.add_argument("--input", required=True)
    import_dimensions.set_defaults(handler=command_import_dimension_coding)

    derive_dimensions = subparsers.add_parser("derive-dimensions")
    derive_dimensions.add_argument("--input")
    derive_dimensions.add_argument("--export-templates", action="store_true")
    derive_dimensions.add_argument("--output-dir")
    derive_dimensions.set_defaults(handler=command_derive_dimensions)

    subparsers.add_parser("select-indicators").set_defaults(
        handler=command_select_indicators
    )

    citation = subparsers.add_parser("citation-track")
    citation.add_argument("--iteration", type=int, required=True)
    citation.add_argument(
        "--scope",
        choices=(
            "included",
            "indicator_sources",
            "reviews_and_indicator_sources",
        ),
        default="reviews_and_indicator_sources",
    )
    citation.set_defaults(handler=command_citation_track)

    saturation = subparsers.add_parser("record-saturation-round")
    saturation.add_argument("--iteration", type=int, required=True)
    saturation.add_argument("--new-records", type=int, required=True)
    saturation.add_argument("--new-terms", type=int, required=True)
    saturation.add_argument(
        "--new-indicator-families",
        type=int,
        required=True,
    )
    saturation.add_argument(
        "--decision",
        choices=("continue", "freeze"),
        required=True,
    )
    saturation.add_argument("--notes", required=True)
    saturation.set_defaults(handler=command_record_saturation)

    subparsers.add_parser("audit").set_defaults(handler=command_audit)
    subparsers.add_parser("status").set_defaults(handler=command_status)
    return parser


def main() -> None:
    """Execute one v3 state-machine command."""
    parser = build_parser()
    args = parser.parse_args()
    connection = initialize(Path(args.database).resolve())
    try:
        result = args.handler(connection, args)
        _print(result)
    finally:
        connection.close()


if __name__ == "__main__":
    main()

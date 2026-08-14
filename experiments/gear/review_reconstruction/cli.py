"""CLI for deterministic packaging and validated no-API session handoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from gear.trace import sha256_value

from .contracts import (
    ReconstructionSessionPackage,
    ReconstructionSessionResponse,
)
from .sessions import (
    build_reconstruction_package,
    import_session_response,
    load_jsonl,
    validate_session_response,
    write_session_handoff,
)


def _select_case(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.manifest)
    if args.case_id:
        for row in rows:
            values = {
                str(row.get("case_id") or ""),
                str(row.get("article_id") or ""),
                str(row.get("paper_id") or ""),
            }
            if args.case_id in values:
                return row
        raise ValueError(f"case not found: {args.case_id}")
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"case index out of range: {args.index}")
    return rows[args.index]


def _build_one(args: argparse.Namespace) -> int:
    case = _select_case(args)
    package, paper_ir = build_reconstruction_package(case)
    target = write_session_handoff(package, paper_ir, args.output_dir)
    print(json.dumps({"package_id": package.package_id, "path": str(target)}, indent=2))
    return 0


def _build_batch(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.manifest)
    selected = rows[: args.limit] if args.limit else rows
    built: list[dict[str, str]] = []
    for case in selected:
        package, paper_ir = build_reconstruction_package(case)
        target = args.output_dir / _safe_id(package.paper_id) / "reconstruction"
        write_session_handoff(package, paper_ir, target)
        built.append(
            {
                "package_id": package.package_id,
                "paper_id": package.paper_id,
                "path": str(target),
            }
        )
    manifest = {
        "schema_version": "aspr_gear",
        "contract": "reconstruction_handoff_batch",
        "dataset_id": args.dataset_id,
        "development_non_confirmatory": args.dataset_id == "nature_dev100",
        "paper_count": len(built),
        "sessions": built,
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _seal_response(args: argparse.Namespace) -> int:
    payload = json.loads(args.response.read_text(encoding="utf-8"))
    payload["output_sha256"] = "sha256:" + "0" * 64
    response = ReconstructionSessionResponse.model_validate(payload)
    payload["output_sha256"] = sha256_value(response.hash_payload())
    target = args.output or args.response
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(target), "output_sha256": payload["output_sha256"]}, indent=2
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    package = ReconstructionSessionPackage.model_validate_json(
        args.package.read_text(encoding="utf-8")
    )
    response = ReconstructionSessionResponse.model_validate_json(
        args.response.read_text(encoding="utf-8")
    )
    validate_session_response(package, response)
    print(
        json.dumps({"passed": True, "output_sha256": response.output_sha256}, indent=2)
    )
    return 0


def _import(args: argparse.Namespace) -> int:
    target = import_session_response(args.package, args.response, args.output_dir)
    print(json.dumps({"imported": str(target)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GEAR graph-blind Nature review reconstruction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("build-one")
    one.add_argument("--manifest", type=Path, required=True)
    one.add_argument("--case-id")
    one.add_argument("--index", type=int, default=0)
    one.add_argument("--output-dir", type=Path, required=True)
    one.set_defaults(handler=_build_one)
    batch = commands.add_parser("build-batch")
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--limit", type=int)
    batch.add_argument("--dataset-id", default="nature_dev100")
    batch.set_defaults(handler=_build_batch)
    seal = commands.add_parser("seal-response")
    seal.add_argument("--response", type=Path, required=True)
    seal.add_argument("--output", type=Path)
    seal.set_defaults(handler=_seal_response)
    validate = commands.add_parser("validate-response")
    validate.add_argument("--package", type=Path, required=True)
    validate.add_argument("--response", type=Path, required=True)
    validate.set_defaults(handler=_validate)
    imported = commands.add_parser("import-response")
    imported.add_argument("--package", type=Path, required=True)
    imported.add_argument("--response", type=Path, required=True)
    imported.add_argument("--output-dir", type=Path, required=True)
    imported.set_defaults(handler=_import)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


def _safe_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:160]


__all__ = ["build_parser", "main"]

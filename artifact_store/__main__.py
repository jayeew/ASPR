"""Command-line access to immutable cross-module artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import validate_dependency
from .contracts import ArtifactReference
from .store import ArtifactStore


def _reference(value: str) -> ArtifactReference:
    """Load a pinned dependency reference from a JSON file."""
    return ArtifactReference.model_validate_json(
        Path(value).read_text(encoding="utf-8")
    )


def _publish(args: argparse.Namespace) -> int:
    dependencies = [_reference(value) for value in args.dependency]
    for dependency in dependencies:
        validate_dependency(args.producer, dependency.producer)
    reference = ArtifactStore(args.store).publish_directory(
        producer=args.producer,
        artifact=args.artifact,
        release=args.release,
        source=args.source,
        dependencies=dependencies,
    )
    print(reference.model_dump_json(indent=2))
    return 0


def _resolve(args: argparse.Namespace) -> int:
    reference = _reference(args.reference)
    path = ArtifactStore(args.store).resolve(reference)
    print(json.dumps({"path": str(path), "reference": reference.model_dump()}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ASPR immutable artifact exchange")
    parser.add_argument("--store", type=Path, default=Path("artifacts"))
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--producer", required=True)
    publish.add_argument("--artifact", required=True)
    publish.add_argument("--release", required=True)
    publish.add_argument("--source", required=True, type=Path)
    publish.add_argument("--dependency", action="append", default=[])
    publish.set_defaults(handler=_publish)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--reference", required=True)
    resolve.set_defaults(handler=_resolve)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

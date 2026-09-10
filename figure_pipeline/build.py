"""Build an auditable draft or fail closed before publication export."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path

from figure_pipeline.specs.contract import REQUIREMENTS, publication_missing, require_publication
from figure_pipeline.qa.checks import audit_exports

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/FROM_WEB_v2'


def source_manifest() -> dict[str, str]:
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'outputs/FROM_WEB/data').glob('*')) if p.is_file()}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['draft','publication'],default='draft')
    parser.add_argument('--figures',nargs='+',type=int,default=list(range(1,11)))
    args=parser.parse_args()
    artifacts_path=OUT/'audit/artifact_attestations.json'
    artifacts=json.loads(artifacts_path.read_text()) if artifacts_path.exists() else {}
    if args.mode=='publication':
        require_publication(args.figures,artifacts)
        reviews_path=OUT/'audit/manual_visual_review.json'
        reviews=json.loads(reviews_path.read_text()) if reviews_path.exists() else {}
        unreviewed=[f for f in args.figures if reviews.get(str(f),{}).get('approved_for_publication') is not True]
        if unreviewed:
            raise SystemExit(f'Publication blocked: final-size visual approval missing for {unreviewed}')
        target=OUT/'publication'
        target.mkdir(exist_ok=True)
        for number in args.figures:
            for path in OUT.glob(f'fig{number:02d}_*'):
                if path.suffix in {'.png','.svg','.pdf'}:
                    shutil.copy2(path,target/path.name)
        return
    before=source_manifest()
    for name,ids in [('benchmark_revision',{4}),('graph_revision',{1,3}),('case_revision',{2,8,9,10})]:
        if ids.intersection(args.figures):
            module=importlib.import_module(f'figure_pipeline.{name}')
            module.main()
    after=source_manifest()
    if before!=after:
        raise RuntimeError('Frozen source files changed during build')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit/source_manifest.json').write_text(json.dumps(before,indent=2))
    (OUT/'audit/export_qa.json').write_text(json.dumps(audit_exports(OUT),indent=2))
    (OUT/'audit/publication_status.json').write_text(json.dumps({str(f):{'required_artifacts':REQUIREMENTS[f],'missing':publication_missing(f,artifacts)} for f in args.figures},indent=2))


if __name__=='__main__': main()

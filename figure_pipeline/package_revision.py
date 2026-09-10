"""Create a new revision ZIP without overwriting the audited original archive."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/packages/ASPR_main_code_figures_data_revision_v2_20260909.zip'
SKIP={'__pycache__','.pytest_cache','.git','node_modules','.ruff_cache','versions'}
EXT={'.py','.md','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.sh','.bash','.txt','.sql','.js','.ts','.r','.R'}


def files() -> list[Path]:
    paths=[]
    for name in ['gear','experiments','scripts','configs','docs','tests','figure_pipeline']:
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in EXT and not SKIP.intersection(p.parts) and not p.name.startswith('.env'))
    for name in ['outputs/FROM_WEB','outputs/FROM_WEB_v2']:
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and not p.is_symlink() and not SKIP.intersection(p.parts))
    paths.extend(ROOT/n for n in ['README.md','AGENTS.md','Makefile','requirements.txt'] if (ROOT/n).exists())
    return sorted(set(paths))


def main() -> None:
    entries=[]
    requirements='\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in ['matplotlib','numpy','pandas','scipy','networkx','scikit-learn','Pillow','pypdf'])+'\n'
    with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files():
            relative=path.relative_to(ROOT).as_posix(); body=path.read_bytes()
            archive.writestr('ASPR/'+relative,body)
            entries.append(dict(path=relative,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
        for name,body in [('requirements-plot-replay.txt',requirements.encode()),('PACKAGE_README.md',(ROOT/'figure_pipeline/README.md').read_bytes())]:
            archive.writestr('ASPR/'+name,body)
            entries.append(dict(path=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
        archive.writestr('ASPR/MANIFEST.json',json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),source='working tree including existing user changes; frozen original figure snapshot and separate v2 derived outputs',files=entries),indent=2))
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read('ASPR/'+row['path'])).hexdigest()==row['sha256'] for row in entries)
    digest=hashlib.sha256(DEST.read_bytes()).hexdigest()
    DEST.with_suffix('.zip.sha256').write_text(f'{digest}  {DEST.name}\n')
    print(json.dumps(dict(path=str(DEST),files=len(entries)+1,size_mib=DEST.stat().st_size/1048576,sha256=digest)))


if __name__=='__main__': main()

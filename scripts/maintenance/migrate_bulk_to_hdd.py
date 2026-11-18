"""Move selected large inactive files to HDD with SHA-256 verified local links."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path('/mnt/d/ASPR-data')
STATE = ROOT/'docs/hdd_migration.json'


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024**2):h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert os.path.ismount('/mnt/d')
    items=json.loads((ROOT/'docs/hdd_large_files.json').read_text())
    state={'strategy':'Files >=128 MiB; small files and directory indices remain on SSD', 'destination':str(DEST),'items':{}}
    if STATE.exists():
        old=json.loads(STATE.read_text())
        if old.get('strategy')==state['strategy']:state=old
    def save() -> None:
        temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)
    for size,rel in items:
        src,dst=ROOT/rel,DEST/rel
        entry=state['items'].setdefault(rel,{'bytes':size,'target':str(dst)})
        if src.is_symlink():
            assert src.resolve()==dst.resolve() and entry['status']=='complete'
            continue
        before=src.stat()
        assert before.st_size==size
        assert shutil.disk_usage(DEST.parent).free>size+10*1024**3
        dst.parent.mkdir(parents=True,exist_ok=True)
        part=dst.with_name(dst.name+'.partial')
        h=hashlib.sha256();copied=0;last=time.monotonic()
        entry.update(status='copying',copied_bytes=0);save()
        print(f'Copying {rel}: {size/1024**3:.2f} GiB',flush=True)
        with src.open('rb') as source,part.open('wb') as target:
            while chunk:=source.read(8*1024**2):
                target.write(chunk);h.update(chunk);copied+=len(chunk)
                if time.monotonic()-last>10:
                    entry['copied_bytes']=copied;save();last=time.monotonic()
            target.flush();os.fsync(target.fileno())
        entry.update(status='verifying',copied_bytes=copied);save()
        print(f'Verifying SHA-256: {rel}',flush=True)
        assert digest(part)==h.hexdigest(),'Destination checksum mismatch; source retained'
        after=src.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed; retained'
        os.utime(part,ns=(before.st_atime_ns,before.st_mtime_ns))
        part.replace(dst)
        link=src.with_name(src.name+'.hdd_link')
        assert not link.exists()
        link.symlink_to(dst)
        assert link.stat().st_size==size
        # Atomic replacement removes the local file only after full-content verification.
        link.replace(src)
        entry.update(status='complete',sha256=h.hexdigest());save()
        print(f'Complete; released {size/1024**3:.2f} GiB: {rel}',flush=True)
    state['complete']=True;save();print('All selected large files migrated',flush=True)

if __name__=='__main__':main()

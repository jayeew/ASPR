"""Package local source plus aggregate graph summaries, never raw research data."""
from __future__ import annotations
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'outputs/packages/ASPR_local_code_and_graph_statistics.zip'
SKIP = {'__pycache__', '.git', '.pytest_cache', '.ruff_cache', 'node_modules'}
SOURCE_EXTENSIONS = {'.py', '.sh', '.bash', '.js', '.ts', '.tsx', '.jsx', '.r', '.R', '.sql', '.html', '.css', '.md'}


def selected_files() -> list[Path]:
    paths = []
    for directory in ['gear', 'experiments', 'scripts', 'figure_pipeline', 'tests']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in SOURCE_EXTENSIONS and not SKIP.intersection(p.parts))
    paths.extend(p for p in (ROOT/'experiments').rglob('config.json') if p.is_file())
    paths.extend(p for p in (ROOT/'figure_pipeline/specs').glob('*.json') if p.is_file())
    for directory in ['outputs/FROM_WEB/scripts']:
        paths.extend(p for p in (ROOT/directory).glob('*.py'))
    for p in (ROOT/'configs').rglob('*'):
        if p.is_file() and p.suffix in {'.json', '.yaml', '.yml', '.toml'} and not any('metadata' in part for part in p.parts):
            paths.append(p)
    names = ['README.md','AGENTS.md','Makefile','requirements.txt','requirements-figures.txt','docs/module_architecture.md','docs/innovation_v2_implementation.md','docs/figure_audit_runtime_corrections.md','experiments/innovation_200/README.md','figure_pipeline/README.md','figure_pipeline/specs/research_protocol.md']
    paths.extend(ROOT/name for name in names if (ROOT/name).is_file())
    paths.extend(p for p in (ROOT/'outputs/GRAPH_STATISTICS').rglob('*') if p.is_file() and p.suffix in {'.md','.json','.csv','.html'})
    return sorted(set(paths))


def main() -> None:
    DEST.parent.mkdir(parents=True,exist_ok=True)
    files=selected_files(); records=[]
    readme='''# 本地代码与图谱统计包

包含打包时工作区源码（含未提交改动）、运行配置、绘图程序、测试、必要说明，以及 outputs/GRAPH_STATISTICS/ 中的汇总统计。

不包含 data/ 原始数据库/Parquet/论文/嵌入、逐论文或逐claim实验记录、图稿源快照、日志、模型权重、.env、Git历史。configs中的单论文metadata也未纳入。代码中的接口/常量/测试样例不属于本次导出的研究原始数据。

图谱统计以数据库、保存历史结果、当前运行观察和展示子图分别列示；文件SHA-256证明包内容完整，不证明科学判断正确。缺少原始资产时此包不能独立重绘旧图或重新运行完整GEAR，符合本次仅要代码与统计数据的范围。

统计重算入口：scripts/reporting/summarize_claim_graph.py、summarize_paper_graph.py、summarize_runtime_graph.py；汇总入口assemble_graph_statistics.py；运行这些脚本需要原工作区资产。
'''
    with zipfile.ZipFile(DEST,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            relative=path.relative_to(ROOT).as_posix()
            if relative.startswith('data/') or path.name.startswith('.env'):
                raise ValueError('Raw data or environment file selected')
            body=path.read_bytes();archive.writestr('ASPR/'+relative,body)
            records.append({'path':relative,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
        body=readme.encode();archive.writestr('ASPR/PACKAGE_README.md',body)
        records.append({'path':'PACKAGE_README.md','bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
        archive.writestr('ASPR/MANIFEST.json',json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'local source and aggregate statistics only','files':records},indent=2))
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read('ASPR/'+r['path'])).hexdigest()==r['sha256'] for r in records)
        assert not any(n.startswith('ASPR/data/') or '/.env' in n or n.endswith(('.parquet','.sqlite','.npy','.faiss')) for n in archive.namelist())
    checksum=hashlib.sha256(DEST.read_bytes()).hexdigest()
    DEST.with_suffix('.zip.sha256').write_text(checksum+'  '+DEST.name+'\n')
    print(json.dumps({'path':str(DEST),'files':len(records)+1,'mib':round(DEST.stat().st_size/1048576,2),'sha256':checksum}))


if __name__=='__main__':main()

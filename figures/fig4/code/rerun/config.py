from __future__ import annotations

from pathlib import Path

from figure_pipeline.fig3_revision.config import Config as BaseConfig, ROOT
from figure_pipeline.fig3_revision.models import Record


NAMES = {
    'T': '纯论文报告', 'E': '仅 GEAR', 'G': '仅 Graph', 'F': '完整系统',
    'F_noJ': '移除联合图', 'F_noM': '屏蔽结构数值', 'F_noP': '屏蔽引用路径',
}
CONDITIONS = tuple(NAMES)
METRICS = {'C': '回应覆盖率', 'H': '正确历史比较覆盖率', 'S': '范围正确回应率'}
CONTRASTS = {'GEAR': 'G', 'Graph branch': 'E', 'Joint': 'F_noJ',
             'Structural summaries': 'F_noM', 'Paper paths': 'F_noP'}


class Config(BaseConfig):
    evaluation_round: str = 'v2_scientific_content'
    source: Path = ROOT / 'outputs/fig4_mechanisms'
    output: Path = ROOT / 'outputs/fig4_rerun'
    executable: str = str(ROOT / 'figure_pipeline/fig4_rerun/codex_http')
    download_fulltext: bool = False
    repair_attempts: int = 0
    network_retries: int = 0
    cli_initial: int = 64
    timeout_seconds: int = 900
    synthesis_timeout_seconds: int = 900
    material_max_chars: int = 384000
    request_max_chars: int = 448000
    request_max_bytes: int = 1800000


class Report(Record):
    body: str
    cited_source_ids: list[str]


class SourceEvidence(Record):
    source_id: str
    source_quote: str
    source_location: str


class CoreJudgment(Record):
    core_id: str
    addressed: bool
    report_quotes: list[str]
    historical_comparison_correct: bool | None
    scope_correct: bool | None
    evidence: list[SourceEvidence]
    reason: str
    unresolved_reason: str


class Evaluation(Record):
    items: list[CoreJudgment]

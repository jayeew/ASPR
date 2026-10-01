"""中文终端进度及普通追加日志；不参与实验结果判定。"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

from .config import Config

_LOCK = threading.Lock()
STAGE_NAMES = {
    'claims': '共享贡献抽取', 'manuscript_notes': '中立正文阅读', 'direct_a': 'Direct-A生成',
    'gear': 'GEAR生成', 'graph': 'Graph生成', 'full': 'Full生成',
    'prepare': '数据准备', 'core': '核心贡献', 'review_sections': '审稿分段',
    'tone': '审稿语气', 'checklist': '真人问题重建', 'review_dynamics': '跨轮审稿变化',
    'eacl': 'EACL基线', 'reviewgrounder': 'ReviewGrounder基线', 'evidence_pool': '中立历史来源池',
    'reference': '历史参考R', 'extract': '报告断言与判断P', 'support': '来源核验',
    'novelty': '核心判断评价V', 'quality_checklist': '五维评价清单', 'quality': '五维评分',
    'concerns': '真人问题比较', 'clusters': '六方法信息聚类', 'fusion': '融合对应',
    'preference': '双顺序偏好', 'recheck': '重复评价与控制', 'baseline': '基线模型步骤',
    'search': '检索规划', 'importance': '信息重要性', 'aggregate': '统计汇总',
    'render': '组件及panel绘制', 'assemble': '母版组装',
}


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f'{hours}小时{minutes:02d}分{rest:02d}秒' if hours else f'{minutes}分{rest:02d}秒'


def log(config: Config, event: str, message: str, **fields: Any) -> None:
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    stage = fields.get('stage', '')
    context = ' | '.join(str(v) for v in (
        STAGE_NAMES.get(stage, stage), fields.get('paper_id'), fields.get('method'), fields.get('step')) if v)
    line = f'[{now}] [{event}]'+(f' [{context}]' if context else '')+' '+message
    record = {'time': now, 'event': event, 'message': message, **fields}
    directory = config.output/'logs'
    with _LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'progress.log').open('a', encoding='utf-8') as handle:
            handle.write(line+'\n')
        with (directory/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
        if event not in {'网络请求', '网络重试'}:
            print(line, flush=True)

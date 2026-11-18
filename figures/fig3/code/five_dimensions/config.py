from __future__ import annotations
from figure_pipeline.paths import PROJECT_ROOT

from pathlib import Path

ROOT = PROJECT_ROOT
STUDY = ROOT / 'outputs/fig3_reference/study'
OUTPUT = ROOT / 'outputs/innovation_five_dimensions'
MODEL = 'gpt-5.6-luna'
PROTOCOL = 'five_dimensions_20261117'
METHODS = ('direct_a', 'eacl', 'reviewgrounder', 'gear', 'graph', 'full')
METHOD_LABELS = {'direct_a': 'GPT-5.6', 'eacl': 'EACL', 'reviewgrounder': 'ReviewGrounder',
                 'gear': 'GEAR', 'graph': 'Graph', 'full': 'Full'}
DIMENSIONS = {
    'historical_originality': '历史原创性',
    'understanding_capability': '认识与能力推进',
    'knowledge_organisation': '知识组织与整合',
    'evidential_basis': '证据基础',
    'scientific_value': '科学价值',
}
DAMAGE_ID = 's41467-026-69179-5'
PILOT_IDS = ('s41467-026-68311-9', 's41467-026-68346-y', 's42004-026-01947-2',
             's41467-026-68499-w', 's41467-026-68290-x')

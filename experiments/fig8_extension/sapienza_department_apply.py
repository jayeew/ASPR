"""Apply and register the officially corroborated explicit Sapienza department signature."""
from __future__ import annotations

import fcntl
import json

from .official_units_apply import run as apply_rules, resolve
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT/'supplement/official_entities'
    path = 'supplement/official_entities/sapienza_department_rules.json'
    with (folder/'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        result = apply_rules(rules_path=path, status_name='sapienza_department_status.json')
        state_path = folder/'current_rule_reuse_scope.json'
        state = json.loads(state_path.read_text())
        state['next_rule_paths'] = list(dict.fromkeys([*state['next_rule_paths'], path]))
        atomic_json(state_path, state)
        atomic_json(folder/'sapienza_department_registration.json',
                    {'result':result,'incremental_reuse_registered':True,'network_requests':0})
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()

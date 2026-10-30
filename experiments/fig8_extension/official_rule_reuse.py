"""Reuse reviewed exact institution rules on incrementally changed author records."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .official_units_apply import resolve, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def verify_changes(low: int, ids: set[str], paths: list[str]) -> int:
    aliases = {}
    for path in paths:
        for rule in json.loads((ROOT / path).read_text())["rules"]:
            for name in rule["exact_names"]:
                aliases.setdefault(name, {**rule, "_rule_receipt": path})
    db = connect()
    checked = 0
    rows = db.execute(
        "SELECT work_id,before_value,after_value FROM changes "
        "WHERE rowid>? AND field='authors_json' AND receipt=?",
        (low, json.dumps(paths)),
    )
    for wid, before, after in rows:
        if wid not in ids:
            continue
        old, new = json.loads(before), json.loads(after)
        assert new == [resolve(a, aliases, paths[0])[0] for a in old]
        assert all(resolve(a, aliases, paths[0])[1] == 0 for a in new)
        for a, b in zip(old, new):
            assert a.get("countries") == b.get("countries")
            assert [s.get("country_codes") for s in a.get("publisher_affiliation_slots") or []] == [
                s.get("country_codes") for s in b.get("publisher_affiliation_slots") or []]
            if a.get("nature_v1_slots") is not None:
                assert a == b
        checked += 1
    db.close()
    return checked


def _run() -> None:
    folder = ROOT / "supplement/official_entities"
    previous = json.loads((folder / "new_qualified_rule_reuse_scope.json").read_text())
    state = folder / "current_rule_reuse_scope.json"
    old = json.loads(state.read_text()) if state.exists() else {}
    paths = old.get("next_rule_paths", previous["rule_paths"])
    low = old.get("completed_through_change_rowid", previous["changes_rowid_before"])
    db = connect()
    high = db.execute("SELECT max(rowid) FROM changes").fetchone()[0]
    ids = {r[0] for r in db.execute(
        "SELECT DISTINCT work_id FROM changes "
        "WHERE rowid>? AND rowid<=? AND field='authors_json'", (low, high))}
    db.close()
    scope = {"next_rule_paths": paths, "rule_paths": paths,
             "completed_through_change_rowid": low, "changes_rowid_after": low,
             "changes_rowid_through": high, "selected_papers": len(ids),
             "started_at": datetime.now(timezone.utc).isoformat(), "network_requests": 0}
    atomic_json(state, scope)
    result = apply_rules(rule_paths=paths, status_name="current_rule_reuse_status.json",
                         selected_work_ids=ids)
    checked = verify_changes(high, ids, paths)
    assert checked == result.get("papers_changed", 0)
    scope.update(completed_through_change_rowid=high, result=result, validation={
        "changed_papers_reproduced": checked, "repeat_safe": True,
        "countries_and_nature_preserved": True})
    atomic_json(state, scope)
    if checked:
        refresh()
    print(json.dumps({"selected_papers": len(ids), "rule_files": len(paths),
                      "result": result, "validation": scope["validation"]}), flush=True)


def run() -> None:
    """Serialize the incremental cursor when several metadata consumers finish."""
    with (ROOT / 'supplement/official_entities/rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        _run()


if __name__ == "__main__":
    run()

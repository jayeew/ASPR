#!/usr/bin/env bash
set -u -o pipefail

# API credentials and proxy routing are intentionally supplied by the caller.
while true; do
  python3 pipeline.py bootstrap-saturation --maximum-queries 1
  remaining="$(python3 - <<'PY'
import sqlite3
connection = sqlite3.connect('outputs/evidence_derived_v4.sqlite3')
row = connection.execute('''
    SELECT COUNT(*)
    FROM discovery_queries AS q
    LEFT JOIN discovery_query_runs AS r
      ON r.discovery_query_id = q.discovery_query_id
    WHERE q.status = 'active' AND COALESCE(r.complete, 0) = 0
''').fetchone()
print(int(row[0]))
PY
)"
  if [ "$remaining" = "0" ]; then
    exit 0
  fi
done

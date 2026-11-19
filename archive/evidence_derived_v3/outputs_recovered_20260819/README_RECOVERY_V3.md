# Evidence-v3 session recovery

Status: **RECOVERED_WITH_DISCLOSED_LIMITATION**

This directory is a usable replacement definition bundle, but it is not a byte-for-byte copy of the deleted ignored SQLite/output tree.

## Exact session-log recovery

- 432 feature IDs, English labels, roles, and inventory flags.
- 66 candidate-dimension IDs and labels.
- Exact nested feature memberships: 7 / 16 / 154 / 221.
- Exact nested dimension memberships: 4 / 10 / 48 / 55.
- Exact feature-to-dimension mapping for all 221 Broad-T0 features.

## Deterministically reconstructed fields

- The 211 features excluded from Broad T0 were assigned to candidate dimensions from their English names and recovered dimension labels.
- Per-gate rows reproduce the four frozen-set memberships, but do not claim the original deleted CSV bytes or every original diagnostic field.
- Original full source-location evidence and the SQLite retrieval tables are not restored by this bundle.

Use `recovery_preflight_v3.json` and `recovery_manifest_v3.json` for machine-readable verification and provenance.

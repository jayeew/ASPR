# Independent Codex review language addendum v3

This addendum clarifies the already frozen English-only eligibility boundary.
It does not change the scientific-topic inclusion rules.

For every screening record:

- an explicit non-English OpenAlex language value is presumptive evidence of
  a non-English publication and must be resolved from the supplied metadata;
- a clearly non-English original title is ineligible even when OpenAlex
  supplies an English translated abstract;
- a clearly non-English abstract is ineligible even when the title is
  translated into English;
- mojibake or broken character decoding in an otherwise English title or
  abstract is not, by itself, evidence of a non-English publication;
- where the supplied title/abstract metadata cannot establish an English
  publication, use `language_judgment=non_en`,
  `decision=exclude`, and
  `exclusion_reason=E_LANGUAGE_NON_ENGLISH`;
- `language_evidence` and `evidence_span` must still be exact substrings of
  the supplied title or abstract.

The prior review of topic relevance may be preserved for records whose
language judgment remains `en`. Every earlier completed round must be
rechecked against this language rule and issued as a new versioned CSV and
manifest. Superseded versions remain in the audit trail but may not supply
the final H2 decision.

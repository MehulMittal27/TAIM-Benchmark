# TAIM paper analysis protocol: corpus-identity amendment

**Protocol ID:** `taim-paper-analysis-protocol-2026-08-31-v2`

**Status:** Frozen. This dated amendment was recorded before inspecting any effectiveness
result produced from the recipe-v3 corpus. It supersedes only the corpus and Evaluation
Package identities in Section 2 of
[`taim-paper-analysis-protocol-2026-08-29-v1`](paper-analysis-protocol-2026-08-29.md)
and the admission treatment that follows from those identities. Every other registered
research question, System, metric, contrast, statistical rule, failure rule, stop rule,
table shell, and claim boundary in v1 remains unchanged.

**Decision record:** The corpus-content decision and the measured identity change were
recorded on GitHub Issue #99 on 2026-08-31, before a recipe-v3 effectiveness run. The
decision accepts the recipe-v3 `[HEALTHY_VOLUNTEERS] {true|false}` line as intended
Benchmark Release 0.1 content. It does not approve the release itself or close any
other release gate.

## 1. Replacement identities

For every fresh paper-bearing run, replace the Prepared Snapshot ID and Evaluation
Package ID in v1 Section 2 with:

- Prepared Snapshot ID:
  `sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8`
- Evaluation Package ID:
  `sha256:fa47e8e588e06a0d69f3fc32ea34f59113c875bdc8219327ce8faaa8930e9dba`

The recipe-v3 corpus has the same 375,580-trial membership as recipe v1. It adds the
healthy-volunteer line to 95,542 trials. No judgment, topic, or section payload changed.
The Evaluation Package identity changes because it binds the prepared-data identity; it
is not a hash of judgment bytes alone.

These identifiers record the accepted prepared-data output observed during release
preparation. They do **not** constitute the immutable public release identity. Final
execution remains blocked until Issue #99 produces an approved Benchmark Release 0.1
manifest, public tag, package/lock inputs, and an executable declaration of the headline
System. If the release gate produces different prepared-data identities, no run may
proceed under this amendment: a new dated amendment must explain and bind the difference
before effectiveness results are inspected.

## 2. Treatment of recipe-v1 evidence

The identities superseded by this amendment are:

- Prepared Snapshot ID:
  `sha256:65a3ecb67c65730f77dcdcb14f70db64e0baed85e0a3df4ced407f464dca0ad3`
- Evaluation Package ID:
  `sha256:bc7d90d250b464c17921c545c11b0159334ad343d0b5036d55468c6f02d36c0c`

Every result bound to either recipe-v1 identity remains immutable historical development
evidence for that exact corpus. Its measurement is not retracted or rewritten. It is not
eligible for a final paper table, figure, headline claim, or Benchmark Release 0.1
Published Result Bundle, and it must not be compared directly with a recipe-v3 result as
if the corpus were held constant.

All paper-bearing rows, including BM25, are fresh runs from the eventual immutable public
release. A command or reconstruction recipe is not a substitute for the resulting
identity-bound run manifest and admitted evidence bundle.

## 3. Unchanged analysis contract

This amendment makes no outcome-dependent method change. In particular:

- v1 Sections 3-14 remain authoritative except where they refer to the superseded
  identities or admit a recipe-v1 value as a paper result;
- the registered metrics, endpoints, contrasts, multiplicity treatment, statistical
  tests, seeds, validity gates, exclusions, and stop rules are unchanged;
- TREC Clinical Trials 2021 remains the sole authorized paper track, and TREC 2022/2023
  remain protected and out of scope;
- the development values named in v1 remain registration provenance only; and
- public-release scope, licensing, approval, publication, and executable-System gates
  remain owned by Issue #99.

The final paper must cite both v1 and this amendment. Where the two differ, this amendment
wins only for prepared-data identity and the resulting admission classification.

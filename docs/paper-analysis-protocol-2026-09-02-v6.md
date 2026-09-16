# TAIM paper analysis protocol: low-memory release amendment

**Protocol ID:** `taim-paper-analysis-protocol-2026-09-02-v6`

**Status:** Frozen under repository-owner authorization on 2026-09-02, before any effectiveness
result from the four TrialGPT rows was inspected. These bytes are immutable; later changes require
a new dated amendment.

This amendment changes protocol v5 only for the executable public release identity and for reuse
of the already frozen query plans. Protocol v5 still governs the admitted targets, query method,
retrieval method, consensus, provider budget, evidence, and admission rules. Protocol v4 governs
the experiment matrix, Task Inputs, Evaluation Packages, metrics, comparisons, and reporting.
Protocol v1 Section 8 remains the sole canonical statement of the recall-floor asymmetry.

## 1. Executable release identity

Every final TrialGPT row must execute the public TAIM Benchmark 0.1.2 wheel. Release 0.1.2 replaces
0.1.1 only as the executable release for fresh paper runs. It does not alter or overwrite the
published 0.1.0 or 0.1.1 tags, commits, manifests, trees, packages, or release assets.

Release 0.1.2 changes prepared-package loading from retained in-memory `TrialDocument` objects to
an immutable, repeatable, disk-backed sequence. The loader still verifies the complete package,
reconstructs the same Benchmark Snapshot and Evaluation Package identities, validates ordering,
provenance, capabilities, record counts, and content hashes, and rejects malformed input. This is
an execution-resource correction. It does not change a Track, Profile, Task Input, System, query,
retrieval score, ranking rule, metric, schema version, or artifact identity rule.

Stop before retrieval if the 0.1.2 loader reconstructs an identity different from the one frozen
for that row. Release validation must bind the exact 0.1.2 manifest, public tree, wheel, source
commit, and dependency lock.

## 2. Query-plan continuity

The 735 completed query-generation calls remain admitted without repetition. Their logical-call
identities bind patient text, topic and arm, prompt bytes and schema, model, reasoning effort,
provider configuration, and returned output. They do not bind the TAIM source commit or public
release tag. Release and source identities enter the later retrieval lock instead.

The admitted query-plan files are exactly:

| Target | Query-plan SHA-256 |
| --- | --- |
| TREC Clinical Trials 2021 external fidelity | `sha256:1a60326de204673575d37bd68e7e79ca73df5bb349c6c781800d8bcdfe760641` |
| TREC Clinical Trials 2022 judgment union | `sha256:f7745291be02242b5e2dd89ddfe6607343c2e65bb93e78f5c13bab3ffaac5217` |
| SIGIR Clinical Trials 2016 description judgment union | `sha256:90a1b7ec70f3d2cde21e815fb980e3287b7047523e94ca0cce45bbc03b9b5ca4` |
| SIGIR Clinical Trials 2016 summary judgment union | `sha256:ba2f8c28315913b4405d1f8654208d0297d78fb368b4258dfa07f95dd2ef4e16` |

The 0.1.2 retrieval producer must revalidate every topic, patient-text hash, arm, prompt contract,
provider configuration, logical-call identity, attempt trace, and output hash before retrieval.
Any mismatch stops the row. Retrieval artifacts must be generated fresh under 0.1.2 and may not be
relabelled from 0.1.1.

## 3. Laptop execution boundary

The low-memory qualification loaded and fully validated the 5.9 GB TREC Clinical Trials 2021
prepared package with 75 topics and 375,580 trials on an 18 GB Apple-silicon laptop. It reproduced
Prepared Snapshot ID
`sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8` and Evaluation Package ID
`sha256:fa47e8e588e06a0d69f3fc32ea34f59113c875bdc8219327ce8faaa8930e9dba` in 394.18 seconds with
79,118,336 bytes peak resident memory.

This measurement supports full prepared-package loading and projection into the bounded paper
Profiles on an 18 GB laptop. It does not claim that every System can rank the full `official-full`
corpus in that memory. Dense full-corpus indexes, rerankers, external Systems, and provider-backed
execution retain their own hardware and runtime requirements.

## 4. Unchanged rules

All other v5 rules remain unchanged, including the four admitted Track/Profile pairs, 500 System
candidates, up to 1,500 logical System calls per topic, exactly 48 generation workers, failure
handling, evidence retention, score-access boundary, and prohibition on TrialGPT for TREC 2023.
Missing evidence, provider failure, timeout, or resource limits remain missing and never become an
eligibility judgment.

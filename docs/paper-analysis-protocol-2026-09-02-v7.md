# TAIM paper analysis protocol: single-release amendment

**Protocol ID:** `taim-paper-analysis-protocol-2026-09-02-v7`

**Status:** Frozen under repository-owner authorization on 2026-09-02. These bytes are immutable;
later changes require a new dated amendment. This amendment was prompted by an executable support
catalogue failure, not by an effectiveness value. A collaborator reported that one BM25 run had
completed under the withdrawn release identity, but no metric value from that run was used to choose
this correction. That run is not admitted as a final paper row.

This amendment changes protocol v6 only for the executable release identity and the staged System's
catalogue closure. Protocol v6 still governs low-memory execution and query-plan continuity.
Protocol v5 governs the TrialGPT targets, query and retrieval methods, provider budget, evidence,
and admission rules. Protocol v4 governs the experiment matrix, Task Inputs, Evaluation Packages,
metrics, comparisons, and reporting. Protocol v1 Section 8 remains the sole canonical statement of
the recall-floor asymmetry.

## 1. One executable public release

Every final paper row must execute the rebuilt public TAIM Benchmark 0.1.0 release. The earlier
0.1.0 release and the later 0.1.1 and 0.1.2 release objects and tags were withdrawn before this
freeze. The rebuilt 0.1.0 incorporates the TrialGPT producer and memory-bounded prepared-package
loader previously developed under those patch-version identities.

The version string alone does not establish release identity. Every final run must bind the exact
0.1.0 manifest, public tree, package artifact, source commit, and dependency lock. An artifact from
any withdrawn release remains historical evidence and cannot be relabelled or admitted under the
rebuilt release.

## 2. Query-plan continuity

The 735 query-generation calls admitted by protocol v6 remain admitted without repetition. Their
four query-plan files and SHA-256 identities remain exactly those listed in v6. Query-plan identity
does not bind a TAIM release. The fresh retrieval artifacts and all downstream System artifacts must
bind the rebuilt 0.1.0 release.

## 3. Staged System catalogue closure

The executable support catalogue must declare `bm25-folded` and
`dense-qwen3-embedding-0.6b-no-template` on Profile
`trec-ct-2021-external-fidelity-26149`. The staged System already declares that Profile and accepts
only complete depth-5,000 runs from those two component Systems on the same Task Input, Evaluation
Package, Profile, Prepared Snapshot, folded-query bundle, release, and dependency lock.

This correction authorizes the existing component methods on the Profile consumed by the staged
System. It does not change either component, the staged ranking method, or the seven reportable rows
in protocol v4. The two component runs remain supporting stages, not additional reportable rows.

## 4. Fresh-run boundary

All seven rows in the TREC 2021 external-fidelity stratum must be rerun from the rebuilt 0.1.0
release and share its exact manifest, public tree, package artifact, and dependency lock. Runs made
against a withdrawn release may be retained as diagnostic evidence but cannot enter the paper.

All four final TrialGPT rows must also produce fresh retrieval and System artifacts under rebuilt
0.1.0. Protocol v6's exact query plans remain reusable. Missing evidence, provider failure, timeout,
or resource limits remain missing and never become an eligibility judgment.

# TAIM paper analysis protocol: the admission-to-comparison rule, restated in public terms

**Protocol ID:** `taim-paper-analysis-protocol-2026-09-26-v10`

**Status:** Frozen on 2026-09-26. These bytes are immutable; a later change requires a new dated
amendment. Written under the repository owner's ruling of 2026-09-26 that the rule of protocol v8
be restated in terms a reader holding only the release can resolve, because v8's own bytes are not
part of the release. This amendment adds no rule, removes none, and changes none. It authorizes no
run, no protected-data access, no provider call, and no result admission.

## 1. What this restates

Protocol v8, `taim-paper-analysis-protocol-2026-09-09-v8`, SHA-256
`sha256:51b27f393496f345f8ce5887f4c75bf662c2c88a937238ba179ea9c3372b3ba0`, replaced the
admission-to-comparison bullet of protocol v1 Section 2
([`taim-paper-analysis-protocol-2026-08-29-v1`](paper-analysis-protocol-2026-08-29.md)). v8 remains in
force and is the source of the rule. This document carries the same rule for a reader of the
release.

## 2. The clause

> Admission to comparison: two Primary Rankings enter Default-Track Compare only when their Prepared
> Snapshot ID, Task Input ID, Evaluation Package identity, Benchmark Profile, budget K, **Package
> Artifact ID and Dependency Lock digest** match; Pipeline Depth may differ but must be disclosed.
> **The two added identities are required together because they cover different axes: the Package
> Artifact ID hashes the built code and the Dependency Lock digest hashes the resolved environment,
> and neither implies the other. A release version string is not sufficient in their place - it is
> assigned by hand and can hold constant across either kind of change. Rankings that differ in
> either identity may still be reported side by side, but are not paired, differenced, or entered
> into the significance machinery of protocol v1 Section 6.**

As v8 states, the sentence beginning "The two added identities are required together" is part of
the rule, not commentary on it.

## 3. What differs from v8's text, and why

Three changes of wording, none of substance:

- v8 names the two identities by the field names `run.package_artifact_id` and
  `run.dependency_lock_sha256` of the data model it was written against. This release records them
  under other paths, so they are named here by identity only; the clause itself says what each one
  hashes.
- v8 cites a design record, which the release does not carry, for the disclosure of Pipeline Depth.
  The requirement to disclose stands; the citation is dropped.
- v8's "Section 6" is written here as "protocol v1 Section 6", because v8's clause was written to
  sit inside v1.

v8's remaining sections record why it was written, measured against releases of an earlier
repository. They state no further rule and are not restated.

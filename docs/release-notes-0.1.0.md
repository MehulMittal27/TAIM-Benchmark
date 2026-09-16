# Benchmark Release 0.1.0

Release 0.1.0 is the sole supported public release identity. It replaces the withdrawn release
objects and tags created during release preparation. Runs made against those earlier artifacts
remain diagnostic evidence and cannot be relabelled as final paper runs.

The release provides the executable multi-track benchmark, frozen Profiles and method contracts,
direction-specific run and result-bundle schemas, deterministic reference Systems, external-System
adapters, and the staged TREC 2021 pipeline.

It also includes the complete `trialgpt-three-luna-consensus-v2` retrieval producer for the four
TrialGPT rows admitted by protocols v5 through v7. Protocol v7 retains the 735 frozen query plans by
exact SHA-256. Retrieval and System artifacts must be generated fresh against this release. The
primary TrialGPT System uses 500 candidates and allows up to 1,500 logical System calls per topic.

Prepared-package loading is memory-bounded. On an 18 GB Apple-silicon laptop, the loader validated
the 5.9 GB TREC Clinical Trials 2021 package with 375,580 trials in 394.18 seconds at 79,118,336
bytes peak resident memory and reproduced the frozen Snapshot and Evaluation Package identities.
This measurement covers loading and bounded-Profile projection. It does not claim that dense
full-corpus retrieval, rerankers, or external Systems fit in laptop memory.

The support catalogue declares the staged pipeline's two depth-5,000 component Systems on the
26,149-trial external-fidelity Profile. This closes the staged execution path without changing a
method or adding reportable rows. All seven external-fidelity rows must bind the same 0.1.0
manifest, public tree, package artifact, and dependency lock.

Release 0.1.0 contains no paper results. It excludes protected benchmark data, provider traces,
model weights, dense indexes, concept stores, external repository checkouts, and internal
reproduction and unreleased variants.

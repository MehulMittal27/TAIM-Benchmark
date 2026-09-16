# Reporting policy

The synthetic fixture is installation evidence only. Its scores are not benchmark findings.

Results on real collections are descriptive development evidence unless the analysis plan, Track,
Task, Profile, System, subset rule, metrics, inclusion rule, and reporting tables were frozen before
outcome inspection. A
Published Result Bundle records those admission inputs and states that measured quality was not
used for selection. Only a current-schema, clean-commit run that passes that gate may enter a
release.

A bundle-eligible run receives the frozen release manifest and the exact wheel or source archive at
execution time through `--release-manifest`, `--package-artifact`, and `--dependency-lock`. The CLI
validates the manifest, checks the archive's complete `taim` package file set against it, checks the
executing package against the archive, verifies that `uv.lock` is the manifest-recorded lock, and
binds the installed Python distribution set and runtime platform before deriving the release
manifest, public tree, package artifact, and dependency-environment IDs. It repeats those checks and
the clean Git-state check immediately before writing the run. The bundle builder rejects a missing,
dirty, changed, or mismatched producer identity. Synthetic and full-corpus non-paper development
runs may omit all three producer inputs, but they cannot become Published Result Bundles.
Judgment-union runs and their staged query preparation never permit that omission.

The Evaluation Package contains pooled relevance Judgments, not exhaustive clinical truth. Tables,
figures, captions, and prose must therefore call recall values **judged eligible recall** or
**judged relevant-or-eligible recall**. Versioned JSON artifacts retain their established
`eligible_recall_*` and `relevant_or_eligible_recall_*` field names; those names do not broaden the
denominator beyond the supplied Judgments.

Both TREC 2021 real Profiles preserve the frozen fixed-cutoff
`relevant_or_eligible_precision` field, where labels 1 and 2 are positive. Their scorecards also
reports **eligible precision**, including eligible P@10, with only label 2 positive. This required
additional output does not change the frozen field or its meaning. The TREC 2022 and 2023 real
Profiles report **eligible precision**, with only label 2 positive. SIGIR real Profiles report
fixed-cutoff **referral-candidate precision**, where labels 1 and 2 are positive under the separate
SIGIR referral Judgment Scheme. Synthetic metrics must still be identified as conformance evidence
rather than substituted into a real result.

Release 0.1 TREC 2021 results must use Snapshot Source Recipe
`trec-ct-2021-snapshot-recipe-v3` with definition digest
`sha256:86a437499cb1363df09040154f20c6e5f63931664719d77d764f4ea48b02233a`. Results produced against
the historical campaign Prepared Snapshot ID
`sha256:65a3ecb67c65730f77dcdcb14f70db64e0baed85e0a3df4ced407f464dca0ad3` are not Release 0.1
results and must not be placed in the same comparative table. Every publishable TREC 2021 result
must be rerun after freeze against the immutable Release 0.1 preparation.

Every result states its release identity, Track, Task, Profile, System, Primary Ranking, Pipeline
Depth, Prepared Snapshot ID, Task Input ID, System Input ID, Evaluation Package ID, and budget.
Release runs use portable System Input identity `2.0`; older Run Manifest v7 artifacts without that
field retain the original `1.0` recipe and remain readable, but cannot be admitted as Release 0.1
Published Result Bundles.
The bundle additionally records the capability evidence scope, complete prepared dataset and
Snapshot Source Recipe identity, official Source Bundle identity for real Profiles, the safe System
Input projection, producer identity, and the projected patient/trial entity-version
membership used to validate every ranking row. Changing any of those fields changes the
bundle identity and must fail validation if it no longer matches the committed Task/System Input
identities. For a judgment-union run, the bundle also carries the complete Benchmark Profile pool
projection. Its count, ordered-ID digest, Evaluation Package binding, and content-addressed receipt
ID are revalidated against the projected Task Input membership.
Bundle-eligible RRF requires clean BM25 and BGE-M3 component producers from the same release
identity. Their exact dependency environments may differ, but each must use the same frozen
dependency lock and each identity is retained in the fusion provenance.
Results with different
comparison identities do not share one comparative table.

The release capability matrix offers `bm25`, `dense-bge-m3`, instructed
`dense-qwen3-embedding-0.6b`, and their declared `rrf` reference on each qualified complete and
judgment-union patient-to-trial Profile. Capability does not by itself authorize a paper
experiment. The shipped dated protocol selects the TREC 2021, TREC 2022, TREC 2023, SIGIR
description, and SIGIR summary judgment-union cells as separate result strata. Raw metrics must not
be pooled across Tracks or across the two SIGIR query Profiles. Missing or unauthorized cells
remain missing; they are not imputed from another Track.

A second, clearly separated TREC 2021 table may report the rows declared under
`trec-ct-2021-external-fidelity-26149`. Every row must share its exact caller-supplied membership,
prepared-order receipt, Task Input, Evaluation Package, release, package, and dependency identity.
It must not be combined with or substituted for the 26,162-trial primary table. TrialGPT and
TrialMatchAI paper scores may appear only as non-comparable literature context.
The selected TAIM
adaptations are not exact reproductions of those paper Systems.

TrialGPT rows on TREC 2022 and both SIGIR query Profiles share the existing judgment-union Task
Inputs.
TrialMatchAI rows on TREC 2022, and the later current-fork TrialMatchAI path on TREC 2023, share
the existing judgment-union Task Inputs.
They may be compared only with rows that reproduce the same Track,
Profile, Task Input, Evaluation Package, and pool receipt. Published external pool counts remain
literature context and are not TAIM receipt identities.
Protocol v4 predeclares 33 unique reportable rows across the primary and external-fidelity strata;
a release declares the subset its catalogue lists.

Every judgment-union table, figure, caption, and prose claim must say **within-pool ranking over the
frozen judgment-union corpus**. The union contains every trial ID present anywhere in the Track's
Evaluation Package and is selected without System scores. TREC 2021 requires 26,162 trials; it is
not the historical 26,149-trial TrialGPT common pool. The other exact counts, ordered-ID hashes, and
receipt IDs come from locally derived, pre-effectiveness receipts. Full-corpus and judgment-union
results never share a comparative table, and a judgment-union result is not described as
full-corpus retrieval or an official-run reproduction.

The staged System is a separate TREC 2021 complete-corpus, judgment-union, and external-fidelity
release capability; the primary paper protocol selects the judgment union. Its frozen sequence is
depth-5,000 folded BM25 plus
depth-5,000 no-template Qwen3, RRF at constant 60, a depth-2,000 scoring pool, and the pinned
Qwen3-Reranker-4B top-1,000 Primary Ranking. The component rankings, RRF pool, and full scored
reranker pool are preserved as named forward Stage Rankings. Stage scorecards must be labelled
diagnostic and must identify the stage artifact hash and Pipeline Depth. They cannot be substituted
for the Primary Ranking or mixed into a direction-neutral leaderboard. The raw reranker score
artifact is conformance and provenance evidence until admitted under a predeclared analysis; it is
not itself an effectiveness claim.

Patient-to-trial and trial-to-patient results never share a leaderboard. The TREC 2021 reverse
viability result must bind the Release 0.1 recipe-v3 Prepared Snapshot ID and may be described only
as the frozen complete ten-trial by 75-patient matrix. It is not evidence for a general reverse TREC
benchmark. TREC 2022, TREC 2023, and SIGIR 2016 have no real reverse effectiveness Profile in this
release.

Later small reference runs and final experiments must execute against the immutable public release.
They must bind the exact release manifest, public tree, package, Task Input, System Input,
Evaluation Package, and, for judgment-union Profiles, frozen pool receipt. Subsets must be chosen
without inspecting scores. TREC 2022 or 2023 effectiveness analysis additionally requires the
approved protocol digest before protected topics or qrels are accessed. Packaged fixtures and fixed
parser smokes remain conformance evidence only.

The public repository ships the dated, checksum-identifiable paper protocol. The project tracker
may record later planning, but it cannot silently change the frozen analysis authority. Release
capability never authorizes access to TREC 2022 or 2023 protected inputs without that protocol
identity.

Every row on the external-fidelity Profile requires the separately approved v4 protocol bytes; the
earlier v3 digest cannot authorize it.
Every external-baseline row requires the separately approved v4 protocol bytes. The earlier v3
digest cannot authorize it.
TrialGPT retrieval production additionally requires the frozen v5
producer amendment and its separate provider authorization.
TrialMatchAI execution requires the
pinned checkout and separately licensed runtime inputs.

SIGIR recall names refer to referral judgments under the SIGIR Judgment Scheme. They must not be
reported as eligible recall. Unjudged pairs are not negatives. A zero-gain treatment inside a
declared pooled metric does not convert an unjudged pair into a judgment.

Benchmark effectiveness does not establish patient eligibility, clinical validity, deployment
safety, enrollment performance, or equivalence to an upstream implementation.

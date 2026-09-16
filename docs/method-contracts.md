# Reference method contracts

TAIM fixes the executable comparison contract. Systems remain replaceable and choose their own
query representation, retrieval, fusion, reranking, eligibility logic, and other matching methods.
The support catalog lists only Systems that this release can execute through the public interface.

The Systems listed in the release catalogue produce Primary Rankings on the Profiles the catalogue
declares for each.
Release-selected external Systems use the Profiles the catalogue declares for each.
Each System's Profile eligibility is declared in the catalogue.
Capabilities are enumerated by `trial-benchmark support list`; no full
Cartesian product is implied. Every evaluation consumes a direction-specific, checksum-bound
Evaluation Package.

Each `System` implements `normalize_options()` as part of its public contract. Unknown method
options are errors, so misspelled configuration cannot disappear from the content-addressed System
input. `StrictSystemOptions` provides the default exact-key implementation; a System may override
it when values need additional normalization. Paths, caches, providers, and component-run assembly
belong to eval-side adapters and are not method options.

## BM25

`bm25` indexes `TrialDocument.canonical_text` and queries with
`BenchmarkTopic.canonical_text`. It case-folds text and extracts `[a-z0-9]+` tokens. Defaults are
`k1 = 1.2` and `b = 0.75`. For corpus size `N`, document frequency `df`, term frequency `tf`, query
term frequency `qtf`, document length `dl`, and average document length `avgdl`, it uses:

```text
idf = ln(1 + (N - df + 0.5) / (df + 0.5))
term = qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
score = sum(term)
```

The index is an exact in-memory inverted index. Ranking is score descending, then trial ID
ascending. Every corpus document participates, including zero-score documents.

## BGE-M3

`dense-bge-m3` pins `BAAI/bge-m3` model and tokenizer revision
`5617a9f61b028005a4858fdac845db406aefb181`. Patient and trial canonical text are passed unchanged.
The encoder uses a maximum length of 8,192, right truncation, dynamic right padding, CLS pooling,
float32 model and output values, and L2 normalization. Retrieval is exact inner product over the
checksum-verified float32 matrix, which is cosine similarity after normalization. Ties use trial ID
ascending.

## Qwen3-Embedding-0.6B

`dense-qwen3-embedding-0.6b` pins `Qwen/Qwen3-Embedding-0.6B` model and tokenizer revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Trial canonical text is unchanged. Patient canonical
text is formatted as:

```text
Instruct: Given a patient description, retrieve clinical trials for which the patient may be eligible.
Query: {text}
```

The encoder uses a maximum length of 8,192, right truncation, dynamic left padding, last-token
pooling, bfloat16 model values, float32 output, and encoder L2 normalization. Retrieval and ties use
the same exact policy as BGE-M3. The implementation verifies the pinned model and tokenizer file
hashes before use.

## Folded BM25

`bm25-folded` uses the frozen BM25 implementation but replaces each patient query with a
checksum-bound folded-query bundle. The bundle is derived locally from the exact Task Input using
medspaCy 1.3.1 ConText rules and a user-supplied SNOMED CT description file. Only asserted topic
spans, age, and sex are retained; duplicate source spans remain duplicated. No SNOMED description
or concept identifier enters the bundle. The bundle records the Task Input ID, medspaCy and spaCy
versions, packaged-rule hash and count, SNOMED release name, and description-file SHA-256.

## No-template Qwen3

`dense-qwen3-embedding-0.6b-no-template` uses the same pinned model, tokenizer, pooling, numeric,
normalization, and exact-retrieval policy as the instructed Qwen3 reference System. Its patient
query format is exactly `{text}`. It is a separate System Input and is not an option that silently
changes the main-matrix Qwen3 System.

## Reciprocal rank fusion

`rrf` consumes complete top-1,000 `bm25` and `dense-bge-m3` runs with matching task, prepared
Snapshot, Evaluation Package, Benchmark Profile, and budget identities. It ignores component scores
and sums `1 / (60 + component_rank)` with exact rational arithmetic. Output ranks use fused score
descending, then trial ID ascending. RRF does not consume the Qwen run. For a bundle-eligible
fusion, both component runs must have clean producers, the same verified release identity as the
fusion process, and dependency environments bound to the same manifest-recorded `uv.lock`.

## Staged paper System

`staged-bm25-qwen3-rrf-rerank` is declared in
`staged-bm25-qwen3-rrf-rerank-v1.json`. It accepts only complete depth-5,000 `bm25-folded` and
`dense-qwen3-embedding-0.6b-no-template` component runs for the same TREC 2021 `official-full` or
`trec-ct-2021-judgment-union` or `trec-ct-2021-external-fidelity-26149` Task Input, Evaluation
Package, Profile, Prepared Snapshot ID, and folded-query bundle. Bundle-eligible assembly
additionally requires clean component producers from
the same release and dependency lock.

The System ignores component scores and applies RRF with constant 60 at component depth 5,000.
Fused scores descend, with trial ID as the deterministic tie-break. The top 2,000 candidates per
topic form the reranker scoring pool. The reranker pins `Qwen/Qwen3-Reranker-4B` model and tokenizer
revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, float16 weights, batch size 16, maximum length
2,048, and a folded-query/F-summary input. F-summary contains condition, brief title, intervention,
and eligibility-provenance lines; a title-or-first-line fallback is recorded. All 2,000 pairs are
scored. Output ordering is score descending, then fused rank ascending, and the Primary Ranking is
the top 1,000.

The run stores four direction-specific Stage Rankings: both depth-5,000 components, the depth-2,000
RRF pool, and the complete depth-2,000 reranked pool. Stage evaluation is explicit and diagnostic;
none is a neutral ranking artifact or an alternative Primary Ranking. The raw score artifact is
bound to the ordered fused pairs and the exact rendered query/document text. Its resumable
checkpoint is valid only at frozen batch boundaries. The selected attention backend is the sole
enabled SDPA kernel, so execution fails rather than silently dispatching another backend. Runtime
provenance records that executed backend, the batch-composition hash, CUDA stack, GPU, peak memory,
truncation count, and checkpoint-resume count.

## Reverse BM25

`bm25-trial-to-patient` indexes `PatientEntityVersion.topic.canonical_text` and queries with
`TrialVersion.trial.canonical_text`. It uses the same frozen BM25 formula and defaults, with score
ties resolved by patient ID. Its Primary Ranking contains patient candidates keyed by trial query.
It cannot consume a patient-to-trial Task Input or emit a patient-to-trial run.

## Profile admission

Every release run and Published Result Bundle carries its capability evidence scope, prepared
dataset ID, Snapshot Source Recipe ID and checksum, and full checksum metadata for the Source
Bundle. Real Profiles are accepted only when the prepared dataset and recipe identify the complete
Track preparation, the source metadata equals the packaged official Track lock, and the resulting
Snapshot has the frozen full-preparation topic and trial counts: TREC 2021 75/375,580, TREC 2022
50/375,580, TREC 2023 40/451,538, and either SIGIR query Profile 60/204,855. Parser-smoke or
truncated preparations cannot enter an effectiveness Profile.

A judgment-union Profile then derives its effective Task Input from every and only trial ID present
in that complete preparation's Evaluation Package. Prepared-corpus order is retained. Before
scoring, `trial-benchmark data pool inspect` emits the Profile definition hash, Prepared Snapshot
ID, Evaluation Package ID, selected count, ordered-ID digest, and content-addressed pool receipt ID.
Every run must reproduce all three pool pins. The writer then checks that the Task Input contains
every and only Evaluation Package trial ID. TREC 2021 additionally requires 26,162 selected trials.
Other Tracks do not carry a guessed count in the release; their exact locally derived receipts are
frozen before effectiveness execution. The source Snapshot stays complete while the Task Input
records the smaller effective corpus and its distinct identity.

The external-fidelity Profile instead takes a checksum-verified caller-supplied 26,149-ID
membership from the official TrialGPT corpus distribution. It verifies that every ID is present in
the complete recipe-v3 preparation and Evaluation Package, then orders the membership by the
prepared corpus and freezes a separate receipt. Its direction-specific Evaluation Package retains
only source judgment rows for trials in that Task Input, preserves the source package identity in
provenance, and leaves absent pairs unjudged. It cannot reuse the 26,162-trial judgment-union receipt
or the historical recipe-v1 TrialGPT Snapshot.

## Release inputs and run integrity

The release streams verified trial records from the prepared package during validation and
bounded-Profile projection instead of retaining the full structured corpus in memory.

No upstream checkout, provider response,
model weight, concept store, index, or historical result enters the release tree.

Synthetic Profiles also require their exact canonical fixture Recipe, Source Bundle, and
record counts. A bundle-eligible run
derives the release manifest, public tree, and package artifact IDs by checking the supplied frozen
manifest and distribution archive against the executing package. It also verifies that the supplied
`uv.lock` is the manifest-recorded lock, checks every installed locked distribution version, and
records a content-addressed environment projection containing the interpreter, platform, machine,
and complete installed distribution set. These producer checks run before execution and again
immediately before the run is written; a changed archive, environment, commit, or dirty state aborts
the write. Wheel and source-distribution admission also binds the
reviewed project identity, README description, dependency and extra sets, license file, CLI entry
point, and pure-Python wheel tag; undeclared package metadata or payload fails closed. Bundles also
carry a hash-bound, redistributable System Input
projection: effective options, capability declarations, model identity, and index identity, with
caller-local path fields explicitly listed as omitted. Path-shaped keys are removed
recursively, including inside lists. Release runs use portable System Input identity `2.0`, which
omits those machine-local path values from the identity while retaining every effective method
option. The bundle also projects the Task Input's sorted patient and trial entity-version
membership; every direction-specific ranking must contain the contiguous declared depth for each
query and every row must belong to those sets. The System Input ID
commits that membership, identity version, effective method options, cutoffs, and capability
declarations, so rehashing a substituted bundle cannot retain the original input IDs.

The synthetic reverse Profile preserves absent pairs as unknown. The real
`trec-ct-2021-reverse-complete10` Profile instead requires the exact complete 750-pair matrix. Its
ten trial IDs, source checksums, label counts, transposition policy, and denominator are frozen in
the public contract. It is bound to Release 0.1 Prepared Snapshot ID
`sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8` and Evaluation Package ID
`sha256:fa47e8e588e06a0d69f3fc32ea34f59113c875bdc8219327ce8faaa8930e9dba`. Labels retain their
original meanings: 0 not relevant, 1 excluded, 2 eligible.

## External Systems

`TrialGPT-TAIM-Luna-v1` consumes a fresh, lock-bound depth-2,000 retrieval ranking over the exact
selected Task Input, takes the top 500 per topic, and applies the TAIM-controlled
failure-aware TrialGPT matching and aggregation path through the separately authorized
`gpt-5.6-luna` provider. Its clean-commit publication contract binds the Prepared Snapshot,
Evaluation Package, retrieval lock and artifact, provider configuration, method hashes, workers,
and every selected topic before a provider call. Paper execution uses exactly 48 generation
workers. The public contract supports TREC 2021, TREC 2022, and the two explicit SIGIR query
Profiles, but not TREC 2023. It is a TAIM adaptation, not the upstream paper's GPT-4 System.

Release 0.1.0 freezes the retrieval producer as `trialgpt-three-luna-consensus-v2`. Three
`gpt-5.6-luna` query plans feed `taim-controlled-bm25-v2` and pinned MedCPT channels at depth
2,000; fixed condition-weighted RRF and a fixed cross-arm consensus produce the final ranking.
BM25 v2 counts document frequency once per document and is not interchangeable with the older v1
implementation. The producer guide records the exact revisions, call budget, cluster command, and
retained evidence.

Protocol v7
binds fresh retrieval and System artifacts to the 0.1.0 release while retaining the exact completed
query plans by their unchanged provider-call identities.

Release-selected TrialMatchAI Systems invoke a clean external checkout at commit
`7eba8f399336fcd988b00ce98f2e025e5ec04119`.
The adapter verifies the config digest, model and LoRA
identities, adapter weight hashes, concept-store manifest, search database, all effective
environment overrides, and output membership before normalizing the ranking. Its entity linking
and rebuilt concept store differ from the published paper implementation, so it is not an exact
paper reproduction.
The TREC 2023 identity denotes the later current-repository path rather than a
claim that the TrialMatchAI paper evaluated TREC 2023.

Release-selected external Systems require separately acquired inputs.

## Track query and label contracts

TREC 2021 retains `official-full` evaluation semantics and applies the same label and metric
semantics to `trec-ct-2021-judgment-union`. Release 0.1 prepares its source corpus with
Snapshot Source Recipe `trec-ct-2021-snapshot-recipe-v3`, whose definition digest is
`sha256:86a437499cb1363df09040154f20c6e5f63931664719d77d764f4ea48b02233a`. The recipe includes
typed `[MINIMUM_AGE]`, `[MAXIMUM_AGE]`, `[SEX]`, and `[HEALTHY_VOLUNTEERS]` lines when those source
fields are present. This is a new Release 0.1 prepared-corpus identity. It is not equivalent to the
historical campaign Prepared Snapshot ID
`sha256:65a3ecb67c65730f77dcdcb14f70db64e0baed85e0a3df4ced407f464dca0ad3`, and preserving the
`official-full` Profile does not make results across those two corpus identities comparable.

TREC 2022 and 2023 use the separately named `confirmation-full` Profile. TREC 2023 questionnaire
queries use the frozen template-and-field rendering, and trial alias repairs are limited to
source-authored NCT aliases. Their judgment-union Profiles preserve those query, label, and parsing
contracts while changing only the effective trial membership.

SIGIR 2016 exposes separate `description` and `summary` query Profiles. Its labels are referral
judgments, not eligibility judgments: 0 would not refer, 1 would consider referral after further
investigation, and 2 is highly likely to refer. Unjudged pooled pairs remain unjudged for every
Track. Each query view has its own separately named judgment-union Profile.

Both TREC 2021 real Profiles preserve fixed-cutoff `relevant_or_eligible_precision`, with labels 1
and 2 positive. Their scorecards also report the required `eligible_precision`, including eligible
P@10, with only label 2 positive. The TREC 2022 and 2023 real Profiles report fixed-cutoff
`eligible_precision`, with label 2 as the sole precision positive. Real SIGIR Profiles report
fixed-cutoff `referral_candidate_precision`, with labels 1 and 2 positive under the referral
scheme.

## Claim limits

The method identities cover these implementations and protocol inputs. They do not claim bitwise
equivalence to upstream examples, superiority outside the declared benchmark, patient eligibility,
clinical safety, or deployment validity. Runtime measurements exclude prepared-data loading,
evaluation, and artifact serialization unless a run states a narrower scope. Judgment-union
results are within-pool rankings, not full-corpus retrieval results or official-run reproductions.

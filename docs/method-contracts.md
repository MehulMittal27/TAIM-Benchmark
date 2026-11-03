# Reference method contracts

TAIM fixes the executable comparison contract. Each System chooses its own query representation,
retrieval, fusion, reranking, eligibility logic, and other matching methods.
The support catalog lists only Systems that this release can execute through the public interface.
The public CLI's run validation, run evaluation, and Result Bundle validation refuse any other System.
A pipeline outside the catalog is scored through the separate `submission` commands in
[Score your own pipeline](external-pipelines.md). Its ranking is checked against the same Task
Input and scored by the same evaluation rules, and it stays a submission, never a catalog System.

The Systems listed in the release catalogue produce Primary Rankings on the Profiles the catalogue
declares for each.
Release-selected external Systems use the Profiles the catalogue declares for each.
Each System's Profile eligibility is declared in the catalogue.
Capabilities are enumerated by `trial-benchmark support list`; no full
Cartesian product is implied. Every evaluation consumes a direction-specific, checksum-bound
Evaluation Package.

Each `PatientToTrialSystem` implements `normalize_options()` as part of its public contract. Unknown method
options are errors, so misspelled configuration cannot disappear from the content-addressed System
input. `StrictSystemOptions` provides the default exact-key implementation; a System may override
it when values need additional normalization. Paths, caches, providers, and component-run assembly
belong to eval-side adapters and are not method options.

## BM25

`bm25` indexes `Trial.canonical_text` and queries with
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

Hardware: CPU only. The TREC 2021 judgment-union run peaked at about 9 GB of memory on a laptop.
Run the complete-corpus `official-full` Profile on a server: on a 24 GB laptop it passed a 43 GB
memory footprint without finishing.

## BGE-M3

`dense-bge-m3` pins `BAAI/bge-m3` model and tokenizer revision
`5617a9f61b028005a4858fdac845db406aefb181`. Patient and trial canonical text are passed unchanged.
The encoder uses a maximum length of 8,192, right truncation, dynamic right padding, CLS pooling,
float32 model and output values, and L2 normalization. Retrieval is exact inner product over the
checksum-verified float32 matrix, which is cosine similarity after normalization. Ties use trial ID
ascending.

Hardware: `--device cpu`, the default, works, but a real Profile needs `--device cuda`.
CUDA needs compute capability 7.5 or newer: the locked torch 2.13.0 is built for CUDA 13 with
kernels for `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, and `sm_120`, so a V100 (7.0) cannot
run it. The run names an unsupported card and its capability, and refuses before any model loads.
The float32 weights take about 2.3 GB of GPU memory, and encoding batches are sized to a 2 GiB
attention-score budget. This System has run on A40, A100, and H200 cards.

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

Hardware: as for BGE-M3, with bfloat16 weights of about 1.2 GB. Its bfloat16 scores differ slightly
between card models, so record the card beside a result. The TREC 2021 judgment-union run peaked at
9.1 GB of host memory on an A40.

## Folded BM25

`bm25-folded` uses the frozen BM25 implementation but replaces each patient query with a
checksum-bound folded-query bundle. The bundle is derived locally from the exact Task Input using
medspaCy 1.3.1 ConText rules and a user-supplied SNOMED CT description file. Only asserted topic
spans, age, and sex are retained; duplicate source spans remain duplicated. No SNOMED description
or concept identifier enters the bundle. The bundle records the Task Input ID, medspaCy and spaCy
versions, packaged-rule hash and count, SNOMED release name, and description-file SHA-256.

**Licensed input (required).** You must supply a SNOMED CT RF2 release directory that contains
`Snapshot/Terminology/sct2_Concept_Snapshot*.txt` and `sct2_Description_Snapshot-en*.txt`.
`bm25-folded` is qualified on the International Edition released 2026-08-01, identified by the
SHA-256 of its description file in `src/taim/data/locks/snomed-ct-folded-query-lexicon.json`. A
different edition or date produces a different query bundle and System Input, which is not
comparable, so `queries prepare` refuses it before loading the corpus. Obtain SNOMED CT from your
National Release Centre or SNOMED International (MLDS). This repository does not ship SNOMED CT
because its licence does not permit us to redistribute it, and no SNOMED CT content enters a query
bundle.

Hardware: CPU only, with the memory needs of BM25.

## No-template Qwen3

`dense-qwen3-embedding-0.6b-no-template` uses the same pinned model, tokenizer, pooling, numeric,
normalization, and exact-retrieval policy as the instructed Qwen3 reference System. Its patient
query format is exactly `{text}`. It is a separate System Input and is not an option that silently
changes the main-matrix Qwen3 System.

Hardware: as for Qwen3-Embedding-0.6B.

## Reciprocal rank fusion

`rrf` consumes complete top-1,000 `bm25` and `dense-bge-m3` runs with matching task, prepared
Snapshot, Evaluation Package, Benchmark Profile, and budget identities. It ignores component scores
and sums `1 / (60 + component_rank)` with exact rational arithmetic. Output ranks use fused score
descending, then trial ID ascending. RRF does not consume the Qwen run. For a bundle-eligible
fusion, both component runs must have clean producers, the same verified release identity as the
fusion process, and dependency environments bound to the same manifest-recorded `uv.lock`.

Hardware: CPU only. Its BGE-M3 component run has the GPU needs.

## Staged paper System

`staged-bm25-qwen3-rrf-rerank` is declared in
`staged-bm25-qwen3-rrf-rerank-v1.json`. It accepts only complete depth-5,000 `bm25-folded` and
`dense-qwen3-embedding-0.6b-no-template` component runs for the same Task Input, Evaluation Package,
Profile, Prepared Snapshot ID, and folded-query bundle, on a Profile its support rows name. The
declaration names the TREC 2021 Profiles its recipe was frozen on and is the same on every Track,
so its definition SHA-256 is too; a run records its own Track. Bundle-eligible assembly
additionally requires clean component producers from the same release and dependency lock.

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
bound to the ordered fused pairs and the exact rendered query/document text. It is schema 2.0 from
Release 0.3.0, and a Release 0.2.0 artifact (schema 1.0, which names no kernels) is refused by name.
Its resumable checkpoint is valid only at frozen batch boundaries. The selected attention backend is
the one SDPA kernel enabled besides cuDNN, which stays as torch starts it. Runtime provenance
records all four SDPA flags, the kernels the run's own batches ran with batch counts, read from
torch's raw profiler event list (resumed pairs are counted by the checkpoint-resume count),
batch-composition hash, CUDA stack, GPU, peak memory and truncation.

Hardware: the reranker runs only on CUDA, with compute capability 7.5 or newer under the locked
torch 2.13.0, so a V100 (7.0) cannot run it; the run names an unsupported card and refuses before
downloading the model, as it does for a torch whose raw profiler event list it cannot read. Its
float16 weights take about 8 GB of GPU memory. PyTorch provides the `flash` backend only from
compute capability 8.0. The components have the hardware needs of their own Systems.

## Reverse BM25

`bm25-trial-to-patient` indexes `PatientEntityVersion.topic.canonical_text` and queries with
`Trial.canonical_text`. It uses the same frozen BM25 formula and defaults, with score
ties resolved by patient ID. Its Primary Ranking contains patient candidates keyed by trial query.
It cannot consume a patient-to-trial Task Input or emit a patient-to-trial run.

Hardware: CPU only.

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
workers. The public contract supports TREC 2021, TREC 2022, TREC 2023, and the two explicit SIGIR
query Profiles. It is a TAIM adaptation, not the upstream paper's GPT-4 System.

The TrialGPT-TAIM-Luna-v1 row searches upstream TrialGPT's TREC 2021 corpus, which is drawn from
the TREC 2021 judged union: 26,149 trials, each judged for at least one topic, 7.0% of the
375,580-trial collection. Its retrieval is TREC-2021-calibrated, in-sample development evidence;
effectiveness claims come only from held-out tracks.

Relative to upstream TrialGPT 1.0 (commit `1b3242c`), `TrialGPT-TAIM-Luna-v1` keeps the method,
the keyword prompt, upstream's criterion parser, the user prompts, the scoring arithmetic and
upstream's 26,149-trial TREC 2021 pool, and differs in ten ways: TAIM reimplements every stage and
runs no upstream code; the model is Codex `gpt-5.6-luna`, not GPT-4; retrieval is a three-query
Luna consensus chosen from ten arms by judged recall on the same 75 TREC 2021 topics the row
reports; both matching system prompts add a sentence requiring sentence IDs as a JSON array of
non-negative integers, and a sentence stating that there are exactly N criteria; the aggregation
system prompt drops upstream's definitions of what E=-R, E=R and E=0 mean; aggregation output is
validated and repaired instead of scoring 0 when it is invalid; a trial whose criterion
matching still fails after three attempts ranks as an abstention at -3.0; a criterion answer
that leaves out its sentence list counts with its label and no sentences cited, where upstream skips
the criterion; and the parser reads inclusion and exclusion sections TAIM cuts from the complete
ClinicalTrials.gov text, not NCBI's pre-separated fields. The model, the retrieval choice, the
dropped definitions and the criterion input could change results; the other six mainly make the
run reliable. The reimplementation, the model, the exactly-N sentence, the dropped definitions and
the replaced score-0 fallback have no recorded reason for the deviation; the method spec states the
design of the reimplementation and the model.

Hardware: matching runs through the provider. Only the retrieval producer's MedCPT encoders run
locally, and on CUDA they need compute capability 7.5 or newer under the locked torch 2.13.0, so a
V100 (7.0) cannot run them; the producer refuses an unsupported card before they load.

Release 0.1 freezes the retrieval producer as `trialgpt-three-luna-consensus-v2`. Three
`gpt-5.6-luna` query plans feed `taim-controlled-bm25-v2` and pinned MedCPT channels at depth
2,000; fixed condition-weighted RRF and a fixed consensus across query-plan variants produce the
final ranking.
BM25 v2 counts document frequency once per document and is not interchangeable with the older v1
implementation. The producer guide records the exact revisions, call budget, cluster command, and
retained evidence.

Protocol v7 requires fresh retrieval and System artifacts while retaining the exact completed query
plans by their unchanged provider-call identities.

Release-selected TrialMatchAI Systems invoke a clean external checkout at commit
`99aa499e7296ebde836a0ebf00a5ed4948892a83`.
The adapter verifies the config digest, model and LoRA
identities, adapter weight hashes, concept-store manifest, search database, all effective
environment overrides, and output membership before normalizing the ranking. Its entity linking
and rebuilt concept store differ from the published paper implementation, so it is not an exact
paper reproduction. Its hardware needs are in the
[TrialMatchAI external baseline guide](trialmatchai-external-baseline.md).

Relative to upstream TrialMatchAI `v0.7.0` (commit `071a76e`), the pinned fork and its
configuration change outputs in twelve ways: Phi-4 serves 4-bit NF4 weights where upstream
serves bfloat16; the Gemma-2 reranker serves 4-bit NF4 weights where upstream serves
unquantized weights; both engines use an fp8 KV cache; eligibility uses guided JSON decoding
with bounded whitespace, which upstream leaves off; age and sex hard filters fire, which
upstream never fires on a text-note patient; Phi-4's context and output ceilings are 16,384
and 12,288 tokens instead of 8,192 and 5,000; eligibility assesses 20 trials per topic where
upstream's TREC preset assesses 250; query expansion is on, as in upstream's TREC preset but
not in its general configuration; concept linking reranks 50 candidates instead of 10;
concept search pins ANN probes at 64 and the refine factor at 4; the concept store is a
rebuild searched through an approximate IVF-PQ vector index; and eligibility output repairs a
missing JSON comma and retries invalid outputs individually. The 4-bit weights and fp8 KV
cache came in fork commit `263a227` with an empty commit message and were never a recorded
decision.
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

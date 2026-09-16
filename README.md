# TAIM benchmark

This tree is an immutable benchmark release. Its identity is recorded in `RELEASE-PROVENANCE.json`
and the release manifest.

TAIM is an executable benchmark for clinical trial matching. It fixes data and method identities,
the public execution path, evaluation rules, validation, and result bundles. Matching pipelines are
replaceable Systems evaluated through that contract; TAIM does not prescribe one pipeline.

The benchmark is multi-track and bidirectional. Its public capability catalog covers TREC
Clinical Trials 2021, 2022, and 2023 and SIGIR Clinical Trials 2016. The `patient_to_trial` and
`trial_to_patient` Tasks have separate inputs, runs, evaluators, and claim limits. TAIM reuses these
existing test collections; it does not introduce a new dataset or redistribute benchmark source
files.

## Quick start

Install the locked base environment and inspect the machine-readable support matrix:

```console
uv sync --locked
. .venv/bin/activate
trial-benchmark support list
```

Run and evaluate the patient-to-trial fixture:

```console
trial-benchmark patient-to-trial fixture run \
  --track trec-ct-2021 \
  --run-id quick-start-patient-to-trial \
  --output-dir runs \
  --top-k 8
trial-benchmark patient-to-trial run validate \
  --run-dir runs/quick-start-patient-to-trial
trial-benchmark patient-to-trial run evaluate \
  --run-dir runs/quick-start-patient-to-trial
```

Run the separate trial-to-patient fixture:

```console
trial-benchmark trial-to-patient fixture run \
  --track trec-ct-2021 \
  --run-id quick-start-trial-to-patient \
  --output-dir runs \
  --top-k 3
trial-benchmark trial-to-patient run validate \
  --run-dir runs/quick-start-trial-to-patient
trial-benchmark trial-to-patient run evaluate \
  --run-dir runs/quick-start-trial-to-patient
```

Both direction-specific validate and evaluate commands print the exact release, Track, Task,
Profile, System, Prepared Snapshot ID, Task Input ID, System Input ID, and Evaluation Package ID.

Both fixtures contain invented records. They cover eligible, excluded, irrelevant, unknown, and
mixed-direction failure behavior. Their scores are conformance evidence, not effectiveness results.

## Real tracks

Acquire each Track locally using its guide, then prepare checksum-verified files:

Real preparation and execution are bound to the packaged Track source lock. The public CLI has no
override that can substitute an unofficial lock under a real effectiveness Profile.

TREC 2021 Release 0.1 preparation uses `trec-ct-2021-snapshot-recipe-v3`. It is a new immutable
prepared-corpus identity and is not interchangeable with historical TAIM campaign Snapshots. Final
publishable runs must be produced from this release after freeze. Complete-source preparation is
shared by the full-corpus and compute-bounded Profiles; only the effective Task Input differs.

```console
uv run trial-benchmark data prepare \
  --track trec-ct-2021 \
  --source /path/to/official-files \
  --output-dir prepared/trec-ct-2021
uv run trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 \
  --profile official-full \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --system bm25 \
  --run-id bm25-official-full \
  --output-dir runs
```

The `official-full` command above is the complete 375,580-trial capability. The paper's separately
named TREC 2021 Profile ranks the score-blind union of all trial IDs present in the Evaluation
Package. Inspect and freeze its identity before producing any System score:

```console
PROTOCOL_ID="sha256:$(shasum -a 256 docs/paper-analysis-protocol-2026-08-31-v3.md | awk '{print $1}')"
trial-benchmark data pool inspect \
  --track trec-ct-2021 \
  --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --protocol-approval-id "$PROTOCOL_ID" \
  > prepared/trec-ct-2021-judgment-union-receipt.json
```

Review and freeze that receipt before running. It must report 26,162 trials. Pass its exact
`pool_count`, `pool_ids_sha256`, and `pool_receipt_id` values to every command in the stratum:

```console
trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 \
  --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --pool-count 26162 \
  --pool-ids-sha256 sha256:<digest-from-receipt> \
  --pool-receipt-id sha256:<id-from-receipt> \
  --protocol-approval-id "$PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --system bm25 \
  --run-id bm25-trec-2021-judgment-union \
  --output-dir runs
```

SIGIR preparation requires `--query-profile description` or `--query-profile summary`, and the run
must name the matching full or judgment-union Profile. TREC 2022 and 2023 full-corpus runs use
`confirmation-full`; their paper runs use their separately named judgment-union Profiles.
Effectiveness work on either protected Track requires the frozen protocol identity before topics or
qrels are accessed.

Use the same `data pool inspect` gate for the remaining paper Profiles, substituting the prepared
directory below. The receipt, rather than a guessed count, supplies all three run pins.

| Track/view | Judgment-union Profile | Complete source Profile |
| --- | --- | --- |
| TREC 2022 | `trec-ct-2022-judgment-union` | `confirmation-full` |
| TREC 2023 | `trec-ct-2023-judgment-union` | `confirmation-full` |
| SIGIR description | `sigir-ct-2016-description-judgment-union` | `description` |
| SIGIR summary | `sigir-ct-2016-summary-judgment-union` | `summary` |

```console
trial-benchmark data pool inspect \
  --track <track> \
  --profile <judgment-union-profile> \
  --data-dir <prepared-directory> \
  --protocol-approval-id "$PROTOCOL_ID"
```

## Paper-ready release capabilities

This release makes `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, and `rrf` executable on both
the complete-corpus and judgment-union Profile declared for each real forward Track/query view.
The complete Profiles remain public capabilities. The frozen
[paper protocol](docs/paper-analysis-protocol-2026-08-31-v4.md) selects the five judgment-union strata for the
paper: TREC 2021, TREC 2022, TREC 2023, SIGIR description, and SIGIR summary. Each Track × Task ×
Profile × System row remains separate, and raw metrics are not pooled across Tracks. TREC 2022 and
2023 remain inaccessible to an execution worker until the protocol digest is supplied.

The benchmark also contains one TREC 2021-only staged System for the paper. It is declared for the
TREC 2021 complete, judgment-union, and separate external-fidelity Profiles, but the primary paper
protocol selects the judgment-union Profile. It is not declared for the other Tracks:

1. folded-query BM25 at depth 5,000;
2. no-template Qwen3-Embedding-0.6B at depth 5,000;
3. two-arm RRF with constant 60 and a 2,000-pair-per-topic scoring pool;
4. `Qwen/Qwen3-Reranker-4B` revision
   `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, scoring all 2,000 pairs and emitting a top-1,000
   Primary Ranking.

Run the staged System under CPython 3.11 or 3.12 because the licensed local folding toolchain is not
declared for newer Python versions. SNOMED CT is supplied by the user and is never copied into an
output. Full-corpus dense retrieval and reranking belong on a suitable compute node.

```console
uv sync --locked --extra folding

trial-benchmark patient-to-trial queries prepare \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --profile trec-ct-2021-judgment-union \
  --pool-count 26162 \
  --pool-ids-sha256 sha256:<digest-from-receipt> \
  --pool-receipt-id sha256:<id-from-receipt> \
  --protocol-approval-id "$PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --snomed-release /path/to/licensed/SnomedCT_InternationalRF2 \
  --output derived/trec-2021-folded-queries.json
```

```console
uv sync --locked --extra folding

trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --pool-count 26162 --pool-ids-sha256 sha256:<digest-from-receipt> \
  --pool-receipt-id sha256:<id-from-receipt> \
  --protocol-approval-id "$PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --system bm25-folded --top-k 5000 \
  --folded-query-bundle derived/trec-2021-folded-queries.json \
  --run-id paper-bm25-folded-d5000 --output-dir runs
```

```console
uv sync --locked --extra dense

trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --pool-count 26162 --pool-ids-sha256 sha256:<digest-from-receipt> \
  --pool-receipt-id sha256:<id-from-receipt> \
  --protocol-approval-id "$PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --system dense-qwen3-embedding-0.6b-no-template --top-k 5000 \
  --index-dir indexes/qwen3-no-template \
  --device cuda \
  --run-id paper-qwen3-no-template-d5000 --output-dir runs
```

```console
uv sync --locked --extra dense --extra staged

trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --pool-count 26162 --pool-ids-sha256 sha256:<digest-from-receipt> \
  --pool-receipt-id sha256:<id-from-receipt> \
  --protocol-approval-id "$PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --system staged-bm25-qwen3-rrf-rerank --top-k 1000 \
  --folded-query-bundle derived/trec-2021-folded-queries.json \
  --folded-bm25-run runs/paper-bm25-folded-d5000 \
  --no-template-qwen-run runs/paper-qwen3-no-template-d5000 \
  --reranker-score-artifact derived/paper-reranker-scores.json \
  --reranker-checkpoint derived/paper-reranker-checkpoint.json \
  --reranker-attention-backend flash \
  --run-id paper-staged-pipeline --output-dir runs
```

The final command preserves the two component rankings, the RRF scoring pool, the complete scored
reranker pool, and the top-1,000 Primary Ranking. It forces the selected backend as the sole enabled
SDPA kernel and records it with the batch schedule,
model and tokenizer revision, rendered-input identity, checkpoint-resume count, and score-artifact
hash. The checkpoint is batch-aligned and may only resume the identical scoring input. Evaluate the
Primary Ranking normally or a named stage explicitly:

```console
trial-benchmark patient-to-trial run evaluate \
  --run-dir runs/paper-staged-pipeline
trial-benchmark patient-to-trial run evaluate \
  --run-dir runs/paper-staged-pipeline \
  --stage rrf-scoring-pool-depth2000
```

A stage scorecard is marked `stage_diagnostic`; it does not replace the declared Primary Ranking.
Judgment-union commands require the receipt's `--pool-count`, `--pool-ids-sha256`, and
`--pool-receipt-id` plus `--release-manifest`, `--package-artifact`, and `--dependency-lock` before
they can produce queries or scores. The component and fusion producers must resolve to the same
immutable release and lock identity. The mutable paper plan and its approval state remain in the
project tracker. The dated paper protocol ships in this repository. The CLI accepts only the
SHA-256 of those exact protocol bytes. Its Track and method choices are frozen, but each locally
derived pool receipt and the eventual immutable release identities must still be recorded before
paper execution.

## External baseline strata

The benchmark also prepares a separate TREC 2021 comparison over the exact TrialGPT 26,149-trial
membership, using the Systems its catalogue declares on that Profile.
Release-selected external Systems are compared beside the TAIM rows on that Profile.
This stratum has its
own Profile, Task Input, receipt, protocol digest, and reporting table; it is not the 26,162-trial
primary TREC 2021 stratum.

The public package verifies the external corpus checksum, extracts an ID-only membership file, and
rejects historical or identity-mismatched artifacts. It ships no external checkout, benchmark source
bytes, provider outputs, model weights, concept store, or index.
For TrialGPT, it also preflights fresh inputs. Its commands and runtime prerequisites are in the
[TrialGPT external baseline guide](docs/trialgpt-external-baseline.md).
For TrialMatchAI, it also executes the selected adapter. Its commands, runtime prerequisites, and
license boundaries are in the
[TrialMatchAI external baseline guide](docs/trialmatchai-external-baseline.md).
The strata and their reporting boundary are in the
[external baseline guide](docs/external-baselines.md).

The release includes the memory-bounded prepared-package loader.
For TrialGPT, the release includes the complete, pinned retrieval producer. Its admitted targets,
reusable query-generation calls, MedCPT revisions, execution command, and evidence boundary are in the
[TrialGPT retrieval producer guide](docs/trialgpt-retrieval-producer.md). Protocol v7 binds fresh
retrieval and System artifacts to this release while retaining the exact completed query plans.
The frozen v5 through v7 producer protocols are additive to v4; they do not add a TREC 2023
TrialGPT row.

The repository owner froze the v4 amendment before outcome inspection. Its exact digest is bound by
the runtime and an older protocol digest fails closed. Execution still requires the immutable
release, approved dependency notices, and - where applicable - the protected-Track access gate.
TrialGPT additionally requires separately authorized provider inputs.
Release-selected external Systems additionally require separately authorized model inputs.

The same v4 amendment adds five TAIM-adaptation rows on the existing judgment-union Profiles:
TrialGPT on TREC 2022 and both SIGIR query Profiles, plus track-specific TrialMatchAI System
identities on TREC 2022 and TREC 2023. These rows use the exact same pool receipt, Task Input,
Evaluation Package, topics, and query representation as their Track's reference rows. They do not
copy the published 26,581, 3,621, or 17,103 counts into TAIM identities.
The frozen paper matrix predeclares 33 unique rows; a release declares the subset its catalogue
lists. The release and input-specific gates still apply to every row.

TREC 2021 also exposes `trec-ct-2021-reverse-complete10`, a specifically named viability profile
over the frozen complete ten-trial by 75-patient matrix. It accepts only the Release 0.1 recipe-v3
Prepared Snapshot ID and Evaluation Package ID. It is not a general reverse TREC benchmark.
The other Tracks expose only the reverse contract and synthetic fixture until a later protocol
freezes an exhaustive denominator.

For TREC 2022 or 2023, both preparation and the real benchmark command require
`--protocol-approval-id sha256:<digest>`. The digest identifies the separately approved frozen
analysis protocol; the CLI stores it in every confirmation run and later Result Bundle. Synthetic
fixtures never accept a protocol approval identity.

Acquisition guides: [TREC 2021](docs/trec-2021-acquisition.md),
[TREC 2022](docs/trec-2022-acquisition.md), [TREC 2023](docs/trec-2023-acquisition.md), and
[SIGIR 2016](docs/sigir-2016-acquisition.md).
External-fidelity acquisition is in
[the external-fidelity Profile guide](docs/external-fidelity-profile.md).

## Scope

The executable catalog, not a Cartesian-product assumption, decides which Track, Task, Profile,
and System combinations are supported. Dense Systems require the `dense` extra and a local index.
RRF consumes same-direction BM25 and BGE-M3 component runs with identical benchmark identities.
Bundle-eligible RRF also requires both components to come from clean producers of the same frozen
release; their lock-bound dependency-environment identities are retained by the fusion run.
The supported Python and artifact versions are listed in [compatibility](docs/compatibility.md).
Development and confirmatory claim rules, including judged-recall terminology, are in the
[reporting policy](docs/reporting-policy.md).
Release owners use the [release verification contract](docs/release-verification.md).
Contribution and disclosure processes are in [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md). License and attribution information is in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Results produced here measure reference implementations under the stated benchmark protocol. They
do not establish clinical validity, deployment readiness, or equivalence to another implementation.

To make any run eligible for later Result Bundle admission, pass all three producer inputs:
`--release-manifest`, `--package-artifact`, and `--dependency-lock`. The dependency lock must be the
exact `uv.lock` recorded by the manifest. Local runs may omit all three, but cannot be admitted to a
Published Result Bundle.

A Result Bundle refuses any published string that is itself an absolute filesystem path, and names
each file and JSON location where it found one.
To bundle a TrialGPT run, pass a bare executable
name such as the default `--codex-executable codex`, not an absolute location.
TrialMatchAI L4
runs bundle with a location-free System Input: the release gives the TrialMatchAI checkout,
workspace, prepared corpus folders, search database, and timeout to the System beside it, and the
System Input binds the prepared corpus by its preparation receipt instead. A run is refused before
it starts when a token of its command is an absolute location, so name the executable bare in
`--trialmatchai-command`.

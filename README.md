# TAIM benchmark

An immutable benchmark release records its identity in `RELEASE-PROVENANCE.json` and the release
manifest.

TAIM is an executable benchmark for clinical trial matching. It fixes data and method identities,
the public execution path, evaluation rules, validation, and result bundles. Matching pipelines are
Systems; TAIM prescribes none. The CLI runs its catalog's Systems and scores any other's rankings.

TAIM defines the `patient_to_trial` and `trial_to_patient` Tasks with separate inputs, runs,
evaluators, and claim limits. Its Track definitions cover TREC Clinical Trials 2021, 2022, and 2023
and SIGIR Clinical Trials 2016. The installed capability catalog lists the combinations this
package can execute. TAIM reuses these existing test collections; it does not introduce a new
dataset or redistribute benchmark source files.

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
  --output-dir runs
trial-benchmark patient-to-trial run validate \
  --run-dir runs/quick-start-patient-to-trial
trial-benchmark patient-to-trial run evaluate \
  --run-dir runs/quick-start-patient-to-trial
```

Run the BGE-M3 dense System on the same fixture. It needs the `dense` extra, downloads the pinned
model (about 2.3 GB) on first use, and encodes the fixture on the CPU:

```console
uv sync --locked --extra dense
trial-benchmark patient-to-trial fixture run \
  --track trec-ct-2021 \
  --system dense-bge-m3 \
  --index-dir indexes/quick-start-dense-bge-m3 \
  --run-id quick-start-dense-bge-m3 \
  --output-dir runs
```

Run the Qwen3 embedding System the same way. It needs the same `dense` extra, downloads its pinned
model (about 1.2 GB) on first use, and encodes the fixture on the CPU:

```console
uv sync --locked --extra dense
trial-benchmark patient-to-trial fixture run \
  --track trec-ct-2021 \
  --system dense-qwen3-embedding-0.6b \
  --index-dir indexes/quick-start-dense-qwen3-embedding-0.6b \
  --run-id quick-start-dense-qwen3-embedding-0.6b \
  --output-dir runs
```

Fuse the BM25 and BGE-M3 runs with RRF. Each component must rank to the default depth of 1,000,
which is why the BM25 command above sets no `--top-k`:

```console
trial-benchmark patient-to-trial fixture run \
  --track trec-ct-2021 \
  --system rrf \
  --bm25-run runs/quick-start-patient-to-trial \
  --dense-run runs/quick-start-dense-bge-m3 \
  --run-id quick-start-rrf \
  --output-dir runs
trial-benchmark patient-to-trial run evaluate \
  --run-dir runs/quick-start-rrf
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

The direction-specific validate and evaluate commands print the exact release, Track, Task,
Profile, System, Prepared Snapshot ID, Task Input ID, System Input ID, and Evaluation Package ID.

The packaged fixture records are invented. They cover eligible, excluded, irrelevant, unknown, and
mixed-direction failure behavior. Their scores are conformance evidence, not effectiveness results.

### Licensed inputs you supply

Some Systems need licensed terminology that this repository cannot ship. You need your own licence
and your own copy. TAIM never downloads, bundles or accepts a licence for you.

| System | Licensed input | Required? | Without it |
| --- | --- | --- | --- |
| `bm25-folded` | SNOMED CT RF2 release, International Edition 2026-08-01; another edition is refused | Required | `queries prepare` refuses |
| `staged-bm25-qwen3-rrf-rerank` | the same, through `bm25-folded`'s query bundle | Required | cannot be assembled |
| TrialMatchAI L4 Systems (`trialmatchai-current-cuda-l4-*-v4`) | OMOP vocabulary files (`CONCEPT.csv`, `CONCEPT_SYNONYM.csv`); the qualified concept store includes SNOMED CT | Required | the run refuses without a registered concept store |

The other Systems need no licensed terminology.

**Getting SNOMED CT.** In a SNOMED International Member country, apply to your National Release
Centre. Elsewhere, apply to SNOMED International as an Affiliate through its Member Licensing and
Distribution Service (MLDS). Terms, fees and permitted territory depend on where you are. A
SNOMED CT download from OHDSI Athena does **not** by itself give you the right to use SNOMED CT
outside OHDSI's own tools; you still need your own licence.

## Real tracks

Acquire each Track locally using its guide, then prepare checksum-verified files:

Real preparation and execution are bound to the packaged Track source lock. The public CLI has no
override that can substitute an unofficial lock under a real effectiveness Profile.

TREC 2021 preparation uses `trec-ct-2021-snapshot-recipe-v3`, introduced in Release 0.1 and still used. It is a new immutable
prepared-corpus identity and is not interchangeable with historical TAIM campaign Snapshots. Final
publishable runs must be produced from this release after freeze. Complete-source preparation is
shared by the full-corpus and compute-bounded Profiles; only the effective Task Input differs.

```console
uv run trial-benchmark data prepare \
  --track trec-ct-2021 \
  --source /path/to/official-files \
  --output-dir prepared/trec-ct-2021
```

Run BM25 over the prepared complete corpus:

```console
uv run trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 \
  --profile official-full \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --system bm25 \
  --run-id bm25-official-full \
  --output-dir runs
```

The complete prepared corpus contains 375,580 trials. The paper's separately named TREC 2021
Profile ranks the score-blind union of all trial IDs present in the Evaluation Package. Inspect
and freeze its identity before producing any System score:

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
`pool_count`, `pool_ids_sha256`, and `pool_receipt_id` values to every command in the stratum.

Run BM25 with the frozen receipt ([its TREC 2021 reference scores](docs/reference-scores.json)):

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
  --package-artifact /path/to/taim_benchmark-0.3.0-py3-none-any.whl \
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

The installed support catalogue declares which Systems run on each complete-corpus or
judgment-union Profile. Use `trial-benchmark support list` to inspect this selection.
The complete Profiles remain public capabilities. The release freezes two protocols. The
[v3 paper protocol](docs/paper-analysis-protocol-2026-08-31-v3.md) selects the five judgment-union
strata for the paper: TREC 2021, TREC 2022, TREC 2023, SIGIR description, and SIGIR summary. The
judgment-union commands in this README pass its digest as `PROTOCOL_ID`. The
[v4 external-baseline amendment](docs/paper-analysis-protocol-2026-08-31-v4.md) retains that
matrix and governs the TREC 2021 external-fidelity Profile and the external baseline Systems; their
commands pass the v4 digest instead. Each Track × Task × Profile × System row remains separate, and
raw metrics are not pooled across Tracks. Where v4 and the v3 note say v4 supersedes v3, they mean
the paper's row allocation, not the digest: TREC 2022 and 2023 preparation and runs take v3 too.

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
  --package-artifact /path/to/taim_benchmark-0.3.0-py3-none-any.whl \
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
  --package-artifact /path/to/taim_benchmark-0.3.0-py3-none-any.whl \
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
  --package-artifact /path/to/taim_benchmark-0.3.0-py3-none-any.whl \
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
  --package-artifact /path/to/taim_benchmark-0.3.0-py3-none-any.whl \
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
The component and fusion producers must resolve to the same immutable release and lock identity.
The dated paper protocols ship in this repository. The CLI accepts only the SHA-256 of those exact
protocol bytes. Their Track and method choices are frozen, but each locally derived pool receipt and
the eventual immutable release identities must still be recorded before paper execution.

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
For the TrialMatchAI L4 Systems, it also executes the selected adapter. Their commands, runtime
prerequisites, and license boundaries are in the
[TrialMatchAI external baseline guide](docs/trialmatchai-external-baseline.md).
For TrialMatchAI v0.9.1, the authors' release run unchanged, it also executes the pinned upstream
tag. Its commands, hardware, model staging, concept store download, and license boundaries are in
the [TrialMatchAI v0.9.1 external baseline guide](docs/trialmatchai-v091-external-baseline.md).
The strata and their reporting boundary are in the
[external baseline guide](docs/external-baselines.md).

The release includes the memory-bounded prepared-package loader.
For TrialGPT, the release includes the complete, pinned retrieval producer. Its admitted targets,
query-plan receipts, MedCPT revisions, execution command, and evidence boundary are in the
[TrialGPT retrieval producer guide](docs/trialgpt-retrieval-producer.md). Fresh retrieval and System
artifacts bind this release. The four query-plan files protocol v6 admits are not available, so
query plans are generated per run and recorded by receipt, as protocol v11 defines it.
The frozen v5 through v7 and v11 producer protocols are additive to v4; a later amendment adds the
TREC 2023 target.

The repository owner froze the v4 amendment before outcome inspection. The runtime binds its exact
digest to the external-fidelity Profile and the external baseline Systems, and refuses any other
digest there; the judgment-union rows of the other Systems take the v3 digest. Execution still
requires the immutable release, approved dependency notices, and - where applicable - the
protected-Track access gate.
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
over the frozen complete ten-trial by 75-patient matrix. It accepts only the recipe-v3
Prepared Snapshot ID and Evaluation Package ID, introduced in Release 0.1 and still used. It is not a general reverse TREC benchmark.
The other Tracks expose only the reverse contract and synthetic fixture until a later protocol
freezes an exhaustive denominator.

A TREC 2022 or 2023 confirmation run stores the v3 digest it was given, as does any Result Bundle
made from it.

Acquisition guides: [TREC 2021](docs/trec-2021-acquisition.md),
[TREC 2022](docs/trec-2022-acquisition.md), [TREC 2023](docs/trec-2023-acquisition.md), and
[SIGIR 2016](docs/sigir-2016-acquisition.md).
External-fidelity acquisition is in
[the external-fidelity Profile guide](docs/external-fidelity-profile.md).

## Plug in your pipeline

A pipeline this release does not ship needs no change to the repository and never imports TAIM.
`trial-benchmark <direction> submission export` writes the exact Task Input the catalog's Systems
receive. Your pipeline ranks it on its own machine, and `submission import` scores that ranking
against judgments the pipeline never sees. [Score your own pipeline](docs/external-pipelines.md)
walks through it on the synthetic fixture and shows how to compare the result with a reference
System on the same Task Input.
The [reference scores](docs/reference-scores.json) cover `bm25` on the TREC 2021 and both SIGIR 2016
judgment-union Profiles.

## Scope

The executable catalog, not a Cartesian-product assumption, decides which Track, Task, Profile,
and System combinations are supported.
Dense Systems require the `dense` extra; the run builds its
own index under `--index-dir`. On CUDA they need compute capability 7.5 or newer, so not a V100; each
System's hardware is stated with its [method contract](docs/method-contracts.md).
RRF consumes same-direction BM25 and BGE-M3 component runs with identical benchmark identities.
Bundle-eligible RRF also requires both components to come from clean producers of the same frozen
release; their lock-bound dependency-environment identities are retained by the fusion run.
Every command and option is described in the [command reference](docs/cli-reference.md), which is
generated from the CLI's own help text.
The supported Python and artifact versions are listed in [compatibility](docs/compatibility.md).
Development and confirmatory claim rules, including judged-recall terminology, are in the
[reporting policy](docs/reporting-policy.md).
Release owners use the [release verification contract](docs/release-verification.md).
Contribution and disclosure processes are in [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md). License and attribution information is in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Results produced here measure reference implementations under the stated benchmark protocol. They
do not establish clinical validity, deployment readiness, or equivalence to another implementation.

A Result Bundle refuses any published string that is itself an absolute filesystem path, and names
each file and JSON location where it found one.

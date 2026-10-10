# TrialMatchAI external baseline

The selected System is the current pinned TrialMatchAI fork at commit
`99aa499e7296ebde836a0ebf00a5ed4948892a83`, not the original paper implementation. The TAIM
adapter verifies the clean checkout, origin, commit, config digest, model and adapter-weight
identities, concept store, index inputs, environment overrides, and normalized ranking.

Provision the upstream checkout, its own locked runtime, licensed model snapshots, a concept store,
and a prepared TrialMatchAI corpus outside this repository.

The prepared corpus is the authors' own, Zenodo record `15516900` (CC BY 4.0, by Majd Abdallah), as
their `bootstrap-data` command downloads it. Cut to the TREC 2021 Snapshot's trials (below), it
holds 26,149 trials and 425,771 criteria.

**These Systems cannot run without a SNOMED CT licence.** Their concept store is built from OMOP
vocabulary files that include SNOMED CT. This repository ships no vocabulary file and no concept
store: you build your own store from files you are licensed to use, with `build-concept-store`
below. See **Licensed vocabulary** at the end of this page, and **A store you build is an outside
build** for how a run accepts it and how its results must be reported.

Build the checkout's locked runtime with TAIM, not with a hand-written `uv sync`:

```console
trial-benchmark patient-to-trial trialmatchai build-env \
  --checkout /path/to/pinned/TrialMatchAI \
  --receipt /path/to/trialmatchai-environment-receipt.json
```

Upstream's extras are exact: `uv sync` removes every package the extras it is given do not need,
and still exits 0. The command reads the extras from the backends the frozen config selects and
refuses a backend it cannot map. It also refuses, before changing anything, when the lock does not
install a needed package on this machine; vLLM installs on Linux only. It prints what the sync will
remove before it runs. The receipt is written only after the environment matches the lock for
those extras and every needed module imports. Do not run `uv sync` in the checkout afterwards.

Find the selected TrialMatchAI System for your Track with `trial-benchmark support list`, then add
its ID and inputs:

```console
  --system <selected-trialmatchai-system-id> \
  --trialmatchai-cwd /path/to/pinned/TrialMatchAI \
  --trialmatchai-corpus-dir /path/to/trialmatchai-corpus \
  --trialmatchai-search-db-path /path/to/trialmatchai-corpus/search \
  --trialmatchai-workspace /path/to/empty-workspace \
  --top-k 20
```

A run refuses a prepared corpus whose trial IDs differ from the Snapshot's. Cut TrialMatchAI's
prepared corpus to exactly the Snapshot's trials into a new folder. The command hard-links the
prepared files, and so their embeddings, when the filesystem allows and copies them otherwise. It
refuses a Snapshot trial the prepared corpus lacks and writes `filter-manifest.json`, whose digest
the run records in its index identity:

```console
trial-benchmark patient-to-trial trialmatchai filter-corpus \
  --snapshot-trials /path/to/snapshot/trials.jsonl \
  --source-trials /path/to/prepared/processed_trials \
  --source-criteria /path/to/prepared/processed_criteria \
  --output /path/to/trialmatchai-corpus
```

The manifest lists the Snapshot trials that have no `processed_criteria/` folder; they stay in the
corpus. After an interrupted cut, `--reuse-existing` completes the same output and refuses any file
whose contents differ.

A run refuses a prepared corpus without a preparation receipt. Write the receipt once for each
prepared corpus. The command hashes every file under `processed_trials/` and
`processed_criteria/` and writes `trialmatchai-corpus-receipt.json` beside those folders:

```console
trial-benchmark patient-to-trial trialmatchai adopt-corpus \
  --corpus-dir /path/to/trialmatchai-corpus
```

The receipt names no location, so it travels with a copy of the corpus directory. At run start
the run binds the receipt's content digest into its System Input and checks the corpus's file
names and counts against the receipt. It does not hash the corpus again, so after changing a
prepared file, delete the receipt and adopt the corpus again. The run hashes the search database
it is given at every start, such as a staged node-local copy, and records that digest in its index
identity and the hash's wall time in its run record.

Build the concept store with TAIM, in the checkout's environment, from your own OMOP vocabulary
files:

```console
trial-benchmark patient-to-trial trialmatchai build-concept-store \
  --checkout /path/to/pinned/TrialMatchAI \
  --concept-csv /path/to/omop-vocabulary/CONCEPT.csv \
  --synonym-csv /path/to/omop-vocabulary/CONCEPT_SYNONYM.csv \
  --source-release '<your-vocabulary-release>' \
  --db-path /path/to/concept-store \
  --work-dir /path/to/concept-store-work
```

The command uses the qualified store's recipe: the frozen config's pinned BGE-M3 embedder, the OMOP
vocabularies listed under **Licensed vocabulary**, TrialMatchAI's open dictionaries at the versions
the qualified store was built from, and the pinned chunking and index parameters. It downloads each
open dictionary into the work directory from an address that serves that version, and uses it only
if its SHA-256 is the pinned one. Two of those versions are no longer served by their sources: NCBI
Gene's `Homo_sapiens.gene_info.gz` as of 2026-08-05, and Cellosaurus release 56.0. The project
archives both, unmodified, on Zenodo under the version DOI `10.5281/zenodo.23088342`, and the
command fetches them from there. Cellosaurus is by the SIB Swiss Institute of Bioinformatics, under
CC BY 4.0; the archive's `NOTICE.md` gives the credit to keep if you pass the file on. The command
embeds on a GPU when one is available, and checkpoints every chunk in the work directory, so an
interrupted build resumes where it stopped. It writes the store with its provenance manifest,
`concept-store-manifest.json`. A run reads the store from the directory that
`TRIALMATCHAI_CONCEPT_DB_PATH` names.

Before any model loads, a run compares the store with the qualified store, `candidate-21352305`,
recorded in `src/taim/pipelines/trialmatchai/data/trialmatchai-l4-concept-store.json`: first the row count,
embedding mode and vector digest its manifest states, then the bytes of every file in it, including
the trained vector and full-text indexes that entity linking searches. A store that matches is the
qualified store, the one the published rows ran on.

**A store you build is an outside build.** No rebuild gives the qualified store's bytes. LanceDB
names every data and index file at random and stamps every table version; it trains the vector
index, and orders full-text ties, afresh on every build, with no seed to fix; and BGE-M3 vectors
differ in their last bits between GPU models. So a run accepts a store with other bytes as an
outside build of the qualified store when:

- its manifest states the qualified recipe - the embedding model and settings, the index
  parameters and the row count - and the pinned dictionary hashes; and
- its rows, read from the store in the checkout's environment, have the digest recorded in
  `src/taim/pipelines/trialmatchai/data/trialmatchai-l4-concept-store-rows.json`. That digest covers the text of
  every row and leaves out row order and vectors.

The run records such a store as `outside-build`, with its own vector and content digests next to
the qualified store's name. Its vectors and indexes are its own, and the trained index decides which
concepts linking finds, so its results are that build's, not the published rows: report them as
results on an outside build, as you would a run on other hardware. A store with other rows - another
vocabulary release or selection, or other dictionary versions - is a different method input, and
the run refuses it and names each field that differs.

Matching the rows also needs the OMOP vocabulary rows of the qualified store's Athena v20260227
selection. Whether a download of that release made today reproduces them has not been tested.

The checkout, workspace, corpus folders, search database, and timeout are not part of the System
Input, so its System Input ID does not depend on where they are, and the run can be bundled.

Before a run, in the same job on the same node, check that the environment can run this pipeline:

```console
trial-benchmark patient-to-trial trialmatchai preflight \
  --system <selected-trialmatchai-system-id> \
  --trialmatchai-cwd /path/to/pinned/TrialMatchAI \
  --trialmatchai-corpus-dir /path/to/trialmatchai-corpus
```

The preflight loads no model and changes nothing. Its first check is the run's own, in the run's
order and stopping where the run would: the required and forbidden variables, the checkout, config
and adapter weights, and the qualified concept store. Then come the prepared corpus and its search
tables, and whether the checkout's environment is exactly its lock with the extras the config's
backends need, derived as `build-env` derives them. In the checkout's own interpreter it checks that
each module those backends need imports, that the card's compute capability is one the installed
torch was built for (CUDA being available is not enough), that the card has room for the frozen
memory shares, and that each pinned model resolves offline from the cache. Each check prints `PASS`
or `FAIL` with its reason. The last line is the total, and the exit status is nonzero on any `FAIL`.
A pass covers these static checks only. It starts no engine, so it cannot show that the engines load
and reach an eligibility decision.

Hardware: Linux with CUDA. The System runs in the pinned checkout's own locked runtime, not TAIM's:
torch 2.11.0 built for CUDA 13 with kernels for compute capability 7.5 and newer, and vLLM 0.23.0.
That runtime has completed 75-topic runs on A40 (48 GB), A100, and H200 cards. A V100 (7.0) cannot
run it, and a 24 GB L4 cannot hold its reranker's KV cache; the preflight refuses both. Plan about
100 GB of disk for the model snapshots, prepared corpus, concept store, and runtime.

**Licensed vocabulary (required).** The concept store is built from OMOP vocabulary files
(`CONCEPT.csv` and `CONCEPT_SYNONYM.csv`). The qualified store for these Systems was built from the
OHDSI Athena v20260227 selection that includes SNOMED CT, LOINC, RxNorm, RxNorm Extension, ATC,
ICD10, ICD10CM, MeSH and OMOP Extension. Reproducing it therefore needs your own SNOMED CT licence:
an OHDSI Athena download does not grant one. Obtain SNOMED CT rights from your National Release
Centre or SNOMED International (MLDS). Check each other vocabulary's own terms on Athena. This
repository ships no vocabulary file and no concept store; their licences govern who may use them.

**Without it** the run refuses: it needs a registered concept store and will not use
TrialMatchAI's unregistered default. A store built from a different vocabulary selection is a
different method input, and its results are not comparable to the published rows.

Gemma 2 is gated by its own terms. The LoRA adapters do not remove the base-model terms. No model,
concept store, index, external checkout, or license acceptance is included or performed by TAIM.

The TREC 2023 row is the later current-fork adaptation, not a TrialMatchAI paper result. Do not use
this System's retrieved top 500 as a neutral corpus because that set depends on System scores.

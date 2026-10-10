# TrialMatchAI v0.9.1 external baseline

The System `trialmatchai-upstream-v091-cuda-trec21-development-v1` runs TrialMatchAI as its
authors release it: `cbib/TrialMatchAI` tag `v0.9.1`, commit
`3bc0e5f44d742ef069f8ef73bbc4187573d9db6a`, unchanged, on the tag's own TREC path. TAIM supplies
the topics, the trial pool and the evaluator; it changes none of their code or settings. The System
is declared on the TREC 2021 external-fidelity Profile only.

TAIM pins what the tag leaves open, and the run refuses anything else:

- the six model revisions (the tag names none);
- the concept store, by the SHA-256 of its manifest;
- the checkout, by commit, origin and a clean tree;
- the config, which is the tag's own with only those revisions filled in.

**No SNOMED CT licence is needed.** The concept store is the tag's open-only recipe, built from six
openly licensed sources. It contains no SNOMED CT content.

## What you need

| | |
| --- | --- |
| Card | Linux with one CUDA card of about 64 GB or more. The tag's runtime is torch 2.11.0 built for CUDA 13, which has no kernels for a V100. Their config serves phi-4 unquantized (29.3 GB in bf16) at `gpu_memory_utilization` 0.5, so an A40 or a 40 GB A100 cannot load it |
| Measured | one NVIDIA H200: all 75 TREC 2021 topics in 6.6 hours of System time, about 5 minutes a topic. The whole job took 6 h 44 min, which NCSA Delta billed as about 20 GPU-hours. Card peak 137 GB, process peak 29.9 GB of RAM |
| Not measured | an 80 GB A100. It should hold the models; no run has shown it |
| CPU | no route: their eligibility and reranking engines are vLLM on CUDA |
| Gated model | `google/gemma-2-2b-it` is under the Gemma Terms of Use. Accept them on its Hugging Face page and create an access token, `HF_TOKEN`, to stage it. The reranker LoRA does not remove the base model's terms |
| Downloads | their prepared corpus, about 28.9 GB compressed; the concept store, 2.7 GB; the six models; their environment |

## 1. Their checkout and environment

The run accepts the checkout only at the pinned commit, with origin exactly
`https://github.com/cbib/TrialMatchAI.git` and no local changes. Build their environment with the
tag's own lock and the extras their TREC path needs:

```console
git clone https://github.com/cbib/TrialMatchAI.git /path/to/TrialMatchAI
git -C /path/to/TrialMatchAI checkout --detach 3bc0e5f44d742ef069f8ef73bbc4187573d9db6a
cd /path/to/TrialMatchAI && uv sync --frozen --python 3.11 --extra llm --extra gpu --extra entity
```

Keep `.env` files out of the checkout and every directory above it. Their config loader reads them
and fills unset variables after TAIM has checked the environment.

## 2. The six models

Stage the pinned revisions into one Hugging Face hub cache, outside the checkout. The command runs
in their environment, downloads each revision and points each `refs/main` at its pin, because two
of their loaders resolve `main` whatever revision they are given:

```console
export HF_TOKEN=<your-hugging-face-token>
trial-benchmark patient-to-trial trialmatchai stage-models \
  --checkout /path/to/TrialMatchAI \
  --cache /path/to/hf-hub
```

| Role | Repository | Revision |
| --- | --- | --- |
| Entity extraction | `fastino/gliner2-base-v1` | `f5b2ecedebe4381b088c1cf276f5bf72a52cac54` |
| Embeddings | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| Eligibility | `microsoft/phi-4` | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` |
| Eligibility adapter | `majdabd33/trialmatchai-phi4-reasoning-lora` | `9eaddaf048c8d4266a291884ff89db1cf05b07fc` |
| Reranker | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| Reranker adapter | `majdabd33/trialmatchai-gemma2-reranker-lora` | `3118ba76d545f71f3aaa7952d2f031ad5fdef8c9` |

The run works offline from that cache. At its start it checks every `refs/main`, every pinned
snapshot, and the files of gliner2 and the two adapters, which no loader binds by revision.

## 3. The concept store

The store cannot be rebuilt byte for byte: LanceDB names its files at random, and the embeddings
differ in their last bits between GPU models. So the run binds the exact store, and the project
publishes it on Zenodo, version DOI `10.5281/zenodo.23245102`, with the six source files it was
built from. The store archive is published in six parts, because Zenodo did not accept it as one
2.7 GB file. Download `trialmatchai-v091-concept-store.tar.part-aa` to `.part-af` and `SHA256SUMS`
from that record. Check the parts, join them in order, check the joined archive, and extract it:

```console
cd /path/to/store-download && grep ' trialmatchai-v091-concept-store.tar.part-a.$' SHA256SUMS | sha256sum -c -
cat /path/to/store-download/trialmatchai-v091-concept-store.tar.part-a? \
  > /path/to/store-download/trialmatchai-v091-concept-store.tar
echo "b4afb84e935e516e0bac6284c73fbef28aa739a98e019fe3cb7242dd3dffab78  /path/to/store-download/trialmatchai-v091-concept-store.tar" | sha256sum -c
tar -xf /path/to/store-download/trialmatchai-v091-concept-store.tar -C /path/to/trialmatchai-store
sha256sum /path/to/trialmatchai-store/concepts/concept-store-manifest.json
```

The manifest's SHA-256 must be
`1cd46b8634f3d01b02991940effe4e5d50ccc825b9f75245be56915d0ea556b3`. The run reads the store from
the directory `TRIALMATCHAI_CONCEPT_DB_PATH` names, which is the extracted `concepts/`. At its start
it checks the manifest and the size and SHA-256 of every file it lists, and refuses a store with
any other file in it.

The store holds 616,925 concepts from NCBI Gene, the Human Disease Ontology, ChEBI, Cellosaurus,
the Cell Ontology and the Human Phenotype Ontology. Its rows are derived from those sources; they
are not the sources. The record's `NOTICE.md` gives each source's credit and licence, including the
Human Phenotype Ontology's conditions. Keep them if you pass the store on.

## 4. Their corpus and its search database

The run uses the authors' own prepared corpus, Zenodo record `15516900` (CC BY 4.0, by Majd
Abdallah), as their tag's TREC route does. Their `bootstrap-data` command downloads
`processed_trials.tar.gz` and `criteria_part_0.zip` to `criteria_part_5.zip` and extracts them
under `--root`, outside the checkout:

```console
cd /path/to/TrialMatchAI && uv run --frozen trialmatchai bootstrap-data --root /path/to/trialmatchai-data
```

A run refuses a corpus whose trial IDs differ from its Task Input's. The external-fidelity Task
Input is the 26,149 trials of `prepared/trec-ct-2021-external-fidelity-26149-ids.json` (see the
external-fidelity Profile guide). Write them as JSON Lines, cut the corpus to exactly those trials
into a new folder, and write the corpus's preparation receipt:

```console
python3 -c 'import json, sys; [print(json.dumps({"trial_id": i})) for i in json.load(sys.stdin)]' \
  < prepared/trec-ct-2021-external-fidelity-26149-ids.json > /path/to/pool-trials.jsonl

trial-benchmark patient-to-trial trialmatchai filter-corpus \
  --snapshot-trials /path/to/pool-trials.jsonl \
  --source-trials /path/to/trialmatchai-data/data/processed_trials \
  --source-criteria /path/to/trialmatchai-data/data/processed_criteria \
  --output /path/to/trialmatchai-corpus

trial-benchmark patient-to-trial trialmatchai adopt-corpus \
  --corpus-dir /path/to/trialmatchai-corpus
```

`filter-corpus` hard-links the prepared files when the filesystem allows, and writes
`filter-manifest.json`, which the run records. `adopt-corpus` hashes every prepared file and writes
`trialmatchai-corpus-receipt.json` beside the folders; the run binds its digest.

Build the search database with their `index` command before the run. The run checks that the
database is unchanged when it ends, so it refuses one their pipeline had to build or rebuild
during the run:

```console
cd /path/to/TrialMatchAI && TRIALMATCHAI_SEARCH_DB_PATH=/path/to/trialmatchai-corpus/search \
  uv run --frozen trialmatchai index \
  --processed-trials-folder /path/to/trialmatchai-corpus/processed_trials \
  --processed-criteria-folder /path/to/trialmatchai-corpus/processed_criteria
```

The run records the database's digest; it does not pin it.

## 5. The run

Set the run's environment. `TRIALMATCHAI_CONCEPT_DB_PATH` is the only `TRIALMATCHAI_` variable the
run accepts; it sets the others itself and refuses any it did not set:

```console
export TRIALMATCHAI_CONCEPT_DB_PATH=/path/to/trialmatchai-store/concepts HF_HUB_CACHE=/path/to/hf-hub HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Then add the System's options to the external-fidelity Profile's common command (the same
`--pool-file`, receipt values, protocol ID, release manifest, package artifact and dependency lock
as every row in that stratum):

```console
  --system trialmatchai-upstream-v091-cuda-trec21-development-v1 \
  --trialmatchai-cwd /path/to/TrialMatchAI \
  --trialmatchai-corpus-dir /path/to/trialmatchai-corpus \
  --trialmatchai-workspace /path/to/empty-workspace \
  --top-k 60
```

K = 60 is the budget of the project's v0.9.1 run. Their TREC preset ranks up to 250 trials a topic;
the run keeps the first K. `--trialmatchai-timeout-seconds` applies per topic.

The run executes their TREC track steps in their order, in their environment: query expansion,
the search index, then matching with first-level retrieval, the Gemma reranker and phi-4
eligibility. It refuses a run in which their concept linking fell back silently: any of their
degradation lines in the log, or no topic whose query plan carries linker terms. It records each
topic's assessment mode, the model and store identities, and the corpus and database digests.

Their eligibility model answers some candidates without parseable JSON; their code skips each such
candidate and marks the topic `partial`. That is their pipeline's behaviour, recorded, not
repaired.

**Results vary slightly from run to run.** The cause is GPU arithmetic in the eligibility step.
Their tag already asks for the most repeatable generation its settings allow: temperature 0,
`top_p` 1.0 and seed 1234. But vLLM groups the eligibility prompts into large batches, and the
arithmetic of a batch can change the last bits of a model output. Where two next words are nearly
tied, that is enough to change a long eligibility answer, and with it a trial's place in the
ranking. Everything before eligibility reproduces byte for byte: query expansion, first-level
retrieval and the reranked shortlist. In the project's runs on H200 cards, a rerun's final top 60
for a topic shared 43 to 59 of its 60 trials with the project's own run, and 51 at the median.

## Licences

TrialMatchAI is MIT-licensed (Majd Abdallah, Macha Nikolski). The models are under their own
licences: gliner2 Apache-2.0, bge-m3 MIT, phi-4 and its LoRA MIT, Gemma 2 and its reranker LoRA
under the Gemma Terms of Use. Their corpus is CC BY 4.0. The concept store's sources keep their
own terms, listed in the Zenodo record's `NOTICE.md`. This repository ships no model, corpus,
concept store, search database or external checkout, and accepts no licence for you.

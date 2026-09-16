# TrialMatchAI external baseline

The selected System is the current pinned TrialMatchAI fork at commit
`7eba8f399336fcd988b00ce98f2e025e5ec04119`, not the original paper implementation. The TAIM
adapter verifies the clean checkout, origin, commit, config digest, model and adapter-weight
identities, concept-store manifest, index inputs, environment overrides, and normalized ranking.

Provision the upstream checkout, its own locked runtime, licensed model snapshots, a frozen
concept store, and a prepared TrialMatchAI corpus outside this repository. Then add:

```console
  --system trialmatchai-current-cuda-l4-trec21-development-v3 \
  --trialmatchai-cwd /path/to/pinned/TrialMatchAI \
  --trialmatchai-corpus-dir /path/to/trialmatchai-corpus \
  --trialmatchai-search-db-path /path/to/trialmatchai-corpus/search \
  --trialmatchai-workspace /path/to/empty-workspace
```

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

The checkout, workspace, corpus folders, search database, and timeout are not part of the System
Input, so its System Input ID does not depend on where they are. A run is refused before it starts
when a token of `--trialmatchai-command` is an absolute location; name the executable bare.

Gemma 2 is gated by its own terms. The LoRA adapters do not remove the base-model terms. The
concept store may depend on separately licensed terminology sources. No model, concept store,
index, external checkout, or license acceptance is included or performed by TAIM.

The TREC 2023 row is the later current-fork adaptation, not a TrialMatchAI paper result. Do not use
this System's retrieved top 500 as a neutral corpus because that set depends on System scores.

# Command reference

Every `trial-benchmark` command and option, generated from the command's own argument parser:
`trial-benchmark <command> --help` prints the same descriptions. Do not edit this page; change the
help text in the code and regenerate it with `python -m taim.cli_reference > docs/cli-reference.md`.
`trial-benchmark support list` reports the Systems this release runs.

## `trial-benchmark support`

inspect executable release capabilities

### `trial-benchmark support list`

print the Track, Task, Profile and System combinations this release executes

No options.

## `trial-benchmark patient-to-trial`

patient query against trial corpus

### `trial-benchmark patient-to-trial fixture`

run deterministic conformance data

### `trial-benchmark patient-to-trial fixture run`

run one System on the packaged synthetic fixture; its scores are conformance evidence, not effectiveness

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | `trec-ct-2021` | Track whose synthetic fixture to run |
| `--system` | `SYSTEM` |  | System to run; support list reports the Systems each Track and Profile declare. Omitted, the System the release's quick start runs |
| `--run-id` | text |  | name of the run and of its directory under --output-dir; omitted, public-<System>-<UTC time> |
| `--output-dir` | path | `runs` | directory that receives the run directory |
| `--top-k` | positive integer | `1000` | ranking depth per query, the run's budget K; bound into the System Input ID |
| `--metric-cutoff` | positive integer | `5` | primary metric cutoff, bound into the System Input ID; a Result Bundle admits only the Profile's cutoffs |
| `--k1` | number | `1.2` | BM25 term-frequency saturation parameter |
| `--b` | number | `0.75` | BM25 document-length normalisation |
| `--folded-query-bundle` | path |  | folded queries written by queries prepare for the same Task Input; required by the Systems that read folded queries |
| `--index-dir` | path |  | directory where a dense System builds its corpus index, resumes an interrupted build or reuses a finished one; required by the dense Systems |
| `--device` | text | `cpu` | PyTorch device that encodes, such as cpu or cuda |
| `--document-batch-size` | positive integer | `8` | trial documents encoded per batch |
| `--query-batch-size` | positive integer | `8` | queries encoded per batch |
| `--rebuild-index` | none |  | delete the index and build checkpoints in --index-dir and encode the corpus again |
| `--bm25-run` | path |  | run directory of the BM25 component; required by RRF |
| `--dense-run` | path |  | run directory of the BGE-M3 component; required by RRF |
| `--folded-bm25-run` | path |  | run directory of the folded-query BM25 component; required by the staged System |
| `--no-template-qwen-run` | path |  | run directory of the no-template Qwen3 component; required by the staged System |
| `--reranker-score-artifact` | path |  | create or reuse raw, unevaluated scores for the staged reranker pool; required by the staged System |
| `--reranker-checkpoint` | path |  | resume the pinned reranker from an identity-bound batch checkpoint |
| `--model-cache` | path |  | Hugging Face cache directory for the pinned reranker; omitted, the default cache |
| `--reranker-attention-backend` | one of `flash`, `mem_efficient`, `math` | `flash` | the only SDPA kernel the reranker may use; recorded with its scores |
| `--reranker-offline` | none |  | require the pinned reranker revision to exist in the local model cache |
| `--trialgpt-frozen-retrieval` | path |  | fresh frozen retrieval artifact for this Task Input, from the retrieval producer; required by TrialGPT |
| `--trialgpt-frozen-retrieval-lock` | path |  | the lock written with that artifact; required by TrialGPT |
| `--codex-executable` | text | `codex` | Codex CLI that makes the provider calls; name it bare, not by absolute location, or the run cannot be bundled |
| `--trialgpt2-checkout` | path |  | a clean checkout of TrialGPT 2.0 at the pinned commit; required by TrialGPT 2.0 |
| `--trialgpt2-workspace` | path |  | an empty directory, or the workspace of an interrupted TrialGPT 2.0 run to resume in place; omitted, .trialgpt2/<run ID> under --output-dir |
| `--trialgpt2-generation-workers` | positive integer | `48` | TrialGPT 2.0 candidate pairs scored at once, 1 to 48 |
| `--trialgpt-publication-contract` | path |  | contract written by trialgpt preflight; the run must match it exactly. Required by TrialGPT |
| `--trialgpt-workspace` | path |  | an empty directory, or the workspace of an interrupted run to resume in place; omitted, .trialgpt/<run ID> under --output-dir |
| `--trialgpt-generation-workers` | positive integer | `48` | concurrent provider calls, 1 to 48; must equal the preflight contract's |
| `--trialmatchai-cwd` | path |  | the pinned TrialMatchAI checkout the command runs in; required by TrialMatchAI |
| `--trialmatchai-workspace` | path |  | an empty working directory; omitted, .trialmatchai/<run ID> under --output-dir |
| `--trialmatchai-corpus-dir` | path |  | prepared corpus holding processed_trials/, processed_criteria/ and the receipt trialmatchai adopt-corpus wrote; required by TrialMatchAI |
| `--trialmatchai-search-db-path` | path |  | search database, hashed at every run start; omitted, search/ in the corpus |
| `--trialmatchai-timeout-seconds` | number | `3600.0` | wall-time limit of one TrialMatchAI invocation |
| `--trialmatchai-command` | text, one or more |  | command that runs TrialMatchAI, with {input}, {processed_trials} and {processed_criteria} placeholders; name the executable bare, since an absolute location is refused. Omitted, the pinned end-to-end command |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; only runs given all three can enter a Result Bundle, and judgment-union and external-fidelity Profiles require them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |

### `trial-benchmark patient-to-trial benchmark`

run a qualified real-data profile

### `trial-benchmark patient-to-trial benchmark run`

run one System on a real Profile of a prepared Track

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | required | Track of the prepared files |
| `--profile` | text |  | Benchmark Profile to run; omitted, official-full on TREC 2021 and confirmation-full on TREC 2022/2023. SIGIR 2016 requires one |
| `--system` | `SYSTEM` |  | System to run; support list reports the Systems each Track and Profile declare. Omitted, the System the release's quick start runs |
| `--run-id` | text |  | name of the run and of its directory under --output-dir; omitted, public-<System>-<UTC time> |
| `--output-dir` | path | `runs` | directory that receives the run directory |
| `--top-k` | positive integer | `1000` | ranking depth per query, the run's budget K; bound into the System Input ID |
| `--metric-cutoff` | positive integer | `5` | primary metric cutoff, bound into the System Input ID; a Result Bundle admits only the Profile's cutoffs |
| `--k1` | number | `1.2` | BM25 term-frequency saturation parameter |
| `--b` | number | `0.75` | BM25 document-length normalisation |
| `--folded-query-bundle` | path |  | folded queries written by queries prepare for the same Task Input; required by the Systems that read folded queries |
| `--index-dir` | path |  | directory where a dense System builds its corpus index, resumes an interrupted build or reuses a finished one; required by the dense Systems |
| `--device` | text | `cpu` | PyTorch device that encodes, such as cpu or cuda |
| `--document-batch-size` | positive integer | `8` | trial documents encoded per batch |
| `--query-batch-size` | positive integer | `8` | queries encoded per batch |
| `--rebuild-index` | none |  | delete the index and build checkpoints in --index-dir and encode the corpus again |
| `--bm25-run` | path |  | run directory of the BM25 component; required by RRF |
| `--dense-run` | path |  | run directory of the BGE-M3 component; required by RRF |
| `--folded-bm25-run` | path |  | run directory of the folded-query BM25 component; required by the staged System |
| `--no-template-qwen-run` | path |  | run directory of the no-template Qwen3 component; required by the staged System |
| `--reranker-score-artifact` | path |  | create or reuse raw, unevaluated scores for the staged reranker pool; required by the staged System |
| `--reranker-checkpoint` | path |  | resume the pinned reranker from an identity-bound batch checkpoint |
| `--model-cache` | path |  | Hugging Face cache directory for the pinned reranker; omitted, the default cache |
| `--reranker-attention-backend` | one of `flash`, `mem_efficient`, `math` | `flash` | the only SDPA kernel the reranker may use; recorded with its scores |
| `--reranker-offline` | none |  | require the pinned reranker revision to exist in the local model cache |
| `--trialgpt-frozen-retrieval` | path |  | fresh frozen retrieval artifact for this Task Input, from the retrieval producer; required by TrialGPT |
| `--trialgpt-frozen-retrieval-lock` | path |  | the lock written with that artifact; required by TrialGPT |
| `--codex-executable` | text | `codex` | Codex CLI that makes the provider calls; name it bare, not by absolute location, or the run cannot be bundled |
| `--trialgpt2-checkout` | path |  | a clean checkout of TrialGPT 2.0 at the pinned commit; required by TrialGPT 2.0 |
| `--trialgpt2-workspace` | path |  | an empty directory, or the workspace of an interrupted TrialGPT 2.0 run to resume in place; omitted, .trialgpt2/<run ID> under --output-dir |
| `--trialgpt2-generation-workers` | positive integer | `48` | TrialGPT 2.0 candidate pairs scored at once, 1 to 48 |
| `--trialgpt-publication-contract` | path |  | contract written by trialgpt preflight; the run must match it exactly. Required by TrialGPT |
| `--trialgpt-workspace` | path |  | an empty directory, or the workspace of an interrupted run to resume in place; omitted, .trialgpt/<run ID> under --output-dir |
| `--trialgpt-generation-workers` | positive integer | `48` | concurrent provider calls, 1 to 48; must equal the preflight contract's |
| `--trialmatchai-cwd` | path |  | the pinned TrialMatchAI checkout the command runs in; required by TrialMatchAI |
| `--trialmatchai-workspace` | path |  | an empty working directory; omitted, .trialmatchai/<run ID> under --output-dir |
| `--trialmatchai-corpus-dir` | path |  | prepared corpus holding processed_trials/, processed_criteria/ and the receipt trialmatchai adopt-corpus wrote; required by TrialMatchAI |
| `--trialmatchai-search-db-path` | path |  | search database, hashed at every run start; omitted, search/ in the corpus |
| `--trialmatchai-timeout-seconds` | number | `3600.0` | wall-time limit of one TrialMatchAI invocation |
| `--trialmatchai-command` | text, one or more |  | command that runs TrialMatchAI, with {input}, {processed_trials} and {processed_criteria} placeholders; name the executable bare, since an absolute location is refused. Omitted, the pinned end-to-end command |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; only runs given all three can enter a Result Bundle, and judgment-union and external-fidelity Profiles require them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |
| `--data-dir` | path | required | directory written by data prepare for this Track |
| `--clinical-as-of` | ISO 8601 time with offset | required | clinical as-of time, bound into the Task Input ID: another value is another Task Input |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer |  | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest |  | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest |  | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest |  | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |

### `trial-benchmark patient-to-trial run`

validate or evaluate a closed run

### `trial-benchmark patient-to-trial run validate`

check a run directory against its manifest and print its identities

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--run-dir` | path | required | run directory a run command wrote |

### `trial-benchmark patient-to-trial run evaluate`

validate a run directory, then score its Primary Ranking

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--run-dir` | path | required | run directory a run command wrote |
| `--stage` | text |  | evaluate one named stage as a diagnostic, leaving the Primary Ranking unchanged |

### `trial-benchmark patient-to-trial submission`

export inputs and evaluate an external pipeline's rankings

### `trial-benchmark patient-to-trial submission export`

write the Task Input a pipeline ranks and, apart from it, the judgments that score it

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | `trec-ct-2021` | Track of the task to export |
| `--fixture` | none |  | export packaged synthetic data without downloads |
| `--data-dir` | path |  | directory written by data prepare; required without --fixture |
| `--profile` | text |  | Benchmark Profile of the task; required without --fixture |
| `--clinical-as-of` | ISO 8601 time with offset |  | clinical as-of time, bound into the Task Input ID; required without --fixture |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer |  | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest |  | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest |  | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest |  | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; every export without --fixture requires them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |
| `--top-k` | positive integer | `1000` | depth the pipeline must rank each query to |
| `--output` | path | required | new task directory: input.json only, the one file the pipeline reads |
| `--evaluator-output` | path | required | new evaluator directory: evaluation-package.json only; keep it from the pipeline |

### `trial-benchmark patient-to-trial submission import`

check a pipeline's ranking and declaration against the exported task and score it

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--task-dir` | path | required | task directory export wrote; it may hold input.json only |
| `--evaluator-dir` | path | required | evaluator directory export wrote |
| `--ranking` | path | required | the pipeline's ranking.jsonl, one JSON row per ranked pair |
| `--pipeline` | path | required | the pipeline's pipeline.json declaration |
| `--output` | path | required | directory that receives the checked submission and its metrics |

### `trial-benchmark patient-to-trial submission validate`

recompute an imported submission's checksums and metrics

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--task-dir` | path | required | task directory export wrote; it may hold input.json only |
| `--evaluator-dir` | path | required | evaluator directory export wrote |
| `--submission-dir` | path | required | directory import wrote |

### `trial-benchmark patient-to-trial queries`

prepare the licensed local folded-query transformation

### `trial-benchmark patient-to-trial queries prepare`

fold the TREC 2021 queries of one Task Input with a local SNOMED CT release and write the folded-query bundle

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--data-dir` | path | required | directory written by data prepare for this Track |
| `--clinical-as-of` | ISO 8601 time with offset | required | clinical as-of time, bound into the Task Input ID: another value is another Task Input |
| `--profile` | text | `official-full` | TREC 2021 Profile whose Task Input the queries are folded for |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer |  | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest |  | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest |  | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest |  | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |
| `--snomed-release` | path | required | your SNOMED CT RF2 release directory, which must be the SNOMED CT International Edition, 2026-08-01. It is licensed data you supply under your own SNOMED CT licence; TAIM ships none. It is read, never copied into an output |
| `--output` | path | required | file that receives the folded-query bundle |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; judgment-union and external-fidelity Profiles require them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |

### `trial-benchmark patient-to-trial trialgpt2`

qualify TrialGPT 2.0 on TREC 2021, and validate its qualification runs

### `trial-benchmark patient-to-trial trialgpt2 qualify`

run one qualification gate over TREC 2021's external-fidelity frozen retrieval and write its record; makes provider calls, computes no effectiveness value

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--gate` | one of `validation`, `repeat` | required | validation, the one-topic run, or repeat, one of the three ten-topic runs |
| `--run-id` | text | required | a new run ID for this gate run |
| `--data-dir` | path | required | directory written by data prepare for this Track |
| `--clinical-as-of` | ISO 8601 time with offset | required | clinical as-of time, bound into the Task Input ID: another value is another Task Input |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer | required | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest | required | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest | required | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest | required | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |
| `--trialgpt-frozen-retrieval` | path |  | fresh frozen retrieval artifact for this Task Input, from the retrieval producer; required by TrialGPT |
| `--trialgpt-frozen-retrieval-lock` | path |  | the lock written with that artifact; required by TrialGPT |
| `--codex-executable` | text | `codex` | Codex CLI that makes the provider calls; name it bare, not by absolute location, or the run cannot be bundled |
| `--trialgpt2-checkout` | path |  | a clean checkout of TrialGPT 2.0 at the pinned commit; required by TrialGPT 2.0 |
| `--trialgpt2-workspace` | path |  | an empty directory, or the workspace of an interrupted TrialGPT 2.0 run to resume in place; omitted, .trialgpt2/<run ID> under --output-dir |
| `--trialgpt2-generation-workers` | positive integer | `48` | TrialGPT 2.0 candidate pairs scored at once, 1 to 48 |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; qualification requires them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |

### `trial-benchmark patient-to-trial trialgpt2 validate-qualification`

re-account for gate runs from the files they left: one validation record, or the three repeat records; reads no judgment

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--gate` | one of `validation`, `repeat` | required | the gate validated |
| `--record` | path | required | a gate run's trialgpt2-qualification.json; once for validation, three times for the repeats |

### `trial-benchmark patient-to-trial trialgpt`

prepare the provider-free TrialGPT execution contract, qualify TrialGPT on TREC 2021, and validate its qualification runs

### `trial-benchmark patient-to-trial trialgpt preflight`

check a frozen retrieval artifact against one Task Input and write the publication contract a TrialGPT run must match; makes no provider call

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022` | `trec-ct-2021` | Track of the run the contract is for |
| `--profile` | text | `trec-ct-2021-external-fidelity-26149` | Profile of the run the contract is for |
| `--data-dir` | path | required | directory written by data prepare for this Track |
| `--clinical-as-of` | ISO 8601 time with offset | required | clinical as-of time, bound into the Task Input ID: another value is another Task Input |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer | required | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest | required | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest | required | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest | required | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |
| `--frozen-retrieval` | path | required | fresh frozen retrieval artifact for this Task Input, from the retrieval producer |
| `--frozen-retrieval-lock` | path | required | the lock written with that artifact |
| `--generation-workers` | positive integer | `48` | concurrent provider calls the run will make, 1 to 48 |
| `--codex-executable` | text | `codex` | Codex CLI that makes the provider calls; name it bare, not by absolute location, or the run cannot be bundled |
| `--output` | path | required | file that receives the publication contract |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; preflight requires them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |

### `trial-benchmark patient-to-trial trialgpt qualify`

run one qualification gate over TREC 2021's external-fidelity frozen retrieval and write its record; makes provider calls, computes no effectiveness value

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--gate` | one of `validation`, `repeat` | required | validation, the one-topic run, or repeat, one of the three ten-topic runs |
| `--run-id` | text | required | a new run ID for this gate run |
| `--data-dir` | path | required | directory written by data prepare for this Track |
| `--clinical-as-of` | ISO 8601 time with offset | required | clinical as-of time, bound into the Task Input ID: another value is another Task Input |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer | required | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest | required | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest | required | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest | required | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |
| `--trialgpt-frozen-retrieval` | path |  | fresh frozen retrieval artifact for this Task Input, from the retrieval producer; required by TrialGPT |
| `--trialgpt-frozen-retrieval-lock` | path |  | the lock written with that artifact; required by TrialGPT |
| `--codex-executable` | text | `codex` | Codex CLI that makes the provider calls; name it bare, not by absolute location, or the run cannot be bundled |
| `--trialgpt-publication-contract` | path |  | contract written by trialgpt preflight; the run must match it exactly. Required by TrialGPT |
| `--trialgpt-workspace` | path |  | an empty directory, or the workspace of an interrupted run to resume in place; omitted, .trialgpt/<run ID> under --output-dir |
| `--trialgpt-generation-workers` | positive integer | `48` | concurrent provider calls, 1 to 48; must equal the preflight contract's |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; qualification requires them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |

### `trial-benchmark patient-to-trial trialgpt validate-qualification`

re-account for gate runs from the files they left: one validation record, or the three repeat records; reads no judgment

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--gate` | one of `validation`, `repeat` | required | the gate validated |
| `--record` | path | required | a gate run's trialgpt-qualification.json; once for validation, three times for the repeats |

### `trial-benchmark patient-to-trial trialmatchai`

cut or identify a TrialMatchAI prepared corpus, or build or check what the installed TrialMatchAI Systems need before a run

### `trial-benchmark patient-to-trial trialmatchai filter-corpus`

cut a TrialMatchAI-prepared corpus to exactly a Snapshot's trial IDs, linking its prepared files, and write filter-manifest.json

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--snapshot-trials` | path | required | Snapshot trial records, JSON Lines with a trial_id on each |
| `--source-trials` | path | required | TrialMatchAI's processed_trials/, one JSON file per trial named by its trial ID |
| `--source-criteria` | path | required | TrialMatchAI's processed_criteria/, one folder per trial named by its trial ID |
| `--output` | path | required | new corpus folder that receives processed_trials/, processed_criteria/ and the manifest |
| `--reuse-existing` | none |  | complete an existing output, refusing any file whose contents differ |

### `trial-benchmark patient-to-trial trialmatchai adopt-corpus`

hash a prepared corpus once and write its preparation receipt beside its folders

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--corpus-dir` | path | required | prepared corpus holding processed_trials/ and processed_criteria/ |
| `--workers` | positive integer | `8` | files hashed concurrently |

### `trial-benchmark patient-to-trial trialmatchai build-env`

install exactly the extras the frozen config's backends need into the pinned checkout's environment, verify them, and write an environment receipt

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--checkout` | path | required | the clean pinned TrialMatchAI checkout |
| `--receipt` | path | required | file that receives the environment receipt, outside the checkout |

### `trial-benchmark patient-to-trial trialmatchai build-concept-store`

build the concept store from your own licensed OMOP vocabulary files in the pinned checkout's environment, with its provenance manifest; resumes an interrupted build

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--checkout` | path | required | the clean pinned TrialMatchAI checkout, its environment built by build-env |
| `--concept-csv` | path | required | CONCEPT.csv of your own OMOP vocabulary download, tab-delimited |
| `--synonym-csv` | path | required | CONCEPT_SYNONYM.csv of the same download |
| `--source-release` | text | required | the vocabulary release the files came from, recorded in the manifest |
| `--db-path` | path | required | directory that receives the concept store, outside the checkout |
| `--work-dir` | path | required | directory for the downloaded open dictionaries and the embedding checkpoint, outside the checkout; a build resumes from it |
| `--force` | none |  | delete the checkpoint and the store and embed every row again |

### `trial-benchmark patient-to-trial trialmatchai preflight`

check, loading no model, that this environment can run the TrialMatchAI L4 pipeline; the last line is the total

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--system` | `SYSTEM` | required | the TrialMatchAI System the run will run; support list reports the Systems |
| `--trialmatchai-cwd` | path | required | the pinned TrialMatchAI checkout the run will use |
| `--trialmatchai-corpus-dir` | path | required | the prepared corpus the run will use |
| `--trialmatchai-search-db-path` | path |  | the search database the run will use; omitted, search/ in the corpus |

### `trial-benchmark patient-to-trial trialmatchai stage-models`

download the six models of TrialMatchAI v0.9.1 at their pinned revisions into one Hugging Face hub cache, in the pinned checkout's environment, and point each refs/main at its pin

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--checkout` | path | required | the clean pinned TrialMatchAI v0.9.1 checkout, with its environment built |
| `--cache` | path | required | the Hugging Face hub cache a run names in HF_HUB_CACHE, outside the checkout |

### `trial-benchmark patient-to-trial trialmatchai verify-concept-store`

verify a TrialMatchAI v0.9.1 concept store against its source snapshot's receipt, in the pinned checkout's environment, and write the manifest a run binds

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--checkout` | path | required | the clean pinned TrialMatchAI v0.9.1 checkout, with its environment built |
| `--concept-cache` | path | required | the concept cache the store was built from: raw/ with the six source files, and the converted dict_<domain>.txt files |
| `--store` | path | required | the concept store, the directory a run names in TRIALMATCHAI_CONCEPT_DB_PATH |
| `--output` | path | required | file that receives the manifest; a run binds it as concept-store-manifest.json at the store's root |

## `trial-benchmark trial-to-patient`

trial query against patient corpus

### `trial-benchmark trial-to-patient fixture`

run deterministic conformance data

### `trial-benchmark trial-to-patient fixture run`

run one System on the packaged synthetic fixture; its scores are conformance evidence, not effectiveness

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | `trec-ct-2021` | Track whose synthetic fixture to run |
| `--run-id` | text |  | name of the run and of its directory under --output-dir; omitted, public-<System>-<UTC time> |
| `--output-dir` | path | `runs` | directory that receives the run directory |
| `--top-k` | positive integer | `20` | ranking depth per query, the run's budget K; bound into the System Input ID |
| `--metric-cutoff` | positive integer | `5` | primary metric cutoff, bound into the System Input ID; a Result Bundle admits only the Profile's cutoffs |
| `--k1` | number | `1.2` | BM25 term-frequency saturation parameter |
| `--b` | number | `0.75` | BM25 document-length normalisation |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; only runs given all three can enter a Result Bundle |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |

### `trial-benchmark trial-to-patient benchmark`

run a qualified real-data profile

### `trial-benchmark trial-to-patient benchmark run`

run the TREC 2021 reverse viability Profile on prepared TREC 2021 files

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--run-id` | text |  | name of the run and of its directory under --output-dir; omitted, public-<System>-<UTC time> |
| `--output-dir` | path | `runs` | directory that receives the run directory |
| `--top-k` | positive integer | `20` | ranking depth per query, the run's budget K; bound into the System Input ID |
| `--metric-cutoff` | positive integer | `5` | primary metric cutoff, bound into the System Input ID; a Result Bundle admits only the Profile's cutoffs |
| `--k1` | number | `1.2` | BM25 term-frequency saturation parameter |
| `--b` | number | `0.75` | BM25 document-length normalisation |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; only runs given all three can enter a Result Bundle |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |
| `--data-dir` | path | required | directory written by data prepare for this Track |
| `--clinical-as-of` | ISO 8601 time with offset | required | clinical as-of time, bound into the Task Input ID: another value is another Task Input |

### `trial-benchmark trial-to-patient run`

validate or evaluate a closed run

### `trial-benchmark trial-to-patient run validate`

check a run directory against its manifest and print its identities

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--run-dir` | path | required | run directory a run command wrote |

### `trial-benchmark trial-to-patient run evaluate`

validate a run directory, then score its Primary Ranking

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--run-dir` | path | required | run directory a run command wrote |

### `trial-benchmark trial-to-patient submission`

export inputs and evaluate an external pipeline's rankings

### `trial-benchmark trial-to-patient submission export`

write the Task Input a pipeline ranks and, apart from it, the judgments that score it

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | `trec-ct-2021` | Track of the task to export |
| `--fixture` | none |  | export packaged synthetic data without downloads |
| `--data-dir` | path |  | directory written by data prepare; required without --fixture |
| `--profile` | text |  | Benchmark Profile of the task; required without --fixture |
| `--clinical-as-of` | ISO 8601 time with offset |  | clinical as-of time, bound into the Task Input ID; required without --fixture |
| `--pool-file` | path |  | trial IDs of the external-fidelity pool, as a JSON array or one ID per line, written by data external-pool prepare; required on that Profile |
| `--pool-count` | positive integer |  | pool_count from the Profile's data pool inspect receipt. Judgment-union and external-fidelity Profiles require it, --pool-ids-sha256 and --pool-receipt-id; full-corpus Profiles refuse all three |
| `--pool-ids-sha256` | `sha256:` digest |  | pool_ids_sha256 from the same receipt |
| `--pool-receipt-id` | `sha256:` digest |  | pool_receipt_id from the same receipt |
| `--protocol-approval-id` | `sha256:` digest |  | SHA-256 of the frozen protocol governing the run: the external-baseline protocol on the external-fidelity Profile and for external baseline Systems, the paper protocol otherwise. Required on judgment-union, external-fidelity and TREC 2022/2023 confirmation Profiles, refused on any other |
| `--release-manifest` | path |  | frozen manifest of the running release, given with --package-artifact and --dependency-lock or not at all; every export without --fixture requires them |
| `--package-artifact` | path |  | the release's wheel; the installed package must match it file by file |
| `--dependency-lock` | path |  | public uv.lock whose bytes are bound by the release manifest |
| `--top-k` | positive integer | `1000` | depth the pipeline must rank each query to |
| `--output` | path | required | new task directory: input.json only, the one file the pipeline reads |
| `--evaluator-output` | path | required | new evaluator directory: evaluation-package.json only; keep it from the pipeline |

### `trial-benchmark trial-to-patient submission import`

check a pipeline's ranking and declaration against the exported task and score it

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--task-dir` | path | required | task directory export wrote; it may hold input.json only |
| `--evaluator-dir` | path | required | evaluator directory export wrote |
| `--ranking` | path | required | the pipeline's ranking.jsonl, one JSON row per ranked pair |
| `--pipeline` | path | required | the pipeline's pipeline.json declaration |
| `--output` | path | required | directory that receives the checked submission and its metrics |

### `trial-benchmark trial-to-patient submission validate`

recompute an imported submission's checksums and metrics

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--task-dir` | path | required | task directory export wrote; it may hold input.json only |
| `--evaluator-dir` | path | required | evaluator directory export wrote |
| `--submission-dir` | path | required | directory import wrote |

## `trial-benchmark data`

prepare checksum-locked local source files

### `trial-benchmark data prepare`

verify acquired source files against the Track's packaged lock and write the prepared files every run reads

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | required | Track to prepare |
| `--query-profile` | text |  | SIGIR 2016 query text, description or summary; required there, ignored elsewhere |
| `--source` | path | required | directory holding the Track's acquired source files, as its acquisition guide names them |
| `--output-dir` | path | required | directory that receives the prepared files |
| `--protocol-approval-id` | `sha256:` digest |  | required before a TREC 2022/2023 source file is opened |

### `trial-benchmark data pool`

inspect a compute-bounded corpus before effectiveness execution

### `trial-benchmark data pool inspect`

print the receipt of a judgment-union or external-fidelity pool, whose values every run on that Profile must pass

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--track` | one of `sigir-ct-2016`, `trec-ct-2021`, `trec-ct-2022`, `trec-ct-2023` | required | Track of the prepared files |
| `--profile` | text | required | judgment-union or external-fidelity Profile to inspect |
| `--data-dir` | path | required | directory written by data prepare |
| `--pool-file` | path |  | trial IDs written by data external-pool prepare; external-fidelity Profile only |
| `--protocol-approval-id` | `sha256:` digest |  | SHA-256 of the frozen protocol that governs the Profile; always required |

### `trial-benchmark data external-pool`

verify an external source lock and extract ID-only pool membership

### `trial-benchmark data external-pool prepare`

check an external corpus against its packaged lock and write its trial IDs, sorted, as a JSON array

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--profile` | one of `trec-ct-2021-external-fidelity-26149` | required | external-fidelity Profile whose source to verify |
| `--source` | path | required | the external corpus file its acquisition guide names |
| `--output` | path | required | file that receives the trial IDs |

## `trial-benchmark release`

validate a generated public candidate

### `trial-benchmark release validate`

check a candidate release tree against its frozen manifest

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--candidate` | path | required | root of the candidate release tree |
| `--manifest` | path | required | the candidate's frozen release manifest |
| `--runtime` | none |  | also execute the manifest's quick-start commands with this executable |

## `trial-benchmark bundle`

validate a Published Result Bundle

### `trial-benchmark bundle validate`

recompute a Published Result Bundle's checksums, identities and metrics

| Option | Value | Default | Description |
| --- | --- | --- | --- |
| `--bundle-dir` | path | required | the Result Bundle directory |

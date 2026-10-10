# TrialGPT three-query consensus retrieval

Benchmark Release 0.1 ships the complete producer for the four TrialGPT rows frozen by protocols
v5 through v7 and v11. The contract is `trialgpt-three-luna-consensus-v2`: three Luna query plans,
`taim-controlled-bm25-v2` and pinned MedCPT retrieval to depth 2,000, fixed three-query consensus,
and a 500-candidate handoff to `TrialGPT-TAIM-Luna-v1`. BM25 v2 corrects the older v1
implementation's document-frequency defect; the two variants are not interchangeable.

Here, the three inputs to consensus are query-plan variants for the same Topic.
Use "three-query consensus" for this stage. Frozen contracts and artifact fields
retain their existing names, including `trialgpt-three-luna-consensus-v2` and `arms`.

The supported targets are:

| `--track` | `--profile` | Topics |
| --- | --- | ---: |
| `trec-ct-2021` | `trec-ct-2021-external-fidelity-26149` | 75 |
| `trec-ct-2022` | `trec-ct-2022-judgment-union` | 50 |
| `sigir-ct-2016` | `sigir-ct-2016-description-judgment-union` | 60 |
| `sigir-ct-2016` | `sigir-ct-2016-summary-judgment-union` | 60 |

Any other pair, including TREC 2023, fails closed. The SIGIR description and summary rows require
their matching prepared packages.

## 1. Inspect and freeze pool receipts

Record the count, ordered-ID SHA-256, and receipt ID of each target's prepared package separately. Protocol roles
remain distinct:

- inspect the TREC 2021 external-fidelity pool under v4,
  `sha256:55bb91f69d6494d58eff62b96f65f7b59759de45a366dc551e4ad5586fdfc896`; and
- inspect TREC 2022 and both SIGIR judgment-union pools under v3,
  `sha256:18a9c9bd3c76455038d01458a5cb2d933059d1e72924285e26500a8aaab1baf6`.

TREC 2021 also requires the frozen 26,149-ID pool file. Judgment-union targets derive membership
from the Evaluation Package and reject `--pool-file`. Protocol v7 is not a pool-inspection
protocol.

## 2. Build the complete-eligibility package

`TrialGPT-TAIM-Luna-v1` reads each trial's complete eligibility text and splits it into criteria.
A prepared package keeps only a bounded excerpt of long source values, so it lacks that text. This
command re-reads the target's pool trials from the acquired Source Bundle with their complete
eligibility text, requires each to reproduce the prepared package's record once that text is
removed, and writes them with the criterion Derived View as a new prepared package. It makes no
provider call.

```console
python scripts/prepare_trialgpt_profile_snapshot.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --data-dir /path/to/prepared/trec-ct-2022 \
  --source /path/to/trec-2022-files \
  --output-dir /path/to/prepared/trec-ct-2022-complete-eligibility \
  --pool-count <frozen-count> \
  --pool-ids-sha256 sha256:<ordered-id-digest> \
  --pool-receipt-id sha256:<receipt-id> \
  --clinical-as-of 2022-01-01T00:00:00Z
```

Pass the receipt values of step 1, and add `--pool-file` only for the TREC 2021 target. The new
package keeps the prepared package's topics, trial IDs and order, judgments and pool. Its Prepared
Snapshot, Evaluation Package, pool receipt and Task Input are new; the command prints them. Use
this package and its receipt in every step below: query generation, the freeze and preflight
refuse a package without complete eligibility text before any provider call or retrieval.

## 3. Generate query plans

This phase reads only the prepared topic file and verifies its manifest before making provider
calls. Across all four targets it makes 735 logical Luna calls. Use a separate run ID, cache, and
output path for each target.

```console
python scripts/generate_trialgpt_retrieval_queries.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --prepared-manifest /path/to/prepared/trec-ct-2022-complete-eligibility/prepared-manifest.json \
  --run-id paper-trialgpt-trec22-query-1 \
  --cache /path/to/local-workspace/trec22-query-generation.sqlite3 \
  --output /path/to/local-workspace/trec22-three-luna-query-plans.jsonl \
  --workers 8
```

Rerunning with the same cache resumes identical completed calls. The command never reads qrels or
trial text and refuses to overwrite its output.

Protocol v6 Section 2 admitted exactly four query-plan files by SHA-256, one per target:

| `--profile` | Admitted query-plan SHA-256 |
| --- | --- |
| `trec-ct-2021-external-fidelity-26149` | `sha256:1a60326de204673575d37bd68e7e79ca73df5bb349c6c781800d8bcdfe760641` |
| `trec-ct-2022-judgment-union` | `sha256:f7745291be02242b5e2dd89ddfe6607343c2e65bb93e78f5c13bab3ffaac5217` |
| `sigir-ct-2016-description-judgment-union` | `sha256:90a1b7ec70f3d2cde21e815fb980e3287b7047523e94ca0cce45bbc03b9b5ca4` |
| `sigir-ct-2016-summary-judgment-union` | `sha256:ba2f8c28315913b4405d1f8654208d0297d78fb368b4258dfa07f95dd2ef4e16` |

These four files are not available. This project does not hold them, and they cannot be recreated,
because each one records hashes of the model's own outputs. So no one can produce a retrieval from
them.

Query plans are therefore generated for each run with the command above, and recorded by a
query-plan receipt written and committed before the freeze, as protocol v11 defines it
([`taim-paper-analysis-protocol-2026-09-28-v11`](paper-analysis-protocol-2026-09-28-v11.md)).
The freeze requires that receipt and binds protocol v11. Its lock carries the receipt and the plans'
own digest, so a retrieval is never reported as the admitted plans. The freeze refuses plans
without a receipt that describes them before it computes anything, and its refusal names the four
digests above.

### 3a. Before resuming an interrupted generation

A resume returns a cached sign-in failure unchanged instead of asking again. So before every
resume, while no process holds the cache, run the check of protocol v11 Section 7.1:

```console
python scripts/clear_trialgpt_failed_calls.py \
  --stage query \
  --run-id paper-trialgpt-trec22-query-1 \
  --cache /path/to/local-workspace/trec22-query-generation.sqlite3 \
  --cause signed-out \
  --record-dir /path/to/receipts \
  --interruptions /path/to/local-workspace/trec22-interruptions.json
```

- **What it removes.** Only unresolved entries whose every attempt is a failed sign-in or transport
  call with no provider output. It never removes an entry that holds a model response.
- **What it records.** A clearing record named by the run ID, with the cache's SHA-256 before and
  after and each removed entry verbatim. It appends the interruption, with its time, cause and
  record, to the interruptions file the receipt reads. Commit the records with the receipt.
- **When it refuses.** An unresolved entry may hold provider output that a resume could not retry
  or would overwrite. Then the check clears nothing, still writes its record, and exits 2. The
  generation stops, and the refusal is disclosed.
- **The cause is generic.** `--cause` takes `usage-allowance`, `signed-out`, `provider-failure`,
  `host-restart`, `interrupted-process` or `codex-version-change`.

The same check runs before a `TrialGPT-TAIM-Luna-v1` row resumes: pass `--stage system --system
TrialGPT-TAIM-Luna-v1 --workspace <its --trialgpt-workspace>` and the row's run ID, in place of
`--stage query` and `--cache`.

### 3b. Write and commit the receipt

Before the freeze, write the protocol v11 Section 6 receipt from the finished generation and
commit it; the freeze may not start before that commit exists.

```console
python scripts/write_trialgpt_query_plan_receipt.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --prepared-manifest /path/to/prepared/trec-ct-2022-complete-eligibility/prepared-manifest.json \
  --run-id paper-trialgpt-trec22-retrieval-1 \
  --generation-run-id paper-trialgpt-trec22-query-1 \
  --cache /path/to/local-workspace/trec22-query-generation.sqlite3 \
  --query-plans /path/to/local-workspace/trec22-three-luna-query-plans.jsonl \
  --release-manifest /path/to/release-manifest.json \
  --package-artifact /path/to/release-wheel.whl \
  --interruptions /path/to/local-workspace/trec22-interruptions.json \
  --output /path/to/receipts/paper-trialgpt-trec22-retrieval-1.json
```

`--run-id` is the freeze's run ID; `--cache` is the value the generation was given. Omit
`--interruptions` for an uninterrupted generation; otherwise pass the file the checks of step 3a
appended to. The command reads the cache without changing it, refuses plans with no resolved call in it, writes once, and applies the freeze's own
receipt checks.

## 4. Run retrieval and freeze evidence

Use a clean checkout of this release's source commit and its verified wheel. The
prepared-package loader is validated on an 18 GB Apple-silicon laptop. That memory result covers
loading and bounded-pool projection. MedCPT runs on a CUDA card for which the installed torch
wheel carries a kernel (the locked build covers compute capability 7.5 and newer, so not a
V100), with an NVIDIA driver of the 580 line or newer; `--device cpu` is also accepted, and
slow. The lock records the hardware its embeddings were made on. Install the locked `dense`
extra (`uv sync --locked --extra dense`) for the pinned MedCPT runtime. Freezing makes no
provider calls.

```console
python scripts/freeze_trialgpt_retrieval.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --data-dir /path/to/prepared/trec-ct-2022-complete-eligibility \
  --pool-count <frozen-count> \
  --pool-ids-sha256 sha256:<ordered-id-digest> \
  --pool-receipt-id sha256:<receipt-id> \
  --protocol-approval-id sha256:ac8be7fc8a618acf282e66ccae6ddb716e75b9cdcb4ccd753bae45378496348b \
  --release-manifest /path/to/release-manifest.json \
  --package-artifact /path/to/release-wheel.whl \
  --clinical-as-of 2022-01-01T00:00:00Z \
  --query-plans /path/to/local-workspace/trec22-three-luna-query-plans.jsonl \
  --query-plan-receipt /path/to/receipts/paper-trialgpt-trec22-retrieval-1.json \
  --run-id paper-trialgpt-trec22-retrieval-1 \
  --workspace /path/to/cluster-workspace/trec22-medcpt \
  --model-cache /path/to/huggingface-cache \
  --batch-size 16 \
  --offline \
  --output-dir /path/to/cluster-workspace/trec22-frozen-retrieval
```

Add `--pool-file /path/to/trialgpt-paper-pool-26149.txt` only for the TREC 2021 target. The command
checks the query-plan receipt first, before anything is computed: its run ID, target, plan digest
and protocol, and later its release and Prepared Snapshot. It then checks the Track/Profile pair,
complete topic set, pool receipt, Snapshot, Evaluation Package, Task Input, patient-text hashes,
release IDs, source commit, lockfile, MedCPT revisions, and every hash-bound sidecar.

### 4b. Recompute and commit

After the freeze and before any effectiveness value is read, recompute the plans from the retained
cache (protocol v11 Section 8). The query method runs against a copy of the cache with Models that
carry the generation's configuration, Codex executable included, and a transport that refuses
every call; the installed Codex CLI must be the version the receipt records. It writes its result
beside the receipt, passed or failed, as a JSON file named by the run ID with the suffix
`.recomputation.json`, and exits non-zero unless the plan file rebuilt byte for byte. Commit it
beside the receipt. A failed recomputation stops the row. It needs no GPU.

```console
python scripts/recompute_trialgpt_query_plans.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --prepared-manifest /path/to/prepared/trec-ct-2022-complete-eligibility/prepared-manifest.json \
  --cache /path/to/local-workspace/trec22-query-generation.sqlite3 \
  --query-plans /path/to/local-workspace/trec22-three-luna-query-plans.jsonl \
  --query-plan-receipt /path/to/receipts/paper-trialgpt-trec22-retrieval-1.json
```

## 5. Run and admit the System

Pass the ranking and lock to `trial-benchmark patient-to-trial trialgpt preflight`, then execute
`TrialGPT-TAIM-Luna-v1`. The primary row uses 500 candidates and up to 1,500 logical System calls
per topic. Preflight and System execution use protocol v4,
`sha256:55bb91f69d6494d58eff62b96f65f7b59759de45a366dc551e4ad5586fdfc896`; protocol v11 is used
only by the fresh retrieval producer. Preflight and System execution refuse a retrieval lock that
does not bind protocol v11 and carry the receipt that describes its query plans. Keep all query
plans, query-plan receipts, generation caches, raw channels, source rankings, consensus features,
provider traces, and locks in the Local Run. Evaluate only after preflight and production
validation pass. Published Result Bundles exclude raw clinical text, provider traces, model files,
and local retrieval indexes.

# TrialGPT retrieval producer

Benchmark Release 0.1.0 ships the complete producer for the four TrialGPT rows frozen by protocols
v5 through v7. The contract is `trialgpt-three-luna-consensus-v2`: three Luna query plans,
`taim-controlled-bm25-v2` and pinned MedCPT retrieval to depth 2,000, fixed three-arm consensus,
and a 500-candidate handoff to `TrialGPT-TAIM-Luna-v1`. BM25 v2 corrects the older v1
implementation's document-frequency defect; the two variants are not interchangeable.

The supported targets are:

| `--track` | `--profile` | Topics |
| --- | --- | ---: |
| `trec-ct-2021` | `trec-ct-2021-external-fidelity-26149` | 75 |
| `trec-ct-2022` | `trec-ct-2022-judgment-union` | 50 |
| `sigir-ct-2016` | `sigir-ct-2016-description-judgment-union` | 60 |
| `sigir-ct-2016` | `sigir-ct-2016-summary-judgment-union` | 60 |

Any other pair, including TREC 2023, fails closed. The SIGIR description and summary rows require
their matching prepared packages.

## 1. Generate query plans

This phase reads only the prepared topic file and verifies its manifest before making provider
calls. Across all four targets it makes 735 logical Luna calls. Use a separate run ID, cache, and
output path for each target.

```console
python scripts/generate_trialgpt_retrieval_queries.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --prepared-manifest /path/to/prepared/trec-ct-2022/prepared-manifest.json \
  --run-id paper-v0.1.0-trialgpt-trec22-query-v2 \
  --cache /path/to/local-workspace/trec22-query-generation.sqlite3 \
  --output /path/to/local-workspace/trec22-three-luna-query-plans.jsonl \
  --workers 8
```

Rerunning with the same cache resumes identical completed calls. Protocol v7 retains the four
already frozen query-plan files by exact SHA-256, so those 735 calls do not need to be repeated.
The command never reads qrels or trial text and refuses to overwrite its output.

## 2. Inspect and freeze pool receipts

Record the count, ordered-ID SHA-256, and receipt ID separately for every target. Protocol roles
remain distinct:

- inspect the TREC 2021 external-fidelity pool under v4,
  `sha256:55bb91f69d6494d58eff62b96f65f7b59759de45a366dc551e4ad5586fdfc896`; and
- inspect TREC 2022 and both SIGIR judgment-union pools under v3,
  `sha256:18a9c9bd3c76455038d01458a5cb2d933059d1e72924285e26500a8aaab1baf6`.

TREC 2021 also requires the frozen 26,149-ID pool file. Judgment-union targets derive membership
from the Evaluation Package and reject `--pool-file`. Protocol v7 is not a pool-inspection
protocol.

## 3. Run retrieval and freeze evidence

Use a clean checkout of the exact public 0.1.0 source commit and the verified 0.1.0 wheel. The
prepared-package loader is validated on an 18 GB Apple-silicon laptop. That memory result covers
loading and bounded-pool projection; MedCPT execution still needs a compatible local or cluster
runtime. Install the candidate's locked `dense` extra (`uv sync --locked --extra dense`) so the
pinned MedCPT runtime is present. The freeze phase makes no provider calls.

```console
python scripts/freeze_trialgpt_retrieval.py \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --data-dir /path/to/prepared/trec-ct-2022 \
  --pool-count <frozen-count> \
  --pool-ids-sha256 sha256:<ordered-id-digest> \
  --pool-receipt-id sha256:<receipt-id> \
  --protocol-approval-id sha256:a617a46c2d61e76ad25963e709cad2c3ff99d4be466f173b8add02f1754927ef \
  --release-manifest /path/to/benchmark-v0.1.0-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --clinical-as-of 2022-01-01T00:00:00Z \
  --query-plans /path/to/local-workspace/trec22-three-luna-query-plans.jsonl \
  --run-id paper-v0.1.0-trialgpt-trec22-retrieval-v2 \
  --workspace /path/to/cluster-workspace/trec22-medcpt \
  --model-cache /path/to/huggingface-cache \
  --batch-size 16 \
  --offline \
  --output-dir /path/to/cluster-workspace/trec22-frozen-retrieval
```

Add `--pool-file /path/to/trialgpt-paper-pool-26149.txt` only for the TREC 2021 target. The command
checks the Track/Profile pair, complete topic set, pool receipt, Snapshot, Evaluation Package, Task
Input, patient-text hashes, release IDs, source commit, lockfile, MedCPT revisions, and every
hash-bound sidecar.

## 4. Run and admit the System

Pass the ranking and lock to `trial-benchmark patient-to-trial trialgpt preflight`, then execute
`TrialGPT-TAIM-Luna-v1`. The primary row uses 500 candidates and up to 1,500 logical System calls
per topic. Preflight and System execution use protocol v4,
`sha256:55bb91f69d6494d58eff62b96f65f7b59759de45a366dc551e4ad5586fdfc896`; protocol v7 is used only
by the fresh retrieval producer. Keep all query plans, raw channels, source rankings, consensus
features, provider traces, and locks in the Local Run. Evaluate only after preflight and production
validation pass. Published Result Bundles exclude raw clinical text, provider traces, model files,
and local retrieval indexes.

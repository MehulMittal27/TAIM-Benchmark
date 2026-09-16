# TAIM paper analysis protocol: TrialGPT retrieval-producer amendment

**Protocol ID:** `taim-paper-analysis-protocol-2026-09-02-v5`

**Status:** Frozen under repository-owner authorization on 2026-09-02. These bytes are immutable;
later changes require a new dated amendment.

This amendment changes protocol v4 only for the fresh retrieval artifacts and provider budget of
`TrialGPT-TAIM-Luna-v1`. Protocol v4 still governs the experiment matrix, Task Inputs, Evaluation
Packages, metrics, comparisons, and reporting. Protocol v1 Section 8 remains the only canonical
statement of the recall-floor asymmetry.

## 1. Admitted targets

The TrialGPT paper System runs on exactly these Track/Profile pairs:

| Track | Profile | Topics | Pool source |
| --- | --- | ---: | --- |
| TREC Clinical Trials 2021 | `trec-ct-2021-external-fidelity-26149` | 75 | frozen 26,149-trial external-fidelity receipt |
| TREC Clinical Trials 2022 | `trec-ct-2022-judgment-union` | 50 | score-blind union of judged trial IDs |
| SIGIR Clinical Trials 2016 | `sigir-ct-2016-description-judgment-union` | 60 | score-blind union for the description query view |
| SIGIR Clinical Trials 2016 | `sigir-ct-2016-summary-judgment-union` | 60 | score-blind union for the summary query view |

TREC 2023 has no TrialGPT row. The two SIGIR query views remain separate experiments and are never
pooled. TREC eligibility and SIGIR referral labels retain their distinct semantics.

## 2. Release and producer identity

Every row must execute the public TAIM Benchmark 0.1.1 wheel. Before any provider call, release
0.1.1 must be published and independently verified. Each run records its exact release manifest
ID, public tree ID, package artifact ID, clean producer commit, and producer `uv.lock` SHA-256.

The source-side method is `trialgpt-three-luna-consensus-v2`. The release must ship its producer,
both command-line scripts, tests, MedCPT model pins, and approved notices. A dirty checkout,
unpublished source commit, mismatched wheel, unresolved licence, or missing identity stops the run.

## 3. Query generation

For every topic, the producer creates three plans in this fixed order:

1. `medium`: `gpt-5.6-luna`, medium reasoning, prompt `trialgpt-paper-keyword-v1`;
2. `xhigh`: `gpt-5.6-luna`, xhigh reasoning, the same prompt;
3. `recall-explicit`: `gpt-5.6-luna`, medium reasoning, prompt
   `trialgpt-recall-explicit-v2`.

Each plan contains one summary and at most 32 ranked conditions. The recall-explicit prompt may use
only facts stated or directly supported by the patient description. Provider configuration,
prompt ID, patient-text SHA-256, logical-call ID, attempt trace, and output SHA-256 are retained.
Cache reuse is allowed only for an identical generation identity.

Across 245 topics this phase has 735 logical provider calls. Query generation sees public patient
text only. It cannot read qrels, judgments, evaluation metrics, or trial-pool scores.

## 4. Retrieval and consensus

Every generated condition retrieves the target Profile through both channels:

- `taim-controlled-bm25-v2` with tokenizer `taim-trialgpt-regex-tokenizer-v1`, BM25 parameters
  `k1=1.2` and `b=0.75`, brief-title terms repeated three times, disease terms repeated twice, and
  document frequency counted once per document; and
- `ncbi/MedCPT-Query-Encoder` revision
  `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` against
  `ncbi/MedCPT-Article-Encoder` revision
  `d05a736da4bb84ee4057b7f7999485be6ed85465`, float32, exact inner product.

Each channel returns depth 2,000. Within one query arm, condition rankings use RRF constant 20 and
condition weight `1 / (zero_based_condition_index + 1)`. Ties use score descending, then trial ID
ascending. Each source arm retains its complete raw channel evidence.

For every candidate in the union of the three source top-2,000 rankings, define:

- `arm_count`: number of source arms containing it;
- `best_rank_quality`: `1 - (best_rank - 1) / 1999`;
- `per_arm_reciprocal_rank`: `1 / rank` when present and zero otherwise, in the frozen arm order;
- `lexical_dense_agreement_count`: arms in which both raw channels exposed it.

The fixed consensus score is:

```text
(6/17) * (arm_count / 3)
+ (4/17) * best_rank_quality
+ (4/17) * (sum(per_arm_reciprocal_rank) / 3)
+ (3/17) * (lexical_dense_agreement_count / 3)
```

Final ties use score descending, then trial ID ascending. The producer freezes 2,000 rows per topic;
the first 500 are the System candidates.

## 5. TrialGPT provider budget

The primary System keeps the publication candidate depth at 500. It makes up to three logical
calls per candidate: inclusion matching, exclusion matching, and aggregation. The frozen ceiling is
therefore 1,500 logical System calls per topic, not 500. Lowering this ceiling changes the System
and may be studied only as a separately named ablation.

The four primary rows have a combined ceiling of 367,500 logical System calls, plus the 735 query
generation calls above. The repository owner authorized protected TREC 2022 access and these
provider-backed experiments for this protocol. No TrialMatchAI, internal baseline, or internal
pipeline run is authorized by this amendment.

## 6. Evidence and admission

Each Local Run retains query plans, raw per-condition rankings, three source rankings, consensus
features, final retrieval, provider traces, run manifest, and all identity locks. The validator
reconstructs the source rankings, recomputes every consensus score, and checks exact topic, pool,
Snapshot, Evaluation Package, Task Input, protocol, release, source commit, and dependency lock
identities.

Do not inspect effectiveness until retrieval production and the TrialGPT publication preflight
both pass. A timeout, unresolved provider response, missing fact, or resource limit is never turned
into ineligibility. Failed or incomplete runs remain Local Runs and cannot enter a paper table.
Only validated, protocol-conforming rows may become Published Result Bundles.

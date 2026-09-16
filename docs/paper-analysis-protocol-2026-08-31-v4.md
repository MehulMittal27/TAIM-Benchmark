# TAIM paper analysis protocol: external-baseline amendment

**Protocol ID:** `taim-paper-analysis-protocol-2026-08-31-v4`

**Status:** Frozen by the repository owner on 2026-09-01, before any effectiveness result from the
external-baseline rows introduced here was inspected. These bytes supersede v3 for paper execution
while retaining protocol v1's metrics, statistical rules, failure rules, stop rules, and claim
boundaries. Protocol v1 Section 8 remains the sole canonical statement of the recall-floor
asymmetry. The accompanying
[v3 supersede record](paper-analysis-protocol-2026-08-31-v3-supersede-note-2026-09-01.md) records
the exact boundary. Approval of this protocol does not authorize provider calls, protected-data
access, result admission, or publication.

## 1. Primary paper matrix retained

The primary experiment remains score-blind within-pool ranking over five separate judgment-union
strata. Raw metrics are not pooled across Tracks or across SIGIR query views.

| Track | Paper Profile | Source Profile | Primary Systems |
| --- | --- | --- | --- |
| TREC Clinical Trials 2021 | `trec-ct-2021-judgment-union` | `official-full` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| TREC Clinical Trials 2022 | `trec-ct-2022-judgment-union` | `confirmation-full` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| TREC Clinical Trials 2023 | `trec-ct-2023-judgment-union` | `confirmation-full` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| SIGIR Clinical Trials 2016 description | `sigir-ct-2016-description-judgment-union` | `description` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| SIGIR Clinical Trials 2016 summary | `sigir-ct-2016-summary-judgment-union` | `summary` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |

`staged-bm25-qwen3-rrf-rerank` remains one additional TREC 2021 primary experiment under
`trec-ct-2021-judgment-union`. Component Systems are supporting method stages, not additional
reportable columns. This yields 21 primary reportable rows.

Each judgment-union Task Input remains every and only trial ID present in its checksum-locked
Evaluation Package, ordered by the complete prepared corpus. Every run reproduces the
pre-effectiveness count, ordered-ID SHA-256, and receipt ID. TREC 2021 requires 26,162 trials. The
other counts are derived only by an authorized operator after immutable release freeze. Unjudged
pairs remain unjudged.

## 2. Separate TREC 2021 external-fidelity stratum

The proposed external-fidelity table uses Profile
`trec-ct-2021-external-fidelity-26149` and exactly seven rows:

| System | Role |
| --- | --- |
| `bm25` | deterministic TAIM reference |
| `dense-bge-m3` | TAIM dense reference |
| `dense-qwen3-embedding-0.6b` | TAIM dense reference |
| `rrf` | TAIM reference fusion |
| `staged-bm25-qwen3-rrf-rerank` | TAIM staged pipeline |
| `TrialGPT-TAIM-Luna-v1` | provider-backed TAIM adaptation of TrialGPT |
| `trialmatchai-current-cuda-l4-trec21-development-v3` | pinned current TrialMatchAI fork through the TAIM adapter |

Together with the primary matrix, this stratum contributes seven reportable rows. Missing,
failed, or unauthorized rows remain missing and are not imputed.

The 26,149 membership is supplied from the separately acquired TrialGPT TREC 2021 corpus at source
SHA-256 `01692c847b2da798c57a8e0a74273ec262a7e42ad3f02b4ff5a87a6442462f9c`. Its 26,149 unique
`_id` values have sorted-ID SHA-256
`fed85fadb5a0e0a42a39ed9cf65984926a13f66d4a2a0a1deb100996ac56ffa7`. It is not derived by
deleting 13 trials from TAIM's 26,162-trial union. Before any score, the CLI verifies
that every supplied ID exists in the recipe-v3 preparation, occurs in the Evaluation Package, and
that prepared-corpus ordering yields count 26,149. It emits the Profile definition hash, Prepared
Snapshot ID, Evaluation Package ID, ordered-ID SHA-256, and receipt ID. Count equality alone does
not establish TrialGPT or TrialMatchAI paper identity.

The external Profile derives one direction-specific Evaluation Package containing every and only
source judgment row whose trial belongs to the 26,149-trial Task Input. Its provenance binds the
complete source Evaluation Package ID. This projection removes out-of-Task-Input judgments; it
does not create judgments for absent pairs, so unjudged pairs remain unjudged.

All seven rows must use the same Task Input, Evaluation Package, pool receipt, release manifest,
public tree, package artifact, and dependency lock. The 26,162 primary table and 26,149
external-fidelity table do not share a comparative table, statistical test, or corpus identity.

## 3. External System identity and limits

The release projects TAIM-owned adapters, pinned source-lock metadata, acquisition instructions,
method contracts, and conformance tests. It does not copy either upstream repository, benchmark
source bytes, model weights, indexes, provider traces, or historical outputs.

`TrialGPT-TAIM-Luna-v1` is a named TAIM adaptation, not an exact reproduction of the upstream GPT-4
paper System. Its run remains separately gated by Issue #93. A fresh retrieval artifact and
publication contract must bind the amended immutable release and exactly 48 generation workers;
historical recipe-v1 artifacts are ineligible.

The selected TrialMatchAI System is the current pinned fork and configuration, not the original
paper implementation. Entity linking and concept-store provenance differ. A fresh final run must
bind the amended immutable release. Historical development runs are not promoted.

Upstream paper scores may appear only as contextual, clearly non-comparable literature values.

## 4. External adaptations on the other compute-bounded Tracks

Five additional rows use TAIM's existing judgment-union Profiles. They do not introduce another
corpus identity or copy a published external count into a TAIM receipt.

| Track and query Profile | Added System | Report role |
| --- | --- | --- |
| TREC Clinical Trials 2022, `trec-ct-2022-judgment-union` | `TrialGPT-TAIM-Luna-v1` | provider-backed TAIM adaptation |
| TREC Clinical Trials 2022, `trec-ct-2022-judgment-union` | `trialmatchai-current-cuda-l4-trec22-development-v3` | pinned current TrialMatchAI fork through the TAIM adapter |
| TREC Clinical Trials 2023, `trec-ct-2023-judgment-union` | `trialmatchai-current-cuda-l4-trec23-development-v3` | later current-repository path, not a paper reproduction |
| SIGIR Clinical Trials 2016 description, `sigir-ct-2016-description-judgment-union` | `TrialGPT-TAIM-Luna-v1` | provider-backed TAIM adaptation over the description query contract |
| SIGIR Clinical Trials 2016 summary, `sigir-ct-2016-summary-judgment-union` | `TrialGPT-TAIM-Luna-v1` | provider-backed TAIM adaptation over the summary query contract |

The primary 21 rows, seven TREC 2021 external-fidelity rows, and these five additions yield 33
unique reportable rows. A row is counted once even when it shares a Task Input with a primary
reference System.

Every added row uses the same Prepared Snapshot, Task Input, Evaluation Package, pool receipt,
topic set, and query representation as the reference rows for that exact Profile. TREC 2022 and
TREC 2023 counts and ordered-ID digests remain locally derived from their authorized Evaluation
Packages. Both SIGIR rows retain all declared topics for their respective public query contracts.
These are TAIM adaptations, so they do not reproduce the external papers' topic subsets, parsed
trial text, retrieval candidates, or reported scores.

TrialGPT requires a fresh track-specific frozen retrieval artifact that passes the same
judgment-blind three-Luna consensus, 2,000-depth, top-500 candidate, exact-topic, and exact-trial
validation used by its TREC 2021 TAIM adaptation. Exactly 48 generation workers remain part of the
System contract. The release provides no TrialGPT TREC 2023 capability because the upstream work
did not evaluate that Track and no TAIM adaptation has been qualified.

All three TrialMatchAI System identities run the same TAIM-owned adapter and frozen L4
configuration against `MehulMittal27/TrialMatchAI` commit
`7eba8f399336fcd988b00ce98f2e025e5ec04119`. The Track name is part of the System identity to
prevent a TREC 2021 run or prepared workspace from being relabeled as TREC 2022 or TREC 2023.
The TREC 2023 row is evidence about this later current-fork path only.

For context, primary-source review found these external initial-pool sizes:

| Track | TrialGPT | TrialMatchAI paper | Later TrialMatchAI repository |
| --- | ---: | ---: | ---: |
| SIGIR 2016 | 3,621 trials, 58 patients | not reported | not reported |
| TREC 2021 | 26,149 trials, 75 topics | 26,149 trials, 75 topics | — |
| TREC 2022 | 26,581 trials, 50 topics | 26,581 trials, 50 topics | — |
| TREC 2023 | not evaluated | not evaluated | 17,103 available of 17,106 judged IDs, 37 judged topics |

TrialGPT describes its initial corpus as the cohort-wide union of judged trials. That selection is
score-blind with respect to the new System but judgment-derived and not exhaustive. Its downstream
top 500 is retrieval-score-derived and cannot be a neutral corpus for comparing retrievers.
TrialMatchAI also passes a retrieved top 500 into later stages.

TAIM's TREC 2022, TREC 2023, and SIGIR judgment-union Profiles provide the score-blind compute
bound for both the reference rows and the five added adaptations. Their identities come from
TAIM's exact source locks and Evaluation Packages. The published counts remain literature context
only. In particular, this protocol does not rename a TAIM pool as the TrialGPT 26,581-trial pool,
the TrialGPT 3,621-trial SIGIR cohort, or the later TrialMatchAI 17,103-trial TREC 2023 corpus.

## 5. Execution, reporting, and stop rules

Every final row must be a fresh run from the amended immutable public release and bind its Track,
Task, Profile, Prepared Snapshot ID, Task Input ID, System Input ID, Evaluation Package ID, pool receipt ID,
Primary Ranking, release manifest, public tree, package artifact, and dependency environment.

Tables and prose use **within-pool ranking**. They do not claim full-corpus retrieval, official
TREC-run reproduction, exhaustive retrieval, a new dataset, upstream equivalence, clinical
validity, eligibility, deployment safety, enrollment benefit, or patient benefit. SIGIR labels
remain referral-oriented. Patient-to-trial and trial-to-patient never share a leaderboard.

Stop before execution if the amended release is not immutable, a pool receipt is absent or
mismatched, a source lock fails, a System is absent from the support matrix, external membership
cannot be reproduced, a protected Track lacks the approved protocol digest, or producer identities
cannot be bound. Do not repair identity failures after inspecting scores.

TREC 2022/2023 access remains behind the protected-data gate. TrialGPT remains behind separate
provider authorization. TrialMatchAI requires separately provisioned licensed models, concept
store, and index. Google Gemma 2 terms, the LoRA and exact-snapshot license caveats, terminology
licenses, provider terms, and the amended third-party notice require explicit human review. This
protocol authorizes none of them.

`published_result_bundles` remains empty in the source release. Historical campaign, recipe-v1,
provider, and development outputs are not retrofitted.

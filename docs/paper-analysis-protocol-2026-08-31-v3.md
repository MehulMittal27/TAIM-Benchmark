# TAIM paper analysis protocol: multi-track compute-bounded amendment

**Protocol ID:** `taim-paper-analysis-protocol-2026-08-31-v3`

**Status:** Frozen by the repository owner on 2026-08-31, before any effectiveness result from
the Profiles introduced here was inspected. This amendment governs the paper's Track, Profile,
System, and corpus-selection matrix. It supersedes the paper-use of the full-corpus identities in
project protocol v2, but it does not remove those full-corpus Profiles from Benchmark Release 0.1.
The registered metrics, statistical rules, failure rules, stop rules, and claim boundaries in
project protocol v1 remain authoritative unless this amendment states a narrower rule. In
particular, v1 Section 8 remains the sole canonical statement of the recall-floor asymmetry. The
public release projects this execution amendment without copying historical campaign outputs from
v1 or v2.

This amendment authorizes future local preparation and execution after the immutable public
release exists. It does not authorize an experiment during release construction, a provider call,
access to protected inputs by the release worker, publication, or admission of a result bundle.

## 1. Decision and rationale

The paper cannot afford full-corpus dense retrieval and reranking across every Track. Its primary
forward experiments therefore use a separately named **judgment-union** Profile for each Track and
SIGIR query representation. The effective trial corpus is exactly the set union of trial IDs that
occur in that Track's frozen Evaluation Package. Selection happens before any System is run and
cannot depend on scores.

The source preparation remains the complete, checksum-locked Track preparation. The smaller
effective corpus is a derived Task Input, not a replacement source collection and not a claim of
full-corpus retrieval. Benchmark Release 0.1 continues to expose the complete-corpus Profiles for
independent use and conformance.

## 2. Primary paper matrix

Each row below is a separate result stratum. Raw metrics are not pooled across Tracks or across the
two SIGIR query representations.

| Track | Paper Profile | Source Profile | Primary Systems |
| --- | --- | --- | --- |
| TREC Clinical Trials 2021 | `trec-ct-2021-judgment-union` | `official-full` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| TREC Clinical Trials 2022 | `trec-ct-2022-judgment-union` | `confirmation-full` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| TREC Clinical Trials 2023 | `trec-ct-2023-judgment-union` | `confirmation-full` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| SIGIR Clinical Trials 2016 description | `sigir-ct-2016-description-judgment-union` | `description` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |
| SIGIR Clinical Trials 2016 summary | `sigir-ct-2016-summary-judgment-union` | `summary` | `bm25`, `dense-bge-m3`, `dense-qwen3-embedding-0.6b`, `rrf` |

The staged `staged-bm25-qwen3-rrf-rerank` System is an additional TREC 2021 experiment under
`trec-ct-2021-judgment-union`. Its folded-BM25 and no-template-Qwen component Systems are supporting
components, not extra primary matrix columns. The staged sequence, depths, fusion constant,
reranker revision, and Primary Ranking are fixed by the public method contract.

No real `trial_to_patient` effectiveness result is part of this paper matrix. Release 0.1's
direction-specific synthetic reverse fixture and named TREC 2021 reverse viability Profile remain
separate contract and viability evidence.

## 3. Pool construction and identity gate

For each paper Profile:

1. prepare the complete Track from the exact packaged source lock;
2. select every and only trial ID present in the resulting Evaluation Package;
3. preserve the prepared-corpus order of those selected trials;
4. emit a pre-effectiveness receipt containing the Profile definition hash, Prepared Snapshot ID,
   Evaluation Package ID, selected-trial count, and SHA-256 of the ordered trial-ID list; and
5. freeze that receipt before the first System score is produced.

The benchmark CLI must reproduce the frozen count and ordered-ID digest when it constructs every
Task Input. A mismatch stops execution. The Evaluation Package is not reduced: every supplied
judgment retains its original label and topic association. A trial judged for one topic is not
thereby judged for another topic. Such cross-topic pairs remain unjudged and receive zero gain only
inside the declared pooled metric; they do not become negative judgments or clinical truth.

For TREC 2021, the expected union contains **26,162 unique trial IDs**. This number is a structural
gate, not a complete identity. The ordered-ID digest and the recipe-v3 Prepared Snapshot and
Evaluation Package identities must still be emitted and frozen from the immutable release before
effectiveness execution.

The historical 26,149-trial TrialGPT common pool is an external-system fidelity profile. It is not
the TREC 2021 paper Profile defined here, and results from the two memberships must not share a
comparative table.

For TREC 2022, TREC 2023, and both SIGIR views, no count or digest is preclaimed in this amendment.
An authorized operator derives each receipt locally from the checksum-verified source files after
the public release is frozen. TREC 2022 and 2023 topics and qrels remain protected until that
operator supplies the SHA-256 identity of this approved protocol to preparation and pool
inspection. If any derived pool exceeds available compute, execution stops and a new dated,
score-blind amendment must narrow or remove the Track before any System output is inspected.

## 4. Execution and comparisons

All final rows are fresh runs from the same immutable Benchmark Release 0.1 manifest, public tree,
package artifact, and dependency lock. Each run binds its exact Track, Task, Profile, Prepared
Snapshot, Task Input, System Input, Evaluation Package, pool receipt, and Primary Ranking.

The four primary Systems use the same effective Task Input within a Track/Profile stratum. RRF uses
the declared BM25 and BGE-M3 runs from that same stratum. The staged System uses only its declared
TREC 2021 components. Missing, failed, or unauthorized rows remain missing; they are not imputed.

Historical recipe-v1, recipe-v3 full-corpus, 26,149-common-pool, campaign, development, or
provider-backed results are not paper rows under this amendment. They may be described only as
clearly separated provenance or external-fidelity evidence where the base protocol permits it.

## 5. Reporting and claim limits

Every main result table and caption must describe these experiments as **within-pool ranking over
the frozen judgment-union corpus**. It must not call them full-corpus retrieval, official TREC run
reproduction, exhaustive retrieval, or a new benchmark dataset. Full-corpus and judgment-union
results never share a comparative table.

TREC 2021 reports judged relevant-or-eligible recall and both relevant-or-eligible and eligible
precision according to its frozen Profile. TREC 2022 and 2023 report judged eligible recall and
eligible precision. SIGIR reports referral-oriented metrics and must not rename its labels as
eligibility. Unjudged pairs remain unjudged for every Track.

The paper may compare System behavior only within the exact frozen strata above and may describe
cross-Track consistency without pooling raw scores. It may not claim clinical validity, patient
eligibility, deployment safety, enrollment benefit, full-corpus effectiveness, or equivalence to
an upstream implementation.

## 6. Stop and admission rules

Before any effectiveness command, stop if the public release is not immutable, a pool receipt is
missing or mismatched, a source lock fails, a selected System is absent from the release support
matrix, a protected Track lacks this protocol's digest, or the run cannot bind all required
producer identities. Do not repair these failures after inspecting scores.

Result bundles remain outside the initial release definition. Later admission requires a separate
review of exact current-schema bundles produced under this amendment. Provider-backed TrialGPT
work in Issue #93 remains post-release and must bind the immutable Release 0.1 identity; it is not
silently substituted for any primary matrix row.

# TAIM paper analysis protocol

**Protocol ID:** `taim-paper-analysis-protocol-2026-08-29-v1`

**Status:** Frozen. This document is the E0 registration the paper plan calls gate B
("Freeze paper protocol"). Once merged it is never edited; an amendment is a new dated
version in this directory that names what it supersedes, plus an additive supersede note
beside this file, following the campaign's correction convention (additive notes beside
unchanged content, `docs/evidence/remediation-2026-08-28.md`).

**Placement.** This file lives in `docs/protocols/` rather than beside the paper drafts in
`docs/paper/`, because the repository already has a convention for exactly this artifact:
`docs/protocols/` holds frozen, dated, immutable protocols with protocol IDs
(`trec-2022-2023-ingestion-2026-08-26.md` and siblings), and the paper drafts' own
placement note records that dated filenames mark immutable snapshots while `docs/paper/`
holds growing, undated deliverables (`docs/paper/stable-sections.md`, placement note). A
pointer to this freeze is recorded in `docs/paper/stable-sections.md`, Section 4, whose
draft explicitly awaits it.

**Provenance.** Registered from the four Notion paper-plan pages, fetched read-only
2026-08-29: root `3cad2bcb09d380cc9f7acead638ab4d8` ("TAIM benchmark paper plan",
carrying the 2026-08-28 gate-3 System decision), "Paper framing and outline"
`3cad2bcb09d381f6959be4c3dcd54d1b`, "Experiment plan and evidence ledger"
`3cad2bcb09d38106a4b9e5eac5ae82a0`, "Readiness and repository audit"
`3cad2bcb09d381cda619d04080a68e3d`; and from the committed campaign record cited inline
throughout. Every figure in this protocol traces to a committed machine-readable source or
to the plan pages; none is invented, estimated, or rounded. Where the plan asks this
protocol to state something the record cannot support, Section 14 says so instead of
smoothing it.

The plan's E0 instruction, quoted (ledger page, "E0. Freeze the protocol"):

> Commit research questions, claims, Systems, fidelity labels, profiles, Pipeline Depths,
> endpoints, primary contrasts, multiplicity, statistical rules, failures, stop rules,
> planned tables, and artifact admission before execution.

Each of those items has a numbered section below. "Endpoints" is not defined anywhere in
the plan; this protocol reads it as measurement endpoints, i.e. the metric family of
Section 7 with the primary endpoint of Section 7.1, and declares that reading here rather
than resolving it silently.

---

## 1. Scope and authority

This protocol governs the analysis and reporting of the paper's registered experiments
(the ledger's E1-E7) and the admission of results into the paper. It binds every
main-paper number, table, comparison, and significance verdict.

It does **not** govern, and deliberately does not touch:

- **Public-release scope selection.** Which Systems, methods, and Published Result
  Bundles enter Benchmark Release 0.1 - the five scope decisions and every approval in
  Issue #99's owner half (`docs/plans/issue-99-hannes-half-2026-08-28.md`) - is decided
  there, not here. This protocol registers what will be run and how it will be analyzed;
  it confers no release admission on anything.
- **The abstract and the results narrative.** Both are gated on experiments that have not
  run (plan sequence: not before gate F). This protocol contains neither.
- **TREC 2022/2023.** Protected and out of scope
  (`docs/adr/0002-evaluation-and-comparison-contract.md`; readiness page "Scope and
  no-gos"; the deliberate deferral decision recorded 2026-08-28 on the readiness page).

## 2. Track, task, profile, and frozen identities

- Track and task: TREC Clinical Trials 2021, patient-to-trial. The only authorized paper
  track (readiness page "Scope and no-gos"; `docs/adr/0002`).
- Benchmark Profile: `official-full` v1.0 - 75 topics, 375,580 trials, budget K=1,000
  (`results/shipping-configuration-trec-ct-2021/metrics.json`, `.contract`).
- Prepared Snapshot ID:
  `sha256:65a3ecb67c65730f77dcdcb14f70db64e0baed85e0a3df4ced407f464dca0ad3`; Evaluation
  Package ID:
  `sha256:bc7d90d250b464c17921c545c11b0159334ad343d0b5036d55468c6f02d36c0c`; snapshot
  contract version 2.0; 35,832 pooled judgments
  (`results/publishable-fusion-trec-ct-2021/manifest.json`), of which 5,570 are label 2
  (`results/shipping-configuration-trec-ct-2021/metrics.json`,
  `.headline.pooled_eligible_judgments`).
- Judgment semantics: NIST labels keep their original meanings; an absent judgment is
  unjudged, never an implicit 0; the evaluator's unjudged policy is `retain_as_zero_gain`;
  recall denominators are qrels denominators and are always described as judged recall
  (`docs/adr/0002`; the contract block above).
- Admission to comparison: two Primary Rankings enter Default-Track Compare only when
  their Prepared Snapshot ID, Task Input ID, Evaluation Package identity, Benchmark
  Profile, and budget K match; Pipeline Depth may differ but must be disclosed
  (`docs/adr/0002`).

## 3. Research questions and claim boundaries

Research questions, registered unchanged from the plan (framing page):

1. Which runs are comparable when Snapshot, task, System input, Evaluation Package,
   Benchmark Profile, budget, and Pipeline Depth must match?
2. How do judged eligible recall, top-rank nDCG, precision, reliability, and compute
   change across retrieval, fusion, and reranking?
3. Can a clean public release reproduce prepared-data identities, rankings, metrics, and
   Published Result Bundles?
4. Which upstream systems can be reproduced, reconstructed with declared changes, or only
   audited because required artifacts are missing?

The paper's claim is the comparison contract and the audit trail, not a leaderboard
(gate-3 decision, plan root page). Claim boundaries, frozen from the framing and
readiness pages:

- TREC 2021 is development evidence; a public rerun is computational reproduction, not
  independent confirmation.
- Patient-to-trial only. Pooled judgments establish neither patient eligibility, clinical
  recall, calibration, deployment safety, nor enrollment impact.
- `official-full` and restricted common-pool results never share a table.
- No current TrialMatchAI score is ever placed beside the published TrialMatchAI result,
  in any table, chart, or sentence of comparison (`results/README.md`, standing caveat).
- Retrieval-only TrialGPT evidence is never relabelled end-to-end effectiveness.
- Missing evidence, timeouts, or compute limits never become clinical rejection.
- Single benchmark: every empirical statement is about TREC CT 2021 under this contract.
  Generalisation is future work by explicit scoping decision (readiness page, decision of
  2026-08-28: single-benchmark is the largest exposure, and deferral is a scoping
  decision, not a claim that the exposure does not exist).

## 4. Systems

### 4.1 Headline System: the two-arm shipping configuration

Decided at gate 3 and recorded on the plan root page, 2026-08-28. Identity, from the
committed bundle (`results/shipping-configuration-trec-ct-2021/metrics.json`):

| | |
|---|---|
| arms | `bm25-folded` + `qwen3-raw-no-template` |
| fusion | `taim.baselines.rrf.fuse_rrf_n`, k=60, ties RRF score desc then trial_id asc |
| component depth / panel cap | 5,000 / 1,000 |
| reranker | `Qwen/Qwen3-Reranker-4B` @ `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, F-summary folded query, float16, max_length 2,048, batch 16 |
| scoring depth / output cut | 2,000 / 1,000 |
| reranked ordering | cross-encoder score descending, ties fused rank ascending |
| eligibility model | none |

Its committed development row (macro judged eligible recall@1000 `0.682157141352177`,
micro `0.6220825852782765`, eligible retrieved 3,465 of 5,570, macro graded nDCG@10
`0.4100272993566693`, macro eligible P@10 `0.304`, net +197, 150,000 pairs reranked;
same bundle) is registration data, not a paper result: the ledger marks it
"Reconstructed from retained private scores. Needs one fresh run." Paper numbers for
this System come only from the registered fresh run (Sections 5 and 6), through the
bundle gate of Section 10.

The three-arm configuration remains what the gate-3 decision says it is: a Publishable
row typed as a candidate, off `main`, at its own branch tip - honest evidence that the
corrected arm works, and not the headline. It is never pooled with, or silently
substituted for, the shipping configuration.

### 4.2 Reference Systems

From the ledger (E2): BM25, BGE-M3, instructed Qwen3-Embedding-0.6B, and BM25 plus BGE
RRF - all 75 topics, all 375,580 trials, `official-full`, K=1,000, metrics recomputed
from rankings.

### 4.3 External systems

Case-study admission follows the ledger's E7 gates exactly: Kusa only as a bounded
common-pool case (or after a preregistered new official-full comparison); TrialMatchAI
only as a sealed reproducibility case study and never beside its published number;
TrialGPT excluded from the v1 headline unless Issues #93 and #22 produce one homogeneous
all-75 run and immutable bundle; SatIR excluded from v1; the MedCPT drift-only control
only if causal attribution matters. Fidelity-matrix rows for excluded systems still
appear (an "audit-only, artifacts missing" row is itself an RQ4 finding).

## 5. Registered contrasts and Pipeline Depths

Pipeline Depth identities for the headline System are those of Section 4.1. The
registered contrasts, from the ledger:

1. **Every reranked configuration against its own unreranked panel baseline** (E4:
   "Compare the reranked output with its own unreranked panel"). This contrast is
   mandatory wherever a reranked number appears: the committed record shows the same
   rankings scoring -0.001 nDCG@10 against another reranked panel and -0.0631 against
   their own pre-rerank baseline
   (`results/shipping-configuration-trec-ct-2021/significance.md`), so a stage's cost is
   only visible against a baseline that lacks the stage.
2. **Depth conditions derived from one scoring pass** (E4): score the deepest frozen list
   to D4000 once; derive D1000, D2000, D4000 without rescoring; the paper configuration
   stays at scoring depth 2,000 and cut 1,000 unless the registered rerun contradicts it.
3. **Method contrasts, conditional on the E3 release decision**: raw vs folded BM25;
   instructed vs no-template Qwen; the panel at depths 1,000, 2,000, 5,000; leave-one-out
   at depth 5,000.
4. **Reproduction agreement analyses** (E5): two same-runtime repeats and one
   second-GPU-architecture run per main dense and reranking path, reporting candidate-set
   overlap, Kendall agreement on the final tie-broken order, top-10 and top-50 churn,
   metric deltas, runtime, and memory. Bitwise cross-hardware equality is never promised,
   and cross-hardware claims are made at the candidate-set and metric level only
   (`docs/agents/benchmarking.md`, "The GPU dispatch is part of the recipe").

Reference-suite rows (Section 4.2) are reported side by side in the main table; any pair
of them the paper narrates comparatively falls under Section 6.4.

Every new comparison is pre-registered before scoring under the discipline of
`docs/agents/benchmarking.md`: predict churn and enrichment rather than boundary effects
("Ordering and membership are not separable"), measure the population a rule sees before
predicting its catch ("The exceptional path is the default path"), and check every arm's
contracted depth ("Every arm in a panel owes the panel its contracted depth", published
via `audit_component_depth`).

## 6. Statistics: the significance rule, frozen as actually applied

### 6.1 Machinery

Paired over the 75 topics: same topics, same judgments, only the compared conditions
differ. Two tests per comparison, computed by
`scripts/analyze_fusion_significance.py` (unchanged machinery across the campaign):

- an **exact two-sided sign test** over the topics that move;
- a **topic-level bootstrap of the mean paired difference**: **10,000 resamples**, seed
  **20260819**, 95% percentile confidence interval (`BOOTSTRAP_RESAMPLES = 10_000`,
  `BOOTSTRAP_SEED = 20260819`, frozen constants of that script).

The seed and resample count are frozen with the rule. Reported per comparison: macro A,
macro B, difference, win/lose/tie topic counts, sign-test p, bootstrap 95% CI, verdict.

### 6.2 The decision rule: a conjunction

**A difference is distinguishable from noise only if the bootstrap 95% CI excludes zero
AND the exact sign test reaches p < 0.05.** Both, always; never either alone. This is the
rule as encoded (`distinguishable_from_noise = (not spans_zero) and probability < 0.05`
in the script) and as applied in every committed significance verdict of the campaign.

The conjunction is load-bearing, not ornamental. Three committed comparisons had a
bootstrap CI excluding zero while the sign test failed, and in all three the rule was
applied as written rather than relaxed:

| comparison | metric | win/lose/tie | sign-test p | bootstrap 95% CI | verdict |
|---|---|---|---|---|---|
| P2 vs P3 after reranking | nDCG@10 | 10 / 4 / 61 | 0.1796 | [+0.000980, +0.008558] | not distinguishable |
| leave-one-out d5000, B vs A (drop `bge-m3-raw`) | nDCG@10 | 34 / 41 / 0 | 0.489 | [-0.0519, -0.0070] | not distinguishable |
| leave-one-out d5000, B vs A (drop `bge-m3-raw`) | eligible P@10 | 17 / 25 / 33 | 0.280 | [-0.0573, -0.0067] | not distinguishable |

(Sources: `results/panel-after-rerank-p2-vs-p3-trec-ct-2021/REPORT.md` and
`significance.md`; `results/leave-one-out-d5000-trec-ct-2021/REPORT.md` and
`significance.md`.)

The P2-vs-P3 report states why the bar is a conjunction, and that statement is frozen
with the rule: "Had the bar been the CI alone, this run would have reported P2
*distinguishably better* on nDCG@10 - which is exactly why the bar was fixed in advance."
The mechanism is on record there too: with 61 of 75 topics exactly tied, sixty-one exact
zeros pin the bootstrap variance down while the sign test sees only the fourteen topics
that move - the two tests disagree precisely when an effect is tiny and confined, which
is when a fixed-in-advance bar matters most.

### 6.3 Interpretation rule

"Not distinguishable" is never reported as "no effect". Failing to reject is not evidence
of absence; the committed formulation is frozen: what such a run establishes is that at
the operating configuration, with 75 topics and this machinery, no effect survives at a
detectable size (`results/panel-after-rerank-p2-vs-p3-trec-ct-2021/REPORT.md`).

### 6.4 No untested asymmetry may carry an argument

Every comparative sentence in the paper either cites its paired verdict under Section 6.2
or names itself untested. The precedent is committed: the depth-1,000 ablation's
strongest argument was an asymmetry it never tested, and when the pre-registered
depth-5,000 re-run finally tested it, the test did not license it
(`results/leave-one-out-d5000-trec-ct-2021/PRE-REGISTRATION.md` and `REPORT.md`).

### 6.5 Multiplicity

Frozen as applied throughout the campaign: **no multiplicity correction**. Every
registered test is reported individually - none is dropped or selected post hoc - and
each comparison family names its full test count where its verdicts are reported, as the
depth-5,000 leave-one-out does ("Nine tests, no multiplicity correction, as
pre-registered and as at depth 1,000", `results/leave-one-out-d5000-trec-ct-2021/`).
This protocol registers the contrast families of Section 5 in advance, which is what
makes the no-correction convention honest: the multiplicity is visible, disclosed, and
fixed before the data.

### 6.6 Fixed sample, no interim analyses

The topic set is fixed at 75 by the track. No registered comparison is stopped, extended,
or re-run based on its interim result; every comparison reports all 75 topics.

## 7. Measurement endpoints and the reporting contract

### 7.1 Primary endpoint

Macro-averaged graded nDCG@10 with linear gains 0, 1, 2 ("The primary reporting metric",
`docs/adr/0002`).

### 7.2 Per-condition reporting contract

Every condition reports, verbatim from the plan's reporting contract (ledger page):

- macro graded nDCG@10;
- eligible P@10;
- macro and micro judged eligible recall at K=100, 500, and 1,000, with retained
  numerator and denominator, and misses;
- all per-topic values;
- the paired exact sign test and the 10,000-resample, seed-20260819 topic bootstrap for
  its registered contrasts (Section 6);
- multiplicity handling (Section 6.5);
- runtime, accelerator-hours, peak memory, index size, and stage throughput;
- failures, retries, and abstentions.

Effectiveness, operational reliability, and artifact completeness remain separate report
sections, so coverage, failures, cost, or abstention cannot be mistaken for ranking
quality (`docs/adr/0004-stage-aware-evaluation-scorecards.md`).

Every judged-recall figure reported under this contract is read under Section 8, by
reference; no table, caption, or limits paragraph restates the asymmetry in its own
words.

## 8. The floor asymmetry - stated once, here; everywhere else points here

This section is the paper's single canonical statement of the recall-floor asymmetry.
Every other surface - metric definitions, table captions, per-study reports, the limits
section, and any later draft - references this section (or, inside a bundle, the
committed sources named below) and does not restate it. Restating is how such caveats
drift: the correctness audit's largest wrongness class was documents that still read as
current after the finding they carry was overtaken
(`docs/evidence/correctness-audit-2026-08-28.md`, Part 6), and the single-statement rule
here is a standing instruction from the campaign captain for exactly that reason.

**The statement.** TREC 2021 judged only what its own participants pooled, and an absent
judgment is unjudged and scores as a miss. Therefore:

1. **Every judged recall figure is a floor, not an estimate.**
2. **The floor is asymmetric against any stage that promotes new candidates** - fusion
   arms and rerankers alike. In any such comparison, **displacement is exact**, because a
   displaced pair was judged eligible by construction; **recovery is a lower bound**,
   because a promotion from outside the 2021 pool scores as a miss however well it
   matches. Recovery is under-counted; displacement is not. Any positive net is
   therefore a floor.
3. **Top-rank reordering of judged pairs is not a floor.** An nDCG@10 change from
   reordering judged pairs inside the top 10 is fully observed and exact. The asymmetry
   runs against the promoting stage on the gain side and not on the cost side.

Committed sources, frozen with the statement:
`results/cross-encoder-full-run-trec-ct-2021/PRE-REGISTRATION.md` ("Both floors, stated
in advance") and `REPORT.md` ("Both floors, and they point the same way") for points 2
and 3; the standing preamble every significance artifact carries ("Every recall figure
here is a floor, not an estimate", `scripts/analyze_fusion_significance.py`) and the
shipping bundle's own floor line
(`results/shipping-configuration-trec-ct-2021/README.md`) for point 1.

## 9. Cost, reliability, and failure policy

- Cost and reliability are reported per condition as in Section 7.2, in a report section
  separate from effectiveness (`docs/adr/0004`).
- Failures, retries, and abstentions are reported as operational outcomes. Missing
  evidence, timeouts, or compute limits never become clinical rejection, and no
  operational failure is ever converted into a relevance label (readiness page "Scope
  and no-gos"; `docs/adr/0004`).
- Execution identity is part of the recipe: every scored condition records the exact
  model, tokenizer, runtime, driver, CUDA, and attention-backend identity that actually
  ran - the backend as reported at runtime, not as requested - and raw candidate scores
  are persisted for every configuration a probe compares
  (`docs/agents/benchmarking.md`, "The GPU dispatch is part of the recipe").

## 10. Artifact admission: the required paper bundle

A result enters the paper only from a frozen, validated bundle containing, verbatim from
the plan's "Required paper bundle" (ledger page):

1. Frozen protocol and release tag.
2. Current run manifest with Snapshot, task, System, Evaluation Package, profile,
   budget, Primary Ranking, and Pipeline Depth identities.
3. Complete component and stage rankings, or stable archive identifiers plus hashes.
4. Metrics, scorecards, significance output, per-topic table, and resource use.
5. Exact model, tokenizer, runtime, driver, CUDA, and attention-backend identity.
6. Raw reranker scores and batch-composition record.
7. Redistribution declaration and complete checksums.
8. Published Result Bundle selected independently of measured quality.

Supporting rules, all standing contract or committed lesson:

- Bundle selection criteria are independent of measured quality (`docs/adr/0002`).
- A digest that points to missing bytes fails the paper gate; every cited artifact must
  be retrievable by a second operator from stable storage (ledger E6;
  `docs/campaigns/cluster-dependency-manifest.json` and
  `scripts/check_cluster_dependencies.py` make not-looking a visible failing state).
- Gates assert effect, not presence: for any setting that is supposed to bound, filter,
  or enrich, the bundle asserts on an artifact the setting changed
  (`docs/agents/benchmarking.md`, "Assert effect, not presence").
- Component-depth shortfalls are either declared in advance with a reason or they stop
  admission (`src/taim/baselines/panel_depth.py`; `docs/agents/benchmarking.md`).
- Result-bearing evidence branches are frozen provenance: never rebased, renamed, or
  deleted while anything cites them (`docs/agents/benchmarking.md`, "Result-bearing
  branches").

## 11. Planned tables and figures (shells only)

Row-admission rules, frozen:

- **One main table, containing only Default-Track-comparable rows** (Section 2's
  admission rule). Restricted-profile studies get separate tables; `official-full` and
  restricted common-pool rows never share a table (framing page; readiness page).
- Main-table columns are the Section 7.2 contract: condition identity (System, stage,
  Pipeline Depth), the effectiveness metrics with numerators and denominators, and the
  Section 6 verdicts for registered contrasts; cost and reliability columns live in the
  separate operational table (`docs/adr/0004`).

Planned figures and tables, registered from the framing page without content:

1. TAIM boundaries from Source Bundle to Benchmark Snapshot, System execution,
   evaluator-owned Evaluation Package, and Published Result Bundle.
2. Fidelity and artifact-completeness matrix for every adapted System.
3. Pareto plot of judged eligible recall, graded nDCG@10, and compute, with Pipeline
   Depth visible.
4. Candidate funnel from component reach to fusion cap, reranker recovery and
   displacement, and final ranking (read under Section 8).
5. Cross-hardware panel separating candidate-set agreement from order-sensitive
   agreement.
6. The main table and its operational companion.

No figure, table cell, or caption may carry a number that does not trace to a committed
bundle admitted under Section 10.

## 12. Fidelity labels and audit vocabulary

Frozen from the committed audit records, and applied to our own campaign first and to
adapted external systems second:

- Reproduction labels: **FULL** / **PARTIAL** (naming which artifacts and why) /
  **NOT DEMONSTRATED** (`docs/evidence/reproducibility-audit-2026-08-28.md`).
- Figure findings: **ERROR** / **STALE** / **UNLABELLED** / **CLEAN** /
  **UNTRACEABLE** (`docs/evidence/correctness-audit-2026-08-28.md`).

## 13. Stop rules

1. Fixed sample, no interim analyses, no data-dependent stopping (Section 6.6).
2. The paper configuration stays at scoring depth 2,000 and cut 1,000 unless the
   registered E4 rerun contradicts it (ledger E4).
3. A bundle whose reproduction gate fails does not enter the paper; the gates' failing
   branches are demonstrated, not assumed (`tests/test_script_reproduction_gates.py`).
4. An undeclared component-depth shortfall stops admission (Section 10).
5. A digest pointing to missing bytes fails the paper gate (Section 10).
6. Any encounter with protected TREC 2022/2023 files: report only their paths and stop
   (`docs/adr/0002`; repository standing rule).
7. Pre-registration precedes scoring for every new comparison; predictions are never
   edited after data; a pre-registered rule that fails is preserved verbatim, as the
   depth-5,000 droppability rule was (`results/leave-one-out-d5000-trec-ct-2021/`).

## 14. What this protocol does not state, because the record does not support it

Reported as gaps rather than smoothed over:

1. **The release tag** (required-bundle item 1). No public release tag exists: `taim
   release freeze` exits pending Issue #99's human approvals
   (`docs/evidence/release-readiness-audit-2026-08-28.md`, Part 0). The tag is recorded
   by amendment (a new dated protocol version) when gate D produces it; nothing else in
   this protocol depends on its value.
2. **Release scope.** Whether folded BM25, the fused panel, the no-template Qwen arm,
   the reranked configurations, or dense-medcpt enter the public release is Issue #99's
   owner half and is not decided here (Section 1). The registered contrasts of Section 5
   item 3 are conditional on that decision, as the ledger's E3 states.
3. **The tasking's "18-to-7 record".** The direction to freeze this protocol described
   the P2-vs-P3 conjunction case as "applied against an 18-to-7 record". No committed
   significance output contains an 18-to-7 record; the committed record for that
   comparison is 10 wins / 4 losses / 61 ties at p = 0.1796
   (`results/panel-after-rerank-p2-vs-p3-trec-ct-2021/significance.json`). This protocol
   cites the committed record. The observation (the conjunction case is real and the
   rule held) and the interpretation (its win/lose split) are separated here so the
   second could be wrong without corrupting the first.
4. **Statistical stop rules beyond Section 13.** The record contains no data-dependent
   stopping machinery, so none is registered; Section 13's stop rules are gates and
   fixed-design commitments, which is all the record supports.
5. **A frozen main-table row list.** The final row list depends on the E3 release
   decision and the E7 case-study gates; freezing rows now would either invent a release
   decision or invite silent substitution later. The admission rules of Sections 10 and
   11 are frozen instead, and they determine the rows mechanically once the gates close.

## 15. Sources

- Notion paper plan, fetched read-only 2026-08-29: root
  `3cad2bcb09d380cc9f7acead638ab4d8` (with the gate-3 decision of 2026-08-28); framing
  `3cad2bcb09d381f6959be4c3dcd54d1b`; ledger `3cad2bcb09d38106a4b9e5eac5ae82a0`;
  readiness `3cad2bcb09d381cda619d04080a68e3d`. No page was edited.
- `docs/adr/0002-evaluation-and-comparison-contract.md`,
  `docs/adr/0004-stage-aware-evaluation-scorecards.md`,
  `docs/adr/0005-separate-public-benchmark-release.md`, `docs/agents/benchmarking.md`.
- `results/shipping-configuration-trec-ct-2021/` (headline System identity and
  development row), `results/publishable-fusion-trec-ct-2021/manifest.json` (frozen
  identities), `results/panel-after-rerank-p2-vs-p3-trec-ct-2021/` and
  `results/leave-one-out-d5000-trec-ct-2021/` (the conjunction rule's record),
  `results/cross-encoder-full-run-trec-ct-2021/` (the floor asymmetry's committed
  statement), `scripts/analyze_fusion_significance.py` (seed, resamples, verdict
  encoding).
- `docs/evidence/correctness-audit-2026-08-28.md`,
  `docs/evidence/reproducibility-audit-2026-08-28.md`,
  `docs/evidence/release-readiness-audit-2026-08-28.md`,
  `docs/evidence/remediation-2026-08-28.md`.
- `docs/paper/stable-sections.md` (Section 4 draft, which this freeze completes) and
  `docs/plans/issue-99-hannes-half-2026-08-28.md` (the selection question this protocol
  does not touch).

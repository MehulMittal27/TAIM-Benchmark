# TAIM paper analysis protocol: one TrialGPT retrieval for two Systems

**Protocol ID:** `taim-paper-analysis-protocol-2026-10-07-v14`

**Status:** Frozen on 2026-10-07, the date in its protocol ID, as the repository owner ruled.
Written on 2026-10-06 for the owner's review, and frozen with the owner's answers to that draft
applied. These bytes are immutable; a later change requires a new dated amendment. It is frozen
after v13, and before any query generation, retrieval freeze or System call is made for any row it
governs (Section 3). It ships with the release that first ships `TrialGPT2-TAIM-Luna-v1`, as v10,
v12 and v13 ship with theirs, because a reader of that release's TrialGPT rows needs the rule. It
authorizes no run, no release, no provider call and no result admission. Its one authorization is
the extension of protected TREC 2022 access in Section 8, as the owner ruled.

No effectiveness value prompted this amendment. It was prompted by a seam between two frozen
amendments. Protocol v11 gives each of `TrialGPT-TAIM-Luna-v1`'s four targets exactly one
query-plan generation and binds those rows to Release 0.1.1. Protocol v12 Section 3 condition 4
binds a row's frozen retrieval input to the row's own release. TrialGPT 2.0, registered here as
`TrialGPT2-TAIM-Luna-v1`, re-ranks that same frozen retrieval and can first ship only in a release
later than 0.1.1. As written, a target's one generation could feed one of the two Systems and never
both. This amendment moves `TrialGPT-TAIM-Luna-v1`'s four rows to the release that first ships
`TrialGPT2-TAIM-Luna-v1`, admits them under v12, and lets one generation, one receipt and one
retrieval freeze per target serve both Systems on identical candidates.

## 1. What this amendment changes

In this amendment, **the TrialGPT 2.0 release** is the immutable public release of this repository
that first ships `TrialGPT2-TAIM-Luna-v1`. It also carries the receipt route that protocol v11
relies on (commits `962b0130` and `881ea83f` of the private source repository from which releases
are generated) and this amendment.

For `TrialGPT-TAIM-Luna-v1`'s four targets, and for `TrialGPT2-TAIM-Luna-v1` on the same four
targets, it changes these clauses and nothing else:

- **Protocol v11 Section 4, adaptation 1.** "The four rows execute Release 0.1.1 as protocol v9
  Section 5 requires" is replaced: the four rows execute the TrialGPT 2.0 release and are admitted
  under protocol v12 Section 3 (Section 4 below).
- **Protocol v11 Section 4, "The query plans of every target selected for execution are finished
  before any retrieval or System scoring begins".** For these rows a target is selected for
  execution one at a time, in the order of Section 9, so each target's query plans are finished
  before its own retrieval, and a later target's generation may follow an earlier target's freeze,
  gates and rows.
- **Protocol v11 Section 6, the `release` field.** It holds the TrialGPT 2.0 release's release
  manifest ID, public tree ID and package artifact ID, not Release 0.1.1's. Every other field of the
  receipt is unchanged, including `protocol_id`, `protocol_sha256` and `system_id` (Section 5).
- **Protocol v11 Sections 6, 7, 7.1 and 8, wherever they name Release 0.1.1 as the code that
  generates, classifies, records or recomputes.** They read the TrialGPT 2.0 release. That release
  checks the receipt itself, so v11 Section 6's sentence "Release 0.1.1 carries no receipt
  machinery, so the receipt is procedural" no longer describes the release; the receipt is still
  written and committed before the freeze, as v11 Section 6 requires. Before this freeze it was
  confirmed, in the source from which that release is to be generated, that `taim.model_codex` and
  `taim.generation` classify a failed provider call as `authentication` or `transport` and store a
  sign-in failure as a final unresolved result that a resume returns unchanged, exactly as v11
  Section 7.1 describes Release 0.1.1 doing. Section 7.1 therefore applies as written.
- **Protocol v11 Section 12, items 5 and 6.** Item 5 reads: the `protocol_approval_id` recorded in
  each retrieval lock of the TrialGPT 2.0 release is protocol v11's digest, and this amendment
  governs the row together with v11. Item 6 reads: the TrialGPT 2.0 release checks the receipt
  against the retrieval lock and refuses a mismatch, and does not recompute generation traces, so
  each row's plan provenance still rests on its retained cache and traces and its committed
  recomputation.
- **Protocol v11 Section 1,** the item "v9: Release 0.1.1 as the executable release" in its list of
  rules that stand, for these four rows. Its statement that in a later release that records the
  receipt in its retrieval lock, the lock cites v11's digest, describes the TrialGPT 2.0 release and
  stands.
- **Protocol v5 Section 2,** "Every row must execute the public TAIM Benchmark 0.1.1 wheel", and
  **protocol v9 Section 5,** which binds every final paper row to Release 0.1.1: for these four
  rows, both are replaced by Section 4 of this amendment.
- **Protocol v12 Section 1,** the sentence "A row governed by v11 still executes Release 0.1.1 as
  v11 requires", is replaced for these four rows by Section 4 of this amendment.
- **Protocol v12 Section 6,** for `TrialGPT2-TAIM-Luna-v1` only: it enters no paired comparison
  (Section 7). v12 Section 6 permits a comparison only when it is registered; this amendment forbids
  registering one for that System.
- **Protocol v12 Section 7,** for the rows of both Systems on these targets: each declaration
  carries one field beside those v12 lists (Section 5).
- **Protocol v5 Section 5,** the owner's authorization of protected TREC 2022 access: extended to
  `TrialGPT2-TAIM-Luna-v1` (Section 8).

Every other rule stands unchanged:
- v11, including the load-bearing rule of its Section 5 (one generation per target), Sections 7
  and 7.1 (interruption, resume and clearing), Section 8 (retention and recomputation), Sections 9
  to 11, and every disclosure of its Section 12 not adapted above;
- v12, all of it, including Section 3 conditions 1 to 8 and the environment record of Section 5;
- v13;
- v9 Section 4: K = 20, retrieval depth 2,000, 500 candidates into criterion matching and
  aggregation, up to 1,500 logical provider calls per topic, exactly 48 generation workers;
- v5: the four targets of Section 1 (TREC 2023 has no TrialGPT row, for either System), the query
  method of Section 3, retrieval and consensus of Section 4, the provider budget of Section 5,
  evidence and admission of Section 6;
- v4, v8 as v10 restates it, and v1, with Section 8 of v1 remaining the sole canonical statement of
  the recall-floor asymmetry.

The System id of the retrieval's producer stays `TrialGPT-TAIM-Luna-v1`. Nothing in either System
id changes.

## 2. Why

- **Two Systems, one first stage.** `TrialGPT2-TAIM-Luna-v1` runs TrialGPT 2.0's own scorer,
  unchanged, from `github.com/TrialGPT2/TrialGPT2.0` commit
  `1c5f9ebc3b98d1f780ceea68043efdaa2daba880`, over the top 500 per topic of
  `TrialGPT-TAIM-Luna-v1`'s frozen three-query consensus retrieval (v5 Sections 3 and 4). It has no
  first stage of its own.
- **The two frozen rules cannot both be met for one generation.** v11 Section 5 permits one
  generation per target and v11 Section 4 binds its rows to Release 0.1.1. v12 Section 3
  condition 4 requires a row's frozen retrieval input to carry the row's own release, and
  `TrialGPT2-TAIM-Luna-v1` can be in no release earlier than the TrialGPT 2.0 release. A Release
  0.1.1 freeze cannot be consumed by a row of a later release, and v11 forbids a second generation
  to make a later freeze from.
- **The code enforces both sides.** A freeze refuses a receipt that names another release than its
  own, and a receipt whose `system_id` is not `TrialGPT-TAIM-Luna-v1`; preflight and execution
  refuse a retrieval lock that does not bind v11 and carry its receipt. Release 0.1.1 does not
  carry this route, so the rows of v11 could not be produced on Release 0.1.1 as v11 wrote them in
  any case.
- **Moving the four rows is the smallest change that keeps every other rule.**
  `TrialGPT-TAIM-Luna-v1` has no paper row, no generation and no provider call for a paper row
  (Section 3), so moving its rows changes no value and re-cuts nothing. The alternatives each need
  an amendment too and are worse: a second freeze from the same generation under the later release
  leaves two retrievals of one generation; a second generation for `TrialGPT2-TAIM-Luna-v1` breaks
  v11 Section 5's one generation per target and gives the two Systems different candidates.

## 3. What was known when this was written

- **No `TrialGPT-TAIM-Luna-v1` effectiveness value existed**, as v11 Section 3 records, and none
  informed this amendment. **No generation under v11 had started** for any of the four targets
  when this amendment was written, and none had when it was frozen: no receipt and no row
  declaration had been committed.
- **No `TrialGPT2-TAIM-Luna-v1` output existed** when it was written or frozen. No TrialGPT 2.0
  scorer call had been made on any TAIM input, and this project had not executed TrialGPT 2.0's
  code.
- **Values published by TrialGPT 2.0's authors existed and had been read.** The TrialGPT 2.0
  paper (arXiv 2609.01202v1, submitted 2026-09-01), including its Results, had been read in
  scoping. Its values come from other backbones, other candidate sets and its authors' own
  evaluator. Random-order scores over the judged candidate sets its repository ships had also been
  computed in scoping (on TREC 2021, P@10 0.322 and nDCG@10 0.238), to show what that setting
  rewards. None of these informed this amendment, and none is a TAIM stratum.
- **Historical TrialGPT values existed, from a different System**: the ten-topic operational
  baseline frozen on 2026-08-10, as v11 Section 3 records. It did not inform this amendment.
- **The licence.** On 2026-10-03 TrialGPT 2.0's authors added a National Library of Medicine
  public-domain notice to their repository, at commit `1c5f9eb`, with a request to cite their
  paper. On 2026-10-06 the owner ruled that `TrialGPT2-TAIM-Luna-v1` may ship in a public release
  and become a paper row under v12, with its code pinned at that commit or later and the citation
  carried into the notices.
- **The cost was known.** Each System's four rows and their qualification were costed in provider
  calls while this was written. Cost bears on the order and pace of work, and on no rule here.

**The owner's rulings of 2026-10-06**, verbatim:

> **D1: yes - one amendment** letting our TrialGPT retrieval rows run on the release that first
> ships Way 2, admitted under v12. **One generation and one freeze serve both Systems.** Draft it
> for my review, not frozen.
>
> **D2: qualification gates once, on TREC 2021.**
>
> **D3: yes, extend the TREC 2022 authorisation** to Way 2.
>
> **D4: same stratum table, no paired comparison, with a caption note.**
>
> **D5: SIGIR description first** - after D1 is frozen and the System is built.

and, of TrialGPT 2.0 run through Codex:

> Record that through Codex it is comparable only to itself.

"Way 2" is `TrialGPT2-TAIM-Luna-v1`. "Our TrialGPT retrieval rows" are `TrialGPT-TAIM-Luna-v1`'s
four rows.

**The owner's answers of 2026-10-07 to the draft**, verbatim:

> 1. Ship the amendment with the release.
> 2. Keep v11's protocol ID on the receipt, no code change.
> 3. Yes: Way 2's qualification runs check coverage and rank stability only and never look at
>    scores.
> 4. Order (a): TREC 2021 generation, freeze and qualification first, then the SIGIR description.
> 5. Yes: "qualify once on TREC 2021" also covers TrialGPT 1.0.

and, on the same day, of the gates' pass rule:

> Qualification pass: option (a) - every call accounted for and the three repeats share every
> identity. Rank stability reported, no numeric bar.

## 4. The rule adopted: the release of the four rows

`TrialGPT-TAIM-Luna-v1`'s four rows (v5 Section 1) execute **the TrialGPT 2.0 release**, and are
admitted under protocol v12 Section 3, conditions 1 to 8, on the strata of v12 Section 4 that their
Profiles name. In particular:

1. **Declared before it runs.** Each row's v12 Section 7 declaration is committed before its run
   starts. Its `release` is the TrialGPT 2.0 release's.
2. **One release per row.** The retrieval lock each row consumes, its receipt, and every other
   artefact it consumes that carries a release identity, carry the TrialGPT 2.0 release's (v12
   Section 3 condition 4).
3. **Scoring.** Each row is re-scored by the Release 0.1.1 evaluator (v12 Section 3 condition 2).
4. **No Release 0.1.1 TrialGPT row.** No `TrialGPT-TAIM-Luna-v1` row is produced on Release 0.1.1,
   and none produced on any other release is admitted. A run of `TrialGPT-TAIM-Luna-v1` on another
   release is historical evidence, not a second row (v12 Section 3 condition 8).

`TrialGPT2-TAIM-Luna-v1`'s rows are admitted under v12 Section 3 on the same terms, on the same
four targets. Its rows are new runs: v12 Section 8's existing-run rule does not apply to them.

## 5. One generation, one receipt, one freeze per target, two Systems

**This is the load-bearing rule of this amendment.**

- **One generation per target**, as v11 Section 5 defines it, made on the TrialGPT 2.0 release.
  Its plans are the plans of both Systems' rows on that target. A generation is not repeated for
  the second System.
- **One receipt per target.** It is the v11 Section 6 receipt of that generation's freeze. Its
  `system_id` is `TrialGPT-TAIM-Luna-v1`, the System whose producer makes the retrieval, also when
  `TrialGPT2-TAIM-Luna-v1` consumes it. Its `release` is the TrialGPT 2.0 release's. Its
  `protocol_id` and `protocol_sha256` stay protocol v11's: the receipt is v11's record, and the
  code that checks it is not changed. This amendment is identified from what is written all the
  same. The receipt, the retrieval lock and each run record the TrialGPT 2.0 release's release
  manifest ID, and that manifest lists this amendment by its SHA-256, because the release ships
  it. Each row's declaration also names it.
- **One freeze per target.** It is the v11 Section 6 retrieval freeze, on the TrialGPT 2.0
  release, recording the receipt. Both Systems' rows on that target consume this one retrieval
  lock, by its SHA-256. A second freeze from the same generation is not made.
- **Identical candidates.** Both Systems are given, for each topic, the first 500 rows of that
  lock, which the lock fixes by its per-topic top-500 digests. `TrialGPT-TAIM-Luna-v1` matches and
  aggregates them (v5 Section 5); `TrialGPT2-TAIM-Luna-v1` scores them with TrialGPT 2.0's scorer.
  Neither System makes a query call.
- **The recomputation of v11 Section 8** is made once per target, after the freeze and before any
  effectiveness value of either System's row on that target is inspected. A failed recomputation
  stops both rows.
- **Two runs, two declarations, both before the generation.** Each System's row on a target is its
  own run, with its own run id, generation cache and v12 Section 7 declaration. Both declarations
  are committed before the target's generation starts, so each System is committed to the target
  before any of the target's provider calls is made. Beside the fields v12 Section 7 lists, each
  declaration carries `amendment`: this amendment's protocol ID and SHA-256, and the statement
  that the row consumes its target's one retrieval under it. A declaration precedes the
  retrieval, so it cannot name the lock; the row's run record names the retrieval lock's SHA-256
  it consumed, its disclosure names the receipt's commit, and both rows on a target must name the
  same.
- **The order of the two runs is free.** Either System's row on a target may run first. Neither
  row's run starts before the recomputation has been committed.
- **Failure.** An operational failure of the generation is handled once, under v11 Section 5, for
  both Systems. An operational failure of one System's run is handled under v12 Section 3
  condition 7 for that row alone; it never causes a second generation or a second freeze.
- **The provider account.** v11 Section 10 applies to both Systems: no rule, receipt field or row
  depends on the account, subscription tier or billing route through which the calls are made,
  and the runs are made on whichever account executes them. The Codex CLI version is part of the
  System and is held fixed for each generation and each run, as v11 Section 7 requires.

## 6. Qualification gates

Owner's rulings D2 and answer 5: the qualification gates of **both Systems** run **once, on TREC
2021**, on the external-fidelity target's frozen retrieval (Section 5), and are not repeated on the
other three targets. They are not waived.

- **Their shape.** Each System's gates are a one-topic validation and then three ten-topic repeats,
  on the topics of `TrialGPT-TAIM-Luna-v1`'s gates. For `TrialGPT-TAIM-Luna-v1` that is up to
  1,500 calls and then 3 x 15,000. For `TrialGPT2-TAIM-Luna-v1`, scaled to its one call per
  candidate, 500 calls and then 3 x 5,000, 15,500 calls in all.
- **What they read.** Answer 3: the gates judge coverage (resolved calls, failures, invalid outputs,
  dropped pairs) and run-to-run stability of the ranking. No effectiveness value is computed from a
  gate run against judgments, and none is read, so no score of the external-fidelity stratum's own
  topics is seen before that stratum's rows run.
- **When they pass.** The owner's ruling of 2026-10-07, verbatim: "Qualification pass: option
  (a) - every call accounted for and the three repeats share every identity. Rank stability
  reported, no numeric bar." For each System's gates: every call of the validation and of each
  repeat is accounted for, as resolved or as a recorded failure, with no candidate dropped; and the
  three repeats record the same identities, including the System Input, the retrieval lock and the
  release. Rank stability across the three repeats is measured and reported. No numeric bar applies
  to it, and no value of it passes or fails a gate.
- **When.** Answer 4: the TREC 2021 generation, freeze, recomputation and both Systems' gates come
  first in the order of work, before any row of either System on any target (Section 9).
- **Not rows.** A gate run is a qualification run. It is not a declared row, is never admitted, and
  does not count under v12 Section 3 condition 7. Its generation cache is never seeded into a
  row's (v9 Section 4). Every gate run is disclosed with its call counts and outcome.

## 7. Comparability of `TrialGPT2-TAIM-Luna-v1`

Owner's ruling of 2026-10-06, "Record that through Codex it is comparable only to itself", and
ruling D4, "same stratum table, no paired comparison, with a caption note". Recorded as:

> Through Codex, `TrialGPT2-TAIM-Luna-v1` is comparable only to itself: the Codex bridge sends
> TrialGPT 2.0's system and user messages as one prompt, cannot apply its temperature of 0.0, wraps
> every call in Codex's own harness, and is served by `gpt-5.6-luna` rather than the GPT-4.1 or
> GPT-5.4 its authors ran, so its numbers enter no paired comparison, difference or significance
> test with the TrialGPT 2.0 paper's, with `TrialGPT-TAIM-Luna-v1`'s or with any other System's,
> and stand in no table whose caption implies a shared model condition.

- **No paired comparison.** No v12 Section 6 comparison that includes a `TrialGPT2-TAIM-Luna-v1`
  row is registered, in its own declarations or in any other row's. Its declarations carry
  `comparisons: []`. That two Systems share a retrieval and identical candidates (Section 5) does
  not make them comparable: no difference between the two Systems' values on a target is computed,
  reported or tested.
- **The same stratum table.** Its rows are reported in the table of their stratum, beside the
  stratum's other rows, as v12 Section 6 permits.
- **The caption note.** Every table holding a `TrialGPT2-TAIM-Luna-v1` row carries, in its caption:
  "`TrialGPT2-TAIM-Luna-v1` runs TrialGPT 2.0's scorer through Codex on `gpt-5.6-luna`, re-ranking
  `TrialGPT-TAIM-Luna-v1`'s top 500. It is comparable only to itself: it enters no paired
  comparison, and its row does not imply a model condition shared with any other row."
- **The reason on its row.** Under v12 Section 11 item 6, the reason it enters no paired comparison
  is the recorded statement above.
- This section binds `TrialGPT2-TAIM-Luna-v1` only. `TrialGPT-TAIM-Luna-v1`'s comparisons remain
  under v12 Section 6, unchanged.

## 8. TREC 2022

Owner's ruling D3: the authorization protocol v5 Section 5 records, "The repository owner
authorized protected TREC 2022 access and these provider-backed experiments for this protocol", is
**extended to `TrialGPT2-TAIM-Luna-v1`** on the TREC 2022 judgment-union target
(`trec-ct-2022-judgment-union`, 50 topics).

- **What it covers**, and no more: giving that System the target's patient texts and the 500
  candidates per topic of the target's one retrieval (Section 5); sending them to the provider
  through the System's declared transport, as v5 already authorizes for `TrialGPT-TAIM-Luna-v1`;
  and re-scoring the row against the target's Evaluation Package by the Release 0.1.1 evaluator
  (v12 Section 3 condition 2), the only use of the judgments.
- **What it does not cover:** TREC 2023, for either System; TREC 2022 for any other System or
  experiment; qualification runs on TREC 2022 (Section 6); and any run. Each run still needs the
  owner's decision to run.
- This amendment is the frozen, version-controlled analysis protocol under which that access is
  made, and the owner's ruling D3 is the explicit authorization, as this repository's rule on
  protected Tracks requires.

## 9. Order of work

Answer 4 and ruling D5. First, after this amendment is frozen and `TrialGPT2-TAIM-Luna-v1` is
built and released, the TREC 2021 external-fidelity target: steps 1 to 6 below, ending with both
Systems' gates. Then the SIGIR description target, steps 1 to 8. Then the TREC 2021 rows and the
remaining targets. For each target, in order:

1. both rows' v12 Section 7 declarations, committed (Section 5);
2. the target's one generation, on the TrialGPT 2.0 release;
3. the v11 Section 6 receipt, committed;
4. the freeze;
5. the v11 Section 8 recomputation, committed;
6. on TREC 2021 only, both Systems' qualification gates (Section 6);
7. each System's row, in either order;
8. each row's re-score by the Release 0.1.1 evaluator.

## 10. Disclosure

Every reported row of either System, and the paper, discloses, beside v11 Section 12 and v12
Section 11 as adapted in Section 1:

1. this amendment by protocol ID, freeze date and SHA-256, and that it moved
   `TrialGPT-TAIM-Luna-v1`'s four rows from Release 0.1.1 to the TrialGPT 2.0 release before any of
   their generations, so that one generation could serve two Systems;
2. for each target, the generation's run id, the receipt's commit, the retrieval lock's SHA-256 and
   the per-topic top-500 digests, and that both Systems' rows on that target consumed that one
   lock;
3. that the two Systems were given identical candidates and are not compared (Section 7);
4. on every `TrialGPT2-TAIM-Luna-v1` row, the recorded statement of Section 7 and the caption
   note;
5. `TrialGPT2-TAIM-Luna-v1`'s upstream commit, its licence (the National Library of Medicine
   public-domain notice) and its authors' requested citation;
6. every qualification run of either System, with its call counts and outcome, and that no gate
   score against judgments was computed or read (Section 6);
7. that TREC 2022 access for `TrialGPT2-TAIM-Luna-v1` was authorized by the owner's ruling of
   2026-10-06 under this amendment (Section 8);
8. what was known when this was written (Section 3), including that the TrialGPT 2.0 paper's
   values had been read and did not inform it.

## 11. What this authorizes

This amendment defines on which release `TrialGPT-TAIM-Luna-v1`'s four rows execute, how one
generation, receipt and freeze per target serve that System and `TrialGPT2-TAIM-Luna-v1`, how both
are qualified, and how the latter is compared and disclosed. Its one authorization is the extension
of protected TREC 2022 access in Section 8, by the owner's ruling. It authorizes no run, no
release, no System registration, no provider call and no result admission. Each generation, gate
and row still requires the owner's decision to run, and every provider-backed phase remains under
its own authorization.

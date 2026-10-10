# TAIM paper analysis protocol: content equality by what a System reads

**Protocol ID:** `taim-paper-analysis-protocol-2026-10-05-v13`

**Status:** Frozen on 2026-10-05, the date in its protocol ID, as the repository owner ruled.
Written on 2026-10-03 for the owner's review, and revised the same day with the owner's rulings on
the first draft (Section 2). These bytes are immutable; a later change requires a new dated
amendment. It is frozen after v12. It ships with the next public release of this repository, as
v10 and v12 do, because readers comparing results outside this project need the rule. It
authorizes no run, no protected-data access, no provider call and no result admission
(Section 8).

This amendment replaces one clause: the content-equality check of protocol v12 Section 8 item 5. It
was prompted by that check's first and only use, which failed. The check compared every part of the
trial records an existing run was given, and most of them differed from the stratum's; it was then
found that the System that run executed never reads those parts, and reads its trial content from a
corpus of its own whose trial IDs were equal to the stratum's. The v12 check compared data the
System does not read. This amendment was written after that failure and because of it; Section 2
records when, why, and what was known. Under it, the parts on which that run failed no longer decide
its verdict. The amendment is general: the rule in Sections 3 to 6 names no System and applies to
every existing run that v12 Section 8 item 5 covers.

## 1. What this amendment changes

- **Protocol v12 Section 8 item 5**, the list of what the content-equality check compares (patient
  texts, trial texts as whole records, the clinical as-of date, the patient evidence profile, the
  judgments with their Judgment Scheme, and the rest of the Task Input). Section 3 of this amendment
  replaces that list. Everything else in item 5 stands: when the check applies, that both sides are
  derived with one codebase, that each derivation must reproduce the recorded identities, that the
  check reads no ranking and computes no score, that it is committed before it runs and runs once,
  that it rewrites no identity, and that a run that fails it is not admitted.
- **Protocol v12 Section 8 item 4**, the no-adjustment record. For a run checked under this
  amendment it also cites the consumption record of Section 4, by commit.
- **Protocol v12 Section 11 items 2 and 7**, the disclosure of a row admitted under v12 Section 8
  item 5. For a row admitted under this amendment, "its inputs were shown equal in content" reads
  "its inputs were shown equal in what its System reads". Item 7 required the paper itself to
  disclose, for such a row, the check's commit and its result field by field, that the clause was
  written after a score had been read, and that v4 Section 5's stop rule was relaxed. For a row
  admitted under this amendment, the paper meets item 7 by citing this amendment by protocol ID,
  freeze date and SHA-256: this amendment's own Section 2 and the row's supplementary record carry
  that disclosure (Section 7).

Every other rule stands unchanged, including all of v12 except the clauses above. In particular:

- the check exists only for an existing run whose recorded identities differ from its stratum's
  (v12 Section 8 item 5);
- the stop rule of protocol v4 Section 5, "Do not repair identity failures after inspecting
  scores", is relaxed for a run that passes the check exactly as v12 relaxed it, and no further;
- a run that fails is not admitted. Its ranking remains historical evidence, and its re-score may be
  reported beside the rows of its stratum, labelled as not a paper row (v12 Section 8);
- the comparison rules of v12 Section 6 and the environment record of v12 Section 5.

## 2. Why this amendment exists: what was known when it was written

This section is the record of how this amendment came about. It is part of the amendment, so that a
reader who has only this text can see when and why it was written, and what its authors knew.

- **2026-09-27.** TrialMatchAI v0.9.1, as its authors ship it, ran on TREC 2021, all 75 topics,
  Delta job 22464546. It recorded Profile `official-full`, Prepared Snapshot
  `sha256:d1b2258ced57c193e5ae6b9f6d29c393f6179be5b7b1a8a07370f15ca1518ac1`, Task Input
  `sha256:2d102c4d0d502d1fcad2ce39be66fd751b6d83a77dea26ceb7280a68dcf4cfb5` and Evaluation Package
  `sha256:42c67370c3908988a86a7d34676fab39f88b05fd3b94fd4b6abec42a6fa47c9f`, budget K 60. Its
  nDCG@10 under `official-full`, 0.557, was read that day (v12 Section 9).
- **2026-09-28.** Protocol v12 was frozen with the content-equality clause of its Section 8 item 5,
  written after that score had been read. Its check, `check_content_equality.py` at
  commit `7b33fc39cfde892d4645de2ecae14ca96d051edf` of the private source repository from which
  releases are generated, compared the whole Task Input record, trial by trial.
- **2026-10-03, 15:21:25Z to 16:03:50Z.** The v12 check ran once, on that run against the TREC 2021
  external-fidelity stratum. **It failed.** Its one reason was "trial_texts differ":
  - all six recorded identities were reproduced, the run's three and the stratum's three;
  - equal: the 75 patient texts in the same order, the clinical as-of date 2021-04-27, the patient
    evidence profile, the 35,811 judgments with their Judgment Scheme, and the rest of the Task
    Input;
  - the 26,149 trial IDs were the same and in the same order;
  - **25,611 of the 26,149 trial records differed**: all 25,611 in their typed clinical core, and
    5,294 of them also in their canonical text. The run's Prepared Snapshot was prepared by version
    2.0 of the TREC 2021 connector, the stratum's by version 2.2.

  Under v12 the run is not admitted. That verdict stands as v12's verdict; this amendment does not
  reverse it (Section 6).
- **2026-10-03, after the check.** As v12 Section 9 requires, the run's existing ranking was
  re-scored by the Release 0.1.1 evaluator against the stratum's Evaluation Package, and the value
  was read: nDCG@10 0.5567895841074783, equal bit for bit to the 2026-09-27 score. It is recorded as
  not a paper row, on a development branch of the private source repository.
- **2026-10-03, the finding.** Examining whether to run the System again on the stratum's own
  Snapshot, it was found that TrialMatchAI v0.9.1 never reads the trial parts that differed. In the
  code that run executed, its adapter gives the System only each patient's canonical text, and uses
  the Task Input's trials only to check that the System's own prepared corpus holds exactly the same
  trial IDs. The System reads trial content from that prepared corpus and from its search database,
  which were selected by trial-ID membership alone. A new run on the stratum's Snapshot would give
  the System the same inputs, and would measure only its run-to-run variation.
- **2026-10-03, the owner's ruling,** verbatim:

  > **Option 3, with a v13.** The content-equality check compared trial text that TrialMatchAI never
  > reads - **it reads its own corpus, whose trial IDs already match.** A re-run would measure only
  > randomness. **No re-run.**
  >
  > **Draft v13, not frozen**: the check compares **what each System actually consumes** - patient
  > texts, trial IDs, judgments, and any System-owned corpus by its pinned identity - **not data the
  > System does not read.** **General, for every System, not written around v0.9.1.** Show me the
  > draft.
  >
  > **Disclose plainly that v13 came after the v0.9.1 check failed**, and why. **Then re-run the
  > check under v13 on the existing run** - a few minutes, no GPU.

  The owner also ruled that day that the paper does not narrate this history but cites the
  protocol, so this amendment carries it; and that this amendment is first reviewed as a draft, then
  frozen with its check committed, and that the check then runs once on that run.
- **2026-10-03, the owner's rulings on the first draft.** Having read a first draft and its open
  points, the owner ruled:
  - the clinical as-of date, the patient evidence profile, the contract version, the task, the
    lineage, the available capabilities and the Derived Views are compared for every System,
    whether or not it reads them (Section 3);
  - trial IDs are compared in order for every System (Section 3);
  - an identity of a System-owned input that the run recorded is accepted, with the record saying
    how the run obtained it (Section 5);
  - every part classified as not read needs a committed test showing that changing it leaves what
    the System is given unchanged (Section 4);
  - a refusal before any comparison is not the check's one run, and no later amendment may
    re-check a run that fails under this one (Section 6);
  - the disclosure lives in this amendment's own record (Sections 1 and 7);
  - a System-owned input is compared with its own recorded and pinned identity only, not with an
    identity a release pins for the stratum (Section 5).
- **2026-10-03, where the run's corpus came from.** Before this amendment was frozen, the run's
  prepared corpus was compared with TrialMatchAI v0.9.1's own distribution. It is, byte for byte,
  the prepared TREC 2021 corpus that v0.9.1's `bootstrap-data` command downloads from its authors'
  Zenodo record 15516900, which is v0.9.1's documented route for TREC 2021; v0.9.1 has no way to
  prepare TREC 2021 trials from their 2021 texts. On that corpus, two of v0.9.1's scoring features
  never take effect: the 0.25 weight on exclusion criteria, because the corpus labels its criteria
  "Exclusion Criteria" and "Inclusion Criteria" while the code matches only "exclusion" and
  "inclusion"; and constraint scoring, because no criterion in the corpus carries constraints. The
  owner ruled that this be recorded, with the two effects disclosed beside it, and that this
  amendment then be frozen and its check run once.

**So this amendment was written knowing:** that the v12 check had failed on that run, and on which
parts; that the System does not read those parts; the run's score under `official-full` and its
re-score against the stratum; where the run's corpus came from, and the two scoring effects above;
and that under this amendment the parts on which the run failed no
longer decide its verdict. It was not written knowing the result of the check it defines, which has
not run.

## 3. The rule: compare what the System reads

The owner's ruling sets the principle: the check compares "what each System actually consumes -
patient texts, trial IDs, judgments, and any System-owned corpus by its pinned identity - not data
the System does not read".

An existing run whose recorded identities differ from its stratum's is treated as meeting v12
Section 3 condition 1 only when a content-equality check shows all of the following.

For every System:

1. **The judgments** of its Evaluation Package, with their Judgment Scheme, equal the stratum's.
2. **The trial IDs** equal the stratum's, in the same order.
3. **The rest of the Task Input** equals the stratum's: the clinical as-of date, the patient
   evidence profile, the contract version, the task, the lineage, the available capabilities and
   the Derived Views. They define the stratum's Task Input, and are compared whether or not the
   System reads them.

As the System reads them:

4. **Every part of a patient record that the System reads** equals the stratum's, patient for
   patient and in the same order.
5. **Every part of a trial record that the System reads** equals the stratum's, trial for trial and
   in the same order.
6. **Every System-owned input** has its recorded and pinned identity (Section 5).

The Prepared Snapshot ID that its Task Input names is not compared, as in v12.

**Parts the System does not read** are compared too, and their result is published with the
check's result, part by part. They never decide the verdict. A difference in a part the System does
not read is disclosed, not hidden.

**What a System reads** is fixed by its consumption record (Section 4) before the check runs. The
check does not decide it.

## 4. The consumption record: how what a System reads is established

**From the code, not the name.** What a System reads is established by reading the code the run
executed: the commit of this repository it ran from, or its release, and the System's own source at
the commit the run recorded. The code read is the System's adapter, which turns a Task Input into
what the System is given, and everything it calls on that path. A System's name, its documentation
and its authors' description are not evidence.

**What it classifies.** The consumption record classifies every part of a patient record and every
part of a trial record, as the Task Input carries them (for example canonical text, sections, typed
core, extensions, source identity and derived evidence), as read or not read. It also names every
System-owned input (Section 5). Each classification carries its evidence: file and line at the
commit the run executed.

**What counts as read.** A part is read when any of its content reaches the System, directly or
through something built from it: a prompt, a query, a file written for the System, an index, an
embedding, a cache. A System-owned input built from Task Input content makes that content read.
Where the code does not show that a part is not read, it is classified as read.

**A test for every part not read.** Each part classified as not read names a test, committed with
the consumption record, that shows it. The test runs the System's adapter on a Task Input, and again
on the same Task Input with that one part changed, and shows two things: that the change reaches
the Task Input, and that everything the adapter gives the System is unchanged. For a part whose
value the Task Input contract fixes, so that no two Task Inputs of one contract can differ in it,
the test shows that it is fixed. A part classified as not read without such a test refuses the
check.

**Fused and staged Systems.** For a System that consumes component runs, what each component read
is read by the row. The record covers every component as the row executed it.

**Fail closed.** A part that a side's Task Input carries and the consumption record does not
classify refuses the check before any comparison. A refusal before any comparison is not the
check's one run: nothing was compared. The record is completed, its tests added, and both are
committed again before the check runs.

**When it is written.** The consumption record is a JSON object named by the run id and committed
to the private source repository from which releases are generated, before the check runs, with its
tests. It records the System id, the commits it was read from, every classification with its
evidence and tests, and every System-owned input. It is never rewritten after the check has run.

## 5. System-owned inputs and their identity

**What they are.** A System-owned input is data the System reads that its Task Input does not carry:
a prepared corpus, a search database, an index, a concept store, a frozen retrieval input, a query
bundle, a cache of scores. The consumption record names each one.

**Its identity.** A content digest over its bytes, computed by a stated digest rule that the check
implements. A path, a timestamp, a file count, or a manifest that records where or how a copy was
built, is not an identity.

**Where it is recorded.** At least one of:

- **recorded by the run**, in a record the run wrote before its System started, which holds no
  ranking. The consumption record states how the run obtained that identity: whether it hashed the
  bytes it was given, or took the identity from an earlier record of them and checked something
  less, such as file names and counts;
- **pinned by the code**: a value fixed in the System's code, lock or release at the commit the run
  executed, which the System checks before it starts and refuses to run without.

**What it is compared against.** The check re-derives the digest from the bytes, from the copy that
was kept, and requires it to equal the recorded identity and the pinned identity, each where the
consumption record names one. An input whose bytes no longer exist cannot be shown and fails the
check. The stratum's rows are other Systems and carry no counterpart of this input, so it is
compared with its own recorded and pinned identity, not with the stratum.

**What it was built from.** The consumption record states, with evidence, what each System-owned
input was built from. Where it was built from Task Input content, that content is read (Section 4),
and is compared under Section 3 whether or not the input itself is equal. Where it was selected from
a larger set by trial ID, the trial IDs are read.

## 6. The check, and runs already checked under v12

**The check.** It is defined as a script at a named commit of the private source repository,
separate from the v12 check, which stays as it is. It is committed with the freeze of this
amendment, together with each consumption record and its tests, and it runs after the freeze, once
per run. As in v12, it derives both sides' Task Inputs and Evaluation Packages with one codebase,
the way a benchmark run builds them, and stops before any System call; it holds only if those
derivations reproduce the three identities the run recorded and the stratum's three; it reads the
prepared packages, the System-owned inputs and the run-start records that hold their recorded
identities; it reads judgments only to compare them; it reads no ranking and computes no score; it
rewrites no identity. A run that fails it is not admitted.

**A run already checked under v12.** The run of Section 2 failed the v12 check. It gets one check
under this amendment. That is not a second try under the same rule: this check compares different
parts, chosen by a different rule. It is a second check of the same run, under a rule written after
the first check failed and after the run's score and re-score had been read. That is why Section 2
records it. The v12 verdict is not withdrawn: the run failed v12's check and was not admitted under
v12. If it passes this check, it is admitted under this amendment, with both checks and both results
disclosed (Section 7). If it fails, it is not admitted, and its re-score stays where it is, labelled
as not a paper row.

**No further check.** A run checked under this amendment is not checked again under it, and no
later amendment may check again a run that fails under it.

## 7. Disclosure

Every table holding a row admitted under this amendment, and the paper, cite this amendment by
protocol ID, freeze date and SHA-256. This amendment's Section 2 is the account of how it came
about. The supplementary record of such a row carries:

1. its recorded identities beside the stratum's;
2. its consumption record and the tests of its parts not read, by commit;
3. the check's commit and its result, part by part, including every part not read and whether it
   was equal;
4. for a run checked earlier under v12, that check's commit and result, field by field, and that it
   failed;
5. that this amendment was written after that failure, and after the run's score had been read
   (Section 2);
6. that v4 Section 5's stop rule was relaxed for it, as v12 relaxed it.

## 8. What this authorizes

This amendment defines what the content-equality check compares, and how what a System reads and
the identity of a System-owned input are established. It authorizes no run, no release, no System
registration, no protected-data access, no provider call and no result admission. The check of the
run of Section 2 runs after the freeze, once, from the committed script; its admission still
requires its no-adjustment record and the owner's decision.

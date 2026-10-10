# TAIM paper analysis protocol: TrialGPT query plans generated per run

**Protocol ID:** `taim-paper-analysis-protocol-2026-09-28-v11`

**Status:** Frozen on 2026-09-28 under the repository owner's approval. These bytes are immutable;
a later change requires a new dated amendment.

No effectiveness value prompted this amendment. It was prompted by an artefact that cannot be
found: the four query-plan files protocol v6 Section 2 admits by SHA-256. It is frozen before any
`TrialGPT-TAIM-Luna-v1` query generation, retrieval or System call is made for a paper row
(Section 3).

## 1. What this amendment changes

It changes the query-plan rules for `TrialGPT-TAIM-Luna-v1` on its four protocol v5 targets, and
nothing else:

- **Protocol v6 Section 2** and **protocol v7 Section 2** admit exactly four query-plan files by
  SHA-256. For the four rows this governs, they are replaced by per-run query plans recorded by
  receipt (Sections 4 to 9).
- **Protocol v7, as the recorded `protocol_approval_id`.** Release 0.1.1 fixes the TrialGPT
  retrieval protocol digest in code to v7's
  (`sha256:a617a46c2d61e76ad25963e709cad2c3ff99d4be466f173b8add02f1754927ef`), and its retrieval
  freeze refuses any other value. Every retrieval lock a 0.1.1 row produces will therefore record
  v7. **For these rows this amendment overrides v7**, including v7 Section 2 and v7 in its role as
  the recorded `protocol_approval_id`. The governing protocol of each row is this amendment, and
  the row's receipt records it (Section 6).
- **What each record cites.** A per-run record cites this amendment's digest. On Release 0.1.1
  that record is the receipt, because the release's retrieval lock can only record v7. In a later
  release that records the receipt in its retrieval lock, the lock cites this amendment's digest.

Every other rule stands unchanged:
- v5: targets, the three-arm query method, retrieval, consensus, the provider budget, evidence and
  admission;
- v9: Release 0.1.1 as the executable release, and the top-20 Primary Ranking;
- v4: the matrix, reporting and stop rules;
- v8: admission to comparison;
- v1: everything not narrowed elsewhere, with Section 8 remaining the sole canonical statement of
  the recall-floor asymmetry.

v8 and v9 are identified by protocol ID and SHA-256 in the current rules document,
`paper-analysis-rules.md`. This release does not carry their text.

The System id stays `TrialGPT-TAIM-Luna-v1`. What distinguishes a row built from per-run plans is
its run-specific query-plan digest, recorded in its receipt. Nothing in the System id changes.

## 2. Why

The four files protocol v6 Section 2 admits are not reachable from anything this project can
search. They also cannot be recreated. A query-plan file records, for each call, the hashes of the
provider's raw and parsed outputs, so repeating the 735 calls yields new bytes and a new digest.
Without this amendment the four rows cannot be produced by anyone.

## 3. What was known when this was written

- **No `TrialGPT-TAIM-Luna-v1` effectiveness value existed**, and none informed this amendment.
  Protocol v9 Section 4, drafted on 2026-09-18, records that no `TrialGPT-TAIM-Luna-v1` assessment
  existed when its budget K was declared. The owner has authorized no provider call for it since.
- **Historical TrialGPT values existed, from a different System.** The known one is the ten-topic
  operational baseline frozen on 2026-08-10. It came from a historical TrialGPT configuration that
  no release carries: a single keyword query per topic, `taim-controlled-bm25-v1` retrieval, ten
  topics, and a legacy identity contract. Its values were known when protocols v5 to v7 were
  frozen. They did not inform this amendment.
- This amendment is frozen before any query generation, retrieval freeze or System call is made
  for any of the four rows. A generation made before the freeze date cannot become a paper row's
  generation.

## 4. The rule adopted

This amendment adopts the rule of an earlier amendment, "TrialGPT query plans as run inputs",
dated 2026-09-07 and written by Hannes Ill for a separate development line of this benchmark,
held in a private repository. No release of this repository carries it. Its bytes are identified
by commit `3eb63b0472d51d6d2d09ba2ab712364c7bebf5d5` and SHA-256
`sha256:c19742e96ac36e657955f7def60e4166017eda64bad0adeefd4eb4d11a274a0e`. Its text is not
reproduced here. The rule it states, as adopted here, is:

- Every new run may produce query plans by the declared three-arm Luna method, which is the
  method of protocol v5 Section 3, or may reuse compatible plans (narrowed by adaptation 2 below).
- Plans are compatible only if all of the following validate: the full set of topics, the real
  patient text, the contract for prompts and output schema, the model with its reasoning
  configuration, the identity of each logical call, the evidence of resolved attempts, and the
  output hashes.
- No plan is generated or chosen with any retrieval or effectiveness outcome in view.
  Alternatives are not generated and chosen between by their scores.
- Before retrieval, a query-plan receipt is frozen. It names the run ID, Track, Profile, the
  SHA-256 of the exact query-plan file, and the protocol digest. It is addressed by its content and
  cannot change for that run.
- The query plans of every target selected for execution are finished before any retrieval or
  System scoring begins.

With these adaptations:

1. **Release.** The 2026-09-07 amendment requires that a release manifest and package be newly
   verified before reportable execution. That requirement is overridden. The four rows execute
   Release 0.1.1 as protocol v9 Section 5 requires. Release 0.1.1 carries no receipt machinery, so
   the receipt is procedural (Section 6).
2. **"Or reuse compatible plans"** is narrowed. The only plans a row may use are the plans of its
   target's one generation (Section 5). Plans from another run, another target, or any generation
   made before this amendment's freeze date are not reused.
3. **The four files protocol v6 Section 2 admitted** are not used by these rows. They are not
   available (Section 2).
4. **Reuse of precomputed deterministic artifacts**, which the 2026-09-07 amendment permits when
   their full identities and checksums match, is not adopted. Protocol v7 Section 4's fresh-run
   rule, applied to Release 0.1.1 by protocol v9 Section 5, stands: every row produces fresh
   retrieval and System artifacts.

## 5. One generation per target

**This is the load-bearing rule of this amendment.**

- A **generation** is one execution of the v5 Section 3 query method over a target's complete
  topic set, in one run workspace, with one generation cache.
- Each of the four targets has **exactly one generation**. Its plans are the plans of that target's
  row. No alternative plan set is generated, compared or selected.
- **Every started generation is disclosed**, whether it completed or not, with its run ID, start
  and end times, call counts and outcome.
- A second generation for a target is permitted **only for an operational failure** that makes the
  first generation impossible to complete in its own workspace, such as a lost or corrupted
  workspace or cache. The failure, its evidence, and the reason the first generation could not be
  resumed are recorded before the second generation starts. A result is never an operational
  failure. Neither is any retrieval, effectiveness or plan-quality outcome, nor dissatisfaction
  with the plans.
- A generation that completed is never replaced.

## 6. The procedural receipt

Before the retrieval freeze of a row runs, a receipt, a JSON file named by the run ID, is written
and **committed** to the private source repository from which releases are generated. The commit
that adds it is the receipt's anchor, and the freeze may not start before that commit exists. The
receipt and the SHA of that commit are disclosed with the row (Section 12).

The receipt is a JSON object with exactly these fields:

| Field | Value |
| --- | --- |
| `artifact_type` | `taim-trialgpt-query-plan-receipt` |
| `schema_version` | `1.0` |
| `protocol_id` | this amendment's protocol ID |
| `protocol_sha256` | this amendment's SHA-256 |
| `system_id` | `TrialGPT-TAIM-Luna-v1` |
| `run_id` | the run ID the freeze will be given |
| `track`, `profile` | the target |
| `prepared_snapshot_id` | the Prepared Snapshot the plans were generated from |
| `release` | Release 0.1.1's release manifest ID, public tree ID and package artifact ID |
| `query_plans_sha256` | the SHA-256 of the completed query-plan file |
| `logical_calls` | three times the target's topic count |
| `generation` | model, the three arms' prompt contracts and efforts, the Codex CLI version and executable path, the SHA-256 of the generation cache at completion, total provider attempts, and the list of interruptions with their clearing records (Section 7) |
| `generation_ordinal` | `1`, or a higher number only under Section 5's operational-failure rule, in which case it names the earlier generations and their recorded failures |

The retrieval lock that follows must record the same run ID, target and `query_plans_sha256`. A
mismatch stops the row. A later receipt for the same run ID is never written.

## 7. Interruption and resume

- An interruption of a generation is **operational**. Examples: a usage allowance running out, a
  provider or transport failure, an expired or signed-out session, a host restart, an interrupted
  process.
- The same generation **resumes from its cache in the same run workspace**. Completed calls are
  not repeated. The resume is not a new generation and not a re-roll, and it does not count under
  Section 5.
- Each interruption is recorded in the receipt's `generation.interruptions` with time and cause.
  Its cause is described generically, for example "usage allowance exhausted", and never by
  account or plan name.
- The Codex CLI executable path and version are held fixed for a whole generation. Release 0.1.1
  records both in every call's provider configuration, which is part of the call's identity, so a
  resume under a different path or version would silently repeat every call. If an unavoidable
  version change invalidates cached calls, it is recorded as an interruption. Only the invalidated
  calls are made again, within the same generation.

### 7.1 Clearing cached failures before a resume

Terms used in this subsection:
- An **entry** is one logical call's record in the generation cache.
- An **attempt holds provider output** when any of its `raw_output`, `parsed_output`,
  `raw_output_sha256` or `parsed_output_sha256` is present.
- A **failed provider call** is an attempt that Release 0.1.1 classifies as `authentication` (a
  sign-in failure) or `transport`.

Rules:
1. **What may be cleared.** Only an unresolved entry whose every attempt is a failed provider call
   classified `authentication` or `transport`, and of which no attempt holds provider output.
   **Nothing holding a model response is ever removed.** Resolved entries are never altered or
   removed.
2. **Why clearing is needed.** Release 0.1.1 stores a sign-in failure as a final unresolved
   result, and a resume returns it unchanged instead of retrying it. The release retries a
   transport failure itself. It may still be cleared under rule 1.
3. **The check runs before every resume**, in the query stage and in the System stage, while no
   process holds the cache.
4. **The resume is refused**, and nothing is cleared, if either of the following exists:
   - (a) an unresolved entry that has a failed provider call and also holds provider output.
     Such an entry cannot be cleared under rule 1, and the release would not retry it.
   - (b) an unresolved entry that holds provider output and that Release 0.1.1 would ask again on
     resume, overwriting it. In the query stage, that is every unresolved entry holding provider
     output whose last attempt is not a transport failure, for example three invalid structured
     outputs. In the System stage the release keeps such entries, as failure-aware abstentions.

   A refused resume stops the generation. The owner decides what follows, and the refusal is
   disclosed.
5. **Each clearing is recorded**, and the record is committed with the receipt. It holds:
   - the UTC time;
   - the run ID and stage;
   - the cache's SHA-256 before and after;
   - the number of entries removed;
   - for each removed entry: its logical-call identity, topic, trial where there is one, stage,
     attempt count, the failure class of each attempt, the SHA-256 of its canonical JSON, and the
     entry itself, verbatim;
   - the refusal check's result: zero entries matching rule 4.

The System stage follows the same rules. Its generation cache resumes in place in its run
workspace, as protocol v9 Section 4 already requires, and each interruption and clearing is
recorded in the run's evidence.

## 8. Retention and post-hoc recomputation

Release 0.1.1 checks that generation traces are well-formed. It does not recompute them. So the
evidence that each plan was actually generated rests on retained artefacts:

- For every generation, the run keeps the complete generation cache, the provider traces, and the
  query-plan file, in addition to what v5 Section 6 retains. They are kept local unless the terms
  that govern their access and redistribution allow them to be moved.
- After the freeze and before any effectiveness value for the row is inspected, the query-plan file
  is **recomputed from the retained cache**. The Release 0.1.1 query method is re-executed against
  that cache with a provider that refuses every call. It must rebuild a byte-identical file, which
  shows that every row's output hash, logical-call identity and trace resolve from the recorded
  provider outputs. The result is committed beside the receipt, as a JSON file named by the run ID
  with the suffix `.recomputation.json`. A failed recomputation stops the row.

## 9. No reproduction of the admitted plans

A row this amendment governs cannot be described, reported or relabelled as reproducing the plans
protocol v6 admitted. As Section 1 states, per-run records cite this amendment's digest.

## 10. Independent of the provider account

No rule, receipt field or row depends on the account, subscription tier or billing route through
which provider calls are made. The account determines only how fast calls can be made. Section 7
covers a usage allowance running out. The transport is recorded because it is part of the System:
the Codex CLI version, model and effort. The account type is disclosed as a fact about execution,
not as an identity.

## 11. What these rows are not

The rows are not reproductions of an experiment on protocol v6's admitted plans, and not of
upstream TrialGPT. Query plans are now a per-run stochastic input. Their variance is not measured,
because each row is one generation and one replicate.

## 12. Disclosure

Every reported row, and the paper, discloses:

1. that the query plans were generated per run, for each target, with the protocol v5 Section 3
   method;
2. why: the four files admitted by protocol v6 Section 2 could not be located, and cannot be
   recreated byte for byte;
3. this amendment, by protocol ID, date and SHA-256, and the 2026-09-07 rule it adopts with its
   adaptations;
4. each row's run-specific query-plan SHA-256, beside the protocol v6 digest it replaces;
5. that the `protocol_approval_id` recorded in each Release 0.1.1 retrieval lock is protocol v7's
   digest, fixed by the release's code, and that this amendment governs the row;
6. that Release 0.1.1 does not check query-plan digests or recompute generation traces, that
   each row's plan provenance rests on its retained cache and traces and its committed
   recomputation, and where that evidence can be obtained;
7. one generation per target, no alternatives, every started generation, every interruption and
   every clearing of cached failures, and the freeze date of this amendment relative to the first
   provider call;
8. the historical TrialGPT values that existed when it was written (Section 3), and that they came
   from a different System and did not inform it;
9. that plan-sampling variance is not measured, and that no equivalence with upstream TrialGPT or
   with the unavailable admitted plans is claimed;
10. the transport: Codex CLI version, model and reasoning efforts, and that the System sends each
    prompt as a single message with no sampling controls, by its own definition.

## 13. What this authorizes

This amendment defines how the four rows obtain and record query plans. It authorizes no
protected-data access and no result admission. Provider-backed execution remains under protocol
v5's authorization and the repository owner's decision to run.

## Source of the 2026-09-07 rule

- Title: "TrialGPT query plans as run inputs".
- Author: Hannes Ill. Dated 2026-09-07. Committed 2026-09-07T10:43:04Z as commit
  `3eb63b0472d51d6d2d09ba2ab712364c7bebf5d5`.
- SHA-256 of its bytes: `sha256:c19742e96ac36e657955f7def60e4166017eda64bad0adeefd4eb4d11a274a0e`.
- It belongs to a separate development line of this benchmark, held in a private repository, and
  no release of this repository carries it. The repository owner holds its bytes. Any copy can be
  checked against the digest above.

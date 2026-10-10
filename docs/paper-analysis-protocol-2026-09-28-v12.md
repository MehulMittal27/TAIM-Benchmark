# TAIM paper analysis protocol: rows from later releases, admitted by evaluation identity

**Protocol ID:** `taim-paper-analysis-protocol-2026-09-28-v12`

**Status:** Frozen. Approved by the repository owner on 2026-09-28, the date in its protocol
ID, with the owner's rulings of that date applied, including the content-equality clause of
Section 8 item 5. These bytes are immutable; a later change requires a new dated amendment. It is
frozen after v11, and it ships with the release, as v10 does, because readers comparing results
outside this project need the rule. It authorizes no run, no protected-data access, no provider
call and no result admission (Section 12).

No effectiveness value prompted the admission rule. It was prompted by a structural limit: protocol
v9 Section 5 makes Release 0.1.1 the only release a paper row may execute, so a System first
registered in any later release can never be a row. This amendment is general: it names no System.
It admits new runs that are declared before they start (Section 7) and runs that already exist,
with their timing disclosed, unless something about the run was adjusted after its results were
seen (Section 8). The existing-run rule was ruled by the owner after a score of one existing run
had been read (Section 2); the paper says so (Section 11, item 7). So was the content-equality
clause of Section 8 item 5, which relaxes a stop rule of protocol v4 Section 5 for an existing run
whose recorded identities differ from its stratum's but whose inputs are equal to it; the paper
says that too.

## 1. What this amendment changes

It adds one way for a row to be admitted, and changes nothing about the rows already admitted:

- **Protocol v9 Section 5** (`taim-paper-analysis-protocol-2026-09-18-v9`, SHA-256
  `sha256:49f02a27bcee9737cc99344cf1716ee40f143295641db895971f3325fdc30aa9`, lines 88-114). v9 is
  identified by protocol ID and SHA-256 in the current rules document, `paper-analysis-rules.md`;
  this release does not carry its text. Release 0.1.1 remains the executable release of every row
  admitted under it, and those rows are unchanged (Section 10). A row may also be admitted under
  Section 3 of this amendment. v9 Section 5 rule 4, "Rows are paired, differenced or pooled only
  within Release 0.1.1", is replaced for pairs that include a row admitted here by Section 6.
- **Protocol v7 Section 4**, as v9 Section 5 applies it to 0.1.1. Its requirement that the rows of
  the TREC 2021 external-fidelity stratum "share its exact manifest, public tree, package artifact,
  and dependency lock" does not apply between a row admitted here and a Release 0.1.1 row.
- **Protocol v4 Section 2**, the sentence "All seven rows must use the same Task Input, Evaluation
  Package, pool receipt, release manifest, public tree, package artifact, and dependency lock" (v4
  lines 93-94 in TAIM's mirror). For a row admitted here the Task Input, Evaluation Package and pool
  receipt must still be the same; the release manifest, public tree, package artifact and dependency
  lock are recorded, not required to match.
- **Protocol v4 Section 5**, the words "Every final row must be a fresh run" (v4 lines 172-174). A
  run that already exists when this amendment is frozen may be admitted without being run again,
  under Section 8. Every other requirement of that sentence stands.
- **Protocol v4 Section 5**, the stop rule "Do not repair identity failures after inspecting
  scores". An existing run whose recorded identities differ from its stratum's is treated as
  meeting Section 3 condition 1 when a check, committed before it runs, shows that its inputs equal
  the stratum's in content (Section 8 item 5). For such a run this relaxes that stop rule. The
  relaxation was decided after a score of an existing run had been read, and the paper says so
  (Section 11, item 7). No identity the run recorded is rewritten.
- **Protocol v8 Section 2**, as restated by protocol v10 Section 2. For two Primary Rankings with
  different System ids where at least one is a row admitted here, Section 6 replaces the requirement
  that Package Artifact ID and Dependency Lock digest match. For two rankings with the same System
  id, v8 stands unchanged.

Every other rule stands unchanged:
- v4 Section 5: every row still binds its Track, Task, Profile, Prepared Snapshot ID, Task Input
  ID, System Input ID, Evaluation Package ID, pool receipt ID, Primary Ranking, release manifest,
  public tree, package artifact and dependency environment. "The amended immutable public release"
  is, for a row admitted here, that row's own release (Section 3, condition 3), or for an existing
  run that executed no release, its recorded commit (Section 8). Its stop rules, including "Do not
  repair identity failures after inspecting scores", apply to every row admitted here, new or
  existing, except where Section 8 item 5 relaxes that one stop rule for an existing run whose
  inputs equal its stratum's in content.
- v1: metrics, statistics, failure rules, stop rules and claim boundaries. Section 8 remains the
  sole canonical statement of the recall-floor asymmetry. The required paper bundle of Section 10,
  including item 5, applies to every row admitted here.
- v3, v4, v5 to v7, v9 Sections 1 to 4, v10, and v11, which is frozen before this amendment. A row
  governed by v11 still executes Release 0.1.1 as v11 requires.

## 2. What was known when this was written

- **The 25 Release 0.1.1 rows existed and their values had been read**: the BM25 row, the BGE-M3 and
  Qwen3-Embedding-0.6B dense rows and the reciprocal-rank-fusion row on each of the six strata of
  Section 4, and the Release 0.1.1 TrialMatchAI L4 row on the external-fidelity stratum. This
  amendment changes none of their values, admits none of them a second time, and changes no
  comparison between any two of them, since all of them share Release 0.1.1. Each ran on an NVIDIA
  A40, driver 595.71.05, as the job-start record of each of their seven jobs reports.
- **A measurement existed of at least one System that may later be registered, and its score had
  been read**: TrialMatchAI v0.9.1 on TREC 2021, all 75 topics, Delta job 22464546 on 2026-09-27,
  recorded as "not a paper row" on a development branch of the private source repository. Its
  nDCG@10 of 0.557, under Profile `official-full` over a 75-topic Snapshot, was read on
  2026-09-27. The owner ruled the existing-run rule of Section 8 on 2026-09-28 with that run in
  view. The rule is general and names no System; Section 9 applies it to that run.
- **When the content-equality clause of Section 8 item 5 was written,** on 2026-09-28, that score
  had been read. Four other things were known:
  - the run recorded a Profile, Prepared Snapshot, Task Input and Evaluation Package that differ
    from every stratum's;
  - it recorded the external-fidelity stratum's 26,149 trial IDs in the same order, its 75 topics,
    its clinical as-of date and the qrels file its judgments were parsed from;
  - the manifests of the two Prepared Snapshots record different versions of the TREC 2021
    connector, 2.0 and 2.2;
  - a full-corpus package prepared by connector 2.0 carries a different trial content hash from the
    stratum's full-corpus package, so the connector change altered some trial records.

  Nobody had checked which records differ, or whether any of the 26,149 pool trials, any patient
  or any judgment differs between the two. The clause and its check are committed before the
  check runs.
- **The card sensitivity of the bfloat16 encoder was known** and is the reason for Section 5: on the
  SIGIR summary stratum, the Qwen3-Embedding-0.6B dense row's nDCG@10 differed by 0.0045573195
  between an A100 and an H200 under one release, and by 0.0011772899 between an A40 and that A100
  across two releases, while the BM25 row, the BGE-M3 dense row and the reciprocal-rank-fusion row
  were bit-identical (a private evidence record of 2026-09-22, commit
  `87dedcdc14d9a029ee088489abc835d24977fffc` of the private source repository from which releases
  are generated).
- The rule is stated for every System and every stratum alike.

## 3. The admission rule

A Primary Ranking is admitted as a paper row on one of the strata of Section 4 only when all of the
following hold. Conditions 1 to 4 and 8 apply to every row. Conditions 5 to 7 apply to a run that
starts after the freeze; a run that started before it meets Section 8 in their place.

1. **Evaluation identity.** Its run records exactly the values Section 4 lists for that stratum:
   Profile id and Profile definition SHA-256, Prepared Snapshot ID, Task Input ID, Evaluation
   Package ID, pool count, pool IDs SHA-256 and pool receipt ID. The Task Input ID fixes the
   patients, the trials, the clinical as-of date and the patient evidence profile. The Evaluation
   Package ID fixes the judgments. The Profile definition fixes the scoring rules: cutoffs, gains,
   discount, unjudged policy, precision metric and denominator, ideal and topic policy. An existing
   run whose recorded identities differ can meet this condition only under Section 8 item 5.
2. **Scoring.** Its scorecard is recomputed from the row's own candidates and the stratum's
   Evaluation Package by the Release 0.1.1 evaluator,
   `taim.public_patient_to_trial._evaluate_profile_cutoff` from the Release 0.1.1 wheel, SHA-256
   `sha256:2e09a6d288a96f791c3d0f77e05d754ce5b56f8158c3a039063d0670d765c277`. Where the row carries
   its own scorecard, every value the Release 0.1.1 scorecard carries, on the aggregate and on every
   topic, at every cutoff of the Profile, must equal the row's own scorecard exactly. Where the row
   carries none, because its System had no release identity when it ran, the recomputed values are
   its values. The paper reports the recomputed values. A row whose candidates the 0.1.1 evaluator
   cannot read is not admitted.
3. **Release.** The run executes an immutable public release of this repository at 0.1.1 or later,
   recorded in the release ledger of the private source repository from which releases are
   generated, whose support matrix declares the System on that Track, Task and Profile. The run
   records everything v4 Section 5 requires it to bind. Section 8 states the one alternative, for
   an existing run.
4. **One release per row.** Every artefact the row consumes that carries a release identity carries
   the row's own: for a fused or staged row, every component run; for a row with a frozen retrieval
   input, that input; any index. An artefact from another release is not consumed, relabelled or
   admitted, as v9 Section 5 rule 3 already states for 0.1.1. An existing run admitted from a
   recorded commit consumes artefacts produced from that commit or recorded with it, never an
   artefact of a release.
5. **Declared before it runs.** A row declaration (Section 7) is committed to the private research
   repository before the run starts.
6. **Run after the freeze.** The run starts after the commit that freezes this amendment.
7. **One run per declared row.** A second run for a declared row is permitted only for an
   operational failure that prevents the first from completing, such as a lost workspace, a node
   failure or a wall-time kill. The failure and its evidence are recorded before the second run
   starts. A score is never an operational failure. A completed run is never replaced, and every
   started run is disclosed.
8. **One row per System per stratum.** A System id that already has a paper row on the stratum,
   admitted under v9 Section 5 or under this amendment, is not admitted there again. Re-running a
   Release 0.1.1 System in a later release produces historical evidence, not a second row.

A row that fails any condition is not admitted. It is not repaired, relabelled or re-run under a
different identity after any of its values has been read. Section 8 item 5 does not repair a row:
every identity the run recorded stays as it was recorded.

## 4. The strata and their identities

These are the identities every Release 0.1.1 row of the stratum records. On 2026-09-28 the code on
`main` at `852ea28` reproduced every value in this table from the same prepared packages, and the
Release 0.1.2 wheel reproduced the SIGIR summary row and all six Profile definitions.

| Stratum | Profile | Profile definition | Prepared Snapshot ID | Task Input ID | Evaluation Package ID | Pool count | Pool IDs SHA-256 | Pool receipt ID | Clinical as-of |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| TREC 2021 judgment-union | `trec-ct-2021-judgment-union` | `sha256:6bc44336d0e3d51be3dbcd09a89ede547b81d7acdee82bd62e02158b011ab941` | `sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8` | `sha256:c393b4f1034db49a2e5fb5a60d8ee49ea7c488b59a8d224dd1530d905298df30` | `sha256:fa47e8e588e06a0d69f3fc32ea34f59113c875bdc8219327ce8faaa8930e9dba` | 26,162 | `sha256:1b0d2f34b76dbc96fb5c7c88d73271ca205a39ce4db32c6407b7031b250b55da` | `sha256:8c8e7f4b8ad03254fec917cdf71ec32306a55242a144b249fa9e8c12b2d8f6f5` | 2021-04-27 |
| TREC 2021 external-fidelity | `trec-ct-2021-external-fidelity-26149` | `sha256:358292a857f071ba3fb39653eeb1368cebb75e265c2e7ee4a79e553a21db0ef9` | `sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8` | `sha256:019f47c8e4fafb6a6f7709ea7accc0c882b915aff4f3bc21f4d68c53e0d520ca` | `sha256:d35d98a80b59be82aaa27da7b624bd60eba82939bd78015f707b471477015c28` | 26,149 | `sha256:fed85fadb5a0e0a42a39ed9cf65984926a13f66d4a2a0a1deb100996ac56ffa7` | `sha256:b6e7f202c72c260f423a71d733c092cc4eba24fde1f5095d1c9f334e7d753d7a` | 2021-04-27 |
| TREC 2022 judgment-union | `trec-ct-2022-judgment-union` | `sha256:3a83d2ddb9cb33c83dd983cf0466858fac80325146d3de471cb468eda6f93a3b` | `sha256:17d6af463adc9827ccdfe3ac69151d8fcb63996086d0b0186c4d51e0266b8d9f` | `sha256:23dc7a334eb13b578bb55e5bfc554fcb13b94efd6ecde74f0024b7dcdfc8a9cb` | `sha256:08865f0aace04f6dd3a805f536a292ed78621837bf66ceb8fd4bbd68ad47b8e9` | 26,585 | `sha256:c56129596aac032d62644fc851d61af95da77f470a164650d0a4b953f261c553` | `sha256:3cb4876310d49cf03fdced7dc2958f4971d64bf357d18ce4401f0f3fd108411f` | 2022-01-01 |
| TREC 2023 judgment-union | `trec-ct-2023-judgment-union` | `sha256:c38d7f0983bdf15954a74d76f63b4f9367aba3c99b42117d5c33f6c3d213aafd` | `sha256:87723d564332819d3b1ee67507ff5ec747305d355b42a09fdbc24c458cddc00a` | `sha256:52ab379ec2f2778f131a0c201c93b95474d42beab31cfbe9a6251ed99d10f521` | `sha256:11c7f35b2b83aab03e63a54bb4c0784654cf698ff47182b77e9592946ef7c370` | 17,105 | `sha256:68d02ba21e41ab48a86bd717c19a762715a190b5c0d2000dbe3a670f57d488ad` | `sha256:b3d4debb19ddaf830a9ce9241c6a4900f8c3e179d1ac1c73b1cea6a51b072de6` | 2023-05-08 |
| SIGIR 2016 summary | `sigir-ct-2016-summary-judgment-union` | `sha256:0ef0aedf49dc34b5e85ce47ff0b1a8d5a94c1d781ca3186f868341336946e743` | `sha256:3c571a03f10419ee5782c2d9aedae1e4825dbf012035bbf1bdbf27d099fc8cfc` | `sha256:e371944d1bfc5b9a8aab24639c5d2c1547422ec0d9c0bdd18fef9899be960938` | `sha256:aa811def8436f58445116fef62539876c45164c97afec318d87a34fa27e89760` | 3,626 | `sha256:9a106146c9e1a5da49e15cda3dcc44add3c028e6b0d26a55a40bf60813f5ec5a` | `sha256:8696755dfeaf1a6029f803d59c29bae62cb38db5ea76a0c37c856e668b18acf4` | 2015-12-16 |
| SIGIR 2016 description | `sigir-ct-2016-description-judgment-union` | `sha256:ccaf5ad32ee527ef2c1831276111e67678819befcd2a73f5f345d242281d4391` | `sha256:c8fe1bce5118520f6e07972cece00e5d1953a3d96a118e29d7b5e1b2fa0f97f8` | `sha256:c05154f17b6be8168c0c89e4fdb62813913cf50cec2088796ed9293c6e957578` | `sha256:3067020a9f9bf8613700a8737418a2130b0b72951e62dfd009562f6afdea3f6c` | 3,626 | `sha256:9a106146c9e1a5da49e15cda3dcc44add3c028e6b0d26a55a40bf60813f5ec5a` | `sha256:10ff30fa39ca1e35fab675396b7648dfc78b2cf8c7a0780312be107b02941217` | 2015-12-16 |

The clinical as-of date is part of the Task Input ID: the same Profile run at another date keeps
its trials and scores and changes its Task Input ID, so a row at any other date does not match.
Rows on any other Profile, including `official-full`, are not admitted under this amendment. The
one exception is an existing run that meets Section 8 item 5 on one of these strata. The
26,162 and 26,149 strata still share no comparative table, statistical test or corpus identity (v4
Section 2).

## 5. Hardware, lock and the environment record

**What the evaluation identity does not fix.** The identities of Section 4 fix what a System is
given and how its ranking is scored. They do not fix how the ranking was produced: the System's
code and configuration, model weights and revisions, the dependency lock, the hardware, caches and
indexes, provider responses and any non-determinism. Each of these is part of the row, and is
recorded with it.

**The environment record.** Every paper row, whether admitted here or under v9 Section 5, carries,
beside the identities v4 Section 5 binds:

1. the release version, release manifest ID, public tree ID, package artifact ID and dependency
   lock digest, with the resolved distributions the run records; for an existing run admitted from
   a recorded commit (Section 8), that commit, the System's own source commit and tree, and every
   dependency lock digest the run used, in their place;
2. **the card**: the card as the executing node reported it at job start - name, memory, driver
   version and compute capability - read from a record written before the row ran, not from the
   run id. A row that ran on no GPU records the processor architecture and operating system in its
   place. A row with no such record written before it ran records "card not recorded"; that is
   disclosed and does not refuse the row;
3. the model identity the run records: model and tokenizer ids and revisions, dtype, attention
   implementation;
4. for a provider-backed System, the provider, model, transport and client version, as v5 and v11
   already require of TrialGPT.

This is protocol v1 Section 10 item 5, "Exact model, tokenizer, runtime, driver, CUDA, and
attention-backend identity", with the card added for every row. The owner ruled the card record for
the qwen rows on 2026-09-22 (recorded in the private source repository at commit
`87dedcdc14d9a029ee088489abc835d24977fffc`) and for every row on 2026-09-28. For the 25 Release
0.1.1 rows, the card is read from each job's `00-job-start.txt`; recording it changes no value
(Section 10).

**No tolerance.** No numeric tolerance admits or refuses a row. A tolerance could only be checked
against a second run of the same System on another card or release, which is the re-running this
amendment exists to avoid, and any value chosen now would be chosen knowing the measured spread.
Card and lock variation is carried by the environment record and disclosed (Section 11), not
absorbed into a threshold.

**The measured spread is disclosed, not corrected.** A row's value is one measurement on one
environment. Where a System is known to be card-sensitive, the paper states the measured spread
beside the row and does not adjust the value.

## 6. Comparison across releases

- Every row admitted here may be reported beside Release 0.1.1 rows of its stratum.
- **Two rankings with different System ids** enter Default-Track Compare and the significance
  machinery of v1 Section 6 when both are paper rows of the same stratum, their budget K matches
  as v1 Section 2 requires, and the comparison is registered before it is computed:
  - for a new run, in the declaration of the row admitted here before that row's run starts; when
    both rows are admitted here, in the declaration of the row whose run starts later;
  - for an existing run (Section 8), by rule rather than by choice: it enters every comparison with
    every other paper row of its stratum that has a different System id and the same budget K, and
    its admission record lists them all before its re-score is computed.

  Their Package Artifact IDs and Dependency Lock digests need not match. Scoring equality is
  established by Section 3 condition 2, and evaluation equality by Section 3 condition 1. A paired
  verdict compares the two Systems as each was executed, each in its own release and environment,
  and is reported with both environment records.
- **The hardware spread beside every paired comparison.** Every paired comparison that includes a
  row admitted here is shown with, beside its verdict: the card of each of the two rows; and for
  each of the two Systems, its measured spread across cards on that stratum and metric, or "not
  measured" where no measurement of that System on another card exists. The spread is shown, not
  applied: it does not adjust, gate or reverse a verdict, and a missing measurement does not block
  the comparison.
- **Two rankings with the same System id** from different releases are never paired, differenced,
  pooled, or presented as replicates of one another. v8 stands for them unchanged.
- No new comparison family is registered by this amendment. v1 Section 6.5 applies: every
  registered test is reported, and each family names its full test count.

## 7. The row declaration

Before the run of a new row starts, a JSON object named by its run id is committed to the private
source repository from which releases are generated. The commit that adds it is the declaration's
anchor, and the run may not start before that commit exists. A later declaration for the same run id
is never written. An existing run admitted under Section 8 has no declaration; its admission record
takes that place.

| Field | Value |
| --- | --- |
| `artifact_type` | `taim-paper-row-declaration` |
| `schema_version` | `1.0` |
| `protocol_id`, `protocol_sha256` | this amendment |
| `run_id` | the run id the row will be given |
| `system_id` | the System, as the release's support matrix names it |
| `release` | release version, release manifest ID, public tree ID, package artifact ID, dependency lock digest |
| `stratum` | Track, Task, Profile, and the Section 4 identities the run must record |
| `budget_k`, `metric_cutoff` | as the run will be given them |
| `environment` | the intended card class, or CPU |
| `comparisons` | every paired comparison this row enters under Section 6: the other row's run id and the metrics, per v1 Section 7 |
| `ordinal` | `1`, or a higher number only under Section 3 condition 7, naming the earlier runs and their recorded failures |
| `declared_by`, `declared_at` | who committed it, and when |

## 8. Freeze, new runs and existing runs

**The freeze.** This amendment is frozen by the repository owner, after v11. The freeze date is the
approval date and is recorded in the protocol ID. It is frozen before the run of the next row of a
System that has no paper row yet starts.

**New runs.** A run that starts after the freeze commit is admitted only under Section 3, with
conditions 5 to 7. An undeclared run that starts after the freeze is never admitted.

**Existing runs.** A run that started before the freeze commit is admitted with its timing
disclosed, and is not run again, when all of the following hold:

1. **Section 3 conditions 1, 2, 4 and 8.** Its identities match its stratum's, or, where its
   recorded identities differ, it meets item 5. Its existing ranking is re-scored by the Release
   0.1.1 evaluator against the stratum's Evaluation Package, and the paper reports the re-scored
   values.
2. **Release, or the exact commit.** It executed a release as Section 3 condition 3 requires, or,
   where it executed no published release, its exact source is recorded: the commit of this
   repository it ran from, reachable from a branch or tag of the private repository and kept
   there; the System's own source commit and tree; and every dependency lock digest it used. A run
   of a withdrawn release, or of a release earlier than 0.1.1, is not admitted (v9 Section 5
   rule 3).
3. **Not adjusted.** Nothing about the run was adjusted after any of its results was seen. A run is
   adjusted when any of the following was changed, chosen or re-done after any score of that run
   was seen:
   - its configuration or System options, including budget K and any truncation of its ranking;
   - its candidate set: the pool, the topics or the Snapshot it was given, including any topic or
     trial kept, dropped or re-run;
   - its prompts or templates;
   - its code: the commit of this repository, its driver or job script, the System's own source;
   - its data: the prepared package, indexes, concept stores, caches, model weights, revisions or
     adapters;
   - which of several runs is put forward. When more than one completed run of the System on the
     stratum exists, the earliest-started completed run is the one considered, and every other run
     is disclosed.

   A score is any effectiveness value computed from the run's ranking against judgments: under any
   Profile or Evaluation Package, for all topics or some, while the run was still going or after it
   finished. A score of an earlier run that this run repeats or extends, such as a probe on some of
   its topics, counts as a score of this run. A change made before any score was seen, such as one
   made because of a probe's timing or operational failure, is not an adjustment; it is listed in
   the record.
4. **The no-adjustment record.** Before the run is re-scored under this amendment, a JSON object
   named by its run id is committed to the private source repository from which releases are
   generated. It states: the run id, System id and stratum; when the run started and finished; its
   source (item 2); when its ranking was fixed and the evidence for it; the first time any score of
   it was computed and seen, and what that score was; every earlier or later run of the System on
   the stratum, including probes and failed jobs; every comparison it enters (Section 6); for a run
   under item 5, the content-equality check's commit and its result, field by field; and that
   nothing listed in item 3 was changed after any score of the run was seen, with the evidence for
   each item. A run with no such record is not admitted. A run whose record cannot say this
   truthfully is not admitted, and may be run again only as a new, declared run.
5. **Content equality, where the recorded identities differ.** An existing run can record
   identities that differ from its stratum's: a different Profile, Prepared Snapshot ID, Task Input
   ID, Evaluation Package ID or pool receipt. Such a run is treated as meeting Section 3
   condition 1 only when a content-equality check shows that each of the following equals the
   stratum's:
   - **patient texts:** every patient record its Task Input gives a System, patient for patient
     and in the same order;
   - **trial texts:** every trial record its Task Input gives a System, trial for trial and in the
     same order. The comparison covers the whole record: source identity, canonical text,
     sections, typed core, extensions, other fields and derived evidence;
   - **the clinical as-of date;**
   - **the patient evidence profile;**
   - **the judgments** of its Evaluation Package, with their Judgment Scheme;
   - **the rest of its Task Input:** contract version, task, lineage, available capabilities and
     Derived Views.

   The Prepared Snapshot ID that its Task Input names is the one field allowed to differ.

   The check derives both sides' Task Inputs and Evaluation Packages with one codebase, the way a
   benchmark run builds them, and stops before any System call. It holds only if that derivation
   exactly reproduces two sets of identities:
   - the Prepared Snapshot, Task Input and Evaluation Package IDs the run recorded;
   - the stratum's values in Section 4.

   A derivation that does not reproduce a recorded identity cannot show what the run was given.

   The check reads the prepared packages. It reads judgments only to compare them. It reads no
   ranking and computes no score.

   The check is defined as a script at a named commit of the private source repository. It is
   committed before it runs, and it runs once. It does not rewrite, relabel or re-derive any
   identity the run recorded: the run keeps its recorded identities, and the paper reports them
   beside the stratum's.

   A run that fails the check is not admitted, whether a field differs or a derivation misses a
   recorded identity.

   This relaxes, for a run that passes, the stop rule of v4 Section 5: "Do not repair identity
   failures after inspecting scores". The relaxation was decided after a score of an existing run
   had been read (Section 2). The paper says so (Section 11, item 7).

A ranking that fails any of these remains historical evidence. Its re-score may be reported beside
the rows of its stratum, labelled as not a paper row, and it enters no comparison.

## 9. Existing runs known when this was written

This section applies Section 8 to the runs known on 2026-09-28. It does not change the rule.

- **Runs that Section 3 condition 8 or Section 8 item 2 already excludes.** The Release 0.1.2
  reference-score runs and the rows of the sibling public 0.2.x line are runs of Systems that
  already have a paper row on each stratum they cover, and are not admitted there again. The
  Release 0.1.0 TrialMatchAI `-v3` run executed a withdrawn release and is not admitted.
- **TrialMatchAI v0.9.1, Delta job 22464546.**
  - *Timing.* The job started at 2026-09-27T12:14:45Z and its System ran from 12:19:51Z to
    18:56:11Z. It predates the freeze, was declared nowhere, and is admitted, if at all, under
    Section 8 and disclosed as such.
  - *System.* TrialMatchAI v0.9.1 as its authors ship it: `github.com/cbib/TrialMatchAI` commit
    `3bc0e5f44d742ef069f8ef73bbc4187573d9db6a`, tree `e02e659c45513387e3b235e4c1bdafadea16e701`,
    recorded clean; configuration
    `sha256:3d6a1540f7a1ab0bcaa1e77bcbc23030fa4bcab7f12dcbe356741275cd7b571b`;
    driver `sha256:a33bb10b554ef5fc374086823d87626dc2e73dc7afd7d7ed81663b5026f25872`.
  - *Source.* It executed no published release. It ran from commit
    `c0d2430d6ec362cdac13df548e3bbedbc317898b` of the private source repository from which
    releases are generated, on a development branch, with the job script of that commit,
    SHA-256 `sha256:9c7e418d92c5f9783b5d3676a4cbc8f0f3d5a377121f38bc779106a1b49ba31b`, submitted
    with its placeholders filled as
    `sha256:516d008088775ef7f9f9d450b5a7b771bceab0cffa9b2400c01096fb011098b2`. When this was
    written, that branch existed in one local clone only; Section 8 item 2 requires the commit
    to be kept on a branch or tag of the private repository before admission.
  - *Card.* NVIDIA H200, compute capability 9.0, 143,771 MiB, driver 595.71.05, from
    `nvidia-smi.csv` written at 12:14:50Z, before the run.
  - *Ranking fixed before any score.* Budget K 60 is set in the submitted job script. The run record
    holding the 60-deep ranking was written at 18:56Z. The first score of it was computed afterwards
    by a scoring script committed at 19:26Z, in commit
    `cfa46aab4a8aa715d9506f4b0180780cf372e458`, and read on 2026-09-27: nDCG@10 0.557 under
    `official-full`.
  - *Other runs.* Probe 22463311, one topic, refused at staging; probe 22463875, one topic,
    completed, measured throughput and was not scored. No other run of the full row exists.
  - *Re-score.* Its existing ranking is re-scored by the Release 0.1.1 evaluator against the
    stratum's Evaluation Package (Section 3 condition 2). No run is repeated. The re-score is
    computed only after the content-equality check below has run. It is reported whether or not the
    run is admitted.
  - *Identity.* The run records:
    - Profile `official-full`;
    - Prepared Snapshot `sha256:d1b2258ced57c193e5ae6b9f6d29c393f6179be5b7b1a8a07370f15ca1518ac1`;
    - Task Input `sha256:2d102c4d0d502d1fcad2ce39be66fd751b6d83a77dea26ceb7280a68dcf4cfb5`;
    - Evaluation Package `sha256:42c67370c3908988a86a7d34676fab39f88b05fd3b94fd4b6abec42a6fa47c9f`.

    The external-fidelity stratum records `trec-ct-2021-external-fidelity-26149`, `5ddeded8…`,
    `019f47c8…` and `d35d98a8…`.

    Several things are the same as the stratum's:
    - its pool: 26,149 trial IDs in the same order, pool IDs SHA-256 `fed85fad…`;
    - its 75 topics;
    - its clinical as-of date, 2021-04-27;
    - the qrels file its judgments were parsed from, `sha256:ba7a2cdd…`.

    Its recorded identities still differ. As recorded, it meets Section 3 condition 1 on no
    stratum, so Section 8 item 5 applies. On 2026-09-28 the owner chose to test it by content
    equality against the external-fidelity stratum, the stratum whose pool, topics, date and qrels
    file it matches.

    The check is `check_content_equality.py` at commit
    `7b33fc39cfde892d4645de2ecae14ca96d051edf` of the private source repository. It is committed
    before it runs, and it runs once: the run's Prepared Snapshot against the stratum's.

    Its result, field by field, is disclosed with the run's row or with its re-score (Section 11,
    item 7). If the run is admitted, its no-adjustment record also cites that result
    (Section 8 item 4). This section does not record the result, so it reads the same whether this
    amendment is frozen before or after the check runs.
    - If it passes, the run is admitted on the external-fidelity stratum with its recorded
      identities unchanged. The paper discloses that v4 Section 5 was relaxed after its score was
      read.
    - If it fails, on any field or on a recorded identity, the run is not admitted. Its re-score
      is reported beside the stratum's rows, labelled as not a paper row.

    Either way, no identity it recorded is rewritten.
  - *Comparisons: none.* Its budget K is 60. No Release 0.1.1 row on the external-fidelity stratum
    has that K: the four retrieval rows carry 1,000 and the TrialMatchAI v4 row carries 20.

    It is paired with none of them. Truncating its ranking to 20 now would be an adjustment made
    after its score was read (Section 8 item 3), so it is not truncated.

    Its row carries this statement: "Not paired. It ran at K = 60, while the stratum's rows carry
    K = 1,000 and K = 20. Truncating it now would be an adjustment made after its score was read."

## 10. The Release 0.1.1 rows

Every row admitted under v9 Section 5 stays admitted, with its values, identities and release
unchanged. This amendment does not re-admit, re-score for the paper, re-cut, relabel or remove any
of them. They are the reference rows whose identities Section 4 lists. It adds one thing to their
record, the card of Section 5, read from their jobs' start records: an NVIDIA A40, driver 595.71.05,
for all 25. That changes no value. Otherwise this amendment only admits more rows.

## 11. Disclosure

Every table holding rows from more than one release, and the paper, discloses:

1. each row's release version and release manifest ID, with its package artifact ID and dependency
   lock digest in the supplementary record; for an existing run from a recorded commit, that commit,
   the System's source commit and the lock digests, and that it executed no published release;
2. that the rows come from different releases; that their Prepared Snapshot, Task Input, Evaluation
   Package, Profile definition and pool receipt are identical, except for a row admitted under
   Section 8 item 5, whose recorded identities are shown beside the stratum's with the statement
   that its inputs were shown equal in content by that item's check; and that every row was scored
   by the Release 0.1.1 evaluator (Section 3, condition 2);
3. what that equality does not cover: the System's code, model weights, dependency lock, hardware,
   caches, provider responses and non-determinism, each recorded per row (Section 5);
4. each row's card, or CPU platform, or that its card was not recorded;
5. for a card-sensitive System, the measured spread: for the Qwen3-Embedding-0.6B dense row on the
   SIGIR strata, nDCG@10 moved by 0.0012 to 0.0046 across A40, A100 and H200 cards, nDCG at any
   cutoff by up to 0.0062, and single recall cells by up to 0.018, while the BM25 and BGE-M3 dense
   rows were bit-identical across those cards and the reciprocal-rank-fusion row differed only at
   one SIGIR description cutoff-20 value, by 8.5e-6;
6. that a paired verdict between rows of different releases compares the Systems as executed in
   their releases and environments, with the two cards and each System's measured spread across
   cards, or "not measured", beside it (Section 6); and, on the row of an admitted row that enters
   no paired comparison, the reason;
7. this amendment by protocol ID, freeze date and SHA-256; each admitted row's run start relative
   to the freeze; and, for every row whose run started before the freeze:
   - that it predates the freeze, was not declared before it ran, and was admitted under Section 8
     without being run again;
   - its no-adjustment record, by commit;
   - every score of it that was seen before its admission, with the Profile, the value and the
     date;
   - that the existing-run rule was ruled after a score of an existing run had been read;
   - for a row admitted under Section 8 item 5: its recorded identities beside the stratum's; the
     content-equality check's commit and its result, field by field; that the clause was written
     after a score of that run had been read; and that v4 Section 5's stop rule "Do not repair
     identity failures after inspecting scores" was relaxed for it;

   and, for any admitted System, whether a measurement of it existed before the freeze and was read;
8. every started run of a declared row and every operational-failure replacement, and, for an
   existing run, every other run of its System on its stratum, including probes and failed jobs;
9. the `protocol_approval_id` each row's release recorded, which the release's code fixes
   (Release 0.1.1 records the v3 digest `sha256:18a9c9bd…` for judgment-union Profiles and the v4
   digest `sha256:55bb91f6…` for the external-fidelity Profile), and that this amendment governs
   the row's admission.

## 12. What this authorizes

This amendment defines when a row from a later release, or a run that already exists, is admitted
and how it is compared and disclosed. It authorizes no run, no release, no System registration, no
protected-data access, no provider call and no result admission. Each new row still requires the
owner's decision to run, each existing run's admission still requires its re-score and its
no-adjustment record, and, where its recorded identities differ, its content-equality check, and
provider-backed Systems remain under their own authorization.

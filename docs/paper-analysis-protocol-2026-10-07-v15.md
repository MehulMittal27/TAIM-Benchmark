# TAIM paper analysis protocol: complete eligibility text for the TrialGPT rows

**Protocol ID:** `taim-paper-analysis-protocol-2026-10-07-v15`

**Status:** Frozen on 2026-10-07, the date in its protocol ID, as the repository owner ruled.
Written on 2026-10-07 for the owner's review, and frozen the same day with the owner's answers to
that draft applied (Section 11). These bytes are immutable; a later change requires a new dated
amendment. It is frozen after v14, and before any query generation, retrieval freeze or System call
is made for any row it governs (Section 11). It ships with the release that first ships
`TrialGPT2-TAIM-Luna-v1`, as v14 does, because a reader of that release's TrialGPT rows needs the
identities it binds. It authorizes no run, no release, no provider call, no protected-data access
beyond what v14 Section 8 already authorizes, and no result admission.

No effectiveness value prompted this amendment. It was prompted by a capability that every stratum
lacks. Both TrialGPT Systems read each pool trial's complete eligibility text, and
`TrialGPT-TAIM-Luna-v1` also reads that text split into criteria. No Prepared Snapshot of protocol
v12 Section 4 carries either, so both Systems refuse the Task Input of each of their four targets.
This amendment gives the rows of both Systems, on those four targets, a profile-sized copy of each
stratum that carries the complete text and its criteria and nothing else new, and binds its four
new identities in place of the stratum's.

## 1. What this amendment changes

For `TrialGPT-TAIM-Luna-v1` and `TrialGPT2-TAIM-Luna-v1`, on the four targets of protocol v5
Section 1, it changes these clauses and nothing else:

- **Protocol v12 Section 3 condition 1, with Section 4.** A row of either System records the
  identities of Section 4 of this amendment: the Prepared Snapshot ID, Task Input ID, Evaluation
  Package ID and pool receipt ID of the target's complete-eligibility package (Section 3), and v12
  Section 4's Profile id, Profile definition SHA-256, pool count, pool IDs SHA-256 and clinical
  as-of date, which do not change.
- **Protocol v14 Section 5.** The target's one generation, one receipt and one freeze are made on
  the complete-eligibility package. The receipt's `prepared_snapshot_id` is that package's, and
  both Systems' rows on the target consume that one retrieval, as v14 Section 5 requires.
- **Protocol v14 Section 9.** Each target's order of work begins with a step 0: the package is built
  and its identities are checked against Section 4 (Section 9).
- **Protocol v12 Section 7.** In each row's declaration, `stratum` carries the identities of
  Section 4 of this amendment.
- **Protocol v12 Section 11 item 2.** For these rows, the disclosure that a row's Prepared
  Snapshot, Task Input, Evaluation Package and pool receipt are identical to its stratum's other
  rows reads as Section 10 item 3 below.
- **Protocol v12 Section 6, for `TrialGPT-TAIM-Luna-v1`.** Its rows keep v12 Section 6's paired
  comparisons, with the evaluation equality of Section 6 of this amendment in place of v12
  Section 3 condition 1, and every table holding one carries the caption note of Section 6.

Every other rule stands unchanged:
- v14, all of it except as adapted above: one generation, one receipt and one freeze per target
  for both Systems (Section 5), the qualification gates once on TREC 2021 (Section 6), the
  comparability of `TrialGPT2-TAIM-Luna-v1` (Section 7), the extension of protected TREC 2022
  access (Section 8), and the order of work (Section 9);
- v12, all of it except as adapted above, including Section 3 condition 2: each row is re-scored
  against the stratum's Evaluation Package by the Release 0.1.1 evaluator (Section 5 of this
  amendment says why that is unchanged);
- v13; v11, including its Section 5 (one generation per target); v9 Section 4; v5; v4; and v1,
  with Section 8 of v1 remaining the sole canonical statement of the recall-floor asymmetry.

The two System ids do not change. Nothing in either System's method, prompts, model, retrieval or
budget changes.

## 2. Why

- **The Systems require it.** `TrialGPT-TAIM-Luna-v1` has required `complete_eligibility_text`
  and the criterion Derived View `derived_view:eligibility_criteria` since 2026-08-12 (commit
  `2173e401` of the private source repository from which releases are generated), and Releases
  0.1.0, 0.1.1 and 0.1.2 publish it with that requirement. `TrialGPT2-TAIM-Luna-v1` requires
  `complete_eligibility_text`. `TrialGPT-TAIM-Luna-v1` splits each eligibility text into
  criteria, which reads its line structure; `TrialGPT2-TAIM-Luna-v1` gives it to TrialGPT 2.0's
  scorer as the trial's raw criteria, the form that scorer's own trial files carry.
- **No stratum carries it.** A stratum's trial record keeps each eligibility text twice: once
  whitespace-collapsed, as the section text, and once as source provenance, which a prepared
  package bounds at 4,096 bytes for long values. Neither keeps the line structure of a text longer
  than that. None of the four Prepared Snapshots of v12 Section 4 declares either capability, so
  both Systems refuse their Task Inputs. A stratum record carries only a cut excerpt of the
  eligibility text for 3,748 of the 26,145 TREC 2021 external-fidelity pool trials with eligibility
  text (14.3%), 3,671 of 26,582 for TREC 2022, and 275 of 3,625 for each SIGIR view (Section 3).
- **The refusal came too late.** Before this amendment, query generation, the receipt, the freeze,
  the recomputation and `TrialGPT-TAIM-Luna-v1`'s preflight all accepted a stratum's package; the
  first refusal was each System's own capability check, at the qualification gates or the first
  row. By then the target's one generation (v11 Section 5, v14 Section 5) would have been spent and
  its receipt bound to the stratum's Prepared Snapshot. The release that executes these rows now
  refuses a package without the two capabilities at query generation, at the freeze and at
  preflight, before any provider call or retrieval.
- **A new package is the smallest change that keeps every other rule.** Each alternative is worse:
  - dropping the requirement would change both Systems' input and method, and the repository
    owner's text of 2026-09-03 (Section 8) already rejected it: "Weakening the System requirement
    or substituting canonical text would change the frozen method and is prohibited";
  - re-preparing the strata themselves with the capability would move the identities of every
    Release 0.1.1 row;
  - the private TREC 2021 complete-eligibility builder covers one Track, selects its own topics and
    reads judgments itself, and never checks its records against the stratum's.

## 3. The complete-eligibility package

For each target, the **complete-eligibility package** is the prepared package that the release's
`scripts/prepare_trialgpt_profile_snapshot.py` builds (module `taim.data.complete_eligibility`),
from two inputs: the stratum's prepared package, whose Prepared Snapshot is v12 Section 4's, and
the Track's acquired Source Bundle. The build:

1. verifies every Source Bundle file against the lock the stratum's package records;
2. resolves the target's Profile on the stratum's package, pinned to v12 Section 4's pool count,
   pool IDs SHA-256 and pool receipt ID;
3. re-reads each pool trial from the verified trial archives, keeping its complete eligibility
   text;
4. requires each re-read trial, once the three complete-text fields are removed, to equal the
   stratum's record exactly. A trial that differs, a pool trial the archives do not hold, or a file
   that does not verify stops the build, and nothing is written;
5. splits each complete eligibility text into criteria with the criterion splitter
   `taim-eligibility-criteria-split-v3`, as the criterion Derived View;
6. writes the stratum's topics, the pool trials in the stratum's order with their complete text,
   the criterion Derived View, and the stratum's Profile judgments, as a new Prepared Snapshot and
   Evaluation Package. It declares the stratum's capabilities and the two above.

The package's Snapshot Source Recipe is `<dataset>-profile-complete-eligibility-v1`. Its SHA-256
binds the stratum it was built from: its Prepared Snapshot ID, Profile, pool and pool receipt. It
does not bind the stratum package's preparation time, so anyone who prepares the stratum's package
with the release builds the same package. The release accepts a package of this recipe for these
four Profiles only at the recipe SHA-256 of Section 4.

**What stays the same and what changes.** The owner asked whether the topics, judgments and pool
stay identical and only the trial text representation changes. **Yes.** Measured on 2026-10-07 on
all four targets, from the four stratum packages of v12 Section 4 (Section 4):

| | TREC 2021 external-fidelity | TREC 2022 | SIGIR description | SIGIR summary |
| --- | --- | --- | --- | --- |
| Topics (patient texts, in order) | equal | equal | equal | equal |
| Judgments, count and content, Judgment Scheme | equal (35,811) | equal (35,394) | equal (3,870) | equal (3,870) |
| Pool: count, trial IDs and order | equal (26,149) | equal (26,585) | equal (3,626) | equal (3,626) |
| Profile, Profile definition, clinical as-of, patient evidence profile | equal | equal | equal | equal |
| Trial records with the complete text removed | 26,149 of 26,149 equal | 26,585 of 26,585 equal | 3,626 of 3,626 equal | 3,626 of 3,626 equal |
| Eligibility sections whose stratum text is the complete text, whitespace collapsed | 26,145 of 26,145 | 26,582 of 26,582 | 3,625 of 3,625 | 3,625 of 3,625 |
| Stratum records whose eligibility provenance is cut at 4,096 bytes | 3,748 | 3,671 | 275 | 275 |

So each trial gains the line-preserving copy of an eligibility text whose words its stratum record
already carries, and the criteria split from it. No trial, topic, judgment or pool member is added,
removed or changed, and nothing outside the eligibility text is added.

## 4. The identities

These replace v12 Section 4's Prepared Snapshot, Task Input, Evaluation Package and pool receipt IDs
for the rows of both Systems on these targets. The other columns are v12 Section 4's, unchanged.

| Target | Profile | Prepared Snapshot ID | Task Input ID | Evaluation Package ID | Pool count | Pool IDs SHA-256 | Pool receipt ID | Clinical as-of |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| TREC 2021 external-fidelity | `trec-ct-2021-external-fidelity-26149` | `sha256:db95af606db708b463dc8bf14515e3662b655de6105dd0f31f28fc10e46b6905` | `sha256:f426d027a92227ec3eea40e5b9ad8ed48422a0c82fbf75147615b31262b340a8` | `sha256:35fa37ea430b7255d904152dc967326fdcf30b27cd5ef93c02109de0b82cb1db` | 26,149 | `sha256:fed85fadb5a0e0a42a39ed9cf65984926a13f66d4a2a0a1deb100996ac56ffa7` | `sha256:1a3432e205252e689ff481914be53f0ed0d78e134b9cf4d06876744dd8fe7cb1` | 2021-04-27 |
| TREC 2022 judgment-union | `trec-ct-2022-judgment-union` | `sha256:d95c9e284cb60f9e9ce2d3eb95d377e35a61217656c95c784213699155cd961f` | `sha256:73e9378c72da535dfa4cf984f7a8f876d5f59a90ca4d965370e39b2723ded536` | `sha256:a6884d8343b34a7cdb76f90da611319abc032f2e5001c0757bc8c3f25a7f2f31` | 26,585 | `sha256:c56129596aac032d62644fc851d61af95da77f470a164650d0a4b953f261c553` | `sha256:e65da82ad2178a74e1a4998b63876db95f306bc78c63f783aa8332df8f7d5c6a` | 2022-01-01 |
| SIGIR 2016 summary | `sigir-ct-2016-summary-judgment-union` | `sha256:de939758bf653505ef83b4efbe9c0a437e9f83a9706238dd4bcb798b8799db70` | `sha256:546348a7b4700cdf26dba1eff17dc6b5f398d0f086665b9617252601f1244a79` | `sha256:6a6e7084b9b16de4d45c6bf8b9d2982698084056d07932e9568ecc49869da8ca` | 3,626 | `sha256:9a106146c9e1a5da49e15cda3dcc44add3c028e6b0d26a55a40bf60813f5ec5a` | `sha256:340cc0636ac0212bfc1ae3e3a7829070ca8ab59e84ba3e37352f98cbc86cfe64` | 2015-12-16 |
| SIGIR 2016 description | `sigir-ct-2016-description-judgment-union` | `sha256:345813701966b573f0c30b89d8110684199ac92b0af90553f2af75c53b48d20c` | `sha256:30f9353ef7d1155fe24f3f1fedb876d968392607f7e1d5273b0de43c2e3b6c4d` | `sha256:36899499f7218e005db9f1f6aa808d6ee34049bea00dc33e0e1404421156e095` | 3,626 | `sha256:9a106146c9e1a5da49e15cda3dcc44add3c028e6b0d26a55a40bf60813f5ec5a` | `sha256:9cba221f7271ffff938d70ffd3b44313f2485e86c5703fa4458807c53bb02a69` | 2015-12-16 |

The Profile definition SHA-256 of each target is v12 Section 4's. The packages' Snapshot Source
Recipe SHA-256s, which the release pins, are: TREC 2021
`sha256:01c577d445c541f03eff7ce6d49087be86346f2d9fafe20aaa931007831936c0`, TREC 2022
`sha256:c468546653876399cb7544e23be75a9ded0f35565ed1fa5fca428b5fdc9d1610`, SIGIR summary
`sha256:76ffcaba7253fe5cc6f8811284390e9e360ffa7852f72e7b61380790f8767253`, SIGIR description
`sha256:0efd6c78cf96ca7c4db9d68626c4604bea6a8af7290cabdf51a55ae458e4a082`.

**How they were computed.** On 2026-10-07, the candidate code of the TrialGPT 2.0 release built each
package from the stratum package of v12 Section 4 and the Track's locked Source Bundle, and derived
every identity above, at commit `126dcc13` of the private source repository from which releases are
generated; the SIGIR and TREC 2022 packages were built earlier that day with byte-identical builder
code. The same code re-derived each stratum's own eight values of v12 Section 4 from the same
stratum packages, and all thirty-two equal v12 Section 4. The derivation read judgments only inside
the code, to project them into the package and to hash them for the comparison of Section 3; it read
no ranking and computed no score.

**Reproduced by the release.** Before the TrialGPT 2.0 release is cut, its wheel rebuilds all four
packages from the stratum packages and reproduces every identity above, and its support accepts
each package only at the recipe SHA-256 above. A release that does not reproduce them is not cut
with this amendment.

## 5. Scoring

v12 Section 3 condition 2 stands as written: each row is re-scored against the stratum's
Evaluation Package by the Release 0.1.1 evaluator. That evaluator reads only an Evaluation
Package's judgments and Judgment Scheme, and the complete-eligibility package's are equal to the
stratum's (Section 3), so the row's values are the same against either package's Evaluation
Package.

## 6. Comparisons

The owner asked: if pairing is honest, keep it; if not, the same stratum table with no paired
comparison and a caption note, as for TrialGPT 2.0.

**Pairing stays honest, so `TrialGPT-TAIM-Luna-v1` keeps v12 Section 6**, as the owner ruled
(Section 11). A paired comparison is computed topic by topic over the same topics, against the same
judgments, within the same pool, under the same Profile, by the same evaluator. Section 3 shows each
of these equal to the stratum's. What differs is what one System is given about each trial: the
line-preserving copy of an eligibility text whose words the stratum's records already carry, and the
criteria split from it. That is part of how `TrialGPT-TAIM-Luna-v1` reads a trial, as a System's own
preprocessing is. It adds no fact about a patient, no judgment and no information about any other
trial. v13 already judges equality by what a System reads; here every other row's System reads a
stratum record that the package carries unchanged.

The reason `TrialGPT2-TAIM-Luna-v1` enters no paired comparison is different and stands: v14
Section 7 excludes it for its model condition through Codex, not for its input. This amendment
does not change that.

So, for `TrialGPT-TAIM-Luna-v1` on these targets, v12 Section 6 applies with this amendment's
evaluation equality in place of v12 Section 3 condition 1: two rankings with different System ids
enter Default-Track Compare when both are paper rows of the stratum, their budget K matches, and the
comparison is registered before it is computed. As v12 Section 6 already requires, no difference is
computed between rows whose budget K differs: among the rows that exist when this is written, only
the Release 0.1.1 TrialMatchAI row on the TREC 2021 external-fidelity stratum carries K = 20, as
`TrialGPT-TAIM-Luna-v1` does (v9 Section 4); the retrieval rows carry K = 1,000.

Every table holding a `TrialGPT-TAIM-Luna-v1` row on these targets carries, in its caption:
"`TrialGPT-TAIM-Luna-v1` reads each trial's complete eligibility text and its criteria, from a
copy of the stratum that differs from it only in that text (protocol v15); its topics, judgments
and pool are the stratum's."

## 7. One generation per target, on the package

v14 Section 5 is unchanged except where it binds. Each target's one generation reads the
complete-eligibility package's patient texts, which are the stratum's (Section 3), so its plans are
the plans of both Systems' rows as v14 Section 5 states. Its receipt records the package's Prepared
Snapshot ID, and the freeze, its lock, the recomputation and both Systems' rows bind the package's
Prepared Snapshot, Evaluation Package, pool receipt and Task Input (Section 4). A generation,
receipt or freeze made on a stratum's own package is not a generation of these rows; none has been
made (Section 11).

## 8. The 2026-09-03 text called "v9"

On 2026-09-03 the repository owner wrote "TAIM paper analysis protocol: TrialGPT capability closure
amendment", protocol ID `taim-paper-analysis-protocol-2026-09-03-v9`, SHA-256
`sha256:022b6ca1a374d07d612f22fbfe7bf6605a6f222f729dcc473e16e56297468237`, in commit `006dea5b`
of the private source repository, with a builder of this design. Its status line reads frozen. It
was never adopted into the chain of frozen documents: it is not in the rules archive, no release
carries it, and no later protocol cites it. Development moved to another line, and the commit is
not an ancestor of the line from which releases are now generated. A different document, frozen
on 2026-09-18, carries the same number (`taim-paper-analysis-protocol-2026-09-18-v9`).

This amendment adopts that text's design for these rows and supersedes it: a reader of the paper
cites this amendment, not the 2026-09-03 text. It differs in four points:
- that text kept the 735 query plans of protocol v6, which are not available; v11 and v14 govern
  query plans;
- that text's recipe bound the stratum package's manifest bytes, which include its preparation
  time; this amendment's binds the stratum by content (Section 3);
- that text named a new patch release of 0.1.0; this amendment ships with the TrialGPT 2.0 release,
  as v14 does;
- that text named only `TrialGPT-TAIM-Luna-v1`; this amendment also covers
  `TrialGPT2-TAIM-Luna-v1`.

## 9. Order of work

v14 Section 9 stands, with one step before its step 1 for each target:

0. build the target's complete-eligibility package with the TrialGPT 2.0 release, from the
   stratum's prepared package and the Track's Source Bundle, and check that the Prepared Snapshot
   ID, Evaluation Package ID, pool receipt ID and Task Input ID the command prints equal Section
   4's. A package that does not reproduce them is not used.

Steps 1 to 8 then use that package: the declarations name its identities, the generation reads its
manifest, and the freeze, preflight and rows take it as their prepared data. The TREC 2021
external-fidelity target still comes first.

## 10. Disclosure

Every reported row of either System on these targets, and the paper, discloses, beside v14
Section 10:

1. this amendment by protocol ID, freeze date and SHA-256, and that it was written before any
   generation of these rows;
2. that both Systems read each trial's complete eligibility text, which no stratum carried, and
   that the rows ran on each stratum's complete-eligibility package (Section 3);
3. in place of v12 Section 11 item 2's statement of identity: that the row's Prepared Snapshot,
   Task Input, Evaluation Package and pool receipt are those of Section 4, not the stratum's, and
   that its topics, judgments, pool, Profile and clinical as-of are the stratum's; with each
   stratum's count of pool trials whose eligibility text it carries only as a cut excerpt;
4. that the row was re-scored against the stratum's Evaluation Package (Section 5);
5. for `TrialGPT-TAIM-Luna-v1`, the caption of Section 6 and each paired comparison registered for
   it; for `TrialGPT2-TAIM-Luna-v1`, v14 Section 7 as before;
6. the 2026-09-03 text of Section 8, and that this amendment supersedes it.

## 11. What was known when this was written

- **No effectiveness value of either TrialGPT System existed**, and none informed this amendment.
  No generation under v11 or v14 had started for any target, and no receipt or row declaration had
  been committed. Since 2026-10-07 every TrialGPT generation has been on hold, by the owner's
  ruling below.
- **The gap was found on 2026-10-07**, while checking whether the SIGIR and TREC 2022 strata carry
  what `TrialGPT-TAIM-Luna-v1` requires. All four strata lack it, TREC 2021 included.
- **The values of Section 4 were computed** while this was written, by the code that builds them.
  No judgment row was opened outside that code, and no ranking or score of any System on any of
  these packages exists.
- **The owner's ruling of 2026-10-07** that started this amendment, verbatim:

  > Complete eligibility: option a - port Hannes's builder (006dea5b) into the public code, build
  > the four targets' prepared data, and draft v15 for my review binding the new IDs, before the
  > TrialGPT 2.0 release. Keep his capability checks in the preflight and freeze, so a gap refuses
  > before any spend. All TrialGPT generation stays on hold until then.
  >
  > On v15's question: say whether the topics, judgments and pool stay identical and only the
  > trial text representation changes. If pairing is honest, keep it; if not, same table with no
  > paired comparison and a caption note, as for TrialGPT 2.0.

**The owner's answers of 2026-10-07 to the draft**, as relayed with the approval:

> 1. Keep paired comparisons for TrialGPT 1.0, with the caption note.
> 2. The release rebuilds all four packages and reproduces their identities.
> 3. The public guide wording is approved: inspect the pool, build the complete-eligibility
>    version, generate, freeze, run.

## 12. What this authorizes

This amendment defines the prepared data and identities of both TrialGPT Systems' rows on their
four targets, how those rows are compared and what they disclose. It authorizes no run, no
release, no System registration, no provider call, no protected-data access beyond v14 Section 8,
and no result admission. Each generation, gate and row still requires the owner's decision to run.

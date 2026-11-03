# TAIM paper analysis protocol: new paper rows on every stratum

TrialGPT on TREC 2023, TrialMatchAI v0.9.1 on all six strata, and the TAIM staged System on all six
strata.

**Protocol ID:** `taim-paper-analysis-protocol-2026-11-02-v17`

**Status:** Frozen on 2026-11-02, the date in its protocol ID, on Mehul's ruling and Hannes Ill's
agreement (Section 11). Written 2026-10-27 from `origin/main` at `b783425e`,
`origin/evidence/trialgpt-paper-rows` at `f3152009`, and a full-history clone of upstream
`cbib/TrialMatchAI` (tag `v0.9.1` at `3bc0e5f4`), and edited after review on 2026-11-02. These
bytes are immutable, and a later change needs a new dated amendment.

It is frozen after v16. It is frozen before any of the following:

- any row of Section 3 is declared;
- any corpus of Section 6 is built, except the three built early on Mehul's decisions (TREC 2023
  and SIGIR from 2026-10-27, TREC 2021 JU from 2026-11-01), from trial texts only and with no
  ranking made (disclosed in Section 13);
- the TrialGPT TREC 2023 package is used for a generation;
- any TREC 2022 or TREC 2023 topic text is given to a System of Section 3.

It ships with the release that first ships the capabilities of Section 10. Its one authorization is
Section 11: Mehul's ruling and Hannes Ill's agreement, both quoted there verbatim. It authorizes no
run, no release, no provider call outside Section 11, and no result admission.

No effectiveness value of any row of Section 3 exists. The amendment was prompted by Mehul's
rules, not by any value:

- no dataset is left out of the paper;
- TrialMatchAI v0.9.1, the version the TrialMatchAI paper uses, runs on every stratum;
- the TAIM staged System, the paper's main claim, runs on every stratum;
- every new row reads its stratum's own Prepared Snapshot.

All of these are Mehul's instructions of 2026-10-27.

## 1. What this amendment changes

### 1.1 TrialGPT on TREC 2023 (two rows)

- **Protocol v4 Section 4, the sentence** "The release provides no TrialGPT TREC 2023 capability
  because the upstream work did not evaluate that Track and no TAIM adaptation has been qualified."
  It is replaced. The release of Section 10 provides that capability for `TrialGPT-TAIM-Luna-v1` and
  `TrialGPT2-TAIM-Luna-v1` on `trec-ct-2023-judgment-union`. The upstream work still did not
  evaluate that Track, so both rows are TAIM adaptations, not reproductions. Qualification is
  Section 9.
- **Protocol v4 Section 4, the table of added rows.** It gains two rows: TREC Clinical Trials 2023,
  `trec-ct-2023-judgment-union`, `TrialGPT-TAIM-Luna-v1`, provider-backed TAIM adaptation; and the
  same for `TrialGPT2-TAIM-Luna-v1`, comparable only to itself (v14 Section 7).
- **Protocol v5 Section 1,** "TREC 2023 has no TrialGPT row", and its restatements in **v6 Section 4**
  and **v14 Section 1** ("TREC 2023 has no TrialGPT row, for either System"). For these two rows they
  are replaced: the TrialGPT paper System runs on a fifth target, `trec-ct-2023-judgment-union`, 40
  topics, the score-blind union of judged trial IDs. v6 Section 2's four admitted query-plan files
  stay four: TREC 2023 never had one, and its plans are generated under v11 and v14 like the others.
- **Protocol v5 Section 5, the provider budget.** The fifth target adds a ceiling of:
  - 120 query-generation calls (40 topics x 3 arms), the target's one generation, for both Systems;
  - 60,000 logical System calls (40 topics x 500 candidates x 3 calls) for `TrialGPT-TAIM-Luna-v1`;
  - 20,000 calls (40 x 500) for `TrialGPT2-TAIM-Luna-v1`.

  In all 80,120. The calls run on Hannes Ill's own Codex account. Each ceiling bounds one complete
  pass. Calls spent by a run that an operational failure stopped are recorded and do not count
  against it.
- **Protocol v14 Sections 4, 5, 8 and 9.** The TREC 2023 rows of both Systems execute the release of
  Section 10 and are admitted under v12 Section 3, conditions 1 to 8. TREC 2023 has its own one
  generation, one receipt and one freeze on that release, serving both Systems on identical
  candidates. It follows steps 0 to 8 after the four existing targets. v14's words "What it does not
  cover: TREC 2023, for either System" no longer apply to these two rows.
- **Protocol v15 Sections 3, 4 and 9.** TREC 2023 gets a complete-eligibility package built exactly
  as v15 Section 3 says, with the identities of Section 5.2. v15 Section 9's step 0 gains, for TREC
  2023, the two checks of Section 4.1, which must pass before the declarations are committed.
- **Protocol v15 Section 6.** `TrialGPT-TAIM-Luna-v1`'s TREC 2023 row keeps v12 Section 6's paired
  comparisons with v15 Section 6's evaluation equality, and carries v15 Section 6's caption.
- **Protocol v16.** Applies unchanged: `TrialGPT-TAIM-Luna-v1` splits criteria with upstream
  TrialGPT's `parse_criteria` over the sections it cuts from the complete text (deviation D11).
- **Protocol v12 Section 7, the declaration.** Each TREC 2023 TrialGPT declaration is written by the
  release's TrialGPT row-declaration command, as on the other targets. It carries:
  - in `stratum`, the package identities of Section 5.2 (v15 Section 1);
  - a new field `source_stratum`: the stratum's own Prepared Snapshot ID, Evaluation Package ID and
    pool receipt ID (Section 5.1), which the package names as its source. It is the snapshot digest
    the same-snapshot rule names;
  - in `amendment`, v14 and its statement, as the code writes it on every target. v14 Section 5 is
    the rule under which each row consumes its target's one retrieval; this amendment extends it to
    TREC 2023 and is identified by the release that ships it.

### 1.2 TrialMatchAI v0.9.1 on six strata (six rows)

- **The admitted row stays as it is.** `trialmatchai-upstream-v091-trec21-row-measurement-22464546`
  (TREC 2021 external-fidelity, K = 60) stays exactly as v13 admitted it:
  - it is not run again, re-scored differently or truncated;
  - it is not paired with anything (its admission record, `comparisons.statement`).
- **Protocol v12 Section 3 condition 8** ("One row per System per stratum"). One exception:
  `trialmatchai-upstream-v091-cuda-trec21-development-v1` may have a second row on TREC 2021
  external-fidelity, at K = 20, declared under this amendment. Mehul approved this exception
  on 2026-10-27.
  - Under v12 Section 6 the K = 60 and K = 20 rows share a System id. So they are never paired,
    differenced, pooled or presented as replicates.
  - They are reported side by side, each labelled with its K. Only the K = 20 row enters
    Section 8.
- **Protocol v4 Section 4, the table of added rows.** It gains five v0.9.1 rows:
  - TREC 2021 judgment-union, under the existing System id;
  - TREC 2022, TREC 2023 and the two SIGIR strata, under the System ids of Section 3.
- **Mehul's ruling of 2026-09-26 on SIGIR**, as recorded in
  `docs/agents/trialmatchai-upstream-v091-run-commands.md:117-120`: "A SIGIR identity would use the
  **description** text. **Summary is declared absent**, because their code has no summary view."
  - For the SIGIR summary row it is replaced. TAIM gives the SIGIR summary text through the same
    route it uses for every Track: the driver's `build_profile_from_topic(_clean(text))`.
  - Their code still has no summary view. The summary row is TAIM's choice of input, and it is
    disclosed (Section 12).
  - Mehul approved this change on 2026-10-27, on one condition: upstream's code stays
    unchanged. TAIM supplies only the topic text, through the same importer call it makes for
    every other Track.
- **Trial text per stratum (Section 6).** TREC 2021 and TREC 2022 read upstream's published
  preparation. TREC 2023 and both SIGIR strata read text prepared by v0.9.1's own preparation code,
  because the tool that made the published preparation is no longer in upstream's code (Section
  6.3). Mehul confirmed this route on 2026-10-27. One measurement on TREC 2021
  external-fidelity, approved on the same day if an allocation allows it, measures what
  that difference does to a ranking (Section 6.4).
- **Three new System ids,** minted as named by Mehul on 2026-10-27:
  - `trialmatchai-upstream-v091-cuda-trec22-development-v1`;
  - `trialmatchai-upstream-v091-cuda-trec23-development-v1`;
  - `trialmatchai-upstream-v091-cuda-sigir16-development-v1`.
- **Nothing else about the System changes.** Each new System id is the TREC 2021 System with the
  tag's own track key and topic id prefix (`trec/corpus.py:17-26` at `3bc0e5f4`). The tag's
  `_track_config` sets the same funnel for every track (`trec/runner.py:31-53`). The pin, the
  concept store S2 and the model revisions stay those of the v0.9.1 System's adapter module.

### 1.3 The TAIM staged System on six strata (six rows)

- **Protocol v1 Section 2** ("TREC Clinical Trials 2021 ... The only authorized paper track") and
  **v1 Section 3** ("Single benchmark ... Generalisation is future work"). For
  `staged-bm25-qwen3-rrf-rerank` they are changed:
  - TREC 2022, TREC 2023 and both SIGIR strata become its held-out test strata;
  - the two TREC 2021 strata stay its development strata, and the paper says so.

  Mehul confirmed this change on 2026-10-27. Hannes Ill's agreement is quoted verbatim in Section 11
  at the freeze.
- **Protocol v3 Section 2 and v4 Section 1** ("one additional TREC 2021 primary experiment"). The
  System gets one row on each of the six strata. Its two component Systems stay supporting stages,
  not reportable columns.
- **Protocol v3 Section 4** ("The staged System uses only its declared TREC 2021 components"). On
  each stratum it uses that stratum's own components, from its own release.
- **Protocol v1 Section 5 item 1** ("Every reranked configuration against its own unreranked panel
  baseline"). It is registered as a paired comparison on each of the six strata (Section 8). The
  baseline is defined in Section 7.3, as Mehul approved on 2026-10-27.
- **One more registered pair per stratum.** By Mehul's decision of 2026-10-27, each staged row
  is also paired with its stratum's Release 0.1.1 BM25 reference row, at K = 1,000 (Section 8). The
  other three reference rows (BGE-M3, Qwen3 and RRF) are shown beside it, unpaired.
- **Protocol v1 Sections 6.1 and 6.6** ("Paired over the 75 topics"; "The topic set is fixed at
  75"). These read as the stratum's topics: 75, 75, 50, 40, 60 and 60.
- **The recipe does not change:**
  - component depth 5,000, fusion and scoring depth 2,000, output depth 1,000;
  - RRF constant 60;
  - `Qwen/Qwen3-Reranker-4B` at `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, float16, batch 16,
    maximum length 2,048;
  - F-summary with the folded query.

  These are the values of the staged System's declaration file. The declaration does not change
  (Section 10).

### 1.4 For every row of this amendment

- **The same-snapshot rule** of Section 4 applies, with its pre-run content check.
- **Protocol v12 Section 6, where a comparison is registered.** Some comparisons of Section 8 join
  two rows of this amendment. Each such comparison is listed in both rows' declarations, and the
  earlier of the two declaration commits precedes both runs. A comparison with an existing row is
  listed in the new row's declaration, as v12 Section 6 already says.
- **Protocol v1 Section 6.5** ("no multiplicity correction"). For the tests this amendment
  registers, it is replaced by Holm's procedure within each stratum (Section 8.3). Tests registered
  before this amendment keep v1 Section 6.5.

Every other rule stands unchanged. That covers v16, v15, v14 (including Section 7:
`TrialGPT2-TAIM-Luna-v1` is comparable only to itself), v13, v12 (including Section 3 condition 2, the
Release 0.1.1 evaluator), v11, v9, v5, v4 and v1. v1 Section 8 stays the sole statement of the
recall-floor asymmetry.

## 2. Why

- **No dataset left out.** TREC 2023 has only the four Release 0.1.1 reference rows. Before this
  amendment, TrialMatchAI v0.9.1 had a row on one stratum and the staged System on none.
- **TrialGPT's TREC 2023 exclusion.** v4 Section 4 gave two reasons. The first stays true and is
  disclosed: the upstream work did not evaluate TREC 2023. The second was overtaken: no TAIM
  adaptation had been qualified, but v14 Section 6 now qualifies both TrialGPT Systems once, on TREC
  2021, for all their targets.
- **v0.9.1 everywhere.** It is the version the TrialMatchAI paper uses. Its tag defines tracks 21,
  22, 23 and sigir itself (`trec/corpus.py:17`). Its own published evaluation covers TREC 2021 and
  2022 only (`docs/trec-evaluation.md`).
- **The staged System everywhere.** It is the paper's main claim. Its only figures so far are TREC
  2021 development data, which v1 Section 4.1 calls "registration data, not a paper result". It was
  built on TREC 2021, so the other four strata are its first held-out tests.
- **TrialMatchAI -v4 is out.** Mehul put the fork's TREC 2022, TREC 2023 and SIGIR rows on hold
  on 2026-10-27. Its TREC 2021 external-fidelity row is unchanged. Its TREC 2022 and TREC 2023 rows
  stay registered and not run (v4 Section 4, v9 Section 1); this amendment does not touch them.

## 3. The rows

| # | System | Stratum | K | Release | Runs on | Trial text the System reads |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | `TrialGPT-TAIM-Luna-v1` | TREC 2023 JU | 20 | Section 10 release | CPU, Hannes Ill's Codex account | TREC 2023 complete-eligibility copy (built by us) |
| 2 | `TrialGPT2-TAIM-Luna-v1` | TREC 2023 JU | 20 | Section 10 release | CPU, Hannes Ill's Codex account | the same copy |
| 3 | `trialmatchai-upstream-v091-cuda-trec21-development-v1` | TREC 2021 JU | 20 | Section 10 release | NVIDIA H200 | upstream's published TREC 2021 preparation, cut to the pool |
| 4 | `trialmatchai-upstream-v091-cuda-trec21-development-v1` | TREC 2021 EF | 20 | 0.2.0 | NVIDIA H200 | the same corpus as the admitted K = 60 row |
| 5 | `trialmatchai-upstream-v091-cuda-trec22-development-v1` | TREC 2022 JU | 20 | Section 10 release | NVIDIA H200 | upstream's preparation for 26,584 trials, ours for 1 |
| 6 | `trialmatchai-upstream-v091-cuda-trec23-development-v1` | TREC 2023 JU | 20 | Section 10 release | NVIDIA H200 | prepared by us with v0.9.1's own code, all 17,105 trials |
| 7 | `trialmatchai-upstream-v091-cuda-sigir16-development-v1` | SIGIR 2016 summary | 20 | Section 10 release | NVIDIA H200 | prepared by us with v0.9.1's own code, all 3,626 trials |
| 8 | the same | SIGIR 2016 description | 20 | Section 10 release | NVIDIA H200 | the same corpus as row 7 |
| 9 | `staged-bm25-qwen3-rrf-rerank` | TREC 2021 JU | 1,000 | 0.2.0 | NVIDIA A40 | the stratum's Prepared Snapshot |
| 10 | the same | TREC 2021 EF | 1,000 | 0.2.0 | NVIDIA A40 | the stratum's Prepared Snapshot |
| 11 | the same | TREC 2022 JU | 1,000 | Section 10 release | NVIDIA A40 | the stratum's Prepared Snapshot |
| 12 | the same | TREC 2023 JU | 1,000 | Section 10 release | NVIDIA A40 | the stratum's Prepared Snapshot |
| 13 | the same | SIGIR 2016 summary | 1,000 | Section 10 release | NVIDIA A40 | the stratum's Prepared Snapshot |
| 14 | the same | SIGIR 2016 description | 1,000 | Section 10 release | NVIDIA A40 | the stratum's Prepared Snapshot |

JU is judgment-union and EF is external-fidelity. The new System ids of rows 5 to 8 were minted as
named by Mehul on 2026-10-27. The measurement of Section 6.4 is not a row of this table.

Releases, as Mehul ruled on 2026-10-27:
- rows 4, 9 and 10 run on Release 0.2.0, whose support matrix already declares them;
- row 9 (TREC 2021 JU) is the first staged row to start;
- every other row runs on the Section 10 release.

Each row:
- is declared under v12 Section 7 before its run;
- runs to completion once (v12 Section 3 condition 7, as Section 10 step 7 widens it);
- is re-scored by the Release 0.1.1 evaluator against its stratum's Evaluation Package (v12 Section 3
  condition 2).

v12 Section 3 condition 8 holds for every row except row 4, which Section 1.2 states.

The card of the v0.9.1 rows follows the System's own requirement: "one CUDA card of about 64 GB or
more ... an A40 or a 40 GB A100 cannot load it" (the v0.9.1 external-baseline guide, line 23).

## 4. The same-snapshot rule and the pre-run content check

**The rule.** Every row of Section 3 reads its stratum's own Prepared Snapshot. That means the same
trial IDs, the same clinical as-of date, and the same Evaluation Package as the reference rows of
that stratum (Section 5).

- A row whose System reads the Prepared Snapshot directly records the Section 5 identities. These
  are rows 9 to 14, and the trial-ID side of rows 3 to 8.
- A row whose System reads trial text from a corpus we built must pass the content check below
  before its declaration is committed. These are rows 1, 2, 5 (its one built trial), 6, 7 and 8,
  and the measurement of Section 6.4.

**The content check.** It follows v13's check. It is run by a script committed before it runs, and
its result is committed before the row's declaration. A run of the check that stops for an
operational reason before it writes its result may be run again, and the stopped run is recorded.
For each corpus it checks:

1. **Trial ID set.** The corpus's trial IDs equal the stratum's pool exactly: same count, and the
   sorted-ID SHA-256 equals the Section 5 pool IDs SHA-256. For v0.9.1 this is also what
   the TrialMatchAI adapter's own snapshot-corpus check enforces at run start.
2. **Trial text traced to the snapshot.** Every trial the corpus holds as built by us records the
   SHA-256 of the source record it was converted from. The check recomputes each digest from the
   stratum's Prepared Snapshot source (the locked Source Bundle behind the Section 5 Prepared Snapshot
   ID) and requires equality for every such trial.
3. **What is not ours.** A trial taken from an upstream preparation is not traced to the snapshot.
   This covers the 26,584 TREC 2022 trials in upstream's published corpus, whose use Mehul
   approved on 2026-10-27. The check records each trial's upstream member digest instead, and the
   row lists it under Section 12.
4. **As-of and judgments.** The run's recorded clinical as-of date and Evaluation Package equal the
   Section 5 values, or for rows 1 and 2, the complete-eligibility identities of Section 5.2.

A corpus that fails any part is not used. It is not repaired after any System output on that
stratum exists. A failed check is disclosed, and a rebuilt corpus is checked again from the start.

### 4.1 Rows 1 and 2: checks A and B of the TrialGPT copy

Rows 1 and 2 read the TREC 2023 complete-eligibility copy, which we build (Section 5.2). Two checks
run on the copy, and their two records are committed before the declarations. Both are required.
**If either fails, the copy is not used, no TrialGPT TREC 2023 row is declared, and nothing after
step 0 runs.** A rebuilt copy is checked again from the start, and the failure is disclosed.

Both checks are one script, committed before it runs: `private/scripts/check_complete_eligibility_copy.py`
(commit `c29571b5` on the local branch `fm/taim-trialgpt-trec23-package`). It runs under the
interpreter of the Section 10 release. It reads trial texts and identities, prints identities,
digests and counts only, reads no ranking, computes no score and calls no provider.

**Check A, the copy against the stratum (`content`).** It is this section's content check for rows
1 and 2, and requires all of:

1. the stratum package resolves the Profile at Section 5.1's pool count, pool IDs SHA-256 and pool
   receipt, and its Prepared Snapshot and Evaluation Package are Section 5.1's;
2. the copy names that Prepared Snapshot, Profile, Profile definition, pool and pool receipt as its
   source, and that Evaluation Package as its judgments' source; its Source Bundle is the stratum's;
3. trial IDs: the copy's equal the stratum pool's, in the same order (17,105);
4. topics equal, in order (40);
5. judgments and the Judgment Scheme equal (33,869 judgments);
6. every trial record equal to the stratum's once the complete-text fields are removed;
7. full text: every Source Bundle file hashes to the stratum's lock, and every copy trial's complete
   eligibility text equals the eligibility text of that trial's XML record in the archives, and
   hashes to the SHA-256 the copy records.

It also reports, without gating, the count of stratum section texts equal to the complete text
whitespace-collapsed and of complete texts beyond the 4,096-byte provenance bound.

**Check B, criterion health against TREC 2022 (`criteria`).** No upstream TrialGPT publishes TREC
2023 trial files (TrialGPT 1.0 at `1b3242cd` and TrialGPT 2.0 at `1c5f9ebc` ship SIGIR, TREC 2021
and TREC 2022 only), so deviation D11's agreement with an upstream file has no reference on this
stratum. Check B runs `TrialGPT-TAIM-Luna-v1`'s own criterion source (`trial_view`:
`criterion_sections`, then upstream's `parse_criteria`) over every trial of the TREC 2023 copy and of
the TREC 2022 copy the TREC 2022 rows ran on, and counts per copy: trials with eligibility text; those
with no criterion on either side; those losing a paragraph to an inline heading (D11's limit);
criteria per trial, median and 95th percentile. **It passes when the TREC 2023 share of trials with
eligibility text but no criterion is at most twice TREC 2022's or at most 0.5%, whichever is
larger.** This is Mehul's rule (b), written at 2026-10-27T18:03:06Z, before the TREC 2022 copy
existed. A difference of a few dozen trials does not stop the rows; a connector or parser break,
which empties hundreds or thousands of trials, does. The other counts are reported, not gated.

**Measured before freeze** (2026-10-27). The TREC 2022 copy was rebuilt from the locked Source Bundle
(every file matched the lock by size and SHA-256 before the build) and reproduced v15 Section 4's
four TREC 2022 identities exactly. Check B passed:

| | TREC 2023 copy | TREC 2022 copy (bar) |
| --- | ---: | ---: |
| Trials with eligibility text | 17,103 of 17,105 | 26,582 of 26,585 |
| With text but no criterion (gated) | 38 (0.222%) | 31 (0.117%) |
| Losing a paragraph to an inline heading (reported) | 1,190 (6.96%) | 1,676 (6.31%) |
| Criteria per trial, median / 95th percentile | 11 / 42 | 11 / 41 |

The limit is 0.5%, the floor, since twice TREC 2022's share is 0.233%. No TREC 2021 or SIGIR copy was
on the machine that measured this, so neither is in the table.

## 5. The strata and their identities

### 5.1 The stratum identities (v12 Section 4, unchanged)

Rows 3 to 14 record these values. Each declaration carries them in `stratum`. Its
`prepared_snapshot_id` is the snapshot digest the same-snapshot rule names.

| Stratum | Profile | Prepared Snapshot ID | Task Input ID | Evaluation Package ID | Pool | Pool IDs SHA-256 | Pool receipt ID | As-of | Topics |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: |
| TREC 2021 JU | `trec-ct-2021-judgment-union` | `sha256:5ddeded8...` | `sha256:c393b4f1...` | `sha256:fa47e8e5...` | 26,162 | `sha256:1b0d2f34...` | `sha256:8c8e7f4b...` | 2021-04-27 | 75 |
| TREC 2021 EF | `trec-ct-2021-external-fidelity-26149` | `sha256:5ddeded8...` | `sha256:019f47c8...` | `sha256:d35d98a8...` | 26,149 | `sha256:fed85fad...` | `sha256:b6e7f202...` | 2021-04-27 | 75 |
| TREC 2022 JU | `trec-ct-2022-judgment-union` | `sha256:17d6af46...` | `sha256:23dc7a33...` | `sha256:08865f0a...` | 26,585 | `sha256:c5612959...` | `sha256:3cb48763...` | 2022-01-01 | 50 |
| TREC 2023 JU | `trec-ct-2023-judgment-union` | `sha256:87723d56...` | `sha256:52ab379e...` | `sha256:11c7f35b...` | 17,105 | `sha256:68d02ba2...` | `sha256:b3d4debb...` | 2023-05-08 | 40 |
| SIGIR summary | `sigir-ct-2016-summary-judgment-union` | `sha256:3c571a03...` | `sha256:e371944d...` | `sha256:aa811def...` | 3,626 | `sha256:9a106146...` | `sha256:8696755d...` | 2015-12-16 | 60 |
| SIGIR description | `sigir-ct-2016-description-judgment-union` | `sha256:c8fe1bce...` | `sha256:c05154f1...` | `sha256:3067020a...` | 3,626 | `sha256:9a106146...` | `sha256:10ff30fa...` | 2015-12-16 | 60 |

The full 64-hex values are those of v12 Section 4. The declaration drafts carry them in full. Topic
counts are each Release 0.1.1 scorecard's `topic_count`.

### 5.2 The TrialGPT TREC 2023 package (rows 1 and 2)

Rows 1 and 2 record, in place of the Prepared Snapshot, Task Input, Evaluation Package and pool
receipt IDs, those of the TREC 2023 complete-eligibility package (v15 Section 4). The other columns
are Section 5.1's, unchanged:

| Target | Profile | Prepared Snapshot ID | Task Input ID | Evaluation Package ID | Pool count | Pool IDs SHA-256 | Pool receipt ID | Clinical as-of |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| TREC 2023 judgment-union | `trec-ct-2023-judgment-union` | `sha256:8835dff29f6635a752f3062287f08a51208e89f2fa039e2294d96997f7797613` | `sha256:c0d21e30fd152b9dbc9a2451cf0bdbed01f50dbd1d73068b48223c24d9f60781` | `sha256:75e7c3ae02fec8dc12ad11364f57d05b07df9ca9cc2b022acf27aeda7cb41e41` | 17,105 | `sha256:68d02ba21e41ab48a86bd717c19a762715a190b5c0d2000dbe3a670f57d488ad` | `sha256:4d5c87e6fae98433f9d1ebd7a9df5e90a554d9285f98d3a9061d8a8d339d52ce` | 2023-05-08 |

The Profile definition SHA-256 is Section 5.1's,
`sha256:c38d7f0983bdf15954a74d76f63b4f9367aba3c99b42117d5c33f6c3d213aafd`. The package's Snapshot
Source Recipe SHA-256, which the release pins, is
`sha256:28340bd4cb74b7eb1fff65ba927104d70e1219bb311615badefb98fa15a0b45f`. The table has the shape of
v15 Section 4's, so the release's test that holds every pinned stratum to its protocol reads both
tables the same way.

**What stays the same and what changes** (check A of Section 4.1, in v15 Section 3's terms):

| | TREC 2023 judgment-union |
| --- | --- |
| Topics (patient texts, in order) | equal (40) |
| Judgments, count and content, Judgment Scheme | equal (33,869) |
| Pool: count, trial IDs and order | equal (17,105) |
| Trial records with the complete text removed | 17,105 of 17,105 equal |
| Complete texts equal to the archive XML's eligibility text | 17,103 of 17,103 (2 pool trials have no eligibility text) |
| Eligibility sections whose stratum text is the complete text, whitespace collapsed | 17,103 of 17,103 |
| Complete texts beyond the 4,096-byte provenance bound | 2,151 |

**How they were computed.** On 2026-10-27, the code of the local branch
`fm/taim-trialgpt-trec23-package` (candidate code for the Section 10 release; the builder is
unchanged since Release 0.2.0) prepared the stratum package from the locked 2023-05-08 Source Bundle
with `trial-benchmark data prepare` under protocol v3, whose pool inspection re-derived all six of
Section 5.1's TREC 2023 values exactly, then built the package with
`scripts/prepare_trialgpt_profile_snapshot.py` and derived every identity above. Every one of the
17,105 pool trials reproduced the stratum's record. Check A then ran on the two packages and the
Source Bundle and passed with no failure; the table is its record. The derivation read judgments only
inside the code, to project them into the package and to compare and count them; it read no ranking
and computed no score.

**Reproduced by the release.** Before the Section 10 release is cut, its wheel rebuilds the package
from the stratum package and reproduces every identity above, and its support accepts the package
only at the recipe SHA-256 above. A release that does not reproduce them is not cut with this
amendment.

The existing TrialGPT rows record their packages' identities the same way:

- on TREC 2021 EF, Prepared Snapshot `sha256:db95af60...` and Evaluation Package
  `sha256:35fa37ea...`, against the stratum's `5ddeded8...` and `d35d98a8...`;
- on TREC 2022, `sha256:d95c9e28...` and `sha256:a6884d83...`, against `17d6af46...` and
  `08865f0a...`.

The source is each row's declaration on `origin/evidence/trialgpt-paper-rows`.

For rows 1 and 2, as for those existing rows, "the same Prepared Snapshot" therefore holds by
content, not by ID. Mehul
ruled so on 2026-10-27, through the package's `source_stratum` record. The complete-eligibility
builder writes that record into the package's recipe. It names the stratum's Prepared Snapshot ID,
Profile and Profile definition digest, pool count, pool IDs SHA-256 and pool receipt ID
(`public/src/taim/data/complete_eligibility.py:216-223` and `:244` at `b783425e`). The package's
Evaluation Package records the same values, plus `source_evaluation_package_id`, under
`profile_complete_eligibility_projection` (`:229-235`). The content check of Section 4 requires
every one of them to equal the stratum's Section 5.1 value. Each TREC 2023 TrialGPT declaration
also carries the stratum's Prepared Snapshot, Evaluation Package and pool receipt IDs in its own
`source_stratum` field (Section 1.1), derived by the release's declaration command from those same
package records.

The rule needs all of the following, and check A of Section 4.1 records each:

- the same pool IDs SHA-256 and as-of date;
- judgments equal in count and content to the stratum's Evaluation Package;
- every trial record equal to the stratum's once the complete eligibility text is removed;
- the complete text traced to the 2023-05-08 Source Bundle.

The re-score is against the stratum's own Evaluation Package `11c7f35b...` (v12 Section 3
condition 2; v15 Section 5), taken from the Release 0.1.1 BM25 row `paper-v0.1.1-ju23-bm25-a40`.

## 6. TrialMatchAI v0.9.1 corpora

### 6.1 The route per stratum

| Stratum | Corpus | Who prepared the trial text, and with what | What must be built |
| --- | --- | --- | --- |
| TREC 2021 EF | the admitted row's corpus, receipt `trialmatchai-corpus-receipt.json` of `common-trialgpt-75` | upstream, with its earlier BERN2-based parser (Zenodo record 15516900) | nothing |
| TREC 2021 JU | upstream's published preparation, cut to the 26,162-trial pool | upstream, as above | a filter and index. Whether it holds the 13 pool trials outside the EF pool is not measured; any missing trial is prepared by us under Section 6.2 |
| TREC 2022 JU | upstream's published preparation for 26,584 trials, plus `NCT00000771` | upstream, as above, for 26,584; us, with v0.9.1's own code, for 1 | the one trial, then the filter and index |
| TREC 2023 JU | built from the 2023-05-08 snapshot | us, with v0.9.1's own code | all 17,105 trials, then the index |
| SIGIR 2016, both strata | built from the 2015-12-16 snapshot; one corpus serves both strata, whose pools are identical (pool IDs `9a106146...`) | us, with v0.9.1's own code | all 3,626 trials, then the index |
| TREC 2021 EF, Section 6.4 measurement only | built from the 2021-04-27 snapshot | us, with v0.9.1's own code | all 26,149 trials, then the index |

The coverage figures are from `docs/evidence/trialmatchai-corpus-build-cost-measurement-2026-09-27.md`,
lines 36-51.

### 6.2 How a corpus is built

1. **Source.** The stratum's own Prepared Snapshot source records, read from the locked Source
   Bundle. No call to the live ClinicalTrials.gov registry is made. Upstream's own route for
   track 23 fetches from the live registry (`trec/backfill.py`) and is not used.
2. **Conversion.** A TAIM-owned converter, committed with its tests before the build, writes each
   trial as the ClinicalTrials.gov v2 JSON that upstream's `normalize_study` reads. The same
   `normalization.py` is used by v0.9.1 and the fork: blob `95055ee0` in both checkouts. Each output
   records the SHA-256 of its source record. The converter reads the TREC 2021/2022 XML, the TREC
   2023 XML and the SIGIR 2016 XML.
3. **Preparation.** Upstream TrialMatchAI at `3bc0e5f44d742ef069f8ef73bbc4187573d9db6a`, unchanged:
   - its own `normalize_study` and preparation functions;
   - in its own environment;
   - with the bound S2 concept store (manifest `sha256:1cd46b86...`) and the pinned model revisions
     of the v0.9.1 System's adapter module.

   Shards are allowed. Preparation is per trial and resumes by skipping prepared trials. No trial is
   prepared twice into the final corpus.
4. **Identity.** A corpus receipt (a content digest over every file) and the search database
   identity are committed before the content check runs. The run records both before its System
   starts.
5. **No score before the corpus is fixed.** No ranking of the System on any topic of that stratum is
   made before the receipt and the content check are committed.

### 6.3 Why the preparations differ, and the build gate

Mehul's rule of 2026-10-27 compares our preparation route with upstream's published TREC 2021
preparation. If they match, the disclosure says "prepared by us with upstream's code, matches their
published TREC 2021 preparation". If not, it lists the differences, and no corpus of Section 6.2 is
built until Mehul rules.

**The counts** (from kept outputs, read-only, before freeze). Job 22464791 kept per-trial counts but
no prepared files. Its inputs were the trial records of the published TREC 2021 preparation itself,
so these counts test the preparation step alone, not the converter. On its 193 trials (ID digest
`sha256:46d553b3...`, in `docs/evidence/trialmatchai-upstream-v091-prepare-cost-22464791-trials.jsonl`):

- criteria per trial equal on 193 of 193, 3,174 both (the timing job was given the published
  chunks, so this count is by construction);
- recognised entities equal on 23 of 193: 4,903 ours against 6,224 in the published preparation.
  Ours had fewer on 127 trials and more on 43;
- of the published preparation's 6,092 linked IDs, 5,253 have forms the bound store never writes:
  2,915 are `C` followed by seven digits, 1,934 carry `mesh:`, 350 `omim:` and 54 `NCBI:txid`.

That is a mismatch.

**The cause** (read-only, from upstream `cbib/TrialMatchAI`'s full history and the Zenodo record's
metadata):

- **The published preparation was made by an earlier tool.** Zenodo record 15516900 ("TrialMatchAI
  resources", created 2025-05-26) holds the prepared trials and criteria with `resources.tar.gz`
  and `models.tar.gz`. It names no tool. Upstream's tree of that day (`d54af22`) downloads those
  files (`setup.sh:6-9`) and prepares criteria with a BERN2-based parser under `src/Parser/`:
  - recognition by BERN2's multi-type NER (`biomedner_init.py:502-508`) and a GLiNER server over
    seven clinical labels (`gner_server.py:52-59`);
  - linking by BERN2's normalizer: genes through GNormPlus as `EntrezGene:` (`normalizer.py:250`,
    `:286`), diseases through a normalizer daemon (`:197`), species as `NCBI:txid` (`:232`),
    procedure and sign-symptom dictionaries (`:28-29`), and `CUI-less` when nothing links (`:49`).
- **The published files carry that tool's output.** On the 193 trials, the entity groups are its
  labels, including `treatment`, `surgical procedure` and `genomic analysis technique`, which
  v0.9.1's schema does not have. Each group carries the IDs of its own legacy normalizer: diseases
  `mesh:` and `omim:`, genes `EntrezGene:`, species `NCBI:txid`, the seven GLiNER labels `C` plus
  seven digits.
- **Upstream removed that tool on 2026-06-21** (`9e8e9da`, "Replace BioNER services with schema
  entity linker"). v0.9.1's changelog: "GLiNER2 in-process NER replaces the BERN2 socket daemons;
  concept linking moves to a LanceDB FTS+vector store" (`CHANGELOG.md:461-462`).
- **v0.9.1's own code cannot reproduce it.** It recognises with GLiNER2 over its own schema
  (`entity_schemas/trialmatchai.yaml`). It links against the bound store, whose six open sources
  (`entities/concept_sources.py:49-75`) hold no MeSH, OMIM, `C`-plus-digits or NCBI-taxid IDs.
  - A store imported from BERN2-style dictionaries (`build-concepts --dictionary`,
    `cli/build_concepts.py:38-44`) could hold MeSH and OMIM IDs. But it is not the bound store, and
    recognition would still differ.
  - Both tools embed criteria with `BAAI/bge-m3`. They do not split criteria alike. Measured on
    2026-10-27, CPU only, on the same 193 trials, with route A from the snapshot text: exactly the
    published criterion texts on 90 of 193 trials; 2,812 of 3,174 published criteria (88.6%)
    reproduced; 3,390 criteria against 3,174; inclusion and exclusion counts equal on 122 of 193
    trials, with labels mapped, and 263 criteria left untyped by v0.9.1. The eligibility text
    itself equals the published text on 181 of 193 trials, so the difference is mostly splitting.
- **What the existing rows read.** TAIM's v0.9.1 route runs no preparation or linking on TREC 2021.
  It builds its index over the prepared corpus
  (`docs/agents/trialmatchai-upstream-v091-run-commands.md:101`). So the admitted K = 60 row read the
  published, BERN2-era preparation, and the bound store linked only the topics, at match time
  (`:178-179`).
- **Upstream's own tag splits the same way.** Tracks 21 and 22 read the published corpus
  (`cli/bootstrap_data.py:22-24`). Track 23 "needs a backfill + prepare pass first"
  (`trec/corpus.py:13-16`), through "the standard build stages (prepare -> index)"
  (`trec/backfill.py:1-11`).

**Route A, the build plan.** TREC 2021 and TREC 2022 read upstream's published preparation. TREC
2023 and both SIGIR strata read text prepared by us with v0.9.1's own code under Section 6.2. This is
the split upstream's own tag makes, except that our text comes from each stratum's snapshot and not
from the live registry. The few trials that TREC 2021 JU or TREC 2022 lack are prepared under
Section 6.2 too. Mehul confirmed route A on 2026-10-27.

**The disclosure (the mismatch outcome).** It applies to rows 6, 7 and 8, and to any trial of rows 3
and 5 prepared by us:

> Prepared by us with upstream TrialMatchAI v0.9.1's own preparation code at `3bc0e5f4`, unchanged.
> Upstream's published TREC 2021 and 2022 preparation was made with its earlier BERN2-based parser,
> which upstream removed on 2026-06-21, so the two differ. Entity recognition: a different model and
> label set; on 193 TREC 2021 trials, entity counts were equal on 23, 4,903 against 6,224. Linked
> vocabularies: an open store of NCBI Gene, DOID, ChEBI, Cellosaurus, CL and HPO, against MeSH,
> OMIM, NCBI Gene, ChEBI, CL, NCBI Taxonomy and `C`-plus-digits IDs. Criteria splitting: exactly
> the published criterion texts on 90 of 193 trials; 2,812 of 3,174 published criteria (88.6%)
> reproduced; 3,390 criteria against 3,174. The same: the embedding model (`BAAI/bge-m3`).

One of the two Section 6.4 sentences follows that text: the measured effect if the measurement ran,
or the unmeasured sentence if it was skipped.

**The GPU check is dropped,** by Mehul's decision of 2026-10-27. The A40 check proposed before
(about one GPU-hour on the same 193 trials) would re-measure a difference whose cause is now known.
The converter is checked by its own tests and by the Section 4 content check.

### 6.4 The corpus-effect measurement

Mehul approved it on 2026-10-27, on these conditions, as loosened on 2026-11-02 (Section 11):
- **Where.** Its corpus is prepared on CPU nodes of any cluster available to us, and it runs on one
  NVIDIA H200.
- **When.** It may run once every row of Section 3 is declared, and nothing in Section 3 waits for
  it. Its result is read only after every row of Section 3 has run.
- **If the allocation is short.** If no allocation is available for it on any cluster, the
  measurement is skipped.

- **What.** `trialmatchai-upstream-v091-cuda-trec21-development-v1` on TREC 2021 external-fidelity,
  K = 20, reading all 26,149 pool trials prepared by us under Section 6.2 from the 2021-04-27
  snapshot. Draft identifier: `measurement-trialmatchai-v091-trec21-ef-own-prep-1`.
- **What it is not.** It is a measurement, not a paper row. It enters no paper table, and v12
  Section 3 condition 8 does not count it.
- **Its comparison.** With row 4: the same System, stratum and K, on the published preparation. The three K = 20 tests of Section 8.2 are registered in the measurement's declaration
  before it runs. They form their own family of three under Section 8.3, never part of the TREC 2021
  EF family.
- **The noise floor.** Two runs of this System on identical inputs already differ. Between Delta
  jobs 22464546 and 22755345, all pre-eligibility files were equal, and the top-60 overlap per topic
  ran from 43 to 59 of 60, median 51. The measurement's top-20 overlap with row 4 is reported beside
  the same overlap of those two runs.
- **Order.** Once every row of Section 3 is declared; its result is read only after every row of
  Section 3 has run. Its corpus passes the Section 4 content check before its declaration is
  committed.
- **Use.** Its result is disclosed with rows 6, 7 and 8, as the measured effect of the preparation
  route on TREC 2021. It changes no rule and no row.

**The two outcomes, as disclosed with rows 6, 7 and 8:**
- **Run:** "On TREC 2021 external-fidelity, the same System on our preparation against upstream's
  published preparation", followed by the three paired tests with their Holm levels, and the top-20
  overlap beside the rerun overlap.
- **Skipped:** "The effect of this preparation difference on rankings was not measured. The planned
  measurement on TREC 2021 external-fidelity was skipped because the compute allocation for it was
  not available."

## 7. The staged System on the new strata

### 7.1 Folded queries

The `bm25-folded` component needs a folded-query bundle bound to each stratum's Task Input ID.
`queries prepare` builds it on the CPU, from the Task Input, with medspaCy ConText rules and the
user-supplied SNOMED CT International Edition of 2026-08-01, identified in
TAIM's SNOMED CT folded-query lexicon lock (`public/docs/method-contracts.md:89-110`).

- Each stratum gets its own bundle. TREC 2021 JU and EF have different Task Input IDs, so they get
  one each.
- Each bundle is built and committed before the row's declaration. The declaration's amendment
  statement names its bundle ID.
- A topic whose folded query is empty is reported as it is, never patched.

The builder now refuses any lineage but TREC 2021: "folded queries are qualified only for TREC CT
2021" (the folded BM25 extension module, lines 80-81). Section 10 lifts that refusal. The folding
rules themselves do not change.

### 7.2 Components

On each stratum, `bm25-folded` and `dense-qwen3-embedding-0.6b-no-template` run at K = 5,000 on that
stratum's Task Input, from the row's own release, before the staged row (`staged/extension.py:98-126`).
They are components, not rows.

### 7.3 The unreranked baseline

The staged run stores four Stage Rankings, including "the depth-2,000 RRF pool"
(`public/docs/method-contracts.md:155-158`, stage `rrf-scoring-pool-depth2000`).

The baseline of v1 Section 5 item 1 is the top 1,000 of that stage, in its recorded order: fused
score descending, then trial ID ascending. Its identifier in declarations is:

`<run_id>/stage/rrf-scoring-pool-depth2000/top-1000`

It is the same run's own pre-rerank ranking, at the same K as the Primary Ranking. It is scored by the
Release 0.1.1 evaluator against the stratum's Evaluation Package, exactly as the row is. It is a
diagnostic stage that this amendment registers for one comparison. It is not a paper row and not an
alternative Primary Ranking.

## 8. Comparisons

### 8.1 The registered pairs

Each pair below shares a stratum and a K, and is registered in a declaration before its first run
starts:

| Stratum | A | B | K | Registered in | Tests |
| --- | --- | --- | ---: | --- | ---: |
| TREC 2021 JU | row 9, staged | its own baseline (Section 7.3) | 1,000 | row 9's declaration | 5 |
| TREC 2021 JU | row 9, staged | `paper-v0.1.1-ju21-bm25-a40` (BM25, Release 0.1.1) | 1,000 | row 9's declaration | 5 |
| TREC 2021 EF | row 4, v0.9.1 | `paper-trialgpt1-trec21-row-1` (TrialGPT 1.0, existing) | 20 | row 4's declaration | 3 |
| TREC 2021 EF | row 4, v0.9.1 | `paper-v0.1.1-trialmatchai-l4-trec21-external-fidelity-a40` (TrialMatchAI -v4, existing) | 20 | row 4's declaration | 3 |
| TREC 2021 EF | row 10, staged | its own baseline | 1,000 | row 10's declaration | 5 |
| TREC 2021 EF | row 10, staged | `paper-v0.1.1-ef21-bm25-a40` | 1,000 | row 10's declaration | 5 |
| TREC 2022 | row 5, v0.9.1 | `paper-trialgpt1-trec22-row-1` (existing) | 20 | row 5's declaration | 3 |
| TREC 2022 | row 11, staged | its own baseline | 1,000 | row 11's declaration | 5 |
| TREC 2022 | row 11, staged | `paper-v0.1.1-ju22-bm25-a40` | 1,000 | row 11's declaration | 5 |
| TREC 2023 | row 1, TrialGPT 1.0 | row 6, v0.9.1 | 20 | both declarations, before either run | 3 |
| TREC 2023 | row 12, staged | its own baseline | 1,000 | row 12's declaration | 5 |
| TREC 2023 | row 12, staged | `paper-v0.1.1-ju23-bm25-a40` | 1,000 | row 12's declaration | 5 |
| SIGIR summary | row 7, v0.9.1 | `paper-trialgpt1-sigir-sum-row-1` (existing) | 20 | row 7's declaration | 3 |
| SIGIR summary | row 13, staged | its own baseline | 1,000 | row 13's declaration | 5 |
| SIGIR summary | row 13, staged | `paper-v0.1.1-sigir-s-bm25-a40` | 1,000 | row 13's declaration | 5 |
| SIGIR description | row 8, v0.9.1 | `paper-trialgpt1-sigir-desc-row-1` (existing) | 20 | row 8's declaration | 3 |
| SIGIR description | row 14, staged | its own baseline | 1,000 | row 14's declaration | 5 |
| SIGIR description | row 14, staged | `paper-v0.1.1-sigir-d-bm25-a40` | 1,000 | row 14's declaration | 5 |

78 tests in all: 18 at K = 20, 30 staged against its baseline, and 30 staged against BM25.

The TREC 2023 K = 20 pair joins `paper-trialgpt1-trec23-row-1` (row 1) and
`paper-trialmatchai-v091-trec23-row-1` (row 6). Row 1's declaration names row 6, and row 6's names
row 1, with the same three metrics of Section 8.2; `paper-trialgpt2-trec23-row-1` (row 2) registers
none, and its stratum table
carries v14 Section 7's caption note.

The BM25 rows are the Release 0.1.1 reference rows of v12 Section 2. Each ranked 1,000 trials and
is scored against its stratum's own Evaluation Package (`budget_k` 1,000 and
`evaluation_package_id` in `docs/evidence/derived-recall-100-500-2026-09-27.json`). BM25 was
bit-identical across cards (v12 Section 2), so its card adds no spread.

### 8.2 Metrics and tests

**Metrics.**

- **On the TREC strata:** macro graded nDCG@10, eligible P@10, and macro and micro judged eligible
  recall at K = 100, 500 and 1,000. These are the metrics the TrialGPT declarations already name.
- **At K = 20,** both rankings end at rank 20, so recall at 100, 500 and 1,000 is one test of
  recall at 20. This is how the registered TREC 2021 comparison was computed
  (`trec21/registered-comparison-trialgpt1-trialmatchai.md:11-15`).
- **On SIGIR,** the Profile's precision is referral-candidate P@10 (labels 1 and 2), and
  "eligible" recall is label-2 recall. The SIGIR scorecards' `precision_policy` names
  `referral_candidate_precision`.
- **Micro recall** is reported and not tested.

**Tests.** Each test is the exact two-sided sign test and the 10,000-resample, seed-20260819 topic
bootstrap, distinguishable only if both reject (v1 Sections 6.1-6.2), at the levels Section 8.3
sets. Each comparison is computed only after both of its rankings are re-scored, by a script
committed before its data.

### 8.3 Multiplicity: Holm within each stratum

This replaces v1 Section 6.5 for the tests of this amendment only. It is declared before any of
their results exists. Mehul confirmed it as written here on 2026-10-27.

**Families.** One family per stratum: every test this amendment registers on that stratum.

| Stratum | K = 20 tests | Staged vs baseline | Staged vs BM25 | Family size m |
| --- | ---: | ---: | ---: | ---: |
| TREC 2021 JU | 0 | 5 | 5 | 10 |
| TREC 2021 EF | 6 | 5 | 5 | 16 |
| TREC 2022 | 3 | 5 | 5 | 13 |
| TREC 2023 | 3 | 5 | 5 | 13 |
| SIGIR summary | 3 | 5 | 5 | 13 |
| SIGIR description | 3 | 5 | 5 | 13 |

The Section 6.4 measurement, if it runs, is a separate family of 3.

**The procedure,** at a family-wise level of 0.05:

1. Order the family's tests by their sign-test p-value, smallest first. Test j (counting from 1)
   gets the level a_j = 0.05 / (m - j + 1).
2. Walk down the list. A test passes its step if its sign-test p is below a_j. At the first test
   that does not pass, it and every later test fail.
3. A test that passed its step is distinguishable only if its bootstrap percentile interval at
   level 1 - a_j excludes zero. The interval uses the same 10,000 resamples and seed; only its level
   changes.

With m = 1 this is exactly v1 Section 6.2.

**The size of each family is fixed now.** A registered test whose row never runs, or whose corpus
fails its check, counts as not rejected with p = 1. A family never shrinks.

**Reported per test:** the raw sign-test p, its Holm level a_j, the bootstrap interval at that level,
and the verdict. A sentence about several strata cites each stratum's verdict. No pooled test across
strata is registered.

**Why Holm within each stratum:**

- **Valid under any dependence.** The tests of one pair are strongly dependent: the recall cut-offs
  are nested, and nDCG@10 and P@10 read the same top 10. Holm needs no assumption about that
  dependence. Hochberg's and Hommel's procedures do, and Benjamini-Hochberg controls the false
  discovery rate, not the per-verdict error that "distinguishable" implies.
- **Never weaker than Bonferroni,** at the same guarantee.
- **The bootstrap step can only remove rejections.** The distinguishable set is a subset of the
  sign test's Holm rejections. That set holds the family-wise error at 0.05, so the conjunction does
  too.
- **One stratum, one benchmark.** Each stratum has its own topics and judgments, and the paper
  reports each as its own result. A single family of 78 would make TREC 2023's verdicts depend on how
  many tests other strata carry. One family per pair (3 or 5 tests) would leave the per-stratum
  tables uncorrected.

**What it costs.** At m = 16 the first step needs a sign-test p below 0.0031 and a 99.69% interval.
- The sign test then needs at least 10 topics that move: the smallest p for 9 moving topics is
  2 x 0.5^9 = 0.0039.
- 10,000 resamples leave about 16 in each tail of a 99.69% interval. Its ends are coarse, but
  deterministic under the fixed seed.

**Tests registered before this amendment** keep v1 Section 6.5 (no correction). Each verdict in the
paper names the rule it was computed under.

### 8.4 What is never paired

- `TrialGPT2-TAIM-Luna-v1`, anywhere (v14 Section 7);
- the two v0.9.1 rows on TREC 2021 EF with each other (Section 1.2);
- v0.9.1 on TREC 2021 JU: no other K = 20 row exists there;
- any K = 20 row with any K = 1,000 row (v12 Section 6).

The staged rows sit beside the other three reference rows (BGE-M3, Qwen3 and RRF, K = 1,000) without
a registered pair. Any comparative sentence about them must cite a paired verdict or call itself
untested (v1 Section 6.4).

Every paired comparison is shown with each row's card or CPU platform and each System's measured
spread across cards, or "not measured" (v12 Section 6).

## 9. Qualification

1. **TrialGPT, rows 1 and 2.** The qualification gates of both TrialGPT Systems ran once, on TREC
   2021, under v14 Section 6, and passed. They are not repeated on TREC 2023, as they were not
   repeated on TREC 2022 or SIGIR. TREC 2023 topics are questionnaire-derived texts (connector
   rendering `trec-2023-questionnaire-canonical-text-v1`), unlike the narrative topics of the
   qualification Track. This is disclosed (Section 12 item 10) and changes no method.
2. **v0.9.1, rows 3 to 8.** No qualification gate: it is a direct pin, run as the tag runs TREC.
   Their topic route takes age and sex by regular expressions (`trec/topics.py:196-233`), and its
   behaviour on questionnaire-derived TREC 2023 text and on SIGIR text is disclosed, not adjusted.
3. **The staged System, rows 9 to 14.** No qualification gate: its recipe was fixed on TREC 2021
   and is not tuned on any other stratum. The folded-query builder is opened to the new Tracks with
   its rules unchanged (Section 7.1).

## 10. Release and order of work

**The Section 10 release** is the first release that ships this amendment. It ships:
- the TrialGPT TREC 2023 capability: the fifth TrialGPT target, both Systems' support rows on
  `trec-ct-2023-judgment-union`, and the pins of Section 5.2's identities and recipe SHA-256;
- the v0.9.1 support rows for TREC 2021 JU, TREC 2022, TREC 2023 and both SIGIR strata, with the
  System ids of Section 3;
- the staged System's support rows for TREC 2022, TREC 2023 and both SIGIR strata, with those of its
  two components;
- the folded-query builder opened to those Tracks. The staged System reaches them through the
  support rows; the declaration and its definition SHA-256 are unchanged.

Release 0.2.0 already declares `trialmatchai-upstream-v091-cuda-trec21-development-v1` and
`staged-bm25-qwen3-rrf-rerank` on TREC 2021 EF, and the staged System on TREC 2021 JU
(`release-support-v0.1.json`).

**Order, for every row:**

1. This amendment is frozen.
2. The release that serves the row exists in the release ledger.
3. For rows 5 to 8, and the Section 6.4 measurement: the converter is committed, the corpus is
   built under route A (Section 6.3), its receipt is committed, and the Section 4 content check
   passes and is committed. The corpora that Section 13 lists as built before the freeze took this
   step before steps 1 and 2.
4. For rows 1 and 2: the TREC 2023 complete-eligibility package is built, its identities equal
   Section 5.2's, and checks A and B of Section 4.1 pass; both check records are committed.
5. For rows 9 to 14: the stratum's folded-query bundle is built and committed.
6. The row's declaration is committed. Rows 1 and 6 both carry their comparison, and the earlier of
   the two commits precedes both runs.
7. The row runs to completion once: no score of a run is visible before the run completes, and a
   completed run is never replaced. A run that stops before it writes its ranking for an
   operational reason may be replaced, as many times as needed. Operational reasons are a node
   failure, a preemption, a wall-time kill, a lost workspace, or a crash of the job or of its
   environment. For the rows of this amendment this widens v12 Section 3 condition 7, which speaks
   of a second run.

   Before each replacement starts, the failure, its job record and its last progress line are
   committed. Each replacement has a new run id and is declared with the next `ordinal` (2, 3 and
   so on). Its declaration names every earlier failed run, with every other field of the first
   declaration unchanged.

   A replacement may resume from the failed run's checkpoint, or start from nothing, whichever is
   practical. If it resumes, its record names exactly which batches it took from the checkpoint,
   and the row discloses them (Section 12 item 12).

   A failure that needs a change to the release, the System, its inputs or its configuration is
   not repaired by a replacement. The row is then declared again on the changed release, and every
   attempt is disclosed. A score is never an operational failure.
8. The row is re-scored by the Release 0.1.1 evaluator. For rows 9 to 14 the baseline of Section 7.3
   is re-scored too.
9. The registered comparisons are computed, with Section 8.3 applied once a stratum's family is
   complete.

**Order for rows 1 and 2 in detail** (v14 Section 9 with v15's step 0, on the Section 10 release):

0. Build the TREC 2023 complete-eligibility package from the stratum package and the 2023-05-08
   Source Bundle; check that its four identities and recipe SHA-256 equal Section 5.2's; run checks A
   and B of Section 4.1, which must pass; commit both check records.
1. Write both declarations with the release's command, `TrialGPT-TAIM-Luna-v1`'s naming row 6's
   comparison; commit them. This commit, or row 6's declaration commit if earlier, precedes both runs
   of the Section 8 pair.
2. The target's one generation (120 calls).
3. The query-plan receipt, committed.
4. The one retrieval freeze, on an NVIDIA GPU (MedCPT), no provider call.
5. The recomputation, committed.
6. No gates (Section 9).
7. Both rows, on the one frozen retrieval.
8. Re-score each row with the Release 0.1.1 evaluator against `paper-v0.1.1-ju23-bm25-a40`'s
   Evaluation Package.

A generation or a retrieval freeze that stops for an operational reason before it writes its
receipt may be run again from the start. The stopped attempt and its calls are recorded, and its
partial output is never used.

**Order across rows.** Row 9 (TREC 2021 JU) is the first staged row to start; row 10 may start
while row 9 runs. Row 4 can run as soon as its declaration is committed. The timing sample of
Section 11 runs after this amendment is frozen. The Section 6.4 measurement may run once every row
of Section 3 is declared, and its result is read only after every row of Section 3 has run; or it
is skipped (Section 6.4).

## 11. Authorization (Mehul's ruling and Hannes Ill's agreement)

At the freeze, this section quotes both, verbatim.

**Mehul's authorization ruling:**

> I authorize the uses of TREC 2022 and TREC 2023 data listed in Section 11 of this amendment, on
> its rules, and no others.

(Mehul Mittal, 2026-11-02)

**Hannes Ill's agreement:**

- On 2026-11-02, to this amendment as sent to him for review: "Approved"
- On 2026-11-02, to the changes on operational failures (Section 10 step 7 and its use in Section
  3, Section 12 item 12, and the operational wording of Sections 1.1, 1.2, 4, 6.4, 10, 11 and 14):
  "go ahead with what you feel is right"

On that ruling and that agreement, this amendment is the frozen, version-controlled analysis
protocol that the repository's rule on TREC 2022/2023 access asks for. It covers these purposes
only:

- **TrialGPT.** Giving `TrialGPT-TAIM-Luna-v1` and `TrialGPT2-TAIM-Luna-v1` the TREC 2023 patient
  texts and the 500 candidates per topic of the target's one retrieval. Sending them to the provider
  through each System's declared transport, on Hannes Ill's own Codex account, as v5 and v14 already
  authorize on the other targets.
- **TrialGPT copy checks.** Running checks A and B of Section 4.1 on the TREC 2023 copy, and
  building the TREC 2022 complete-eligibility copy as check B's bar. Check A reads the judgments only
  to compare the copy's with the stratum's inside the script, and prints their count; check B reads
  no judgment; building the TREC 2022 copy reads judgments only inside the code that projects and
  hashes them.
- **TrialMatchAI v0.9.1.** Giving the v0.9.1 Systems of Section 3 the TREC 2022 and TREC 2023
  patient texts on the executing cluster.
- **Corpora.** Building the corpora of Section 6, the Section 6.4 corpus included, and running the
  content checks of Section 4. The Section 6.4 measurement is covered on the conditions of that
  section.
- **The staged System.** Building the folded-query bundles of Section 7.1 and running the staged
  System's components on TREC 2022 and TREC 2023.
- **A timing sample** of the staged reranker on the new strata. It reads topic texts and candidate
  trial texts and no judgment. Its result sets no rule.
- **Judgments.** Building the TREC 2023 complete-eligibility package, which reads judgments only
  inside the code that projects and hashes them. Re-scoring every row of Section 3 against its
  stratum's Evaluation Package with the Release 0.1.1 evaluator, and computing Section 8. These are
  the only uses of the judgments.

It does not cover:
- any other System or experiment on TREC 2022 or TREC 2023;
- any qualification run on them;
- any run. Each run still needs Mehul's decision to run. A replacement under Section 10 step 7
  runs under the decision that started its row.

## 12. Disclosure

Every reported row of Section 3, and the paper, discloses the following. This is in addition to v12
Section 11, v13 Section 7, v14 Section 10, v15 Section 10 and v16, as they apply.

1. This amendment, by protocol ID, freeze date and SHA-256. That it was written after the values in
   Section 13 had been read.
2. **Rows whose trial text was prepared by us, not upstream:**
   - row 1: the TREC 2023 complete-eligibility copy, built by TAIM with its own builder
     (`taim.data.complete_eligibility`, v15 Section 3) from the stratum package and the locked
     2023-05-08 Source Bundle. Neither upstream TrialGPT 1.0 nor TrialGPT 2.0 publishes TREC 2023
     trial files, so unlike SIGIR, TREC 2021 and TREC 2022, no criterion agreement with an upstream
     file (deviation D11) can be reported. The row names the copy's identities (Section 5.2), the
     commit of check A's and check B's records, and check B's counts beside TREC 2022's;
   - row 2: the same copy;
   - row 5: one trial (`NCT00000771`) built by us; 26,584 trials from upstream's published
     preparation;
   - row 6: all 17,105 trials built by us from the 2023-05-08 snapshot;
   - rows 7 and 8: all 3,626 trials built by us from the 2015-12-16 snapshot;
   - row 3, only if any of its 13 extra pool trials is missing from the published preparation.

   Each of these rows names its converter commit, corpus receipt and content-check record. Rows 5 to
   8 (and row 3, if it applies) also carry the Section 6.3 disclosure, in its words, followed by
   the Section 6.4 outcome sentence that applies.
3. That neither upstream TrialGPT nor the TrialMatchAI paper evaluated TREC 2023, nor
   TrialMatchAI SIGIR summary. So rows 1, 2 and 6 to 8 are TAIM extensions, and row 7 reads a SIGIR
   view the tag has no route for (Section 1.2).
4. That the staged System was developed on TREC 2021, so rows 9 and 10 are development strata and
   rows 11 to 14 its first held-out tests.
5. That `trialmatchai-upstream-v091-cuda-trec21-development-v1` has two rows on TREC 2021 EF, at K
   = 60 and K = 20, never paired. The K = 60 row's v13 disclosures stand.
6. The registered comparisons of Section 8, with test counts, and the reason each other row enters
   none.
7. For rows 1 and 2: that their Prepared Snapshot, Task Input and Evaluation Package IDs differ from
   the stratum's by construction, and that the same-snapshot rule holds by content (Section 5.2):
   check A found the copy's topics, judgments, pool and records equal to the stratum's and its
   complete text equal to the archives'. Each row also discloses its declaration's
   `source_stratum`, and v15 Section 10 items 1 to 6 as they apply to TREC 2023, with this amendment
   in place of v15 for the TREC 2023 package and the TREC 2023 count of pool trials whose eligibility
   text the stratum carries only as a cut excerpt (2,151).
8. That the tests of Section 8 are Holm-corrected within each stratum (Section 8.3), while tests
   registered before this amendment are not. Each verdict names its rule.
9. For rows 1 and 2, **existing parser behaviour, not new to TREC 2023:** `TrialGPT-TAIM-Luna-v1`'s
   criterion source (deviation D11) drops a whole paragraph that holds a heading phrase. On TREC 2023
   this touches 1,190 of 17,103 trials with eligibility text (6.96%), against 1,676 of 26,582
   (6.31%) on TREC 2022; 38 TREC 2023 trials (0.222%) get no criterion, against 31 (0.117%) on TREC
   2022 (Section 4.1). Check B's record carries both.
10. For rows 1 and 2: that the TrialGPT Systems were qualified on TREC 2021 only, and that TREC 2023
    topics are questionnaire-derived (Section 9).
11. For rows 9 and 10, which run on Release 0.2.0: the run record names the requested attention
    kernel, `flash`, as the one that ran. Two kernels ran. cuDNN attention served every padded
    batch, because the flash kernel takes no padding mask. Flash served every batch without padding.
    The same holds for every staged row on Release 0.2.0. Each row states how many batches each
    kernel served, recomputed from its record's `batch_composition_sha256`. The recipe's settings
    are unchanged.
12. For any row that resumed from a failed run's checkpoint (Section 10 step 7): exactly which
    batches came from that run, as its record names them. For a staged row, also the attention
    kernels that run's record names for those batches (item 11).

## 13. What was known when this was written

- Every Release 0.1.1 reference row on all six strata existed, and its values had been read.
- All eight TrialGPT rows on the other four targets existed and had been read. The registered TREC
  2021 comparison of TrialGPT 1.0 and TrialMatchAI -v4 had been computed.
- The admitted v0.9.1 K = 60 row's values had been read (nDCG@10 0.5568). The released-route
  rerun of the same row (Delta job 22755345, not scored) had been compared with it by top-60 overlap,
  not by score.
- The staged System's TREC 2021 development values had been read (v1 Section 4.1: nDCG@10 0.4100).
- No output of any row of Section 3 existed. No folded-query bundle existed for any stratum but
  TREC 2021.
- Three v0.9.1 corpora had been built, on Mehul's decisions of 2026-10-27 (TREC 2023 and SIGIR) and
  2026-11-01 (TREC 2021 JU). Each was built from trial texts only, and no ranking was made on any
  of them:
  - TREC 2023, for row 6: all 17,105 trials, prepared by us under Section 6.2 from the 2023-05-08
    snapshot. Delta job 23319527 recorded its receipt and search database identity, and its
    Section 4 content check (job 23319574) passed.
  - SIGIR 2016, for rows 7 and 8: all 3,626 trials, prepared by us under Section 6.2 from the
    2015-12-16 snapshot. Job 23293141 recorded its receipt and search database identity, and its
    content check (job 23293490) passed.
  - TREC 2021 JU, for row 3: upstream's published preparation (Zenodo record 15516900), cut to the
    26,162-trial pool and indexed (job 23417016, 2026-11-02). The pool's trial IDs come from TREC
    2021's judgments, through the release's own pool resolution (the judgment-union Profile), as
    every JU row's do; no topic or label was used or written. The pool equalled v12 Section 4's,
    and the corpus's EF-pool part equalled the admitted EF row's corpus. Coverage was
    complete: all 13 pool trials outside the EF pool are in the published preparation, so no
    trial was prepared by us and Section 12 item 2 does not apply to row 3. Of those 13, five have
    no published criteria (NCT00819130, NCT02343068, NCT02383082, NCT02771119 and NCT02957071).
    In all, 42 of the 26,162 pool trials have none: these five, and 37 in the EF pool, which the
    admitted EF row's corpus holds the same way. Under Section 6.1 they are used as published:
    only a trial missing from the published preparation is prepared by us.

  Both content checks ran their checks script as committed before either ran.
- An unscored 3-topic engineering smoke test of the staged System on Release 0.2.0 ran on
  2026-11-01. Apart from the release's own resolution of the TREC 2021 judgment-union pool, no
  Evaluation Package, judgment or metric was used, and its outputs were deleted.
- Cost estimates and the Section 6.3 count comparison were made before freeze. Cost bears on order
  and pace, not on any rule here.
- The cause in Section 6.3 was traced read-only from upstream's history and the Zenodo record's
  metadata. No archive of that record was downloaded for that trace. Its archives were downloaded
  on 2026-11-01, only to build the TREC 2021 JU corpus (above).
- The rerun overlap of Section 6.4 (Delta jobs 22464546 and 22755345) had been read. Neither
  ranking had been scored against the other.
- Section 8.3 was written before any test of Section 8 had a result. No row of Section 3 existed.
- The TREC 2023 TrialGPT package of Section 5.2 had been built and checks A and B of Section 4.1 had
  been run; check B's rule (b) was written at 2026-10-27T18:03:06Z, before the TREC 2022 copy existed.
  No TrialGPT generation, receipt, freeze, declaration or row on TREC 2023 existed.

## 14. What this authorizes

Section 11 only. No run, no release, no System registration, no provider call outside Section 11, and
no result admission. Each corpus build and each row still needs Mehul's decision to run. A
replacement under Section 10 step 7 runs under the decision that started its row.

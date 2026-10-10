# TAIM paper analysis protocol: TrialGPT 1.0 splits criteria with upstream's parser

**Protocol ID:** `taim-paper-analysis-protocol-2026-10-09-v16`

**Status:** Frozen on 2026-10-09, the date in its protocol ID, as the repository owner ruled.
Written on 2026-10-09 for the owner's review, and frozen the same day. These bytes are immutable;
a later change requires a new dated amendment. It is frozen after v15, and before any query
generation, retrieval freeze or System call for any row it governs (Section 7). It ships with the
release that first ships the TrialGPT 2.0 System (0.2.0), as v14 and v15 do. It authorizes no
run, no release, no provider call, no protected-data access beyond v14 Section 8, and no result
admission.

No effectiveness value prompted this amendment. It was prompted by how far TrialGPT 1.0's criteria
are from upstream TrialGPT's.

## 1. What this amendment changes

For `TrialGPT-TAIM-Luna-v1` on the four targets of protocol v5 Section 1, it changes these clauses
of v15 and nothing else:

- **v15 Section 2, first bullet, and Section 3 step 5, for what the System reads.**
  `TrialGPT-TAIM-Luna-v1` no longer reads the criterion Derived View. It reads each trial's
  complete eligibility text, cuts it into an inclusion and an exclusion section, and splits each
  with upstream TrialGPT's own `parse_criteria` (Section 3). It requires
  `complete_eligibility_text` and no longer requires `derived_view:eligibility_criteria`.
- **v15 Section 6, the caption.** Every table holding a `TrialGPT-TAIM-Luna-v1` row on these
  targets carries, in place of v15's caption: "`TrialGPT-TAIM-Luna-v1` reads each trial's complete
  eligibility text from a copy of the stratum that differs from it only in that text (protocol
  v15), and splits its criteria with upstream TrialGPT's parser (protocol v16); its topics,
  judgments and pool are the stratum's."
- **v15 Section 10, the disclosure,** gains Section 6's two items.

Every other rule stands unchanged:
- **v15, all of it except as above.** The complete-eligibility package is built exactly as v15
  Section 3 says, step 5 included. The criterion Derived View stays in the package, and no
  published System reads it. Section 4's identities, Section 5's scoring, Section 6's paired
  comparisons, Section 7's one generation per target, and Section 9's order of work are unchanged.
- **v14, v13, v12, v11, v9 Section 4, v5, v4 and v1,** as v15 Section 1 lists them.
- **The TrialGPT 2.0 System** is not touched: it never read the criterion view, and its code,
  method and identities do not change.

The System ID `TrialGPT-TAIM-Luna-v1` does not change (Section 5). Its prompts, model, retrieval,
budget and scoring do not change; only the source of its criteria does.

## 2. Why

- **Upstream TrialGPT splits criteria with `parse_criteria`** (`trialgpt_matching/TrialGPT.py:20-37`
  at the pinned commit `1b3242cd`) over NCBI's pre-separated `inclusion_criteria` and
  `exclusion_criteria` fields. Every other published System with its own criterion split uses
  that split as shipped. `TrialGPT-TAIM-Luna-v1` was the exception: it read TAIM's split-v3
  criteria, with upstream's rule only as a filter.
- **Its criteria agreed with upstream's on about 72% of trials:** 18,897 of 26,145 TREC 2021 pool
  trials (72.3%), 19,388 of 26,579 TREC 2022 (72.9%), 2,638 of 3,620 per SIGIR view (72.9%).
  "Agree" means the same criteria on the same side, ignoring list markers, case, punctuation and
  whitespace. The deviation list said "Identical output"; the function was, its input was not.
- **With this amendment they agree on about 85%:** 22,063 of 26,145 (84.4%), 22,934 of 26,579
  (86.3%) and 3,025 of 3,620 (83.6%). Every remaining difference comes from upstream's input
  data: text NCBI's fields lose or carry in another registry version, or the layout NCBI gave
  them (deviation D11).
- **Using NCBI's fields directly was rejected.** They lose text the complete text carries in 1,939
  TREC 2021 pool trials, and 901 carry a different registry version of the text. The rows would
  then read text from outside the Task Input, which v15 Section 6's paired comparisons rule out.
  It would also add two locked downloads.
- **Deriving the criteria inside the System moves no shared identity.** Changing the criteria in
  the package would move all sixteen v15 Section 4 identities and the four recipe pins, for both
  Systems.

## 3. The criterion source

For each trial, `TrialGPT-TAIM-Luna-v1`:

1. reads the trial's complete eligibility text from the package;
2. finds where split-v3 (`taim-eligibility-criteria-split-v3`) changes polarity, reading the text as
   split-v3 does, and cuts there. A section opened by a change starts at its first item, with its
   list marker, so the heading that opens it is in neither section, as NCBI's fields leave the
   phrase "exclusion criteria" out. Text before the first item stays with it. Criteria of
   unspecified polarity count as inclusion, as upstream's fields put every criterion outside an
   exclusion section in inclusion;
3. converts CRLF and CR line endings to LF, since upstream's rule splits only on blank lines;
4. splits each section with `parse_criteria`, copied byte for byte from the pinned commit and
   pinned by a test to its SHA-256; the matching prompt renders each section with it.

The method contract records this as `criterion_source`
`upstream-parse-criteria-over-split-v3-polarity-sections-v1`, and its implementation hash covers
the four functions that do it.

## 4. The identities

**Unchanged,** checked on 2026-10-09 against the four complete-eligibility packages:
- every identity of v15 Section 4 for all four targets (Prepared Snapshot, Task Input,
  Evaluation Package and pool receipt IDs, Profile, Profile definition, pool count, pool IDs and
  clinical as-of), re-derived from the packages by the release's row-declaration code with this
  change, and the four recipe SHA-256s;
- the frozen retrieval's inputs: the title, conditions and complete text of every pool trial, the
  only fields retrieval reads, are byte-identical before and after on all four targets;
- the TrialGPT 2.0 System: its code, constants, capability contract and row declaration.

**Moves,** for `TrialGPT-TAIM-Luna-v1` only: the publication method contract (it gains
`criterion_source`; `prompt_and_scoring_implementation_sha256` moves from `sha256:33112523...` to
`sha256:e1997fda...`), the publication contract ID that binds it, and each run's System Input ID,
which binds the required capabilities. None of these is fixed by any protocol, and none has been
written for any row.

## 5. The System ID and the 2026-09-03 rule

- **The System ID stays `TrialGPT-TAIM-Luna-v1`, by the owner's decision of 2026-10-09.** v15
  Section 1 said nothing in the System's method changes. This amendment changes one part of it,
  before any generation, freeze or row exists. The method contract records the change, and the
  2026-10-07 repair set the precedent of a fidelity change under the same ID.
- **The 2026-09-03 rule quoted in v15 Section 2.** That text says "Weakening the System
  requirement or substituting canonical text would change the frozen method and is prohibited".
  This amendment drops the criterion-view requirement. It substitutes no canonical text: the
  System still requires and reads the complete eligibility text, and only its criterion source
  changes, to upstream's own parser, by the owner's ruling of 2026-10-09. Hannes Ill, the
  author of that rule, agreed to this change the same day (Section 7).

## 6. Disclosure

Every reported `TrialGPT-TAIM-Luna-v1` row on these targets, and the paper, disclose, beside v15
Section 10:

1. this amendment by protocol ID, freeze date and SHA-256, and that it was written before any
   generation of these rows;
2. that the row's criteria come from upstream's `parse_criteria` over sections cut from the
   complete text, with the agreement figures of Section 2 and deviation D11 of the deviation list.

## 7. What was known when this was written

- **No effectiveness value of either TrialGPT System existed,** and none informed this amendment.
  No generation, receipt, freeze or row declaration under v14 or v15 had been made for any
  target.
- **The agreement figures** were measured on 2026-10-09 from the four packages' trial texts and
  upstream's public trial files. No judgment, qrel, ranking or model call was used.
- **Hannes Ill, the author of the 2026-09-03 rule, agreed to this change on 2026-10-09,** by
  WhatsApp message to the repository owner, who relayed it.
- **The owner's ruling of 2026-10-09,** after asking Hannes Ill for his view, verbatim as relayed:

  > Finalise it for 0.2.0: TrialGPT 1.0 uses upstream's parse_criteria on the sections TAIM
  > separates, inside the System. Prove packages, retrieval freeze and TrialGPT 2.0 identities
  > unchanged. Show me protocol v16 and the deviation entry (agreement ~85%, remaining differences
  > explained as input-data causes: NCBI field text loss, registry version differences) for my
  > freeze.

## 8. What this authorizes

This amendment defines where `TrialGPT-TAIM-Luna-v1` takes its criteria from on its four targets,
and what its rows disclose. It authorizes no run, no release, no System registration, no provider
call, no protected-data access beyond v14 Section 8, and no result admission.

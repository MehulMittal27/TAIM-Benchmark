# TAIM paper analysis protocol: a sentence-boundary fix for folded queries, and row 12

**Protocol ID:** `taim-paper-analysis-protocol-2026-11-06-v18`

**Status:** Frozen on 2026-11-06, the date in its protocol ID, on Mehul's ruling and Hannes Ill's
agreement (Section 7). It amends `taim-paper-analysis-protocol-2026-11-02-v17` in one place, its
Section 7.1, for one row, row 12 of its Section 3. Every other rule of v17 stands. These bytes are
immutable, and a later change needs a new dated amendment.

## 1. What was wrong

v17 Section 7.1 builds each staged row's folded queries with medspaCy ConText. A negation cue there
reaches the rest of its sentence: forward to the sentence end, or back to its start. Only a
terminate cue or another cue of the same category stops it sooner. Two of TAIM's five added polarity
rules cap their reach at 1 and 3 tokens; the 69 packaged negation rules have no cap
(`medspacy/context/context_modifier.py:126-191`, medspaCy 1.3.1; `taim/query_folding.py:115-177`,
Release 0.3.0). The sentence is PyRuSH's, the default splitter (`query_folding.py:481`, Release
0.3.0).

PyRuSH does not end a sentence at a line break when the line has no closing punctuation and the next
line starts with `[`. TREC 2023 topics are written one item per line, and every line starts with
`[` (346 of 346 lines). So:

- 38 of the 40 TREC 2023 topics were one sentence, and in the other 2, one sentence spanned all
  their lines;
- a single negation cue negated every term after it, on every later line;
- 188 of 313 extracted terms (60.1%) were negated, against 3.6% to 11.8% on the other strata;
- 15 of 40 topics folded to an empty query: 5, 6, 7, 8, 9, 12, 13, 15, 18, 23, 25, 27, 28, 29 and
  30.

Release 0.3.0's bundle validator refuses an empty query, so its `queries prepare` wrote no bundle for
TREC 2023, and row 12 was never declared, run or scored. v17 Section 7.1's rule, "A topic whose
folded query is empty is reported as it is, never patched", was written for a topic that states
nothing assertable. These 15 topics did state assertable items, and the splitter's sentence
boundaries removed them.

## 2. The fix

A line break always ends a sentence: a spaCy component after PyRuSH starts a sentence at the token
after any whitespace that holds a line break (`query_folding.py`, component
`taim_line_break_sentences`). Nothing else changes:

- the 102 packaged ConText rules (`sha256:5a0096af3761232288475a2b1f8cfbcc1aa2b3eb50c92b8bd12f8032cafec7c2`)
  and TAIM's five polarity rules
  (`sha256:7e0b4072a2876c6e560c114f0650712dcccd5811b8c0625800e5ff9342753c6a`) are unchanged;
- the SNOMED CT release and its lock, the lexicon, the clinical tags, the demographic patterns and
  the folding policy are unchanged;
- text without a line break splits exactly as before.

It ships in Release 0.3.1, which differs from 0.3.0 only by this change, its tests, a known-issues
entry, this amendment and the version.

**Effect on TREC 2023**, measured with the fix before this amendment by building the bundle, with no
retrieval, ranking or score:

- empty topics: 15 before, 0 after;
- negated terms: 188 of 313 (60.1%) before, 44 of 313 (14.1%) after; every remaining negated term is
  negated by a cue on its own line;
- the extracted terms are the same 313;
- folded query words per topic: 0 to 15 (mean 4.0) before, 2 to 15 (mean 8.6) after;
- 29 of 40 queries change. The 11 unchanged topics are 11, 17, 19, 20, 21, 22 and 31 to 35.

## 3. The other rows

The fix leaves every other row's inputs byte-identical: rebuilt with it, the folded-query bundles of
rows 9, 10, 11, 13 and 14 equal the ones their declarations name (bundle IDs `d6a0ff63...`,
`28ca71e2...`, `fb8ec270...`, `dc21311d...` and `6a91e0ec...`), and no other row reads a folded
query.

## 4. Row 12

Row 12 (the staged System, TREC 2023 JU, K = 1,000, NVIDIA A40) runs on **Release 0.3.1** in place of
the Section 10 release. Under v17 Section 10 step 7, a failure that needs a change to the release is
not repaired by a replacement: the row is declared on the changed release, and every attempt is
disclosed.

- Its folded-query bundle is built with Release 0.3.1's `queries prepare` and committed before its
  declaration, as v17 Section 10 step 5 states.
- Its declaration names this amendment and that bundle ID.
- It runs to completion once, under v17 Section 10 step 7.
- Its stratum identities, recipe, components, K, card and registered comparisons (v17 Sections 5,
  7.2, 7.3 and 8) are unchanged.

## 5. What was known when this was written

- No row 12 output existed. No TREC 2023 retrieval, ranking or score of the staged System or its
  components existed.
- The following had been read: the TREC 2023 counts of Section 1; the per-topic line, sentence, term
  and negated-term counts; and the counts with the fix in Section 2. Every one is a count or a topic
  ID. They came from the release's own folding code on the stratum's Task Input topics. No judgment
  was read except inside the release's own pool resolution, which `queries prepare` runs.
- The bundle comparison of Section 3 had been made. Rows 9 and 10 had run and been re-scored. Rows 11,
  13 and 14 were running (Delta jobs 23597123, 23597124 and 23597125), with no output.
- The Release 0.1.1 re-scores of rows 3, 4, 6, 9 and 10 existed, and their macro means had been read,
  including row 6's on TREC 2023. The author saw them while drafting this amendment, after its scope
  and the fix were fixed and agreed. No rule or count in it was chosen after they were seen.
- The fix was found on TREC 2023, a held-out test stratum of the staged System (v17 Section 1.3). It
  was chosen as a structural rule about where a sentence ends, before any effectiveness value of
  the staged System on TREC 2023 existed, and it changes no query outside TREC 2023.

## 6. Disclosure

Row 12, and the paper wherever it reports row 12, adds to v17 Section 12:

> **13. Row 12 folded-query fix.** Release 0.3.0's folding treated each multi-line TREC 2023 topic
> as one sentence, so one negation cue negated every later item: 15 of 40 topics folded to an empty
> query, and 60.1% of extracted terms were negated. The bundle could not be built, and row 12 never
> ran on 0.3.0. Release 0.3.1 ends a sentence at every line break, and changes nothing else. With
> it, no topic is empty, and 14.1% of terms are negated. Row 12 runs on 0.3.1 under amendment v18.
> The fix was found on TREC 2023 counts, before any effectiveness value existed. Rebuilt with it,
> the folded-query bundles of the other four strata are byte-identical to those their rows used.

Rows 9, 10, 11, 13 and 14 need no new disclosure.

## 7. Authorization

Mehul Mittal ruled on 6 Nov 2026 that this amendment covers row 12 only, with Section 3's sentence on
the other rows, that the fix ships as the patch release 0.3.1, and that 0.3.0 gets a known-issues
entry.

Hannes Ill agreed on 6 Nov 2026 to the bug fix, to the rerun of row 12 only, to reporting the fix in
the paper, to the 0.3.1 patch release and to the held-out point of Section 5.

The TREC 2023 uses are the ones v17 Section 11 already authorizes ("Building the folded-query
bundles of Section 7.1 and running the staged System's components on TREC 2022 and TREC 2023"). This
amendment adds none. It authorizes no run: row 12 still needs Mehul's decision to run.

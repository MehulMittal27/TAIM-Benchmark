# Known issues

## Folded queries on multi-line topics: a negation could reach later lines, fixed in 0.3.1

**In short.** In Releases 0.1.0 to 0.3.0, folded-query assertion
(`taim.query_folding.build_context_pipeline`) took its sentence boundaries from medspaCy's
default splitter, PyRuSH. A ConText negation cue reaches the rest of its sentence. PyRuSH does not
end a sentence at a line break when the line has no closing punctuation and the next line starts
with `[`. In a topic written one item per line in that form, a single negation cue therefore
negated every item on the lines after it. From Release 0.3.1, a line break always ends a sentence
(the pipeline component `taim_line_break_sentences`). Nothing else in folding changes.

**Who is affected.** Only topics with line breaks. Releases before 0.3.0 build folded queries for
TREC CT 2021 only.
- TREC CT 2021, TREC CT 2022 and both SIGIR 2016 views have none. Their folded-query bundles
  built with 0.3.1 are byte-identical to those built with 0.3.0.
- TREC CT 2023 topics are written one `[`-led item per line. With 0.3.0, 38 of its 40 topics were
  one sentence, 188 of 313 extracted terms (60.1%) were negated, and 15 topics folded to an empty
  query. `queries prepare` then refuses to write the bundle ("folded query rows require non-empty
  topic_id and canonical_text"), so no TREC CT 2023 folded-query bundle can be built with 0.3.0.
- With 0.3.1, no TREC CT 2023 topic is empty, and 44 of 313 terms (14.1%) are negated.

**What to do.** Build TREC CT 2023 folded-query bundles with 0.3.1 or later. Bundles for the other
Tracks need no rebuild.

**What stays the same.** The ConText rules and TAIM's polarity rules (their digests are unchanged),
the SNOMED CT lexicon lock and the folding policy. The bundle's provenance records none of the
pipeline's components, so a 0.3.1 bundle differs from a 0.3.0 one only where its queries differ.

## nltk 3.10.3: CVE-2026-81726 (GHSA-8mgp-746c-j5xp), fixed in 0.3.0

**In short.** From Release 0.3.0, `uv.lock` pins nltk 3.10.4, the first version the advisory lists
as patched (published 2026-10-21). Releases 0.1.0 to 0.2.0 pin nltk 3.10.3, which the advisory
covers: it covers every nltk up to and including 3.10.3. When this was first checked on
2026-10-08, no fixed nltk existed.

nltk is installed only with the optional `folding` extra, and only as an indirect dependency. This
package never calls the six affected functions, so they cannot be reached from it, with either
version. The evidence follows.

### Who has nltk installed

- **Only `folding` installs it.** nltk is installed only by `uv sync --locked --extra folding`, on
  CPython below 3.13. No other extra and no base install includes it.
- **The chain is indirect.** The `folding` extra pins medspacy 1.3.1. medspacy requires
  medspacy-quickumls 3.2, and medspacy-quickumls requires nltk.
- **No direct dependency.** This package declares no dependency on nltk.

### What the advisory covers

The advisory ([GHSA-8mgp-746c-j5xp](https://github.com/advisories/GHSA-8mgp-746c-j5xp), rated high)
concerns six of nltk's model-file functions:
- `TransitionParser.train` and `TransitionParser.parse`;
- `AveragedPerceptron.save` and `AveragedPerceptron.load`;
- `PerceptronTagger.save_to_json`;
- `save_maxent_params`.

These functions open caller-supplied paths with plain file APIs. They therefore bypass nltk's
opt-in path sandbox (`nltk.pathsec`). An application at risk is one that turns that sandbox on and
lets an untrusted caller choose a model path.

### Why the affected functions are not reachable

**The call path.** Line numbers are those of the files in this repository and of the locked
medspacy 1.3.1, medspacy-quickumls 3.2 and nltk 3.10.3.

- The package builds folded queries in one place. `trial-benchmark patient-to-trial queries
  prepare` is handled by `_prepare_folded_queries` (`src/taim/pipelines/bm25/folded_extension.py:49`).
  That handler calls `build_folded_query_bundle` at `:130`, which is defined at
  `src/taim/folded_query_bundle.py:138`.
- The package imports medspacy in one place too: `build_context_pipeline`
  (`src/taim/query_folding.py:466`).
  - It runs `import medspacy` (`:478`).
  - It runs `medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_context"])` (`:481`).
    These are the sentence splitter and the ConText assertion component.
  - `build_extraction` (`:604`) then reaches `assert_terms` (`:512`), which runs the pipeline with
    `nlp(text)` (`:518`).
- **How nltk gets imported.**
  - `medspacy/__init__.py:4` imports `medspacy/util.py`.
  - Line 14 of `medspacy/util.py` runs `from quickumls import spacy_component`.
  - `quickumls/__init__.py:1` imports `quickumls/core.py`.
  - Line 13 of `quickumls/core.py` runs `import nltk`.
- **When quickumls calls nltk.** Its only nltk calls are `nltk.corpus.stopwords` and
  `nltk.download`, at `quickumls/core.py:109-126`. Both sit in `QuickUMLS.__init__`. That
  constructor runs only when the `medspacy_quickumls` component is enabled, and folding does not
  enable it.
- **Nothing else calls the six functions.** A search of the installed `folding` environment,
  outside `nltk/` itself, finds none of the six names. The only other modules that import nltk are
  the top-level `benchmarks/` scripts of pysbd's wheel, which nothing imports.
- **The System that scores folded queries does not load nltk.** Its run,
  `ReleaseFoldedBM25System.run` (`src/taim/pipelines/bm25/folded.py:29`), imports only the BM25
  retriever and `validate_folded_query_bundle`. Importing its modules leaves `nltk`, `medspacy` and
  `quickumls` out of `sys.modules`.

**The runtime check, 2026-10-08.**

The setup:
- CPython 3.12.13 on macOS (arm64), in a fresh environment installed from `uv.lock` with the
  `folding` extra: nltk 3.10.3, medspacy 1.3.1, medspacy-quickumls 3.2 and spaCy 3.8.16.
- The run called `build_folded_query_bundle` on two sets of topics:
  - the three topics of the `trec-mini` fixture;
  - three sentences that exercise ConText, with affirmed, negated and historical mentions.
- The SNOMED CT input was a synthetic RF2-format release of seven invented concept IDs. It holds
  no SNOMED CT content.
- **What was recorded:**
  - a profile hook recorded every Python call into nltk;
  - an audit hook recorded every file open and refused every network call.

The results:

| Phase | Calls into nltk | Calls to the six functions | Files opened under `nltk_data` |
| --- | ---: | ---: | ---: |
| `import medspacy` | 1,604 | 0 | 0 |
| Building the folded queries | 0 | 0 | 0 |

- **What nltk read.** Apart from its own compiled modules, nltk read only its `VERSION` file.
- **The one refused network call** was urllib3's import-time IPv6 probe, a loopback bind at
  `urllib3/util/connection.py:127`. Nothing tried to connect anywhere.
- **The hook works.** As a positive control, `AveragedPerceptron.save` was called directly after
  the run, and the hook recorded it.

**The same check on nltk 3.10.4, 2026-11-03.** CPython 3.12.13 on macOS (arm64), in a fresh
environment installed from Release 0.3.0's `uv.lock` with the `folding` extra: nltk 3.10.4,
medspacy 1.3.1, medspacy-quickumls 3.2 and spaCy 3.8.16. The topics were the three topics of the
TREC 2021 synthetic fixture and three sentences with affirmed, negated and historical mentions. The
SNOMED CT input was a synthetic RF2-format release of nine invented concept IDs.

| Phase | Calls into nltk | Calls to the six functions | Files opened under `nltk_data` |
| --- | ---: | ---: | ---: |
| `import medspacy` | 3,931 | 0 | 0 |
| Building the folded queries | 0 | 0 | 0 |

nltk again read only its `VERSION` file, and the one refused network call was the same urllib3
probe. The positive control was recorded too, and nltk 3.10.4 itself refused it:
`AveragedPerceptron.save` to a path outside nltk's allowed roots raises `PermissionError`.

**Earlier releases.** The same `import medspacy` and `medspacy.load` lines are in
`src/taim/query_folding.py` at the tags `v0.1.0`, `v0.1.1` and `v0.1.2`. The `uv.lock` of each tag
pins nltk 3.10.3.

### Why nltk is not simply removed

Folding cannot drop nltk on its own:
- `import medspacy` imports quickumls when it loads, and quickumls imports nltk when it loads, so
  removing nltk breaks `import medspacy`.
- Removing nltk would mean a different medspacy or a different assertion library. Either would
  change the assertion step whose medspaCy version and rules every folded query bundle records.

Neither change is needed while the six functions are unreachable.

### The fixed nltk

nltk 3.10.4, published on 2026-10-21, is the advisory's first patched version. Release 0.3.0's
`uv.lock` pins it, and nltk is the only package that changed in the lock for it. The six functions
stay unreachable from this package, as the check above shows.

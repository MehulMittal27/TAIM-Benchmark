# Score your own pipeline

This guide is for a matching pipeline this release does not ship. You do not add it to the
repository and your pipeline never imports TAIM. The release exports the exact input its own
Systems receive. Your pipeline ranks that input wherever it runs and writes two files. The release
then checks the ranking and scores it against judgments your pipeline never sees.

Every command below runs from a clone of this repository, starting from nothing:

```console
uv sync --locked
. .venv/bin/activate
```

## 1. Export a task

Start with the packaged synthetic fixture, which needs no downloads:

```console
trial-benchmark patient-to-trial submission export --fixture \
  --output submission/task \
  --evaluator-output submission/evaluator
```

The export writes two directories, and they are kept apart on purpose:

- `submission/task/` holds only `input.json`. It carries no judgments. This is the only file your
  pipeline reads.
- `submission/evaluator/` holds only `evaluation-package.json`, the judgments. Your pipeline never
  needs it, so do not give it access. Import and validation refuse a task directory that holds
  anything besides `input.json`.

The command prints the task's `task_input_id` and `evaluation_package_id`. They identify exactly
what your ranking will be scored on.

## 2. Run your pipeline

`examples/external-pipeline/pipeline.py` is a complete pipeline in about 60 lines of standard-library
Python. It scores candidates by the words they share with the query. Run it on the exported input:

```console
python examples/external-pipeline/pipeline.py submission/task/input.json submission/predictions
```

It writes `submission/predictions/ranking.jsonl` and `submission/predictions/pipeline.json`. To
plug in your own method, write those same two files. The rules follow.

**What `input.json` contains.** `task` is `patient_to_trial` or `trial_to_patient`. `top_k` is the
depth you must rank to. `input.task_input_id` is the identity you copy into every ranking row. A
`patient_to_trial` input lists the patients in `input.query_patient_versions` and the candidate
trials in `input.trial_corpus_versions`. A `trial_to_patient` input lists the query trials in
`input.trial_versions` and the candidate patients in `input.patient_versions`. Each entry carries
its `patient_id` or `trial_id`. Patient text is `topic.canonical_text` and trial text is
`trial.canonical_text`, and both records also keep their structured fields.

**`ranking.jsonl`** has one JSON object per line, one line per ranked pair, with exactly these
fields:

```json
{"task": "patient_to_trial", "task_input_id": "sha256:<from input.json>", "system_id": "external:my-pipeline", "topic_id": "<patient ID>", "trial_id": "<trial ID>", "rank": 1, "score": 0.9}
```

A `trial_to_patient` row has `trial_id` for the query and `patient_id` for the candidate instead.
Every query needs ranks 1, 2, 3 and so on with no gaps, down to `top_k`, or to the number of
candidates when there are fewer. Each candidate appears at most once per query. Scores must be
finite numbers. Every ID must come from `input.json`. A missing row makes the whole submission
fail: TAIM never counts missing output as a wrong answer.

**`pipeline.json`** declares your System, with exactly these fields:

- `system_id`: `external:` followed by a lowercase name, the same one your ranking rows carry.
  This declaration is the whole registration. There is nothing to add to TAIM.
- `source_revision`: a version or commit of your code.
- `source_sha256` and `dependency_lock_sha256`: `sha256:` and the hex digest of your source and
  of your dependency lock. The example computes both for itself.
- `configuration`: an object recording whatever someone needs to reproduce your run, such as
  models and their revisions, seeds, tie policy and runtime.

TAIM checks the declaration's shape. It cannot check that the ranking came from the code you
declared: that needs your source and lock in the hands of whoever reviews the result.

## 3. Import and validate

```console
trial-benchmark patient-to-trial submission import \
  --task-dir submission/task \
  --evaluator-dir submission/evaluator \
  --ranking submission/predictions/ranking.jsonl \
  --pipeline submission/predictions/pipeline.json \
  --output submission/result
trial-benchmark patient-to-trial submission validate \
  --task-dir submission/task \
  --evaluator-dir submission/evaluator \
  --submission-dir submission/result
```

Import refuses an incomplete, duplicated or foreign ranking before it writes anything. It writes
`submission/result/` with your ranking and declaration, `metrics.json` and a checksummed
`submission.json`. The validate command recomputes every checksum and every metric, so an edited
file fails it. `submission.json` records `paper_matrix_admitted: false`: a submission is your
result, not a row of TAIM's paper.

## 4. Compare with a reference System

A score means something next to another System's score on the same task. Run the reference `bm25`
System on the same input and score it:

```console
trial-benchmark patient-to-trial fixture run \
  --run-id reference-bm25 \
  --output-dir submission/runs
trial-benchmark patient-to-trial run evaluate \
  --run-dir submission/runs/reference-bm25 > submission/reference-scorecard.json
python -c "import json; r = json.load(open('submission/reference-scorecard.json')); s = json.load(open('submission/result/submission.json')); m = json.load(open('submission/result/metrics.json')); print('same task:', all(r[k] == s[k] for k in ('task_input_id', 'evaluation_package_id'))); print('bm25 nDCG@10:', r['metrics_by_cutoff']['10']['aggregate']['ndcg_at_10']); print('yours nDCG@10:', m['10']['aggregate']['ndcg_at_10'])"
```

Two results are comparable when their `task_input_id` and `evaluation_package_id` are equal. Then
your `metrics.json` has the same shape as the reference scorecard's `metrics_by_cutoff`, computed
by the same code: a submission of the `bm25` ranking itself reproduces the `bm25` scorecard
exactly. A result on a different Task Input is a different experiment, however similar it looks.

Compare with a reference score the README publishes when both ids match; else produce the reference
row yourself, as above. Do not compare against numbers quoted in the dated protocol documents: they
were measured on an earlier prepared-corpus recipe that is not interchangeable with this release's.

The fixture's records are invented. Its scores show that the path works, not how good a method is.

## 5. The other direction

The `trial-to-patient` commands work the same way, and the example handles both directions:

```console
trial-benchmark trial-to-patient submission export --fixture \
  --top-k 20 \
  --output submission-reverse/task \
  --evaluator-output submission-reverse/evaluator
python examples/external-pipeline/pipeline.py submission-reverse/task/input.json submission-reverse/predictions
trial-benchmark trial-to-patient submission import \
  --task-dir submission-reverse/task \
  --evaluator-dir submission-reverse/evaluator \
  --ranking submission-reverse/predictions/ranking.jsonl \
  --pipeline submission-reverse/predictions/pipeline.json \
  --output submission-reverse/result
trial-benchmark trial-to-patient submission validate \
  --task-dir submission-reverse/task \
  --evaluator-dir submission-reverse/evaluator \
  --submission-dir submission-reverse/result
```

A reverse `run evaluate` scorecard is laid out differently from a reverse `metrics.json`, and
names the same measures differently: its `aggregate.mean.ndcg_at_5` is the submission's
`5.aggregate.ndcg_at_5`, and its `eligible_recall_at_5` and `relevant_or_eligible_recall_at_5` are
the submission's `5.aggregate.recall.eligible_at_5` and
`5.aggregate.recall.relevant_or_eligible_at_5`.

## 6. Real data

A real export takes the same flags as the matching `benchmark run` in the
[README](../README.md#real-tracks): the prepared data, the Profile, its pool pins, the protocol
identity where that Profile requires one, the release manifest and wheel from this release's page
under the repository's GitHub Releases, and the `uv.lock` at the root of this repository. Prepare
the Track and inspect its pool as the README describes, then export. The pool inspection's receipt
gives the three pool values. For the TREC 2021 judgment-union Profile, the protocol identity is the
digest of the v3 paper protocol that ships in `docs/`:

```console
PROTOCOL_ID="sha256:$(shasum -a 256 docs/paper-analysis-protocol-2026-08-31-v3.md | awk '{print $1}')"
trial-benchmark patient-to-trial submission export \
  --track trec-ct-2021 \
  --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --pool-count 26162 \
  --pool-ids-sha256 sha256:<digest-from-receipt> \
  --pool-receipt-id sha256:<id-from-receipt> \
  --protocol-approval-id "$PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.2.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --output submission-2021/task \
  --evaluator-output submission-2021/evaluator
```

Measured on an Apple M5 Pro laptop with 24 GB of memory, this export took 14 minutes, most of it
loading and checking the prepared package, and peaked at 3.7 GB resident. It wrote a 618 MB
`input.json` holding the 75 patients and 26,162 trials, and a 4 MB `evaluation-package.json`. The
example pipeline ranked it in 10 seconds. Import and validation each took under 2 minutes at about
3 GB.

Your reference row is the README's `bm25` command on the same Profile with the same pins. Its
scorecard carries the same `task_input_id` and `evaluation_package_id` as your export, and
section 4's comparison applies unchanged.

Every real export requires the manifest, the wheel and `uv.lock`, which bind the result to this
release. The export refuses a missing or wrong protocol identity before it opens any prepared file.
The only real reverse Profile is `trec-ct-2021-reverse-complete10`. An exported `input.json`
contains patient and trial text from the collection, so it stays under the collection's terms: do
not commit or upload it.

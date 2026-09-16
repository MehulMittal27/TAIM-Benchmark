# Release verification

The release owner runs these gates from a clean generated candidate. CI runs the same lint, type,
test, package, fixture, evaluation, bundle, and Python-matrix checks.

```console
uv sync --locked --extra dev
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
uv build
```

```console
uv sync --locked --extra dev --extra folding
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

```console
uv sync --locked --extra dev --extra staged
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

`uv run --locked --no-sync taim doctor` is a private TAIM source-tree pre-freeze gate. The public
candidate deliberately ships only `trial-benchmark`; its release validator replaces internal
external-checkout diagnostics with closed-tree, import, license, fixture, and package checks.

Install the generated source distribution and wheel into separate empty environments. In each
environment, execute the [documented quick start](../README.md#quick-start). The public test suite
also builds and semantically validates a synthetic Published Result Bundle.

Before package tests, compare `trial-benchmark support list` with the manifest. Verify all four
connector parser smokes from synthetic inputs, both direction-specific quick starts,
mixed-direction rejections, public import closure, protected-source exclusion, dependency and
license closure, deterministic regeneration, frozen full-preparation record counts, producer
identity revalidation, and rejection of RRF components from dirty or unrelated release producers.
Verify all five judgment-union Profile definitions, source-Profile mappings, pool inspection
receipts, missing or mismatched count/digest rejection, the fixed TREC 2021 count of 26,162, and
content-addressed receipt-ID validation. Confirm separation of full-corpus and within-pool evidence
scopes. These are structural and synthetic checks; do not open protected Track bytes during release
verification.
Verify the external source-lock checksum and synthetic ID extraction, the 26,149-count gate,
separation from the 26,162 judgment union, exclusion of external repository directories, and exact
license closure over every model and runtime input the release binds.
For TrialGPT, verify the fresh retrieval-lock requirement.
For TrialMatchAI, verify normalization and the source-lock checks.
Do not run a release-selected external System.
Verify the distinct instructed and no-template Qwen policies.
Also verify the exact staged declaration, depth-5,000 component validation, rendered-input-bound
reranker scores, batch-aligned checkpoint rejection, named Stage Ranking round trips, explicit
stage evaluation, and rejection of staged components from dirty or unrelated release producers.
These checks use declarations and synthetic artifacts only and do not run the reranker.
Release verification does not inspect real effectiveness outcomes.
For bundle-eligible run tests, provide the exact manifest-recorded `uv.lock` with
`--dependency-lock` and verify that its content-addressed installed-environment projection survives
the Published Result Bundle round trip. The initial manifest must retain
`published_result_bundles: []`. Also verify that a Result Bundle is refused when any published
string is itself an absolute filesystem path.
A TrialMatchAI L4 System Input carries no filesystem location,
so its bundles meet that refusal
like any other bundle.

Finally, validate the exact candidate and execute the manifest's quick-start argv from the installed
package:

```console
trial-benchmark release validate \
  --candidate /path/to/generated-candidate \
  --manifest /path/to/frozen-release-manifest.json \
  --runtime
```

The frozen release manifest is a reviewed release input held outside the generated tree to avoid a
self-referential tree hash. Publish it beside the release so another auditor can repeat validation.
Do not create or push a public remote until these gates pass and the candidate receives final human
approval.

`trial-benchmark release validate` also checks the release commit: its tracked tree, its author,
committer and timestamp, and the parent the manifest names, which is none for the first release. Any
checkout of that commit validates: the generated candidate, a plain clone, a clone of the release tag
and a shallow clone. Paths the release's own `.gitignore` names, such as the `.venv/` that
`uv sync` creates, are not part of the release tree. The validator does not check how a clone is
wired or which other refs it holds.

A release is published with explicit release refspecs only: `refs/heads/main:refs/heads/main` and
`refs/tags/<tag>:refs/tags/<tag>`. Never a mirror, `--all`, `--tags`, a bare `git push`, or any
forced push.

**Guarantee:** deny-by-default makes leaking a FILE impossible.

**It does not cover:**
1. Content inside a named file. That takes a separate content rule.
2. Re-verification on a published release. The excluded-reference scan is a private build-time gate - it searches for held-back pipeline names, so it cannot ship publicly without publishing them.
3. Any push shape other than explicit release refspecs. The public release check does not see the history a push carries, so a mirror push publishes whatever the repository holds.
4. Release assets and release notes. They live outside the release tree, which is all the deny-by-default walk sees. Assets are verified against what was built and uploaded; release notes remain editable after publication and are not covered by any check.

**Worked example of limit 1:** the frozen protocol `docs/paper-analysis-protocol-2026-08-29.md` ships byte-unchanged, as the external-fidelity rows require, and its section 4.3 names Systems the protocol did not admit and this release does not carry. The release's reference checks search for System identifiers, not for such names, and the protocol is a dated record that is not edited.

Candidate construction requires the six human-owned approved inputs. A missing approval must stop
freeze; validators must never accept placeholders or fabricate an approval value.

The generated candidate must include the complete normative protocol chain as
`docs/paper-analysis-protocol-2026-08-29.md`,
`docs/paper-analysis-protocol-2026-08-31-v2.md`,
`docs/paper-analysis-protocol-2026-08-31-v3.md`, and
`docs/paper-analysis-protocol-2026-08-31-v4.md`.
For TrialGPT, the candidate also includes the frozen producer amendment as
`docs/paper-analysis-protocol-2026-09-02-v5.md`, its low-memory execution amendment as
`docs/paper-analysis-protocol-2026-09-02-v6.md`, and the single-release amendment as
`docs/paper-analysis-protocol-2026-09-02-v7.md`.
External runs require explicitly approved v4 bytes.
TrialGPT retrieval production additionally requires the exact v7 bytes.
Drafts are not
sufficient. The release must also include the approved v3 supersede record as
`docs/paper-analysis-protocol-2026-08-31-v3-supersede-note-2026-09-01.md`. Hash the exact bytes and
verify that the same `sha256:<digest>` is supplied to every pool inspection and protected-Track run.
Pool receipts are frozen later from authorized local preparations; they are not fabricated as part
of release construction.

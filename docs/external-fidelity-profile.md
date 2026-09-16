# TREC 2021 external-fidelity Profile

## Acquire and verify the 26,149-trial membership

Download `trec_2021_corpus.jsonl` from the official TrialGPT distribution URL recorded in
`src/taim/data/locks/trialgpt-trec-2021-external-fidelity.json`. Do not commit it. Prepare an
ID-only local membership file:

```console
trial-benchmark data external-pool prepare \
  --profile trec-ct-2021-external-fidelity-26149 \
  --source /path/to/trec_2021_corpus.jsonl \
  --output prepared/trec-ct-2021-external-fidelity-26149-ids.json
```

The command verifies byte size `137406767`, source SHA-256
`01692c847b2da798c57a8e0a74273ec262a7e42ad3f02b4ff5a87a6442462f9c`, 26,149 unique `_id`
values, and sorted-ID SHA-256
`fed85fadb5a0e0a42a39ed9cf65984926a13f66d4a2a0a1deb100996ac56ffa7`. It writes only trial IDs;
it never copies trial text into the repository.

After preparing TREC 2021 with the Release 0.1 recipe, inspect the effective Task Input. Use the
digest of the human-approved v4 protocol, not the earlier v3 protocol:

```console
EXTERNAL_PROTOCOL_ID="sha256:$(shasum -a 256 docs/paper-analysis-protocol-2026-08-31-v4.md | awk '{print $1}')"

trial-benchmark data pool inspect \
  --track trec-ct-2021 \
  --profile trec-ct-2021-external-fidelity-26149 \
  --data-dir prepared/trec-ct-2021 \
  --pool-file prepared/trec-ct-2021-external-fidelity-26149-ids.json \
  --protocol-approval-id "$EXTERNAL_PROTOCOL_ID" \
  > prepared/trec-ct-2021-external-fidelity-26149-receipt.json
```

The inspection verifies that every supplied ID exists in the recipe-v3 prepared corpus and the
Evaluation Package. It then orders the IDs by the prepared corpus and emits the exact count,
ordered-ID SHA-256, and content-addressed receipt ID. Count equality alone is not identity.
The Profile derives a separate Evaluation Package containing only source judgment rows for trials
inside that Task Input and records the complete source Evaluation Package ID in its provenance.
It does not turn absent pairs into negatives; they remain unjudged.

The repository owner froze the projected v4 file before outcome inspection. The runtime accepts
only its exact digest. This protocol approval does not authorize provider calls, protected-Track
access, model or terminology terms, result admission, or publication; those gates remain separate.

## Run TAIM reference Systems

Pass the same `--pool-file`, receipt values, external protocol ID, immutable release manifest,
package artifact, and dependency lock to every row. For example:

```console
trial-benchmark patient-to-trial benchmark run \
  --track trec-ct-2021 \
  --profile trec-ct-2021-external-fidelity-26149 \
  --data-dir prepared/trec-ct-2021 \
  --clinical-as-of 2021-04-27T00:00:00Z \
  --pool-file prepared/trec-ct-2021-external-fidelity-26149-ids.json \
  --pool-count 26149 \
  --pool-ids-sha256 sha256:<ordered-digest-from-receipt> \
  --pool-receipt-id sha256:<receipt-id> \
  --protocol-approval-id "$EXTERNAL_PROTOCOL_ID" \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --system bm25 \
  --run-id external-fidelity-bm25 \
  --output-dir runs
```

The other reference Systems' commands use the same System-specific options documented in the main
README. Their effective Task Input must be identical to the BM25 row.

# TrialGPT-TAIM-Luna-v1 external baseline

This System is a provider-backed TAIM adaptation, not an exact reproduction of the upstream GPT-4
paper System. The release does not include an upstream checkout, provider credentials, generated
outputs, a retrieval index, or the historical recipe-v1 retrieval artifact.

Execution requires a fresh 2,000-deep frozen retrieval artifact and matching lock created against
the amended immutable release, followed by a clean-commit publication preflight. The lock binds
the Track, exact topic set, complete Task Input trial set, and top-500 hashes. The release will not
accept historical campaign artifacts or a retrieval artifact from another Track. The paper
execution freezes 48 generation workers. After that preflight, add these options to the common
command:

The release ships the producer commands and their complete instructions in
`docs/trialgpt-retrieval-producer.md`. Producer locks require the exact v7 digest
`sha256:a617a46c2d61e76ad25963e709cad2c3ff99d4be466f173b8add02f1754927ef`. The primary
System keeps 500 candidates and permits up to 1,500 logical provider calls per topic; a 500-call
configuration would be a separately named ablation.

```console
  --system TrialGPT-TAIM-Luna-v1 \
  --trialgpt-frozen-retrieval /path/to/fresh-retrieval.jsonl \
  --trialgpt-frozen-retrieval-lock /path/to/fresh-retrieval-lock.json \
  --trialgpt-publication-contract /path/to/all75-contract.json \
  --trialgpt-workspace /path/to/empty-or-in-place-resume-workspace \
  --trialgpt-generation-workers 48 \
  --top-k 10
```

Provider calls require separate explicit authorization. A timeout, provider failure, or missing
output is not a clinical judgment and cannot be imputed into the ranking.

The SIGIR rows use all declared topics under their respective TAIM query contracts. They are not
reproductions of TrialGPT's 58-patient subset. Do not use this System's retrieved top 500 as a
neutral corpus because that set depends on System scores.

## Preflight

For TrialGPT preflight, select the same Track and Profile used by the run:

```console
trial-benchmark patient-to-trial trialgpt preflight \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --data-dir prepared/trec-ct-2022 \
  --clinical-as-of 2022-01-01T00:00:00Z \
  --pool-count <count-from-receipt> \
  --pool-ids-sha256 sha256:<ordered-digest-from-receipt> \
  --pool-receipt-id sha256:<receipt-id> \
  --protocol-approval-id "$EXTERNAL_PROTOCOL_ID" \
  --frozen-retrieval /path/to/track-specific-retrieval.jsonl \
  --frozen-retrieval-lock /path/to/track-specific-retrieval-lock.json \
  --generation-workers 48 \
  --release-manifest /path/to/benchmark-release-manifest.json \
  --package-artifact /path/to/taim_benchmark-0.1.0-py3-none-any.whl \
  --dependency-lock uv.lock \
  --output /path/to/trialgpt-contract.json
```

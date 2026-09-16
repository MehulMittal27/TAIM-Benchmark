# SIGIR Clinical Trials 2016 acquisition

Acquire CSIRO Clinical Trial Data 2016 collection version 4, DOI
`10.4225/08/58e2e83d92c2b`, using the per-file URLs recorded in
`src/taim/data/locks/sigir-ct-2016.json`. Keep all eight source files outside Git. The lock records
their filenames, byte sizes, SHA-256 values, access notes, and the collection's stated license. The
public CLI always uses this packaged lock and does not expose a source-lock override.

Choose the patient query representation explicitly:

```console
trial-benchmark data prepare \
  --track sigir-ct-2016 \
  --query-profile description \
  --source /path/to/sigir-2016-files \
  --output-dir prepared/sigir-ct-2016-description

trial-benchmark data prepare \
  --track sigir-ct-2016 \
  --query-profile summary \
  --source /path/to/sigir-2016-files \
  --output-dir prepared/sigir-ct-2016-summary
```

Preparation verifies all eight files before parsing. `description` and `summary` produce distinct
Snapshot identities and must be named by the matching run Profile. SIGIR labels describe referral
decisions: 0 would not refer, 1 would consider referral after further investigation, and 2 is highly
likely to refer. They are not TREC eligibility labels. Unjudged pooled pairs stay unjudged.

The paper uses a separate score-blind judgment union for each query view. Inspect and freeze the two
receipts independently before scoring:

```console
trial-benchmark data pool inspect \
  --track sigir-ct-2016 \
  --profile sigir-ct-2016-description-judgment-union \
  --data-dir prepared/sigir-ct-2016-description \
  --protocol-approval-id sha256:<protocol-v3-digest>

trial-benchmark data pool inspect \
  --track sigir-ct-2016 \
  --profile sigir-ct-2016-summary-judgment-union \
  --data-dir prepared/sigir-ct-2016-summary \
  --protocol-approval-id sha256:<protocol-v3-digest>
```

Do not guess either count, digest, or receipt ID. Freeze all three values for each query view. The
`description` and `summary` complete-corpus Profiles remain separate public capabilities.
Judgment-union results are within-pool rankings and retain the SIGIR referral semantics.

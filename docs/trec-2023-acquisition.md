# TREC Clinical Trials 2023 acquisition

Release 0.1 ships connector code and checksum metadata, not source bytes. Obtain the official files
from the URLs in `src/taim/data/locks/trec-ct-2023.json` and keep them outside Git. Metadata access
does not imply permission to redistribute the files.

The lock requires `topics2023.xml` (19,307 bytes,
`12987f6311f518fa64abe41ada207a442e4deb72e7926f5185b2be82e29ac122`), `qrels2023.txt`
(636,340 bytes, `4f0cd4ad573e37d08002fb401bb7ab22a643537753880d9427c96119696582a3`), and the six
`ClinicalTrials.2023-05-08.trials0.zip` through `trials5.zip` archives listed in the lock.
Preparation checks every filename, byte count, and SHA-256 before reading a source file. The public
CLI always uses this packaged lock and does not expose a source-lock override.

```console
trial-benchmark data prepare \
  --track trec-ct-2023 \
  --protocol-approval-id sha256:<approved-protocol-digest> \
  --source /path/to/trec-2023-files \
  --output-dir prepared/trec-ct-2023
```

The connector freezes questionnaire rendering as one template line followed by non-empty source
fields in source order. Trial-ID repair accepts only aliases authored in the source record. TREC
2023 is a confirmation Track; a separate approved protocol is required before topics or qrels are
accessed for effectiveness work.

After authorized preparation, derive the paper Profile's exact pool receipt before scoring:

```console
trial-benchmark data pool inspect \
  --track trec-ct-2023 \
  --profile trec-ct-2023-judgment-union \
  --data-dir prepared/trec-ct-2023 \
  --protocol-approval-id sha256:<protocol-v3-digest>
```

Do not guess the count, digest, or receipt ID. Freeze the emitted values and pass all three to every
run. This is a within-pool Profile; `confirmation-full` remains the separate complete-corpus
capability.

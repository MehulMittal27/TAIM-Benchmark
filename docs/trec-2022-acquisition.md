# TREC Clinical Trials 2022 acquisition

Release 0.1 ships connector code and checksum metadata, not source bytes. Obtain the official files
from the URLs in `src/taim/data/locks/trec-ct-2022.json` and keep them outside Git. Metadata access
does not imply permission to redistribute the files.

The lock requires `topics2022.xml` (32,423 bytes,
`c5d37709ba14f6cb341b0bea35a7f43bd1cf93647f939659667975229a7abe91`), `qrels2022.txt`
(666,030 bytes, `e569a531489e03f7b1fab03fe169c8ea66f4a59e8180fa9858b1a6e4bdcb0c5c`), and the five
`ClinicalTrials.2021-04-27.part1.zip` through `part5.zip` archives listed in the lock. Preparation
checks every filename, byte count, and SHA-256 before reading a source file. The public CLI always
uses this packaged lock and does not expose a source-lock override.

```console
trial-benchmark data prepare \
  --track trec-ct-2022 \
  --protocol-approval-id sha256:<approved-protocol-digest> \
  --source /path/to/trec-2022-files \
  --output-dir prepared/trec-ct-2022
```

TREC 2022 is a confirmation Track. Do not access its topics or qrels for effectiveness work without
a separately approved, frozen analysis protocol. Preparation is local, and source and prepared
directories remain outside version control.

After authorized preparation, derive the paper Profile's exact pool receipt before scoring:

```console
trial-benchmark data pool inspect \
  --track trec-ct-2022 \
  --profile trec-ct-2022-judgment-union \
  --data-dir prepared/trec-ct-2022 \
  --protocol-approval-id sha256:<protocol-v3-digest>
```

Do not guess the count, digest, or receipt ID. Freeze the emitted values and pass all three to every
run. This is a within-pool Profile; `confirmation-full` remains the separate complete-corpus
capability.

# TREC Clinical Trials 2021 acquisition

The repository does not download or redistribute official benchmark data. Obtain each file from
its source, place the files together in a local directory, and verify the exact byte size and
SHA-256 before preparation.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `topics2021.xml` | 64,618 | `94bda921ce7c40a0353f251abb2ea938c77331759a9f83a36abd145ab5840aca` |
| `qrels2021.txt` | 676,496 | `ba7a2cddc90285e75cd76adcd483394a6c9bacf7017113222058ba6537e6d8ac` |
| `ClinicalTrials.2021-04-27.part1.zip` | 382,792,518 | `4caa9579290adaa974efae6f5be170c8964c0fb5d110c1bdb935b331d0bf3ec2` |
| `ClinicalTrials.2021-04-27.part2.zip` | 378,478,271 | `6f4800b0c9e57af5bcef039c2d0661b2c5e844bb5025f959d7a940c03c2d902d` |
| `ClinicalTrials.2021-04-27.part3.zip` | 375,998,752 | `1190a5bd629d011dc7011b42b92f94cb7ec0ec21522bafb7701aaee826a42fce` |
| `ClinicalTrials.2021-04-27.part4.zip` | 360,825,058 | `d10ecad6993dea01e9f581e0104cdff6cd8120d0bc851de87d088cf799a7dd98` |
| `ClinicalTrials.2021-04-27.part5.zip` | 296,625,845 | `17c773184de8fea9cfcac5cd0cca45c816f4c4ac7568d35d7b0873f63bfc1602` |

The official locations are recorded in `src/taim/data/locks/trec-ct-2021.json`. The lock also records
the acquisition and redistribution notes used by the preparer. Preparation fails if a filename,
size, or digest differs. The public CLI always uses this packaged lock; it intentionally has no
source-lock override for real release preparation or execution.

```console
uv run trial-benchmark data prepare \
  --track trec-ct-2021 \
  --source /path/to/official-files \
  --output-dir prepared/trec-ct-2021
```

Keep the source and prepared directories outside version control. The prepared output binds the
source lock, preparation recipe, Benchmark Snapshot, and separate Evaluation Package.

The complete preparation supports `official-full`. For the paper's compute-bounded Profile, hash
`docs/paper-analysis-protocol-2026-08-31-v3.md`, then inspect the score-blind judgment union before any System
run:

```console
trial-benchmark data pool inspect \
  --track trec-ct-2021 \
  --profile trec-ct-2021-judgment-union \
  --data-dir prepared/trec-ct-2021 \
  --protocol-approval-id sha256:<protocol-v3-digest>
```

The receipt must contain 26,162 trials, an ordered-ID SHA-256, and a content-addressed receipt ID.
Freeze all three values and supply them to every run. This Profile is within-pool ranking, not
full-corpus retrieval, and is distinct from the historical 26,149-trial TrialGPT common pool.

# TAIM paper analysis protocol: the current rules

**Status:** Current. This is the one rules document that is not frozen. It restates no rule: it
names the frozen documents in force, by protocol ID and SHA-256, and says what each governs,
summarised from the status lines of the frozen documents. Where this summary and a frozen document
differ, the frozen document governs.

**How it is bound.** Each release ships this document, and the release manifest digests it with
every other shipped file. A row records that manifest's id, so a row is bound to the exact bytes
of this document and of every document below that the release carries. The frozen documents are
never edited: an amendment is a new dated document beside the ones it amends, added to the table
below. Between releases this document is edited to record it. Once a release has shipped this
document, those bytes are archived under a new dated name before it is edited again.

**Why this name.** The path `docs/paper-analysis-protocol.md` is spoken for. Frozen protocol v8
Section 5 states that a release ships a document at that path under the protocol ID
`taim-paper-analysis-protocol-2026-09-07-combined-release-0.2.1`, and v8 cannot be edited to say
otherwise. Giving this document that path would let one name resolve to two documents.

Protocol v1 Section 8 remains the sole canonical statement of the recall-floor asymmetry. Point at
it, never restate it.

## Frozen documents this release carries

| Document | Protocol ID | SHA-256 | Governs |
| --- | --- | --- | --- |
| [v1](paper-analysis-protocol-2026-08-29.md) | `taim-paper-analysis-protocol-2026-08-29-v1` | `sha256:411ea6e8d71fe29084b8870d6ea34a366f89c127a4371698da78aa5600af2fd7` | Metrics, statistical rules, failure rules, stop rules and claim boundaries, except where a later document states a narrower rule. Section 8 is the sole canonical statement of the recall-floor asymmetry. The identities of its Section 2 are amended by v2, and its admission-to-comparison clause by v8, which v10 restates. |
| [v2](paper-analysis-protocol-2026-08-31-v2.md) | `taim-paper-analysis-protocol-2026-08-31-v2` | `sha256:c14f7b28689a985de7aaeaece2e131338386bcdfcb489dd1955a5b59aba94594` | The recipe-v3 corpus and Evaluation Package identities. Their paper use is superseded by v3. |
| [v3](paper-analysis-protocol-2026-08-31-v3.md) | `taim-paper-analysis-protocol-2026-08-31-v3` | `sha256:18a9c9bd3c76455038d01458a5cb2d933059d1e72924285e26500a8aaab1baf6` | The Track, Profile, System and corpus-selection matrix, superseded by v4 for paper execution. |
| [v3 supersede note](paper-analysis-protocol-2026-08-31-v3-supersede-note-2026-09-01.md) | - | `sha256:6802d1447fc74b1e150876ea76667c1b3f84ce1b0ceed5bf7f840f2d87d98511` | The boundary between v3 and v4. |
| [v4](paper-analysis-protocol-2026-08-31-v4.md) | `taim-paper-analysis-protocol-2026-08-31-v4` | `sha256:55bb91f69d6494d58eff62b96f65f7b59759de45a366dc551e4ad5586fdfc896` | The experiment matrix, Task Inputs, Evaluation Packages, metrics, comparisons and reporting. Its external-baseline System identities and pinned commit are amended by v9. |
| [v4 supersede note](paper-analysis-protocol-2026-08-31-v4-supersede-note-2026-09-18.md) | - | `sha256:f02c332cb1278e1cb4ed9c62071548e4e96d637aa4476c4e193c6cfd9309cde9` | Records that the System identities and commit v4 names are those pinned when it was frozen. |
| [v5](paper-analysis-protocol-2026-09-02-v5.md) | `taim-paper-analysis-protocol-2026-09-02-v5` | `sha256:68ac1fa5bf2c25f4f54f106343ffcb153ed5f89a78c8fba4b774dc3c5bdc0bbf` | The retrieval producer's targets, query and retrieval methods, provider budget, evidence and admission rules. |
| [v6](paper-analysis-protocol-2026-09-02-v6.md) | `taim-paper-analysis-protocol-2026-09-02-v6` | `sha256:42aaad46b91eeb56a75d416de1a29e93e90cb3d15eb87db33c40fc1bc4a61208` | Low-memory execution and query-plan continuity. Its Section 2's four admitted query-plan files are replaced for the TrialGPT rows by v11. |
| [v7](paper-analysis-protocol-2026-09-02-v7.md) | `taim-paper-analysis-protocol-2026-09-02-v7` | `sha256:a617a46c2d61e76ad25963e709cad2c3ff99d4be466f173b8add02f1754927ef` | The executable release identity and the staged System's catalogue closure. Its Section 1 release is replaced by v9 Section 5, and its Section 2 and its role as the TrialGPT retrieval's recorded protocol by v11. |
| [v5-v7 supersede note](paper-analysis-protocol-2026-09-02-v5-v7-supersede-note-2026-09-19.md) | - | `sha256:0123b6f7258d4aaf2afc1411addb500f94af0b117bcd76ab206463ffe7b61a48` | Records that the release identities v5 to v7 name are not releases of this repository. |
| [v10](paper-analysis-protocol-2026-09-26-v10.md) | `taim-paper-analysis-protocol-2026-09-26-v10` | `sha256:d07cd79d6f8d40bbb4cb663bed604ae60ba088180e8d7538d7740b565b4eb693` | Restates the admission-to-comparison rule of v8 in terms a reader of this release can resolve. It adds, removes and changes no rule. |
| [v11](paper-analysis-protocol-2026-09-28-v11.md) | `taim-paper-analysis-protocol-2026-09-28-v11` | `sha256:ac8be7fc8a618acf282e66ccae6ddb716e75b9cdcb4ccd753bae45378496348b` | TrialGPT query plans generated per run and recorded by a receipt, in place of the four query-plan files v6 Section 2 admitted, which are not available: one generation per target, resume and recomputation rules, and what each row discloses. |
| [v12](paper-analysis-protocol-2026-09-28-v12.md) | `taim-paper-analysis-protocol-2026-09-28-v12` | `sha256:3e7160f1484033d3a43cf9892d70a3f7f3fa7daa1168a1d5927631b3ee5dea4c` | When a paper row from a release later than 0.1.1, or a run that existed before it was frozen, is admitted, including by content equality where a run's recorded identities differ, and how such a row is compared and disclosed. For those rows it amends v4 Sections 2 and 5, v7 Section 4, v8 Section 2 as v10 restates it, and v9 Section 5. |
| [v13](paper-analysis-protocol-2026-10-05-v13.md) | `taim-paper-analysis-protocol-2026-10-05-v13` | `sha256:ff85d21b17a20278a52e50922af3426f0564d9caf34ce994ad6809309cd74e38` | What the content-equality check of v12 Section 8 item 5 compares: what each System reads, established by a committed consumption record, and the identity of each System-owned input. It replaces that item's list of what is compared, and amends v12 Section 8 item 4 and Section 11 items 2 and 7 for a run checked under it. |
| [v14](paper-analysis-protocol-2026-10-07-v14.md) | `taim-paper-analysis-protocol-2026-10-07-v14` | `sha256:1fd0cce1d08be4ebb02b6a01de3eb6b2222262b98270c33a93d325d1b1c42848` | The release `TrialGPT-TAIM-Luna-v1`'s four rows execute: the release that first ships TrialGPT 2.0, so that one query-plan generation, receipt and retrieval freeze per target serve both Systems on identical candidates; and TrialGPT 2.0's qualification, comparability and disclosure. For those rows it amends v5 Section 2, v9 Section 5, v11 Sections 1, 4, 6 to 8 and 12, and v12 Sections 1, 6 and 7. |
| [v15](paper-analysis-protocol-2026-10-07-v15.md) | `taim-paper-analysis-protocol-2026-10-07-v15` | `sha256:59b660e47df8aa17012dae4c79bc268c4641d44f190725db4a79af97d36c3c10` | The prepared data of both TrialGPT Systems' rows on their four targets: a profile-sized copy of each stratum that adds the trials' complete eligibility text and its criteria, and nothing else; the new Prepared Snapshot, Task Input, Evaluation Package and pool receipt IDs those rows record; and that `TrialGPT-TAIM-Luna-v1` keeps paired comparisons. For those rows it amends v12 Sections 3, 6, 7 and 11 and v14 Sections 5 and 9. |
| [v16](paper-analysis-protocol-2026-10-09-v16.md) | `taim-paper-analysis-protocol-2026-10-09-v16` | `sha256:1b2e363d5d845fd2a60120f5b414209da692cb6c6ac3d40a98937948046afb95` | Where `TrialGPT-TAIM-Luna-v1` takes its criteria from on its four targets: upstream TrialGPT's own `parse_criteria` over inclusion and exclusion sections cut from the complete eligibility text, in place of the criterion Derived View; every v15 identity is unchanged |

## Frozen documents in force that this release does not carry

Each is identified by its digest.

| Protocol ID | SHA-256 | Why this release does not carry it |
| --- | --- | --- |
| `taim-paper-analysis-protocol-2026-09-09-v8` | `sha256:51b27f393496f345f8ce5887f4c75bf662c2c88a937238ba179ea9c3372b3ba0` | Its rule is restated in v10, which this release carries. Its other sections describe releases of an earlier repository and state no further rule. |
| `taim-paper-analysis-protocol-2026-09-18-v9` | `sha256:49f02a27bcee9737cc99344cf1716ee40f143295641db895971f3325fdc30aa9` | It governs the paper, binding the paper's rows to Release 0.1.1, not the benchmark release. |

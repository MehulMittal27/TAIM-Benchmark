# Compatibility

This release supports CPython 3.11, 3.12, 3.13, and 3.14. Its compatibility surface is:

The base and dense benchmark paths cover that Python matrix.
The optional `folding` extra is declared for CPython 3.11 and 3.12 because its pinned medspaCy stack
is not released for the newer interpreters.
The `staged` extra supplies the pinned reranker runtime, and the staged System therefore runs on
CPython 3.11 and 3.12.
The no-template dense component also requires `dense`.

| Contract | Version |
| --- | --- |
| Benchmark Snapshot | `2.0` |
| Run Manifest | `7.0` |
| System Input identity | `2.0` only; the retired `1.0` recipe is not read |
| Published Result Bundle | `1.0` |
| Benchmark Release manifest | `1.0` |
| Release support catalog | `1.0` |
| Release dependency environment | `1.0` |
| Patient-to-trial Run Manifest | `7.0` |
| Trial-to-patient contracts and run | `1.0` |
| Folded query bundle | `1.0` |
| Staged pipeline declaration | `1.0` |
| Judgment-union pool receipt | Profile `1.0`, release support `1.0` |
| External-fidelity pool receipt | Profile `1.0`, release support `1.0` |
| External submission task (`input.json`) | `1.0` |
| External submission result (`submission.json`) | `1.0` |
| External System dependency declaration | `1.0` |

The stable entry points are `trial-benchmark`, the explicit `PatientToTrialSystem` contracts, and the explicitly
exported `TrialToPatientTaskInput`, `TrialToPatientRunRequest`, direction-specific Benchmark
Profile, Evaluation Package, run, manifest, and evaluator contracts, and the `submission` input,
ranking, and pipeline-declaration formats in [Score your own pipeline](external-pipelines.md). A
forward run is not accepted by a reverse loader, and a reverse run is not accepted by a forward
loader. There is no neutral ranking artifact or generic direction flag.

Patient-to-trial Python execution contract `2.0` exports `PatientToTrialTaskInput`,
`PatientToTrialRunRequest`, `PatientToTrialRunResult`, and `PatientToTrialSystem` from
`taim.patient_to_trial`. These replace `SystemBenchmarkSnapshot`, `SystemRunRequest`,
`SystemRunResult`, and `System` from the removed `taim.system_contracts` module. Update imports;
there are no compatibility aliases. Serialized inputs and identity recipes retain their versions.

Pipeline implementations live under `taim.pipelines`, grouped by family. Shared retrieval,
generation, eligibility parsing and artifacts keep their shared modules. Each pipeline's CLI and
release binding lives in its family too: `taim.pipelines.<family>.extension` for a family of one
release unit, otherwise `taim.pipelines.<family>.<unit>_extension`. Since 0.1.2 the bindings moved
there from `taim.pipeline_extensions.<unit>`; update imports, there are no compatibility aliases.
`taim.pipeline_extensions` keeps only the binding contract (`PipelineExtension`, `RankingExecution`,
`extension_for`, `pipeline_extensions`). A binding registers only the ranking directions it
implements.
Submission export and validation use the sole installed reference declared for that task. They
refuse an installation with no reference or ambiguous defaults.

New forward run records use `configuration.system_input.trial_corpus_encoding = "lineage-v1"`.
They store ordered `(trial_id, trial_version_id)` pairs, an exact count and the canonical SHA-256
of those pairs instead of full trial documents. Derived Views contain only `name`, `version`,
`input_snapshot_id` and the content-addressed `view_id`; their configuration and content stay in
the source. This is a new encoding inside Run Manifest `7.0`;
readers predating this encoding reject it. The full `PatientToTrialTaskInput.to_dict()` interface,
Task Input IDs, System Input identity recipes, entity versions and evaluator identities are
unchanged. `PatientToTrialRunRequest.system_input_record_dict()` is the only forward System Input
serialization, and every run producer writes it.

Loading a compact run verifies the recorded lineage, patient content, System Input identity and
closed artifact hashes. It cannot recompute the original content-based Task Input ID without the
corpus. `RunManifest.resolve_task_input(source)` requires a matching, profile-resolved
`PatientToTrialTaskInput` before returning entity content; it checks the complete compact record,
including the source's recomputed Task Input ID. Both forward loaders accept `source_snapshot=`
for this additional verification. Retain the checksum-verified prepared package, select the run's
Benchmark Profile, and use its Patient Evidence Profile and clinical cutoff to construct the source.
A matching trial ID alone never authorizes substituting current registry content.
Both Result Bundle builders require `source_snapshot=` for compact forward runs and verify it
before projecting or writing the bundle. Source-free loading alone does not establish publication
integrity.

Forward manifests must record the lineage encoding. Earlier v7 records that embedded full trial
content are no longer read; they are verified from the commit that wrote them. Pre-v7 manifests
remain unsupported. Compact records still contain patient information and are
Local Run artifacts, not safe public projections; Result Bundle privacy rules continue to apply.

Named forward Stage Rankings remain part of their patient-to-trial Run Manifest and closed run
tree. They are validated against the same direction, Task Input, run, System, entity membership,
and artifact hashes as the Primary Ranking. `--stage` evaluation is explicitly diagnostic and does
not change the Primary Ranking identity.

`trial-benchmark support list` returns every executable Track × Task × Profile × System tuple. The
release validator rejects unknown or duplicate tuples, missing projected files, unsupported System
directions, and any catalog that differs from the release manifest.

Full-preparation Profiles and judgment-union Profiles are distinct compatibility identities. A
judgment-union run additionally binds the pre-effectiveness selected-trial count, ordered-ID
SHA-256, and content-addressed pool receipt ID emitted by `trial-benchmark data pool inspect`.
Neither artifact can be relabelled or loaded as the other Profile.

The TREC 2021 external-fidelity Profile is a third compatibility identity. It requires the
checksum-verified 26,149-ID membership file, its own receipt, and the separately approved v4
protocol.
Release-selected external rows additionally require their exact external inputs and source
locks. An installed public package can validate and execute those rows when the inputs are
provided, but it does not vendor or authorize them.
TrialGPT provider execution additionally requires a POSIX runtime for its process-exclusive
generation-cache lease. Package import and the rest of the benchmark remain available elsewhere;
the TrialGPT command fails before provider execution when advisory file locking is unavailable.

TrialGPT on TREC 2022 or SIGIR reuses the exact judgment-union compatibility identity of the
corresponding reference rows.
TrialMatchAI on TREC 2022 or TREC 2023 reuses the exact judgment-union compatibility identity of
the corresponding reference rows.
Each external System's ID, producer inputs, and v4 protocol identity remain distinct.
TrialGPT retrieval artifacts also bind
the v5 method contract and the v6 and v7 executable-release amendments.
A Track-specific external
artifact cannot be relabeled for another Track even when the adapter code and model configuration
match.

This release uses an immutable disk-backed sequence instead of retaining full-corpus trial objects.
This changes resource use, not logical Snapshot content or any public artifact schema. Loading and
bounded-Profile projection are validated on an 18 GB laptop; individual Systems retain their own
hardware requirements, stated with each System in the [method contracts](method-contracts.md).

Release versions use semantic versioning for the documented Python and CLI surface. Artifact schema
versions remain independent: incompatible schema changes increment that schema's major version,
while additive compatible changes increment its minor version. Patch releases do not change method
or artifact identities unless they correct a validation defect and say so in the release notes.

An artifact with a different major schema version is rejected. A changed task input, effective
System request, benchmark profile, evaluation package, or selected ranking product receives a
different identity rather than being treated as compatible output.

# Benchmark Release 0.1.1

Release 0.1.1 follows Release 0.1.0 in this repository. It changes the identities of the three
TrialMatchAI Systems and the final ranking depth of the TrialGPT System, and it states which frozen
protocol each documented command uses. No Track, Profile, Task Input, metric, or reference, dense,
fused or staged System changes.

The three TrialMatchAI Systems are now `trialmatchai-current-cuda-l4-trec21-development-v4`,
`trialmatchai-current-cuda-l4-trec22-development-v4` and
`trialmatchai-current-cuda-l4-trec23-development-v4`. They pin `MehulMittal27/TrialMatchAI` commit
`99aa499e7296ebde836a0ebf00a5ed4948892a83` in place of `7eba8f399336fcd988b00ce98f2e025e5ec04119`.
The newer commit changes code on the execution path, so the System identities change with it. The
frozen L4 configuration and the MIT licence are the same bytes at both commits. The `-v3`
identities are retired, and the CLI refuses them.

`TrialGPT-TAIM-Luna-v1` now emits a top-20 Primary Ranking. Its publication contract requires
retrieval depth 2,000 and final `--top-k 20`, where 0.1.0 required 10. Its candidates, provider
call budget and workers are unchanged. The System Input ID records the depth, so a run at 20 is a
different experiment from a run at 10.

The README now names both frozen protocols. The v3 protocol selects the five judgment-union strata,
and the judgment-union commands pass its digest. The v4 amendment governs the TREC 2021
external-fidelity Profile and the external baseline Systems, and their commands pass the v4 digest.
Neither protocol's bytes changed, so both digests are the ones 0.1.0 accepted. Supersede notes
beside the frozen v4 protocol and beside the frozen v5, v6 and v7 protocols say how the identities
those protocols record relate to this release.

Runs and Result Bundles record the release they were made with. Artifacts made with 0.1.0 are not
0.1.1 artifacts; retrieval and System artifacts must be generated fresh against this release.
Release 0.1.0 and its notes remain available at its own tag.

Release 0.1.1 contains no paper results. It excludes protected benchmark data, provider traces,
model weights, dense indexes, concept stores, external repository checkouts, and internal
reproduction and unreleased variants.

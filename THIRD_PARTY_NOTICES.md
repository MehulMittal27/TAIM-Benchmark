# Third-party notices

This notice records the third-party material reviewed for the benchmark.

## Scope

This release distributes its own source code, documentation, fixtures, and acquisition
metadata. It does not redistribute Python wheels, model weights, workflow-action source trees, or
benchmark source files.

Exact Python versions, artifact URLs, hashes, and platform-specific transitive dependencies remain
in `uv.lock`. The release audit retains a separate machine-checkable license binding for every
dependency. This notice summarizes the third-party material relevant to readers of the public
repository instead of repeating that inventory. The repository owner reviewed and approved every
expression changed during the external-baseline amendment against the accompanying licence-closure
evidence.

## Python dependencies

The package declares its direct dependencies in `pyproject.toml` and freezes the complete resolved
environment in `uv.lock`. Installation obtains those packages from their original distributors
under their own licenses. The public repository and TAIM distribution do not vendor their source or
binary artifacts.

The optional folding path pins medspaCy 1.3.1. Its upstream repository publishes the MIT license.
medspaCy's complete transitive dependency set is fixed in `uv.lock`; the release definition binds
every resolved distribution individually, including platform-conditional packages. The amendment
rechecked the exact PyPI artifacts and replaced all `LicenseRef-Requires-Human-Review` and
`NOASSERTION` sentinels with reviewable expressions. Notable non-MIT entries include Cython, NLTK,
PyFastNER, and Requests under Apache-2.0; blis and pysimstring under BSD-3-Clause; defusedxml under
PSF-2.0; wrapt under BSD-2-Clause; and Unidecode under GPL-2.0-or-later. These packages are resolved
installation dependencies; their source and binary artifacts are not copied into the repository or
TAIM distribution.

On supported GPU platforms, PyTorch installation may resolve CUDA and NVIDIA packages governed by
NVIDIA terms. The `cuda-toolkit` 13.0.3.0 wheel declares no licence in its embedded metadata, so the
release definition records `LicenseRef-NVIDIA-CUDA-EULA` instead of treating the absence as an
approval. TAIM does not redistribute those packages or accept their terms on a user's behalf. See
the exact platform resolution in `uv.lock` and NVIDIA's current CUDA licence terms at
<https://docs.nvidia.com/cuda/eula/index.html>.

## Model revisions

The release can use these separately obtained model revisions:

- `ncbi/MedCPT-Article-Encoder@d05a736da4bb84ee4057b7f7999485be6ed85465` and
  `ncbi/MedCPT-Query-Encoder@d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`, each covered by
  the exact-revision NCBI public-domain notice for United States Government works.
- `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`, licensed MIT according to the
  exact-revision Hugging Face model metadata.
- `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, licensed
  Apache-2.0 according to the exact-revision Hugging Face model metadata.
- `Qwen/Qwen3-Reranker-4B@22e683669bc0f0bd69640a1354a6d0aebcfeede5`, licensed
  Apache-2.0 according to the exact-revision Hugging Face model metadata.

The release records these identities and checksums but does not include model weights. Users obtain
the files from the model repositories and remain responsible for the applicable terms.

## Distribution boundary

TAIM distributes its own source-lock metadata format and adapter interfaces.
Adapters and source-lock metadata for release-selected external Systems ship with those Systems.
It does not redistribute any upstream checkout, benchmark corpora, model weights,
provider outputs, concept stores, indexes, or historical results.

The release declares separately provisioned external Systems. The Systems it declares, their Tracks
and their Profiles are listed in the release catalogue.
TrialGPT inputs are provisioned separately for each Profile the catalogue declares for it.
The pinned TrialMatchAI fork is provisioned separately for each Profile the catalogue declares
for it.

The TrialGPT source lock pins `ncbi-nlp/TrialGPT` commit
`1b3242cd50c153a9c324661d4a0bd94fc68a4226`. Its exact LICENSE states that the upstream code is a
United States Government work/public-domain contribution and includes a warranty disclaimer and
citation request. Its TAIM-owned retrieval producer uses the two pinned MedCPT encoders listed
above. Their exact revisions contain the NCBI public-domain notice; neither model is redistributed.
`TrialGPT-TAIM-Luna-v1` additionally uses the separately authorized OpenAI Codex service with model
`gpt-5.6-luna`. This notice does not accept service terms or authorize provider calls. Every Track
requires a fresh, Track-specific frozen retrieval artifact. The checksum-locked TREC 2021
external-fidelity corpus is acquired separately; availability of its metadata does not authorize
redistribution. The TREC 2022 and SIGIR adaptations use their release-defined judgment-union Task
Inputs rather than copying TrialGPT's published cohort sizes.

The selected TrialMatchAI checkout pins `MehulMittal27/TrialMatchAI` commit
`7eba8f399336fcd988b00ce98f2e025e5ec04119`, whose exact LICENSE is MIT. The original paper
repository also contains an MIT LICENSE, while its Zenodo record declares CC BY 4.0 at the record
level. TAIM does not resolve that metadata discrepancy by copying either repository; it records both
facts for human review and invokes only the separately acquired current checkout.

The selected TrialMatchAI runtime binds these exact model revisions:

- `microsoft/phi-4@2db69c1c3e91a05d2c64a3185acfbaf36f744e25`: MIT metadata.
- `majdabd33/trialmatchai-phi4-reasoning-lora@9eaddaf048c8d4266a291884ff89db1cf05b07fc`:
  MIT metadata, no standalone LICENSE at the exact snapshot, and an upstream card restricting its
  description to research and informational use. The training-data rights are not established by
  the adapter metadata.
- `google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`: gated custom Gemma
  terms. A human must accept the terms through the original distributor before use.
- `majdabd33/trialmatchai-gemma2-reranker-lora@3118ba76d545f71f3aaa7952d2f031ad5fdef8c9`:
  Gemma terms. Obtaining the adapter does not remove the gated base-model terms; its model card's
  base-model naming discrepancy remains a review caveat.
- `fastino/gliner2-base-v1@f5b2ecedebe4381b088c1cf276f5bf72a52cac54`: Apache-2.0 metadata,
  with no standalone LICENSE or NOTICE in the exact model snapshot.
- `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`: MIT metadata, already
  listed above; the exact snapshot has no standalone LICENSE file.

The TrialMatchAI concept store is a user-supplied, manifest-bound runtime input. Its vocabulary and
terminology sources may require separate licenses, including SNOMED CT where applicable. TAIM does
not include the store or accept those licenses for the user.
No external System may execute until the operator provides every required licensed input.
The TREC 2023 adaptation is a TAIM run of the later pinned fork; it is not a reproduction of the
TrialMatchAI paper.

## Workflow actions

The release workflow references these immutable, MIT-licensed commits:

- `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`
- `astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e`

The workflow does not copy either action's source tree into this repository.

## Benchmark acquisition metadata

The TREC Clinical Trials 2021, 2022, and 2023 guides and source locks record official source URLs,
byte sizes, SHA-256 values, and the absence of explicit artifact-specific redistribution terms on
the source pages. Users must acquire the topics, qrels, trial archives, and other source files
themselves. Publishing acquisition metadata does not authorize redistribution of those files.

The SIGIR Clinical Trials 2016 guide and source lock cite CSIRO collection version 4, DOI
`10.4225/08/58e2e83d92c2b`, which states a CC BY-SA 4.0 license. The underlying collection remains
local and is not copied into the release.

The folded-query command requires a user-supplied licensed SNOMED CT release. TAIM records the
release name and SHA-256 of its description file but does not redistribute the description file,
concept identifiers, normalized descriptions, or any other SNOMED CT content. Output queries
contain only spans copied from the benchmark topic text. Possession of this software or its
metadata does not grant a SNOMED CT license.

## Exclusions

E5, BioLORD, generic fused panels, internal reproduction and unreleased variants, unselected TrialGPT or TrialMatchAI
variants, and historical campaign configurations are outside the exported dependency closure.
Selected external Systems, where a release declares any, are limited to the exact source locks and
capability rows its catalogue declares.

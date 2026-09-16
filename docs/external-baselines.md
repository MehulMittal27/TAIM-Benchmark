# External baseline strata

Release 0.1 declares one separate TREC Clinical Trials 2021 external-fidelity Profile:
`trec-ct-2021-external-fidelity-26149`. It compares the Systems the release catalogue declares for
that Profile over the same 26,149-trial Task Input.

This is a separate table from TAIM's 26,162-trial TREC 2021 judgment-union Profile. Neither table
may be relabelled as the other. The external Profile is within-pool ranking, not full-corpus
retrieval or an official TREC-run reproduction.

Acquisition and the reference-System commands are in
[the external-fidelity Profile guide](external-fidelity-profile.md).

## TREC 2022, TREC 2023, and SIGIR 2016

The release catalogue declares the external Systems' adaptation rows on the existing score-blind
judgment-union Profiles.

Run `trial-benchmark data pool inspect` for the selected Profile before any score. Use its exact
count, ordered-ID digest, and receipt for the reference and external rows. TREC 2022 and TREC 2023
require authorized local source preparation. The release worker did not open those protected
inputs and does not preclaim their counts.

Published pool counts of 26,581, 3,621, and 17,103 are literature context rather than TAIM corpus
identities.

## Reporting boundary

Report the TREC 2021 external-fidelity rows only in their named 26,149-trial stratum. The added
adaptation rows use their named TAIM judgment-union Profile and may be compared only with rows
sharing the same Task Input and Evaluation Package. Upstream paper scores are contextual literature
values, not like-for-like baselines. Unjudged pairs remain unjudged. These experiments do not
establish clinical validity, eligibility, deployment safety, enrollment benefit, or patient benefit.

# TAIM TREC-style synthetic fixture

This fixture is synthetic and contains no patient data. It has three patient topics, eight trial
documents, and explicit judgments on the TREC Clinical Trials scale:

- `0`: not relevant;
- `1`: condition-relevant but excluded;
- `2`: eligible.

For topic `T001`, lexical overlap intentionally places `NCT-BREAST-EXCLUDED` above
`NCT-BREAST-ELIGIBLE`. This catches evaluations that incorrectly treat condition relevance as
eligibility.

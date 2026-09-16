"""Deterministic public trial-to-patient conformance fixture."""

from __future__ import annotations

from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.release_fixture import FIXTURE_CLINICAL_AS_OF, load_release_fixture
from taim.trial_to_patient import (
    REVERSE_PRESERVE_JUDGED_PAIRS_V1,
    ResolvedTrialToPatientBenchmark,
    TrialToPatientBenchmarkProfile,
    resolve_trial_to_patient_benchmark,
)

FIXTURE_REVERSE_PROFILE_ID = "synthetic-trial-to-patient"


def load_reverse_fixture(track: str = "trec-ct-2021") -> ResolvedTrialToPatientBenchmark:
    """Transpose only explicit fixture judgments and preserve every absent pair as unknown."""

    return resolve_trial_to_patient_benchmark(
        load_release_fixture(track),
        clinical_as_of=FIXTURE_CLINICAL_AS_OF,
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
        transposition_policy=REVERSE_PRESERVE_JUDGED_PAIRS_V1,
        profile=TrialToPatientBenchmarkProfile(
            profile_id=FIXTURE_REVERSE_PROFILE_ID,
            profile_version="1.0",
            cutoffs=(1, 3, 5),
            corpus_policy="all_packaged_synthetic_patient_versions",
            query_policy="all_packaged_synthetic_trial_versions",
            gain_mapping={"0": 0, "1": 1, "2": 2},
        ),
    )


__all__ = [
    "FIXTURE_CLINICAL_AS_OF",
    "FIXTURE_REVERSE_PROFILE_ID",
    "load_reverse_fixture",
]

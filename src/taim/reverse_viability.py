"""Release 0.1 recipe-v3 TREC 2021 reverse-viability profile qualification."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from taim.data import PreparedBenchmark
from taim.entity_versions import PatientEvidenceProfile, version_patients, version_trials
from taim.schemas import SchemaValidationError
from taim.trial_to_patient import (
    ResolvedTrialToPatientBenchmark,
    TrialToPatientBenchmarkProfile,
    TrialToPatientEvaluationPackage,
    TrialToPatientJudgment,
    TrialToPatientTaskInput,
)

REVERSE_VIABILITY_PROFILE_ID = "trec-ct-2021-reverse-complete10"
RELEASE_0_1_RECIPE_V3_PREPARED_SNAPSHOT_ID = (
    "sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8"
)
RELEASE_0_1_RECIPE_V3_EVALUATION_PACKAGE_ID = (
    "sha256:fa47e8e588e06a0d69f3fc32ea34f59113c875bdc8219327ce8faaa8930e9dba"
)
CONTROLLED_JUDGMENTS_SHA256 = (
    "sha256:85097a6153e2703e82038fad26ad33181b90e257ea313364cd44eee830fabf15"
)
CONTROLLED_SOURCE_IDENTITIES = {
    "topics": "sha256:94bda921ce7c40a0353f251abb2ea938c77331759a9f83a36abd145ab5840aca",
    "qrels": "sha256:ba7a2cddc90285e75cd76adcd483394a6c9bacf7017113222058ba6537e6d8ac",
    "trials_part_1": "sha256:4caa9579290adaa974efae6f5be170c8964c0fb5d110c1bdb935b331d0bf3ec2",
    "trials_part_2": "sha256:6f4800b0c9e57af5bcef039c2d0661b2c5e844bb5025f959d7a940c03c2d902d",
    "trials_part_3": "sha256:1190a5bd629d011dc7011b42b92f94cb7ec0ec21522bafb7701aaee826a42fce",
    "trials_part_4": "sha256:d10ecad6993dea01e9f581e0104cdff6cd8120d0bc851de87d088cf799a7dd98",
    "trials_part_5": "sha256:17c773184de8fea9cfcac5cd0cca45c816f4c4ac7568d35d7b0873f63bfc1602",
}
CONTROLLED_TRIAL_IDS = (
    "NCT01020279",
    "NCT01022476",
    "NCT01022905",
    "NCT01026584",
    "NCT01029873",
    "NCT01991457",
    "NCT01993498",
    "NCT01994200",
    "NCT01994343",
    "NCT01995500",
)
CONTROLLED_LABEL_COUNTS = {
    "NCT01020279": (67, 6, 2),
    "NCT01022476": (75, 0, 0),
    "NCT01022905": (63, 0, 12),
    "NCT01026584": (74, 1, 0),
    "NCT01029873": (74, 0, 1),
    "NCT01991457": (75, 0, 0),
    "NCT01993498": (73, 2, 0),
    "NCT01994200": (75, 0, 0),
    "NCT01994343": (72, 2, 1),
    "NCT01995500": (72, 3, 0),
}
CONTROLLED_TOPIC_IDS = frozenset(str(index) for index in range(1, 76))


def qualify_reverse_viability_profile(
    prepared: PreparedBenchmark,
    *,
    clinical_as_of: datetime,
    patient_evidence_profile: PatientEvidenceProfile,
) -> ResolvedTrialToPatientBenchmark:
    """Reproduce and transpose only the frozen complete 10-trial by 75-patient matrix."""

    if prepared.dataset_id != "trec-ct-2021":
        raise SchemaValidationError("reverse viability is restricted to TREC 2021")
    if prepared.snapshot.snapshot_id != RELEASE_0_1_RECIPE_V3_PREPARED_SNAPSHOT_ID:
        raise SchemaValidationError("reverse viability Prepared Snapshot ID changed")
    if (
        prepared.evaluation_package.evaluation_package_id
        != RELEASE_0_1_RECIPE_V3_EVALUATION_PACKAGE_ID
    ):
        raise SchemaValidationError("reverse viability Evaluation Package ID changed")
    if prepared.package_checksums.get("judgments.jsonl") != CONTROLLED_JUDGMENTS_SHA256:
        raise SchemaValidationError("reverse viability Judgment serialization changed")
    source_identities = {
        artifact.role: artifact.sha256 for artifact in prepared.source_bundle.artifacts
    }
    if source_identities != CONTROLLED_SOURCE_IDENTITIES:
        raise SchemaValidationError("reverse viability source identities changed")
    topics = tuple(sorted(prepared.topics, key=lambda item: item.topic_id))
    if len(topics) != 75 or {topic.topic_id for topic in topics} != CONTROLLED_TOPIC_IDS:
        raise SchemaValidationError("reverse viability requires TREC 2021 Topics 1 through 75")
    judgments_by_trial = defaultdict(list)
    for judgment in prepared.evaluation_package.judgments:
        judgments_by_trial[judgment.trial_id].append(judgment)
    complete_trial_ids = {
        trial_id
        for trial_id, judgments in judgments_by_trial.items()
        if len(judgments) == 75
        and {judgment.topic_id for judgment in judgments} == CONTROLLED_TOPIC_IDS
    }
    if complete_trial_ids != set(CONTROLLED_TRIAL_IDS):
        raise SchemaValidationError("reverse viability complete-matrix selector changed")
    trial_by_id = {trial.trial_id: trial for trial in prepared.trials}
    try:
        trials = tuple(trial_by_id[trial_id] for trial_id in CONTROLLED_TRIAL_IDS)
    except KeyError as exc:
        raise SchemaValidationError("reverse viability trial is absent from the Snapshot") from exc
    judgments = tuple(
        sorted(
            (
                judgment
                for judgment in prepared.evaluation_package.judgments
                if judgment.trial_id in complete_trial_ids
            ),
            key=lambda item: (item.trial_id, item.topic_id),
        )
    )
    if len(judgments) != 750:
        raise SchemaValidationError("reverse viability denominator must be exactly 750 pairs")
    for trial_id in CONTROLLED_TRIAL_IDS:
        counts = Counter(row.label for row in judgments if row.trial_id == trial_id)
        if (counts[0], counts[1], counts[2]) != CONTROLLED_LABEL_COUNTS[trial_id]:
            raise SchemaValidationError(f"reverse viability labels changed for {trial_id}")
    patient_versions = version_patients(
        topics,
        patient_evidence_profile,
        clinical_as_of=clinical_as_of,
        derived_views=prepared.snapshot.derived_views,
    )
    trial_versions = version_trials(trials, derived_views=prepared.snapshot.derived_views)
    profile = TrialToPatientBenchmarkProfile(
        profile_id=REVERSE_VIABILITY_PROFILE_ID,
        profile_version="1.0",
        cutoffs=(5, 10, 20),
        corpus_policy="frozen_complete-ten-trial-matrix",
        query_policy="ten_frozen_complete_matrix_trial_queries",
        gain_mapping={"0": 0, "1": 1, "2": 2},
    )
    task_input = TrialToPatientTaskInput(
        benchmark_lineage=prepared.snapshot.benchmark_lineage,
        prepared_snapshot_id=prepared.snapshot.snapshot_id,
        clinical_as_of=clinical_as_of,
        patient_evidence_profile=patient_evidence_profile,
        benchmark_profile=profile,
        trial_versions=trial_versions,
        patient_versions=patient_versions,
        available_capabilities=prepared.snapshot.available_capabilities,
        derived_views=prepared.snapshot.derived_views,
    )
    patients = {version.patient_id: version for version in patient_versions}
    selected_trials = {version.trial_id: version for version in trial_versions}
    evaluation_package = TrialToPatientEvaluationPackage(
        benchmark_lineage=task_input.benchmark_lineage,
        prepared_snapshot_id=task_input.prepared_snapshot_id,
        query_set_id=task_input.query_set_id,
        patient_corpus_id=task_input.patient_corpus_id,
        benchmark_profile_definition_sha256=profile.definition_sha256,
        judgments=tuple(
            TrialToPatientJudgment(
                trial_id=row.trial_id,
                trial_version_id=selected_trials[row.trial_id].trial_version_id,
                patient_id=row.topic_id,
                patient_version_id=patients[row.topic_id].patient_version_id,
                label=row.label,
            )
            for row in judgments
        ),
        provenance={
            "claim_scope": "reverse viability for this frozen complete matrix only",
            "denominator_pairs": 750,
            "label_semantics": {
                "0": "not relevant",
                "1": "excluded",
                "2": "eligible",
            },
            "source_evaluation_package_id": RELEASE_0_1_RECIPE_V3_EVALUATION_PACKAGE_ID,
            "source_identities": dict(CONTROLLED_SOURCE_IDENTITIES),
            "transposition_policy": "transpose-patient-trial-pairs-v1",
            "trial_ids": list(CONTROLLED_TRIAL_IDS),
        },
    )
    return ResolvedTrialToPatientBenchmark(
        prepared=prepared,
        task_input=task_input,
        profile=profile,
        evaluation_package=evaluation_package,
    )


__all__ = [
    "CONTROLLED_JUDGMENTS_SHA256",
    "CONTROLLED_SOURCE_IDENTITIES",
    "CONTROLLED_TRIAL_IDS",
    "RELEASE_0_1_RECIPE_V3_EVALUATION_PACKAGE_ID",
    "RELEASE_0_1_RECIPE_V3_PREPARED_SNAPSHOT_ID",
    "REVERSE_VIABILITY_PROFILE_ID",
    "qualify_reverse_viability_profile",
]

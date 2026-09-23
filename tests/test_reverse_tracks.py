from __future__ import annotations

# ruff: noqa: S101
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taim.data import PreparedBenchmark, SnapshotPreparationManifest
from taim.entity_versions import source_grounded_patient_evidence_profile
from taim.evaluation_package import EvaluationPackage
from taim.public_patient_to_trial import load_public_patient_to_trial_run
from taim.release_cli import main
from taim.release_support import (
    SUPPORTED_TRACKS,
    load_release_support,
)
from taim.reverse_viability import (
    CONTROLLED_JUDGMENTS_SHA256,
    CONTROLLED_LABEL_COUNTS,
    CONTROLLED_SOURCE_IDENTITIES,
    CONTROLLED_TRIAL_IDS,
    qualify_reverse_viability_profile,
)
from taim.schemas import RelevanceJudgment, SchemaValidationError
from taim.snapshot import (
    CAPABILITY_CANONICAL_PATIENT_TEXT,
    CAPABILITY_CANONICAL_TRIAL_TEXT,
    BenchmarkSnapshot,
    BenchmarkTopic,
    SourceRecordIdentity,
    TrialDocument,
)
from taim.source import SourceArtifact, SourceBundle
from taim.trial_to_patient import load_trial_to_patient_run


def test_reverse_support_preserves_its_single_qualified_real_profile() -> None:
    real_reverse = [
        row
        for row in load_release_support()["support_matrix"]
        if row["task"] == "trial_to_patient" and row["evidence_scope"] != "synthetic_conformance"
    ]
    assert [(row["track"], row["profile"]) for row in real_reverse] == [
        ("trec-ct-2021", "trec-ct-2021-reverse-complete10")
    ]


def _reverse_viability_prepared(
    *,
    prepared_snapshot_id: str = (
        "sha256:5ddeded840f94964e578ca7cfe3e08d2fbb2ac0afe6588e350526eab3531a4c8"
    ),
    evaluation_package_id: str = (
        "sha256:fa47e8e588e06a0d69f3fc32ea34f59113c875bdc8219327ce8faaa8930e9dba"
    ),
) -> PreparedBenchmark:
    topics = tuple(
        sorted(
            (
                BenchmarkTopic(
                    topic_id=str(index),
                    source_identity=SourceRecordIdentity("synthetic-reverse-test", str(index)),
                    canonical_text=f"Synthetic patient {index}",
                )
                for index in range(1, 76)
            ),
            key=lambda item: item.topic_id,
        )
    )
    trials = tuple(
        TrialDocument(
            trial_id=trial_id,
            source_identity=SourceRecordIdentity("synthetic-reverse-test", trial_id),
            canonical_text=f"Synthetic trial {trial_id}",
        )
        for trial_id in CONTROLLED_TRIAL_IDS
    )
    judgments = []
    for trial_id in CONTROLLED_TRIAL_IDS:
        label_zero, label_one, label_two = CONTROLLED_LABEL_COUNTS[trial_id]
        labels = (0,) * label_zero + (1,) * label_one + (2,) * label_two
        judgments.extend(
            RelevanceJudgment(str(index), trial_id, label)
            for index, label in enumerate(labels, start=1)
        )
    snapshot = BenchmarkSnapshot(
        benchmark_lineage="trec-ct-2021",
        snapshot_name="synthetic-reverse-viability",
        topics=topics,
        trials=trials,
        available_capabilities=frozenset(
            {CAPABILITY_CANONICAL_PATIENT_TEXT, CAPABILITY_CANONICAL_TRIAL_TEXT}
        ),
    )
    object.__setattr__(snapshot, "snapshot_id", prepared_snapshot_id)
    evaluation_package = EvaluationPackage(
        benchmark_lineage="trec-ct-2021",
        task_id="synthetic-reverse-viability",
        snapshot_id=prepared_snapshot_id,
        judgments=tuple(sorted(judgments, key=lambda item: (item.topic_id, item.trial_id))),
        provenance={"synthetic": True},
    )
    object.__setattr__(evaluation_package, "evaluation_package_id", evaluation_package_id)
    source_bundle = SourceBundle(
        artifacts=tuple(
            SourceArtifact(
                artifact_id=role,
                role=role,
                filename=f"{role}.fixture",
                byte_size=0,
                sha256=sha256,
                acquisition_date="2026-08-31",
                access_terms="synthetic test metadata",
                redistribution_terms="synthetic test metadata",
                url=f"fixture://{role}",
            )
            for role, sha256 in CONTROLLED_SOURCE_IDENTITIES.items()
        ),
        lock_filename="synthetic-reverse-lock.json",
        lock_sha256="sha256:" + "1" * 64,
    )
    return PreparedBenchmark(
        dataset_id="trec-ct-2021",
        directory=Path("/synthetic-reverse-viability"),
        snapshot=snapshot,
        evaluation_package=evaluation_package,
        source_bundle=source_bundle,
        preparation=SnapshotPreparationManifest(
            source_recipe_id="synthetic-reverse-recipe-v1",
            source_recipe_hash="sha256:" + "2" * 64,
            source_bundle_id=source_bundle.source_bundle_id,
        ),
        diagnostics=(),
        package_checksums={"judgments.jsonl": CONTROLLED_JUDGMENTS_SHA256},
        prepared_manifest={},
        prepared_manifest_hash="sha256:" + "3" * 64,
    )


def test_reverse_viability_accepts_only_the_release_recipe_v3_inputs() -> None:
    resolved = qualify_reverse_viability_profile(
        _reverse_viability_prepared(),
        clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
        patient_evidence_profile=source_grounded_patient_evidence_profile(),
    )
    assert len(resolved.task_input.trial_versions) == 10
    assert len(resolved.task_input.patient_versions) == 75
    assert len(resolved.evaluation_package.judgments) == 750
    assert resolved.evaluation_package.provenance["denominator_pairs"] == 750

    with pytest.raises(SchemaValidationError, match="Prepared Snapshot ID changed"):
        qualify_reverse_viability_profile(
            _reverse_viability_prepared(
                prepared_snapshot_id=(
                    "sha256:65a3ecb67c65730f77dcdcb14f70db64e0baed85e0a3df4ced407f464dca0ad3"
                )
            ),
            clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
            patient_evidence_profile=source_grounded_patient_evidence_profile(),
        )
    with pytest.raises(SchemaValidationError, match="Evaluation Package ID changed"):
        qualify_reverse_viability_profile(
            _reverse_viability_prepared(
                evaluation_package_id=(
                    "sha256:bc7d90d250b464c17921c545c11b0159334ad343d0b5036d55468c6f02d36c0c"
                )
            ),
            clinical_as_of=datetime(2021, 4, 27, tzinfo=UTC),
            patient_evidence_profile=source_grounded_patient_evidence_profile(),
        )


def test_mixed_direction_loaders_fail(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "patient-to-trial",
                "fixture",
                "run",
                "--run-id",
                "forward",
                "--output-dir",
                str(tmp_path),
                "--top-k",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "trial-to-patient",
                "fixture",
                "run",
                "--run-id",
                "reverse",
                "--output-dir",
                str(tmp_path),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    capsys.readouterr()
    with pytest.raises(ValueError):
        load_public_patient_to_trial_run(tmp_path / "reverse")
    with pytest.raises(ValueError):
        load_trial_to_patient_run(tmp_path / "forward")


def test_synthetic_task_inputs_are_track_bound_for_all_four_tracks(tmp_path: Path, capsys) -> None:
    forward_ids = set()
    reverse_ids = set()
    for index, track in enumerate(SUPPORTED_TRACKS):
        for command, target in (
            ("patient-to-trial", forward_ids),
            ("trial-to-patient", reverse_ids),
        ):
            run_id = f"{index}-{command}"
            assert (
                main(
                    [
                        command,
                        "fixture",
                        "run",
                        "--track",
                        track,
                        "--run-id",
                        run_id,
                        "--output-dir",
                        str(tmp_path),
                        "--top-k",
                        "8",
                    ]
                )
                == 0
            )
            report = json.loads(capsys.readouterr().out)
            assert report["track"] == track
            target.add(report["task_input_id"])
    assert len(forward_ids) == len(SUPPORTED_TRACKS)
    assert len(reverse_ids) == len(SUPPORTED_TRACKS)

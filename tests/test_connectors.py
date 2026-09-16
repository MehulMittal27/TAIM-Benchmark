from __future__ import annotations

# ruff: noqa: S101
import io
import json
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taim.data import load_prepared_benchmark
from taim.data.sigir_ct_2016 import (
    DATASET_ID as SIGIR_DATASET_ID,
)
from taim.data.sigir_ct_2016 import (
    EXPECTED_SOURCE_ROLES as SIGIR_SOURCE_ROLES,
)
from taim.data.sigir_ct_2016 import (
    SOURCE_FILENAMES as SIGIR_FILENAMES,
)
from taim.data.sigir_ct_2016 import (
    prepare_sigir_ct_2016,
)
from taim.data.trec_ct import TRACKS, prepare_trec_ct
from taim.data.trec_ct_2021 import TRIAL_ARCHIVE_FILENAMES, prepare_trec_ct_2021
from taim.file_hash import sha256_file
from taim.release_cli import main

FIXED_TIME = datetime(2026, 8, 29, tzinfo=UTC)


def _trial_xml(trial_id: str, *, alias: str | None = None) -> str:
    alias_xml = f"<nct_alias>{alias}</nct_alias>" if alias is not None else ""
    return f"""
    <clinical_study>
      <id_info><nct_id>{trial_id}</nct_id>{alias_xml}</id_info>
      <brief_title>Synthetic alpha treatment</brief_title>
      <condition>Alpha syndrome</condition>
      <eligibility>
        <healthy_volunteers>No</healthy_volunteers>
        <criteria><textblock>Adults with alpha syndrome.</textblock></criteria>
      </eligibility>
    </clinical_study>
    """


def _write_lock(root: Path, dataset_id: str, roles: list[tuple[str, str]]) -> Path:
    source = root / "source"
    entries = []
    for role, filename in roles:
        path = source / filename
        entries.append(
            {
                "role": role,
                "filename": filename,
                "url": f"https://example.test/{filename}",
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "acquisition_date": "2026-08-29",
                "access_terms": "synthetic public connector fixture",
                "redistribution_terms": "synthetic public connector fixture",
            }
        )
    lock = root / "source-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "dataset_id": dataset_id,
                "lock_status": "complete",
                "sources": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return lock


def _trec_source(root: Path, track_id: str) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir()
    if track_id == "trec-ct-2021":
        topics_name = "topics2021.xml"
        qrels_name = "qrels2021.txt"
        archives = TRIAL_ARCHIVE_FILENAMES
        topics = (
            '<topics task="2021 TREC Clinical Trials">'
            '<topic number="1">Synthetic patient.</topic></topics>'
        )
        roles = [("topics", topics_name), ("qrels", qrels_name)] + [
            (f"trials_part_{index}", filename) for index, filename in enumerate(archives, 1)
        ]
    else:
        track = TRACKS[track_id]
        topics_name = track.topics_filename
        qrels_name = track.qrels_filename
        archives = track.trial_archive_filenames
        roles = [(role, track.expected_filenames[role]) for role in track.expected_source_roles]
        topics = (
            '<topics task="2021 TREC Clinical Trials">'
            '<topic number="1">Synthetic patient.</topic></topics>'
            if track_id == "trec-ct-2022"
            else (
                '<topics task="2023 TREC Clinical Trials">'
                '<topic number="1" template="questionnaire-v1">'
                '<field name="diagnosis">alpha</field><field name="age">42</field>'
                "</topic></topics>"
            )
        )
    (source / topics_name).write_text(topics, encoding="utf-8")
    alias = "NCT00000009" if track_id == "trec-ct-2023" else None
    qrels = "1 0 NCT00000001 2\n"
    if alias is not None:
        qrels += f"1 0 {alias} 2\n"
    (source / qrels_name).write_text(qrels, encoding="utf-8")
    for index, filename in enumerate(archives):
        with zipfile.ZipFile(source / filename, "w", compression=zipfile.ZIP_STORED) as archive:
            if index == 0:
                archive.writestr(
                    "records/NCT00000001.xml",
                    _trial_xml("NCT00000001", alias=alias),
                )
    return source, _write_lock(root, track_id, roles)


def test_all_three_trec_connector_paths_and_2023_alias_policy(tmp_path: Path) -> None:
    prepared_by_track = {}
    for track_id in ("trec-ct-2021", "trec-ct-2022", "trec-ct-2023"):
        root = tmp_path / track_id
        root.mkdir()
        source, lock = _trec_source(root, track_id)
        if track_id == "trec-ct-2021":
            result = prepare_trec_ct_2021(
                source,
                root / "prepared",
                source_lock_path=lock,
                created_at=FIXED_TIME,
            )
        else:
            result = prepare_trec_ct(
                track_id,
                source,
                root / "prepared",
                source_lock_path=lock,
                created_at=FIXED_TIME,
            )
        prepared_by_track[track_id] = load_prepared_benchmark(result.output_directory)
    assert set(prepared_by_track) == {"trec-ct-2021", "trec-ct-2022", "trec-ct-2023"}
    trec_2021 = prepared_by_track["trec-ct-2021"]
    assert trec_2021.preparation.source_recipe_id == "trec-ct-2021-snapshot-recipe-v3"
    assert trec_2021.trials[0].canonical_text.endswith("[HEALTHY_VOLUNTEERS] false")
    trec_2023 = prepared_by_track["trec-ct-2023"]
    assert trec_2023.topics[0].canonical_text == (
        "[TEMPLATE] questionnaire-v1\n[FIELD] diagnosis: alpha\n[FIELD] age: 42"
    )
    normalization = trec_2023.evaluation_package.provenance["trial_identity_normalization"]
    assert tuple(dict(item) for item in normalization["resolved_aliases"]) == (
        {"judgment_trial_id": "NCT00000009", "corpus_trial_id": "NCT00000001"},
    )
    with pytest.raises(SystemExit):
        main(
            [
                "patient-to-trial",
                "benchmark",
                "run",
                "--track",
                "trec-ct-2021",
                "--data-dir",
                str(prepared_by_track["trec-ct-2021"].directory),
                "--clinical-as-of",
                "2021-04-27T00:00:00Z",
                "--output-dir",
                str(tmp_path / "runs"),
            ]
        )


def _sigir_source(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir()
    (source / SIGIR_FILENAMES["topics_description"]).write_text(
        "<TOP><NUM>20141</NUM><TITLE>Detailed patient description.</TITLE></TOP>",
        encoding="utf-8",
    )
    (source / SIGIR_FILENAMES["topics_summary"]).write_text(
        "<TOP><NUM>20141</NUM><TITLE>Short patient summary.</TITLE></TOP>",
        encoding="utf-8",
    )
    (source / SIGIR_FILENAMES["qrels"]).write_text("20141 0 NCT00000001 2\n", encoding="utf-8")
    with tarfile.open(
        source / SIGIR_FILENAMES["trials"], mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        serialized = _trial_xml("NCT00000001").encode()
        member = tarfile.TarInfo("records/NCT00000001.xml")
        member.size = len(serialized)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(serialized))
    for role, value in {
        "adhoc_queries": "[]\n",
        "terms": "synthetic terms\n",
        "readme": "<html>synthetic</html>\n",
        "expected_relevant_counts": "topic\tcount\n",
    }.items():
        (source / SIGIR_FILENAMES[role]).write_text(value, encoding="utf-8")
    roles = [(role, SIGIR_FILENAMES[role]) for role in SIGIR_SOURCE_ROLES]
    return source, _write_lock(root, SIGIR_DATASET_ID, roles)


def test_sigir_connector_requires_distinct_description_and_summary_profiles(tmp_path: Path) -> None:
    source, lock = _sigir_source(tmp_path)
    texts = {}
    snapshots = {}
    for query_profile in ("description", "summary"):
        result = prepare_sigir_ct_2016(
            source,
            tmp_path / f"prepared-{query_profile}",
            source_lock_path=lock,
            query_variant=query_profile,
            created_at=FIXED_TIME,
        )
        prepared = load_prepared_benchmark(result.output_directory)
        texts[query_profile] = prepared.topics[0].canonical_text
        snapshots[query_profile] = prepared.snapshot.snapshot_id
        assert prepared.evaluation_package.judgment_scheme.scheme_id == (
            "sigir-clinical-trials-2016-referral-labels"
        )
    assert texts == {
        "description": "Detailed patient description.",
        "summary": "Short patient summary.",
    }
    assert snapshots["description"] != snapshots["summary"]

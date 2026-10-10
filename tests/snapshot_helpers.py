from __future__ import annotations

import io
import json
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from taim.artifacts import sha256_file
from taim.data.sigir_ct_2016 import DATASET_ID as SIGIR_DATASET_ID
from taim.data.sigir_ct_2016 import EXPECTED_SOURCE_ROLES as SIGIR_SOURCE_ROLES
from taim.data.sigir_ct_2016 import SOURCE_FILENAMES as SIGIR_SOURCE_FILENAMES

TREC_FIXED_TIME = datetime(2021, 4, 27, tzinfo=UTC)
TREC_ARCHIVE_FILENAMES = tuple(f"ClinicalTrials.2021-04-27.part{part}.zip" for part in range(1, 6))


def write_trec_source_bundle(
    root: Path,
    *,
    trials: dict[str, str],
    topics_xml: str | None = None,
    qrels: str | None = None,
) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir(parents=True)
    (source / "topics2021.xml").write_text(
        topics_xml
        or (
            '<topics task="2021 TREC Clinical Trials">'
            '<topic number="1">Adult with alpha syndrome.</topic>'
            "</topics>"
        ),
        encoding="utf-8",
    )
    (source / "qrels2021.txt").write_text(
        qrels or "1 0 NCT00000001 2\n",
        encoding="utf-8",
    )

    trial_items = sorted(trials.items())
    for index, filename in enumerate(TREC_ARCHIVE_FILENAMES):
        with zipfile.ZipFile(source / filename, "w", compression=zipfile.ZIP_STORED) as archive:
            if index == 0:
                for trial_id, serialized in trial_items:
                    member = zipfile.ZipInfo(
                        f"records/{trial_id}.xml",
                        date_time=(2021, 4, 27, 0, 0, 0),
                    )
                    member.compress_type = zipfile.ZIP_STORED
                    member.external_attr = 0o600 << 16
                    archive.writestr(member, serialized)

    source_specs = [
        ("topics", "topics2021.xml"),
        ("qrels", "qrels2021.txt"),
        *[
            (f"trials_part_{part}", filename)
            for part, filename in enumerate(TREC_ARCHIVE_FILENAMES, start=1)
        ],
    ]
    sources = []
    for role, filename in source_specs:
        path = source / filename
        sources.append(
            {
                "role": role,
                "filename": filename,
                "url": f"https://example.test/{filename}",
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "acquisition_date": "2021-04-27",
                "access_terms": "synthetic test fixture",
                "redistribution_terms": "synthetic test fixture",
            }
        )
    lock = {
        "schema_version": "2.0",
        "dataset_id": "trec-ct-2021",
        "lock_status": "complete",
        "sources": sources,
    }
    lock_path = root / "source-bundle-lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source, lock_path


def complete_trial_xml(*, trial_id: str = "NCT00000001") -> str:
    return f"""
    <clinical_study>
      <id_info><nct_id>{trial_id}</nct_id></id_info>
      <brief_title>Alpha treatment</brief_title>
      <official_title>Official alpha study</official_title>
      <brief_summary><textblock>Tests alpha therapy.</textblock></brief_summary>
      <detailed_description>
        <textblock>Extended alpha study description.</textblock>
      </detailed_description>
      <condition>Alpha syndrome</condition>
      <condition>Progressive disease</condition>
      <intervention>
        <intervention_type>Drug</intervention_type>
        <intervention_name>AlphaMed</intervention_name>
        <description>Once daily</description>
      </intervention>
      <eligibility>
        <criteria><textblock>Adults with alpha syndrome.
        Excludes cardiac disease.</textblock></criteria>
        <gender>All</gender>
        <minimum_age>18 Years</minimum_age>
        <maximum_age>65 Years</maximum_age>
        <healthy_volunteers>Yes</healthy_volunteers>
      </eligibility>
    </clinical_study>
    """


def write_sigir_source_bundle(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir()
    (source / SIGIR_SOURCE_FILENAMES["topics_description"]).write_text(
        "<TOP><NUM>20141</NUM><TITLE>First detailed patient.</TITLE></TOP>"
        "<TOP><NUM>20142</NUM><TITLE>Second detailed patient.</TITLE></TOP>",
        encoding="utf-8",
    )
    (source / SIGIR_SOURCE_FILENAMES["topics_summary"]).write_text(
        "<TOP><NUM>20141</NUM><TITLE>First summary.</TITLE></TOP>"
        "<TOP><NUM>20142</NUM><TITLE>Second summary.</TITLE></TOP>",
        encoding="utf-8",
    )
    (source / SIGIR_SOURCE_FILENAMES["qrels"]).write_text(
        "20141 0 NCT00000001 2\n20142 0 NCT00000002 0\n",
        encoding="utf-8",
    )
    archive_path = source / SIGIR_SOURCE_FILENAMES["trials"]
    with tarfile.open(archive_path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for trial_id in ("NCT00000001", "NCT00000002"):
            serialized = complete_trial_xml(trial_id=trial_id).encode()
            member = tarfile.TarInfo(f"records/{trial_id}.xml")
            member.size = len(serialized)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(serialized))
    other_contents = {
        "adhoc_queries": "[]\n",
        "terms": "synthetic terms\n",
        "readme": "<html><body>synthetic readme</body></html>\n",
        "expected_relevant_counts": "topic\tT\n",
    }
    for role, content in other_contents.items():
        (source / SIGIR_SOURCE_FILENAMES[role]).write_text(content, encoding="utf-8")

    sources = []
    for role in SIGIR_SOURCE_ROLES:
        filename = SIGIR_SOURCE_FILENAMES[role]
        path = source / filename
        sources.append(
            {
                "role": role,
                "filename": filename,
                "url": f"https://example.test/{filename}",
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "acquisition_date": "2017-04-04",
                "access_terms": "synthetic test fixture",
                "redistribution_terms": "synthetic test fixture",
            }
        )
    lock_path = root / "source-lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "dataset_id": SIGIR_DATASET_ID,
                "lock_status": "complete",
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source, lock_path

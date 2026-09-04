from __future__ import annotations

from pathlib import Path
from typing import List

from study.manifests.models import (
    DATASET_CELEB_DF_V2,
    LABEL_FAKE,
    LABEL_REAL,
    ROLE_EXTERNAL_EVALUATION,
    SOURCE_SPLIT_TEST,
    StudyVideoRecord,
)


CELEB_SOURCE_LABEL_FAKE = "0"
CELEB_SOURCE_LABEL_REAL = "1"


def _study_label_from_source_label(
    source_label: str,
) -> int:
    """
    Convert the Celeb-DF-v2 test-list label convention to the study
    convention.

    Celeb-DF-v2 test list:
        1 = real
        0 = fake

    Study convention:
        0 = real
        1 = fake
    """
    if source_label == CELEB_SOURCE_LABEL_REAL:
        return LABEL_REAL

    if source_label == CELEB_SOURCE_LABEL_FAKE:
        return LABEL_FAKE

    raise ValueError(
        "Unexpected Celeb-DF-v2 source label: {!r}."
        .format(
            source_label
        )
    )


def load_celeb_df_v2_test_records(
    list_path: Path,
) -> List[StudyVideoRecord]:
    """
    Build canonical study records from the author-provided
    Celeb-DF-v2 test list.

    Only videos listed in this file are admitted to the external
    evaluation role.
    """
    records = []
    seen_relative_paths = set()
    seen_base_video_ids = set()

    with list_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split(
                maxsplit=1
            )

            if len(parts) != 2:
                raise ValueError(
                    "Invalid Celeb-DF-v2 test-list line {}: {!r}."
                    .format(
                        line_number,
                        stripped,
                    )
                )

            source_label, relative_text = (
                parts
            )

            study_label = (
                _study_label_from_source_label(
                    source_label
                )
            )

            relative_path = Path(
                relative_text
            )

            if len(relative_path.parts) < 2:
                raise ValueError(
                    "Invalid Celeb-DF-v2 relative path "
                    "at line {}: {!r}."
                    .format(
                        line_number,
                        relative_text,
                    )
                )

            if (
                relative_text
                in seen_relative_paths
            ):
                raise ValueError(
                    "Duplicate Celeb-DF-v2 test path "
                    "at line {}: {!r}."
                    .format(
                        line_number,
                        relative_text,
                    )
                )

            base_video_id = (
                relative_path.stem
            )

            if (
                base_video_id
                in seen_base_video_ids
            ):
                raise ValueError(
                    "Duplicate Celeb-DF-v2 base_video_id "
                    "at line {}: {!r}."
                    .format(
                        line_number,
                        base_video_id,
                    )
                )

            seen_relative_paths.add(
                relative_text
            )

            seen_base_video_ids.add(
                base_video_id
            )

            records.append(
                StudyVideoRecord(
                    dataset=(
                        DATASET_CELEB_DF_V2
                    ),
                    source_split=(
                        SOURCE_SPLIT_TEST
                    ),
                    role=(
                        ROLE_EXTERNAL_EVALUATION
                    ),
                    base_video_id=(
                        base_video_id
                    ),
                    source_group_id=(
                        base_video_id
                    ),
                    source_video_id=(
                        base_video_id
                    ),
                    target_video_id=None,
                    study_label=(
                        study_label
                    ),
                    manipulation=None,
                    relative_source_path=(
                        str(relative_path)
                    ),
                    source_label=(
                        source_label
                    ),
                )
            )

    records.sort(
        key=lambda record: (
            record.study_label,
            record.relative_source_path,
        )
    )

    return records
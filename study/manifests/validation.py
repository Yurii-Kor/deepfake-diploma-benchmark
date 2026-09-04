from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Set, Tuple

from study.manifests.ffpp import (
    FFPP_DEVELOPMENT_PAIRS,
    FFPP_EXPECTED_TEST_PAIRS,
    FFPP_EXPECTED_VAL_PAIRS,
    FFPP_FIT_PAIRS,
    FFPP_MANIPULATION_DIRS,
)
from study.manifests.models import (
    DATASET_FFPP,
    FFPP_MANIPULATIONS,
    LABEL_FAKE,
    LABEL_REAL,
    ROLE_DEVELOPMENT,
    ROLE_FIT,
    ROLE_TEST,
    ROLE_VALIDATION,
    SOURCE_SPLIT_TEST,
    SOURCE_SPLIT_TRAIN,
    SOURCE_SPLIT_VAL,
    StudyVideoRecord,
    canonical_source_group_id,
)
from study.manifests.models import (
    DATASET_CELEB_DF_V2,
    DATASET_FFPP,
    FFPP_MANIPULATIONS,
    LABEL_FAKE,
    LABEL_REAL,
    ROLE_DEVELOPMENT,
    ROLE_EXTERNAL_EVALUATION,
    ROLE_FIT,
    ROLE_TEST,
    ROLE_VALIDATION,
    SOURCE_SPLIT_TEST,
    SOURCE_SPLIT_TRAIN,
    SOURCE_SPLIT_VAL,
    StudyVideoRecord,
    canonical_source_group_id,
)


class ManifestValidationError(ValueError):
    """
    Raised when a study-manifest invariant is violated.
    """


FFPP_ROLE_SOURCE_SPLIT = {
    ROLE_FIT: SOURCE_SPLIT_TRAIN,
    ROLE_DEVELOPMENT: SOURCE_SPLIT_TRAIN,
    ROLE_VALIDATION: SOURCE_SPLIT_VAL,
    ROLE_TEST: SOURCE_SPLIT_TEST,
}


FFPP_EXPECTED_GROUP_COUNTS = {
    ROLE_FIT: FFPP_FIT_PAIRS,
    ROLE_DEVELOPMENT: FFPP_DEVELOPMENT_PAIRS,
    ROLE_VALIDATION: FFPP_EXPECTED_VAL_PAIRS,
    ROLE_TEST: FFPP_EXPECTED_TEST_PAIRS,
}


FFPP_EXPECTED_ROLE_COUNTS = {
    ROLE_FIT: FFPP_FIT_PAIRS * 10,
    ROLE_DEVELOPMENT: FFPP_DEVELOPMENT_PAIRS * 10,
    ROLE_VALIDATION: FFPP_EXPECTED_VAL_PAIRS * 10,
    ROLE_TEST: FFPP_EXPECTED_TEST_PAIRS * 10,
}


FFPP_EXPECTED_REAL_COUNTS = {
    ROLE_FIT: FFPP_FIT_PAIRS * 2,
    ROLE_DEVELOPMENT: FFPP_DEVELOPMENT_PAIRS * 2,
    ROLE_VALIDATION: FFPP_EXPECTED_VAL_PAIRS * 2,
    ROLE_TEST: FFPP_EXPECTED_TEST_PAIRS * 2,
}


FFPP_EXPECTED_FAKE_COUNTS = {
    role: count * 8
    for role, count in FFPP_EXPECTED_GROUP_COUNTS.items()
}


def _source_ids_from_group_id(
    source_group_id: str,
) -> Tuple[str, str]:
    parts = source_group_id.split("_")

    if len(parts) != 2:
        raise ManifestValidationError(
            "Invalid FF++ source_group_id: {!r}"
            .format(source_group_id)
        )

    first, second = parts

    for source_id in (
        first,
        second,
    ):
        if (
            len(source_id) != 3
            or not source_id.isdigit()
        ):
            raise ManifestValidationError(
                "Invalid FF++ source ID inside source_group_id "
                "{!r}: {!r}"
                .format(
                    source_group_id,
                    source_id,
                )
            )

    canonical = canonical_source_group_id(
        first,
        second,
    )

    if canonical != source_group_id:
        raise ManifestValidationError(
            "Non-canonical FF++ source_group_id: {!r}; "
            "expected {!r}."
            .format(
                source_group_id,
                canonical,
            )
        )

    return first, second


def validate_ffpp_record_semantics(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Validate field-level semantics for canonical FF++ records.
    """
    for index, record in enumerate(
        records,
        start=1,
    ):
        if record.dataset != DATASET_FFPP:
            raise ManifestValidationError(
                "Non-FF++ record passed to FF++ validation "
                "at position {}: {!r}."
                .format(
                    index,
                    record.dataset,
                )
            )

        expected_source_split = (
            FFPP_ROLE_SOURCE_SPLIT.get(
                record.role
            )
        )

        if expected_source_split is None:
            raise ManifestValidationError(
                "Unexpected FF++ study role at position {}: {!r}."
                .format(
                    index,
                    record.role,
                )
            )

        if (
            record.source_split
            != expected_source_split
        ):
            raise ManifestValidationError(
                "FF++ role/source-split mismatch for {!r}: "
                "role={!r}, source_split={!r}, expected={!r}."
                .format(
                    record.relative_source_path,
                    record.role,
                    record.source_split,
                    expected_source_split,
                )
            )

        group_first, group_second = (
            _source_ids_from_group_id(
                record.source_group_id
            )
        )

        group_source_ids = {
            group_first,
            group_second,
        }

        if record.source_label is not None:
            raise ManifestValidationError(
                "FF++ record must not use source_label: {!r}."
                .format(
                    record.relative_source_path
                )
            )

        if record.study_label == LABEL_REAL:
            if record.manipulation is not None:
                raise ManifestValidationError(
                    "Real FF++ record has a manipulation label: {!r}."
                    .format(
                        record.relative_source_path
                    )
                )

            if record.target_video_id is not None:
                raise ManifestValidationError(
                    "Real FF++ record has target_video_id: {!r}."
                    .format(
                        record.relative_source_path
                    )
                )

            if (
                record.source_video_id
                not in group_source_ids
            ):
                raise ManifestValidationError(
                    "Real FF++ source ID {!r} is not part of "
                    "source group {!r}."
                    .format(
                        record.source_video_id,
                        record.source_group_id,
                    )
                )

            if (
                record.base_video_id
                != record.source_video_id
            ):
                raise ManifestValidationError(
                    "Real FF++ base_video_id/source_video_id mismatch: "
                    "{!r}."
                    .format(
                        record.relative_source_path
                    )
                )

            expected_path = str(
                Path("original")
                / "{}.mp4".format(
                    record.source_video_id
                )
            )

            if (
                record.relative_source_path
                != expected_path
            ):
                raise ManifestValidationError(
                    "Unexpected FF++ real source path for {!r}: "
                    "{!r}; expected {!r}."
                    .format(
                        record.base_video_id,
                        record.relative_source_path,
                        expected_path,
                    )
                )

        elif record.study_label == LABEL_FAKE:
            if (
                record.manipulation
                not in FFPP_MANIPULATIONS
            ):
                raise ManifestValidationError(
                    "Unexpected FF++ manipulation for {!r}: {!r}."
                    .format(
                        record.relative_source_path,
                        record.manipulation,
                    )
                )

            if record.target_video_id is None:
                raise ManifestValidationError(
                    "Fake FF++ record has no target_video_id: {!r}."
                    .format(
                        record.relative_source_path
                    )
                )

            if (
                record.source_video_id
                == record.target_video_id
            ):
                raise ManifestValidationError(
                    "Fake FF++ record uses identical source and "
                    "target IDs: {!r}."
                    .format(
                        record.relative_source_path
                    )
                )

            directional_source_ids = {
                record.source_video_id,
                record.target_video_id,
            }

            if (
                directional_source_ids
                != group_source_ids
            ):
                raise ManifestValidationError(
                    "Fake FF++ directional IDs do not match "
                    "source group {!r}: {!r}."
                    .format(
                        record.source_group_id,
                        record.relative_source_path,
                    )
                )

            expected_group_id = (
                canonical_source_group_id(
                    record.source_video_id,
                    record.target_video_id,
                )
            )

            if (
                record.source_group_id
                != expected_group_id
            ):
                raise ManifestValidationError(
                    "Fake FF++ source-group mismatch for {!r}: "
                    "{!r}; expected {!r}."
                    .format(
                        record.relative_source_path,
                        record.source_group_id,
                        expected_group_id,
                    )
                )

            expected_base_video_id = (
                "{}_{}".format(
                    record.source_video_id,
                    record.target_video_id,
                )
            )

            if (
                record.base_video_id
                != expected_base_video_id
            ):
                raise ManifestValidationError(
                    "Fake FF++ base_video_id mismatch for {!r}: "
                    "expected {!r}."
                    .format(
                        record.relative_source_path,
                        expected_base_video_id,
                    )
                )

            manipulation_dir = (
                FFPP_MANIPULATION_DIRS[
                    record.manipulation
                ]
            )

            expected_path = str(
                Path(manipulation_dir)
                / "{}.mp4".format(
                    record.base_video_id
                )
            )

            if (
                record.relative_source_path
                != expected_path
            ):
                raise ManifestValidationError(
                    "Unexpected FF++ fake source path for {!r}: "
                    "{!r}; expected {!r}."
                    .format(
                        record.base_video_id,
                        record.relative_source_path,
                        expected_path,
                    )
                )

        else:
            raise ManifestValidationError(
                "Unexpected FF++ study label for {!r}: {!r}."
                .format(
                    record.relative_source_path,
                    record.study_label,
                )
            )


def validate_unique_record_paths(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Require every source-media path to occur exactly once.
    """
    counts = Counter(
        (
            record.dataset,
            record.relative_source_path,
        )
        for record in records
    )

    duplicates = [
        key
        for key, count in counts.items()
        if count != 1
    ]

    if duplicates:
        raise ManifestValidationError(
            "Duplicate study-manifest source paths: {}"
            .format(
                duplicates[:20]
            )
        )


def validate_ffpp_source_ownership(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Ensure every underlying FF++ source identity belongs to exactly one
    source group, one dataset-native split, and one experimental role.

    This catches leakage even if a malformed manifest assigns the same
    source video to different pair identifiers.
    """
    source_groups = defaultdict(set)
    source_splits = defaultdict(set)
    source_roles = defaultdict(set)

    for record in records:
        source_ids = {
            record.source_video_id
        }

        if record.target_video_id is not None:
            source_ids.add(
                record.target_video_id
            )

        for source_id in source_ids:
            source_groups[source_id].add(
                record.source_group_id
            )

            source_splits[source_id].add(
                record.source_split
            )

            source_roles[source_id].add(
                record.role
            )

    for source_id in sorted(
        source_groups
    ):
        groups = source_groups[source_id]
        splits = source_splits[source_id]
        roles = source_roles[source_id]

        if len(groups) != 1:
            raise ManifestValidationError(
                "FF++ source ID {!r} appears in multiple "
                "source groups: {}."
                .format(
                    source_id,
                    sorted(groups),
                )
            )

        if len(splits) != 1:
            raise ManifestValidationError(
                "FF++ source ID {!r} appears in multiple "
                "dataset-native splits: {}."
                .format(
                    source_id,
                    sorted(splits),
                )
            )

        if len(roles) != 1:
            raise ManifestValidationError(
                "FF++ source ID {!r} appears in multiple "
                "experimental roles: {}."
                .format(
                    source_id,
                    sorted(roles),
                )
            )


def validate_ffpp_group_composition(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Validate the nominal composition of every FF++ source-pair group.

    One complete source pair must contain:
        2 real videos
        2 DeepFakes videos
        2 Face2Face videos
        2 FaceSwap videos
        2 NeuralTextures videos
    """
    grouped = defaultdict(list)

    for record in records:
        grouped[
            record.source_group_id
        ].append(record)

    for source_group_id, group_records in sorted(
        grouped.items()
    ):
        first, second = (
            _source_ids_from_group_id(
                source_group_id
            )
        )

        expected_real_ids = {
            first,
            second,
        }

        roles = {
            record.role
            for record in group_records
        }

        source_splits = {
            record.source_split
            for record in group_records
        }

        if len(roles) != 1:
            raise ManifestValidationError(
                "FF++ source group {!r} appears in multiple roles: {}."
                .format(
                    source_group_id,
                    sorted(roles),
                )
            )

        if len(source_splits) != 1:
            raise ManifestValidationError(
                "FF++ source group {!r} appears in multiple "
                "source splits: {}."
                .format(
                    source_group_id,
                    sorted(source_splits),
                )
            )

        real_records = [
            record
            for record in group_records
            if record.study_label == LABEL_REAL
        ]

        fake_records = [
            record
            for record in group_records
            if record.study_label == LABEL_FAKE
        ]

        if len(real_records) != 2:
            raise ManifestValidationError(
                "FF++ source group {!r} has {} real records; "
                "expected 2."
                .format(
                    source_group_id,
                    len(real_records),
                )
            )

        actual_real_ids = {
            record.base_video_id
            for record in real_records
        }

        if actual_real_ids != expected_real_ids:
            raise ManifestValidationError(
                "FF++ source group {!r} has real IDs {}; "
                "expected {}."
                .format(
                    source_group_id,
                    sorted(actual_real_ids),
                    sorted(expected_real_ids),
                )
            )

        if len(fake_records) != 8:
            raise ManifestValidationError(
                "FF++ source group {!r} has {} fake records; "
                "expected 8."
                .format(
                    source_group_id,
                    len(fake_records),
                )
            )

        expected_directional_ids = {
            "{}_{}".format(
                first,
                second,
            ),
            "{}_{}".format(
                second,
                first,
            ),
        }

        for manipulation in FFPP_MANIPULATIONS:
            manipulation_records = [
                record
                for record in fake_records
                if (
                    record.manipulation
                    == manipulation
                )
            ]

            if len(manipulation_records) != 2:
                raise ManifestValidationError(
                    "FF++ source group {!r}, manipulation {!r}, "
                    "has {} records; expected 2."
                    .format(
                        source_group_id,
                        manipulation,
                        len(manipulation_records),
                    )
                )

            actual_directional_ids = {
                record.base_video_id
                for record in manipulation_records
            }

            if (
                actual_directional_ids
                != expected_directional_ids
            ):
                raise ManifestValidationError(
                    "FF++ source group {!r}, manipulation {!r}, "
                    "has directional IDs {}; expected {}."
                    .format(
                        source_group_id,
                        manipulation,
                        sorted(
                            actual_directional_ids
                        ),
                        sorted(
                            expected_directional_ids
                        ),
                    )
                )


def validate_ffpp_expected_composition(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Validate the complete nominal FF++ study composition.
    """
    role_counts = Counter(
        record.role
        for record in records
    )

    role_label_counts = Counter(
        (
            record.role,
            record.study_label,
        )
        for record in records
    )

    role_groups = defaultdict(set)

    for record in records:
        role_groups[
            record.role
        ].add(
            record.source_group_id
        )

    unexpected_roles = (
        set(role_counts)
        - set(FFPP_EXPECTED_ROLE_COUNTS)
    )

    if unexpected_roles:
        raise ManifestValidationError(
            "Unexpected FF++ roles in study manifest: {}."
            .format(
                sorted(unexpected_roles)
            )
        )

    for role, expected_count in (
        FFPP_EXPECTED_ROLE_COUNTS.items()
    ):
        actual_count = role_counts.get(
            role,
            0,
        )

        if actual_count != expected_count:
            raise ManifestValidationError(
                "Unexpected FF++ record count for role {!r}: "
                "{}; expected {}."
                .format(
                    role,
                    actual_count,
                    expected_count,
                )
            )

        actual_group_count = len(
            role_groups.get(
                role,
                set(),
            )
        )

        expected_group_count = (
            FFPP_EXPECTED_GROUP_COUNTS[
                role
            ]
        )

        if (
            actual_group_count
            != expected_group_count
        ):
            raise ManifestValidationError(
                "Unexpected FF++ source-group count for role {!r}: "
                "{}; expected {}."
                .format(
                    role,
                    actual_group_count,
                    expected_group_count,
                )
            )

        actual_real_count = (
            role_label_counts.get(
                (
                    role,
                    LABEL_REAL,
                ),
                0,
            )
        )

        expected_real_count = (
            FFPP_EXPECTED_REAL_COUNTS[
                role
            ]
        )

        if (
            actual_real_count
            != expected_real_count
        ):
            raise ManifestValidationError(
                "Unexpected FF++ real-video count for role {!r}: "
                "{}; expected {}."
                .format(
                    role,
                    actual_real_count,
                    expected_real_count,
                )
            )

        actual_fake_count = (
            role_label_counts.get(
                (
                    role,
                    LABEL_FAKE,
                ),
                0,
            )
        )

        expected_fake_count = (
            FFPP_EXPECTED_FAKE_COUNTS[
                role
            ]
        )

        if (
            actual_fake_count
            != expected_fake_count
        ):
            raise ManifestValidationError(
                "Unexpected FF++ fake-video count for role {!r}: "
                "{}; expected {}."
                .format(
                    role,
                    actual_fake_count,
                    expected_fake_count,
                )
            )

    expected_total = sum(
        FFPP_EXPECTED_ROLE_COUNTS.values()
    )

    if len(records) != expected_total:
        raise ManifestValidationError(
            "Unexpected total FF++ record count: {}; expected {}."
            .format(
                len(records),
                expected_total,
            )
        )


def find_missing_source_files(
    records: Sequence[StudyVideoRecord],
    dataset_roots: Mapping[str, Path],
) -> Tuple[str, ...]:
    """
    Return source files that are absent from the configured dataset roots.

    A dataset root is the directory relative to which a record's
    ``relative_source_path`` is resolved.

    For FF++ this means the directory containing:
        original/
        Deepfakes/
        Face2Face/
        FaceSwap/
        NeuralTextures/
    """
    missing = []

    for record in records:
        dataset_root = (
            dataset_roots.get(
                record.dataset
            )
        )

        if dataset_root is None:
            raise ManifestValidationError(
                "No dataset root configured for {!r}."
                .format(
                    record.dataset
                )
            )

        source_path = (
            dataset_root
            / record.relative_source_path
        )

        if not source_path.is_file():
            missing.append(
                "{}|{}|{}".format(
                    record.dataset,
                    record.role,
                    record.relative_source_path,
                )
            )

    return tuple(
        sorted(missing)
    )


def validate_source_files(
    records: Sequence[StudyVideoRecord],
    dataset_roots: Mapping[str, Path],
) -> None:
    """
    Require all nominal source-media files to exist.
    """
    missing = find_missing_source_files(
        records=records,
        dataset_roots=dataset_roots,
    )

    if not missing:
        return

    preview = "\n".join(
        missing[:20]
    )

    suffix = ""

    if len(missing) > 20:
        suffix = (
            "\n... {} additional missing source files."
            .format(
                len(missing) - 20
            )
        )

    raise ManifestValidationError(
        "Study manifest references {} missing source files:\n{}{}"
        .format(
            len(missing),
            preview,
            suffix,
        )
    )


def validate_ffpp_records(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Run all nominal FF++ manifest invariant checks.

    File-system availability is intentionally separate because validation
    of nominal dataset identity can be performed without access to the
    media storage. Use ``validate_source_files`` when the dataset root is
    available.
    """
    validate_ffpp_record_semantics(
        records
    )

    validate_unique_record_paths(
        records
    )

    validate_ffpp_source_ownership(
        records
    )

    validate_ffpp_group_composition(
        records
    )

    validate_ffpp_expected_composition(
        records
    )

def validate_celeb_df_v2_records(
    records: Sequence[StudyVideoRecord],
) -> None:
    """
    Validate canonical Celeb-DF-v2 external-evaluation records.
    """
    seen_paths = set()
    seen_base_video_ids = set()

    for index, record in enumerate(
        records,
        start=1,
    ):
        if (
            record.dataset
            != DATASET_CELEB_DF_V2
        ):
            raise ManifestValidationError(
                "Non-Celeb-DF-v2 record passed "
                "to Celeb validation at position {}: {!r}."
                .format(
                    index,
                    record.dataset,
                )
            )

        if (
            record.source_split
            != SOURCE_SPLIT_TEST
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 record has unexpected "
                "source_split: {!r}."
                .format(
                    record.source_split
                )
            )

        if (
            record.role
            != ROLE_EXTERNAL_EVALUATION
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 record has unexpected "
                "study role: {!r}."
                .format(
                    record.role
                )
            )

        if (
            record.study_label
            not in (
                LABEL_REAL,
                LABEL_FAKE,
            )
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 record has invalid "
                "study label: {!r}."
                .format(
                    record.study_label
                )
            )

        if record.manipulation is not None:
            raise ManifestValidationError(
                "Celeb-DF-v2 record must not use "
                "FF++ manipulation metadata: {!r}."
                .format(
                    record.base_video_id
                )
            )

        if record.target_video_id is not None:
            raise ManifestValidationError(
                "Celeb-DF-v2 record must not have "
                "target_video_id: {!r}."
                .format(
                    record.base_video_id
                )
            )

        if (
            record.source_group_id
            != record.base_video_id
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 source_group_id must "
                "match base_video_id for {!r}."
                .format(
                    record.base_video_id
                )
            )

        if (
            record.source_video_id
            != record.base_video_id
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 source_video_id must "
                "match base_video_id for {!r}."
                .format(
                    record.base_video_id
                )
            )

        if record.source_label not in (
            "0",
            "1",
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 record has invalid "
                "source label: {!r}."
                .format(
                    record.source_label
                )
            )

        expected_study_label = (
            LABEL_REAL
            if record.source_label == "1"
            else LABEL_FAKE
        )

        if (
            record.study_label
            != expected_study_label
        ):
            raise ManifestValidationError(
                "Celeb-DF-v2 source/study label mismatch "
                "for {!r}: source_label={!r}, "
                "study_label={!r}."
                .format(
                    record.base_video_id,
                    record.source_label,
                    record.study_label,
                )
            )

        if (
            record.relative_source_path
            in seen_paths
        ):
            raise ManifestValidationError(
                "Duplicate Celeb-DF-v2 path: {!r}."
                .format(
                    record.relative_source_path
                )
            )

        if (
            record.base_video_id
            in seen_base_video_ids
        ):
            raise ManifestValidationError(
                "Duplicate Celeb-DF-v2 base_video_id: {!r}."
                .format(
                    record.base_video_id
                )
            )

        seen_paths.add(
            record.relative_source_path
        )

        seen_base_video_ids.add(
            record.base_video_id
        )
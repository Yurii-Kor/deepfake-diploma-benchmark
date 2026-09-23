import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from study.materialization.plan_training_frames import (
    SAMPLING_METHOD,
    TARGET_FRAME_BUDGET,
    temporal_midpoint_indices,
)


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

MANIFEST_PATH = (
    STUDY_ROOT
    / "manifests"
    / "processing_manifest.csv"
)

PLAN_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "temporal_plan.csv"
)

SUMMARY_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "temporal_plan_summary.json"
)

PREFLIGHT_PATHS = [
    (
        STUDY_ROOT
        / "preflight"
        / "ffpp"
        / "original_preflight.csv"
    ),
    (
        STUDY_ROOT
        / "preflight"
        / "ffpp"
        / "Deepfakes_preflight.csv"
    ),
    (
        STUDY_ROOT
        / "preflight"
        / "ffpp"
        / "Face2Face_preflight.csv"
    ),
    (
        STUDY_ROOT
        / "preflight"
        / "ffpp"
        / "FaceSwap_preflight.csv"
    ),
    (
        STUDY_ROOT
        / "preflight"
        / "ffpp"
        / "NeuralTextures_preflight.csv"
    ),
    (
        STUDY_ROOT
        / "preflight"
        / "celeb_df_v2"
        / "test_preflight.csv"
    ),
]

EXPECTED_TOTAL = 1358

EXPECTED_DATASET_ROLE_COUNTS = {
    ("Celeb-DF-v2", "test"): 518,
    ("FaceForensics++", "test"): 700,
    ("FaceForensics++", "validation"): 140,
}

EXPECTED_SUBGROUP_COUNTS = {
    ("Celeb-DF-v2", "test", "Celeb-real"): 108,
    (
        "Celeb-DF-v2",
        "test",
        "Celeb-synthesis",
    ): 340,
    ("Celeb-DF-v2", "test", "YouTube-real"): 70,
    ("FaceForensics++", "test", "Deepfakes"): 140,
    ("FaceForensics++", "test", "Face2Face"): 140,
    ("FaceForensics++", "test", "FaceSwap"): 140,
    (
        "FaceForensics++",
        "test",
        "NeuralTextures",
    ): 140,
    ("FaceForensics++", "test", "original"): 140,
    (
        "FaceForensics++",
        "validation",
        "original",
    ): 140,
}

EXPECTED_CONDITIONS = (
    "RSZ",
    "BLR",
    "H40",
    "PLT",
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Required CSV does not exist: {}".format(
                path
            )
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def manifest_key(row):
    return (
        row["dataset"],
        row["relative_source_path"],
    )


def plan_key(row):
    return (
        row["dataset"],
        row["relative_source_path"],
    )


def preflight_key(row):
    return (
        row["dataset"],
        row["relative_path"],
    )


def assert_unique(
    rows,
    key_function,
    description,
):
    counts = Counter(
        key_function(
            row
        )
        for row in rows
    )

    duplicates = [
        key
        for key, count in counts.items()
        if count > 1
    ]

    if duplicates:
        raise AssertionError(
            "{} contains duplicate keys: {}".format(
                description,
                duplicates[:5],
            )
        )


def load_preflight_index():
    rows = []

    for path in PREFLIGHT_PATHS:
        rows.extend(
            read_csv(
                path
            )
        )

    assert_unique(
        rows=rows,
        key_function=preflight_key,
        description="preflight",
    )

    return {
        preflight_key(
            row
        ): row
        for row in rows
    }


def validate_membership(
    manifest_rows,
    plan_rows,
):
    if len(
        manifest_rows
    ) != EXPECTED_TOTAL:
        raise AssertionError(
            "Manifest row count mismatch: "
            "expected {}, got {}".format(
                EXPECTED_TOTAL,
                len(
                    manifest_rows
                ),
            )
        )

    if len(
        plan_rows
    ) != EXPECTED_TOTAL:
        raise AssertionError(
            "Temporal-plan row count mismatch: "
            "expected {}, got {}".format(
                EXPECTED_TOTAL,
                len(
                    plan_rows
                ),
            )
        )

    assert_unique(
        rows=manifest_rows,
        key_function=manifest_key,
        description="manifest",
    )

    assert_unique(
        rows=plan_rows,
        key_function=plan_key,
        description="temporal plan",
    )

    manifest_keys = {
        manifest_key(
            row
        )
        for row in manifest_rows
    }

    plan_keys = {
        plan_key(
            row
        )
        for row in plan_rows
    }

    if manifest_keys != plan_keys:
        missing = sorted(
            manifest_keys
            - plan_keys
        )

        extra = sorted(
            plan_keys
            - manifest_keys
        )

        raise AssertionError(
            "Temporal-plan membership differs from "
            "processing manifest.\n"
            "Missing: {}\n"
            "Extra: {}".format(
                missing[:5],
                extra[:5],
            )
        )


def validate_frozen_counts(
    plan_rows,
):
    dataset_role_counts = Counter(
        (
            row["dataset"],
            row["role"],
        )
        for row in plan_rows
    )

    if dict(
        dataset_role_counts
    ) != EXPECTED_DATASET_ROLE_COUNTS:
        raise AssertionError(
            "Dataset/role counts differ from frozen "
            "evaluation contract.\n"
            "Expected: {}\n"
            "Actual: {}".format(
                EXPECTED_DATASET_ROLE_COUNTS,
                dict(
                    sorted(
                        dataset_role_counts.items()
                    )
                ),
            )
        )

    subgroup_counts = Counter(
        (
            row["dataset"],
            row["role"],
            row["subgroup"],
        )
        for row in plan_rows
    )

    if dict(
        subgroup_counts
    ) != EXPECTED_SUBGROUP_COUNTS:
        raise AssertionError(
            "Subgroup counts differ from frozen "
            "evaluation contract.\n"
            "Expected: {}\n"
            "Actual: {}".format(
                EXPECTED_SUBGROUP_COUNTS,
                dict(
                    sorted(
                        subgroup_counts.items()
                    )
                ),
            )
        )


def validate_records(
    manifest_rows,
    plan_rows,
    preflight_index,
):
    manifest_index = {
        manifest_key(
            row
        ): row
        for row in manifest_rows
    }

    valid_count = 0
    invalid_count = 0

    decoded_counts = []

    for row in plan_rows:
        key = plan_key(
            row
        )

        manifest = manifest_index[
            key
        ]

        if key not in preflight_index:
            raise AssertionError(
                "Missing preflight record for {}".format(
                    key
                )
            )

        preflight = preflight_index[
            key
        ]

        for field in (
            "dataset",
            "role",
            "subgroup",
            "study_label",
            "source_label",
            "base_video_id",
            "relative_source_path",
            "absolute_source_path",
            "conditions",
        ):
            if (
                row[field]
                != manifest[field]
            ):
                raise AssertionError(
                    "Manifest/plan mismatch for {} field {}: "
                    "manifest={!r}, plan={!r}".format(
                        key,
                        field,
                        manifest[field],
                        row[field],
                    )
                )

        if (
            preflight["subgroup"]
            != row["subgroup"]
        ):
            raise AssertionError(
                "Preflight subgroup mismatch for {}".format(
                    key
                )
            )

        plan_absolute = (
            Path(
                row[
                    "absolute_source_path"
                ]
            )
            .expanduser()
            .resolve()
        )

        preflight_absolute = (
            Path(
                preflight[
                    "absolute_path"
                ]
            )
            .expanduser()
            .resolve()
        )

        if (
            plan_absolute
            != preflight_absolute
        ):
            raise AssertionError(
                "Preflight absolute-path mismatch "
                "for {}".format(
                    key
                )
            )

        if (
            row["sampling_method"]
            != SAMPLING_METHOD
        ):
            raise AssertionError(
                "Unexpected sampling method for {}: {}".format(
                    key,
                    row[
                        "sampling_method"
                    ],
                )
            )

        if (
            int(
                row[
                    "target_frame_budget"
                ]
            )
            != TARGET_FRAME_BUDGET
        ):
            raise AssertionError(
                "Unexpected frame budget for {}".format(
                    key
                )
            )

        conditions = tuple(
            value.strip()
            for value in row[
                "conditions"
            ].split(";")
            if value.strip()
        )

        if (
            conditions
            != EXPECTED_CONDITIONS
        ):
            raise AssertionError(
                "Unexpected condition list for {}: {}".format(
                    key,
                    conditions,
                )
            )

        if (
            preflight["status"]
            != "ok"
        ):
            expected_status = "invalid"

        else:
            try:
                preflight_count = int(
                    preflight[
                        "counted_frame_count"
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):
                expected_status = "invalid"

            else:
                expected_status = (
                    "ok"
                    if (
                        preflight_count
                        >= TARGET_FRAME_BUDGET
                    )
                    else "invalid"
                )

        actual_status = row[
            "temporal_plan_status"
        ]

        if (
            actual_status
            != expected_status
        ):
            raise AssertionError(
                "Temporal-plan status mismatch for {}: "
                "expected={!r}, actual={!r}".format(
                    key,
                    expected_status,
                    actual_status,
                )
            )

        if (
            actual_status
            == "invalid"
        ):
            invalid_count += 1

            if row[
                "target_indices"
            ].strip():
                raise AssertionError(
                    "Invalid record must not contain "
                    "target indices: {}".format(
                        key
                    )
                )

            if not row[
                "failure_stage"
            ].strip():
                raise AssertionError(
                    "Invalid record is missing "
                    "failure_stage: {}".format(
                        key
                    )
                )

            if not row[
                "failure_reason"
            ].strip():
                raise AssertionError(
                    "Invalid record is missing "
                    "failure_reason: {}".format(
                        key
                    )
                )

            continue

        valid_count += 1

        decoded_frame_count = int(
            row[
                "decoded_frame_count"
            ]
        )

        decoded_counts.append(
            decoded_frame_count
        )

        if (
            decoded_frame_count
            != int(
                preflight[
                    "counted_frame_count"
                ]
            )
        ):
            raise AssertionError(
                "Decoded-frame count mismatch "
                "for {}".format(
                    key
                )
            )

        if (
            decoded_frame_count
            < TARGET_FRAME_BUDGET
        ):
            raise AssertionError(
                "Valid temporal plan has fewer than "
                "{} decoded frames: {}".format(
                    TARGET_FRAME_BUDGET,
                    key,
                )
            )

        try:
            target_indices = json.loads(
                row[
                    "target_indices"
                ]
            )

        except json.JSONDecodeError as exc:
            raise AssertionError(
                "Invalid target_indices JSON "
                "for {}".format(
                    key
                )
            ) from exc

        if not isinstance(
            target_indices,
            list,
        ):
            raise AssertionError(
                "target_indices must be a list "
                "for {}".format(
                    key
                )
            )

        if (
            len(
                target_indices
            )
            != TARGET_FRAME_BUDGET
        ):
            raise AssertionError(
                "Target-index count mismatch "
                "for {}".format(
                    key
                )
            )

        if not all(
            isinstance(
                value,
                int,
            )
            for value in target_indices
        ):
            raise AssertionError(
                "All target indices must be integers "
                "for {}".format(
                    key
                )
            )

        if (
            len(
                set(
                    target_indices
                )
            )
            != TARGET_FRAME_BUDGET
        ):
            raise AssertionError(
                "Duplicate temporal indices "
                "for {}".format(
                    key
                )
            )

        if (
            target_indices
            != sorted(
                target_indices
            )
        ):
            raise AssertionError(
                "Temporal indices are not sorted "
                "for {}".format(
                    key
                )
            )

        if (
            target_indices[0] < 0
            or target_indices[-1]
            >= decoded_frame_count
        ):
            raise AssertionError(
                "Temporal index is outside decoded "
                "frame range for {}".format(
                    key
                )
            )

        expected_indices = (
            temporal_midpoint_indices(
                frame_count=(
                    decoded_frame_count
                ),
                budget=(
                    TARGET_FRAME_BUDGET
                ),
            )
        )

        if (
            target_indices
            != expected_indices
        ):
            raise AssertionError(
                "Temporal indices do not match "
                "equal_bin_midpoint_v1 for {}\n"
                "Expected: {}\n"
                "Actual: {}".format(
                    key,
                    expected_indices,
                    target_indices,
                )
            )

        if (
            row[
                "failure_stage"
            ].strip()
        ):
            raise AssertionError(
                "Valid record contains failure_stage "
                "for {}".format(
                    key
                )
            )

        if (
            row[
                "failure_reason"
            ].strip()
        ):
            raise AssertionError(
                "Valid record contains failure_reason "
                "for {}".format(
                    key
                )
            )

    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "decoded_counts": decoded_counts,
    }


def validate_summary(
    plan_rows,
    validation_result,
):
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "Missing temporal-plan summary: {}".format(
                SUMMARY_PATH
            )
        )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if (
        summary[
            "base_video_count"
        ]
        != len(
            plan_rows
        )
    ):
        raise AssertionError(
            "Summary base_video_count mismatch."
        )

    if (
        summary[
            "target_frame_budget"
        ]
        != TARGET_FRAME_BUDGET
    ):
        raise AssertionError(
            "Summary target_frame_budget mismatch."
        )

    if (
        summary[
            "sampling_method"
        ]
        != SAMPLING_METHOD
    ):
        raise AssertionError(
            "Summary sampling_method mismatch."
        )

    actual_status_counts = dict(
        sorted(
            Counter(
                row[
                    "temporal_plan_status"
                ]
                for row in plan_rows
            ).items()
        )
    )

    if (
        summary[
            "status_counts"
        ]
        != actual_status_counts
    ):
        raise AssertionError(
            "Summary status_counts mismatch."
        )

    decoded_counts = (
        validation_result[
            "decoded_counts"
        ]
    )

    expected_min = (
        min(
            decoded_counts
        )
        if decoded_counts
        else None
    )

    expected_max = (
        max(
            decoded_counts
        )
        if decoded_counts
        else None
    )

    if (
        summary[
            "minimum_decoded_frame_count"
        ]
        != expected_min
    ):
        raise AssertionError(
            "Summary minimum decoded-frame "
            "count mismatch."
        )

    if (
        summary[
            "maximum_decoded_frame_count"
        ]
        != expected_max
    ):
        raise AssertionError(
            "Summary maximum decoded-frame "
            "count mismatch."
        )

    actual_plan_hash = sha256_file(
        PLAN_PATH
    )

    if (
        summary[
            "temporal_plan_sha256"
        ]
        != actual_plan_hash
    ):
        raise AssertionError(
            "Temporal-plan SHA-256 mismatch.\n"
            "Summary: {}\n"
            "Actual:  {}".format(
                summary[
                    "temporal_plan_sha256"
                ],
                actual_plan_hash,
            )
        )

    expected_manifest_hash = (
        sha256_file(
            MANIFEST_PATH
        )
    )

    actual_manifest_hash = (
        summary[
            "input_hashes"
        ][
            "processing_manifest.csv"
        ]
    )

    if (
        actual_manifest_hash
        != expected_manifest_hash
    ):
        raise AssertionError(
            "Processing-manifest hash mismatch."
        )

    summary_preflight_hashes = (
        summary[
            "input_hashes"
        ][
            "preflight"
        ]
    )

    for path in PREFLIGHT_PATHS:
        expected_hash = sha256_file(
            path
        )

        actual_hash = (
            summary_preflight_hashes.get(
                path.name
            )
        )

        if (
            actual_hash
            != expected_hash
        ):
            raise AssertionError(
                "Preflight hash mismatch for {}".format(
                    path
                )
            )

    return actual_plan_hash


def main():
    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    plan_rows = read_csv(
        PLAN_PATH
    )

    preflight_index = (
        load_preflight_index()
    )

    validate_membership(
        manifest_rows=manifest_rows,
        plan_rows=plan_rows,
    )

    validate_frozen_counts(
        plan_rows
    )

    validation_result = (
        validate_records(
            manifest_rows=manifest_rows,
            plan_rows=plan_rows,
            preflight_index=(
                preflight_index
            ),
        )
    )

    plan_hash = validate_summary(
        plan_rows=plan_rows,
        validation_result=(
            validation_result
        ),
    )

    print(
        "POST-TRAINING TEMPORAL PLAN VALIDATION"
    )
    print(
        "  base videos:               {}".format(
            len(
                plan_rows
            )
        )
    )
    print(
        "  valid plans:               {}".format(
            validation_result[
                "valid_count"
            ]
        )
    )
    print(
        "  invalid plans:             {}".format(
            validation_result[
                "invalid_count"
            ]
        )
    )
    print(
        "  target frames/video:       {}".format(
            TARGET_FRAME_BUDGET
        )
    )
    print(
        "  sampling method:           {}".format(
            SAMPLING_METHOD
        )
    )

    decoded_counts = (
        validation_result[
            "decoded_counts"
        ]
    )

    if decoded_counts:
        print(
            "  min decoded frames:        {}".format(
                min(
                    decoded_counts
                )
            )
        )
        print(
            "  max decoded frames:        {}".format(
                max(
                    decoded_counts
                )
            )
        )

    print(
        "  temporal-plan SHA-256:     {}".format(
            plan_hash
        )
    )

    print()
    print(
        "TEMPORAL PLAN VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
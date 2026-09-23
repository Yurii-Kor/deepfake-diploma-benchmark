import argparse
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


EXPECTED_TOTAL_BASE_VIDEOS = 1358

EXPECTED_DATASET_ROLE_COUNTS = {
    ("Celeb-DF-v2", "test"): 518,
    ("FaceForensics++", "test"): 700,
    ("FaceForensics++", "validation"): 140,
}

EXPECTED_SUBGROUP_COUNTS = {
    ("Celeb-DF-v2", "test", "Celeb-real"): 108,
    ("Celeb-DF-v2", "test", "Celeb-synthesis"): 340,
    ("Celeb-DF-v2", "test", "YouTube-real"): 70,
    ("FaceForensics++", "test", "Deepfakes"): 140,
    ("FaceForensics++", "test", "Face2Face"): 140,
    ("FaceForensics++", "test", "FaceSwap"): 140,
    ("FaceForensics++", "test", "NeuralTextures"): 140,
    ("FaceForensics++", "test", "original"): 140,
    ("FaceForensics++", "validation", "original"): 140,
}

EXPECTED_PROCESSED_CONDITIONS = (
    "RSZ",
    "BLR",
    "H40",
    "PLT",
)

OUTPUT_FIELDS = [
    "dataset",
    "role",
    "subgroup",
    "study_label",
    "source_label",
    "base_video_id",
    "relative_source_path",
    "absolute_source_path",
    "conditions",
    "decoded_frame_count",
    "target_frame_budget",
    "sampling_method",
    "target_indices",
    "temporal_plan_status",
    "failure_stage",
    "failure_reason",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen post-training temporal sampling plan "
            "for the controlled deepfake robustness study."
        )
    )

    parser.add_argument(
        "--study-root",
        type=Path,
        default=(
            Path.home()
            / "deepfake_lab"
            / "study_data"
        ),
        help=(
            "External study-data root containing manifests/ "
            "and preflight/."
        ),
    )

    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open(
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


def read_csv(path):
    path = Path(path)

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


def manifest_key(row):
    return (
        row["dataset"],
        row["relative_source_path"],
    )


def preflight_key(row):
    return (
        row["dataset"],
        row["relative_path"],
    )


def normalize_bool(value):
    return str(
        value
    ).strip().lower() == "true"


def validate_manifest(manifest_rows):
    if (
        len(manifest_rows)
        != EXPECTED_TOTAL_BASE_VIDEOS
    ):
        raise ValueError(
            "Processing manifest must contain exactly {} "
            "base videos; got {}.".format(
                EXPECTED_TOTAL_BASE_VIDEOS,
                len(manifest_rows),
            )
        )

    keys = [
        manifest_key(
            row
        )
        for row in manifest_rows
    ]

    duplicates = [
        key
        for key, count in Counter(
            keys
        ).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            "Duplicate manifest source keys detected: {}".format(
                duplicates[:5]
            )
        )

    actual_dataset_roles = Counter(
        (
            row["dataset"],
            row["role"],
        )
        for row in manifest_rows
    )

    if dict(
        actual_dataset_roles
    ) != EXPECTED_DATASET_ROLE_COUNTS:
        raise ValueError(
            "Dataset/role counts do not match the frozen "
            "evaluation contract.\n"
            "Expected: {}\n"
            "Actual: {}".format(
                EXPECTED_DATASET_ROLE_COUNTS,
                dict(
                    sorted(
                        actual_dataset_roles.items()
                    )
                ),
            )
        )

    actual_subgroups = Counter(
        (
            row["dataset"],
            row["role"],
            row["subgroup"],
        )
        for row in manifest_rows
    )

    if dict(
        actual_subgroups
    ) != EXPECTED_SUBGROUP_COUNTS:
        raise ValueError(
            "Dataset/role/subgroup counts do not match the "
            "frozen evaluation contract.\n"
            "Expected: {}\n"
            "Actual: {}".format(
                EXPECTED_SUBGROUP_COUNTS,
                dict(
                    sorted(
                        actual_subgroups.items()
                    )
                ),
            )
        )

    for row in manifest_rows:
        listed_conditions = tuple(
            condition.strip()
            for condition in row[
                "conditions"
            ].split(";")
            if condition.strip()
        )

        if (
            listed_conditions
            != EXPECTED_PROCESSED_CONDITIONS
        ):
            raise ValueError(
                "Unexpected processing-condition list for {}: "
                "{}".format(
                    manifest_key(
                        row
                    ),
                    listed_conditions,
                )
            )

        expected_base_video_id = (
            Path(
                row[
                    "relative_source_path"
                ]
            ).stem
        )

        if (
            row["base_video_id"]
            != expected_base_video_id
        ):
            raise ValueError(
                "base_video_id mismatch for {}: "
                "manifest={!r}, expected={!r}".format(
                    manifest_key(
                        row
                    ),
                    row["base_video_id"],
                    expected_base_video_id,
                )
            )


def build_preflight_index(
    preflight_rows,
):
    index = {}

    for row in preflight_rows:
        key = preflight_key(
            row
        )

        if key in index:
            raise ValueError(
                "Duplicate preflight key detected: {}".format(
                    key
                )
            )

        index[
            key
        ] = row

    return index


def validate_join(
    manifest_row,
    preflight_row,
):
    key = manifest_key(
        manifest_row
    )

    if (
        manifest_row["subgroup"]
        != preflight_row["subgroup"]
    ):
        raise ValueError(
            "Subgroup mismatch for {}: manifest={!r}, "
            "preflight={!r}".format(
                key,
                manifest_row["subgroup"],
                preflight_row["subgroup"],
            )
        )

    manifest_absolute = (
        Path(
            manifest_row[
                "absolute_source_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    preflight_absolute = (
        Path(
            preflight_row[
                "absolute_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    if (
        manifest_absolute
        != preflight_absolute
    ):
        raise ValueError(
            "Absolute source-path mismatch for {}:\n"
            "manifest={}\n"
            "preflight={}".format(
                key,
                manifest_absolute,
                preflight_absolute,
            )
        )


def build_temporal_record(
    manifest_row,
    preflight_row,
):
    validate_join(
        manifest_row=manifest_row,
        preflight_row=preflight_row,
    )

    record = {
        field: ""
        for field in OUTPUT_FIELDS
    }

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
        record[
            field
        ] = manifest_row[
            field
        ]

    record[
        "target_frame_budget"
    ] = TARGET_FRAME_BUDGET

    record[
        "sampling_method"
    ] = SAMPLING_METHOD

    preflight_status = (
        preflight_row[
            "status"
        ].strip()
    )

    if preflight_status != "ok":
        record[
            "temporal_plan_status"
        ] = "invalid"

        record[
            "failure_stage"
        ] = "source_preflight"

        record[
            "failure_reason"
        ] = (
            "source preflight status is {!r}: {}".format(
                preflight_status,
                preflight_row.get(
                    "error",
                    "",
                ),
            )
        )

        return record

    counted_value = (
        preflight_row[
            "counted_frame_count"
        ].strip()
    )

    try:
        decoded_frame_count = int(
            counted_value
        )
    except (
        TypeError,
        ValueError,
    ):
        record[
            "temporal_plan_status"
        ] = "invalid"

        record[
            "failure_stage"
        ] = "source_preflight"

        record[
            "failure_reason"
        ] = (
            "counted_frame_count is not a valid integer: "
            "{!r}".format(
                counted_value
            )
        )

        return record

    record[
        "decoded_frame_count"
    ] = decoded_frame_count

    if (
        decoded_frame_count
        < TARGET_FRAME_BUDGET
    ):
        record[
            "temporal_plan_status"
        ] = "invalid"

        record[
            "failure_stage"
        ] = "temporal_plan"

        record[
            "failure_reason"
        ] = (
            "decoded frame count {} is below target "
            "frame budget {}".format(
                decoded_frame_count,
                TARGET_FRAME_BUDGET,
            )
        )

        return record

    if not normalize_bool(
        preflight_row[
            "has_32_frames"
        ]
    ):
        raise ValueError(
            "Preflight inconsistency for {}: decoded_frame_count={} "
            "but has_32_frames={!r}".format(
                manifest_key(
                    manifest_row
                ),
                decoded_frame_count,
                preflight_row[
                    "has_32_frames"
                ],
            )
        )

    target_indices = (
        temporal_midpoint_indices(
            frame_count=decoded_frame_count,
            budget=TARGET_FRAME_BUDGET,
        )
    )

    record[
        "target_indices"
    ] = json.dumps(
        target_indices,
        separators=(
            ",",
            ":",
        ),
    )

    record[
        "temporal_plan_status"
    ] = "ok"

    return record


def main():
    args = parse_args()

    study_root = (
        args.study_root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        study_root
        / "manifests"
        / "processing_manifest.csv"
    )

    preflight_paths = [
        (
            study_root
            / "preflight"
            / "ffpp"
            / "original_preflight.csv"
        ),
        (
            study_root
            / "preflight"
            / "ffpp"
            / "Deepfakes_preflight.csv"
        ),
        (
            study_root
            / "preflight"
            / "ffpp"
            / "Face2Face_preflight.csv"
        ),
        (
            study_root
            / "preflight"
            / "ffpp"
            / "FaceSwap_preflight.csv"
        ),
        (
            study_root
            / "preflight"
            / "ffpp"
            / "NeuralTextures_preflight.csv"
        ),
        (
            study_root
            / "preflight"
            / "celeb_df_v2"
            / "test_preflight.csv"
        ),
    ]

    output_dir = (
        study_root
        / "evaluation"
    )

    output_path = (
        output_dir
        / "temporal_plan.csv"
    )

    summary_path = (
        output_dir
        / "temporal_plan_summary.json"
    )

    manifest_rows = read_csv(
        manifest_path
    )

    validate_manifest(
        manifest_rows
    )

    preflight_rows = []

    for path in preflight_paths:
        preflight_rows.extend(
            read_csv(
                path
            )
        )

    preflight_index = (
        build_preflight_index(
            preflight_rows
        )
    )

    output_rows = []

    missing_preflight = []

    for manifest_row in manifest_rows:
        key = manifest_key(
            manifest_row
        )

        preflight_row = (
            preflight_index.get(
                key
            )
        )

        if preflight_row is None:
            missing_preflight.append(
                key
            )

            continue

        output_rows.append(
            build_temporal_record(
                manifest_row=manifest_row,
                preflight_row=preflight_row,
            )
        )

    if missing_preflight:
        raise ValueError(
            "Manifest rows are missing corresponding preflight "
            "records: {}".format(
                missing_preflight[:5]
            )
        )

    if (
        len(output_rows)
        != len(manifest_rows)
    ):
        raise RuntimeError(
            "Temporal-plan row count mismatch: "
            "manifest={}, output={}".format(
                len(manifest_rows),
                len(output_rows),
            )
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    status_counts = Counter(
        row[
            "temporal_plan_status"
        ]
        for row in output_rows
    )

    valid_counts = [
        int(
            row[
                "decoded_frame_count"
            ]
        )
        for row in output_rows
        if (
            row[
                "temporal_plan_status"
            ]
            == "ok"
        )
    ]

    summary = {
        "schema_version": 1,
        "base_video_count": len(
            output_rows
        ),
        "target_frame_budget": (
            TARGET_FRAME_BUDGET
        ),
        "sampling_method": (
            SAMPLING_METHOD
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "minimum_decoded_frame_count": (
            min(
                valid_counts
            )
            if valid_counts
            else None
        ),
        "maximum_decoded_frame_count": (
            max(
                valid_counts
            )
            if valid_counts
            else None
        ),
        "dataset_role_counts": {
            "{}|{}".format(
                dataset,
                role,
            ): count
            for (
                dataset,
                role,
            ), count in sorted(
                Counter(
                    (
                        row[
                            "dataset"
                        ],
                        row[
                            "role"
                        ],
                    )
                    for row in output_rows
                ).items()
            )
        },
        "subgroup_counts": {
            "{}|{}|{}".format(
                dataset,
                role,
                subgroup,
            ): count
            for (
                dataset,
                role,
                subgroup,
            ), count in sorted(
                Counter(
                    (
                        row[
                            "dataset"
                        ],
                        row[
                            "role"
                        ],
                        row[
                            "subgroup"
                        ],
                    )
                    for row in output_rows
                ).items()
            )
        },
        "input_hashes": {
            "processing_manifest.csv": (
                sha256_file(
                    manifest_path
                )
            ),
            "preflight": {
                path.name: (
                    sha256_file(
                        path
                    )
                )
                for path in preflight_paths
            },
        },
        "temporal_plan_sha256": (
            sha256_file(
                output_path
            )
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    print(
        "POST-TRAINING TEMPORAL PLAN"
    )
    print(
        "  base videos:               {}".format(
            len(
                output_rows
            )
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
    print(
        "  status counts:             {}".format(
            dict(
                sorted(
                    status_counts.items()
                )
            )
        )
    )

    if valid_counts:
        print(
            "  min decoded frames:        {}".format(
                min(
                    valid_counts
                )
            )
        )
        print(
            "  max decoded frames:        {}".format(
                max(
                    valid_counts
                )
            )
        )

    print()
    print(
        "  temporal plan:             {}".format(
            output_path
        )
    )
    print(
        "  summary:                   {}".format(
            summary_path
        )
    )
    print(
        "  temporal-plan SHA-256:     {}".format(
            summary[
                "temporal_plan_sha256"
            ]
        )
    )

    print()
    print(
        "POST-TRAINING TEMPORAL PLAN BUILD PASSED"
    )


if __name__ == "__main__":
    main()
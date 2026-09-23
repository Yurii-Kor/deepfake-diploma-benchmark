import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from study.materialization.face_alignment import (
    ALIGNMENT_METHOD,
    ALIGNMENT_SCALE,
    DETECTOR_UPSAMPLE,
    LANDMARK_INDICES,
    OUTPUT_SIZE,
    align_face_bgr,
    load_dlib_face_components,
)
from study.processing.processing_common import (
    RawVideoReader,
    probe_video,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

DEFAULT_PREDICTOR_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "dlib_tools"
    / "shape_predictor_81_face_landmarks.dat"
)

TARGET_FRAME_BUDGET = 32

EXPECTED_BASE_VIDEOS = 1358
EXPECTED_TARGETS = (
    EXPECTED_BASE_VIDEOS
    * TARGET_FRAME_BUDGET
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
    "temporal_position",
    "source_frame_index",
    "source_frame_sha256",
    "alignment_method",
    "geometry_status",
    "face_count",
    "bbox_json",
    "keypoints_json",
    "affine_matrix_json",
    "aligned_height",
    "aligned_width",
    "aligned_channels",
    "aligned_sha256",
    "failure_stage",
    "failure_reason",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen clean-reference face geometry "
            "for the post-training robustness study."
        )
    )

    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
    )

    parser.add_argument(
        "--predictor-path",
        type=Path,
        default=DEFAULT_PREDICTOR_PATH,
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help=(
            "Optional deterministic prefix limit for smoke testing. "
            "Omit for the full 1358-video build."
        ),
    )

    parser.add_argument(
        "--output-name",
        default="clean_geometry.csv",
        help=(
            "Output CSV filename inside study_root/evaluation/."
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


def sha256_array(array):
    array = np.ascontiguousarray(
        array
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def json_cell(value):
    if value is None:
        return ""

    return json.dumps(
        value,
        separators=(
            ",",
            ":",
        ),
    )


def read_temporal_plan(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Temporal plan does not exist: {}".format(
                path
            )
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    if not rows:
        raise RuntimeError(
            "Temporal plan is empty."
        )

    return rows


def parse_target_indices(row):
    if (
        row[
            "temporal_plan_status"
        ]
        != "ok"
    ):
        raise RuntimeError(
            "Clean geometry cannot be built from "
            "an invalid temporal plan: {}".format(
                (
                    row[
                        "dataset"
                    ],
                    row[
                        "relative_source_path"
                    ],
                )
            )
        )

    try:
        target_indices = json.loads(
            row[
                "target_indices"
            ]
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid target_indices JSON for {}".format(
                row[
                    "relative_source_path"
                ]
            )
        ) from exc

    if not isinstance(
        target_indices,
        list,
    ):
        raise RuntimeError(
            "target_indices must be a list."
        )

    if (
        len(
            target_indices
        )
        != TARGET_FRAME_BUDGET
    ):
        raise RuntimeError(
            "Expected {} temporal positions for {}, "
            "got {}.".format(
                TARGET_FRAME_BUDGET,
                row[
                    "relative_source_path"
                ],
                len(
                    target_indices
                ),
            )
        )

    if not all(
        isinstance(
            value,
            int,
        )
        for value in target_indices
    ):
        raise RuntimeError(
            "All temporal indices must be integers."
        )

    if (
        target_indices
        != sorted(
            target_indices
        )
    ):
        raise RuntimeError(
            "Temporal indices must be sorted."
        )

    if (
        len(
            set(
                target_indices
            )
        )
        != TARGET_FRAME_BUDGET
    ):
        raise RuntimeError(
            "Temporal indices must be unique."
        )

    return target_indices


def base_record(
    video_row,
    temporal_position,
    source_frame_index,
):
    return {
        "dataset": (
            video_row[
                "dataset"
            ]
        ),
        "role": (
            video_row[
                "role"
            ]
        ),
        "subgroup": (
            video_row[
                "subgroup"
            ]
        ),
        "study_label": (
            video_row[
                "study_label"
            ]
        ),
        "source_label": (
            video_row[
                "source_label"
            ]
        ),
        "base_video_id": (
            video_row[
                "base_video_id"
            ]
        ),
        "relative_source_path": (
            video_row[
                "relative_source_path"
            ]
        ),
        "absolute_source_path": (
            video_row[
                "absolute_source_path"
            ]
        ),
        "temporal_position": (
            temporal_position
        ),
        "source_frame_index": (
            source_frame_index
        ),
        "source_frame_sha256": "",
        "alignment_method": (
            ALIGNMENT_METHOD
        ),
        "geometry_status": "invalid",
        "face_count": "",
        "bbox_json": "",
        "keypoints_json": "",
        "affine_matrix_json": "",
        "aligned_height": "",
        "aligned_width": "",
        "aligned_channels": "",
        "aligned_sha256": "",
        "failure_stage": "",
        "failure_reason": "",
    }


def alignment_record(
    video_row,
    temporal_position,
    source_frame_index,
    frame_bgr,
    face_detector,
    predictor,
):
    record = base_record(
        video_row=video_row,
        temporal_position=temporal_position,
        source_frame_index=(
            source_frame_index
        ),
    )

    record[
        "source_frame_sha256"
    ] = sha256_array(
        frame_bgr
    )

    result = align_face_bgr(
        frame_bgr=frame_bgr,
        face_detector=face_detector,
        predictor=predictor,
    )

    record[
        "face_count"
    ] = result.face_count

    record[
        "bbox_json"
    ] = json_cell(
        (
            list(
                result.bbox
            )
            if result.bbox is not None
            else None
        )
    )

    record[
        "keypoints_json"
    ] = json_cell(
        (
            result.keypoints.tolist()
            if result.keypoints is not None
            else None
        )
    )

    record[
        "affine_matrix_json"
    ] = json_cell(
        (
            result.affine_matrix.tolist()
            if result.affine_matrix is not None
            else None
        )
    )

    if not result.ok:
        record[
            "failure_stage"
        ] = (
            result.failure_stage
            or "unknown"
        )

        record[
            "failure_reason"
        ] = (
            result.failure_reason
            or "unspecified clean-reference alignment failure"
        )

        return record

    aligned = (
        result.aligned_bgr
    )

    if aligned is None:
        raise RuntimeError(
            "Successful alignment returned no "
            "aligned image."
        )

    if (
        aligned.dtype
        != np.uint8
    ):
        raise RuntimeError(
            "Aligned image must have dtype uint8."
        )

    if (
        aligned.shape
        != (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
            3,
        )
    ):
        raise RuntimeError(
            "Unexpected aligned-image shape: {}".format(
                aligned.shape
            )
        )

    if (
        result.bbox is None
        or result.keypoints is None
        or result.affine_matrix is None
    ):
        raise RuntimeError(
            "Successful alignment is missing "
            "geometry metadata."
        )

    if (
        result.keypoints.shape
        != (
            5,
            2,
        )
    ):
        raise RuntimeError(
            "Successful alignment has unexpected "
            "keypoint shape: {}".format(
                result.keypoints.shape
            )
        )

    if (
        result.affine_matrix.shape
        != (
            2,
            3,
        )
    ):
        raise RuntimeError(
            "Successful alignment has unexpected "
            "affine-matrix shape: {}".format(
                result.affine_matrix.shape
            )
        )

    if not np.isfinite(
        result.keypoints
    ).all():
        raise RuntimeError(
            "Successful alignment contains "
            "non-finite keypoints."
        )

    if not np.isfinite(
        result.affine_matrix
    ).all():
        raise RuntimeError(
            "Successful alignment contains "
            "non-finite affine values."
        )

    record[
        "geometry_status"
    ] = "valid"

    record[
        "aligned_height"
    ] = int(
        aligned.shape[0]
    )

    record[
        "aligned_width"
    ] = int(
        aligned.shape[1]
    )

    record[
        "aligned_channels"
    ] = int(
        aligned.shape[2]
    )

    record[
        "aligned_sha256"
    ] = sha256_array(
        aligned
    )

    return record


def process_video(
    video_row,
    face_detector,
    predictor,
):
    target_indices = (
        parse_target_indices(
            video_row
        )
    )

    expected_decoded_count = int(
        video_row[
            "decoded_frame_count"
        ]
    )

    source_path = (
        Path(
            video_row[
                "absolute_source_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            "Source video does not exist: {}".format(
                source_path
            )
        )

    probe = probe_video(
        source_path
    )

    width = int(
        probe[
            "width"
        ]
    )

    height = int(
        probe[
            "height"
        ]
    )

    target_by_index = {
        source_frame_index: (
            temporal_position
        )
        for (
            temporal_position,
            source_frame_index,
        ) in enumerate(
            target_indices
        )
    }

    records_by_position = {}

    decoded_index = 0

    with RawVideoReader(
        path=source_path,
        width=width,
        height=height,
    ) as reader:
        while True:
            frame_bgr = (
                reader.read_frame()
            )

            if frame_bgr is None:
                break

            if (
                decoded_index
                in target_by_index
            ):
                temporal_position = (
                    target_by_index[
                        decoded_index
                    ]
                )

                records_by_position[
                    temporal_position
                ] = alignment_record(
                    video_row=video_row,
                    temporal_position=(
                        temporal_position
                    ),
                    source_frame_index=(
                        decoded_index
                    ),
                    frame_bgr=frame_bgr,
                    face_detector=(
                        face_detector
                    ),
                    predictor=predictor,
                )

            decoded_index += 1

    if (
        decoded_index
        != expected_decoded_count
    ):
        raise RuntimeError(
            "Runtime decoded-frame count mismatch for {}: "
            "plan={}, runtime={}.".format(
                video_row[
                    "relative_source_path"
                ],
                expected_decoded_count,
                decoded_index,
            )
        )

    if (
        len(
            records_by_position
        )
        != TARGET_FRAME_BUDGET
    ):
        missing_positions = [
            position
            for position in range(
                TARGET_FRAME_BUDGET
            )
            if (
                position
                not in records_by_position
            )
        ]

        raise RuntimeError(
            "Not all frozen temporal positions were "
            "decoded for {}. Missing positions: {}".format(
                video_row[
                    "relative_source_path"
                ],
                missing_positions,
            )
        )

    return [
        records_by_position[
            position
        ]
        for position in range(
            TARGET_FRAME_BUDGET
        )
    ]


def write_summary(
    path,
    *,
    temporal_plan_path,
    predictor_path,
    output_path,
    selected_video_count,
    records,
):
    status_counts = Counter(
        row[
            "geometry_status"
        ]
        for row in records
    )

    failure_stages = Counter(
        row[
            "failure_stage"
        ]
        for row in records
        if row[
            "geometry_status"
        ]
        == "invalid"
    )

    valid_by_video = Counter()

    all_video_keys = set()

    for row in records:
        key = (
            row[
                "dataset"
            ],
            row[
                "role"
            ],
            row[
                "relative_source_path"
            ],
        )

        all_video_keys.add(
            key
        )

        if (
            row[
                "geometry_status"
            ]
            == "valid"
        ):
            valid_by_video[
                key
            ] += 1

    valid_count_distribution = Counter(
        valid_by_video.get(
            key,
            0,
        )
        for key in all_video_keys
    )

    clean_valid_counts = [
        valid_by_video.get(
            key,
            0,
        )
        for key in all_video_keys
    ]

    summary = {
        "schema_version": 1,
        "alignment_method": (
            ALIGNMENT_METHOD
        ),
        "output_size": (
            OUTPUT_SIZE
        ),
        "alignment_scale": (
            ALIGNMENT_SCALE
        ),
        "detector_upsample": (
            DETECTOR_UPSAMPLE
        ),
        "landmark_indices": list(
            LANDMARK_INDICES
        ),
        "temporal_plan_sha256": (
            sha256_file(
                temporal_plan_path
            )
        ),
        "predictor_sha256": (
            sha256_file(
                predictor_path
            )
        ),
        "selected_video_count": (
            selected_video_count
        ),
        "targets_per_video": (
            TARGET_FRAME_BUDGET
        ),
        "nominal_target_count": len(
            records
        ),
        "geometry_status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "failure_stages": dict(
            sorted(
                failure_stages.items()
            )
        ),
        "clean_valid_frame_count_distribution": {
            str(
                count
            ): videos
            for (
                count,
                videos,
            ) in sorted(
                valid_count_distribution.items()
            )
        },
        "minimum_clean_valid_frames": (
            min(
                clean_valid_counts
            )
            if clean_valid_counts
            else None
        ),
        "maximum_clean_valid_frames": (
            max(
                clean_valid_counts
            )
            if clean_valid_counts
            else None
        ),
        "complete_32_video_count": (
            valid_count_distribution.get(
                TARGET_FRAME_BUDGET,
                0,
            )
        ),
        "zero_valid_video_count": (
            valid_count_distribution.get(
                0,
                0,
            )
        ),
        "clean_geometry_sha256": (
            sha256_file(
                output_path
            )
        ),
    }

    with path.open(
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

    return summary


def main():
    args = parse_args()

    study_root = (
        args.study_root
        .expanduser()
        .resolve()
    )

    predictor_path = (
        args.predictor_path
        .expanduser()
        .resolve()
    )

    temporal_plan_path = (
        study_root
        / "evaluation"
        / "temporal_plan.csv"
    )

    output_dir = (
        study_root
        / "evaluation"
    )

    output_path = (
        output_dir
        / args.output_name
    )

    summary_path = (
        output_path.with_name(
            "{}_summary.json".format(
                output_path.stem
            )
        )
    )

    inprogress_path = (
        output_path.with_name(
            "{}.inprogress.csv".format(
                output_path.stem
            )
        )
    )

    if not predictor_path.is_file():
        raise FileNotFoundError(
            "dlib predictor does not exist: {}".format(
                predictor_path
            )
        )

    plan_rows = read_temporal_plan(
        temporal_plan_path
    )

    if (
        args.limit_videos is None
    ):
        if (
            len(
                plan_rows
            )
            != EXPECTED_BASE_VIDEOS
        ):
            raise RuntimeError(
                "Expected {} evaluation videos, got {}.".format(
                    EXPECTED_BASE_VIDEOS,
                    len(
                        plan_rows
                    ),
                )
            )

        selected_rows = (
            plan_rows
        )

    else:
        if (
            args.limit_videos
            <= 0
        ):
            raise ValueError(
                "--limit-videos must be positive."
            )

        selected_rows = (
            plan_rows[
                :args.limit_videos
            ]
        )

    expected_targets = (
        len(
            selected_rows
        )
        * TARGET_FRAME_BUDGET
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if inprogress_path.exists():
        inprogress_path.unlink()

    face_detector, predictor = (
        load_dlib_face_components(
            predictor_path
        )
    )

    all_records = []

    print(
        "CLEAN-REFERENCE GEOMETRY BUILD"
    )
    print(
        "  videos:                    {}".format(
            len(
                selected_rows
            )
        )
    )
    print(
        "  nominal targets:           {}".format(
            expected_targets
        )
    )
    print(
        "  alignment method:          {}".format(
            ALIGNMENT_METHOD
        )
    )
    print(
        "  output size:               {}".format(
            OUTPUT_SIZE
        )
    )
    print(
        "  predictor SHA-256:         {}".format(
            sha256_file(
                predictor_path
            )
        )
    )
    print()

    with inprogress_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()

        for (
            index,
            video_row,
        ) in enumerate(
            selected_rows,
            start=1,
        ):
            video_records = (
                process_video(
                    video_row=video_row,
                    face_detector=(
                        face_detector
                    ),
                    predictor=predictor,
                )
            )

            writer.writerows(
                video_records
            )

            all_records.extend(
                video_records
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

            valid_count = sum(
                row[
                    "geometry_status"
                ]
                == "valid"
                for row in video_records
            )

            print(
                "  [{}/{}] {} | {} | {} | {} "
                "clean-valid={}/32".format(
                    index,
                    len(
                        selected_rows
                    ),
                    video_row[
                        "dataset"
                    ],
                    video_row[
                        "role"
                    ],
                    video_row[
                        "subgroup"
                    ],
                    video_row[
                        "base_video_id"
                    ],
                    valid_count,
                )
            )

    if (
        len(
            all_records
        )
        != expected_targets
    ):
        raise RuntimeError(
            "Geometry-record count mismatch: "
            "expected {}, got {}.".format(
                expected_targets,
                len(
                    all_records
                ),
            )
        )

    os.replace(
        str(
            inprogress_path
        ),
        str(
            output_path
        ),
    )

    summary = write_summary(
        path=summary_path,
        temporal_plan_path=(
            temporal_plan_path
        ),
        predictor_path=(
            predictor_path
        ),
        output_path=(
            output_path
        ),
        selected_video_count=len(
            selected_rows
        ),
        records=all_records,
    )

    print()
    print(
        "CLEAN-REFERENCE GEOMETRY RESULT"
    )
    print(
        "  records:                   {}".format(
            len(
                all_records
            )
        )
    )
    print(
        "  status counts:             {}".format(
            summary[
                "geometry_status_counts"
            ]
        )
    )
    print(
        "  failure stages:            {}".format(
            summary[
                "failure_stages"
            ]
        )
    )
    print(
        "  clean-valid distribution:  {}".format(
            summary[
                "clean_valid_frame_count_distribution"
            ]
        )
    )
    print(
        "  complete-32 videos:        {}".format(
            summary[
                "complete_32_video_count"
            ]
        )
    )
    print(
        "  zero-valid videos:         {}".format(
            summary[
                "zero_valid_video_count"
            ]
        )
    )
    print()
    print(
        "  geometry CSV:              {}".format(
            output_path
        )
    )
    print(
        "  summary:                   {}".format(
            summary_path
        )
    )
    print(
        "  geometry SHA-256:          {}".format(
            summary[
                "clean_geometry_sha256"
            ]
        )
    )
    print()
    print(
        "CLEAN-REFERENCE GEOMETRY BUILD PASSED"
    )


if __name__ == "__main__":
    main()
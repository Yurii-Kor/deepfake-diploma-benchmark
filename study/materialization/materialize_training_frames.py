import argparse
import csv
import hashlib
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path

import cv2

from study.materialization.face_alignment import (
    ALIGNMENT_METHOD,
    OUTPUT_SIZE,
    align_face_bgr,
    load_dlib_face_components,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
    / "frame_target_plan.csv"
)

DEFAULT_PREDICTOR_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "dlib_tools"
    / "shape_predictor_81_face_landmarks.dat"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "study"
    / "training"
    / "materialized"
    / "rgb"
)

DEFAULT_AUDIT_DIR = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
)

EXPECTED_VIDEOS = 3600
EXPECTED_TARGETS_PER_VIDEO = 32
EXPECTED_TARGETS = (
    EXPECTED_VIDEOS
    * EXPECTED_TARGETS_PER_VIDEO
)

EXPECTED_ROLE_VIDEO_COUNTS = {
    "fit": 2880,
    "development": 720,
}

PLAN_FIELDS = (
    "view",
    "adapter_dataset",
    "execution_split",
    "dataset",
    "source_split",
    "role",
    "base_video_id",
    "source_group_id",
    "source_video_id",
    "target_video_id",
    "study_label",
    "manipulation",
    "relative_source_path",
    "temporal_position",
    "source_frame_index",
    "output_relative_path",
    "status",
)

AUDIT_FIELDS = PLAN_FIELDS + (
    "alignment_method",
    "materialization_status",
    "face_count",
    "bbox_json",
    "keypoints_json",
    "affine_matrix_json",
    "output_height",
    "output_width",
    "output_channels",
    "output_bytes",
    "output_sha256",
    "failure_stage",
    "failure_reason",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the frozen 32-frame FF++ "
            "FIT/DEV study plan using controlled "
            "dlib face alignment."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "FF++ root containing original/, "
            "Deepfakes/, Face2Face/, FaceSwap/, "
            "and NeuralTextures/"
        ),
    )

    parser.add_argument(
        "--plan-path",
        type=Path,
        default=DEFAULT_PLAN_PATH,
    )

    parser.add_argument(
        "--predictor-path",
        type=Path,
        default=DEFAULT_PREDICTOR_PATH,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
    )

    return parser.parse_args()


def sha256_file(
    path,
):
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def json_cell(
    value,
):
    if value is None:
        return ""

    return json.dumps(
        value,
        separators=(
            ",",
            ":",
        ),
    )


def load_and_validate_plan(
    plan_path,
):
    rows = []

    with Path(plan_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        missing = (
            set(
                PLAN_FIELDS
            )
            - set(
                fieldnames
            )
        )

        if missing:
            raise ValueError(
                "frame plan is missing columns: {}".format(
                    sorted(
                        missing
                    )
                )
            )

        for row in reader:
            rows.append(
                row
            )

    if len(
        rows
    ) != EXPECTED_TARGETS:
        raise ValueError(
            "expected {} frame targets, found {}".format(
                EXPECTED_TARGETS,
                len(
                    rows
                ),
            )
        )

    groups = OrderedDict()

    output_paths = set()

    for row in rows:
        if row[
            "view"
        ] != "aggregate":
            raise ValueError(
                "materialization plan must use "
                "the aggregate adapter view"
            )

        if row[
            "adapter_dataset"
        ] != "FaceForensics++":
            raise ValueError(
                "unexpected adapter dataset: {}".format(
                    row[
                        "adapter_dataset"
                    ]
                )
            )

        if row[
            "status"
        ] != "planned":
            raise ValueError(
                "unexpected plan status: {}".format(
                    row[
                        "status"
                    ]
                )
            )

        relative_source_path = row[
            "relative_source_path"
        ]

        groups.setdefault(
            relative_source_path,
            [],
        ).append(
            row
        )

        output_relative_path = row[
            "output_relative_path"
        ]

        if (
            output_relative_path
            in output_paths
        ):
            raise ValueError(
                "duplicate output path: {}".format(
                    output_relative_path
                )
            )

        output_paths.add(
            output_relative_path
        )

    if len(
        groups
    ) != EXPECTED_VIDEOS:
        raise ValueError(
            "expected {} videos, found {}".format(
                EXPECTED_VIDEOS,
                len(
                    groups
                ),
            )
        )

    role_counts = Counter()

    for (
        relative_source_path,
        video_rows,
    ) in groups.items():
        if len(
            video_rows
        ) != EXPECTED_TARGETS_PER_VIDEO:
            raise ValueError(
                "{} has {} targets; expected {}".format(
                    relative_source_path,
                    len(
                        video_rows
                    ),
                    EXPECTED_TARGETS_PER_VIDEO,
                )
            )

        temporal_positions = sorted(
            int(
                row[
                    "temporal_position"
                ]
            )
            for row in video_rows
        )

        if temporal_positions != list(
            range(
                EXPECTED_TARGETS_PER_VIDEO
            )
        ):
            raise ValueError(
                "{} does not contain temporal "
                "positions 0..31".format(
                    relative_source_path
                )
            )

        source_indices = [
            int(
                row[
                    "source_frame_index"
                ]
            )
            for row in video_rows
        ]

        if len(
            set(
                source_indices
            )
        ) != EXPECTED_TARGETS_PER_VIDEO:
            raise ValueError(
                "{} contains duplicate source "
                "frame indices".format(
                    relative_source_path
                )
            )

        roles = {
            row[
                "role"
            ]
            for row in video_rows
        }

        if len(
            roles
        ) != 1:
            raise ValueError(
                "{} has multiple roles: {}".format(
                    relative_source_path,
                    sorted(
                        roles
                    ),
                )
            )

        role_counts[
            next(
                iter(
                    roles
                )
            )
        ] += 1

    if dict(
        role_counts
    ) != EXPECTED_ROLE_VIDEO_COUNTS:
        raise ValueError(
            "unexpected role/video counts: {}".format(
                dict(
                    role_counts
                )
            )
        )

    return (
        rows,
        groups,
    )


def base_audit_record(
    row,
):
    record = {
        field: row[
            field
        ]
        for field in PLAN_FIELDS
    }

    record.update(
        {
            "alignment_method": ALIGNMENT_METHOD,
            "materialization_status": "failed",
            "face_count": "",
            "bbox_json": "",
            "keypoints_json": "",
            "affine_matrix_json": "",
            "output_height": "",
            "output_width": "",
            "output_channels": "",
            "output_bytes": "",
            "output_sha256": "",
            "failure_stage": "",
            "failure_reason": "",
        }
    )

    return record


def remove_existing_output(
    output_path,
):
    if output_path.exists():
        output_path.unlink()


def failure_record(
    row,
    output_root,
    stage,
    reason,
):
    output_path = (
        output_root
        / row[
            "output_relative_path"
        ]
    )

    remove_existing_output(
        output_path
    )

    record = base_audit_record(
        row
    )

    record[
        "failure_stage"
    ] = stage

    record[
        "failure_reason"
    ] = reason

    return record


def materialize_target(
    row,
    frame_bgr,
    output_root,
    face_detector,
    predictor,
):
    record = base_audit_record(
        row
    )

    output_path = (
        output_root
        / row[
            "output_relative_path"
        ]
    )

    remove_existing_output(
        output_path
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
        ] = result.failure_stage

        record[
            "failure_reason"
        ] = result.failure_reason

        return record

    if result.aligned_bgr is None:
        record[
            "failure_stage"
        ] = "alignment_output"

        record[
            "failure_reason"
        ] = (
            "successful alignment result "
            "contains no image"
        )

        return record

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = cv2.imwrite(
        str(
            output_path
        ),
        result.aligned_bgr,
    )

    if not written:
        remove_existing_output(
            output_path
        )

        record[
            "failure_stage"
        ] = "png_write"

        record[
            "failure_reason"
        ] = (
            "cv2.imwrite returned false"
        )

        return record

    readback = cv2.imread(
        str(
            output_path
        ),
        cv2.IMREAD_COLOR,
    )

    if readback is None:
        remove_existing_output(
            output_path
        )

        record[
            "failure_stage"
        ] = "png_readback"

        record[
            "failure_reason"
        ] = (
            "written PNG could not "
            "be read back"
        )

        return record

    if readback.shape != (
        OUTPUT_SIZE,
        OUTPUT_SIZE,
        3,
    ):
        remove_existing_output(
            output_path
        )

        record[
            "failure_stage"
        ] = "png_readback"

        record[
            "failure_reason"
        ] = (
            "unexpected PNG shape: {}".format(
                readback.shape
            )
        )

        return record

    record[
        "output_height"
    ] = int(
        readback.shape[0]
    )

    record[
        "output_width"
    ] = int(
        readback.shape[1]
    )

    record[
        "output_channels"
    ] = int(
        readback.shape[2]
    )

    record[
        "output_bytes"
    ] = output_path.stat().st_size

    record[
        "output_sha256"
    ] = sha256_file(
        output_path
    )

    record[
        "materialization_status"
    ] = "ok"

    return record


def materialize_video(
    relative_source_path,
    video_rows,
    source_root,
    output_root,
    face_detector,
    predictor,
):
    ordered_rows = sorted(
        video_rows,
        key=lambda row: int(
            row[
                "source_frame_index"
            ]
        ),
    )

    video_path = (
        source_root
        / relative_source_path
    )

    if not video_path.is_file():
        return [
            failure_record(
                row=row,
                output_root=output_root,
                stage="source_video",
                reason="source video does not exist",
            )
            for row in ordered_rows
        ]

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        capture.release()

        return [
            failure_record(
                row=row,
                output_root=output_root,
                stage="video_open",
                reason="cv2.VideoCapture could not open video",
            )
            for row in ordered_rows
        ]

    records = []

    target_position = 0
    decoded_index = -1

    try:
        while (
            target_position
            < len(
                ordered_rows
            )
        ):
            ok, frame_bgr = (
                capture.read()
            )

            if not ok:
                break

            decoded_index += 1

            target_row = ordered_rows[
                target_position
            ]

            target_index = int(
                target_row[
                    "source_frame_index"
                ]
            )

            if decoded_index < target_index:
                continue

            if decoded_index > target_index:
                raise RuntimeError(
                    "decode index advanced past "
                    "planned target {} for {}".format(
                        target_index,
                        relative_source_path,
                    )
                )

            records.append(
                materialize_target(
                    row=target_row,
                    frame_bgr=frame_bgr,
                    output_root=output_root,
                    face_detector=face_detector,
                    predictor=predictor,
                )
            )

            target_position += 1

    finally:
        capture.release()

    if (
        target_position
        < len(
            ordered_rows
        )
    ):
        for row in ordered_rows[
            target_position:
        ]:
            records.append(
                failure_record(
                    row=row,
                    output_root=output_root,
                    stage="frame_decode",
                    reason=(
                        "video ended before target "
                        "frame {}; last decoded "
                        "index was {}".format(
                            row[
                                "source_frame_index"
                            ],
                            decoded_index,
                        )
                    ),
                )
            )

    if len(
        records
    ) != EXPECTED_TARGETS_PER_VIDEO:
        raise RuntimeError(
            "{} produced {} audit records; "
            "expected {}".format(
                relative_source_path,
                len(
                    records
                ),
                EXPECTED_TARGETS_PER_VIDEO,
            )
        )

    return records


def write_summary(
    summary_path,
    *,
    plan_path,
    predictor_path,
    source_root,
    output_root,
    total,
    succeeded,
    failed,
    failure_stages,
):
    summary = {
        "alignment_method": ALIGNMENT_METHOD,
        "output_size": OUTPUT_SIZE,
        "plan_sha256": sha256_file(
            plan_path
        ),
        "predictor_sha256": sha256_file(
            predictor_path
        ),
        "source_root": str(
            source_root
        ),
        "output_root": str(
            output_root
        ),
        "expected_videos": EXPECTED_VIDEOS,
        "targets_per_video": (
            EXPECTED_TARGETS_PER_VIDEO
        ),
        "expected_targets": EXPECTED_TARGETS,
        "audit_rows": total,
        "successful_targets": succeeded,
        "failed_targets": failed,
        "failure_stages": dict(
            sorted(
                failure_stages.items()
            )
        ),
        "status": (
            "passed"
            if failed == 0
            else "failed"
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


def main():
    args = parse_args()

    source_root = (
        args.source_root
        .expanduser()
        .resolve()
    )

    plan_path = (
        args.plan_path
        .expanduser()
        .resolve()
    )

    predictor_path = (
        args.predictor_path
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    audit_dir = (
        args.audit_dir
        .expanduser()
        .resolve()
    )

    if not source_root.is_dir():
        raise FileNotFoundError(
            "source root not found: {}".format(
                source_root
            )
        )

    if not plan_path.is_file():
        raise FileNotFoundError(
            "frame plan not found: {}".format(
                plan_path
            )
        )

    if not predictor_path.is_file():
        raise FileNotFoundError(
            "dlib predictor not found: {}".format(
                predictor_path
            )
        )

    rows, groups = (
        load_and_validate_plan(
            plan_path
        )
    )

    print(
        "FULL TRAINING FRAME MATERIALIZATION"
    )
    print(
        "  videos:                    {}".format(
            len(
                groups
            )
        )
    )
    print(
        "  targets/video:             {}".format(
            EXPECTED_TARGETS_PER_VIDEO
        )
    )
    print(
        "  total targets:             {}".format(
            len(
                rows
            )
        )
    )
    print(
        "  alignment:                 {}".format(
            ALIGNMENT_METHOD
        )
    )
    print(
        "  output root:               {}".format(
            output_root
        )
    )
    print()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        audit_dir
        / "frame_materialization.csv"
    )

    inprogress_path = (
        audit_dir
        / "frame_materialization.inprogress.csv"
    )

    summary_path = (
        audit_dir
        / "frame_materialization_summary.json"
    )

    if inprogress_path.exists():
        inprogress_path.unlink()

    face_detector, predictor = (
        load_dlib_face_components(
            predictor_path
        )
    )

    total = 0
    succeeded = 0
    failed = 0

    failure_stages = Counter()

    with inprogress_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=AUDIT_FIELDS,
        )

        writer.writeheader()

        for video_number, (
            relative_source_path,
            video_rows,
        ) in enumerate(
            groups.items(),
            start=1,
        ):
            records = materialize_video(
                relative_source_path=(
                    relative_source_path
                ),
                video_rows=video_rows,
                source_root=source_root,
                output_root=output_root,
                face_detector=face_detector,
                predictor=predictor,
            )

            for record in records:
                writer.writerow(
                    record
                )

                total += 1

                if (
                    record[
                        "materialization_status"
                    ]
                    == "ok"
                ):
                    succeeded += 1
                else:
                    failed += 1

                    failure_stages[
                        record[
                            "failure_stage"
                        ]
                    ] += 1

            file.flush()

            if (
                video_number % 50 == 0
                or video_number
                == EXPECTED_VIDEOS
            ):
                print(
                    "  videos {:4d}/{:4d} | "
                    "targets {:6d}/{:6d} | "
                    "ok {:6d} | "
                    "failed {:6d}".format(
                        video_number,
                        EXPECTED_VIDEOS,
                        total,
                        EXPECTED_TARGETS,
                        succeeded,
                        failed,
                    )
                )

    if total != EXPECTED_TARGETS:
        raise RuntimeError(
            "materialization produced {} audit "
            "rows; expected {}".format(
                total,
                EXPECTED_TARGETS,
            )
        )

    os.replace(
        str(
            inprogress_path
        ),
        str(
            audit_path
        ),
    )

    write_summary(
        summary_path=summary_path,
        plan_path=plan_path,
        predictor_path=predictor_path,
        source_root=source_root,
        output_root=output_root,
        total=total,
        succeeded=succeeded,
        failed=failed,
        failure_stages=failure_stages,
    )

    print()
    print(
        "MATERIALIZATION RESULT"
    )
    print(
        "  audit rows:                {}".format(
            total
        )
    )
    print(
        "  successful targets:        {}".format(
            succeeded
        )
    )
    print(
        "  failed targets:            {}".format(
            failed
        )
    )
    print(
        "  audit:                     {}".format(
            audit_path
        )
    )
    print(
        "  summary:                   {}".format(
            summary_path
        )
    )

    if failed:
        print(
            "  failure stages:            {}".format(
                dict(
                    sorted(
                        failure_stages.items()
                    )
                )
            )
        )

        raise RuntimeError(
            "{} of {} planned frames failed "
            "controlled materialization".format(
                failed,
                total,
            )
        )

    print()
    print(
        "FULL TRAINING FRAME MATERIALIZATION PASSED"
    )


if __name__ == "__main__":
    main()
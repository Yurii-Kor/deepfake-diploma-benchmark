import argparse
import csv
import json
from pathlib import Path

import cv2

from study.materialization.face_alignment import (
    ALIGNMENT_METHOD,
    load_dlib_face_components,
    align_face_bgr,
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
    / "materialization"
    / "artifacts"
    / "smoke_outputs"
)

SMOKE_TEMPORAL_POSITION = 16

FAKE_MANIPULATIONS = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)

EXPECTED_CASES = (
    ("fit", "REAL"),
    ("fit", "DeepFakes"),
    ("fit", "Face2Face"),
    ("fit", "FaceSwap"),
    ("fit", "NeuralTextures"),
    ("development", "REAL"),
    ("development", "DeepFakes"),
    ("development", "Face2Face"),
    ("development", "FaceSwap"),
    ("development", "NeuralTextures"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic real-frame smoke tests "
            "for the controlled FF++ face alignment."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "root containing FF++ original/, "
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

    return parser.parse_args()


def smoke_category(row):
    if row["study_label"] == "0":
        return "REAL"

    return row["manipulation"]


def select_smoke_rows(
    plan_path,
):
    selected = {}

    with Path(plan_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        required_columns = {
            "role",
            "study_label",
            "manipulation",
            "relative_source_path",
            "temporal_position",
            "source_frame_index",
            "output_relative_path",
            "status",
            "base_video_id",
        }

        missing = (
            required_columns
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise ValueError(
                "frame target plan is missing columns: {}".format(
                    sorted(
                        missing
                    )
                )
            )

        for row in reader:
            if row["status"] != "planned":
                continue

            if int(
                row["temporal_position"]
            ) != SMOKE_TEMPORAL_POSITION:
                continue

            key = (
                row["role"],
                smoke_category(
                    row
                ),
            )

            if (
                key in EXPECTED_CASES
                and key not in selected
            ):
                selected[
                    key
                ] = row

            if len(
                selected
            ) == len(
                EXPECTED_CASES
            ):
                break

    missing_cases = [
        key
        for key in EXPECTED_CASES
        if key not in selected
    ]

    if missing_cases:
        raise RuntimeError(
            "could not select all smoke cases: {}".format(
                missing_cases
            )
        )

    return [
        selected[
            key
        ]
        for key in EXPECTED_CASES
    ]


def decode_exact_frame(
    video_path,
    target_index,
):
    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            "could not open video: {}".format(
                video_path
            )
        )

    decoded_index = -1

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            decoded_index += 1

            if decoded_index == target_index:
                return frame

    finally:
        capture.release()

    raise RuntimeError(
        "target frame {} was not decoded from {} "
        "(last decoded index: {})".format(
            target_index,
            video_path,
            decoded_index,
        )
    )


def bbox_to_list(
    bbox,
):
    if bbox is None:
        return None

    return [
        int(
            value
        )
        for value in bbox
    ]


def array_to_list(
    value,
):
    if value is None:
        return None

    return value.tolist()


def make_output_path(
    output_root,
    row,
    category,
):
    source_frame_index = int(
        row["source_frame_index"]
    )

    return (
        output_root
        / row["role"]
        / category
        / row["base_video_id"]
        / "{:06d}.png".format(
            source_frame_index
        )
    )


def run_smoke_case(
    row,
    source_root,
    output_root,
    face_detector,
    predictor,
):
    category = smoke_category(
        row
    )

    relative_source_path = Path(
        row[
            "relative_source_path"
        ]
    )

    video_path = (
        source_root
        / relative_source_path
    )

    source_frame_index = int(
        row[
            "source_frame_index"
        ]
    )

    output_path = make_output_path(
        output_root=output_root,
        row=row,
        category=category,
    )

    audit = {
        "role": row["role"],
        "category": category,
        "manipulation": row[
            "manipulation"
        ],
        "base_video_id": row[
            "base_video_id"
        ],
        "relative_source_path": (
            relative_source_path.as_posix()
        ),
        "temporal_position": int(
            row[
                "temporal_position"
            ]
        ),
        "source_frame_index": (
            source_frame_index
        ),
        "plan_output_relative_path": row[
            "output_relative_path"
        ],
        "alignment_method": (
            ALIGNMENT_METHOD
        ),
        "status": "failed",
        "face_count": None,
        "bbox": None,
        "keypoints": None,
        "affine_matrix": None,
        "smoke_output_path": None,
        "failure_stage": None,
        "failure_reason": None,
    }

    if not video_path.is_file():
        audit[
            "failure_stage"
        ] = "source_video"

        audit[
            "failure_reason"
        ] = (
            "source video does not exist"
        )

        return audit

    try:
        frame_bgr = decode_exact_frame(
            video_path=video_path,
            target_index=source_frame_index,
        )

    except Exception as exc:
        audit[
            "failure_stage"
        ] = "frame_decode"

        audit[
            "failure_reason"
        ] = str(
            exc
        )

        return audit

    result = align_face_bgr(
        frame_bgr=frame_bgr,
        face_detector=face_detector,
        predictor=predictor,
    )

    audit[
        "face_count"
    ] = result.face_count

    audit[
        "bbox"
    ] = bbox_to_list(
        result.bbox
    )

    audit[
        "keypoints"
    ] = array_to_list(
        result.keypoints
    )

    audit[
        "affine_matrix"
    ] = array_to_list(
        result.affine_matrix
    )

    if not result.ok:
        audit[
            "failure_stage"
        ] = result.failure_stage

        audit[
            "failure_reason"
        ] = result.failure_reason

        return audit

    if result.aligned_bgr is None:
        audit[
            "failure_stage"
        ] = "alignment_output"

        audit[
            "failure_reason"
        ] = (
            "successful result has no aligned image"
        )

        return audit

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
        audit[
            "failure_stage"
        ] = "png_write"

        audit[
            "failure_reason"
        ] = (
            "cv2.imwrite returned false"
        )

        return audit

    written_image = cv2.imread(
        str(
            output_path
        ),
        cv2.IMREAD_COLOR,
    )

    if written_image is None:
        audit[
            "failure_stage"
        ] = "png_readback"

        audit[
            "failure_reason"
        ] = (
            "written PNG could not be read back"
        )

        return audit

    if written_image.shape != (
        256,
        256,
        3,
    ):
        audit[
            "failure_stage"
        ] = "png_readback"

        audit[
            "failure_reason"
        ] = (
            "unexpected written image shape: {}".format(
                written_image.shape
            )
        )

        return audit

    audit[
        "status"
    ] = "ok"

    audit[
        "smoke_output_path"
    ] = str(
        output_path.relative_to(
            PROJECT_ROOT
        )
    )

    return audit


def write_audit(
    output_root,
    records,
):
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        output_root
        / "smoke_audit.json"
    )

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    return audit_path


def print_results(
    records,
):
    print(
        "REAL FF++ FACE ALIGNMENT SMOKE"
    )

    print(
        "  temporal position:         {}".format(
            SMOKE_TEMPORAL_POSITION
        )
    )

    print(
        "  cases:                     {}".format(
            len(
                records
            )
        )
    )

    print()

    header = (
        "{:<12} {:<15} {:<22} {:>6} {:>5}  {}"
    )

    print(
        header.format(
            "role",
            "category",
            "video",
            "frame",
            "faces",
            "status",
        )
    )

    print(
        "-" * 78
    )

    for record in records:
        print(
            header.format(
                record[
                    "role"
                ],
                record[
                    "category"
                ],
                record[
                    "base_video_id"
                ],
                record[
                    "source_frame_index"
                ],
                (
                    "-"
                    if record[
                        "face_count"
                    ] is None
                    else record[
                        "face_count"
                    ]
                ),
                record[
                    "status"
                ],
            )
        )

        if (
            record[
                "status"
            ]
            != "ok"
        ):
            print(
                "  failure: {}: {}".format(
                    record[
                        "failure_stage"
                    ],
                    record[
                        "failure_reason"
                    ],
                )
            )

    print()


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

    if not source_root.is_dir():
        raise FileNotFoundError(
            "source root not found: {}".format(
                source_root
            )
        )

    if not plan_path.is_file():
        raise FileNotFoundError(
            "frame target plan not found: {}".format(
                plan_path
            )
        )

    rows = select_smoke_rows(
        plan_path
    )

    face_detector, predictor = (
        load_dlib_face_components(
            predictor_path
        )
    )

    records = []

    for row in rows:
        records.append(
            run_smoke_case(
                row=row,
                source_root=source_root,
                output_root=output_root,
                face_detector=face_detector,
                predictor=predictor,
            )
        )

    audit_path = write_audit(
        output_root=output_root,
        records=records,
    )

    print_results(
        records
    )

    failed = [
        record
        for record in records
        if record[
            "status"
        ] != "ok"
    ]

    print(
        "  audit:                     {}".format(
            audit_path
        )
    )

    if failed:
        raise RuntimeError(
            "{} of {} smoke cases failed".format(
                len(
                    failed
                ),
                len(
                    records
                ),
            )
        )

    print()
    print(
        "REAL FACE ALIGNMENT SMOKE PASSED"
    )


if __name__ == "__main__":
    main()
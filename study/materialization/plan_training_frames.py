import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "study"
    / "manifests"
    / "artifacts"
    / "deepfakebench_membership.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "study"
    / "materialization"
    / "artifacts"
)

VIDEO_PLAN_PATH = (
    ARTIFACT_DIR
    / "video_frame_plan.csv"
)

FRAME_PLAN_PATH = (
    ARTIFACT_DIR
    / "frame_target_plan.csv"
)

SUMMARY_PATH = (
    ARTIFACT_DIR
    / "frame_plan_summary.json"
)

TARGET_FRAME_BUDGET = 32

SAMPLING_METHOD = (
    "equal_bin_midpoint_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic 32-frame FIT/DEV "
            "materialization plan for FaceForensics++."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "FaceForensics++ root containing original/, "
            "Deepfakes/, Face2Face/, FaceSwap/, "
            "and NeuralTextures/."
        ),
    )

    return parser.parse_args()


VIDEO_FIELDS = [
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
    "absolute_source_path",
    "reported_frame_count",
    "fps",
    "width",
    "height",
    "target_frame_budget",
    "sampling_method",
    "target_indices",
    "output_video_dir",
    "status",
    "failure_reason",
]


FRAME_FIELDS = [
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
]


def sha256_file(
    path,
):
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


def temporal_midpoint_indices(
    frame_count,
    budget,
):
    if not isinstance(
        frame_count,
        int,
    ):
        raise TypeError(
            "frame_count must be an integer."
        )

    if not isinstance(
        budget,
        int,
    ):
        raise TypeError(
            "budget must be an integer."
        )

    if budget <= 0:
        raise ValueError(
            "budget must be positive."
        )

    if frame_count < budget:
        raise ValueError(
            "frame_count must be at least the target budget."
        )

    indices = [
        (
            (
                2 * position
                + 1
            )
            * frame_count
        )
        // (
            2
            * budget
        )
        for position in range(
            budget
        )
    ]

    if len(
        indices
    ) != budget:
        raise AssertionError(
            "Unexpected target-index count."
        )

    if len(
        set(
            indices
        )
    ) != budget:
        raise AssertionError(
            "Temporal sampling produced duplicate indices."
        )

    if indices != sorted(
        indices
    ):
        raise AssertionError(
            "Temporal sampling is not monotonic."
        )

    if indices[
        0
    ] < 0:
        raise AssertionError(
            "Temporal sampling produced a negative index."
        )

    if indices[
        -1
    ] >= frame_count:
        raise AssertionError(
            "Temporal sampling exceeded the video frame count."
        )

    return indices


def load_aggregate_membership():
    with MEMBERSHIP_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    aggregate_rows = [
        row
        for row in rows
        if row[
            "view"
        ]
        == "aggregate"
    ]

    split_order = {
        "train": 0,
        "test": 1,
    }

    aggregate_rows.sort(
        key=lambda row: (
            split_order.get(
                row[
                    "execution_split"
                ],
                99,
            ),
            row[
                "relative_source_path"
            ],
        )
    )

    relative_paths = [
        row[
            "relative_source_path"
        ]
        for row in aggregate_rows
    ]

    if len(
        aggregate_rows
    ) != 3600:
        raise ValueError(
            "Expected 3600 aggregate FIT/DEV rows, "
            f"found {len(aggregate_rows)}."
        )

    if len(
        set(
            relative_paths
        )
    ) != 3600:
        raise ValueError(
            "Aggregate membership contains duplicate source paths."
        )

    return aggregate_rows


def inspect_video(
    source_path,
):
    if not source_path.is_file():
        return {
            "reported_frame_count": "",
            "fps": "",
            "width": "",
            "height": "",
            "status": "source_missing",
            "failure_reason": (
                "source file does not exist"
            ),
        }

    capture = cv2.VideoCapture(
        str(
            source_path
        )
    )

    try:
        if not capture.isOpened():
            return {
                "reported_frame_count": "",
                "fps": "",
                "width": "",
                "height": "",
                "status": "video_open_failed",
                "failure_reason": (
                    "cv2.VideoCapture could not open source"
                ),
            }

        frame_count_raw = capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        width = capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        height = capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )

        if (
            not math.isfinite(
                frame_count_raw
            )
            or frame_count_raw <= 0
        ):
            return {
                "reported_frame_count": "",
                "fps": fps,
                "width": width,
                "height": height,
                "status": "invalid_frame_count",
                "failure_reason": (
                    "CAP_PROP_FRAME_COUNT is not positive and finite"
                ),
            }

        frame_count = int(
            frame_count_raw
        )

        if frame_count < (
            TARGET_FRAME_BUDGET
        ):
            return {
                "reported_frame_count": frame_count,
                "fps": fps,
                "width": int(
                    width
                ),
                "height": int(
                    height
                ),
                "status": (
                    "insufficient_reported_frames"
                ),
                "failure_reason": (
                    "reported frame count is below "
                    f"{TARGET_FRAME_BUDGET}"
                ),
            }

        if (
            width <= 0
            or height <= 0
        ):
            return {
                "reported_frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "status": "invalid_video_geometry",
                "failure_reason": (
                    "reported width/height are not positive"
                ),
            }

        return {
            "reported_frame_count": frame_count,
            "fps": fps,
            "width": int(
                width
            ),
            "height": int(
                height
            ),
            "status": "planned",
            "failure_reason": "",
        }

    finally:
        capture.release()


def output_video_dir_for(
    relative_source_path,
):
    source_relative = Path(
        relative_source_path
    )

    without_suffix = (
        source_relative.with_suffix(
            ""
        )
    )

    return (
        Path(
            "FaceForensics++"
        )
        / without_suffix
    ).as_posix()


def main():
    args = parse_args()

    source_root = (
        args.source_root
        .expanduser()
        .resolve()
    )

    if not source_root.is_dir():
        raise FileNotFoundError(
            "source root not found: {}".format(
                source_root
            )
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    membership_rows = (
        load_aggregate_membership()
    )

    video_rows = []
    frame_rows = []

    total = len(
        membership_rows
    )

    print(
        "FRAME MATERIALIZATION PLAN"
    )

    print(
        f"  aggregate videos:          {total}"
    )

    print(
        f"  target frames/video:       {TARGET_FRAME_BUDGET}"
    )

    print(
        f"  sampling method:           {SAMPLING_METHOD}"
    )

    print()

    for (
        row_index,
        membership_row,
    ) in enumerate(
        membership_rows,
        start=1,
    ):
        relative_source_path = (
            membership_row[
                "relative_source_path"
            ]
        )

        source_path = (
            source_root
            / relative_source_path
        )

        inspection = inspect_video(
            source_path
        )

        output_video_dir = (
            output_video_dir_for(
                relative_source_path
            )
        )

        target_indices = []

        if (
            inspection[
                "status"
            ]
            == "planned"
        ):
            target_indices = (
                temporal_midpoint_indices(
                    frame_count=inspection[
                        "reported_frame_count"
                    ],
                    budget=(
                        TARGET_FRAME_BUDGET
                    ),
                )
            )

        video_row = {
            key: membership_row.get(
                key,
                "",
            )
            for key in [
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
            ]
        }

        video_row.update(
            {
                "absolute_source_path": str(
                    source_path
                ),
                "reported_frame_count": (
                    inspection[
                        "reported_frame_count"
                    ]
                ),
                "fps": (
                    inspection[
                        "fps"
                    ]
                ),
                "width": (
                    inspection[
                        "width"
                    ]
                ),
                "height": (
                    inspection[
                        "height"
                    ]
                ),
                "target_frame_budget": (
                    TARGET_FRAME_BUDGET
                ),
                "sampling_method": (
                    SAMPLING_METHOD
                ),
                "target_indices": (
                    json.dumps(
                        target_indices,
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                ),
                "output_video_dir": (
                    output_video_dir
                ),
                "status": (
                    inspection[
                        "status"
                    ]
                ),
                "failure_reason": (
                    inspection[
                        "failure_reason"
                    ]
                ),
            }
        )

        video_rows.append(
            video_row
        )

        for (
            temporal_position,
            source_frame_index,
        ) in enumerate(
            target_indices
        ):
            output_relative_path = (
                Path(
                    output_video_dir
                )
                / (
                    f"{source_frame_index:06d}.png"
                )
            ).as_posix()

            frame_rows.append(
                {
                    "view": membership_row[
                        "view"
                    ],
                    "adapter_dataset": (
                        membership_row[
                            "adapter_dataset"
                        ]
                    ),
                    "execution_split": (
                        membership_row[
                            "execution_split"
                        ]
                    ),
                    "dataset": membership_row[
                        "dataset"
                    ],
                    "source_split": (
                        membership_row[
                            "source_split"
                        ]
                    ),
                    "role": membership_row[
                        "role"
                    ],
                    "base_video_id": (
                        membership_row[
                            "base_video_id"
                        ]
                    ),
                    "source_group_id": (
                        membership_row[
                            "source_group_id"
                        ]
                    ),
                    "source_video_id": (
                        membership_row[
                            "source_video_id"
                        ]
                    ),
                    "target_video_id": (
                        membership_row[
                            "target_video_id"
                        ]
                    ),
                    "study_label": (
                        membership_row[
                            "study_label"
                        ]
                    ),
                    "manipulation": (
                        membership_row[
                            "manipulation"
                        ]
                    ),
                    "relative_source_path": (
                        relative_source_path
                    ),
                    "temporal_position": (
                        temporal_position
                    ),
                    "source_frame_index": (
                        source_frame_index
                    ),
                    "output_relative_path": (
                        output_relative_path
                    ),
                    "status": "planned",
                }
            )

        if (
            row_index % 500
            == 0
            or row_index
            == total
        ):
            print(
                f"  inspected:                 "
                f"{row_index}/{total}"
            )

    with VIDEO_PLAN_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=VIDEO_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            video_rows
        )

    with FRAME_PLAN_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FRAME_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            frame_rows
        )

    status_counts = Counter(
        row[
            "status"
        ]
        for row in video_rows
    )

    summary = {
        "schema_version": 1,
        "source_membership": {
            "path": str(
                MEMBERSHIP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256_file(
                MEMBERSHIP_PATH
            ),
        },
        "source_root": str(
            source_root
        ),
        "sampling": {
            "method": (
                SAMPLING_METHOD
            ),
            "target_frame_budget": (
                TARGET_FRAME_BUDGET
            ),
            "definition": (
                "floor((2*k+1)*N/(2*K)), "
                "k=0..K-1"
            ),
        },
        "counts": {
            "aggregate_video_rows": len(
                video_rows
            ),
            "planned_video_rows": int(
                status_counts.get(
                    "planned",
                    0,
                )
            ),
            "frame_target_rows": len(
                frame_rows
            ),
            "execution_split": dict(
                Counter(
                    row[
                        "execution_split"
                    ]
                    for row in video_rows
                )
            ),
            "role": dict(
                Counter(
                    row[
                        "role"
                    ]
                    for row in video_rows
                )
            ),
            "study_label": dict(
                Counter(
                    row[
                        "study_label"
                    ]
                    for row in video_rows
                )
            ),
            "manipulation": dict(
                Counter(
                    (
                        row[
                            "manipulation"
                        ]
                        or "REAL"
                    )
                    for row in video_rows
                )
            ),
            "status": dict(
                status_counts
            ),
        },
    }

    summary[
        "outputs"
    ] = {
        "video_frame_plan.csv": {
            "sha256": sha256_file(
                VIDEO_PLAN_PATH
            )
        },
        "frame_target_plan.csv": {
            "sha256": sha256_file(
                FRAME_PLAN_PATH
            )
        },
    }

    with SUMMARY_PATH.open(
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

    failed_video_count = (
        len(
            video_rows
        )
        - status_counts.get(
            "planned",
            0,
        )
    )

    print()
    print(
        "PLAN RESULT"
    )

    print(
        f"  video rows:                "
        f"{len(video_rows)}"
    )

    print(
        f"  planned videos:            "
        f"{status_counts.get('planned', 0)}"
    )

    print(
        f"  non-ready videos:          "
        f"{failed_video_count}"
    )

    print(
        f"  target-frame rows:         "
        f"{len(frame_rows)}"
    )

    print()

    print(
        f"  video CSV:                 "
        f"{VIDEO_PLAN_PATH}"
    )

    print(
        f"  frame CSV:                 "
        f"{FRAME_PLAN_PATH}"
    )

    print(
        f"  summary:                   "
        f"{SUMMARY_PATH}"
    )

    if failed_video_count:
        raise RuntimeError(
            "One or more videos could not be "
            "included in the temporal plan. "
            "Inspect video_frame_plan.csv."
        )

    expected_frame_rows = (
        3600
        * TARGET_FRAME_BUDGET
    )

    if len(
        frame_rows
    ) != expected_frame_rows:
        raise AssertionError(
            "Expected "
            f"{expected_frame_rows} "
            "frame targets, found "
            f"{len(frame_rows)}."
        )

    print()

    print(
        "FRAME MATERIALIZATION PLAN PASSED"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

FRAME_RECORDS_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "frame_scoring_smoke.jsonl"
)

CLEAN_GEOMETRY_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "clean_geometry.csv"
)

VIDEO_RECORDS_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "video_scoring_smoke.jsonl"
)

SUMMARY_PATH = (
    STUDY_ROOT
    / "evaluation"
    / "video_scoring_smoke_summary.json"
)

EXPECTED_DETECTORS = {
    "xception",
    "ucf",
    "spsl",
}

EXPECTED_CONDITIONS = {
    "CLN",
    "RSZ",
}

EXPECTED_VIDEO_COUNT = 3
EXPECTED_VIDEO_RECORD_COUNT = (
    len(EXPECTED_DETECTORS)
    * len(EXPECTED_CONDITIONS)
    * EXPECTED_VIDEO_COUNT
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


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Required JSONL does not exist: {}".format(
                path
            )
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSON at {} line {}.".format(
                        path,
                        line_number,
                    )
                ) from exc

            rows.append(
                row
            )

    return rows


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


def write_jsonl(
    path,
    records,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )
            file.write(
                "\n"
            )


def base_video_key(row):
    return (
        row["dataset"],
        row["role"],
        row["relative_source_path"],
    )


def group_key(row):
    return (
        row["detector"],
        row["dataset"],
        row["role"],
        row["relative_source_path"],
        row["condition"],
    )


def load_clean_valid_positions():
    rows = read_csv(
        CLEAN_GEOMETRY_PATH
    )

    result = defaultdict(
        set
    )

    for row in rows:
        if (
            row["geometry_status"]
            != "valid"
        ):
            continue

        key = base_video_key(
            row
        )

        position = int(
            row[
                "temporal_position"
            ]
        )

        if position in result[
            key
        ]:
            raise RuntimeError(
                "Duplicate valid temporal position "
                "in clean geometry: {} position {}".format(
                    key,
                    position,
                )
            )

        result[
            key
        ].add(
            position
        )

    return result


def validate_frame_records(
    frame_rows,
    clean_valid_positions,
):
    if (
        len(
            frame_rows
        )
        != 576
    ):
        raise RuntimeError(
            "Expected 576 frame records, got {}.".format(
                len(
                    frame_rows
                )
            )
        )

    detectors = {
        row[
            "detector"
        ]
        for row in frame_rows
    }

    if (
        detectors
        != EXPECTED_DETECTORS
    ):
        raise RuntimeError(
            "Detector set mismatch: {}".format(
                sorted(
                    detectors
                )
            )
        )

    conditions = {
        row[
            "condition"
        ]
        for row in frame_rows
    }

    if (
        conditions
        != EXPECTED_CONDITIONS
    ):
        raise RuntimeError(
            "Condition set mismatch: {}".format(
                sorted(
                    conditions
                )
            )
        )

    video_keys = {
        base_video_key(
            row
        )
        for row in frame_rows
    }

    if (
        len(
            video_keys
        )
        != EXPECTED_VIDEO_COUNT
    ):
        raise RuntimeError(
            "Expected {} base videos, got {}.".format(
                EXPECTED_VIDEO_COUNT,
                len(
                    video_keys
                ),
            )
        )

    seen_frame_keys = set()

    for row in frame_rows:
        if (
            row[
                "inference_status"
            ]
            != "ok"
        ):
            raise RuntimeError(
                "Smoke aggregation received a "
                "non-valid inference record."
            )

        if (
            row.get(
                "failure_stage",
                "",
            )
        ):
            raise RuntimeError(
                "Valid frame record contains "
                "failure_stage."
            )

        if (
            row.get(
                "failure_reason",
                "",
            )
        ):
            raise RuntimeError(
                "Valid frame record contains "
                "failure_reason."
            )

        score = float(
            row[
                "score"
            ]
        )

        if not math.isfinite(
            score
        ):
            raise RuntimeError(
                "Frame score is non-finite."
            )

        if not (
            0.0
            <= score
            <= 1.0
        ):
            raise RuntimeError(
                "Frame score is outside [0, 1]."
            )

        position = int(
            row[
                "temporal_position"
            ]
        )

        video_key = (
            base_video_key(
                row
            )
        )

        if (
            video_key
            not in clean_valid_positions
        ):
            raise RuntimeError(
                "Frame record video is missing "
                "from clean geometry: {}".format(
                    video_key
                )
            )

        if (
            position
            not in clean_valid_positions[
                video_key
            ]
        ):
            raise RuntimeError(
                "Frame record uses a position that "
                "is not clean-valid: {} position {}".format(
                    video_key,
                    position,
                )
            )

        unique_key = (
            row[
                "detector"
            ],
            row[
                "condition"
            ],
            row[
                "dataset"
            ],
            row[
                "role"
            ],
            row[
                "relative_source_path"
            ],
            position,
        )

        if (
            unique_key
            in seen_frame_keys
        ):
            raise RuntimeError(
                "Duplicate detector-condition-frame "
                "record: {}".format(
                    unique_key
                )
            )

        seen_frame_keys.add(
            unique_key
        )

    return video_keys


def aggregate_records(
    frame_rows,
    clean_valid_positions,
):
    grouped = defaultdict(
        list
    )

    for row in frame_rows:
        grouped[
            group_key(
                row
            )
        ].append(
            row
        )

    if (
        len(
            grouped
        )
        != EXPECTED_VIDEO_RECORD_COUNT
    ):
        raise RuntimeError(
            "Expected {} detector/video/condition "
            "groups, got {}.".format(
                EXPECTED_VIDEO_RECORD_COUNT,
                len(
                    grouped
                ),
            )
        )

    video_records = []

    for key in sorted(
        grouped
    ):
        rows = grouped[
            key
        ]

        (
            detector,
            dataset,
            role,
            relative_source_path,
            condition,
        ) = key

        first = rows[
            0
        ]

        video_key = (
            dataset,
            role,
            relative_source_path,
        )

        expected_positions = (
            clean_valid_positions[
                video_key
            ]
        )

        actual_positions = {
            int(
                row[
                    "temporal_position"
                ]
            )
            for row in rows
        }

        if (
            actual_positions
            != expected_positions
        ):
            missing = sorted(
                expected_positions
                - actual_positions
            )

            extra = sorted(
                actual_positions
                - expected_positions
            )

            raise RuntimeError(
                "Clean-valid completeness failure for "
                "{}.\nMissing: {}\nExtra: {}".format(
                    key,
                    missing,
                    extra,
                )
            )

        expected_count = len(
            expected_positions
        )

        if (
            len(
                rows
            )
            != expected_count
        ):
            raise RuntimeError(
                "Frame-count mismatch for {}: "
                "expected {}, got {}.".format(
                    key,
                    expected_count,
                    len(
                        rows
                    ),
                )
            )

        identity_fields = (
            "subgroup",
            "study_label",
            "source_label",
            "base_video_id",
            "relative_source_path",
            "checkpoint_sha256",
        )

        for field in identity_fields:
            values = {
                str(
                    row[
                        field
                    ]
                )
                for row in rows
            }

            if (
                len(
                    values
                )
                != 1
            ):
                raise RuntimeError(
                    "Inconsistent {} within group {}.".format(
                        field,
                        key,
                    )
                )

        rows = sorted(
            rows,
            key=lambda row: int(
                row[
                    "temporal_position"
                ]
            ),
        )

        scores = np.asarray(
            [
                float(
                    row[
                        "score"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                scores
            )
        ):
            raise RuntimeError(
                "Non-finite scores reached aggregation."
            )

        #
        # Frozen study aggregation rule:
        # unweighted arithmetic mean.
        #
        video_score = float(
            np.mean(
                scores
            )
        )

        if not math.isfinite(
            video_score
        ):
            raise RuntimeError(
                "Video score is non-finite."
            )

        video_records.append(
            {
                "detector": detector,
                "checkpoint_sha256": (
                    first[
                        "checkpoint_sha256"
                    ]
                ),
                "dataset": dataset,
                "role": role,
                "subgroup": (
                    first[
                        "subgroup"
                    ]
                ),
                "study_label": int(
                    first[
                        "study_label"
                    ]
                ),
                "source_label": (
                    first[
                        "source_label"
                    ]
                ),
                "base_video_id": (
                    first[
                        "base_video_id"
                    ]
                ),
                "relative_source_path": (
                    relative_source_path
                ),
                "condition": condition,
                "clean_admissible_frame_count": (
                    expected_count
                ),
                "evaluated_frame_count": (
                    len(
                        rows
                    )
                ),
                "aggregation_method": (
                    "unweighted_arithmetic_mean"
                ),
                "video_score": (
                    video_score
                ),
                "inference_status": (
                    "ok"
                ),
                "failure_stage": "",
                "failure_reason": "",
            }
        )

    return video_records


def validate_cross_condition_support(
    video_records,
):
    grouped = defaultdict(
        dict
    )

    for row in video_records:
        key = (
            row[
                "detector"
            ],
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

        condition = (
            row[
                "condition"
            ]
        )

        if (
            condition
            in grouped[
                key
            ]
        ):
            raise RuntimeError(
                "Duplicate video condition: "
                "{} {}".format(
                    key,
                    condition,
                )
            )

        grouped[
            key
        ][
            condition
        ] = row

    expected_pairs = (
        len(
            EXPECTED_DETECTORS
        )
        * EXPECTED_VIDEO_COUNT
    )

    if (
        len(
            grouped
        )
        != expected_pairs
    ):
        raise RuntimeError(
            "Expected {} detector-video pairs, "
            "got {}.".format(
                expected_pairs,
                len(
                    grouped
                ),
            )
        )

    for key, conditions in grouped.items():
        if (
            set(
                conditions
            )
            != EXPECTED_CONDITIONS
        ):
            raise RuntimeError(
                "Missing paired condition for {}: {}".format(
                    key,
                    sorted(
                        conditions
                    ),
                )
            )

        clean_count = int(
            conditions[
                "CLN"
            ][
                "evaluated_frame_count"
            ]
        )

        rsz_count = int(
            conditions[
                "RSZ"
            ][
                "evaluated_frame_count"
            ]
        )

        if (
            clean_count
            != rsz_count
        ):
            raise RuntimeError(
                "Cross-condition frame-count mismatch "
                "for {}: CLN={}, RSZ={}.".format(
                    key,
                    clean_count,
                    rsz_count,
                )
            )


def main():
    frame_rows = read_jsonl(
        FRAME_RECORDS_PATH
    )

    clean_valid_positions = (
        load_clean_valid_positions()
    )

    video_keys = (
        validate_frame_records(
            frame_rows,
            clean_valid_positions,
        )
    )

    video_records = (
        aggregate_records(
            frame_rows,
            clean_valid_positions,
        )
    )

    validate_cross_condition_support(
        video_records
    )

    write_jsonl(
        VIDEO_RECORDS_PATH,
        video_records,
    )

    detector_counts = Counter(
        row[
            "detector"
        ]
        for row in video_records
    )

    condition_counts = Counter(
        row[
            "condition"
        ]
        for row in video_records
    )

    frame_count_distribution = Counter(
        int(
            row[
                "evaluated_frame_count"
            ]
        )
        for row in video_records
    )

    summary = {
        "schema_version": 1,
        "base_video_count": len(
            video_keys
        ),
        "detectors": sorted(
            EXPECTED_DETECTORS
        ),
        "conditions": sorted(
            EXPECTED_CONDITIONS
        ),
        "input_frame_record_count": len(
            frame_rows
        ),
        "video_record_count": len(
            video_records
        ),
        "detector_video_record_counts": dict(
            sorted(
                detector_counts.items()
            )
        ),
        "condition_video_record_counts": dict(
            sorted(
                condition_counts.items()
            )
        ),
        "evaluated_frame_count_distribution": {
            str(
                key
            ): value
            for (
                key,
                value,
            ) in sorted(
                frame_count_distribution.items()
            )
        },
        "aggregation_method": (
            "unweighted_arithmetic_mean"
        ),
        "frame_weighting_applied": False,
        "majority_voting_applied": False,
        "maximum_score_selection_applied": False,
        "threshold_applied": False,
        "score_calibration_applied": False,
        "frame_records_sha256": (
            sha256_file(
                FRAME_RECORDS_PATH
            )
        ),
        "clean_geometry_sha256": (
            sha256_file(
                CLEAN_GEOMETRY_PATH
            )
        ),
        "video_records_sha256": (
            sha256_file(
                VIDEO_RECORDS_PATH
            )
        ),
        "status": "passed",
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

    print(
        "VIDEO-LEVEL AGGREGATION SMOKE"
    )
    print(
        "  input frame records:       {}".format(
            len(
                frame_rows
            )
        )
    )
    print(
        "  base videos:               {}".format(
            len(
                video_keys
            )
        )
    )
    print(
        "  detector-video-condition:  {}".format(
            len(
                video_records
            )
        )
    )
    print(
        "  aggregation:               "
        "unweighted_arithmetic_mean"
    )
    print(
        "  weighting applied:         False"
    )
    print(
        "  majority voting:           False"
    )
    print(
        "  max selection:             False"
    )
    print(
        "  threshold applied:         False"
    )
    print(
        "  frame-count distribution:  {}".format(
            dict(
                sorted(
                    frame_count_distribution.items()
                )
            )
        )
    )
    print(
        "  video records:             {}".format(
            VIDEO_RECORDS_PATH
        )
    )
    print(
        "  summary:                   {}".format(
            SUMMARY_PATH
        )
    )
    print()
    print(
        "VIDEO-LEVEL AGGREGATION SMOKE PASSED"
    )


if __name__ == "__main__":
    main()
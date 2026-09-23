from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from study.evaluation.inference_validity import (
    STATUS_OK,
    finalize_video_inference,
    paired_valid_records,
)


STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

EVALUATION_ROOT = (
    STUDY_ROOT
    / "evaluation"
)

FRAME_RECORDS_PATH = (
    EVALUATION_ROOT
    / "frame_scoring_smoke.jsonl"
)

CLEAN_GEOMETRY_PATH = (
    EVALUATION_ROOT
    / "clean_geometry.csv"
)

REFERENCE_VIDEO_RECORDS_PATH = (
    EVALUATION_ROOT
    / "video_scoring_smoke.jsonl"
)

CANONICAL_RECORDS_PATH = (
    EVALUATION_ROOT
    / "canonical_inference_smoke.jsonl"
)

SUMMARY_PATH = (
    EVALUATION_ROOT
    / "canonical_inference_smoke_summary.json"
)

EXPECTED_FRAME_RECORDS = 576
EXPECTED_CANONICAL_RECORDS = 18
EXPECTED_PAIRED_RECORDS = 9


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

    records = []

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
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSON at {} line {}.".format(
                        path,
                        line_number,
                    )
                ) from exc

            records.append(
                record
            )

    return records


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


def inference_group_key(row):
    return (
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
        row[
            "condition"
        ],
    )


def reference_key(row):
    return (
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
        row[
            "condition"
        ],
    )


def load_clean_valid_positions():
    rows = read_csv(
        CLEAN_GEOMETRY_PATH
    )

    positions_by_video = defaultdict(
        set
    )

    for row in rows:
        if (
            row[
                "geometry_status"
            ]
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

        if (
            position
            in positions_by_video[
                key
            ]
        ):
            raise RuntimeError(
                "Duplicate clean-valid temporal "
                "position for {}: {}".format(
                    key,
                    position,
                )
            )

        positions_by_video[
            key
        ].add(
            position
        )

    return positions_by_video


def validate_identity_consistency(
    rows,
    key,
):
    fields = (
        "dataset",
        "role",
        "subgroup",
        "study_label",
        "source_label",
        "base_video_id",
        "relative_source_path",
        "detector",
        "condition",
        "checkpoint_sha256",
    )

    for field in fields:
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
                "Inconsistent {} in inference "
                "group {}.".format(
                    field,
                    key,
                )
            )


def build_identity(
    row,
):
    return {
        "dataset": (
            row[
                "dataset"
            ]
        ),
        "role": (
            row[
                "role"
            ]
        ),
        "subgroup": (
            row[
                "subgroup"
            ]
        ),
        "study_label": int(
            row[
                "study_label"
            ]
        ),
        "source_label": (
            row[
                "source_label"
            ]
        ),
        "base_video_id": (
            row[
                "base_video_id"
            ]
        ),
        "relative_source_path": (
            row[
                "relative_source_path"
            ]
        ),
        "checkpoint_sha256": (
            row[
                "checkpoint_sha256"
            ]
        ),
    }


def load_reference_index():
    rows = read_jsonl(
        REFERENCE_VIDEO_RECORDS_PATH
    )

    index = {}

    for row in rows:
        key = reference_key(
            row
        )

        if key in index:
            raise RuntimeError(
                "Duplicate reference video record: {}".format(
                    key
                )
            )

        index[
            key
        ] = row

    if (
        len(
            index
        )
        != EXPECTED_CANONICAL_RECORDS
    ):
        raise RuntimeError(
            "Expected {} independently aggregated "
            "reference video records, got {}.".format(
                EXPECTED_CANONICAL_RECORDS,
                len(
                    index
                ),
            )
        )

    return index


def main():
    frame_records = read_jsonl(
        FRAME_RECORDS_PATH
    )

    if (
        len(
            frame_records
        )
        != EXPECTED_FRAME_RECORDS
    ):
        raise RuntimeError(
            "Expected {} real frame records, "
            "got {}.".format(
                EXPECTED_FRAME_RECORDS,
                len(
                    frame_records
                ),
            )
        )

    clean_valid_positions = (
        load_clean_valid_positions()
    )

    reference_index = (
        load_reference_index()
    )

    grouped = defaultdict(
        list
    )

    for row in frame_records:
        grouped[
            inference_group_key(
                row
            )
        ].append(
            row
        )

    if (
        len(
            grouped
        )
        != EXPECTED_CANONICAL_RECORDS
    ):
        raise RuntimeError(
            "Expected {} real detector-video-condition "
            "groups, got {}.".format(
                EXPECTED_CANONICAL_RECORDS,
                len(
                    grouped
                ),
            )
        )

    canonical_records = []

    score_match_count = 0

    for key in sorted(
        grouped
    ):
        rows = grouped[
            key
        ]

        validate_identity_consistency(
            rows,
            key,
        )

        first = rows[
            0
        ]

        base_key = (
            base_video_key(
                first
            )
        )

        if (
            base_key
            not in clean_valid_positions
        ):
            raise RuntimeError(
                "Inference group has no clean "
                "geometry entry: {}".format(
                    base_key
                )
            )

        frozen_positions = sorted(
            clean_valid_positions[
                base_key
            ]
        )

        canonical = (
            finalize_video_inference(
                identity=(
                    build_identity(
                        first
                    )
                ),
                condition=(
                    first[
                        "condition"
                    ]
                ),
                detector=(
                    first[
                        "detector"
                    ]
                ),
                clean_valid_positions=(
                    frozen_positions
                ),
                frame_records=(
                    rows
                ),
            )
        )

        if (
            canonical[
                "inference_status"
            ]
            != STATUS_OK
        ):
            raise RuntimeError(
                "A previously validated real smoke "
                "record became invalid under the "
                "canonical state machine: {}\n{}"
                .format(
                    key,
                    canonical,
                )
            )

        if (
            canonical[
                "clean_valid_frame_count"
            ]
            != len(
                frozen_positions
            )
        ):
            raise RuntimeError(
                "clean_valid_frame_count mismatch "
                "for {}.".format(
                    key
                )
            )

        if (
            canonical[
                "successful_frame_count"
            ]
            != len(
                frozen_positions
            )
        ):
            raise RuntimeError(
                "successful_frame_count mismatch "
                "for {}.".format(
                    key
                )
            )

        if (
            canonical[
                "retained_temporal_positions"
            ]
            != frozen_positions
        ):
            raise RuntimeError(
                "Frozen temporal support mismatch "
                "for {}.".format(
                    key
                )
            )

        if (
            canonical[
                "aggregation_method"
            ]
            != "unweighted_arithmetic_mean"
        ):
            raise RuntimeError(
                "Unexpected aggregation method "
                "for {}.".format(
                    key
                )
            )

        reference = (
            reference_index.get(
                key
            )
        )

        if reference is None:
            raise RuntimeError(
                "No independent aggregation "
                "reference exists for {}.".format(
                    key
                )
            )

        canonical_score = float(
            canonical[
                "video_score"
            ]
        )

        reference_score = float(
            reference[
                "video_score"
            ]
        )

        if not math.isclose(
            canonical_score,
            reference_score,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                "Canonical/reference score mismatch "
                "for {}: {} != {}.".format(
                    key,
                    canonical_score,
                    reference_score,
                )
            )

        score_match_count += 1

        canonical_records.append(
            canonical
        )

    if (
        score_match_count
        != EXPECTED_CANONICAL_RECORDS
    ):
        raise RuntimeError(
            "Not every canonical score matched "
            "the independent reference."
        )

    clean_records = [
        row
        for row in (
            canonical_records
        )
        if (
            row[
                "condition"
            ]
            == "CLN"
        )
    ]

    rsz_records = [
        row
        for row in (
            canonical_records
        )
        if (
            row[
                "condition"
            ]
            == "RSZ"
        )
    ]

    paired = (
        paired_valid_records(
            clean_records=(
                clean_records
            ),
            processed_records=(
                rsz_records
            ),
        )
    )

    if (
        len(
            paired
        )
        != EXPECTED_PAIRED_RECORDS
    ):
        raise RuntimeError(
            "Expected {} valid CLN-RSZ pairs, "
            "got {}.".format(
                EXPECTED_PAIRED_RECORDS,
                len(
                    paired
                ),
            )
        )

    for (
        clean_record,
        rsz_record,
    ) in paired:
        if (
            clean_record[
                "retained_temporal_positions"
            ]
            != rsz_record[
                "retained_temporal_positions"
            ]
        ):
            raise RuntimeError(
                "Paired CLN/RSZ records do not "
                "share identical temporal support: "
                "{} | {}".format(
                    clean_record[
                        "base_video_id"
                    ],
                    clean_record[
                        "detector"
                    ],
                )
            )

        if (
            clean_record[
                "clean_valid_frame_count"
            ]
            != rsz_record[
                "clean_valid_frame_count"
            ]
        ):
            raise RuntimeError(
                "Paired CLN/RSZ clean-valid "
                "counts differ."
            )

    write_jsonl(
        CANONICAL_RECORDS_PATH,
        canonical_records,
    )

    status_counts = Counter(
        row[
            "inference_status"
        ]
        for row in (
            canonical_records
        )
    )

    frame_count_distribution = Counter(
        int(
            row[
                "successful_frame_count"
            ]
        )
        for row in (
            canonical_records
        )
    )

    detector_counts = Counter(
        row[
            "detector"
        ]
        for row in (
            canonical_records
        )
    )

    condition_counts = Counter(
        row[
            "condition"
        ]
        for row in (
            canonical_records
        )
    )

    summary = {
        "schema_version": 1,
        "input_frame_record_count": len(
            frame_records
        ),
        "canonical_record_count": len(
            canonical_records
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "detector_record_counts": dict(
            sorted(
                detector_counts.items()
            )
        ),
        "condition_record_counts": dict(
            sorted(
                condition_counts.items()
            )
        ),
        "successful_frame_count_distribution": {
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
        "canonical_vs_independent_score_matches": (
            score_match_count
        ),
        "paired_cln_rsz_record_count": len(
            paired
        ),
        "all_paired_temporal_support_identical": (
            True
        ),
        "partial_aggregation_permitted": (
            False
        ),
        "substitution_permitted": (
            False
        ),
        "threshold_applied": (
            False
        ),
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
        "independent_video_records_sha256": (
            sha256_file(
                REFERENCE_VIDEO_RECORDS_PATH
            )
        ),
        "canonical_records_sha256": (
            sha256_file(
                CANONICAL_RECORDS_PATH
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
        "CANONICAL INFERENCE INTEGRATION SMOKE"
    )
    print(
        "  real frame records:          {}".format(
            len(
                frame_records
            )
        )
    )
    print(
        "  canonical video records:     {}".format(
            len(
                canonical_records
            )
        )
    )
    print(
        "  status counts:               {}".format(
            dict(
                sorted(
                    status_counts.items()
                )
            )
        )
    )
    print(
        "  score/reference matches:     {}/{}".format(
            score_match_count,
            EXPECTED_CANONICAL_RECORDS,
        )
    )
    print(
        "  valid CLN-RSZ pairs:         {}".format(
            len(
                paired
            )
        )
    )
    print(
        "  paired temporal support:     IDENTICAL"
    )
    print(
        "  partial aggregation:         FORBIDDEN"
    )
    print(
        "  substitution:                FORBIDDEN"
    )
    print(
        "  threshold applied:           False"
    )
    print(
        "  frame-count distribution:    {}".format(
            dict(
                sorted(
                    frame_count_distribution.items()
                )
            )
        )
    )
    print(
        "  canonical records:           {}".format(
            CANONICAL_RECORDS_PATH
        )
    )
    print(
        "  summary:                     {}".format(
            SUMMARY_PATH
        )
    )
    print()
    print(
        "CANONICAL INFERENCE INTEGRATION SMOKE PASSED"
    )


if __name__ == "__main__":
    main()
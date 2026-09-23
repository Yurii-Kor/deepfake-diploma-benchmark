import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from study.materialization.face_alignment import (
    OUTPUT_SIZE,
    warp_aligned_face,
)
from study.processing.processing_common import (
    RawVideoReader,
    probe_video,
)


DEFAULT_STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

CONDITION = "RSZ"

EXPECTED_SMOKE_VIDEOS = 3

ACCEPTED_QC_STATUSES = {
    "generated_valid",
    "existing_valid",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify frozen clean-reference geometry reuse "
            "on validated RSZ processed videos without "
            "condition-specific face detection."
        )
    )

    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
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
                    "Invalid JSONL record at {} line {}.".format(
                        path,
                        line_number,
                    )
                ) from exc

            records.append(
                record
            )

    return records


def parse_json_cell(
    value,
    *,
    description,
):
    try:
        return json.loads(
            value
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid JSON in {}: {!r}".format(
                description,
                value,
            )
        ) from exc


def video_key(row):
    return (
        row["dataset"],
        row["role"],
        row["relative_source_path"],
    )


def decode_selected_frames(
    path,
    target_indices,
):
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Video does not exist: {}".format(
                path
            )
        )

    target_indices = set(
        target_indices
    )

    probe = probe_video(
        path
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

    selected_frames = {}

    decoded_count = 0

    with RawVideoReader(
        path=path,
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
                decoded_count
                in target_indices
            ):
                selected_frames[
                    decoded_count
                ] = frame_bgr

            decoded_count += 1

    missing = sorted(
        target_indices
        - set(
            selected_frames
        )
    )

    if missing:
        raise RuntimeError(
            "Requested decoded-frame indices were "
            "not found in {}: {}".format(
                path,
                missing,
            )
        )

    return {
        "path": path,
        "probe": probe,
        "decoded_count": decoded_count,
        "frames": selected_frames,
    }


def load_temporal_index(path):
    rows = read_csv(
        path
    )

    index = {}

    for row in rows:
        key = video_key(
            row
        )

        if key in index:
            raise RuntimeError(
                "Duplicate temporal-plan video key: {}".format(
                    key
                )
            )

        if (
            row[
                "temporal_plan_status"
            ]
            != "ok"
        ):
            raise RuntimeError(
                "Temporal plan is not valid for {}".format(
                    key
                )
            )

        index[
            key
        ] = row

    return index


def load_geometry_index(path):
    rows = read_csv(
        path
    )

    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            video_key(
                row
            )
        ].append(
            row
        )

    for key, video_rows in grouped.items():
        if (
            len(
                video_rows
            )
            != 32
        ):
            raise RuntimeError(
                "Expected 32 geometry rows for {}, got {}.".format(
                    key,
                    len(
                        video_rows
                    ),
                )
            )

        video_rows.sort(
            key=lambda row: int(
                row[
                    "temporal_position"
                ]
            )
        )

        positions = [
            int(
                row[
                    "temporal_position"
                ]
            )
            for row in video_rows
        ]

        if (
            positions
            != list(
                range(
                    32
                )
            )
        ):
            raise RuntimeError(
                "Unexpected temporal positions for {}.".format(
                    key
                )
            )

    return grouped


def select_validated_rsz_outputs(
    qc_path,
):
    qc_rows = read_jsonl(
        qc_path
    )

    latest_valid = {}

    for row in qc_rows:
        if (
            row.get(
                "condition"
            )
            != CONDITION
        ):
            continue

        if (
            row.get(
                "status"
            )
            not in ACCEPTED_QC_STATUSES
        ):
            continue

        if (
            row.get(
                "dataset"
            )
            != "FaceForensics++"
            or row.get(
                "role"
            )
            != "validation"
            or row.get(
                "subgroup"
            )
            != "original"
        ):
            continue

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

        latest_valid[
            key
        ] = row

    selected = sorted(
        latest_valid.values(),
        key=lambda row: (
            row[
                "relative_source_path"
            ]
        ),
    )

    if (
        len(
            selected
        )
        != EXPECTED_SMOKE_VIDEOS
    ):
        raise RuntimeError(
            "Expected {} unique validated RSZ smoke "
            "videos, got {}.".format(
                EXPECTED_SMOKE_VIDEOS,
                len(
                    selected
                ),
            )
        )

    return selected


def validate_affine(
    row,
):
    matrix = np.asarray(
        parse_json_cell(
            row[
                "affine_matrix_json"
            ],
            description=(
                "affine_matrix_json"
            ),
        ),
        dtype=np.float64,
    )

    if (
        matrix.shape
        != (
            2,
            3,
        )
    ):
        raise RuntimeError(
            "Saved affine matrix has invalid shape: {}".format(
                matrix.shape
            )
        )

    if not np.isfinite(
        matrix
    ).all():
        raise RuntimeError(
            "Saved affine matrix contains "
            "non-finite values."
        )

    return matrix


def apply_saved_geometry(
    frame_bgr,
    affine_matrix,
):
    if (
        frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
    ):
        raise RuntimeError(
            "Decoded frame must be uint8 HxWx3."
        )

    image_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    aligned_rgb = (
        warp_aligned_face(
            image_rgb=image_rgb,
            affine_matrix=affine_matrix,
            output_size=OUTPUT_SIZE,
        )
    )

    if (
        aligned_rgb.shape
        != (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
            3,
        )
    ):
        raise RuntimeError(
            "Unexpected aligned output shape: {}".format(
                aligned_rgb.shape
            )
        )

    if (
        aligned_rgb.dtype
        != np.uint8
    ):
        raise RuntimeError(
            "Aligned output must have dtype uint8."
        )

    aligned_bgr = cv2.cvtColor(
        aligned_rgb,
        cv2.COLOR_RGB2BGR,
    )

    return aligned_bgr


def process_video(
    qc_row,
    temporal_index,
    geometry_index,
):
    key = (
        qc_row[
            "dataset"
        ],
        qc_row[
            "role"
        ],
        qc_row[
            "relative_source_path"
        ],
    )

    if key not in temporal_index:
        raise RuntimeError(
            "Validated RSZ video is not present "
            "in frozen temporal plan: {}".format(
                key
            )
        )

    if key not in geometry_index:
        raise RuntimeError(
            "Validated RSZ video is not present "
            "in clean geometry: {}".format(
                key
            )
        )

    temporal_row = (
        temporal_index[
            key
        ]
    )

    geometry_rows = (
        geometry_index[
            key
        ]
    )

    if (
        qc_row[
            "subgroup"
        ]
        != temporal_row[
            "subgroup"
        ]
    ):
        raise RuntimeError(
            "QC/temporal subgroup mismatch for {}".format(
                key
            )
        )

    if (
        qc_row[
            "base_video_id"
        ]
        != temporal_row[
            "base_video_id"
        ]
    ):
        raise RuntimeError(
            "QC/temporal base-video ID mismatch for {}".format(
                key
            )
        )

    valid_rows = [
        row
        for row in geometry_rows
        if (
            row[
                "geometry_status"
            ]
            == "valid"
        )
    ]

    invalid_rows = [
        row
        for row in geometry_rows
        if (
            row[
                "geometry_status"
            ]
            == "invalid"
        )
    ]

    if not valid_rows:
        raise RuntimeError(
            "Smoke video has zero clean-valid "
            "positions: {}".format(
                key
            )
        )

    target_indices = [
        int(
            row[
                "source_frame_index"
            ]
        )
        for row in valid_rows
    ]

    if (
        len(
            target_indices
        )
        != len(
            set(
                target_indices
            )
        )
    ):
        raise RuntimeError(
            "Duplicate clean-valid frame indices "
            "for {}".format(
                key
            )
        )

    expected_decoded_count = int(
        temporal_row[
            "decoded_frame_count"
        ]
    )

    source_path = (
        Path(
            temporal_row[
                "absolute_source_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    processed_path = (
        Path(
            qc_row[
                "output_path"
            ]
        )
        .expanduser()
        .resolve()
    )

    if (
        Path(
            qc_row[
                "source_path"
            ]
        )
        .expanduser()
        .resolve()
        != source_path
    ):
        raise RuntimeError(
            "QC source path differs from frozen "
            "temporal-plan source for {}".format(
                key
            )
        )

    source_decoded = (
        decode_selected_frames(
            path=source_path,
            target_indices=target_indices,
        )
    )

    processed_decoded = (
        decode_selected_frames(
            path=processed_path,
            target_indices=target_indices,
        )
    )

    if (
        source_decoded[
            "decoded_count"
        ]
        != expected_decoded_count
    ):
        raise RuntimeError(
            "CLN decoded-frame count mismatch for {}: "
            "expected {}, got {}.".format(
                key,
                expected_decoded_count,
                source_decoded[
                    "decoded_count"
                ],
            )
        )

    if (
        processed_decoded[
            "decoded_count"
        ]
        != expected_decoded_count
    ):
        raise RuntimeError(
            "RSZ decoded-frame count mismatch for {}: "
            "expected {}, got {}.".format(
                key,
                expected_decoded_count,
                processed_decoded[
                    "decoded_count"
                ],
            )
        )

    source_probe = (
        source_decoded[
            "probe"
        ]
    )

    processed_probe = (
        processed_decoded[
            "probe"
        ]
    )

    for dimension in (
        "width",
        "height",
    ):
        if (
            int(
                source_probe[
                    dimension
                ]
            )
            != int(
                processed_probe[
                    dimension
                ]
            )
        ):
            raise RuntimeError(
                "CLN/RSZ {} mismatch for {}.".format(
                    dimension,
                    key,
                )
            )

    frame_records = []

    for row in valid_rows:
        temporal_position = int(
            row[
                "temporal_position"
            ]
        )

        frame_index = int(
            row[
                "source_frame_index"
            ]
        )

        clean_frame_bgr = (
            source_decoded[
                "frames"
            ][
                frame_index
            ]
        )

        processed_frame_bgr = (
            processed_decoded[
                "frames"
            ][
                frame_index
            ]
        )

        clean_frame_hash = (
            sha256_array(
                clean_frame_bgr
            )
        )

        if (
            clean_frame_hash
            != row[
                "source_frame_sha256"
            ]
        ):
            raise RuntimeError(
                "Decoded CLN frame hash differs "
                "from frozen clean_geometry.csv for "
                "{} position {} frame {}.".format(
                    key,
                    temporal_position,
                    frame_index,
                )
            )

        affine_matrix = (
            validate_affine(
                row
            )
        )

        reconstructed_clean_bgr = (
            apply_saved_geometry(
                frame_bgr=(
                    clean_frame_bgr
                ),
                affine_matrix=(
                    affine_matrix
                ),
            )
        )

        reconstructed_clean_hash = (
            sha256_array(
                reconstructed_clean_bgr
            )
        )

        if (
            reconstructed_clean_hash
            != row[
                "aligned_sha256"
            ]
        ):
            raise RuntimeError(
                "Saved affine matrix does not exactly "
                "reproduce the frozen CLN aligned crop "
                "for {} position {} frame {}.".format(
                    key,
                    temporal_position,
                    frame_index,
                )
            )

        processed_frame_hash = (
            sha256_array(
                processed_frame_bgr
            )
        )

        processed_aligned_bgr = (
            apply_saved_geometry(
                frame_bgr=(
                    processed_frame_bgr
                ),
                affine_matrix=(
                    affine_matrix
                ),
            )
        )

        processed_aligned_hash = (
            sha256_array(
                processed_aligned_bgr
            )
        )

        frame_records.append(
            {
                "temporal_position": (
                    temporal_position
                ),
                "source_frame_index": (
                    frame_index
                ),
                "clean_source_frame_sha256": (
                    clean_frame_hash
                ),
                "clean_aligned_sha256": (
                    reconstructed_clean_hash
                ),
                "clean_aligned_matches_frozen": (
                    True
                ),
                "processed_source_frame_sha256": (
                    processed_frame_hash
                ),
                "processed_aligned_sha256": (
                    processed_aligned_hash
                ),
                "affine_source": (
                    "clean_geometry.csv"
                ),
                "geometry_reestimated": (
                    False
                ),
            }
        )

    if (
        len(
            frame_records
        )
        != len(
            valid_rows
        )
    ):
        raise RuntimeError(
            "Processed-frame record count mismatch "
            "for {}".format(
                key
            )
        )

    return {
        "dataset": (
            qc_row[
                "dataset"
            ]
        ),
        "role": (
            qc_row[
                "role"
            ]
        ),
        "subgroup": (
            qc_row[
                "subgroup"
            ]
        ),
        "base_video_id": (
            qc_row[
                "base_video_id"
            ]
        ),
        "relative_source_path": (
            qc_row[
                "relative_source_path"
            ]
        ),
        "condition": CONDITION,
        "qc_status": (
            qc_row[
                "status"
            ]
        ),
        "source_path": str(
            source_path
        ),
        "processed_path": str(
            processed_path
        ),
        "expected_decoded_frame_count": (
            expected_decoded_count
        ),
        "clean_runtime_decoded_frame_count": (
            source_decoded[
                "decoded_count"
            ]
        ),
        "processed_runtime_decoded_frame_count": (
            processed_decoded[
                "decoded_count"
            ]
        ),
        "clean_valid_position_count": len(
            valid_rows
        ),
        "clean_invalid_position_count": len(
            invalid_rows
        ),
        "reused_affine_count": len(
            frame_records
        ),
        "processed_aligned_count": len(
            frame_records
        ),
        "all_clean_reconstructions_match_frozen": (
            all(
                record[
                    "clean_aligned_matches_frozen"
                ]
                for record in (
                    frame_records
                )
            )
        ),
        "frames": frame_records,
        "status": "ok",
    }


def main():
    args = parse_args()

    study_root = (
        args.study_root
        .expanduser()
        .resolve()
    )

    temporal_plan_path = (
        study_root
        / "evaluation"
        / "temporal_plan.csv"
    )

    clean_geometry_path = (
        study_root
        / "evaluation"
        / "clean_geometry.csv"
    )

    qc_path = (
        study_root
        / "qc"
        / "processing_RSZ_geometry_smoke.jsonl"
    )

    output_path = (
        study_root
        / "evaluation"
        / "processed_geometry_RSZ_smoke.json"
    )

    #
    # This smoke must never enter the dlib face-detection path.
    # face_alignment imports dlib only inside its loader function,
    # which is deliberately not imported or called here.
    #
    if "dlib" in sys.modules:
        raise RuntimeError(
            "dlib was unexpectedly loaded before "
            "processed-geometry evaluation."
        )

    temporal_index = (
        load_temporal_index(
            temporal_plan_path
        )
    )

    geometry_index = (
        load_geometry_index(
            clean_geometry_path
        )
    )

    selected_qc = (
        select_validated_rsz_outputs(
            qc_path
        )
    )

    print(
        "PROCESSED GEOMETRY REUSE SMOKE"
    )
    print(
        "  condition:                 {}".format(
            CONDITION
        )
    )
    print(
        "  selected videos:           {}".format(
            len(
                selected_qc
            )
        )
    )
    print(
        "  clean geometry SHA-256:    {}".format(
            sha256_file(
                clean_geometry_path
            )
        )
    )
    print(
        "  dlib loaded:               {}".format(
            "dlib" in sys.modules
        )
    )
    print()

    video_results = []

    for index, qc_row in enumerate(
        selected_qc,
        start=1,
    ):
        result = process_video(
            qc_row=qc_row,
            temporal_index=(
                temporal_index
            ),
            geometry_index=(
                geometry_index
            ),
        )

        video_results.append(
            result
        )

        print(
            "  [{}/{}] {} | {} | {} | {}".format(
                index,
                len(
                    selected_qc
                ),
                result[
                    "dataset"
                ],
                result[
                    "role"
                ],
                result[
                    "subgroup"
                ],
                result[
                    "base_video_id"
                ],
            )
        )

        print(
            "        decoded CLN/RSZ: {} / {}".format(
                result[
                    "clean_runtime_decoded_frame_count"
                ],
                result[
                    "processed_runtime_decoded_frame_count"
                ],
            )
        )

        print(
            "        clean-valid:     {} / 32".format(
                result[
                    "clean_valid_position_count"
                ]
            )
        )

        print(
            "        reused affine:   {} / {}".format(
                result[
                    "reused_affine_count"
                ],
                result[
                    "clean_valid_position_count"
                ],
            )
        )

        print(
            "        CLN hash replay: {}".format(
                (
                    "PASSED"
                    if result[
                        "all_clean_reconstructions_match_frozen"
                    ]
                    else "FAILED"
                )
            )
        )

    if "dlib" in sys.modules:
        raise RuntimeError(
            "dlib was loaded during processed-geometry "
            "reuse. Condition-specific face detection "
            "must not occur."
        )

    total_valid_positions = sum(
        row[
            "clean_valid_position_count"
        ]
        for row in video_results
    )

    total_reused_affines = sum(
        row[
            "reused_affine_count"
        ]
        for row in video_results
    )

    artifact = {
        "schema_version": 1,
        "condition": CONDITION,
        "selected_video_count": len(
            video_results
        ),
        "temporal_plan_sha256": (
            sha256_file(
                temporal_plan_path
            )
        ),
        "clean_geometry_sha256": (
            sha256_file(
                clean_geometry_path
            )
        ),
        "processing_qc_sha256": (
            sha256_file(
                qc_path
            )
        ),
        "geometry_source": (
            "clean_reference"
        ),
        "geometry_reestimated_on_processed": (
            False
        ),
        "condition_specific_face_detection": (
            False
        ),
        "dlib_loaded": (
            "dlib" in sys.modules
        ),
        "total_clean_valid_positions": (
            total_valid_positions
        ),
        "total_reused_affines": (
            total_reused_affines
        ),
        "all_clean_reconstructions_match_frozen": (
            all(
                row[
                    "all_clean_reconstructions_match_frozen"
                ]
                for row in video_results
            )
        ),
        "all_status_ok": (
            all(
                row[
                    "status"
                ]
                == "ok"
                for row in video_results
            )
        ),
        "videos": video_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    print()
    print(
        "PROCESSED GEOMETRY REUSE RESULT"
    )
    print(
        "  videos:                    {}".format(
            len(
                video_results
            )
        )
    )
    print(
        "  clean-valid positions:     {}".format(
            total_valid_positions
        )
    )
    print(
        "  reused CLN affines:        {}".format(
            total_reused_affines
        )
    )
    print(
        "  geometry re-estimated:     False"
    )
    print(
        "  condition face detection:  False"
    )
    print(
        "  dlib loaded:               {}".format(
            "dlib" in sys.modules
        )
    )
    print(
        "  artifact:                  {}".format(
            output_path
        )
    )
    print()
    print(
        "PROCESSED GEOMETRY REUSE SMOKE PASSED"
    )


if __name__ == "__main__":
    main()
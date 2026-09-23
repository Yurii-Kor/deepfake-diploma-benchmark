import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from study.evaluation.detector_scoring import (
    DETECTOR_SPECS,
    FrozenDetectorScorer,
    sha256_file,
)
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

DETECTORS = (
    "xception",
    "ucf",
    "spsl",
)

CONDITIONS = (
    "CLN",
    "RSZ",
)

EXPECTED_VIDEOS = 3

ACCEPTED_QC_STATUSES = {
    "generated_valid",
    "existing_valid",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run real frozen-checkpoint frame scoring "
            "over the current CLN/RSZ geometry smoke set."
        )
    )

    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    return parser.parse_args()


def sha256_array(
    array,
):
    array = np.ascontiguousarray(
        array
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def read_csv(
    path,
):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def read_jsonl(
    path,
):
    records = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(
                        line
                    )
                )

    return records


def parse_json_cell(
    value,
):
    return json.loads(
        value
    )


def video_key(
    row,
):
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
            path
        )

    target_indices = set(
        target_indices
    )

    probe = probe_video(
        path
    )

    frames = {}

    decoded_count = 0

    with RawVideoReader(
        path=path,
        width=int(
            probe[
                "width"
            ]
        ),
        height=int(
            probe[
                "height"
            ]
        ),
    ) as reader:
        while True:
            frame = (
                reader.read_frame()
            )

            if frame is None:
                break

            if (
                decoded_count
                in target_indices
            ):
                frames[
                    decoded_count
                ] = frame

            decoded_count += 1

    missing = sorted(
        target_indices
        - set(
            frames
        )
    )

    if missing:
        raise RuntimeError(
            "Missing decoded target frames in {}: {}".format(
                path,
                missing,
            )
        )

    return (
        decoded_count,
        frames,
    )


def apply_saved_geometry(
    frame_bgr,
    affine_matrix,
):
    image_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    aligned_rgb = (
        warp_aligned_face(
            image_rgb=image_rgb,
            affine_matrix=(
                affine_matrix
            ),
            output_size=(
                OUTPUT_SIZE
            ),
        )
    )

    aligned_bgr = cv2.cvtColor(
        aligned_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if (
        aligned_bgr.shape
        != (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
            3,
        )
    ):
        raise RuntimeError(
            "Unexpected aligned shape: {}".format(
                aligned_bgr.shape
            )
        )

    return aligned_bgr


def load_temporal_index(
    path,
):
    result = {}

    for row in read_csv(
        path
    ):
        key = video_key(
            row
        )

        if key in result:
            raise RuntimeError(
                "Duplicate temporal-plan key: {}".format(
                    key
                )
            )

        result[
            key
        ] = row

    return result


def load_geometry_index(
    path,
):
    result = defaultdict(
        list
    )

    for row in read_csv(
        path
    ):
        result[
            video_key(
                row
            )
        ].append(
            row
        )

    for key in result:
        result[
            key
        ].sort(
            key=lambda row: int(
                row[
                    "temporal_position"
                ]
            )
        )

    return result


def select_qc_videos(
    path,
):
    latest = {}

    for row in read_jsonl(
        path
    ):
        if (
            row.get(
                "condition"
            )
            != "RSZ"
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

        latest[
            video_key(
                row
            )
        ] = row

    selected = sorted(
        latest.values(),
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
        != EXPECTED_VIDEOS
    ):
        raise RuntimeError(
            "Expected {} smoke videos, got {}.".format(
                EXPECTED_VIDEOS,
                len(
                    selected
                ),
            )
        )

    return selected


def build_samples(
    temporal_index,
    geometry_index,
    qc_rows,
):
    samples = []

    video_summary = []

    for qc_row in qc_rows:
        key = video_key(
            qc_row
        )

        if key not in temporal_index:
            raise RuntimeError(
                "QC video missing from temporal plan: {}".format(
                    key
                )
            )

        if key not in geometry_index:
            raise RuntimeError(
                "QC video missing from clean geometry: {}".format(
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

        if not valid_rows:
            raise RuntimeError(
                "Smoke video has zero clean-valid positions: {}".format(
                    key
                )
            )

        frame_indices = [
            int(
                row[
                    "source_frame_index"
                ]
            )
            for row in valid_rows
        ]

        expected_count = int(
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

        (
            clean_count,
            clean_frames,
        ) = decode_selected_frames(
            source_path,
            frame_indices,
        )

        (
            rsz_count,
            rsz_frames,
        ) = decode_selected_frames(
            processed_path,
            frame_indices,
        )

        if (
            clean_count
            != expected_count
            or rsz_count
            != expected_count
        ):
            raise RuntimeError(
                "Decoded-frame-count mismatch for {}: "
                "expected={}, CLN={}, RSZ={}.".format(
                    key,
                    expected_count,
                    clean_count,
                    rsz_count,
                )
            )

        for row in valid_rows:
            position = int(
                row[
                    "temporal_position"
                ]
            )

            frame_index = int(
                row[
                    "source_frame_index"
                ]
            )

            affine_matrix = np.asarray(
                parse_json_cell(
                    row[
                        "affine_matrix_json"
                    ]
                ),
                dtype=np.float64,
            )

            if (
                affine_matrix.shape
                != (
                    2,
                    3,
                )
                or not np.isfinite(
                    affine_matrix
                ).all()
            ):
                raise RuntimeError(
                    "Invalid saved affine matrix "
                    "for {} position {}.".format(
                        key,
                        position,
                    )
                )

            clean_frame = (
                clean_frames[
                    frame_index
                ]
            )

            clean_source_hash = (
                sha256_array(
                    clean_frame
                )
            )

            if (
                clean_source_hash
                != row[
                    "source_frame_sha256"
                ]
            ):
                raise RuntimeError(
                    "Frozen CLN frame hash mismatch "
                    "for {} position {}.".format(
                        key,
                        position,
                    )
                )

            clean_aligned = (
                apply_saved_geometry(
                    clean_frame,
                    affine_matrix,
                )
            )

            clean_aligned_hash = (
                sha256_array(
                    clean_aligned
                )
            )

            if (
                clean_aligned_hash
                != row[
                    "aligned_sha256"
                ]
            ):
                raise RuntimeError(
                    "Frozen CLN aligned hash replay "
                    "failed for {} position {}.".format(
                        key,
                        position,
                    )
                )

            rsz_frame = (
                rsz_frames[
                    frame_index
                ]
            )

            rsz_aligned = (
                apply_saved_geometry(
                    rsz_frame,
                    affine_matrix,
                )
            )

            common = {
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
                "temporal_position": (
                    position
                ),
                "source_frame_index": (
                    frame_index
                ),
            }

            samples.append(
                {
                    **common,
                    "condition": "CLN",
                    "decoded_frame_sha256": (
                        clean_source_hash
                    ),
                    "aligned_frame_sha256": (
                        clean_aligned_hash
                    ),
                    "aligned_bgr": (
                        clean_aligned
                    ),
                }
            )

            samples.append(
                {
                    **common,
                    "condition": "RSZ",
                    "decoded_frame_sha256": (
                        sha256_array(
                            rsz_frame
                        )
                    ),
                    "aligned_frame_sha256": (
                        sha256_array(
                            rsz_aligned
                        )
                    ),
                    "aligned_bgr": (
                        rsz_aligned
                    ),
                }
            )

        video_summary.append(
            {
                "base_video_id": (
                    qc_row[
                        "base_video_id"
                    ]
                ),
                "expected_decoded_frame_count": (
                    expected_count
                ),
                "clean_runtime_decoded_frame_count": (
                    clean_count
                ),
                "rsz_runtime_decoded_frame_count": (
                    rsz_count
                ),
                "clean_valid_position_count": len(
                    valid_rows
                ),
            }
        )

    return (
        samples,
        video_summary,
    )


def write_jsonl(
    path,
    records,
):
    with Path(path).open(
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


def main():
    args = parse_args()

    if (
        args.batch_size
        <= 0
    ):
        raise ValueError(
            "--batch-size must be positive."
        )

    study_root = (
        args.study_root
        .expanduser()
        .resolve()
    )

    evaluation_dir = (
        study_root
        / "evaluation"
    )

    temporal_plan_path = (
        evaluation_dir
        / "temporal_plan.csv"
    )

    clean_geometry_path = (
        evaluation_dir
        / "clean_geometry.csv"
    )

    qc_path = (
        study_root
        / "qc"
        / "processing_RSZ_geometry_smoke.jsonl"
    )

    frame_output_path = (
        evaluation_dir
        / "frame_scoring_smoke.jsonl"
    )

    summary_path = (
        evaluation_dir
        / "frame_scoring_smoke_summary.json"
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

    qc_rows = (
        select_qc_videos(
            qc_path
        )
    )

    (
        samples,
        video_summary,
    ) = build_samples(
        temporal_index=(
            temporal_index
        ),
        geometry_index=(
            geometry_index
        ),
        qc_rows=qc_rows,
    )

    condition_counts = Counter(
        sample[
            "condition"
        ]
        for sample in samples
    )

    print(
        "FROZEN CHECKPOINT FRAME-SCORING SMOKE"
    )
    print(
        "  videos:                    {}".format(
            len(
                qc_rows
            )
        )
    )
    print(
        "  CLN inputs:                {}".format(
            condition_counts[
                "CLN"
            ]
        )
    )
    print(
        "  RSZ inputs:                {}".format(
            condition_counts[
                "RSZ"
            ]
        )
    )
    print(
        "  batch size:                {}".format(
            args.batch_size
        )
    )
    print()

    frame_records = []

    aligned_images = [
        sample[
            "aligned_bgr"
        ]
        for sample in samples
    ]

    labels = [
        sample[
            "study_label"
        ]
        for sample in samples
    ]

    detector_summaries = {}

    for detector in DETECTORS:
        print(
            "  Loading {}...".format(
                detector.upper()
            )
        )

        scorer = FrozenDetectorScorer(
            detector,
            device=args.device,
            batch_size=(
                args.batch_size
            ),
        )

        print(
            "    device:             {}".format(
                scorer.device
            )
        )
        print(
            "    checkpoint SHA-256: {}".format(
                scorer.checkpoint_sha256
            )
        )

        scores = scorer.score(
            aligned_images,
            labels,
        )

        if (
            len(
                scores
            )
            != len(
                samples
            )
        ):
            raise RuntimeError(
                "{} score count mismatch.".format(
                    detector
                )
            )

        detector_records = []

        for sample, score in zip(
            samples,
            scores,
        ):
            if not np.isfinite(
                score
            ):
                raise RuntimeError(
                    "{} produced non-finite score.".format(
                        detector
                    )
                )

            record = {
                "detector": detector,
                "checkpoint_path": str(
                    scorer.checkpoint_path
                ),
                "checkpoint_sha256": (
                    scorer.checkpoint_sha256
                ),
                "dataset": (
                    sample[
                        "dataset"
                    ]
                ),
                "role": (
                    sample[
                        "role"
                    ]
                ),
                "subgroup": (
                    sample[
                        "subgroup"
                    ]
                ),
                "study_label": (
                    sample[
                        "study_label"
                    ]
                ),
                "source_label": (
                    sample[
                        "source_label"
                    ]
                ),
                "base_video_id": (
                    sample[
                        "base_video_id"
                    ]
                ),
                "relative_source_path": (
                    sample[
                        "relative_source_path"
                    ]
                ),
                "condition": (
                    sample[
                        "condition"
                    ]
                ),
                "temporal_position": (
                    sample[
                        "temporal_position"
                    ]
                ),
                "source_frame_index": (
                    sample[
                        "source_frame_index"
                    ]
                ),
                "decoded_frame_sha256": (
                    sample[
                        "decoded_frame_sha256"
                    ]
                ),
                "aligned_frame_sha256": (
                    sample[
                        "aligned_frame_sha256"
                    ]
                ),
                "score_semantics": (
                    "model_return_prob"
                ),
                "score": (
                    score
                ),
                "inference_status": (
                    "ok"
                ),
                "failure_stage": "",
                "failure_reason": "",
            }

            detector_records.append(
                record
            )

        frame_records.extend(
            detector_records
        )

        per_condition = Counter(
            row[
                "condition"
            ]
            for row in detector_records
        )

        detector_summaries[
            detector
        ] = {
            "device": str(
                scorer.device
            ),
            "checkpoint_sha256": (
                scorer.checkpoint_sha256
            ),
            "frame_score_count": len(
                detector_records
            ),
            "condition_counts": dict(
                sorted(
                    per_condition.items()
                )
            ),
            "minimum_score": min(
                scores
            ),
            "maximum_score": max(
                scores
            ),
            "all_scores_finite": True,
            "all_scores_in_unit_interval": (
                all(
                    0.0
                    <= score
                    <= 1.0
                    for score in scores
                )
            ),
        }

        print(
            "    scores:             {}".format(
                len(
                    scores
                )
            )
        )
        print(
            "    range:              [{:.8f}, {:.8f}]".format(
                min(
                    scores
                ),
                max(
                    scores
                ),
            )
        )
        print(
            "    PASSED"
        )
        print()

        del scorer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_records = (
        len(
            samples
        )
        * len(
            DETECTORS
        )
    )

    if (
        len(
            frame_records
        )
        != expected_records
    ):
        raise RuntimeError(
            "Final frame-record count mismatch: "
            "{} != {}.".format(
                len(
                    frame_records
                ),
                expected_records,
            )
        )

    write_jsonl(
        frame_output_path,
        frame_records,
    )

    summary = {
        "schema_version": 1,
        "selected_video_count": len(
            qc_rows
        ),
        "conditions": list(
            CONDITIONS
        ),
        "detectors": list(
            DETECTORS
        ),
        "input_frame_count": len(
            samples
        ),
        "input_condition_counts": dict(
            sorted(
                condition_counts.items()
            )
        ),
        "frame_score_record_count": len(
            frame_records
        ),
        "score_semantics": (
            "raw_model_return_prob"
        ),
        "score_calibration_applied": False,
        "score_normalization_applied": False,
        "score_clipping_applied": False,
        "condition_specific_score_transform": False,
        "video_aggregation_performed": False,
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
        "frame_records_sha256": (
            sha256_file(
                frame_output_path
            )
        ),
        "detector_summaries": (
            detector_summaries
        ),
        "videos": (
            video_summary
        ),
        "status": "passed",
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
        "FRAME-SCORING SMOKE RESULT"
    )
    print(
        "  input frames:              {}".format(
            len(
                samples
            )
        )
    )
    print(
        "  detector-frame records:    {}".format(
            len(
                frame_records
            )
        )
    )
    print(
        "  calibration applied:       False"
    )
    print(
        "  normalization of scores:   False"
    )
    print(
        "  score clipping:            False"
    )
    print(
        "  video aggregation:         False"
    )
    print(
        "  frame records:             {}".format(
            frame_output_path
        )
    )
    print(
        "  summary:                   {}".format(
            summary_path
        )
    )
    print()
    print(
        "FROZEN CHECKPOINT FRAME-SCORING SMOKE PASSED"
    )


if __name__ == "__main__":
    main()
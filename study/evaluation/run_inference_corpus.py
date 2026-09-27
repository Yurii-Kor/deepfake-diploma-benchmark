from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from study.evaluation.detector_scoring import (
    DETECTOR_SPECS,
    FrozenDetectorScorer,
    sha256_file,
)
from study.evaluation.frame_scoring_smoke import (
    apply_saved_geometry,
    decode_selected_frames,
    load_geometry_index,
    load_temporal_index,
    parse_json_cell,
    read_jsonl,
    sha256_array,
    video_key,
)
from study.evaluation.inference_validity import (
    FAILURE_CLEAN_GEOMETRY,
    FAILURE_CONDITION_INTEGRITY,
    FAILURE_DETECTOR_INFERENCE,
    FAILURE_DETECTOR_INPUT,
    FAILURE_SOURCE_DECODE,
    STATUS_OK,
    finalize_video_inference,
)


DEFAULT_STUDY_ROOT = (
    Path.home()
    / "deepfake_lab"
    / "study_data"
)

CONDITIONS = (
    "CLN",
    "RSZ",
    "BLR",
    "H40",
    "PLT",
)

VALID_QC = {
    "generated_valid",
    "existing_valid",
}


def args():
    parser = argparse.ArgumentParser(
        description=(
            "Production frozen-checkpoint inference."
        )
    )

    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
    )

    parser.add_argument(
        "--detector",
        required=True,
        choices=sorted(
            DETECTOR_SPECS
        ),
    )

    parser.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--role",
        required=True,
    )

    parser.add_argument(
        "--subgroup",
    )

    parser.add_argument(
        "--qc-jsonl",
        type=Path,
    )

    parser.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
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

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def write_json(
    path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write(
            "\n"
        )

    temporary.replace(
        path
    )


def append_jsonl(
    path,
    row,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
        )

        file.flush()


def out_key(
    row,
):
    return (
        str(
            row[
                "detector"
            ]
        ),
        str(
            row[
                "dataset"
            ]
        ),
        str(
            row[
                "role"
            ]
        ),
        str(
            row[
                "relative_source_path"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
    )


def select_rows(
    index,
    dataset,
    role,
    subgroup,
    limit,
):
    rows = [
        row
        for row in index.values()
        if (
            str(
                row[
                    "dataset"
                ]
            )
            == dataset
            and str(
                row[
                    "role"
                ]
            )
            == role
            and (
                subgroup is None
                or str(
                    row.get(
                        "subgroup",
                        "",
                    )
                )
                == subgroup
            )
        )
    ]

    rows.sort(
        key=lambda row: str(
            row[
                "relative_source_path"
            ]
        )
    )

    if limit is not None:
        rows = rows[
            :limit
        ]

    return rows


def qc_index(
    path,
    condition,
):
    latest = {}

    for row in read_jsonl(
        path
    ):
        if (
            str(
                row.get(
                    "condition",
                    "",
                )
            )
            == condition
        ):
            latest[
                video_key(
                    row
                )
            ] = row

    return latest


def valid_geometry(
    rows,
):
    return [
        row
        for row in rows
        if (
            row[
                "geometry_status"
            ]
            == "valid"
        )
    ]


def identity(
    row,
    checkpoint_sha,
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
            row.get(
                "subgroup",
                "",
            )
        ),
        "study_label": int(
            row[
                "study_label"
            ]
        ),
        "source_label": (
            row.get(
                "source_label",
                "",
            )
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
            checkpoint_sha
        ),
    }


def validate_geometry(
    rows,
    key,
):
    if not rows:
        raise RuntimeError(
            "Missing clean geometry for {}."
            .format(
                key
            )
        )

    for field in (
        "dataset",
        "role",
        "subgroup",
        "study_label",
        "source_label",
        "base_video_id",
        "relative_source_path",
    ):
        values = {
            str(
                row.get(
                    field,
                    "",
                )
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
                "Inconsistent geometry field "
                "{} for {}."
                .format(
                    field,
                    key,
                )
            )

    positions = [
        int(
            row[
                "temporal_position"
            ]
        )
        for row in rows
    ]

    if (
        len(
            positions
        )
        != len(
            set(
                positions
            )
        )
    ):
        raise RuntimeError(
            "Duplicate temporal positions for {}."
            .format(
                key
            )
        )


def prepare(
    condition,
    temporal,
    geometry,
    qc,
):
    rows = valid_geometry(
        geometry
    )

    if not rows:
        return (
            [],
            [],
            None,
            None,
        )

    indices = [
        int(
            row[
                "source_frame_index"
            ]
        )
        for row in rows
    ]

    expected_count = int(
        temporal[
            "decoded_frame_count"
        ]
    )

    if condition == "CLN":
        path = (
            Path(
                temporal[
                    "absolute_source_path"
                ]
            )
            .expanduser()
            .resolve()
        )

        decode_stage = (
            FAILURE_SOURCE_DECODE
        )

    else:
        if qc is None:
            return (
                [],
                [],
                FAILURE_CONDITION_INTEGRITY,
                (
                    "Missing processing QC record."
                ),
            )

        status = str(
            qc.get(
                "status",
                "",
            )
        )

        if status not in VALID_QC:
            return (
                [],
                [],
                FAILURE_CONDITION_INTEGRITY,
                (
                    "Processing QC status is {!r}."
                    .format(
                        status
                    )
                ),
            )

        output_path = str(
            qc.get(
                "output_path",
                "",
            )
        ).strip()

        if not output_path:
            return (
                [],
                [],
                FAILURE_CONDITION_INTEGRITY,
                (
                    "Accepted QC record has "
                    "no output_path."
                ),
            )

        path = (
            Path(
                output_path
            )
            .expanduser()
            .resolve()
        )

        decode_stage = (
            FAILURE_CONDITION_INTEGRITY
        )

    try:
        (
            decoded_count,
            frames,
        ) = decode_selected_frames(
            path,
            indices,
        )

    except Exception as exc:
        return (
            [],
            [],
            decode_stage,
            "{}: {}".format(
                type(
                    exc
                ).__name__,
                exc,
            ),
        )

    if (
        decoded_count
        != expected_count
    ):
        return (
            [],
            [],
            decode_stage,
            (
                "Decoded-frame-count mismatch: "
                "expected={}, actual={}."
                .format(
                    expected_count,
                    decoded_count,
                )
            ),
        )

    positions = []
    images = []

    for row in rows:
        position = int(
            row[
                "temporal_position"
            ]
        )

        frame = frames[
            int(
                row[
                    "source_frame_index"
                ]
            )
        ]

        try:
            matrix = np.asarray(
                parse_json_cell(
                    row[
                        "affine_matrix_json"
                    ]
                ),
                dtype=np.float64,
            )

        except Exception as exc:
            return (
                [],
                [],
                FAILURE_CLEAN_GEOMETRY,
                "{}: {}".format(
                    type(
                        exc
                    ).__name__,
                    exc,
                ),
            )

        if (
            matrix.shape
            != (
                2,
                3,
            )
            or not np.isfinite(
                matrix
            ).all()
        ):
            return (
                [],
                [],
                FAILURE_CLEAN_GEOMETRY,
                (
                    "Invalid affine matrix at "
                    "position {}."
                    .format(
                        position
                    )
                ),
            )

        if condition == "CLN":
            if (
                sha256_array(
                    frame
                )
                != str(
                    row[
                        "source_frame_sha256"
                    ]
                )
            ):
                return (
                    [],
                    [],
                    FAILURE_CLEAN_GEOMETRY,
                    (
                        "Frozen CLN frame hash "
                        "mismatch at position {}."
                        .format(
                            position
                        )
                    ),
                )

        try:
            aligned = (
                apply_saved_geometry(
                    frame,
                    matrix,
                )
            )

        except Exception as exc:
            stage = (
                FAILURE_CLEAN_GEOMETRY
                if condition == "CLN"
                else FAILURE_DETECTOR_INPUT
            )

            return (
                [],
                [],
                stage,
                "{}: {}".format(
                    type(
                        exc
                    ).__name__,
                    exc,
                ),
            )

        if condition == "CLN":
            if (
                sha256_array(
                    aligned
                )
                != str(
                    row[
                        "aligned_sha256"
                    ]
                )
            ):
                return (
                    [],
                    [],
                    FAILURE_CLEAN_GEOMETRY,
                    (
                        "Frozen CLN aligned hash "
                        "mismatch at position {}."
                        .format(
                            position
                        )
                    ),
                )

        positions.append(
            position
        )

        images.append(
            aligned
        )

    return (
        positions,
        images,
        None,
        None,
    )


def existing(
    path,
    arguments,
):
    if not path.exists():
        return {}

    result = {}

    for row in read_jsonl(
        path
    ):
        if (
            str(
                row.get(
                    "detector",
                    "",
                )
            )
            != arguments.detector
            or str(
                row.get(
                    "condition",
                    "",
                )
            )
            != arguments.condition
            or str(
                row.get(
                    "dataset",
                    "",
                )
            )
            != arguments.dataset
            or str(
                row.get(
                    "role",
                    "",
                )
            )
            != arguments.role
            or (
                arguments.subgroup
                is not None
                and str(
                    row.get(
                        "subgroup",
                        "",
                    )
                )
                != arguments.subgroup
            )
        ):
            raise RuntimeError(
                "Existing output contains a record "
                "outside this run identity."
            )

        key = out_key(
            row
        )

        if key in result:
            raise RuntimeError(
                "Duplicate existing record: {}"
                .format(
                    key
                )
            )

        result[
            key
        ] = row

    return result


def main():
    arguments = args()

    if (
        arguments.batch_size
        <= 0
    ):
        raise ValueError(
            "--batch-size must be positive."
        )

    if (
        arguments.limit
        is not None
        and arguments.limit
        <= 0
    ):
        raise ValueError(
            "--limit must be positive."
        )

    if (
        arguments.condition
        == "CLN"
        and arguments.qc_jsonl
        is not None
    ):
        raise ValueError(
            "--qc-jsonl must be omitted for CLN."
        )

    if (
        arguments.condition
        != "CLN"
        and arguments.qc_jsonl
        is None
    ):
        raise ValueError(
            "--qc-jsonl is required for "
            "processed conditions."
        )

    root = (
        arguments.study_root
        .expanduser()
        .resolve()
    )

    temporal_path = (
        root
        / "evaluation"
        / "temporal_plan.csv"
    )

    geometry_path = (
        root
        / "evaluation"
        / "clean_geometry.csv"
    )

    temporal = (
        load_temporal_index(
            temporal_path
        )
    )

    geometry = (
        load_geometry_index(
            geometry_path
        )
    )

    selected = select_rows(
        temporal,
        arguments.dataset,
        arguments.role,
        arguments.subgroup,
        arguments.limit,
    )

    if not selected:
        raise RuntimeError(
            "No temporal-plan videos matched "
            "the requested filters."
        )

    missing_geometry = [
        video_key(
            row
        )
        for row in selected
        if (
            video_key(
                row
            )
            not in geometry
        )
    ]

    if missing_geometry:
        raise RuntimeError(
            "Selected videos are missing clean "
            "geometry: {}"
            .format(
                missing_geometry[
                    :10
                ]
            )
        )

    qpath = None
    qindex = {}

    if (
        arguments.condition
        != "CLN"
    ):
        qpath = (
            arguments.qc_jsonl
            .expanduser()
            .resolve()
        )

        qindex = qc_index(
            qpath,
            arguments.condition,
        )

    accepted = 0
    missing = 0
    invalid = 0

    if (
        arguments.condition
        != "CLN"
    ):
        for row in selected:
            qrow = qindex.get(
                video_key(
                    row
                )
            )

            if qrow is None:
                missing += 1

            elif (
                str(
                    qrow.get(
                        "status",
                        "",
                    )
                )
                in VALID_QC
            ):
                accepted += 1

            else:
                invalid += 1

    print(
        "PRODUCTION INFERENCE {}"
        .format(
            (
                "DRY RUN"
                if arguments.dry_run
                else "RUN"
            )
        )
    )

    print(
        "  detector:        {}"
        .format(
            arguments.detector
        )
    )

    print(
        "  condition:       {}"
        .format(
            arguments.condition
        )
    )

    print(
        "  dataset:         {}"
        .format(
            arguments.dataset
        )
    )

    print(
        "  role:            {}"
        .format(
            arguments.role
        )
    )

    print(
        "  subgroup:        {}"
        .format(
            arguments.subgroup
        )
    )

    print(
        "  selected videos: {}"
        .format(
            len(
                selected
            )
        )
    )

    if (
        arguments.condition
        != "CLN"
    ):
        print(
            "  QC accepted:     {}"
            .format(
                accepted
            )
        )

        print(
            "  QC missing:      {}"
            .format(
                missing
            )
        )

        print(
            "  QC invalid:      {}"
            .format(
                invalid
            )
        )

    if arguments.dry_run:
        return

    output = (
        arguments.output_jsonl
        .expanduser()
        .resolve()
    )

    if (
        output.exists()
        and not arguments.resume
    ):
        raise FileExistsError(
            "Output exists; use --resume or "
            "another path: {}"
            .format(
                output
            )
        )

    done = (
        existing(
            output,
            arguments,
        )
        if arguments.resume
        else {}
    )

    scorer = FrozenDetectorScorer(
        arguments.detector,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )

    started = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    counts = Counter()
    frame_distribution = Counter()

    written = 0
    skipped = 0

    for (
        number,
        temporal_row,
    ) in enumerate(
        selected,
        start=1,
    ):
        key = video_key(
            temporal_row
        )

        rows = geometry[
            key
        ]

        validate_geometry(
            rows,
            key,
        )

        first = rows[
            0
        ]

        record_key = (
            arguments.detector,
            arguments.dataset,
            arguments.role,
            str(
                temporal_row[
                    "relative_source_path"
                ]
            ),
            arguments.condition,
        )

        if record_key in done:
            row = done[
                record_key
            ]

            if (
                str(
                    row[
                        "checkpoint_sha256"
                    ]
                )
                != scorer.checkpoint_sha256
            ):
                raise RuntimeError(
                    "Existing checkpoint SHA "
                    "mismatch for {}."
                    .format(
                        record_key
                    )
                )

            skipped += 1

            counts[
                row[
                    "inference_status"
                ]
            ] += 1

            frame_distribution[
                int(
                    row[
                        "clean_valid_frame_count"
                    ]
                )
            ] += 1

            print(
                "[{}/{}] {} | SKIP {}"
                .format(
                    number,
                    len(
                        selected
                    ),
                    temporal_row[
                        "relative_source_path"
                    ],
                    row[
                        "inference_status"
                    ],
                )
            )

            continue

        valid_rows = valid_geometry(
            rows
        )

        frozen_positions = [
            int(
                row[
                    "temporal_position"
                ]
            )
            for row in valid_rows
        ]

        frame_distribution[
            len(
                frozen_positions
            )
        ] += 1

        qrow = (
            qindex.get(
                key
            )
            if (
                arguments.condition
                != "CLN"
            )
            else None
        )

        (
            positions,
            images,
            failure_stage,
            failure_reason,
        ) = prepare(
            arguments.condition,
            temporal_row,
            rows,
            qrow,
        )

        base_identity = identity(
            first,
            scorer.checkpoint_sha256,
        )

        if (
            failure_stage
            is not None
        ):
            row = finalize_video_inference(
                identity=base_identity,
                condition=arguments.condition,
                detector=arguments.detector,
                clean_valid_positions=(
                    frozen_positions
                ),
                frame_records=[],
                upstream_failure_stage=(
                    failure_stage
                ),
                upstream_failure_reason=(
                    failure_reason
                ),
            )

        elif not frozen_positions:
            row = finalize_video_inference(
                identity=base_identity,
                condition=arguments.condition,
                detector=arguments.detector,
                clean_valid_positions=[],
                frame_records=[],
            )

        else:
            if (
                positions
                != frozen_positions
            ):
                raise RuntimeError(
                    "Prepared positions differ "
                    "from frozen support for {}."
                    .format(
                        key
                    )
                )

            labels = [
                int(
                    first[
                        "study_label"
                    ]
                )
            ] * len(
                images
            )

            try:
                scores = scorer.score(
                    images,
                    labels,
                )

            except Exception as exc:
                row = (
                    finalize_video_inference(
                        identity=base_identity,
                        condition=arguments.condition,
                        detector=arguments.detector,
                        clean_valid_positions=(
                            frozen_positions
                        ),
                        frame_records=[],
                        upstream_failure_stage=(
                            FAILURE_DETECTOR_INFERENCE
                        ),
                        upstream_failure_reason=(
                            "{}: {}"
                            .format(
                                type(
                                    exc
                                ).__name__,
                                exc,
                            )
                        ),
                    )
                )

            else:
                frame_rows = [
                    {
                        "temporal_position": (
                            position
                        ),
                        "score": (
                            score
                        ),
                        "inference_status": (
                            STATUS_OK
                        ),
                        "failure_stage": "",
                        "failure_reason": "",
                    }
                    for (
                        position,
                        score,
                    ) in zip(
                        positions,
                        scores,
                    )
                ]

                row = finalize_video_inference(
                    identity=base_identity,
                    condition=arguments.condition,
                    detector=arguments.detector,
                    clean_valid_positions=(
                        frozen_positions
                    ),
                    frame_records=(
                        frame_rows
                    ),
                )

        append_jsonl(
            output,
            row,
        )

        written += 1

        counts[
            row[
                "inference_status"
            ]
        ] += 1

        print(
            "[{}/{}] {} | {} | frames {}/{}"
            .format(
                number,
                len(
                    selected
                ),
                temporal_row[
                    "relative_source_path"
                ],
                row[
                    "inference_status"
                ],
                row[
                    "successful_frame_count"
                ],
                row[
                    "clean_valid_frame_count"
                ],
            )
        )

    device = str(
        scorer.device
    )

    checkpoint_path = (
        scorer.checkpoint_path
    )

    checkpoint_sha = (
        scorer.checkpoint_sha256
    )

    del scorer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    invalid_count = sum(
        count
        for (
            status,
            count,
        ) in counts.items()
        if (
            status
            != STATUS_OK
        )
    )

    summary = {
        "schema_version": 1,
        "artifact_type": (
            "production_inference_run_summary"
        ),
        "run_started_utc": (
            started
        ),
        "run_finished_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "detector": (
            arguments.detector
        ),
        "condition": (
            arguments.condition
        ),
        "dataset_filter": (
            arguments.dataset
        ),
        "role_filter": (
            arguments.role
        ),
        "subgroup_filter": (
            arguments.subgroup
        ),
        "selected_video_count": (
            len(
                selected
            )
        ),
        "written_record_count": (
            written
        ),
        "skipped_existing_count": (
            skipped
        ),
        "status_counts": dict(
            sorted(
                counts.items()
            )
        ),
        "invalid_record_count": (
            invalid_count
        ),
        "clean_valid_frame_count_distribution": {
            str(
                key
            ): value
            for (
                key,
                value,
            ) in sorted(
                frame_distribution.items()
            )
        },
        "device": (
            device
        ),
        "batch_size": (
            arguments.batch_size
        ),
        "checkpoint_path": (
            str(
                checkpoint_path
            )
        ),
        "checkpoint_sha256": (
            checkpoint_sha
        ),
        "temporal_plan": {
            "path": (
                str(
                    temporal_path
                )
            ),
            "sha256": (
                sha256_file(
                    temporal_path
                )
            ),
        },
        "clean_geometry": {
            "path": (
                str(
                    geometry_path
                )
            ),
            "sha256": (
                sha256_file(
                    geometry_path
                )
            ),
        },
        "processing_qc": (
            None
            if qpath is None
            else {
                "path": (
                    str(
                        qpath
                    )
                ),
                "sha256": (
                    sha256_file(
                        qpath
                    )
                ),
            }
        ),
        "output_jsonl": {
            "path": (
                str(
                    output
                )
            ),
            "sha256": (
                sha256_file(
                    output
                )
            ),
        },
        "status": (
            "passed"
            if invalid_count == 0
            else "completed_with_invalid"
        ),
    }

    summary_path = (
        arguments.summary_json
        .expanduser()
        .resolve()
    )

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "PRODUCTION INFERENCE RUN FINISHED"
    )

    print(
        "  selected videos:  {}"
        .format(
            len(
                selected
            )
        )
    )

    print(
        "  written records:  {}"
        .format(
            written
        )
    )

    print(
        "  skipped existing: {}"
        .format(
            skipped
        )
    )

    print(
        "  status counts:    {}"
        .format(
            dict(
                sorted(
                    counts.items()
                )
            )
        )
    )

    print(
        "  output:           {}"
        .format(
            output
        )
    )

    print(
        "  summary:          {}"
        .format(
            summary_path
        )
    )

    if (
        arguments.fail_on_invalid
        and invalid_count
    ):
        raise RuntimeError(
            "Inference completed with {} invalid "
            "selected video records."
            .format(
                invalid_count
            )
        )


if __name__ == "__main__":
    main()
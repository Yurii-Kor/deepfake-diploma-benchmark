from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


TARGET_FRAME_BUDGET = 32

STATUS_OK = "ok"
STATUS_INSUFFICIENT_VALID_INPUT = (
    "insufficient_valid_input"
)
STATUS_INVALID_CONDITION = (
    "invalid_condition"
)

FAILURE_TEMPORAL_PLAN = "temporal_plan"
FAILURE_CLEAN_GEOMETRY = "clean_reference_geometry"
FAILURE_CONDITION_INTEGRITY = "condition_integrity"
FAILURE_DETECTOR_INPUT = "detector_input"
FAILURE_DETECTOR_INFERENCE = "detector_inference"
FAILURE_SOURCE_DECODE = "source_decode"


def _normalize_positions(
    positions: Iterable[int],
) -> List[int]:
    normalized = [
        int(
            value
        )
        for value in positions
    ]

    if (
        len(
            normalized
        )
        != len(
            set(
                normalized
            )
        )
    ):
        raise ValueError(
            "Temporal positions must be unique."
        )

    normalized.sort()

    for position in normalized:
        if (
            position < 0
            or position >= TARGET_FRAME_BUDGET
        ):
            raise ValueError(
                "Temporal position outside "
                "the target budget: {}".format(
                    position
                )
            )

    return normalized


def invalid_video_record(
    *,
    identity: Dict[str, Any],
    condition: str,
    detector: str,
    clean_valid_positions: Iterable[int],
    failure_stage: str,
    failure_reason: str,
    inference_status: str = STATUS_INVALID_CONDITION,
    successful_frame_count: int = 0,
) -> Dict[str, Any]:
    positions = _normalize_positions(
        clean_valid_positions
    )

    return {
        **identity,
        "detector": detector,
        "condition": condition,
        "target_frame_budget": (
            TARGET_FRAME_BUDGET
        ),
        "clean_valid_frame_count": len(
            positions
        ),
        "retained_temporal_positions": (
            positions
        ),
        "successful_frame_count": int(
            successful_frame_count
        ),
        "aggregation_method": None,
        "video_score": None,
        "inference_status": (
            inference_status
        ),
        "failure_stage": (
            failure_stage
        ),
        "failure_reason": (
            failure_reason
        ),
    }


def finalize_video_inference(
    *,
    identity: Dict[str, Any],
    condition: str,
    detector: str,
    clean_valid_positions: Iterable[int],
    frame_records: Iterable[Dict[str, Any]],
    upstream_failure_stage: Optional[str] = None,
    upstream_failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    positions = _normalize_positions(
        clean_valid_positions
    )

    #
    # A failure before frame scoring invalidates the entire
    # detector-video-condition record.
    #
    if upstream_failure_stage is not None:
        if not upstream_failure_reason:
            raise ValueError(
                "An upstream failure stage requires "
                "a failure reason."
            )

        return invalid_video_record(
            identity=identity,
            condition=condition,
            detector=detector,
            clean_valid_positions=positions,
            failure_stage=(
                upstream_failure_stage
            ),
            failure_reason=(
                upstream_failure_reason
            ),
        )

    #
    # Frozen protocol:
    # zero clean-valid positions means no inference score.
    #
    if not positions:
        return invalid_video_record(
            identity=identity,
            condition=condition,
            detector=detector,
            clean_valid_positions=positions,
            inference_status=(
                STATUS_INSUFFICIENT_VALID_INPUT
            ),
            failure_stage=(
                FAILURE_CLEAN_GEOMETRY
            ),
            failure_reason=(
                "No clean-valid temporal positions "
                "remain after clean-reference "
                "face construction."
            ),
        )

    rows = list(
        frame_records
    )

    rows_by_position = {}

    for row in rows:
        position = int(
            row[
                "temporal_position"
            ]
        )

        if (
            position
            in rows_by_position
        ):
            return invalid_video_record(
                identity=identity,
                condition=condition,
                detector=detector,
                clean_valid_positions=positions,
                failure_stage=(
                    FAILURE_DETECTOR_INFERENCE
                ),
                failure_reason=(
                    "Duplicate frame-level inference "
                    "record for temporal position {}."
                    .format(
                        position
                    )
                ),
            )

        rows_by_position[
            position
        ] = row

    expected = set(
        positions
    )

    actual = set(
        rows_by_position
    )

    missing = sorted(
        expected
        - actual
    )

    extra = sorted(
        actual
        - expected
    )

    #
    # Missing required positions invalidate the condition.
    # Extra positions are not accepted as replacements.
    #
    if missing or extra:
        reason_parts = []

        if missing:
            reason_parts.append(
                "missing required positions {}".format(
                    missing
                )
            )

        if extra:
            reason_parts.append(
                "unexpected substitute/extra positions {}"
                .format(
                    extra
                )
            )

        successful = sum(
            1
            for position in positions
            if (
                position
                in rows_by_position
                and rows_by_position[
                    position
                ].get(
                    "inference_status"
                )
                == STATUS_OK
            )
        )

        return invalid_video_record(
            identity=identity,
            condition=condition,
            detector=detector,
            clean_valid_positions=positions,
            successful_frame_count=(
                successful
            ),
            failure_stage=(
                FAILURE_DETECTOR_INFERENCE
            ),
            failure_reason="; ".join(
                reason_parts
            ),
        )

    scores = []

    successful = 0

    for position in positions:
        row = rows_by_position[
            position
        ]

        status = row.get(
            "inference_status"
        )

        if (
            status
            != STATUS_OK
        ):
            return invalid_video_record(
                identity=identity,
                condition=condition,
                detector=detector,
                clean_valid_positions=positions,
                successful_frame_count=(
                    successful
                ),
                failure_stage=(
                    row.get(
                        "failure_stage"
                    )
                    or FAILURE_DETECTOR_INFERENCE
                ),
                failure_reason=(
                    row.get(
                        "failure_reason"
                    )
                    or (
                        "Required temporal position {} "
                        "did not produce a valid "
                        "detector output."
                    ).format(
                        position
                    )
                ),
            )

        try:
            score = float(
                row[
                    "score"
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return invalid_video_record(
                identity=identity,
                condition=condition,
                detector=detector,
                clean_valid_positions=positions,
                successful_frame_count=(
                    successful
                ),
                failure_stage=(
                    FAILURE_DETECTOR_INFERENCE
                ),
                failure_reason=(
                    "Required temporal position {} "
                    "has no scalar detector score."
                ).format(
                    position
                ),
            )

        if not math.isfinite(
            score
        ):
            return invalid_video_record(
                identity=identity,
                condition=condition,
                detector=detector,
                clean_valid_positions=positions,
                successful_frame_count=(
                    successful
                ),
                failure_stage=(
                    FAILURE_DETECTOR_INFERENCE
                ),
                failure_reason=(
                    "Required temporal position {} "
                    "produced a non-finite detector "
                    "score."
                ).format(
                    position
                ),
            )

        scores.append(
            score
        )

        successful += 1

    if (
        successful
        != len(
            positions
        )
    ):
        raise RuntimeError(
            "Internal completeness invariant failed."
        )

    video_score = (
        sum(
            scores
        )
        / len(
            scores
        )
    )

    if not math.isfinite(
        video_score
    ):
        raise RuntimeError(
            "Arithmetic aggregation produced "
            "a non-finite result."
        )

    return {
        **identity,
        "detector": detector,
        "condition": condition,
        "target_frame_budget": (
            TARGET_FRAME_BUDGET
        ),
        "clean_valid_frame_count": len(
            positions
        ),
        "retained_temporal_positions": (
            positions
        ),
        "successful_frame_count": (
            successful
        ),
        "aggregation_method": (
            "unweighted_arithmetic_mean"
        ),
        "video_score": (
            video_score
        ),
        "inference_status": (
            STATUS_OK
        ),
        "failure_stage": "",
        "failure_reason": "",
    }


def is_valid_video_record(
    record: Dict[str, Any],
) -> bool:
    return (
        record.get(
            "inference_status"
        )
        == STATUS_OK
        and record.get(
            "video_score"
        )
        is not None
    )


def paired_valid_records(
    *,
    clean_records: Iterable[Dict[str, Any]],
    processed_records: Iterable[Dict[str, Any]],
) -> List[
    tuple
]:
    def key(
        row,
    ):
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
        )

    def build_index(
        records,
        name,
    ):
        index = {}

        for row in records:
            if not is_valid_video_record(
                row
            ):
                continue

            item_key = key(
                row
            )

            if item_key in index:
                raise ValueError(
                    "Duplicate valid {} record "
                    "for pairing key: {}".format(
                        name,
                        item_key,
                    )
                )

            index[
                item_key
            ] = row

        return index

    clean_index = build_index(
        clean_records,
        "clean",
    )

    processed_index = build_index(
        processed_records,
        "processed",
    )

    shared = sorted(
        set(
            clean_index
        )
        & set(
            processed_index
        )
    )

    return [
        (
            clean_index[
                item
            ],
            processed_index[
                item
            ],
        )
        for item in shared
    ]
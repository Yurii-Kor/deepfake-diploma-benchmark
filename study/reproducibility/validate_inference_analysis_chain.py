from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from study.reproducibility.validate_analysis_artifacts import (
    ANALYSIS_STATUS_INVALID,
    ANALYSIS_STATUS_VALID,
    analysis_record_key,
    read_jsonl,
    require_field,
    strict_binary_int,
    strict_bool,
    strict_finite_number,
    strict_nonempty_string,
)


STATUS_OK = "ok"
STATUS_INSUFFICIENT_VALID_INPUT = (
    "insufficient_valid_input"
)
STATUS_INVALID_CONDITION = (
    "invalid_condition"
)

SUPPORTED_INFERENCE_STATUSES = {
    STATUS_OK,
    STATUS_INSUFFICIENT_VALID_INPUT,
    STATUS_INVALID_CONDITION,
}

EXPECTED_AGGREGATION_METHOD = (
    "unweighted_arithmetic_mean"
)


def strict_nonnegative_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if type(value) is not int:
        raise ValueError(
            "{} must be an exact integer; "
            "got {!r} (type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    if value < 0:
        raise ValueError(
            "{} must be non-negative; got {}."
            .format(
                field_name,
                value,
            )
        )

    return value


def strict_positive_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    value = strict_nonnegative_int(
        value,
        field_name=field_name,
    )

    if value == 0:
        raise ValueError(
            "{} must be positive.".format(
                field_name
            )
        )

    return value


def strict_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    value = strict_nonempty_string(
        value,
        field_name=field_name,
    )

    if len(value) != 64:
        raise ValueError(
            "{} must contain 64 hexadecimal "
            "characters.".format(
                field_name
            )
        )

    try:
        int(
            value,
            16,
        )
    except ValueError as exc:
        raise ValueError(
            "{} is not a valid SHA-256 value: {!r}."
            .format(
                field_name,
                value,
            )
        ) from exc

    return value.lower()


def inference_record_key(
    record: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        strict_nonempty_string(
            require_field(
                record,
                "detector",
            ),
            field_name="detector",
        ),
        strict_nonempty_string(
            require_field(
                record,
                "dataset",
            ),
            field_name="dataset",
        ),
        strict_nonempty_string(
            require_field(
                record,
                "relative_source_path",
            ),
            field_name="relative_source_path",
        ),
        strict_nonempty_string(
            require_field(
                record,
                "condition",
            ),
            field_name="condition",
        ),
    )


def validate_temporal_positions(
    record: Dict[str, Any],
    *,
    target_frame_budget: int,
    clean_valid_frame_count: int,
) -> None:
    positions = require_field(
        record,
        "retained_temporal_positions",
    )

    if type(positions) is not list:
        raise ValueError(
            "retained_temporal_positions "
            "must be a list."
        )

    parsed = []

    for index, value in enumerate(
        positions
    ):
        position = strict_nonnegative_int(
            value,
            field_name=(
                "retained_temporal_positions[{}]"
                .format(
                    index
                )
            ),
        )

        if position >= target_frame_budget:
            raise ValueError(
                "Temporal position {} is outside "
                "target frame budget {}.".format(
                    position,
                    target_frame_budget,
                )
            )

        parsed.append(
            position
        )

    if len(parsed) != len(set(parsed)):
        raise ValueError(
            "retained_temporal_positions "
            "contains duplicates."
        )

    if parsed != sorted(parsed):
        raise ValueError(
            "retained_temporal_positions "
            "must be sorted."
        )

    if (
        len(parsed)
        != clean_valid_frame_count
    ):
        raise ValueError(
            "retained_temporal_positions count "
            "does not match clean_valid_frame_count."
        )


def validate_inference_record(
    record: Dict[str, Any],
) -> None:
    strict_nonempty_string(
        require_field(
            record,
            "detector",
        ),
        field_name="detector",
    )

    strict_nonempty_string(
        require_field(
            record,
            "dataset",
        ),
        field_name="dataset",
    )

    strict_nonempty_string(
        require_field(
            record,
            "role",
        ),
        field_name="role",
    )

    strict_nonempty_string(
        require_field(
            record,
            "relative_source_path",
        ),
        field_name="relative_source_path",
    )

    strict_nonempty_string(
        require_field(
            record,
            "condition",
        ),
        field_name="condition",
    )

    strict_nonempty_string(
        require_field(
            record,
            "base_video_id",
        ),
        field_name="base_video_id",
    )

    strict_sha256(
        require_field(
            record,
            "checkpoint_sha256",
        ),
        field_name="checkpoint_sha256",
    )

    strict_binary_int(
        require_field(
            record,
            "study_label",
        ),
        field_name="study_label",
    )

    target_frame_budget = (
        strict_positive_int(
            require_field(
                record,
                "target_frame_budget",
            ),
            field_name=(
                "target_frame_budget"
            ),
        )
    )

    clean_valid_frame_count = (
        strict_nonnegative_int(
            require_field(
                record,
                "clean_valid_frame_count",
            ),
            field_name=(
                "clean_valid_frame_count"
            ),
        )
    )

    successful_frame_count = (
        strict_nonnegative_int(
            require_field(
                record,
                "successful_frame_count",
            ),
            field_name=(
                "successful_frame_count"
            ),
        )
    )

    if (
        clean_valid_frame_count
        > target_frame_budget
    ):
        raise ValueError(
            "clean_valid_frame_count exceeds "
            "target_frame_budget."
        )

    if (
        successful_frame_count
        > clean_valid_frame_count
    ):
        raise ValueError(
            "successful_frame_count exceeds "
            "clean_valid_frame_count."
        )

    validate_temporal_positions(
        record,
        target_frame_budget=(
            target_frame_budget
        ),
        clean_valid_frame_count=(
            clean_valid_frame_count
        ),
    )

    inference_status = (
        strict_nonempty_string(
            require_field(
                record,
                "inference_status",
            ),
            field_name="inference_status",
        )
    )

    if (
        inference_status
        not in SUPPORTED_INFERENCE_STATUSES
    ):
        raise ValueError(
            "Unsupported inference_status: {!r}."
            .format(
                inference_status
            )
        )

    video_score = require_field(
        record,
        "video_score",
    )

    aggregation_method = require_field(
        record,
        "aggregation_method",
    )

    failure_stage = require_field(
        record,
        "failure_stage",
    )

    failure_reason = require_field(
        record,
        "failure_reason",
    )

    if inference_status == STATUS_OK:
        if (
            aggregation_method
            != EXPECTED_AGGREGATION_METHOD
        ):
            raise ValueError(
                "Successful inference record has "
                "unexpected aggregation_method."
            )

        strict_finite_number(
            video_score,
            field_name="video_score",
        )

        if (
            successful_frame_count
            != clean_valid_frame_count
        ):
            raise ValueError(
                "Successful inference must evaluate "
                "all clean-valid temporal positions."
            )

        if failure_stage != "":
            raise ValueError(
                "Successful inference must have "
                "empty failure_stage."
            )

        if failure_reason != "":
            raise ValueError(
                "Successful inference must have "
                "empty failure_reason."
            )

        return

    if aggregation_method is not None:
        raise ValueError(
            "Invalid inference must have "
            "aggregation_method=None."
        )

    if video_score is not None:
        raise ValueError(
            "Invalid inference must have "
            "video_score=None."
        )

    strict_nonempty_string(
        failure_stage,
        field_name="failure_stage",
    )

    strict_nonempty_string(
        failure_reason,
        field_name="failure_reason",
    )


def validate_inference_records(
    records: Iterable[
        Dict[str, Any]
    ],
) -> int:
    seen = set()
    count = 0

    for index, record in enumerate(
        records,
        start=1,
    ):
        try:
            validate_inference_record(
                record
            )

            key = inference_record_key(
                record
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid inference record #{}: {}"
                .format(
                    index,
                    exc,
                )
            ) from exc

        if key in seen:
            raise ValueError(
                "Duplicate inference record key: {}."
                .format(
                    key
                )
            )

        seen.add(
            key
        )

        count += 1

    if count == 0:
        raise ValueError(
            "Inference record collection is empty."
        )

    return count


def expected_analysis_state(
    inference_record: Dict[str, Any],
) -> Tuple[str, bool]:
    if (
        inference_record[
            "inference_status"
        ]
        == STATUS_OK
    ):
        return (
            ANALYSIS_STATUS_VALID,
            True,
        )

    return (
        ANALYSIS_STATUS_INVALID,
        False,
    )


def validate_inference_analysis_chain(
    *,
    inference_records: List[
        Dict[str, Any]
    ],
    analysis_records: List[
        Dict[str, Any]
    ],
) -> None:
    inference_by_key = {
        inference_record_key(
            row
        ): row
        for row in inference_records
    }

    analysis_by_key = {
        analysis_record_key(
            row
        ): row
        for row in analysis_records
    }

    inference_keys = set(
        inference_by_key
    )

    analysis_keys = set(
        analysis_by_key
    )

    if inference_keys != analysis_keys:
        missing_analysis = sorted(
            inference_keys
            - analysis_keys
        )

        unexpected_analysis = sorted(
            analysis_keys
            - inference_keys
        )

        raise ValueError(
            "Inference/analysis identity mismatch; "
            "missing_analysis={} "
            "unexpected_analysis={}.".format(
                missing_analysis[:5],
                unexpected_analysis[:5],
            )
        )

    preserved_fields = (
        "checkpoint_sha256",
        "base_video_id",
        "study_label",
        "condition",
        "target_frame_budget",
        "clean_valid_frame_count",
        "successful_frame_count",
        "aggregation_method",
        "video_score",
        "inference_status",
        "failure_stage",
        "failure_reason",
    )

    for key in sorted(
        inference_keys
    ):
        inference = inference_by_key[
            key
        ]

        analysis = analysis_by_key[
            key
        ]

        for field in preserved_fields:
            if (
                inference.get(
                    field
                )
                != analysis.get(
                    field
                )
            ):
                raise ValueError(
                    "Analysis stage changed {} for {}: "
                    "{!r} != {!r}.".format(
                        field,
                        key,
                        inference.get(
                            field
                        ),
                        analysis.get(
                            field
                        ),
                    )
                )

        if (
            inference[
                "role"
            ]
            != analysis.get(
                "inference_role"
            )
        ):
            raise ValueError(
                "Analysis stage did not preserve "
                "inference role for {}: {!r} != {!r}."
                .format(
                    key,
                    inference[
                        "role"
                    ],
                    analysis.get(
                        "inference_role"
                    ),
                )
            )

        (
            expected_status,
            expected_eligible,
        ) = expected_analysis_state(
            inference
        )

        if (
            analysis.get(
                "analysis_status"
            )
            != expected_status
        ):
            raise ValueError(
                "analysis_status is inconsistent "
                "with inference_status for {}."
                .format(
                    key
                )
            )

        if (
            analysis.get(
                "analysis_eligible"
            )
            is not expected_eligible
        ):
            raise ValueError(
                "analysis_eligible is inconsistent "
                "with inference_status for {}."
                .format(
                    key
                )
            )

        expected_complete = (
            inference[
                "clean_valid_frame_count"
            ]
            == inference[
                "target_frame_budget"
            ]
        )

        strict_bool(
            require_field(
                analysis,
                "complete_target_frame_set",
            ),
            field_name=(
                "complete_target_frame_set"
            ),
        )

        if (
            analysis[
                "complete_target_frame_set"
            ]
            is not expected_complete
        ):
            raise ValueError(
                "complete_target_frame_set is "
                "incorrect for {}.".format(
                    key
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate canonical inference "
            "records and verify preservation into "
            "canonical analysis records."
        )
    )

    parser.add_argument(
        "--inference-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    inference_records = read_jsonl(
        args.inference_jsonl
    )

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    inference_count = (
        validate_inference_records(
            inference_records
        )
    )

    validate_inference_analysis_chain(
        inference_records=(
            inference_records
        ),
        analysis_records=(
            analysis_records
        ),
    )

    print(
        "INFERENCE -> ANALYSIS CHAIN VALIDATION"
    )
    print(
        "  inference records:            {}".format(
            inference_count
        )
    )
    print(
        "  strict inference labels:      PASSED"
    )
    print(
        "  strict frame-count schema:    PASSED"
    )
    print(
        "  temporal-support schema:      PASSED"
    )
    print(
        "  inference state invariants:   PASSED"
    )
    print(
        "  cross-stage identities:       PASSED"
    )
    print(
        "  checkpoint preservation:      PASSED"
    )
    print(
        "  label preservation:           PASSED"
    )
    print(
        "  validity preservation:        PASSED"
    )
    print(
        "  score preservation:           PASSED"
    )
    print()
    print(
        "INFERENCE -> ANALYSIS CHAIN VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
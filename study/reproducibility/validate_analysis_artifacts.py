from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ANALYSIS_STATUS_VALID = "valid_score"
ANALYSIS_STATUS_INVALID = "invalid_inference"

DECISION_STATUS_DECIDED = "decided"
DECISION_STATUS_NOT_SCORED = "not_scored"


def read_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            path
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
                value = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON at {} line {}."
                    .format(
                        path,
                        line_number,
                    )
                ) from exc

            if not isinstance(
                value,
                dict,
            ):
                raise ValueError(
                    "JSONL record at {} line {} "
                    "must be an object."
                    .format(
                        path,
                        line_number,
                    )
                )

            records.append(
                value
            )

    return records


def require_field(
    record: Dict[str, Any],
    field_name: str,
) -> Any:
    if field_name not in record:
        raise ValueError(
            "Missing required field: {}.".format(
                field_name
            )
        )

    return record[
        field_name
    ]


def strict_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if type(value) is not str:
        raise ValueError(
            "{} must be a string; "
            "got {!r} (type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    if not value:
        raise ValueError(
            "{} must not be empty.".format(
                field_name
            )
        )

    return value


def strict_binary_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    #
    # Deliberately use exact type checking.
    #
    # bool is a subclass of int in Python, therefore
    # isinstance(True, int) would incorrectly accept True.
    #
    if type(value) is not int:
        raise ValueError(
            "{} must be an exact integer 0 or 1; "
            "got {!r} (type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    if value not in (
        0,
        1,
    ):
        raise ValueError(
            "{} must be 0 or 1; got {!r}.".format(
                field_name,
                value,
            )
        )

    return value


def strict_bool(
    value: Any,
    *,
    field_name: str,
) -> bool:
    if type(value) is not bool:
        raise ValueError(
            "{} must be an exact boolean; "
            "got {!r} (type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    return value


def strict_finite_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    #
    # Again reject booleans explicitly through exact type
    # checking even though bool is numerically coercible.
    #
    if (
        type(value) is not int
        and type(value) is not float
    ):
        raise ValueError(
            "{} must be a JSON number; "
            "got {!r} (type {}).".format(
                field_name,
                value,
                type(value).__name__,
            )
        )

    parsed = float(
        value
    )

    if not math.isfinite(
        parsed
    ):
        raise ValueError(
            "{} must be finite; got {!r}.".format(
                field_name,
                value,
            )
        )

    return parsed


def analysis_record_key(
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


def validate_analysis_record(
    record: Dict[str, Any],
) -> None:
    strict_binary_int(
        require_field(
            record,
            "study_label",
        ),
        field_name="study_label",
    )

    analysis_status = (
        strict_nonempty_string(
            require_field(
                record,
                "analysis_status",
            ),
            field_name="analysis_status",
        )
    )

    analysis_eligible = strict_bool(
        require_field(
            record,
            "analysis_eligible",
        ),
        field_name="analysis_eligible",
    )

    video_score = require_field(
        record,
        "video_score",
    )

    if (
        analysis_status
        == ANALYSIS_STATUS_VALID
    ):
        if analysis_eligible is not True:
            raise ValueError(
                "valid_score record must have "
                "analysis_eligible=True."
            )

        strict_finite_number(
            video_score,
            field_name="video_score",
        )

        return

    if (
        analysis_status
        == ANALYSIS_STATUS_INVALID
    ):
        if analysis_eligible is not False:
            raise ValueError(
                "invalid_inference record must have "
                "analysis_eligible=False."
            )

        if video_score is not None:
            raise ValueError(
                "invalid_inference record must have "
                "video_score=None."
            )

        return

    raise ValueError(
        "Unsupported analysis_status: {!r}.".format(
            analysis_status
        )
    )


def validate_decision_record(
    record: Dict[str, Any],
) -> None:
    #
    # A decision record is an analysis record plus
    # frozen-threshold decision metadata.
    #
    validate_analysis_record(
        record
    )

    strict_nonempty_string(
        require_field(
            record,
            "threshold_id",
        ),
        field_name="threshold_id",
    )

    strict_finite_number(
        require_field(
            record,
            "threshold_value",
        ),
        field_name="threshold_value",
    )

    decision_status = (
        strict_nonempty_string(
            require_field(
                record,
                "decision_status",
            ),
            field_name="decision_status",
        )
    )

    decision = require_field(
        record,
        "decision",
    )

    if (
        record[
            "analysis_eligible"
        ]
        is True
    ):
        if (
            decision_status
            != DECISION_STATUS_DECIDED
        ):
            raise ValueError(
                "Eligible analysis record must have "
                "decision_status='decided'."
            )

        strict_binary_int(
            decision,
            field_name="decision",
        )

        return

    if (
        decision_status
        != DECISION_STATUS_NOT_SCORED
    ):
        raise ValueError(
            "Ineligible analysis record must have "
            "decision_status='not_scored'."
        )

    if decision is not None:
        raise ValueError(
            "not_scored record must have "
            "decision=None."
        )


def validate_analysis_records(
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
            validate_analysis_record(
                record
            )

            key = analysis_record_key(
                record
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid analysis record #{}: {}"
                .format(
                    index,
                    exc,
                )
            ) from exc

        if key in seen:
            raise ValueError(
                "Duplicate analysis record key: {}."
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
            "Analysis record collection is empty."
        )

    return count


def validate_decision_records(
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
            validate_decision_record(
                record
            )

            key = analysis_record_key(
                record
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid decision record #{}: {}"
                .format(
                    index,
                    exc,
                )
            ) from exc

        if key in seen:
            raise ValueError(
                "Duplicate decision record key: {}."
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
            "Decision record collection is empty."
        )

    return count


def validate_cross_stage_identity(
    *,
    analysis_records: List[
        Dict[str, Any]
    ],
    decision_records: List[
        Dict[str, Any]
    ],
) -> None:
    analysis_by_key = {
        analysis_record_key(
            row
        ): row
        for row in analysis_records
    }

    decision_by_key = {
        analysis_record_key(
            row
        ): row
        for row in decision_records
    }

    analysis_keys = set(
        analysis_by_key
    )

    decision_keys = set(
        decision_by_key
    )

    if analysis_keys != decision_keys:
        missing_decisions = sorted(
            analysis_keys
            - decision_keys
        )

        unexpected_decisions = sorted(
            decision_keys
            - analysis_keys
        )

        raise ValueError(
            "Analysis/decision identity mismatch; "
            "missing_decisions={} "
            "unexpected_decisions={}.".format(
                missing_decisions[:5],
                unexpected_decisions[:5],
            )
        )

    preserved_fields = (
        "study_label",
        "analysis_status",
        "analysis_eligible",
        "video_score",
    )

    for key in analysis_keys:
        analysis = analysis_by_key[
            key
        ]

        decision = decision_by_key[
            key
        ]

        for field in preserved_fields:
            if (
                analysis.get(
                    field
                )
                != decision.get(
                    field
                )
            ):
                raise ValueError(
                    "Decision stage changed {} for {}: "
                    "{!r} != {!r}.".format(
                        field,
                        key,
                        analysis.get(
                            field
                        ),
                        decision.get(
                            field
                        ),
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate strict schema and cross-stage "
            "identity for analysis and frozen-decision "
            "JSONL artifacts."
        )
    )

    parser.add_argument(
        "--analysis-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--decisions-jsonl",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    analysis_records = read_jsonl(
        args.analysis_jsonl
    )

    decision_records = read_jsonl(
        args.decisions_jsonl
    )

    analysis_count = (
        validate_analysis_records(
            analysis_records
        )
    )

    decision_count = (
        validate_decision_records(
            decision_records
        )
    )

    validate_cross_stage_identity(
        analysis_records=(
            analysis_records
        ),
        decision_records=(
            decision_records
        ),
    )

    print(
        "ANALYSIS ARTIFACT STRICT SCHEMA VALIDATION"
    )
    print(
        "  analysis records:             {}".format(
            analysis_count
        )
    )
    print(
        "  decision records:             {}".format(
            decision_count
        )
    )
    print(
        "  strict study_label schema:    PASSED"
    )
    print(
        "  strict decision schema:       PASSED"
    )
    print(
        "  analysis state invariants:    PASSED"
    )
    print(
        "  decision state invariants:    PASSED"
    )
    print(
        "  cross-stage identity:         PASSED"
    )
    print(
        "  score/label preservation:     PASSED"
    )
    print()
    print(
        "ANALYSIS ARTIFACT STRICT SCHEMA VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
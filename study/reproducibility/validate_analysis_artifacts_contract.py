from __future__ import annotations

from copy import deepcopy

from study.reproducibility.validate_analysis_artifacts import (
    strict_binary_int,
    validate_analysis_record,
    validate_decision_record,
)


def expect_value_error(
    function,
    *,
    description: str,
):
    try:
        function()
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError: {}".format(
            description
        )
    )


def valid_analysis_record():
    return {
        "detector": "xception",
        "dataset": "FaceForensics++",
        "relative_source_path": (
            "original/test/001.mp4"
        ),
        "condition": "CLN",
        "study_label": 0,
        "analysis_status": "valid_score",
        "analysis_eligible": True,
        "video_score": 0.25,
    }


def valid_decision_record():
    row = valid_analysis_record()

    row.update(
        {
            "threshold_id": (
                "xception-primary"
            ),
            "threshold_value": 0.5,
            "decision_status": "decided",
            "decision": 0,
        }
    )

    return row


def main():
    assert (
        strict_binary_int(
            0,
            field_name="study_label",
        )
        == 0
    )

    assert (
        strict_binary_int(
            1,
            field_name="study_label",
        )
        == 1
    )

    invalid_binary_values = (
        0.0,
        1.0,
        0.7,
        True,
        False,
        "0",
        "1",
        "0.0",
        "yes",
        2,
        -1,
        None,
    )

    for value in invalid_binary_values:
        expect_value_error(
            lambda value=value: (
                strict_binary_int(
                    value,
                    field_name=(
                        "study_label"
                    ),
                )
            ),
            description=(
                "strict binary value {!r}"
                .format(
                    value
                )
            ),
        )

    validate_analysis_record(
        valid_analysis_record()
    )

    invalid_label = deepcopy(
        valid_analysis_record()
    )

    invalid_label[
        "study_label"
    ] = 0.7

    expect_value_error(
        lambda: validate_analysis_record(
            invalid_label
        ),
        description="0.7 study_label",
    )

    boolean_label = deepcopy(
        valid_analysis_record()
    )

    boolean_label[
        "study_label"
    ] = False

    expect_value_error(
        lambda: validate_analysis_record(
            boolean_label
        ),
        description="boolean study_label",
    )

    inconsistent_analysis = deepcopy(
        valid_analysis_record()
    )

    inconsistent_analysis[
        "analysis_eligible"
    ] = False

    expect_value_error(
        lambda: validate_analysis_record(
            inconsistent_analysis
        ),
        description=(
            "valid_score with "
            "analysis_eligible=False"
        ),
    )

    validate_decision_record(
        valid_decision_record()
    )

    invalid_decision = deepcopy(
        valid_decision_record()
    )

    invalid_decision[
        "decision"
    ] = 0.7

    expect_value_error(
        lambda: validate_decision_record(
            invalid_decision
        ),
        description="0.7 decision",
    )

    boolean_decision = deepcopy(
        valid_decision_record()
    )

    boolean_decision[
        "decision"
    ] = False

    expect_value_error(
        lambda: validate_decision_record(
            boolean_decision
        ),
        description="boolean decision",
    )

    not_scored = valid_analysis_record()

    not_scored[
        "analysis_status"
    ] = "invalid_inference"

    not_scored[
        "analysis_eligible"
    ] = False

    not_scored[
        "video_score"
    ] = None

    not_scored.update(
        {
            "threshold_id": (
                "xception-primary"
            ),
            "threshold_value": 0.5,
            "decision_status": (
                "not_scored"
            ),
            "decision": None,
        }
    )

    validate_decision_record(
        not_scored
    )

    invalid_not_scored = deepcopy(
        not_scored
    )

    invalid_not_scored[
        "decision"
    ] = 0

    expect_value_error(
        lambda: validate_decision_record(
            invalid_not_scored
        ),
        description=(
            "not_scored with decision"
        ),
    )

    print(
        "STRICT SCHEMA CONTRACT VALIDATION"
    )
    print(
        "  exact integer labels 0/1:       PASSED"
    )
    print(
        "  float labels rejected:          PASSED"
    )
    print(
        "  string labels rejected:         PASSED"
    )
    print(
        "  boolean labels rejected:        PASSED"
    )
    print(
        "  analysis state invariants:      PASSED"
    )
    print(
        "  exact integer decisions 0/1:    PASSED"
    )
    print(
        "  float decisions rejected:       PASSED"
    )
    print(
        "  boolean decisions rejected:     PASSED"
    )
    print(
        "  not_scored -> decision=None:    PASSED"
    )
    print()
    print(
        "STRICT SCHEMA CONTRACT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
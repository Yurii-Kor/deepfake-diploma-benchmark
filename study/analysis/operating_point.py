from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, Iterable, List

import numpy as np


PRIMARY_TARGET_FPR = 0.05

SCORE_MIN = 0.0
SCORE_MAX = 1.0

PRIMARY_SELECTION_RULE = (
    "lowest_candidate_with_fpr_lte_target"
)

CANDIDATE_RULE = (
    "unique_observed_real_scores_plus_nextafter_max"
)

TIE_RULE = (
    "lowest_numeric_threshold"
)

DECISION_RULE = (
    "manipulated_if_score_gte_threshold"
)

SCORE_DIRECTION = (
    "higher_means_more_manipulated"
)


def normalize_score(
    value: Any,
) -> float:
    try:
        score = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid detector score: {!r}".format(
                value
            )
        ) from exc

    if not math.isfinite(
        score
    ):
        raise ValueError(
            "Detector score is not finite: {!r}"
            .format(
                score
            )
        )

    if (
        score < SCORE_MIN
        or score > SCORE_MAX
    ):
        raise ValueError(
            "Detector score outside [{}, {}]: {}"
            .format(
                SCORE_MIN,
                SCORE_MAX,
                score,
            )
        )

    return score


def normalize_scores(
    values: Iterable[Any],
) -> List[float]:
    scores = [
        normalize_score(
            value
        )
        for value in values
    ]

    if not scores:
        raise ValueError(
            "At least one valid real score "
            "is required."
        )

    return scores


def maximum_allowed_false_positives(
    *,
    n_real: int,
    target_fpr: float,
) -> int:
    if n_real <= 0:
        raise ValueError(
            "n_real must be positive."
        )

    if (
        not math.isfinite(
            target_fpr
        )
        or target_fpr < 0.0
        or target_fpr > 1.0
    ):
        raise ValueError(
            "target_fpr must be finite and "
            "within [0, 1]."
        )

    #
    # Use the decimal representation of the requested
    # operating point rather than binary floating-point
    # multiplication when determining the admissible
    # integer false-positive count.
    #
    exact_target = Fraction(
        str(
            target_fpr
        )
    )

    allowed = (
        exact_target.numerator
        * n_real
        // exact_target.denominator
    )

    return int(
        allowed
    )


def empirical_false_positive_count(
    *,
    real_scores: Iterable[Any],
    threshold: float,
) -> int:
    scores = normalize_scores(
        real_scores
    )

    threshold = float(
        threshold
    )

    if not math.isfinite(
        threshold
    ):
        raise ValueError(
            "Threshold must be finite."
        )

    return sum(
        1
        for score in scores
        if score >= threshold
    )


def empirical_fpr(
    *,
    real_scores: Iterable[Any],
    threshold: float,
) -> float:
    scores = normalize_scores(
        real_scores
    )

    false_positives = sum(
        1
        for score in scores
        if score >= threshold
    )

    return (
        false_positives
        / len(
            scores
        )
    )


def build_threshold_candidates(
    real_scores: Iterable[Any],
) -> List[float]:
    scores = normalize_scores(
        real_scores
    )

    unique_scores = sorted(
        set(
            scores
        )
    )

    maximum_score = max(
        unique_scores
    )

    #
    # An explicit decision-equivalent boundary above
    # the maximum observed score guarantees that a
    # zero-FPR operating point is always representable.
    #
    upper_boundary = float(
        np.nextafter(
            np.float64(
                maximum_score
            ),
            np.float64(
                np.inf
            ),
        )
    )

    if not math.isfinite(
        upper_boundary
    ):
        raise RuntimeError(
            "Unable to construct a finite "
            "upper threshold candidate."
        )

    if (
        upper_boundary
        <= maximum_score
    ):
        raise RuntimeError(
            "Upper threshold candidate is not "
            "strictly above the maximum score."
        )

    candidates = list(
        unique_scores
    )

    candidates.append(
        upper_boundary
    )

    candidates.sort()

    return candidates


def select_primary_low_fpr_threshold(
    *,
    real_scores: Iterable[Any],
    target_fpr: float = PRIMARY_TARGET_FPR,
) -> Dict[str, Any]:
    scores = normalize_scores(
        real_scores
    )

    n_real = len(
        scores
    )

    max_false_positives = (
        maximum_allowed_false_positives(
            n_real=n_real,
            target_fpr=target_fpr,
        )
    )

    candidates = (
        build_threshold_candidates(
            scores
        )
    )

    selected_threshold = None
    selected_false_positives = None

    #
    # Candidates are numerically ascending.
    #
    # Therefore the first admissible candidate is
    # exactly the lowest candidate threshold whose
    # empirical FPR does not exceed the target.
    #
    # This also provides deterministic tie handling:
    # if several candidates satisfy an equivalent
    # admissible operating point, the numerically
    # lowest one is retained.
    #
    for threshold in candidates:
        false_positives = sum(
            1
            for score in scores
            if score >= threshold
        )

        if (
            false_positives
            <= max_false_positives
        ):
            selected_threshold = (
                threshold
            )

            selected_false_positives = (
                false_positives
            )

            break

    if selected_threshold is None:
        raise RuntimeError(
            "No admissible threshold candidate found."
        )

    realized_fpr = (
        selected_false_positives
        / n_real
    )

    if realized_fpr > target_fpr:
        raise RuntimeError(
            "Selected threshold violates "
            "the target FPR."
        )

    return {
        "threshold_value": (
            selected_threshold
        ),
        "target_fpr": (
            float(
                target_fpr
            )
        ),
        "realized_fpr": (
            realized_fpr
        ),
        "n_real": (
            n_real
        ),
        "false_positives": (
            selected_false_positives
        ),
        "max_allowed_false_positives": (
            max_false_positives
        ),
        "candidate_count": (
            len(
                candidates
            )
        ),
        "selection_rule": (
            PRIMARY_SELECTION_RULE
        ),
        "candidate_rule": (
            CANDIDATE_RULE
        ),
        "tie_rule": (
            TIE_RULE
        ),
        "decision_rule": (
            DECISION_RULE
        ),
        "score_direction": (
            SCORE_DIRECTION
        ),
    }


def decision_from_score(
    *,
    score: Any,
    threshold: float,
) -> int:
    normalized_score = normalize_score(
        score
    )

    threshold = float(
        threshold
    )

    if not math.isfinite(
        threshold
    ):
        raise ValueError(
            "Threshold must be finite."
        )

    return (
        1
        if normalized_score >= threshold
        else 0
    )


def _primary_calibration_rows(
    rows: Iterable[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    selected = []

    for row in rows:
        if (
            row.get(
                "dataset"
            )
            != "FaceForensics++"
        ):
            continue

        if (
            row.get(
                "role"
            )
            != "validation"
        ):
            continue

        if (
            row.get(
                "condition"
            )
            != "CLN"
        ):
            continue

        try:
            label = int(
                row.get(
                    "study_label"
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if label != 0:
            continue

        selected.append(
            row
        )

    return selected


def select_primary_thresholds_from_analysis_records(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    target_fpr: float = PRIMARY_TARGET_FPR,
) -> List[Dict[str, Any]]:
    if not math.isclose(
        float(target_fpr),
        PRIMARY_TARGET_FPR,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Primary study operating point is frozen "
            "at target FPR = {}.".format(
                PRIMARY_TARGET_FPR
            )
        )
    
    calibration_rows = (
        _primary_calibration_rows(
            rows
        )
    )

    if not calibration_rows:
        raise ValueError(
            "No clean FF++ validation real "
            "records are available."
        )

    detectors = sorted(
        {
            str(
                row[
                    "detector"
                ]
            )
            for row in calibration_rows
        }
    )

    output = []

    for detector in detectors:
        detector_rows = [
            row
            for row in calibration_rows
            if (
                str(
                    row[
                        "detector"
                    ]
                )
                == detector
            )
        ]

        checkpoint_values = {
            str(
                row[
                    "checkpoint_sha256"
                ]
            )
            for row in detector_rows
        }

        if len(
            checkpoint_values
        ) != 1:
            raise ValueError(
                "Calibration rows for {} contain "
                "multiple checkpoints.".format(
                    detector
                )
            )

        checkpoint_sha256 = next(
            iter(
                checkpoint_values
            )
        )

        pairing_uids = [
            str(
                row[
                    "pairing_uid"
                ]
            )
            for row in detector_rows
        ]

        if (
            len(
                pairing_uids
            )
            != len(
                set(
                    pairing_uids
                )
            )
        ):
            raise ValueError(
                "Duplicate clean validation real "
                "video for detector {}.".format(
                    detector
                )
            )

        valid_rows = [
            row
            for row in detector_rows
            if (
                row.get(
                    "analysis_eligible"
                )
                is True
                and row.get(
                    "video_score"
                )
                is not None
            )
        ]

        if not valid_rows:
            raise ValueError(
                "Detector {} has no valid clean "
                "validation real scores.".format(
                    detector
                )
            )

        selection = (
            select_primary_low_fpr_threshold(
                real_scores=[
                    row[
                        "video_score"
                    ]
                    for row in valid_rows
                ],
                target_fpr=(
                    target_fpr
                ),
            )
        )

        threshold_id = (
            "{}_ffpp_validation_cln_fpr005_{}"
            .format(
                detector,
                checkpoint_sha256[
                    :12
                ],
            )
        )

        result = {
            "schema_version": 1,

            "threshold_id": (
                threshold_id
            ),

            "detector": (
                detector
            ),

            "checkpoint_sha256": (
                checkpoint_sha256
            ),

            "purpose": (
                "primary_operating_point"
            ),

            "calibration_dataset": (
                "FaceForensics++"
            ),

            "calibration_role": (
                "validation"
            ),

            "calibration_condition": (
                "CLN"
            ),

            "calibration_class": (
                "real_only"
            ),

            "nominal_real_records": (
                len(
                    detector_rows
                )
            ),

            "valid_real_records": (
                len(
                    valid_rows
                )
            ),

            "excluded_invalid_real_records": (
                len(
                    detector_rows
                )
                - len(
                    valid_rows
                )
            ),
        }

        result.update(
            selection
        )

        output.append(
            result
        )

    return output


def apply_frozen_threshold(
    *,
    analysis_record: Dict[str, Any],
    threshold_record: Dict[str, Any],
) -> Dict[str, Any]:
    detector = str(
        analysis_record[
            "detector"
        ]
    )

    if (
        detector
        != str(
            threshold_record[
                "detector"
            ]
        )
    ):
        raise ValueError(
            "Threshold detector mismatch."
        )

    checkpoint_sha256 = str(
        analysis_record[
            "checkpoint_sha256"
        ]
    )

    if (
        checkpoint_sha256
        != str(
            threshold_record[
                "checkpoint_sha256"
            ]
        )
    ):
        raise ValueError(
            "Threshold checkpoint mismatch."
        )

    output = dict(
        analysis_record
    )

    output[
        "threshold_id"
    ] = threshold_record[
        "threshold_id"
    ]

    output[
        "threshold_value"
    ] = float(
        threshold_record[
            "threshold_value"
        ]
    )

    output[
        "threshold_selection_rule"
    ] = threshold_record[
        "selection_rule"
    ]

    output[
        "threshold_source_dataset"
    ] = threshold_record[
        "calibration_dataset"
    ]

    output[
        "threshold_source_role"
    ] = threshold_record[
        "calibration_role"
    ]

    output[
        "threshold_source_condition"
    ] = threshold_record[
        "calibration_condition"
    ]

    if (
        analysis_record.get(
            "analysis_eligible"
        )
        is not True
        or analysis_record.get(
            "video_score"
        )
        is None
    ):
        output[
            "decision"
        ] = None

        output[
            "decision_status"
        ] = "not_scored"

        return output

    output[
        "decision"
    ] = decision_from_score(
        score=analysis_record[
            "video_score"
        ],
        threshold=threshold_record[
            "threshold_value"
        ],
    )

    output[
        "decision_status"
    ] = "decided"

    return output
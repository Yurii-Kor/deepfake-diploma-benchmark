from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.operating_point import (
    normalize_score,
)


REAL_LABEL = 0
MANIPULATED_LABEL = 1


def _normalize_scores(
    values: Iterable[Any],
    *,
    name: str,
) -> List[float]:
    scores = [
        normalize_score(
            value
        )
        for value in values
    ]

    if not scores:
        raise ValueError(
            "{} must contain at least one score."
            .format(
                name
            )
        )

    return scores


def _normalize_binary_value(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        return int(
            value
        )

    try:
        normalized = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "{} must contain binary values, got {!r}."
            .format(
                name,
                value,
            )
        ) from exc

    if normalized not in (
        REAL_LABEL,
        MANIPULATED_LABEL,
    ):
        raise ValueError(
            "{} must contain only 0 or 1, got {}."
            .format(
                name,
                normalized,
            )
        )

    return normalized


def _normalize_binary_values(
    values: Iterable[Any],
    *,
    name: str,
) -> List[int]:
    normalized = [
        _normalize_binary_value(
            value,
            name=name,
        )
        for value in values
    ]

    if not normalized:
        raise ValueError(
            "{} must not be empty."
            .format(
                name
            )
        )

    return normalized


def binary_auc(
    *,
    real_scores: Iterable[Any],
    manipulated_scores: Iterable[Any],
) -> float:
    """
    Compute video-level binary AUC with the manipulated
    class treated as positive.

    Higher scores indicate stronger evidence of
    manipulation.

    Tied real/manipulated scores contribute 0.5,
    equivalent to the standard Mann-Whitney
    interpretation of ROC AUC.
    """

    real = _normalize_scores(
        real_scores,
        name="real_scores",
    )

    manipulated = _normalize_scores(
        manipulated_scores,
        name="manipulated_scores",
    )

    observations: List[
        Tuple[
            float,
            int,
        ]
    ] = []

    observations.extend(
        (
            score,
            REAL_LABEL,
        )
        for score in real
    )

    observations.extend(
        (
            score,
            MANIPULATED_LABEL,
        )
        for score in manipulated
    )

    observations.sort(
        key=lambda item: item[
            0
        ]
    )

    manipulated_rank_sum = 0.0

    index = 0
    total = len(
        observations
    )

    while index < total:
        end = index + 1

        current_score = (
            observations[
                index
            ][
                0
            ]
        )

        while (
            end < total
            and observations[
                end
            ][
                0
            ]
            == current_score
        ):
            end += 1

        #
        # observations[index:end] occupy one-based
        # ranks:
        #
        #   index + 1, ..., end
        #
        # Their average rank is therefore:
        #
        #   ((index + 1) + end) / 2
        #
        average_rank = (
            (
                index
                + 1
                + end
            )
            / 2.0
        )

        manipulated_in_group = sum(
            1
            for _, label in (
                observations[
                    index:end
                ]
            )
            if (
                label
                == MANIPULATED_LABEL
            )
        )

        manipulated_rank_sum += (
            average_rank
            * manipulated_in_group
        )

        index = end

    n_real = len(
        real
    )

    n_manipulated = len(
        manipulated
    )

    mann_whitney_u = (
        manipulated_rank_sum
        - (
            n_manipulated
            * (
                n_manipulated
                + 1
            )
            / 2.0
        )
    )

    auc = (
        mann_whitney_u
        / (
            n_manipulated
            * n_real
        )
    )

    if (
        auc < -1e-15
        or auc > 1.0 + 1e-15
    ):
        raise RuntimeError(
            "Computed AUC is outside [0, 1]: {}"
            .format(
                auc
            )
        )

    #
    # Protect only against insignificant floating-point
    # boundary drift.
    #
    auc = min(
        1.0,
        max(
            0.0,
            auc,
        ),
    )

    return auc


def confusion_counts(
    *,
    labels: Iterable[Any],
    decisions: Iterable[Any],
) -> Dict[str, int]:
    """
    Compute confusion counts using:

        real        = negative class = 0
        manipulated = positive class = 1
    """

    normalized_labels = (
        _normalize_binary_values(
            labels,
            name="labels",
        )
    )

    normalized_decisions = (
        _normalize_binary_values(
            decisions,
            name="decisions",
        )
    )

    if (
        len(
            normalized_labels
        )
        != len(
            normalized_decisions
        )
    ):
        raise ValueError(
            "labels and decisions must have "
            "the same length."
        )

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for label, decision in zip(
        normalized_labels,
        normalized_decisions,
    ):
        if (
            label
            == MANIPULATED_LABEL
            and decision
            == MANIPULATED_LABEL
        ):
            true_positive += 1

        elif (
            label
            == REAL_LABEL
            and decision
            == MANIPULATED_LABEL
        ):
            false_positive += 1

        elif (
            label
            == REAL_LABEL
            and decision
            == REAL_LABEL
        ):
            true_negative += 1

        elif (
            label
            == MANIPULATED_LABEL
            and decision
            == REAL_LABEL
        ):
            false_negative += 1

        else:
            raise RuntimeError(
                "Unexpected binary classification state."
            )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def false_positive_rate(
    *,
    false_positives: int,
    true_negatives: int,
) -> float:
    false_positives = int(
        false_positives
    )

    true_negatives = int(
        true_negatives
    )

    if (
        false_positives < 0
        or true_negatives < 0
    ):
        raise ValueError(
            "Confusion counts must be non-negative."
        )

    denominator = (
        false_positives
        + true_negatives
    )

    if denominator <= 0:
        raise ValueError(
            "FPR is undefined without real videos."
        )

    return (
        false_positives
        / denominator
    )


def false_negative_rate(
    *,
    false_negatives: int,
    true_positives: int,
) -> float:
    false_negatives = int(
        false_negatives
    )

    true_positives = int(
        true_positives
    )

    if (
        false_negatives < 0
        or true_positives < 0
    ):
        raise ValueError(
            "Confusion counts must be non-negative."
        )

    denominator = (
        false_negatives
        + true_positives
    )

    if denominator <= 0:
        raise ValueError(
            "FNR is undefined without "
            "manipulated videos."
        )

    return (
        false_negatives
        / denominator
    )


def half_total_error_rate(
    *,
    fpr: float,
    fnr: float,
) -> float:
    try:
        normalized_fpr = float(
            fpr
        )

        normalized_fnr = float(
            fnr
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "FPR and FNR must be numeric."
        ) from exc

    if (
        not math.isfinite(
            normalized_fpr
        )
        or not math.isfinite(
            normalized_fnr
        )
    ):
        raise ValueError(
            "FPR and FNR must be finite."
        )

    if (
        normalized_fpr < 0.0
        or normalized_fpr > 1.0
        or normalized_fnr < 0.0
        or normalized_fnr > 1.0
    ):
        raise ValueError(
            "FPR and FNR must be within [0, 1]."
        )

    return (
        normalized_fpr
        + normalized_fnr
    ) / 2.0


def fixed_threshold_metrics(
    *,
    labels: Iterable[Any],
    decisions: Iterable[Any],
) -> Dict[str, Any]:
    counts = confusion_counts(
        labels=labels,
        decisions=decisions,
    )

    fpr = false_positive_rate(
        false_positives=(
            counts[
                "fp"
            ]
        ),
        true_negatives=(
            counts[
                "tn"
            ]
        ),
    )

    fnr = false_negative_rate(
        false_negatives=(
            counts[
                "fn"
            ]
        ),
        true_positives=(
            counts[
                "tp"
            ]
        ),
    )

    hter = half_total_error_rate(
        fpr=fpr,
        fnr=fnr,
    )

    n_real = (
        counts[
            "tn"
        ]
        + counts[
            "fp"
        ]
    )

    n_manipulated = (
        counts[
            "tp"
        ]
        + counts[
            "fn"
        ]
    )

    return {
        "n_real": (
            n_real
        ),
        "n_manipulated": (
            n_manipulated
        ),

        "tp": (
            counts[
                "tp"
            ]
        ),
        "fp": (
            counts[
                "fp"
            ]
        ),
        "tn": (
            counts[
                "tn"
            ]
        ),
        "fn": (
            counts[
                "fn"
            ]
        ),

        "fpr": (
            fpr
        ),
        "fnr": (
            fnr
        ),
        "hter": (
            hter
        ),
    }
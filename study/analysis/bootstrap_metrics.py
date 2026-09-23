from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Tuple

from study.analysis.bootstrap_sampling import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FFPP_DATASET,
    bootstrap_stratum,
    bootstrap_unit_id,
    sample_bootstrap_replicate,
)
from study.analysis.metrics import (
    binary_auc,
    fixed_threshold_metrics,
)
from study.analysis.operating_point import (
    PRIMARY_TARGET_FPR,
    normalize_score,
    select_primary_low_fpr_threshold,
)


def derive_stream_seed(
    *,
    seed: int,
    stream_name: str,
) -> int:
    if not stream_name:
        raise ValueError(
            "stream_name must not be empty."
        )

    payload = "{}::{}".format(
        int(seed),
        stream_name,
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def _score(
    row: Dict[str, Any],
) -> float:
    return normalize_score(
        row.get(
            "video_score"
        )
    )


def _label(
    row: Dict[str, Any],
) -> int:
    try:
        label = int(
            row[
                "study_label"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid or missing study_label."
        ) from exc

    if label not in (
        0,
        1,
    ):
        raise ValueError(
            "study_label must be 0 or 1."
        )

    return label


def _valid_score_row(
    row: Dict[str, Any],
) -> bool:
    return (
        row.get(
            "analysis_eligible"
        )
        is True
        and row.get(
            "video_score"
        )
        is not None
    )


def _sampling_key(
    row: Dict[str, Any],
) -> Tuple[str, str]:
    return (
        bootstrap_stratum(
            row
        ),
        bootstrap_unit_id(
            row
        ),
    )


def _index_sampling_rows(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    context: str,
) -> Dict[
    Tuple[str, str],
    Dict[str, Any],
]:
    index = {}

    for row in rows:
        if not _valid_score_row(
            row
        ):
            raise ValueError(
                "Invalid score record admitted to {}."
                .format(
                    context
                )
            )

        key = _sampling_key(
            row
        )

        if key in index:
            raise ValueError(
                "Duplicate bootstrap sampling key "
                "in {}: {}"
                .format(
                    context,
                    key,
                )
            )

        index[
            key
        ] = row

    return index


def _materialize_sample(
    *,
    index: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
    sampled_units: Dict[
        str,
        List[str],
    ],
    context: str,
) -> List[Dict[str, Any]]:
    output = []

    for stratum in sorted(
        sampled_units
    ):
        for unit_id in (
            sampled_units[
                stratum
            ]
        ):
            key = (
                stratum,
                unit_id,
            )

            row = index.get(
                key
            )

            if row is None:
                raise ValueError(
                    "Sampled unit missing from {}: {}"
                    .format(
                        context,
                        key,
                    )
                )

            #
            # Append the same immutable-style record
            # reference repeatedly when the unit is
            # drawn repeatedly. Multiplicity is what
            # matters to the bootstrap statistic.
            #
            output.append(
                row
            )

    return output


def _metric_values(
    *,
    real_rows: List[
        Dict[str, Any]
    ],
    manipulated_rows: List[
        Dict[str, Any]
    ],
    threshold: float,
) -> Dict[str, float]:
    if not real_rows:
        raise ValueError(
            "Bootstrap metric requires real rows."
        )

    if not manipulated_rows:
        raise ValueError(
            "Bootstrap metric requires manipulated rows."
        )

    if any(
        _label(
            row
        )
        != 0
        for row in real_rows
    ):
        raise ValueError(
            "Real bootstrap population contains "
            "a manipulated label."
        )

    if any(
        _label(
            row
        )
        != 1
        for row in manipulated_rows
    ):
        raise ValueError(
            "Manipulated bootstrap population "
            "contains a real label."
        )

    real_scores = [
        _score(
            row
        )
        for row in real_rows
    ]

    manipulated_scores = [
        _score(
            row
        )
        for row in manipulated_rows
    ]

    auc = binary_auc(
        real_scores=(
            real_scores
        ),
        manipulated_scores=(
            manipulated_scores
        ),
    )

    labels = (
        [
            0
            for _ in real_rows
        ]
        + [
            1
            for _ in manipulated_rows
        ]
    )

    decisions = (
        [
            (
                1
                if score >= threshold
                else 0
            )
            for score in real_scores
        ]
        + [
            (
                1
                if score >= threshold
                else 0
            )
            for score in (
                manipulated_scores
            )
        ]
    )

    fixed = fixed_threshold_metrics(
        labels=labels,
        decisions=decisions,
    )

    return {
        "auc": (
            auc
        ),
        "fpr": (
            fixed[
                "fpr"
            ]
        ),
        "fnr": (
            fixed[
                "fnr"
            ]
        ),
        "hter": (
            fixed[
                "hter"
            ]
        ),
    }


def _validate_clean_calibration_rows(
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
            != FFPP_DATASET
        ):
            raise ValueError(
                "Primary threshold bootstrap must "
                "use FaceForensics++ validation."
            )

        if (
            row.get(
                "role"
            )
            != "validation"
        ):
            raise ValueError(
                "Primary threshold bootstrap must "
                "use role=validation."
            )

        if (
            row.get(
                "condition"
            )
            != "CLN"
        ):
            raise ValueError(
                "Primary threshold bootstrap must "
                "use condition=CLN."
            )

        if _label(
            row
        ) != 0:
            raise ValueError(
                "Primary threshold bootstrap must "
                "use real validation videos only."
            )

        if not _valid_score_row(
            row
        ):
            continue

        selected.append(
            row
        )

    if not selected:
        raise ValueError(
            "No valid clean FF++ validation real "
            "records are available."
        )

    _index_sampling_rows(
        selected,
        context=(
            "clean validation calibration"
        ),
    )

    return selected


def _bootstrap_clean_threshold(
    *,
    clean_validation_real_rows: List[
        Dict[str, Any]
    ],
    replicate_index: int,
    seed: int,
) -> float:
    stream_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            "primary_clean_validation"
        ),
    )

    sampled_units = (
        sample_bootstrap_replicate(
            clean_validation_real_rows,
            replicate_index=(
                replicate_index
            ),
            seed=stream_seed,
        )
    )

    index = _index_sampling_rows(
        clean_validation_real_rows,
        context=(
            "clean validation calibration"
        ),
    )

    sampled_rows = _materialize_sample(
        index=index,
        sampled_units=(
            sampled_units
        ),
        context=(
            "clean validation calibration"
        ),
    )

    selection = (
        select_primary_low_fpr_threshold(
            real_scores=[
                _score(
                    row
                )
                for row in sampled_rows
            ],
            target_fpr=(
                PRIMARY_TARGET_FPR
            ),
        )
    )

    return float(
        selection[
            "threshold_value"
        ]
    )


def _split_sample_by_label(
    rows: Iterable[
        Dict[str, Any]
    ],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    real = []
    manipulated = []

    for row in rows:
        label = _label(
            row
        )

        if label == 0:
            real.append(
                row
            )
        else:
            manipulated.append(
                row
            )

    return (
        real,
        manipulated,
    )


def bootstrap_absolute_metrics(
    *,
    real_rows: Iterable[
        Dict[str, Any]
    ],
    manipulated_rows: Iterable[
        Dict[str, Any]
    ],
    clean_validation_real_rows: Iterable[
        Dict[str, Any]
    ],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    stream_name: str = "absolute_evaluation",
) -> Dict[str, Any]:
    real_rows = list(
        real_rows
    )

    manipulated_rows = list(
        manipulated_rows
    )

    if not real_rows:
        raise ValueError(
            "Absolute bootstrap requires real rows."
        )

    if not manipulated_rows:
        raise ValueError(
            "Absolute bootstrap requires "
            "manipulated rows."
        )

    if replicates <= 0:
        raise ValueError(
            "replicates must be positive."
        )

    calibration_rows = (
        _validate_clean_calibration_rows(
            clean_validation_real_rows
        )
    )

    population = (
        real_rows
        + manipulated_rows
    )

    population_index = (
        _index_sampling_rows(
            population,
            context=(
                "absolute evaluation"
            ),
        )
    )

    evaluation_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            stream_name
        ),
    )

    replicate_results = []

    for replicate_index in range(
        replicates
    ):
        threshold = (
            _bootstrap_clean_threshold(
                clean_validation_real_rows=(
                    calibration_rows
                ),
                replicate_index=(
                    replicate_index
                ),
                seed=seed,
            )
        )

        sampled_units = (
            sample_bootstrap_replicate(
                population,
                replicate_index=(
                    replicate_index
                ),
                seed=(
                    evaluation_seed
                ),
            )
        )

        sampled_rows = (
            _materialize_sample(
                index=(
                    population_index
                ),
                sampled_units=(
                    sampled_units
                ),
                context=(
                    "absolute evaluation"
                ),
            )
        )

        (
            sampled_real,
            sampled_manipulated,
        ) = _split_sample_by_label(
            sampled_rows
        )

        values = _metric_values(
            real_rows=(
                sampled_real
            ),
            manipulated_rows=(
                sampled_manipulated
            ),
            threshold=(
                threshold
            ),
        )

        replicate_results.append(
            {
                "replicate_index": (
                    replicate_index
                ),

                "threshold": (
                    threshold
                ),

                "auc": (
                    values[
                        "auc"
                    ]
                ),

                "fpr": (
                    values[
                        "fpr"
                    ]
                ),

                "fnr": (
                    values[
                        "fnr"
                    ]
                ),

                "hter": (
                    values[
                        "hter"
                    ]
                ),
            }
        )

    return {
        "replicate_count": (
            replicates
        ),

        "seed": (
            seed
        ),

        "evaluation_stream": (
            stream_name
        ),

        "n_real_units": (
            len(
                {
                    _sampling_key(
                        row
                    )
                    for row in real_rows
                }
            )
        ),

        "n_manipulated_units": (
            len(
                {
                    _sampling_key(
                        row
                    )
                    for row in (
                        manipulated_rows
                    )
                }
            )
        ),

        "replicates": (
            replicate_results
        ),
    }


def _paired_population(
    *,
    clean_rows: Iterable[
        Dict[str, Any]
    ],
    degraded_rows: Iterable[
        Dict[str, Any]
    ],
    context: str,
) -> Tuple[
    Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
    Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
]:
    clean_index = _index_sampling_rows(
        clean_rows,
        context=(
            "{} clean".format(
                context
            )
        ),
    )

    degraded_index = (
        _index_sampling_rows(
            degraded_rows,
            context=(
                "{} degraded".format(
                    context
                )
            ),
        )
    )

    if (
        set(
            clean_index
        )
        != set(
            degraded_index
        )
    ):
        raise ValueError(
            "{} does not contain identical paired "
            "bootstrap units in CLN and degraded "
            "conditions."
            .format(
                context
            )
        )

    return (
        clean_index,
        degraded_index,
    )


def bootstrap_paired_degradation_metrics(
    *,
    clean_real_rows: Iterable[
        Dict[str, Any]
    ],
    clean_manipulated_rows: Iterable[
        Dict[str, Any]
    ],
    degraded_real_rows: Iterable[
        Dict[str, Any]
    ],
    degraded_manipulated_rows: Iterable[
        Dict[str, Any]
    ],
    clean_validation_real_rows: Iterable[
        Dict[str, Any]
    ],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    stream_name: str = "paired_evaluation",
) -> Dict[str, Any]:
    clean_real_rows = list(
        clean_real_rows
    )

    clean_manipulated_rows = list(
        clean_manipulated_rows
    )

    degraded_real_rows = list(
        degraded_real_rows
    )

    degraded_manipulated_rows = list(
        degraded_manipulated_rows
    )

    if replicates <= 0:
        raise ValueError(
            "replicates must be positive."
        )

    calibration_rows = (
        _validate_clean_calibration_rows(
            clean_validation_real_rows
        )
    )

    clean_population = (
        clean_real_rows
        + clean_manipulated_rows
    )

    degraded_population = (
        degraded_real_rows
        + degraded_manipulated_rows
    )

    (
        clean_index,
        degraded_index,
    ) = _paired_population(
        clean_rows=(
            clean_population
        ),
        degraded_rows=(
            degraded_population
        ),
        context=(
            "paired degradation evaluation"
        ),
    )

    evaluation_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            stream_name
        ),
    )

    replicate_results = []

    for replicate_index in range(
        replicates
    ):
        threshold = (
            _bootstrap_clean_threshold(
                clean_validation_real_rows=(
                    calibration_rows
                ),
                replicate_index=(
                    replicate_index
                ),
                seed=seed,
            )
        )

        sampled_units = (
            sample_bootstrap_replicate(
                clean_population,
                replicate_index=(
                    replicate_index
                ),
                seed=(
                    evaluation_seed
                ),
            )
        )

        sampled_clean = (
            _materialize_sample(
                index=clean_index,
                sampled_units=(
                    sampled_units
                ),
                context=(
                    "paired clean evaluation"
                ),
            )
        )

        sampled_degraded = (
            _materialize_sample(
                index=degraded_index,
                sampled_units=(
                    sampled_units
                ),
                context=(
                    "paired degraded evaluation"
                ),
            )
        )

        (
            clean_real,
            clean_manipulated,
        ) = _split_sample_by_label(
            sampled_clean
        )

        (
            degraded_real,
            degraded_manipulated,
        ) = _split_sample_by_label(
            sampled_degraded
        )

        clean_values = _metric_values(
            real_rows=(
                clean_real
            ),
            manipulated_rows=(
                clean_manipulated
            ),
            threshold=(
                threshold
            ),
        )

        degraded_values = (
            _metric_values(
                real_rows=(
                    degraded_real
                ),
                manipulated_rows=(
                    degraded_manipulated
                ),
                threshold=(
                    threshold
                ),
            )
        )

        replicate_results.append(
            {
                "replicate_index": (
                    replicate_index
                ),

                "threshold": (
                    threshold
                ),

                "clean_auc": (
                    clean_values[
                        "auc"
                    ]
                ),

                "degraded_auc": (
                    degraded_values[
                        "auc"
                    ]
                ),

                "delta_auc": (
                    degraded_values[
                        "auc"
                    ]
                    - clean_values[
                        "auc"
                    ]
                ),

                "clean_fpr": (
                    clean_values[
                        "fpr"
                    ]
                ),

                "degraded_fpr": (
                    degraded_values[
                        "fpr"
                    ]
                ),

                "delta_fpr": (
                    degraded_values[
                        "fpr"
                    ]
                    - clean_values[
                        "fpr"
                    ]
                ),

                "clean_fnr": (
                    clean_values[
                        "fnr"
                    ]
                ),

                "degraded_fnr": (
                    degraded_values[
                        "fnr"
                    ]
                ),

                "delta_fnr": (
                    degraded_values[
                        "fnr"
                    ]
                    - clean_values[
                        "fnr"
                    ]
                ),

                "clean_hter": (
                    clean_values[
                        "hter"
                    ]
                ),

                "degraded_hter": (
                    degraded_values[
                        "hter"
                    ]
                ),

                "delta_hter": (
                    degraded_values[
                        "hter"
                    ]
                    - clean_values[
                        "hter"
                    ]
                ),
            }
        )

    return {
        "replicate_count": (
            replicates
        ),

        "seed": (
            seed
        ),

        "evaluation_stream": (
            stream_name
        ),

        "n_paired_real_units": (
            len(
                {
                    _sampling_key(
                        row
                    )
                    for row in (
                        clean_real_rows
                    )
                }
            )
        ),

        "n_paired_manipulated_units": (
            len(
                {
                    _sampling_key(
                        row
                    )
                    for row in (
                        clean_manipulated_rows
                    )
                }
            )
        ),

        "replicates": (
            replicate_results
        ),
    }


def _validation_real_index(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    condition: str,
) -> Dict[str, Dict[str, Any]]:
    index = {}

    for row in rows:
        if (
            row.get(
                "dataset"
            )
            != FFPP_DATASET
            or row.get(
                "role"
            )
            != "validation"
            or row.get(
                "condition"
            )
            != condition
            or _label(
                row
            )
            != 0
            or not _valid_score_row(
                row
            )
        ):
            continue

        unit_id = bootstrap_unit_id(
            row
        )

        if unit_id in index:
            raise ValueError(
                "Duplicate validation real "
                "bootstrap unit: {} | {}"
                .format(
                    condition,
                    unit_id,
                )
            )

        index[
            unit_id
        ] = row

    return index


def bootstrap_diagnostic_threshold_displacement(
    *,
    clean_validation_rows: Iterable[
        Dict[str, Any]
    ],
    degraded_validation_rows: Iterable[
        Dict[str, Any]
    ],
    degraded_condition: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    if (
        degraded_condition
        == "CLN"
    ):
        raise ValueError(
            "Diagnostic degraded condition "
            "must not be CLN."
        )

    if replicates <= 0:
        raise ValueError(
            "replicates must be positive."
        )

    clean_index = _validation_real_index(
        clean_validation_rows,
        condition="CLN",
    )

    degraded_index = (
        _validation_real_index(
            degraded_validation_rows,
            condition=(
                degraded_condition
            ),
        )
    )

    if not clean_index:
        raise ValueError(
            "No valid CLN validation real rows."
        )

    if not degraded_index:
        raise ValueError(
            "No valid degraded validation real rows."
        )

    paired_ids = sorted(
        set(
            clean_index
        )
        & set(
            degraded_index
        )
    )

    if not paired_ids:
        raise ValueError(
            "No paired valid validation real IDs "
            "remain for diagnostic bootstrap."
        )

    paired_clean = [
        clean_index[
            unit_id
        ]
        for unit_id in paired_ids
    ]

    paired_degraded_index = {
        (
            "real",
            unit_id,
        ): degraded_index[
            unit_id
        ]
        for unit_id in paired_ids
    }

    paired_clean_index = {
        (
            "real",
            unit_id,
        ): clean_index[
            unit_id
        ]
        for unit_id in paired_ids
    }

    diagnostic_seed = derive_stream_seed(
        seed=seed,
        stream_name=(
            "diagnostic_validation_{}"
            .format(
                degraded_condition
            )
        ),
    )

    replicate_results = []

    for replicate_index in range(
        replicates
    ):
        sampled_units = (
            sample_bootstrap_replicate(
                paired_clean,
                replicate_index=(
                    replicate_index
                ),
                seed=(
                    diagnostic_seed
                ),
            )
        )

        sampled_clean = (
            _materialize_sample(
                index=(
                    paired_clean_index
                ),
                sampled_units=(
                    sampled_units
                ),
                context=(
                    "diagnostic CLN validation"
                ),
            )
        )

        sampled_degraded = (
            _materialize_sample(
                index=(
                    paired_degraded_index
                ),
                sampled_units=(
                    sampled_units
                ),
                context=(
                    "diagnostic degraded validation"
                ),
            )
        )

        clean_selection = (
            select_primary_low_fpr_threshold(
                real_scores=[
                    _score(
                        row
                    )
                    for row in (
                        sampled_clean
                    )
                ],
                target_fpr=(
                    PRIMARY_TARGET_FPR
                ),
            )
        )

        degraded_selection = (
            select_primary_low_fpr_threshold(
                real_scores=[
                    _score(
                        row
                    )
                    for row in (
                        sampled_degraded
                    )
                ],
                target_fpr=(
                    PRIMARY_TARGET_FPR
                ),
            )
        )

        clean_threshold = float(
            clean_selection[
                "threshold_value"
            ]
        )

        degraded_threshold = float(
            degraded_selection[
                "threshold_value"
            ]
        )

        replicate_results.append(
            {
                "replicate_index": (
                    replicate_index
                ),

                "clean_threshold": (
                    clean_threshold
                ),

                "diagnostic_threshold": (
                    degraded_threshold
                ),

                "threshold_displacement": (
                    degraded_threshold
                    - clean_threshold
                ),
            }
        )

    return {
        "replicate_count": (
            replicates
        ),

        "seed": (
            seed
        ),

        "degraded_condition": (
            degraded_condition
        ),

        "n_clean_valid_real": (
            len(
                clean_index
            )
        ),

        "n_degraded_valid_real": (
            len(
                degraded_index
            )
        ),

        "n_paired_valid_real": (
            len(
                paired_ids
            )
        ),

        "replicates": (
            replicate_results
        ),
    }


def percentile_value(
    values: Iterable[Any],
    *,
    probability: float,
) -> float:
    normalized = [
        float(
            value
        )
        for value in values
    ]

    if not normalized:
        raise ValueError(
            "Percentile input is empty."
        )

    if (
        probability < 0.0
        or probability > 1.0
    ):
        raise ValueError(
            "probability must be within [0, 1]."
        )

    if any(
        not math.isfinite(
            value
        )
        for value in normalized
    ):
        raise ValueError(
            "Percentile input contains "
            "non-finite values."
        )

    normalized.sort()

    if len(
        normalized
    ) == 1:
        return normalized[
            0
        ]

    position = (
        probability
        * (
            len(
                normalized
            )
            - 1
        )
    )

    lower_index = int(
        math.floor(
            position
        )
    )

    upper_index = int(
        math.ceil(
            position
        )
    )

    if (
        lower_index
        == upper_index
    ):
        return normalized[
            lower_index
        ]

    fraction = (
        position
        - lower_index
    )

    lower_value = normalized[
        lower_index
    ]

    upper_value = normalized[
        upper_index
    ]

    return (
        lower_value
        + fraction
        * (
            upper_value
            - lower_value
        )
    )


def percentile_interval(
    values: Iterable[Any],
    *,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> Dict[str, float]:
    if (
        not math.isfinite(
            ci_level
        )
        or ci_level <= 0.0
        or ci_level >= 1.0
    ):
        raise ValueError(
            "ci_level must lie strictly "
            "between 0 and 1."
        )

    values = list(
        values
    )

    if not values:
        raise ValueError(
            "Confidence-interval input is empty."
        )

    tail = (
        1.0
        - ci_level
    ) / 2.0

    return {
        "ci_level": (
            ci_level
        ),

        "lower": percentile_value(
            values,
            probability=tail,
        ),

        "upper": percentile_value(
            values,
            probability=(
                1.0
                - tail
            ),
        ),

        "replicate_count": (
            len(
                values
            )
        ),

        "method": (
            "percentile_linear_interpolation"
        ),
    }
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_SEED = 1024

FFPP_DATASET = "FaceForensics++"
CELEB_DATASET = "Celeb-DF-v2"

FFPP_MANIPULATIONS = (
    "DeepFakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)


def _required_string(
    row: Dict[str, Any],
    field: str,
) -> str:
    if field not in row:
        raise ValueError(
            "Missing required field: {}".format(
                field
            )
        )

    value = str(
        row[
            field
        ]
    ).strip()

    if not value:
        raise ValueError(
            "Required field is empty: {}".format(
                field
            )
        )

    return value


def _label(
    row: Dict[str, Any],
) -> int:
    try:
        value = int(
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

    if value not in (
        0,
        1,
    ):
        raise ValueError(
            "study_label must be 0 or 1."
        )

    return value


def bootstrap_stratum(
    row: Dict[str, Any],
) -> str:
    dataset = _required_string(
        row,
        "dataset",
    )

    label = _label(
        row
    )

    if label == 0:
        return "real"

    if dataset == FFPP_DATASET:
        manipulation = (
            _required_string(
                row,
                "manipulation",
            )
        )

        if (
            manipulation
            not in FFPP_MANIPULATIONS
        ):
            raise ValueError(
                "Unsupported FF++ manipulation: {}"
                .format(
                    manipulation
                )
            )

        return manipulation

    if dataset == CELEB_DATASET:
        return "manipulated"

    raise ValueError(
        "Unsupported dataset: {}".format(
            dataset
        )
    )


def bootstrap_unit_id(
    row: Dict[str, Any],
) -> str:
    dataset = _required_string(
        row,
        "dataset",
    )

    if dataset == FFPP_DATASET:
        return _required_string(
            row,
            "source_video_id",
        )

    if dataset == CELEB_DATASET:
        return _required_string(
            row,
            "base_video_id",
        )

    raise ValueError(
        "Unsupported dataset: {}".format(
            dataset
        )
    )


def bootstrap_sampling_key(
    row: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Bootstrap units are sampled inside explicitly
    defined strata.

    The same source identifier may legitimately occur
    in more than one FF++ stratum. Therefore the
    sampling identity is:

        (stratum, bootstrap_unit_id)

    rather than bootstrap_unit_id alone.
    """

    return (
        bootstrap_stratum(
            row
        ),
        bootstrap_unit_id(
            row
        ),
    )


def build_stratum_units(
    rows: Iterable[
        Dict[str, Any]
    ],
) -> Dict[
    str,
    List[str],
]:
    units_by_stratum = defaultdict(
        set
    )

    rows = list(
        rows
    )

    if not rows:
        raise ValueError(
            "Bootstrap input is empty."
        )

    datasets = {
        _required_string(
            row,
            "dataset",
        )
        for row in rows
    }

    if len(
        datasets
    ) != 1:
        raise ValueError(
            "One bootstrap sampling population "
            "must belong to exactly one dataset."
        )

    for row in rows:
        stratum = bootstrap_stratum(
            row
        )

        unit_id = bootstrap_unit_id(
            row
        )

        units_by_stratum[
            stratum
        ].add(
            unit_id
        )

    return {
        stratum: sorted(
            units
        )
        for (
            stratum,
            units
        ) in sorted(
            units_by_stratum.items()
        )
    }


def sample_units_with_replacement(
    units: List[str],
    *,
    rng: random.Random,
) -> List[str]:
    if not units:
        raise ValueError(
            "Cannot bootstrap an empty stratum."
        )

    return [
        units[
            rng.randrange(
                len(
                    units
                )
            )
        ]
        for _ in range(
            len(
                units
            )
        )
    ]


def sample_stratified_units(
    *,
    units_by_stratum: Dict[
        str,
        List[str],
    ],
    rng: random.Random,
) -> Dict[
    str,
    List[str],
]:
    if not units_by_stratum:
        raise ValueError(
            "No bootstrap strata are available."
        )

    sampled = {}

    for stratum in sorted(
        units_by_stratum
    ):
        sampled[
            stratum
        ] = sample_units_with_replacement(
            units_by_stratum[
                stratum
            ],
            rng=rng,
        )

    return sampled


def replicate_rng(
    *,
    replicate_index: int,
    seed: int = BOOTSTRAP_SEED,
) -> random.Random:
    if replicate_index < 0:
        raise ValueError(
            "replicate_index must be non-negative."
        )

    #
    # Derive a deterministic independent RNG stream
    # for each replicate.
    #
    # This makes replicate r reproducible without
    # depending on whether earlier replicates were
    # executed in the same process.
    #
    derived_seed = (
        int(
            seed
        )
        + int(
            replicate_index
        )
    )

    return random.Random(
        derived_seed
    )


def sample_bootstrap_replicate(
    rows: Iterable[
        Dict[str, Any]
    ],
    *,
    replicate_index: int,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[
    str,
    List[str],
]:
    units_by_stratum = (
        build_stratum_units(
            rows
        )
    )

    rng = replicate_rng(
        replicate_index=(
            replicate_index
        ),
        seed=seed,
    )

    return sample_stratified_units(
        units_by_stratum=(
            units_by_stratum
        ),
        rng=rng,
    )
"""
Executable validation of the study-controlled training samplers.

Run from the repository root:

    python -m study.training.validation
"""

from collections import Counter
from typing import Callable, Dict, List

from .sampling import (
    FAKE_METHODS,
    TARGET_FAKE_EXPOSURES_PER_METHOD,
    BalancedRealFakeBatchSampler,
    FixedMethodExposureSampler,
)


FROZEN_FIT_SOURCE_COUNTS = {
    "FF-real": 18426,
    "FF-DF": 18398,
    "FF-F2F": 18423,
    "FF-FS": 18423,
    "FF-NT": 18407,
}

FROZEN_REPEAT_COUNTS = {
    "FF-DF": 34,
    "FF-F2F": 9,
    "FF-FS": 9,
    "FF-NT": 25,
}


def _expect_failure(
    description: str,
    operation: Callable[[], None],
) -> None:
    try:
        operation()
    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Expected failure was not raised: {}".format(
            description
        )
    )


def _make_frame_population(
    source_counts: Dict[str, int],
):
    source_labels = []
    labels = []

    for source_label in (
        "FF-real",
        "FF-DF",
        "FF-F2F",
        "FF-FS",
        "FF-NT",
    ):
        count = source_counts[
            source_label
        ]

        source_labels.extend(
            [source_label] * count
        )

        binary_label = (
            0
            if source_label == "FF-real"
            else 1
        )

        labels.extend(
            [binary_label] * count
        )

    return (
        labels,
        source_labels,
    )


def _validate_balanced_batches(
    labels: List[int],
    source_labels: List[str],
    sampler: BalancedRealFakeBatchSampler,
):
    batches = list(
        sampler
    )

    if len(batches) != len(
        sampler
    ):
        raise AssertionError(
            "reported sampler length does not match generated batches"
        )

    fake_source_indices = {
        index
        for index, label
        in enumerate(labels)
        if label == 1
    }

    fake_seen = []
    method_exposures = Counter()

    real_exposures = 0
    fake_exposures = 0

    for batch_number, batch in enumerate(
        batches
    ):
        if len(
            batch
        ) != sampler.batch_size:
            raise AssertionError(
                "batch {} has size {}, expected {}".format(
                    batch_number,
                    len(batch),
                    sampler.batch_size,
                )
            )

        real_count = sum(
            labels[index] == 0
            for index in batch
        )

        fake_count = sum(
            labels[index] == 1
            for index in batch
        )

        if (
            real_count
            != sampler.half_batch_size
        ):
            raise AssertionError(
                "batch {} contains {} real items, expected {}".format(
                    batch_number,
                    real_count,
                    sampler.half_batch_size,
                )
            )

        if (
            fake_count
            != sampler.half_batch_size
        ):
            raise AssertionError(
                "batch {} contains {} fake items, expected {}".format(
                    batch_number,
                    fake_count,
                    sampler.half_batch_size,
                )
            )

        real_exposures += real_count
        fake_exposures += fake_count

        for index in batch:
            if labels[
                index
            ] != 1:
                continue

            fake_seen.append(
                index
            )

            method_exposures[
                source_labels[
                    index
                ]
            ] += 1

    fake_counter = Counter(
        fake_seen
    )

    if set(
        fake_counter
    ) != fake_source_indices:
        missing = sorted(
            fake_source_indices
            - set(
                fake_counter
            )
        )

        unexpected = sorted(
            set(
                fake_counter
            )
            - fake_source_indices
        )

        raise AssertionError(
            "fake population mismatch; "
            "missing={}, unexpected={}".format(
                missing[:10],
                unexpected[:10],
            )
        )

    for source_index in fake_source_indices:
        if fake_counter[
            source_index
        ] < 1:
            raise AssertionError(
                "valid fake source item was not exposed"
            )

    if (
        real_exposures
        != fake_exposures
    ):
        raise AssertionError(
            "real/fake epoch exposures are not balanced"
        )

    return {
        "batches": len(
            batches
        ),
        "real_exposures": real_exposures,
        "fake_exposures": fake_exposures,
        "method_exposures": dict(
            method_exposures
        ),
        "fake_counter": fake_counter,
    }


def _validate_small_synthetic_case() -> None:
    source_counts = {
        "FF-real": 8,
        "FF-DF": 4,
        "FF-F2F": 5,
        "FF-FS": 5,
        "FF-NT": 3,
    }

    target_per_method = 5

    (
        labels,
        source_labels,
    ) = _make_frame_population(
        source_counts
    )

    sampler = (
        BalancedRealFakeBatchSampler(
            labels=labels,
            source_labels=source_labels,
            batch_size=8,
            target_fake_exposures_per_method=(
                target_per_method
            ),
            seed=1024,
        )
    )

    sampler.set_epoch(
        0
    )

    stats = (
        _validate_balanced_batches(
            labels=labels,
            source_labels=source_labels,
            sampler=sampler,
        )
    )

    expected_method_exposures = {
        method: target_per_method
        for method in FAKE_METHODS
    }

    if (
        stats[
            "method_exposures"
        ]
        != expected_method_exposures
    ):
        raise AssertionError(
            "unexpected fake-method exposures: {}".format(
                stats[
                    "method_exposures"
                ]
            )
        )

    summary = (
        sampler.contract_summary()
    )

    expected_repeats = {
        "FF-DF": 1,
        "FF-F2F": 0,
        "FF-FS": 0,
        "FF-NT": 2,
    }

    if (
        summary[
            "fake_method_repeat_counts"
        ]
        != expected_repeats
    ):
        raise AssertionError(
            "unexpected synthetic top-up counts"
        )

    batches_epoch_0_a = list(
        sampler
    )

    sampler_same = (
        BalancedRealFakeBatchSampler(
            labels=labels,
            source_labels=source_labels,
            batch_size=8,
            target_fake_exposures_per_method=(
                target_per_method
            ),
            seed=1024,
        )
    )

    sampler_same.set_epoch(
        0
    )

    batches_epoch_0_b = list(
        sampler_same
    )

    if (
        batches_epoch_0_a
        != batches_epoch_0_b
    ):
        raise AssertionError(
            "same seed and epoch did not reproduce identical batches"
        )

    sampler.set_epoch(
        1
    )

    batches_epoch_1 = list(
        sampler
    )

    if (
        batches_epoch_1
        == batches_epoch_0_a
    ):
        raise AssertionError(
            "different epochs unexpectedly produced identical batches"
        )

    print(
        "SMALL SYNTHETIC TOP-UP CASE"
    )

    print(
        "  real source items:          8"
    )

    print(
        "  fake source items:          17"
    )

    print(
        "  fake target/method:         5"
    )

    print(
        "  fake exposures/epoch:       20"
    )

    print(
        "  real exposures/epoch:       20"
    )

    print(
        "  batches/epoch:              {}".format(
            stats[
                "batches"
            ]
        )
    )

    print(
        "  top-up counts:              {}".format(
            expected_repeats
        )
    )

    print(
        "  every valid fake:           exposed >= 1 time"
    )

    print(
        "  same seed + epoch:          reproducible"
    )

    print(
        "  next epoch:                 different ordering/top-up"
    )


def _validate_frozen_ffpp_fit_case() -> None:
    (
        labels,
        source_labels,
    ) = _make_frame_population(
        FROZEN_FIT_SOURCE_COUNTS
    )

    sampler = (
        BalancedRealFakeBatchSampler(
            labels=labels,
            source_labels=source_labels,
            batch_size=32,
            target_fake_exposures_per_method=(
                TARGET_FAKE_EXPOSURES_PER_METHOD
            ),
            seed=1024,
        )
    )

    sampler.set_epoch(
        0
    )

    stats = (
        _validate_balanced_batches(
            labels=labels,
            source_labels=source_labels,
            sampler=sampler,
        )
    )

    expected_method_exposures = {
        method: (
            TARGET_FAKE_EXPOSURES_PER_METHOD
        )
        for method in FAKE_METHODS
    }

    if (
        stats[
            "method_exposures"
        ]
        != expected_method_exposures
    ):
        raise AssertionError(
            "frozen fake-method exposure targets were not preserved"
        )

    if len(
        sampler
    ) != 4608:
        raise AssertionError(
            "expected 4608 optimizer steps"
        )

    if (
        sampler.fake_exposures_per_epoch
        != 73728
    ):
        raise AssertionError(
            "unexpected fake exposure count"
        )

    if (
        sampler.real_exposures_per_epoch
        != 73728
    ):
        raise AssertionError(
            "unexpected real exposure count"
        )

    if (
        sampler.image_exposures_per_epoch
        != 147456
    ):
        raise AssertionError(
            "unexpected total image exposure count"
        )

    summary = (
        sampler.contract_summary()
    )

    if (
        summary[
            "fake_method_source_counts"
        ]
        != {
            method: (
                FROZEN_FIT_SOURCE_COUNTS[
                    method
                ]
            )
            for method in FAKE_METHODS
        }
    ):
        raise AssertionError(
            "frozen fake source counts changed"
        )

    if (
        summary[
            "fake_method_repeat_counts"
        ]
        != FROZEN_REPEAT_COUNTS
    ):
        raise AssertionError(
            "frozen minimal top-up counts changed"
        )

    print()
    print(
        "FROZEN FF++ FIT CASE"
    )

    print(
        "  real valid source frames:   {}".format(
            FROZEN_FIT_SOURCE_COUNTS[
                "FF-real"
            ]
        )
    )

    for method in FAKE_METHODS:
        print(
            "  {:<6} valid/target/repeat:   "
            "{}/{}/{}".format(
                method,
                FROZEN_FIT_SOURCE_COUNTS[
                    method
                ],
                TARGET_FAKE_EXPOSURES_PER_METHOD,
                FROZEN_REPEAT_COUNTS[
                    method
                ],
            )
        )

    print(
        "  optimizer steps/epoch:      4608"
    )

    print(
        "  effective images/step:      32"
    )

    print(
        "  real exposures/epoch:       73728"
    )

    print(
        "  fake exposures/epoch:       73728"
    )

    print(
        "  image exposures/epoch:      147456"
    )

    print(
        "  materialization failures:   no temporal replacement"
    )


def _validate_ucf_sampler() -> None:
    source_counts = {
        1: 4,
        2: 5,
        3: 5,
        4: 3,
    }

    method_labels = []

    for method_label in (
        1,
        2,
        3,
        4,
    ):
        method_labels.extend(
            [
                method_label
            ]
            * source_counts[
                method_label
            ]
        )

    method_names = {
        1: "FF-DF",
        2: "FF-F2F",
        3: "FF-FS",
        4: "FF-NT",
    }

    method_targets = {
        method_label: 5
        for method_label in (
            1,
            2,
            3,
            4,
        )
    }

    sampler = (
        FixedMethodExposureSampler(
            method_labels=method_labels,
            method_order=[
                1,
                2,
                3,
                4,
            ],
            method_targets=method_targets,
            method_names=method_names,
            seed=1024,
        )
    )

    sampler.set_epoch(
        0
    )

    plan_a = list(
        sampler
    )

    if len(
        plan_a
    ) != 20:
        raise AssertionError(
            "unexpected UCF pair exposure count"
        )

    if set(
        plan_a
    ) != set(
        range(
            len(
                method_labels
            )
        )
    ):
        raise AssertionError(
            "UCF sampler did not preserve every valid fake source item"
        )

    exposure_counts = Counter(
        method_labels[
            index
        ]
        for index in plan_a
    )

    if exposure_counts != Counter(
        method_targets
    ):
        raise AssertionError(
            "UCF fake-method targets were not preserved"
        )

    sampler_same = (
        FixedMethodExposureSampler(
            method_labels=method_labels,
            method_order=[
                1,
                2,
                3,
                4,
            ],
            method_targets=method_targets,
            method_names=method_names,
            seed=1024,
        )
    )

    sampler_same.set_epoch(
        0
    )

    if plan_a != list(
        sampler_same
    ):
        raise AssertionError(
            "UCF sampler is not reproducible for same seed and epoch"
        )

    sampler.set_epoch(
        1
    )

    if plan_a == list(
        sampler
    ):
        raise AssertionError(
            "UCF sampler did not change between epochs"
        )

    summary = (
        sampler_same.contract_summary()
    )

    if (
        summary[
            "fake_method_repeat_counts"
        ]
        != {
            "FF-DF": 1,
            "FF-F2F": 0,
            "FF-FS": 0,
            "FF-NT": 2,
        }
    ):
        raise AssertionError(
            "unexpected UCF top-up counts"
        )

    print()
    print(
        "UCF FIXED-METHOD SAMPLER"
    )

    print(
        "  valid fake pair-items:      17"
    )

    print(
        "  target pair-items:          20"
    )

    print(
        "  method balance:             5 each"
    )

    print(
        "  every valid fake:           exposed >= 1 time"
    )

    print(
        "  same seed + epoch:          reproducible"
    )

    print(
        "  next epoch:                 different ordering/top-up"
    )

    print(
        "  real partner selection:     remains pairDataset responsibility"
    )


def _validate_negative_cases() -> None:
    _expect_failure(
        "odd batch size",
        lambda: BalancedRealFakeBatchSampler(
            labels=[
                0,
                1,
            ],
            source_labels=[
                "FF-real",
                "FF-DF",
            ],
            batch_size=3,
            target_fake_exposures_per_method=1,
        ),
    )

    _expect_failure(
        "non-binary label",
        lambda: BalancedRealFakeBatchSampler(
            labels=[
                0,
                2,
            ],
            source_labels=[
                "FF-real",
                "FF-DF",
            ],
            batch_size=2,
            target_fake_exposures_per_method=1,
        ),
    )

    _expect_failure(
        "metadata length mismatch",
        lambda: BalancedRealFakeBatchSampler(
            labels=[
                0,
                1,
            ],
            source_labels=[
                "FF-real",
            ],
            batch_size=2,
            target_fake_exposures_per_method=1,
        ),
    )

    _expect_failure(
        "missing fake manipulation group",
        lambda: BalancedRealFakeBatchSampler(
            labels=[
                0,
                1,
            ],
            source_labels=[
                "FF-real",
                "FF-DF",
            ],
            batch_size=2,
            target_fake_exposures_per_method=1,
        ),
    )

    _expect_failure(
        "valid source population exceeds target",
        lambda: BalancedRealFakeBatchSampler(
            labels=[
                0,
                1,
                1,
                1,
                1,
                1,
            ],
            source_labels=[
                "FF-real",
                "FF-DF",
                "FF-DF",
                "FF-F2F",
                "FF-FS",
                "FF-NT",
            ],
            batch_size=2,
            target_fake_exposures_per_method=1,
        ),
    )

    valid_sampler = (
        BalancedRealFakeBatchSampler(
            labels=[
                0,
                1,
                1,
                1,
                1,
            ],
            source_labels=[
                "FF-real",
                "FF-DF",
                "FF-F2F",
                "FF-FS",
                "FF-NT",
            ],
            batch_size=2,
            target_fake_exposures_per_method=1,
        )
    )

    _expect_failure(
        "negative epoch",
        lambda: valid_sampler.set_epoch(
            -1
        ),
    )

    _expect_failure(
        "UCF target-key mismatch",
        lambda: FixedMethodExposureSampler(
            method_labels=[
                1,
                2,
                3,
                4,
            ],
            method_order=[
                1,
                2,
                3,
                4,
            ],
            method_targets={
                1: 1,
                2: 1,
                3: 1,
            },
            method_names={
                1: "FF-DF",
                2: "FF-F2F",
                3: "FF-FS",
                4: "FF-NT",
            },
        ),
    )

    print()
    print(
        "NEGATIVE / INVARIANT TESTS"
    )

    print(
        "  invalid configurations:     rejected"
    )

    print(
        "  missing method group:       rejected"
    )

    print(
        "  dropping valid inputs:      rejected"
    )

    print(
        "  metadata mismatch:          rejected"
    )


def main() -> None:
    _validate_small_synthetic_case()
    _validate_frozen_ffpp_fit_case()
    _validate_ucf_sampler()
    _validate_negative_cases()

    print()
    print(
        "SAMPLER VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
"""
Executable validation of the study-controlled training sampler.

Run from the repository root:

    python -m study.training.validation
"""

from collections import Counter
from typing import Callable, Dict, List

from .sampling import BalancedRealFakeBatchSampler


def _expect_failure(
    description: str,
    operation: Callable[[], None],
) -> None:
    try:
        operation()
    except (TypeError, ValueError):
        return

    raise AssertionError(
        "Expected failure was not raised: {}".format(
            description
        )
    )


def _validate_batches(
    labels: List[int],
    sampler: BalancedRealFakeBatchSampler,
) -> Dict[str, int]:
    batches = list(sampler)

    if len(batches) != len(sampler):
        raise AssertionError(
            "reported sampler length does not match generated batches"
        )

    fake_source = {
        index
        for index, label in enumerate(labels)
        if label == 1
    }

    fake_seen = []
    real_exposures = 0
    fake_exposures = 0

    for batch_number, batch in enumerate(batches):
        if len(batch) != sampler.batch_size:
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

        if real_count != sampler.half_batch_size:
            raise AssertionError(
                "batch {} contains {} real items, expected {}".format(
                    batch_number,
                    real_count,
                    sampler.half_batch_size,
                )
            )

        if fake_count != sampler.half_batch_size:
            raise AssertionError(
                "batch {} contains {} fake items, expected {}".format(
                    batch_number,
                    fake_count,
                    sampler.half_batch_size,
                )
            )

        real_exposures += real_count
        fake_exposures += fake_count

        fake_seen.extend(
            index
            for index in batch
            if labels[index] == 1
        )

    fake_counter = Counter(
        fake_seen
    )

    if set(fake_counter) != fake_source:
        missing = sorted(
            fake_source - set(fake_counter)
        )
        unexpected = sorted(
            set(fake_counter) - fake_source
        )

        raise AssertionError(
            "fake population mismatch; missing={}, unexpected={}".format(
                missing[:10],
                unexpected[:10],
            )
        )

    duplicated_fake = [
        index
        for index, count in fake_counter.items()
        if count != 1
    ]

    if duplicated_fake:
        raise AssertionError(
            "fake items were not exposed exactly once: {}".format(
                duplicated_fake[:10]
            )
        )

    if real_exposures != len(fake_source):
        raise AssertionError(
            "real/fake epoch exposures are not balanced"
        )

    return {
        "batches": len(batches),
        "real_exposures": real_exposures,
        "fake_exposures": fake_exposures,
    }


def _validate_small_synthetic_case() -> None:
    # 8 source real items.
    real_count = 8

    # 16 fake items:
    # 4 DF + 4 F2F + 4 FS + 4 NT.
    fake_methods = (
        ["DF"] * 4
        + ["F2F"] * 4
        + ["FS"] * 4
        + ["NT"] * 4
    )

    labels = (
        [0] * real_count
        + [1] * len(fake_methods)
    )

    method_by_index = {
        real_count + offset: method
        for offset, method in enumerate(
            fake_methods
        )
    }

    sampler = BalancedRealFakeBatchSampler(
        labels=labels,
        batch_size=8,
        seed=1024,
    )

    sampler.set_epoch(0)

    stats = _validate_batches(
        labels,
        sampler,
    )

    batches_epoch_0_a = list(
        sampler
    )

    sampler_same = BalancedRealFakeBatchSampler(
        labels=labels,
        batch_size=8,
        seed=1024,
    )
    sampler_same.set_epoch(0)

    batches_epoch_0_b = list(
        sampler_same
    )

    if batches_epoch_0_a != batches_epoch_0_b:
        raise AssertionError(
            "same seed and epoch did not reproduce identical batches"
        )

    sampler.set_epoch(1)
    batches_epoch_1 = list(
        sampler
    )

    if batches_epoch_1 == batches_epoch_0_a:
        raise AssertionError(
            "different epochs unexpectedly produced identical batches"
        )

    method_counts = Counter()

    for batch in batches_epoch_0_a:
        for index in batch:
            if labels[index] == 1:
                method_counts[
                    method_by_index[index]
                ] += 1

    expected_method_counts = {
        "DF": 4,
        "F2F": 4,
        "FS": 4,
        "NT": 4,
    }

    if dict(method_counts) != expected_method_counts:
        raise AssertionError(
            "fake-method exposure changed: {}".format(
                dict(method_counts)
            )
        )

    print("SMALL SYNTHETIC CASE")
    print(
        "  real source items:          {}".format(
            real_count
        )
    )
    print(
        "  fake source items:          {}".format(
            len(fake_methods)
        )
    )
    print(
        "  batch size:                 {}".format(
            sampler.batch_size
        )
    )
    print(
        "  real/fake per batch:        {}/{}".format(
            sampler.half_batch_size,
            sampler.half_batch_size,
        )
    )
    print(
        "  batches per epoch:          {}".format(
            stats["batches"]
        )
    )
    print(
        "  fake exposure:              exactly once"
    )
    print(
        "  real sampling:              with replacement"
    )
    print(
        "  fake-method counts:         {}".format(
            dict(method_counts)
        )
    )
    print(
        "  same seed + epoch:          reproducible"
    )
    print(
        "  different epoch:            different ordering"
    )


def _validate_nominal_ffpp_fit_case() -> None:
    frames_per_video = 32

    real_videos = 576

    fake_videos_per_method = {
        "DF": 576,
        "F2F": 576,
        "FS": 576,
        "NT": 576,
    }

    real_frames = (
        real_videos
        * frames_per_video
    )

    fake_frames_by_method = {
        method: video_count * frames_per_video
        for method, video_count
        in fake_videos_per_method.items()
    }

    fake_frames = sum(
        fake_frames_by_method.values()
    )

    labels = (
        [0] * real_frames
        + [1] * fake_frames
    )

    sampler = BalancedRealFakeBatchSampler(
        labels=labels,
        batch_size=32,
        seed=1024,
    )
    sampler.set_epoch(0)

    stats = _validate_batches(
        labels,
        sampler,
    )

    expected_steps = 4608
    expected_real_exposures = 73728
    expected_fake_exposures = 73728
    expected_total_exposures = 147456

    if len(sampler) != expected_steps:
        raise AssertionError(
            "expected {} optimizer steps, got {}".format(
                expected_steps,
                len(sampler),
            )
        )

    if (
        sampler.real_exposures_per_epoch
        != expected_real_exposures
    ):
        raise AssertionError(
            "unexpected real exposure count"
        )

    if (
        sampler.fake_exposures_per_epoch
        != expected_fake_exposures
    ):
        raise AssertionError(
            "unexpected fake exposure count"
        )

    if (
        sampler.image_exposures_per_epoch
        != expected_total_exposures
    ):
        raise AssertionError(
            "unexpected total image exposure count"
        )

    print()
    print("NOMINAL FF++ FIT CASE")
    print(
        "  frames/video:               {}".format(
            frames_per_video
        )
    )
    print(
        "  real source frames:         {}".format(
            real_frames
        )
    )
    print(
        "  fake source frames:         {}".format(
            fake_frames
        )
    )

    for method in (
        "DF",
        "F2F",
        "FS",
        "NT",
    ):
        print(
            "  {:<4} source frames:         {}".format(
                method,
                fake_frames_by_method[method],
            )
        )

    print(
        "  effective batch size:       {}".format(
            sampler.batch_size
        )
    )
    print(
        "  real/fake per batch:        {}/{}".format(
            sampler.half_batch_size,
            sampler.half_batch_size,
        )
    )
    print(
        "  optimizer steps/epoch:      {}".format(
            stats["batches"]
        )
    )
    print(
        "  real exposures/epoch:       {}".format(
            stats["real_exposures"]
        )
    )
    print(
        "  fake exposures/epoch:       {}".format(
            stats["fake_exposures"]
        )
    )
    print(
        "  total image exposures:      {}".format(
            sampler.image_exposures_per_epoch
        )
    )


def _validate_negative_cases() -> None:
    _expect_failure(
        "odd batch size",
        lambda: BalancedRealFakeBatchSampler(
            labels=[0, 0, 1, 1],
            batch_size=3,
        ),
    )

    _expect_failure(
        "non-binary label",
        lambda: BalancedRealFakeBatchSampler(
            labels=[0, 1, 2, 1],
            batch_size=2,
        ),
    )

    _expect_failure(
        "missing real class",
        lambda: BalancedRealFakeBatchSampler(
            labels=[1, 1, 1, 1],
            batch_size=2,
        ),
    )

    _expect_failure(
        "missing fake class",
        lambda: BalancedRealFakeBatchSampler(
            labels=[0, 0, 0, 0],
            batch_size=2,
        ),
    )

    _expect_failure(
        "fake population not divisible by half batch",
        lambda: BalancedRealFakeBatchSampler(
            labels=[
                0, 0, 0,
                1, 1, 1,
            ],
            batch_size=4,
        ),
    )

    sampler = BalancedRealFakeBatchSampler(
        labels=[0, 0, 1, 1],
        batch_size=2,
    )

    _expect_failure(
        "negative epoch",
        lambda: sampler.set_epoch(-1),
    )

    print()
    print("NEGATIVE / INVARIANT TESTS")
    print(
        "  invalid configurations:     rejected"
    )


def main() -> None:
    _validate_small_synthetic_case()
    _validate_nominal_ffpp_fit_case()
    _validate_negative_cases()

    print()
    print("SAMPLER VALIDATION PASSED")


if __name__ == "__main__":
    main()
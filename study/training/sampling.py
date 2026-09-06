"""
Study-controlled exposure sampling for Xception, UCF, and SPSL.

The frozen FIT materialization contains slightly fewer than the nominal
32 valid face crops for a small number of videos. Training therefore does
not replace failed temporal positions during materialization. Instead, the
training sampler preserves the predetermined epoch exposure budget by
deterministically reusing already-valid FIT inputs.

Study contract
--------------
Nominal FIT inventory per fake manipulation method:

    576 videos * 32 target frames = 18,432 exposures

Actual valid FIT inputs after clean-face materialization:

    FF-DF   18,398
    FF-F2F  18,423
    FF-FS   18,423
    FF-NT   18,407

The sampler exposes exactly 18,432 fake inputs from every method in every
epoch. Every valid fake input is exposed at least once. Only the shortfall
relative to the nominal budget is filled by deterministic reuse:

    FF-DF   +34
    FF-F2F   +9
    FF-FS    +9
    FF-NT   +25

This gives exactly 73,728 fake exposures per epoch.

For Xception and SPSL, every optimizer batch contains equal numbers of real
and fake inputs. Real inputs are sampled with replacement, yielding 73,728
real exposures per epoch.

For UCF, the same fake-method exposure plan is applied to indices of the
detector-specific pairDataset. Each pairDataset item then supplies one fake
and one real input according to the original UCF pair semantics.

All sampling decisions use sampler-local RNG state derived from the fixed
study seed and epoch. No failed materialization target is reconstructed,
temporally replaced, or silently padded here.
"""

import random
from collections import Counter
from typing import Dict, Iterator, List, Mapping, Sequence

from torch.utils.data import Sampler


FAKE_METHODS = (
    "FF-DF",
    "FF-F2F",
    "FF-FS",
    "FF-NT",
)

TARGET_FAKE_EXPOSURES_PER_METHOD = 18432


def _validate_non_negative_epoch(
    epoch: int,
) -> None:
    if not isinstance(epoch, int):
        raise TypeError(
            "epoch must be an integer"
        )

    if epoch < 0:
        raise ValueError(
            "epoch must be non-negative"
        )


def _validate_positive_integer(
    value,
    name: str,
) -> int:
    if not isinstance(value, int):
        raise TypeError(
            "{} must be an integer".format(
                name
            )
        )

    if value <= 0:
        raise ValueError(
            "{} must be positive".format(
                name
            )
        )

    return value


def _normalize_integer_labels(
    labels: Sequence[int],
    name: str,
) -> List[int]:
    if labels is None:
        raise ValueError(
            "{} must not be None".format(
                name
            )
        )

    normalized = []

    for index, label in enumerate(
        labels
    ):
        try:
            integer_label = int(
                label
            )
            numeric_label = float(
                label
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "{} at index {} is not numeric: {!r}".format(
                    name,
                    index,
                    label,
                )
            ) from exc

        if (
            numeric_label
            != float(integer_label)
        ):
            raise ValueError(
                "{} at index {} is not an integer value: {!r}".format(
                    name,
                    index,
                    label,
                )
            )

        normalized.append(
            integer_label
        )

    if not normalized:
        raise ValueError(
            "{} must contain at least one item".format(
                name
            )
        )

    return normalized


def _normalize_binary_labels(
    labels: Sequence[int],
) -> List[int]:
    normalized = (
        _normalize_integer_labels(
            labels=labels,
            name="labels",
        )
    )

    for index, label in enumerate(
        normalized
    ):
        if label not in (
            0,
            1,
        ):
            raise ValueError(
                "label at index {} must be binary 0/1, got {!r}".format(
                    index,
                    label,
                )
            )

    return normalized


def _derive_sampler_seed(
    seed: int,
    epoch: int,
    stream: int,
) -> int:
    """
    Derive an integer seed without depending on Python's randomized hash().
    """
    return (
        seed * 1_000_003
        + epoch * 9_176
        + stream * 104_729
        + 0x9E3779B1
    )


def _extend_group_to_target(
    source_indices: Sequence[int],
    target_count: int,
    rng: random.Random,
) -> List[int]:
    """
    Preserve every source item once, then add only the required reuse.

    When target_count exceeds one additional pass, deterministic shuffled
    passes are added until the target is reached. In the frozen study data
    the actual shortfalls are only 9--34 items.
    """
    source_indices = list(
        source_indices
    )

    if not source_indices:
        raise ValueError(
            "exposure group must contain at least one source item"
        )

    if len(source_indices) > target_count:
        raise ValueError(
            "source group contains {} items but target exposure is {}; "
            "the study sampler does not drop valid source inputs".format(
                len(source_indices),
                target_count,
            )
        )

    first_pass = list(
        source_indices
    )
    rng.shuffle(
        first_pass
    )

    exposures = list(
        first_pass
    )

    remaining = (
        target_count
        - len(exposures)
    )

    while remaining > 0:
        reuse_pass = list(
            source_indices
        )

        rng.shuffle(
            reuse_pass
        )

        take = min(
            remaining,
            len(reuse_pass),
        )

        exposures.extend(
            reuse_pass[
                :take
            ]
        )

        remaining -= take

    if len(exposures) != target_count:
        raise RuntimeError(
            "internal exposure-plan length mismatch"
        )

    return exposures


def _build_group_exposure_plan(
    index_groups: Mapping,
    target_by_group: Mapping,
    group_order: Sequence,
    seed: int,
    epoch: int,
) -> List[int]:
    """
    Build one deterministic epoch exposure sequence.

    Every group's valid source items are retained. A group is reused only
    when its valid source count is below its frozen target exposure count.
    """
    _validate_non_negative_epoch(
        epoch
    )

    plan = []

    for group_position, group in enumerate(
        group_order
    ):
        if group not in index_groups:
            raise ValueError(
                "missing source group {!r}".format(
                    group
                )
            )

        if group not in target_by_group:
            raise ValueError(
                "missing target for source group {!r}".format(
                    group
                )
            )

        target_count = (
            _validate_positive_integer(
                target_by_group[
                    group
                ],
                "target exposure for {!r}".format(
                    group
                ),
            )
        )

        group_rng = random.Random(
            _derive_sampler_seed(
                seed=seed,
                epoch=epoch,
                stream=(
                    group_position
                    + 1
                ),
            )
        )

        group_plan = (
            _extend_group_to_target(
                source_indices=index_groups[
                    group
                ],
                target_count=target_count,
                rng=group_rng,
            )
        )

        plan.extend(
            group_plan
        )

    expected_total = sum(
        target_by_group[
            group
        ]
        for group in group_order
    )

    if len(plan) != expected_total:
        raise RuntimeError(
            "combined exposure-plan length mismatch: "
            "expected {}, got {}".format(
                expected_total,
                len(plan),
            )
        )

    combined_rng = random.Random(
        _derive_sampler_seed(
            seed=seed,
            epoch=epoch,
            stream=10_000,
        )
    )

    combined_rng.shuffle(
        plan
    )

    return plan


class BalancedRealFakeBatchSampler(Sampler):
    """
    Study batch sampler for Xception and SPSL.

    Each batch contains:

        batch_size / 2 real inputs
        batch_size / 2 fake inputs

    Fake inputs follow the fixed per-method epoch exposure plan. Real inputs
    are sampled with replacement from the valid clean FIT real-input pool.

    Parameters
    ----------
    labels:
        Binary 0/1 labels aligned with the underlying dataset indices.
    source_labels:
        Original DeepfakeBench labels aligned with the same indices:
        FF-real / FF-DF / FF-F2F / FF-FS / FF-NT.
    batch_size:
        Effective image batch size. Must be even.
    target_fake_exposures_per_method:
        Fixed epoch exposure target for each fake method.
    seed:
        Base study seed.
    """

    def __init__(
        self,
        labels: Sequence[int],
        source_labels: Sequence[str],
        batch_size: int = 32,
        target_fake_exposures_per_method: int = (
            TARGET_FAKE_EXPOSURES_PER_METHOD
        ),
        seed: int = 1024,
    ) -> None:
        self.labels = (
            _normalize_binary_labels(
                labels
            )
        )

        if source_labels is None:
            raise ValueError(
                "source_labels must not be None"
            )

        self.source_labels = [
            str(label)
            for label in source_labels
        ]

        if (
            len(self.labels)
            != len(self.source_labels)
        ):
            raise ValueError(
                "labels and source_labels must have equal length"
            )

        self.batch_size = (
            _validate_positive_integer(
                batch_size,
                "batch_size",
            )
        )

        if self.batch_size % 2 != 0:
            raise ValueError(
                "batch_size must be even so every batch can "
                "contain equal numbers of real and fake inputs"
            )

        self.half_batch_size = (
            self.batch_size
            // 2
        )

        self.target_fake_exposures_per_method = (
            _validate_positive_integer(
                target_fake_exposures_per_method,
                "target_fake_exposures_per_method",
            )
        )

        if not isinstance(
            seed,
            int,
        ):
            raise TypeError(
                "seed must be an integer"
            )

        self.seed = seed
        self.epoch = 0

        self.real_indices = []
        self.fake_indices_by_method = {
            method: []
            for method in FAKE_METHODS
        }

        for index, (
            binary_label,
            source_label,
        ) in enumerate(
            zip(
                self.labels,
                self.source_labels,
            )
        ):
            if binary_label == 0:
                if source_label != "FF-real":
                    raise ValueError(
                        "binary real item at index {} has source label "
                        "{!r}, expected 'FF-real'".format(
                            index,
                            source_label,
                        )
                    )

                self.real_indices.append(
                    index
                )

                continue

            if source_label not in FAKE_METHODS:
                raise ValueError(
                    "binary fake item at index {} has unsupported "
                    "source label {!r}".format(
                        index,
                        source_label,
                    )
                )

            self.fake_indices_by_method[
                source_label
            ].append(
                index
            )

        if not self.real_indices:
            raise ValueError(
                "at least one real source item is required"
            )

        for method in FAKE_METHODS:
            source_count = len(
                self.fake_indices_by_method[
                    method
                ]
            )

            if source_count == 0:
                raise ValueError(
                    "fake source group {} is empty".format(
                        method
                    )
                )

            if (
                source_count
                > self.target_fake_exposures_per_method
            ):
                raise ValueError(
                    "{} contains {} valid source items but the "
                    "frozen target is {}; dropping valid inputs is "
                    "not permitted".format(
                        method,
                        source_count,
                        self.target_fake_exposures_per_method,
                    )
                )

        self.fake_target_by_method = {
            method: (
                self.target_fake_exposures_per_method
            )
            for method in FAKE_METHODS
        }

        self.fake_exposure_count = sum(
            self.fake_target_by_method.values()
        )

        if (
            self.fake_exposure_count
            % self.half_batch_size
            != 0
        ):
            raise ValueError(
                "total fake exposure count {} is not divisible by "
                "fake items per batch {}".format(
                    self.fake_exposure_count,
                    self.half_batch_size,
                )
            )

        self.num_batches = (
            self.fake_exposure_count
            // self.half_batch_size
        )

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        _validate_non_negative_epoch(
            epoch
        )

        self.epoch = epoch

    def _fake_epoch_plan(
        self,
    ) -> List[int]:
        return _build_group_exposure_plan(
            index_groups=(
                self.fake_indices_by_method
            ),
            target_by_group=(
                self.fake_target_by_method
            ),
            group_order=FAKE_METHODS,
            seed=self.seed,
            epoch=self.epoch,
        )

    def __iter__(
        self,
    ) -> Iterator[List[int]]:
        fake_plan = (
            self._fake_epoch_plan()
        )

        real_rng = random.Random(
            _derive_sampler_seed(
                seed=self.seed,
                epoch=self.epoch,
                stream=20_000,
            )
        )

        batch_rng = random.Random(
            _derive_sampler_seed(
                seed=self.seed,
                epoch=self.epoch,
                stream=30_000,
            )
        )

        for start in range(
            0,
            len(fake_plan),
            self.half_batch_size,
        ):
            fake_batch = fake_plan[
                start:
                start
                + self.half_batch_size
            ]

            if (
                len(fake_batch)
                != self.half_batch_size
            ):
                raise RuntimeError(
                    "incomplete fake batch produced by study sampler"
                )

            real_batch = [
                real_rng.choice(
                    self.real_indices
                )
                for _ in range(
                    self.half_batch_size
                )
            ]

            batch = (
                real_batch
                + fake_batch
            )

            batch_rng.shuffle(
                batch
            )

            yield batch

    def __len__(
        self,
    ) -> int:
        return self.num_batches

    @property
    def real_source_count(
        self,
    ) -> int:
        return len(
            self.real_indices
        )

    @property
    def fake_source_count(
        self,
    ) -> int:
        return sum(
            len(indices)
            for indices
            in self.fake_indices_by_method.values()
        )

    @property
    def fake_exposures_per_epoch(
        self,
    ) -> int:
        return self.fake_exposure_count

    @property
    def real_exposures_per_epoch(
        self,
    ) -> int:
        return self.fake_exposure_count

    @property
    def image_exposures_per_epoch(
        self,
    ) -> int:
        return (
            self.real_exposures_per_epoch
            + self.fake_exposures_per_epoch
        )

    def contract_summary(
        self,
    ) -> Dict[str, object]:
        source_counts = {
            method: len(
                self.fake_indices_by_method[
                    method
                ]
            )
            for method in FAKE_METHODS
        }

        repeat_counts = {
            method: (
                self.target_fake_exposures_per_method
                - source_counts[
                    method
                ]
            )
            for method in FAKE_METHODS
        }

        return {
            "batch_size": self.batch_size,
            "real_per_batch": self.half_batch_size,
            "fake_per_batch": self.half_batch_size,
            "real_source_items": self.real_source_count,
            "fake_source_items": self.fake_source_count,
            "fake_method_source_counts": source_counts,
            "fake_method_target_exposures": {
                method: (
                    self.target_fake_exposures_per_method
                )
                for method in FAKE_METHODS
            },
            "fake_method_repeat_counts": repeat_counts,
            "real_exposures_per_epoch": (
                self.real_exposures_per_epoch
            ),
            "fake_exposures_per_epoch": (
                self.fake_exposures_per_epoch
            ),
            "image_exposures_per_epoch": (
                self.image_exposures_per_epoch
            ),
            "optimizer_steps_per_epoch": len(
                self
            ),
            "real_sampling": "with_replacement",
            "seed": self.seed,
            "epoch": self.epoch,
        }


class FixedMethodExposureSampler(Sampler):
    """
    Individual-index sampler used for UCF pairDataset.

    ``pairDataset`` is indexed by fake examples. This sampler preserves all
    valid fake examples and deterministically tops up each manipulation
    method to its frozen epoch target.

    The dataset remains responsible for choosing the corresponding real
    member of each pair.
    """

    def __init__(
        self,
        method_labels: Sequence[int],
        method_order: Sequence[int],
        method_targets: Mapping[int, int],
        method_names: Mapping[int, str],
        seed: int = 1024,
    ) -> None:
        self.method_labels = (
            _normalize_integer_labels(
                labels=method_labels,
                name="method_labels",
            )
        )

        if not method_order:
            raise ValueError(
                "method_order must not be empty"
            )

        self.method_order = [
            int(label)
            for label in method_order
        ]

        if (
            len(self.method_order)
            != len(
                set(
                    self.method_order
                )
            )
        ):
            raise ValueError(
                "method_order must contain unique labels"
            )

        self.method_targets = {
            int(label): (
                _validate_positive_integer(
                    int(target),
                    "method target for {}".format(
                        label
                    ),
                )
            )
            for label, target
            in method_targets.items()
        }

        self.method_names = {
            int(label): str(name)
            for label, name
            in method_names.items()
        }

        expected = set(
            self.method_order
        )

        if set(
            self.method_targets
        ) != expected:
            raise ValueError(
                "method_targets keys must exactly match method_order"
            )

        if set(
            self.method_names
        ) != expected:
            raise ValueError(
                "method_names keys must exactly match method_order"
            )

        actual_labels = set(
            self.method_labels
        )

        if actual_labels != expected:
            raise ValueError(
                "pairDataset fake method labels {} do not match "
                "expected labels {}".format(
                    sorted(
                        actual_labels
                    ),
                    sorted(
                        expected
                    ),
                )
            )

        if not isinstance(
            seed,
            int,
        ):
            raise TypeError(
                "seed must be an integer"
            )

        self.seed = seed
        self.epoch = 0

        self.indices_by_method = {
            label: []
            for label
            in self.method_order
        }

        for index, method_label in enumerate(
            self.method_labels
        ):
            self.indices_by_method[
                method_label
            ].append(
                index
            )

        for method_label in self.method_order:
            source_count = len(
                self.indices_by_method[
                    method_label
                ]
            )

            target = self.method_targets[
                method_label
            ]

            if source_count == 0:
                raise ValueError(
                    "UCF fake method {} is empty".format(
                        self.method_names[
                            method_label
                        ]
                    )
                )

            if source_count > target:
                raise ValueError(
                    "UCF fake method {} contains {} valid inputs but "
                    "target exposure is {}; dropping valid inputs is "
                    "not permitted".format(
                        self.method_names[
                            method_label
                        ],
                        source_count,
                        target,
                    )
                )

        self.total_exposures = sum(
            self.method_targets.values()
        )

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        _validate_non_negative_epoch(
            epoch
        )

        self.epoch = epoch

    def __iter__(
        self,
    ) -> Iterator[int]:
        plan = (
            _build_group_exposure_plan(
                index_groups=(
                    self.indices_by_method
                ),
                target_by_group=(
                    self.method_targets
                ),
                group_order=(
                    self.method_order
                ),
                seed=self.seed,
                epoch=self.epoch,
            )
        )

        return iter(
            plan
        )

    def __len__(
        self,
    ) -> int:
        return self.total_exposures

    def contract_summary(
        self,
    ) -> Dict[str, object]:
        source_counts = {}
        targets = {}
        repeats = {}

        for method_label in self.method_order:
            method_name = (
                self.method_names[
                    method_label
                ]
            )

            source_count = len(
                self.indices_by_method[
                    method_label
                ]
            )

            target = self.method_targets[
                method_label
            ]

            source_counts[
                method_name
            ] = source_count

            targets[
                method_name
            ] = target

            repeats[
                method_name
            ] = (
                target
                - source_count
            )

        return {
            "fake_source_items": len(
                self.method_labels
            ),
            "fake_method_source_counts": (
                source_counts
            ),
            "fake_method_target_exposures": (
                targets
            ),
            "fake_method_repeat_counts": (
                repeats
            ),
            "pair_exposures_per_epoch": (
                self.total_exposures
            ),
            "seed": self.seed,
            "epoch": self.epoch,
        }
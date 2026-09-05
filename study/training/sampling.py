"""
Study-controlled balanced batch sampling for detector training.

The sampler is intentionally independent of DeepfakeBench dataset internals.
It operates only on binary dataset labels and yields batches of dataset
indices.

Study contract
--------------
For Xception and SPSL training:

* one epoch is defined by one complete pass over all fake frame items;
* every fake frame is exposed exactly once per epoch;
* an equal number of real frame exposures is sampled with replacement;
* every batch contains 50% real and 50% fake items;
* the effective image batch size is therefore identical to the configured
  batch size;
* fake-method composition is preserved at the epoch level because no fake
  frame is dropped, duplicated, or reweighted;
* ordering and real-frame sampling are reproducible for a fixed
  ``seed`` and ``epoch``.

UCF does not use this sampler. Its detector-specific ``pairDataset`` already
constructs one real sample for every fake sample. This sampler provides the
corresponding study-level exposure policy for ordinary frame datasets such
as Xception and SPSL.
"""

import random
from typing import Dict, Iterator, List, Sequence

from torch.utils.data import Sampler


class BalancedRealFakeBatchSampler(Sampler):
    """
    Yield balanced batches containing equal numbers of real and fake items.

    Labels follow the study convention:

    * 0 -> real
    * 1 -> fake

    The fake population defines the epoch length. Fake indices are shuffled
    and used exactly once. Real indices are sampled with replacement.

    Parameters
    ----------
    labels:
        Binary label for every item in the underlying dataset.
    batch_size:
        Effective number of images in one optimizer step. Must be even.
    seed:
        Base pseudorandom seed.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int = 32,
        seed: int = 1024,
    ) -> None:
        self.labels = self._normalize_labels(labels)

        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size % 2 != 0:
            raise ValueError(
                "batch_size must be even so that every batch can contain "
                "equal numbers of real and fake items"
            )

        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.seed = seed
        self.epoch = 0

        self.real_indices = [
            index
            for index, label in enumerate(self.labels)
            if label == 0
        ]

        self.fake_indices = [
            index
            for index, label in enumerate(self.labels)
            if label == 1
        ]

        if not self.real_indices:
            raise ValueError(
                "at least one real item is required"
            )

        if not self.fake_indices:
            raise ValueError(
                "at least one fake item is required"
            )

        if len(self.fake_indices) % self.half_batch_size != 0:
            raise ValueError(
                "the number of fake items must be divisible by half of "
                "batch_size; the strict study sampler does not silently "
                "drop or pad fake items"
            )

        self.num_batches = (
            len(self.fake_indices)
            // self.half_batch_size
        )

    @staticmethod
    def _normalize_labels(
        labels: Sequence[int],
    ) -> List[int]:
        if labels is None:
            raise ValueError("labels must not be None")

        normalized = []

        for index, label in enumerate(labels):
            try:
                integer_label = int(label)
                numeric_label = float(label)
            except (TypeError, ValueError):
                raise ValueError(
                    "label at index {} is not numeric: {!r}".format(
                        index,
                        label,
                    )
                )

            if (
                integer_label not in (0, 1)
                or numeric_label != float(integer_label)
            ):
                raise ValueError(
                    "label at index {} must be binary 0/1, got {!r}".format(
                        index,
                        label,
                    )
                )

            normalized.append(integer_label)

        if not normalized:
            raise ValueError(
                "labels must contain at least one item"
            )

        return normalized

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Select the deterministic random stream used for an epoch.

        This mirrors the usual ``set_epoch`` concept used by distributed
        samplers while remaining independent of global Python RNG state.
        """
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")

        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        self.epoch = epoch

    def _epoch_seed(self) -> int:
        """
        Derive a stable per-epoch seed from the study seed.

        A sampler-local RNG is used so that dataset splitting, training
        sampling, and unrelated Python random operations cannot consume
        one another's RNG state.
        """
        return (
            self.seed * 1_000_003
            + self.epoch * 9_176
            + 0x9E3779B1
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(
            self._epoch_seed()
        )

        fake_indices = list(
            self.fake_indices
        )
        rng.shuffle(fake_indices)

        for start in range(
            0,
            len(fake_indices),
            self.half_batch_size,
        ):
            fake_batch = fake_indices[
                start:start + self.half_batch_size
            ]

            real_batch = [
                rng.choice(self.real_indices)
                for _ in range(
                    self.half_batch_size
                )
            ]

            batch = (
                real_batch
                + fake_batch
            )

            # Xception and SPSL have no requirement that real and fake
            # samples occupy fixed positions within the batch.
            rng.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        return self.num_batches

    @property
    def real_source_count(self) -> int:
        return len(
            self.real_indices
        )

    @property
    def fake_source_count(self) -> int:
        return len(
            self.fake_indices
        )

    @property
    def fake_exposures_per_epoch(self) -> int:
        return len(
            self.fake_indices
        )

    @property
    def real_exposures_per_epoch(self) -> int:
        return len(
            self.fake_indices
        )

    @property
    def image_exposures_per_epoch(self) -> int:
        return (
            self.real_exposures_per_epoch
            + self.fake_exposures_per_epoch
        )

    def contract_summary(self) -> Dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "real_per_batch": self.half_batch_size,
            "fake_per_batch": self.half_batch_size,
            "real_source_items": self.real_source_count,
            "fake_source_items": self.fake_source_count,
            "real_exposures_per_epoch": self.real_exposures_per_epoch,
            "fake_exposures_per_epoch": self.fake_exposures_per_epoch,
            "image_exposures_per_epoch": self.image_exposures_per_epoch,
            "optimizer_steps_per_epoch": len(self),
            "seed": self.seed,
            "epoch": self.epoch,
        }
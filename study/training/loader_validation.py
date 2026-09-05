"""
Executable validation of the study-controlled DataLoader integration.

Run:

    python -m study.training.loader_validation
"""

from collections import Counter

import torch
from torch.utils.data import Dataset

from .loader import (
    build_study_training_loader,
    effective_image_batch_size,
    loader_contract_summary,
    set_study_training_epoch,
)


class SyntheticFrameDataset(Dataset):
    def __init__(
        self,
    ) -> None:
        self.label_list = (
            [0] * 8
            + [1] * 16
        )

    def __len__(self) -> int:
        return len(
            self.label_list
        )

    def __getitem__(
        self,
        index: int,
    ):
        return {
            "index": index,
            "label": self.label_list[
                index
            ],
        }


def _collect_frame_loader(
    model_name: str,
    epoch: int,
):
    dataset = (
        SyntheticFrameDataset()
    )

    config = {
        "study_controlled_training": True,
        "model_name": model_name,
        "train_batchSize": 8,
        "workers": 0,
        "manualSeed": 1024,
        "ddp": False,
    }

    loader = (
        build_study_training_loader(
            dataset,
            config,
        )
    )

    set_study_training_epoch(
        loader,
        epoch,
    )

    batches = []

    for batch in loader:
        batches.append(
            {
                "index": [
                    int(value)
                    for value
                    in batch[
                        "index"
                    ].tolist()
                ],
                "label": [
                    int(value)
                    for value
                    in batch[
                        "label"
                    ].tolist()
                ],
            }
        )

    return (
        loader,
        config,
        batches,
    )


def _validate_frame_model(
    model_name: str,
) -> None:
    (
        loader_a,
        config,
        batches_a,
    ) = _collect_frame_loader(
        model_name,
        epoch=0,
    )

    (
        _,
        _,
        batches_b,
    ) = _collect_frame_loader(
        model_name,
        epoch=0,
    )

    if batches_a != batches_b:
        raise AssertionError(
            "{} loader is not reproducible "
            "for the same seed and epoch".format(
                model_name
            )
        )

    (
        _,
        _,
        batches_epoch_1,
    ) = _collect_frame_loader(
        model_name,
        epoch=1,
    )

    if batches_a == batches_epoch_1:
        raise AssertionError(
            "{} loader did not change "
            "between epochs".format(
                model_name
            )
        )

    fake_indices = []

    for batch_number, batch in enumerate(
        batches_a
    ):
        label_counts = Counter(
            batch["label"]
        )

        if label_counts != {
            0: 4,
            1: 4,
        }:
            raise AssertionError(
                "{} batch {} is not balanced: {}".format(
                    model_name,
                    batch_number,
                    dict(
                        label_counts
                    ),
                )
            )

        fake_indices.extend(
            index
            for index, label
            in zip(
                batch["index"],
                batch["label"],
            )
            if label == 1
        )

    expected_fake_indices = list(
        range(
            8,
            24,
        )
    )

    if sorted(
        fake_indices
    ) != expected_fake_indices:
        raise AssertionError(
            "{} did not expose every fake "
            "item exactly once".format(
                model_name
            )
        )

    if len(
        fake_indices
    ) != len(
        set(
            fake_indices
        )
    ):
        raise AssertionError(
            "{} duplicated fake items".format(
                model_name
            )
        )

    summary = (
        loader_contract_summary(
            loader_a,
            config,
        )
    )

    if (
        summary[
            "effective_image_batch_size"
        ]
        != 8
    ):
        raise AssertionError(
            "unexpected effective batch size"
        )

    if (
        summary[
            "optimizer_steps_per_epoch"
        ]
        != 4
    ):
        raise AssertionError(
            "unexpected optimizer-step count"
        )

    print(
        "{} FRAME LOADER".format(
            model_name.upper()
        )
    )
    print(
        "  balanced batch:             4 real + 4 fake"
    )
    print(
        "  fake exposure:              exactly once"
    )
    print(
        "  same seed + epoch:          reproducible"
    )
    print(
        "  next epoch:                 different ordering"
    )
    print(
        "  effective images/step:      {}".format(
            summary[
                "effective_image_batch_size"
            ]
        )
    )
    print(
        "  optimizer steps/epoch:      {}".format(
            summary[
                "optimizer_steps_per_epoch"
            ]
        )
    )


def _validate_ucf_batch_semantics() -> None:
    config = {
        "study_controlled_training": True,
        "model_name": "ucf",
        "train_batchSize": 16,
        "workers": 0,
        "manualSeed": 1024,
        "ddp": False,
    }

    effective_batch = (
        effective_image_batch_size(
            config
        )
    )

    if effective_batch != 32:
        raise AssertionError(
            "UCF effective batch must be "
            "32 images for 16 pair-items"
        )

    print()
    print("UCF BATCH SEMANTICS")
    print(
        "  loader pair-items/step:     16"
    )
    print(
        "  effective images/step:      {}".format(
            effective_batch
        )
    )
    print(
        "  pairDataset replacement:    not performed"
    )


def _validate_negative_cases() -> None:
    dataset = (
        SyntheticFrameDataset()
    )

    bad_config = {
        "study_controlled_training": False,
        "model_name": "xception",
        "train_batchSize": 8,
        "workers": 0,
        "manualSeed": 1024,
        "ddp": False,
    }

    try:
        build_study_training_loader(
            dataset,
            bad_config,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "disabled study mode was not rejected"
        )

    ddp_config = dict(
        bad_config
    )
    ddp_config[
        "study_controlled_training"
    ] = True
    ddp_config[
        "ddp"
    ] = True

    try:
        build_study_training_loader(
            dataset,
            ddp_config,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "DDP study mode was not rejected"
        )

    print()
    print("NEGATIVE / INVARIANT TESTS")
    print(
        "  implicit activation:        rejected"
    )
    print(
        "  unsupported DDP mode:       rejected"
    )


def main() -> None:
    _validate_frame_model(
        "xception"
    )

    print()

    _validate_frame_model(
        "spsl"
    )

    _validate_ucf_batch_semantics()
    _validate_negative_cases()

    print()
    print(
        "LOADER INTEGRATION VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
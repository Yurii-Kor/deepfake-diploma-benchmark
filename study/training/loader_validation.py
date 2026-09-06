"""
Executable validation of the study-controlled DataLoader integration.

Run from the repository root:

    python -m study.training.loader_validation
"""

from collections import Counter

from torch.utils.data import Dataset

from .loader import (
    build_study_testing_loader,
    build_study_training_loader,
    effective_image_batch_size,
    loader_contract_summary,
    set_study_training_epoch,
)
from .sampling import (
    FAKE_METHODS,
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

UCF_LABEL_DICT = {
    "FF-real": 0,
    "FF-DF": 1,
    "FF-F2F": 2,
    "FF-FS": 3,
    "FF-NT": 4,
}


class SyntheticFrameDataset(
    Dataset
):
    """
    Metadata-only synthetic equivalent of the frozen aggregate FIT dataset.

    No materialized PNG is read by this validation.
    """

    def __init__(
        self,
    ) -> None:
        self.label_list = []
        self.source_label_list = []

        for source_label in (
            "FF-real",
            "FF-DF",
            "FF-F2F",
            "FF-FS",
            "FF-NT",
        ):
            count = (
                FROZEN_FIT_SOURCE_COUNTS[
                    source_label
                ]
            )

            binary_label = (
                0
                if source_label == "FF-real"
                else 1
            )

            self.label_list.extend(
                [
                    binary_label
                ]
                * count
            )

            self.source_label_list.extend(
                [
                    source_label
                ]
                * count
            )

    def __len__(
        self,
    ) -> int:
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


class SyntheticUcfPairDataset(
    Dataset
):
    """
    Metadata-only equivalent of the frozen UCF pairDataset.

    fake_imglist retains the tuple shape required by loader.py:

        (image_placeholder, specific_label, binary_label)

    Real partner selection is detector-specific pairDataset behavior and is
    tested separately by the real UCF no-augmentation smoke.
    """

    def __init__(
        self,
    ) -> None:
        self.fake_imglist = []
        self.real_imglist = []

        for index in range(
            FROZEN_FIT_SOURCE_COUNTS[
                "FF-real"
            ]
        ):
            self.real_imglist.append(
                (
                    index,
                    0,
                    0,
                )
            )

        fake_counter = 0

        for method in FAKE_METHODS:
            specific_label = (
                UCF_LABEL_DICT[
                    method
                ]
            )

            for _ in range(
                FROZEN_FIT_SOURCE_COUNTS[
                    method
                ]
            ):
                self.fake_imglist.append(
                    (
                        fake_counter,
                        specific_label,
                        1,
                    )
                )

                fake_counter += 1

    def __len__(
        self,
    ) -> int:
        return len(
            self.fake_imglist
        )

    def __getitem__(
        self,
        index: int,
    ):
        return {
            "pair_index": index,
            "specific_label": (
                self.fake_imglist[
                    index
                ][
                    1
                ]
            ),
        }


class TinyEvaluationDataset(
    Dataset
):
    def __init__(
        self,
        size: int = 33,
    ) -> None:
        self.size = size

    def __len__(
        self,
    ) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ):
        return index


def _base_config(
    model_name: str,
):
    config = {
        "study_controlled_training": True,
        "model_name": model_name,
        "workers": 0,
        "manualSeed": 1024,
        "ddp": False,
        "test_batchSize": 32,
    }

    if model_name == "ucf":
        config[
            "train_batchSize"
        ] = 16

        config[
            "label_dict"
        ] = dict(
            UCF_LABEL_DICT
        )

    else:
        config[
            "train_batchSize"
        ] = 32

    return config


def _frame_plan(
    model_name: str,
    epoch: int,
):
    dataset = (
        SyntheticFrameDataset()
    )

    config = (
        _base_config(
            model_name
        )
    )

    loader = (
        build_study_training_loader(
            train_set=dataset,
            config=config,
        )
    )

    set_study_training_epoch(
        loader,
        epoch,
    )

    batch_sampler = getattr(
        loader,
        "batch_sampler",
        None,
    )

    if not isinstance(
        batch_sampler,
        BalancedRealFakeBatchSampler,
    ):
        raise AssertionError(
            "{} did not receive the balanced study sampler".format(
                model_name
            )
        )

    plan = list(
        batch_sampler
    )

    return (
        dataset,
        loader,
        config,
        plan,
    )


def _validate_frame_model(
    model_name: str,
) -> None:
    (
        dataset_a,
        loader_a,
        config,
        plan_a,
    ) = _frame_plan(
        model_name=model_name,
        epoch=0,
    )

    (
        _,
        _,
        _,
        plan_b,
    ) = _frame_plan(
        model_name=model_name,
        epoch=0,
    )

    if plan_a != plan_b:
        raise AssertionError(
            "{} loader is not reproducible for "
            "the same seed and epoch".format(
                model_name
            )
        )

    (
        _,
        _,
        _,
        plan_epoch_1,
    ) = _frame_plan(
        model_name=model_name,
        epoch=1,
    )

    if plan_a == plan_epoch_1:
        raise AssertionError(
            "{} loader did not change between epochs".format(
                model_name
            )
        )

    if len(
        loader_a
    ) != 4608:
        raise AssertionError(
            "{} must expose 4608 optimizer steps".format(
                model_name
            )
        )

    fake_method_exposures = Counter()

    fake_source_indices = {
        index
        for index, label
        in enumerate(
            dataset_a.label_list
        )
        if label == 1
    }

    fake_seen = []

    for batch_number, batch in enumerate(
        plan_a
    ):
        if len(
            batch
        ) != 32:
            raise AssertionError(
                "{} batch {} has incorrect size".format(
                    model_name,
                    batch_number,
                )
            )

        label_counts = Counter(
            dataset_a.label_list[
                index
            ]
            for index in batch
        )

        if label_counts != {
            0: 16,
            1: 16,
        }:
            raise AssertionError(
                "{} batch {} is not 16/16 balanced: {}".format(
                    model_name,
                    batch_number,
                    dict(
                        label_counts
                    ),
                )
            )

        for index in batch:
            if (
                dataset_a.label_list[
                    index
                ]
                != 1
            ):
                continue

            fake_seen.append(
                index
            )

            fake_method_exposures[
                dataset_a.source_label_list[
                    index
                ]
            ] += 1

    if set(
        fake_seen
    ) != fake_source_indices:
        raise AssertionError(
            "{} did not expose every valid fake source input".format(
                model_name
            )
        )

    expected_method_exposures = {
        method: 18432
        for method in FAKE_METHODS
    }

    if dict(
        fake_method_exposures
    ) != expected_method_exposures:
        raise AssertionError(
            "{} fake method exposures are incorrect: {}".format(
                model_name,
                dict(
                    fake_method_exposures
                ),
            )
        )

    summary = (
        loader_contract_summary(
            train_data_loader=loader_a,
            config=config,
        )
    )

    if (
        summary[
            "effective_image_batch_size"
        ]
        != 32
    ):
        raise AssertionError(
            "unexpected effective image batch size"
        )

    if (
        summary[
            "optimizer_steps_per_epoch"
        ]
        != 4608
    ):
        raise AssertionError(
            "unexpected optimizer-step count"
        )

    if (
        summary[
            "image_exposures_per_epoch"
        ]
        != 147456
    ):
        raise AssertionError(
            "unexpected image-exposure count"
        )

    if (
        summary[
            "fake_method_repeat_counts"
        ]
        != FROZEN_REPEAT_COUNTS
    ):
        raise AssertionError(
            "unexpected fake top-up counts"
        )

    print(
        "{} FRAME LOADER".format(
            model_name.upper()
        )
    )

    print(
        "  batches:                    4608"
    )

    print(
        "  effective images/step:      32"
    )

    print(
        "  real/fake per batch:        16/16"
    )

    print(
        "  fake exposures/method:      18432"
    )

    print(
        "  top-up counts:              {}".format(
            FROZEN_REPEAT_COUNTS
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


def _ucf_plan(
    epoch: int,
):
    dataset = (
        SyntheticUcfPairDataset()
    )

    config = (
        _base_config(
            "ucf"
        )
    )

    loader = (
        build_study_training_loader(
            train_set=dataset,
            config=config,
        )
    )

    set_study_training_epoch(
        loader,
        epoch,
    )

    sampler = getattr(
        loader,
        "sampler",
        None,
    )

    if not isinstance(
        sampler,
        FixedMethodExposureSampler,
    ):
        raise AssertionError(
            "UCF did not receive FixedMethodExposureSampler"
        )

    return (
        dataset,
        loader,
        config,
        list(
            sampler
        ),
    )


def _validate_ucf_loader() -> None:
    (
        dataset,
        loader,
        config,
        plan_a,
    ) = _ucf_plan(
        epoch=0
    )

    (
        _,
        _,
        _,
        plan_b,
    ) = _ucf_plan(
        epoch=0
    )

    if plan_a != plan_b:
        raise AssertionError(
            "UCF sampler is not reproducible for same seed and epoch"
        )

    (
        _,
        _,
        _,
        plan_epoch_1,
    ) = _ucf_plan(
        epoch=1
    )

    if plan_a == plan_epoch_1:
        raise AssertionError(
            "UCF sampler did not change between epochs"
        )

    if len(
        loader
    ) != 4608:
        raise AssertionError(
            "UCF must expose 4608 optimizer steps"
        )

    if len(
        plan_a
    ) != 73728:
        raise AssertionError(
            "UCF must expose 73728 pair-items per epoch"
        )

    if set(
        plan_a
    ) != set(
        range(
            len(
                dataset.fake_imglist
            )
        )
    ):
        raise AssertionError(
            "UCF did not preserve every valid fake pair-item"
        )

    exposure_by_specific_label = Counter(
        dataset.fake_imglist[
            index
        ][
            1
        ]
        for index in plan_a
    )

    expected = {
        UCF_LABEL_DICT[
            method
        ]: 18432
        for method in FAKE_METHODS
    }

    if dict(
        exposure_by_specific_label
    ) != expected:
        raise AssertionError(
            "UCF specific-label exposures are incorrect"
        )

    summary = (
        loader_contract_summary(
            train_data_loader=loader,
            config=config,
        )
    )

    if (
        summary[
            "effective_image_batch_size"
        ]
        != 32
    ):
        raise AssertionError(
            "UCF effective batch must be 32 images"
        )

    if (
        summary[
            "image_exposures_per_epoch"
        ]
        != 147456
    ):
        raise AssertionError(
            "UCF image-exposure budget is incorrect"
        )

    if (
        summary[
            "fake_method_repeat_counts"
        ]
        != FROZEN_REPEAT_COUNTS
    ):
        raise AssertionError(
            "UCF top-up counts are incorrect"
        )

    print()
    print(
        "UCF PAIR LOADER"
    )

    print(
        "  pair-items/step:            16"
    )

    print(
        "  effective images/step:      32"
    )

    print(
        "  pair-items/epoch:           73728"
    )

    print(
        "  optimizer steps/epoch:      4608"
    )

    print(
        "  fake exposures/method:      18432"
    )

    print(
        "  top-up counts:              {}".format(
            FROZEN_REPEAT_COUNTS
        )
    )

    print(
        "  every valid fake:           exposed >= 1 time"
    )

    print(
        "  pairDataset real sampling:  preserved"
    )

    print(
        "  same seed + epoch:          reproducible"
    )


def _validate_evaluation_drop_policy() -> None:
    dataset = (
        TinyEvaluationDataset(
            size=33
        )
    )

    config = (
        _base_config(
            "xception"
        )
    )

    loader = (
        build_study_testing_loader(
            test_set=dataset,
            config=config,
            test_name="DeepFakeDetection",
        )
    )

    if loader.drop_last:
        raise AssertionError(
            "study evaluation must never drop the final technical batch"
        )

    observed = 0

    for batch in loader:
        observed += len(
            batch
        )

    if observed != 33:
        raise AssertionError(
            "evaluation loader dropped valid study inputs"
        )

    if len(
        loader
    ) != 2:
        raise AssertionError(
            "33 samples with test batch 32 must produce two batches"
        )

    print()
    print(
        "EVALUATION BATCH POLICY"
    )

    print(
        "  technical batch size:       32"
    )

    print(
        "  synthetic valid inputs:     33"
    )

    print(
        "  loader batches:             2"
    )

    print(
        "  drop_last:                  False"
    )

    print(
        "  retained valid inputs:      33/33"
    )


def _validate_negative_cases() -> None:
    dataset = (
        SyntheticFrameDataset()
    )

    bad_config = (
        _base_config(
            "xception"
        )
    )

    bad_config[
        "study_controlled_training"
    ] = False

    try:
        build_study_training_loader(
            train_set=dataset,
            config=bad_config,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "disabled study mode was not rejected"
        )

    ddp_config = (
        _base_config(
            "xception"
        )
    )

    ddp_config[
        "ddp"
    ] = True

    try:
        build_study_training_loader(
            train_set=dataset,
            config=ddp_config,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "DDP study mode was not rejected"
        )

    broken_dataset = (
        SyntheticFrameDataset()
    )

    broken_dataset.source_label_list[
        0
    ] = "FF-DF"

    try:
        build_study_training_loader(
            train_set=broken_dataset,
            config=_base_config(
                "xception"
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "frozen FIT source-count mismatch was not rejected"
        )

    print()
    print(
        "NEGATIVE / INVARIANT TESTS"
    )

    print(
        "  implicit activation:        rejected"
    )

    print(
        "  unsupported DDP mode:       rejected"
    )

    print(
        "  frozen FIT mismatch:        rejected"
    )


def main() -> None:
    _validate_frame_model(
        "xception"
    )

    print()

    _validate_frame_model(
        "spsl"
    )

    _validate_ucf_loader()
    _validate_evaluation_drop_policy()
    _validate_negative_cases()

    print()
    print(
        "LOADER INTEGRATION VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
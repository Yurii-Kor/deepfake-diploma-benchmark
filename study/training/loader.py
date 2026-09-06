"""
Study-controlled DeepfakeBench DataLoader construction.

Training
--------
Xception / SPSL:
    * ordinary frame dataset;
    * fixed fake-method exposure targets;
    * 50/50 real/fake batches;
    * real sampling with replacement.

UCF:
    * detector-specific pairDataset preserved;
    * one fake + one real per pair item preserved;
    * fake pair indices follow the same fixed method exposure targets used
      for Xception/SPSL.

All three detectors therefore execute:

    4 * 18,432 = 73,728 fake exposures / epoch
    73,728 real exposures / epoch
    147,456 effective image exposures / epoch
    4,608 optimizer steps / epoch

Evaluation
----------
Evaluation loaders never drop an incomplete final technical GPU batch.
"""

from collections import Counter
from typing import Dict

from torch.utils.data import DataLoader

from .reproducibility import (
    make_data_loader_generator,
    seed_data_loader_worker,
    validate_seed,
)
from .sampling import (
    FAKE_METHODS,
    TARGET_FAKE_EXPOSURES_PER_METHOD,
    BalancedRealFakeBatchSampler,
    FixedMethodExposureSampler,
)


SUPPORTED_STUDY_MODELS = {
    "xception",
    "ucf",
    "spsl",
}

BALANCED_FRAME_MODELS = {
    "xception",
    "spsl",
}


# Frozen valid FIT frame inventory produced by clean-face materialization.
EXPECTED_FIT_SOURCE_COUNTS = {
    "FF-real": 18426,
    "FF-DF": 18398,
    "FF-F2F": 18423,
    "FF-FS": 18423,
    "FF-NT": 18407,
}


def study_controlled_training_enabled(
    config: Dict,
) -> bool:
    return bool(
        config.get(
            "study_controlled_training",
            False,
        )
    )


def _validate_epoch(
    epoch: int,
) -> None:
    if not isinstance(
        epoch,
        int,
    ):
        raise TypeError(
            "epoch must be an integer"
        )

    if epoch < 0:
        raise ValueError(
            "epoch must be non-negative"
        )


def _derive_epoch_seed(
    base_seed: int,
    epoch: int,
) -> int:
    validate_seed(
        base_seed
    )

    _validate_epoch(
        epoch
    )

    return (
        base_seed
        + epoch * 1_000_003
    )


def _validate_common_loader_config(
    config: Dict,
) -> None:
    model_name = config.get(
        "model_name"
    )

    if (
        model_name
        not in SUPPORTED_STUDY_MODELS
    ):
        raise ValueError(
            "study-controlled training supports only "
            "Xception, UCF, and SPSL; got {!r}".format(
                model_name
            )
        )

    if config.get(
        "ddp",
        False,
    ):
        raise ValueError(
            "study-controlled training does not use DDP"
        )

    batch_size = config.get(
        "train_batchSize"
    )

    if not isinstance(
        batch_size,
        int,
    ):
        raise TypeError(
            "train_batchSize must be an integer"
        )

    if batch_size <= 0:
        raise ValueError(
            "train_batchSize must be positive"
        )

    workers = config.get(
        "workers",
        0,
    )

    try:
        workers = int(
            workers
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            "workers must be integer-compatible"
        ) from exc

    if workers < 0:
        raise ValueError(
            "workers must be non-negative"
        )

    validate_seed(
        config.get(
            "manualSeed"
        )
    )


def _validate_frozen_frame_dataset(
    train_set,
) -> None:
    if not hasattr(
        train_set,
        "label_list",
    ):
        raise AttributeError(
            "frame training dataset must expose label_list"
        )

    if not hasattr(
        train_set,
        "source_label_list",
    ):
        raise AttributeError(
            "frame training dataset must expose source_label_list"
        )

    if (
        len(
            train_set.label_list
        )
        != len(
            train_set.source_label_list
        )
    ):
        raise ValueError(
            "label_list and source_label_list are not index-aligned"
        )

    actual_counts = Counter(
        train_set.source_label_list
    )

    if dict(
        actual_counts
    ) != EXPECTED_FIT_SOURCE_COUNTS:
        raise ValueError(
            "frozen FIT source counts do not match the materialized "
            "study contract; expected {}, got {}".format(
                EXPECTED_FIT_SOURCE_COUNTS,
                dict(
                    sorted(
                        actual_counts.items()
                    )
                ),
            )
        )


def _ucf_specific_label_mapping(
    config: Dict,
) -> Dict[str, int]:
    label_dict = config.get(
        "label_dict",
        {}
    )

    if label_dict.get(
        "FF-real"
    ) != 0:
        raise ValueError(
            "UCF FF-real specific label must be 0"
        )

    mapping = {}

    for method in FAKE_METHODS:
        if method not in label_dict:
            raise ValueError(
                "UCF label_dict is missing {}".format(
                    method
                )
            )

        method_label = label_dict[
            method
        ]

        if not isinstance(
            method_label,
            int,
        ):
            raise TypeError(
                "UCF specific label for {} must be an integer".format(
                    method
                )
            )

        if method_label == 0:
            raise ValueError(
                "UCF fake specific label for {} cannot be 0".format(
                    method
                )
            )

        mapping[
            method
        ] = method_label

    if (
        len(
            set(
                mapping.values()
            )
        )
        != len(
            FAKE_METHODS
        )
    ):
        raise ValueError(
            "UCF fake manipulation labels must be distinct"
        )

    return mapping


def _validate_frozen_ucf_dataset(
    train_set,
    config: Dict,
) -> Dict[str, int]:
    if not hasattr(
        train_set,
        "fake_imglist",
    ):
        raise AttributeError(
            "UCF training requires pairDataset.fake_imglist"
        )

    if not hasattr(
        train_set,
        "real_imglist",
    ):
        raise AttributeError(
            "UCF training requires pairDataset.real_imglist"
        )

    mapping = (
        _ucf_specific_label_mapping(
            config
        )
    )

    if (
        len(
            train_set.real_imglist
        )
        != EXPECTED_FIT_SOURCE_COUNTS[
            "FF-real"
        ]
    ):
        raise ValueError(
            "UCF real source count mismatch: expected {}, got {}".format(
                EXPECTED_FIT_SOURCE_COUNTS[
                    "FF-real"
                ],
                len(
                    train_set.real_imglist
                ),
            )
        )

    fake_specific_counts = Counter(
        int(
            item[1]
        )
        for item in train_set.fake_imglist
    )

    expected_specific_counts = {
        mapping[
            method
        ]: (
            EXPECTED_FIT_SOURCE_COUNTS[
                method
            ]
        )
        for method in FAKE_METHODS
    }

    if dict(
        fake_specific_counts
    ) != expected_specific_counts:
        readable_actual = {
            method: fake_specific_counts.get(
                mapping[
                    method
                ],
                0,
            )
            for method in FAKE_METHODS
        }

        expected_readable = {
            method: (
                EXPECTED_FIT_SOURCE_COUNTS[
                    method
                ]
            )
            for method in FAKE_METHODS
        }

        raise ValueError(
            "UCF fake source counts do not match the frozen "
            "materialized FIT contract; expected {}, got {}".format(
                expected_readable,
                readable_actual,
            )
        )

    return mapping


def effective_image_batch_size(
    config: Dict,
) -> int:
    _validate_common_loader_config(
        config
    )

    loader_batch_size = config[
        "train_batchSize"
    ]

    if config[
        "model_name"
    ] == "ucf":
        return (
            2
            * loader_batch_size
        )

    return loader_batch_size


def _attach_study_rng(
    loader: DataLoader,
    generator,
    base_seed: int,
) -> DataLoader:
    loader._study_generator = (
        generator
    )

    loader._study_base_seed = (
        base_seed
    )

    return loader


def build_study_training_loader(
    train_set,
    config: Dict,
) -> DataLoader:
    _validate_common_loader_config(
        config
    )

    if not study_controlled_training_enabled(
        config
    ):
        raise ValueError(
            "study_controlled_training must be enabled"
        )

    model_name = config[
        "model_name"
    ]

    seed = config[
        "manualSeed"
    ]

    workers = int(
        config.get(
            "workers",
            0,
        )
    )

    collate_fn = getattr(
        train_set,
        "collate_fn",
        None,
    )

    generator = (
        make_data_loader_generator(
            seed
        )
    )

    if (
        model_name
        in BALANCED_FRAME_MODELS
    ):
        _validate_frozen_frame_dataset(
            train_set
        )

        batch_sampler = (
            BalancedRealFakeBatchSampler(
                labels=(
                    train_set.label_list
                ),
                source_labels=(
                    train_set.source_label_list
                ),
                batch_size=config[
                    "train_batchSize"
                ],
                target_fake_exposures_per_method=(
                    TARGET_FAKE_EXPOSURES_PER_METHOD
                ),
                seed=seed,
            )
        )

        loader = DataLoader(
            dataset=train_set,
            batch_sampler=batch_sampler,
            num_workers=workers,
            collate_fn=collate_fn,
            worker_init_fn=(
                seed_data_loader_worker
            ),
            generator=generator,
        )

    else:
        # UCF keeps its pairDataset behavior. The sampler controls only
        # which fake pair-item index is requested and how often.
        mapping = (
            _validate_frozen_ucf_dataset(
                train_set=train_set,
                config=config,
            )
        )

        method_order = [
            mapping[
                method
            ]
            for method in FAKE_METHODS
        ]

        method_targets = {
            mapping[
                method
            ]: (
                TARGET_FAKE_EXPOSURES_PER_METHOD
            )
            for method in FAKE_METHODS
        }

        method_names = {
            mapping[
                method
            ]: method
            for method in FAKE_METHODS
        }

        method_labels = [
            int(
                item[1]
            )
            for item
            in train_set.fake_imglist
        ]

        sampler = (
            FixedMethodExposureSampler(
                method_labels=method_labels,
                method_order=method_order,
                method_targets=method_targets,
                method_names=method_names,
                seed=seed,
            )
        )

        loader = DataLoader(
            dataset=train_set,
            batch_size=config[
                "train_batchSize"
            ],
            sampler=sampler,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=False,
            worker_init_fn=(
                seed_data_loader_worker
            ),
            generator=generator,
        )

    return _attach_study_rng(
        loader=loader,
        generator=generator,
        base_seed=seed,
    )


def build_study_testing_loader(
    test_set,
    config: Dict,
    test_name: str,
    seed_offset: int = 0,
) -> DataLoader:
    _validate_common_loader_config(
        config
    )

    if not study_controlled_training_enabled(
        config
    ):
        raise ValueError(
            "study_controlled_training must be enabled"
        )

    if not isinstance(
        seed_offset,
        int,
    ):
        raise TypeError(
            "seed_offset must be an integer"
        )

    if seed_offset < 0:
        raise ValueError(
            "seed_offset must be non-negative"
        )

    evaluation_seed = (
        config[
            "manualSeed"
        ]
        + seed_offset
    )

    generator = (
        make_data_loader_generator(
            evaluation_seed
        )
    )

    loader = DataLoader(
        dataset=test_set,
        batch_size=config[
            "test_batchSize"
        ],
        shuffle=False,
        num_workers=int(
            config.get(
                "workers",
                0,
            )
        ),
        collate_fn=getattr(
            test_set,
            "collate_fn",
            None,
        ),
        # Technical GPU batching must never discard a valid study input.
        drop_last=False,
        worker_init_fn=(
            seed_data_loader_worker
        ),
        generator=generator,
    )

    return _attach_study_rng(
        loader=loader,
        generator=generator,
        base_seed=evaluation_seed,
    )


def set_study_training_epoch(
    train_data_loader: DataLoader,
    epoch: int,
) -> None:
    """
    Address all study-controlled stochastic training behavior by epoch.

    Xception / SPSL:
        * fake-method top-up identity and ordering;
        * real replacement sampling;
        * batch ordering;
        * DataLoader worker RNG.

    UCF:
        * fake-method top-up identity and ordering;
        * pairDataset worker RNG, including random real partner choice.
    """
    _validate_epoch(
        epoch
    )

    batch_sampler = getattr(
        train_data_loader,
        "batch_sampler",
        None,
    )

    if isinstance(
        batch_sampler,
        BalancedRealFakeBatchSampler,
    ):
        batch_sampler.set_epoch(
            epoch
        )

    sampler = getattr(
        train_data_loader,
        "sampler",
        None,
    )

    if isinstance(
        sampler,
        FixedMethodExposureSampler,
    ):
        sampler.set_epoch(
            epoch
        )

    generator = getattr(
        train_data_loader,
        "_study_generator",
        None,
    )

    base_seed = getattr(
        train_data_loader,
        "_study_base_seed",
        None,
    )

    if (
        generator is None
        or base_seed is None
    ):
        raise RuntimeError(
            "training loader is missing study RNG metadata"
        )

    generator.manual_seed(
        _derive_epoch_seed(
            base_seed=base_seed,
            epoch=epoch,
        )
    )


def loader_contract_summary(
    train_data_loader: DataLoader,
    config: Dict,
) -> Dict[str, object]:
    _validate_common_loader_config(
        config
    )

    effective_batch = (
        effective_image_batch_size(
            config
        )
    )

    summary = {
        "model_name": config[
            "model_name"
        ],
        "loader_batch_size": config[
            "train_batchSize"
        ],
        "effective_image_batch_size": (
            effective_batch
        ),
        "workers": int(
            config.get(
                "workers",
                0,
            )
        ),
        "seed": config[
            "manualSeed"
        ],
        "optimizer_steps_per_epoch": len(
            train_data_loader
        ),
        "image_exposures_per_epoch": (
            len(
                train_data_loader
            )
            * effective_batch
        ),
    }

    batch_sampler = getattr(
        train_data_loader,
        "batch_sampler",
        None,
    )

    if isinstance(
        batch_sampler,
        BalancedRealFakeBatchSampler,
    ):
        summary.update(
            batch_sampler.contract_summary()
        )

        summary[
            "sampling_mode"
        ] = (
            "balanced_fixed_method_exposure"
        )

        return summary

    sampler = getattr(
        train_data_loader,
        "sampler",
        None,
    )

    if isinstance(
        sampler,
        FixedMethodExposureSampler,
    ):
        summary.update(
            sampler.contract_summary()
        )

        summary[
            "sampling_mode"
        ] = (
            "ucf_fixed_method_exposure"
        )

        summary[
            "pair_items_per_step"
        ] = config[
            "train_batchSize"
        ]

        summary[
            "real_exposures_per_epoch"
        ] = len(
            sampler
        )

        summary[
            "fake_exposures_per_epoch"
        ] = len(
            sampler
        )

        return summary

    raise RuntimeError(
        "study training loader does not expose a recognized "
        "study sampler"
    )
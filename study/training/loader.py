"""
Study-controlled DeepfakeBench DataLoader construction.

Xception / SPSL:
    * ordinary frame dataset;
    * balanced real/fake batch sampler;
    * every fake item exactly once per epoch;
    * real sampling with replacement.

UCF:
    * detector-specific pairDataset preserved;
    * pair-item batching preserved.

For all training loaders:
    * DataLoader worker RNG is derived deterministically from
      the study seed and current epoch.
"""

from typing import Dict

from torch.utils.data import DataLoader

from .reproducibility import (
    make_data_loader_generator,
    seed_data_loader_worker,
    validate_seed,
)
from .sampling import BalancedRealFakeBatchSampler


SUPPORTED_STUDY_MODELS = {
    "xception",
    "ucf",
    "spsl",
}

BALANCED_FRAME_MODELS = {
    "xception",
    "spsl",
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
    if not isinstance(epoch, int):
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

    if model_name not in SUPPORTED_STUDY_MODELS:
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
            2 * loader_batch_size
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

    if model_name in BALANCED_FRAME_MODELS:
        if not hasattr(
            train_set,
            "label_list",
        ):
            raise AttributeError(
                "balanced training requires train_set.label_list"
            )

        batch_sampler = (
            BalancedRealFakeBatchSampler(
                labels=train_set.label_list,
                batch_size=config[
                    "train_batchSize"
                ],
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
        # UCF retains pairDataset semantics.
        loader = DataLoader(
            dataset=train_set,
            batch_size=config[
                "train_batchSize"
            ],
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
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
        config["manualSeed"]
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
        drop_last=(
            test_name
            == "DeepFakeDetection"
        ),
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
    Make the complete stochastic training-input path addressable by epoch.

    For Xception/SPSL this controls both:
        * balanced batch ordering;
        * DataLoader worker RNG.

    For UCF this controls:
        * shuffled pairDataset ordering;
        * worker RNG used for real-pair selection and augmentation.
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
        ] = "balanced_frame_sampler"

    else:
        summary[
            "sampling_mode"
        ] = "detector_specific_pair_dataset"

        summary[
            "pair_items_per_step"
        ] = config[
            "train_batchSize"
        ]

    return summary
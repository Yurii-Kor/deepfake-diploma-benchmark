# author: Zhiyuan Yan
# study runtime adaptation: controlled Xception / UCF / SPSL training

import argparse
import datetime
import os
import sys

import torch
import torch.optim as optim
import yaml


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(
        PROJECT_ROOT
    )


from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import (
    DeepfakeAbstractBaseDataset,
    pairDataset,
)
from metrics.utils import parse_metric_for_print
from logger import create_logger

from study.training import (
    build_study_testing_loader,
    build_study_training_loader,
    completed_epoch_indices,
    configure_training_reproducibility,
    effective_image_batch_size,
    loader_contract_summary,
    set_study_training_epoch,
    study_controlled_training_enabled,
    training_budget_summary,
)


SUPPORTED_MODELS = {
    "xception",
    "ucf",
    "spsl",
}


parser = argparse.ArgumentParser(
    description=(
        "Controlled training launcher for the "
        "Xception/UCF/SPSL robustness study."
    )
)

parser.add_argument(
    "--detector_path",
    type=str,
    default="./training/config/detector/study_xception.yaml",
    help=(
        "path to study_xception.yaml, study_ucf.yaml, "
        "or study_spsl.yaml"
    ),
)

parser.add_argument(
    "--no-save_ckpt",
    dest="save_ckpt",
    action="store_false",
    default=True,
)

parser.add_argument(
    "--no-save_feat",
    dest="save_feat",
    action="store_false",
    default=True,
)

parser.add_argument(
    "--local_rank",
    type=int,
    default=0,
    help="CUDA device index for single-process study training",
)

parser.add_argument(
    "--task_target",
    type=str,
    default=None,
)

args = parser.parse_args()


def load_study_config(
    detector_path: str,
):
    with open(
        "./training/config/train_config.yaml",
        "r",
    ) as file:
        common_config = yaml.safe_load(
            file
        )

    with open(
        detector_path,
        "r",
    ) as file:
        detector_config = yaml.safe_load(
            file
        )

    # Study semantics:
    # common configuration provides defaults,
    # detector/study configuration is authoritative.
    config = dict(
        common_config
    )

    config.update(
        detector_config
    )

    return config


def validate_study_config(
    config,
) -> None:
    model_name = config.get(
        "model_name"
    )

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            "unsupported study detector: {!r}; "
            "expected one of {}".format(
                model_name,
                sorted(
                    SUPPORTED_MODELS
                ),
            )
        )

    if not study_controlled_training_enabled(
        config
    ):
        raise ValueError(
            "study_controlled_training must be true "
            "in the study detector configuration"
        )

    if config.get(
        "lmdb",
        False,
    ):
        raise ValueError(
            "study training requires lmdb: false so that "
            "the materialized study JSON/frame paths remain authoritative"
        )

    if config.get(
        "lr_scheduler"
    ) is not None:
        raise ValueError(
            "the controlled study protocol requires lr_scheduler: null"
        )

    if config.get(
        "manualSeed"
    ) != 1024:
        raise ValueError(
            "the controlled study protocol requires manualSeed: 1024"
        )

    if config.get(
        "nEpochs"
    ) != 10:
        raise ValueError(
            "the controlled study protocol requires nEpochs: 10"
        )

    if not (
        0
        <= config.get(
            "start_epoch",
            0,
        )
        <= config[
            "nEpochs"
        ]
    ):
        raise ValueError(
            "start_epoch must be within the 10-epoch study budget"
        )

    frame_num = config.get(
        "frame_num",
        {},
    )

    if frame_num.get(
        "train"
    ) != 32:
        raise ValueError(
            "the controlled study protocol requires "
            "32 training frames per video"
        )

    if frame_num.get(
        "test"
    ) != 32:
        raise ValueError(
            "the controlled study protocol requires "
            "32 DEV/test frames per video"
        )

    if config.get(
        "test_batchSize"
    ) != 32:
        raise ValueError(
            "study test_batchSize must be 32"
        )

    if model_name in {
        "xception",
        "spsl",
    }:
        if config.get(
            "train_batchSize"
        ) != 32:
            raise ValueError(
                "{} train_batchSize must be 32".format(
                    model_name
                )
            )

        if config.get(
            "dataset_type"
        ) not in (
            None,
            "frame",
        ):
            raise ValueError(
                "{} must use the ordinary frame dataset".format(
                    model_name
                )
            )

    elif model_name == "ucf":
        if config.get(
            "train_batchSize"
        ) != 16:
            raise ValueError(
                "UCF train_batchSize must be 16 pair-items"
            )

        if config.get(
            "dataset_type"
        ) != "pair":
            raise ValueError(
                "UCF must preserve dataset_type: pair"
            )

    optimizer = config.get(
        "optimizer",
        {},
    )

    if optimizer.get(
        "type"
    ) != "adam":
        raise ValueError(
            "the controlled study protocol requires Adam"
        )

    adam = optimizer.get(
        "adam",
        {},
    )

    expected_adam = {
        "lr": 0.0002,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0005,
        "amsgrad": False,
    }

    for key, expected in expected_adam.items():
        actual = adam.get(
            key
        )

        if actual != expected:
            raise ValueError(
                "unexpected Adam {}: {!r}; expected {!r}".format(
                    key,
                    actual,
                    expected,
                )
            )


def prepare_training_data(
    config,
):
    model_name = config[
        "model_name"
    ]

    if model_name == "ucf":
        train_set = pairDataset(
            config=config,
            mode="train",
        )

    elif model_name in {
        "xception",
        "spsl",
    }:
        train_set = (
            DeepfakeAbstractBaseDataset(
                config=config,
                mode="train",
            )
        )

    else:
        raise ValueError(
            "unsupported study detector: {}".format(
                model_name
            )
        )

    return build_study_training_loader(
        train_set=train_set,
        config=config,
    )


def prepare_testing_data(
    config,
):
    test_data_loaders = {}

    for test_index, test_name in enumerate(
        config[
            "test_dataset"
        ]
    ):
        test_config = dict(
            config
        )

        test_config[
            "test_dataset"
        ] = test_name

        test_set = (
            DeepfakeAbstractBaseDataset(
                config=test_config,
                mode="test",
            )
        )

        test_data_loaders[
            test_name
        ] = build_study_testing_loader(
            test_set=test_set,
            config=test_config,
            test_name=test_name,
            seed_offset=(
                10_000
                + test_index
            ),
        )

    return test_data_loaders


def choose_optimizer(
    model,
    config,
):
    adam = config[
        "optimizer"
    ]["adam"]

    return optim.Adam(
        params=model.parameters(),
        lr=adam[
            "lr"
        ],
        weight_decay=adam[
            "weight_decay"
        ],
        betas=(
            adam[
                "beta1"
            ],
            adam[
                "beta2"
            ],
        ),
        eps=adam[
            "eps"
        ],
        amsgrad=adam[
            "amsgrad"
        ],
    )


def choose_metric(
    config,
):
    metric = config.get(
        "metric_scoring"
    )

    if metric != "video_auc":
        raise ValueError(
            "study checkpoint selection requires metric_scoring: video_auc"
        )

    return metric


def main():
    config = load_study_config(
        args.detector_path
    )

    config[
        "local_rank"
    ] = args.local_rank

    config[
        "ddp"
    ] = False

    config[
        "save_ckpt"
    ] = args.save_ckpt

    config[
        "save_feat"
    ] = args.save_feat

    if args.task_target is not None:
        config[
            "task_target"
        ] = args.task_target

    validate_study_config(
        config
    )

    if config.get(
        "cuda",
        False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training requested but CUDA is unavailable"
            )

        torch.cuda.set_device(
            args.local_rank
        )

    configure_training_reproducibility(
        seed=config[
            "manualSeed"
        ],
        use_cuda=config.get(
            "cuda",
            False,
        ),
        deterministic_cudnn=True,
    )

    timestamp = (
        datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
    )

    task_target = config.get(
        "task_target"
    )

    task_suffix = (
        "_{}".format(
            task_target
        )
        if task_target
        else ""
    )

    logger_path = os.path.join(
        config[
            "log_dir"
        ],
        "{}{}_{}".format(
            config[
                "model_name"
            ],
            task_suffix,
            timestamp,
        ),
    )

    os.makedirs(
        logger_path,
        exist_ok=True,
    )

    logger = create_logger(
        os.path.join(
            logger_path,
            "training.log",
        )
    )

    logger.info(
        "Save log to {}".format(
            logger_path
        )
    )

    logger.info(
        "--------------- Study Configuration ---------------"
    )

    for key, value in config.items():
        logger.info(
            "%s: %s",
            key,
            value,
        )

    train_data_loader = (
        prepare_training_data(
            config
        )
    )

    loader_summary = (
        loader_contract_summary(
            train_data_loader,
            config,
        )
    )

    logger.info(
        "Study training loader contract: %s",
        loader_summary,
    )

    test_data_loaders = (
        prepare_testing_data(
            config
        )
    )

    model_class = DETECTOR[
        config[
            "model_name"
        ]
    ]

    model = model_class(
        config
    )

    optimizer = choose_optimizer(
        model,
        config,
    )

    metric_scoring = (
        choose_metric(
            config
        )
    )

    trainer = Trainer(
        config,
        model,
        optimizer,
        None,
        logger,
        metric_scoring,
        time_now=timestamp,
    )

    budget = training_budget_summary(
        steps_per_epoch=len(
            train_data_loader
        ),
        effective_images_per_step=(
            effective_image_batch_size(
                config
            )
        ),
        start_epoch=config[
            "start_epoch"
        ],
        n_epochs=config[
            "nEpochs"
        ],
    )

    logger.info(
        "Study training budget: %s",
        budget,
    )

    best_metric = None

    for epoch in completed_epoch_indices(
        start_epoch=config[
            "start_epoch"
        ],
        n_epochs=config[
            "nEpochs"
        ],
    ):
        set_study_training_epoch(
            train_data_loader,
            epoch,
        )

        trainer.model.epoch = (
            epoch
        )

        best_metric = (
            trainer.train_epoch(
                epoch=epoch,
                train_data_loader=(
                    train_data_loader
                ),
                test_data_loaders=(
                    test_data_loaders
                ),
            )
        )

        if best_metric is not None:
            logger.info(
                "===> Epoch[%s] end with DEV %s: %s",
                epoch,
                metric_scoring,
                parse_metric_for_print(
                    best_metric
                ),
            )

    if best_metric is None:
        logger.info(
            "Training completed without a DEV metric."
        )
    else:
        logger.info(
            "Training completed. Best DEV metric: %s",
            parse_metric_for_print(
                best_metric
            ),
        )

    for writer in trainer.writers.values():
        writer.close()


if __name__ == "__main__":
    main()
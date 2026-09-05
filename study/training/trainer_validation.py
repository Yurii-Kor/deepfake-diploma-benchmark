import json
import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from training.trainer.trainer import Trainer


class DummyModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

    def forward(
        self,
        data_dict,
        inference=False,
    ):
        batch_size = data_dict[
            "image"
        ].shape[
            0
        ]

        probability = torch.sigmoid(
            self.weight
        ).expand(
            batch_size
        )

        return {
            "prob": probability
        }


class SyntheticTrainer(
    Trainer
):
    """
    Trainer test double.

    Training computation itself is intentionally replaced by a counter.
    The production Trainer.train_epoch() control flow remains unchanged.
    """

    def __init__(
        self,
        *args,
        expected_steps_per_epoch,
        synthetic_dev_metrics,
        **kwargs,
    ):
        self.expected_steps_per_epoch = (
            expected_steps_per_epoch
        )

        self.synthetic_dev_metrics = list(
            synthetic_dev_metrics
        )

        self.total_train_steps = 0
        self.steps_since_dev = 0
        self.dev_call_steps = []

        super().__init__(
            *args,
            **kwargs,
        )

    def train_step(
        self,
        data_dict,
    ):
        self.total_train_steps += 1
        self.steps_since_dev += 1

        return {
            "overall": torch.tensor(
                1.0,
                device=self.device,
            )
        }

    def evaluate_dev(
        self,
        test_data_loaders,
    ):
        if (
            self.steps_since_dev
            != self.expected_steps_per_epoch
        ):
            raise AssertionError(
                "DEV evaluation occurred before a complete "
                "optimizer epoch finished: "
                f"{self.steps_since_dev}/"
                f"{self.expected_steps_per_epoch} steps."
            )

        evaluation_index = len(
            self.dev_call_steps
        )

        if (
            evaluation_index
            >= len(
                self.synthetic_dev_metrics
            )
        ):
            raise AssertionError(
                "DEV was evaluated more times than expected."
            )

        self.dev_call_steps.append(
            self.total_train_steps
        )

        self.steps_since_dev = 0

        return (
            "FaceForensics++",
            dict(
                self.synthetic_dev_metrics[
                    evaluation_index
                ]
            ),
        )


def _make_logger():
    logger = logging.getLogger(
        "study_trainer_validation"
    )

    logger.handlers.clear()

    logger.addHandler(
        logging.NullHandler()
    )

    logger.propagate = False

    return logger


def _make_train_loader(
    steps_per_epoch,
):
    batches = []

    for _ in range(
        steps_per_epoch
    ):
        batches.append(
            {
                "image": torch.zeros(
                    2,
                    3,
                    4,
                    4,
                    dtype=torch.float32,
                ),
                "label": torch.zeros(
                    2,
                    dtype=torch.long,
                ),
            }
        )

    return batches


def _dev_metrics(
    frame_auc,
    video_auc,
):
    return {
        "frame_auc": float(
            frame_auc
        ),
        "video_auc": float(
            video_auc
        ),
        "frame_count": 128,
        "video_count": 4,
        "real_video_count": 2,
        "fake_video_count": 2,
        "min_frames_per_video": 32,
        "max_frames_per_video": 32,
    }


def validate_trainer_control_flow():
    steps_per_epoch = 4

    synthetic_dev_metrics = [
        _dev_metrics(
            frame_auc=0.99,
            video_auc=0.60,
        ),
        _dev_metrics(
            frame_auc=0.50,
            video_auc=0.80,
        ),
        _dev_metrics(
            frame_auc=0.999,
            video_auc=0.80,
        ),
    ]

    with tempfile.TemporaryDirectory() as temporary_dir:
        config = {
            "study_controlled_training": True,
            "model_name": "xception",
            "ddp": False,
            "cuda": False,
            "lr_scheduler": None,
            "optimizer": {
                "type": "adam"
            },
            "log_dir": temporary_dir,
            "task_target": (
                "trainer_validation"
            ),
            "save_ckpt": True,
            "manualSeed": 1024,
            "rec_iter": 1000,
            "test_dataset": [
                "FaceForensics++"
            ],
        }

        model = DummyModel()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

        trainer = SyntheticTrainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            logger=_make_logger(),
            metric_scoring="video_auc",
            time_now="synthetic",
            expected_steps_per_epoch=(
                steps_per_epoch
            ),
            synthetic_dev_metrics=(
                synthetic_dev_metrics
            ),
        )

        train_loader = (
            _make_train_loader(
                steps_per_epoch
            )
        )

        synthetic_dev_loader = {
            "FaceForensics++": object()
        }

        for epoch in range(
            3
        ):
            trainer.train_epoch(
                epoch=epoch,
                train_data_loader=(
                    train_loader
                ),
                test_data_loaders=(
                    synthetic_dev_loader
                ),
            )

        # ------------------------------------------------------------
        # DEV must run exactly once after each COMPLETE epoch.
        # ------------------------------------------------------------

        assert (
            trainer.dev_call_steps
            == [
                4,
                8,
                12,
            ]
        )

        assert (
            trainer.total_train_steps
            == 12
        )

        assert (
            trainer.steps_since_dev
            == 0
        )

        # ------------------------------------------------------------
        # Selection must use VIDEO AUC, not frame AUC.
        #
        # Epoch 0:
        #   frame AUC = 0.99
        #   video AUC = 0.60
        #
        # Epoch 1:
        #   frame AUC = 0.50
        #   video AUC = 0.80
        #
        # Therefore epoch 1 must replace epoch 0.
        # ------------------------------------------------------------

        assert (
            trainer.best_epoch
            == 1
        )

        assert (
            trainer.best_video_auc
            == 0.80
        )

        best_summary = (
            trainer
            .best_metrics_all_time[
                "DEV"
            ]
        )

        assert (
            best_summary[
                "completed_epoch"
            ]
            == 1
        )

        assert (
            best_summary[
                "video_auc"
            ]
            == 0.80
        )

        assert (
            best_summary[
                "frame_auc"
            ]
            == 0.50
        )

        # ------------------------------------------------------------
        # Epoch 2 has a much larger frame AUC but equal video AUC.
        # Strict comparison must retain the EARLIER epoch 1 checkpoint.
        # ------------------------------------------------------------

        checkpoint_dir = (
            Path(
                trainer.checkpoint_dir
            )
        )

        checkpoint_path = (
            checkpoint_dir
            / "best.pth"
        )

        metadata_path = (
            checkpoint_dir
            / "best_metadata.json"
        )

        assert checkpoint_path.is_file()
        assert metadata_path.is_file()

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        assert (
            metadata[
                "selection_metric"
            ]
            == "video_auc"
        )

        assert (
            metadata[
                "selection_value"
            ]
            == 0.80
        )

        assert (
            metadata[
                "completed_epoch"
            ]
            == 1
        )

        assert (
            metadata[
                "selection_rule"
            ]
            == (
                "strictly_greater_video_auc_earliest_tie"
            )
        )

        assert (
            metadata[
                "optimizer_steps_completed"
            ]
            == 8
        )

        # ------------------------------------------------------------
        # DEV history must retain every completed epoch.
        # ------------------------------------------------------------

        history_path = (
            Path(
                trainer.dev_dir
            )
            / "dev_history.jsonl"
        )

        assert history_path.is_file()

        with history_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            history = [
                json.loads(
                    line
                )
                for line in file
                if line.strip()
            ]

        assert (
            len(
                history
            )
            == 3
        )

        assert [
            record[
                "completed_epoch"
            ]
            for record in history
        ] == [
            0,
            1,
            2,
        ]

        assert [
            record[
                "optimizer_steps_completed"
            ]
            for record in history
        ] == [
            4,
            8,
            12,
        ]

        assert [
            record[
                "improved_best"
            ]
            for record in history
        ] == [
            True,
            True,
            False,
        ]

        assert [
            record[
                "selection_value"
            ]
            for record in history
        ] == [
            0.60,
            0.80,
            0.80,
        ]

    print(
        "TRAINER CONTROL FLOW"
    )

    print(
        "  optimizer steps/epoch:    4"
    )

    print(
        "  completed epochs:         3"
    )

    print(
        "  DEV calls:                after steps 4, 8, 12"
    )

    print(
        "  DEV evaluations/epoch:    exactly 1"
    )

    print()
    print(
        "CHECKPOINT SELECTION"
    )

    print(
        "  selection metric:         video_auc"
    )

    print(
        "  epoch 0 video_auc:        0.60"
    )

    print(
        "  epoch 1 video_auc:        0.80 -> best"
    )

    print(
        "  epoch 2 video_auc:        0.80 -> tie retained"
    )

    print(
        "  selected epoch:           1"
    )

    print(
        "  frame-AUC distraction:    ignored"
    )

    print()
    print(
        "ARTIFACT CONTRACT"
    )

    print(
        "  best.pth:                 created"
    )

    print(
        "  best_metadata.json:       created"
    )

    print(
        "  dev_history.jsonl:        3 completed epochs"
    )

    print()
    print(
        "TRAINER VALIDATION PASSED"
    )


def main():
    validate_trainer_control_flow()


if __name__ == "__main__":
    main()
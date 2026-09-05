import datetime
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from tqdm import tqdm


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        2
    ]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.append(
        str(
            PROJECT_ROOT
        )
    )


from study.training.dev_metrics import (
    compute_dev_metrics,
)


SUPPORTED_STUDY_MODELS = {
    "xception",
    "ucf",
    "spsl",
}

CHECKPOINT_SELECTION_METRIC = (
    "video_auc"
)


class Trainer:
    """
    Controlled trainer for the Xception/UCF/SPSL robustness study.

    Study contract:
      * single-process training only;
      * Adam optimizer only;
      * no scheduler;
      * one complete optimizer epoch before DEV evaluation;
      * exactly one DEV evaluation per completed epoch;
      * checkpoint selection by aggregate DEV video-level AUC;
      * a strictly greater DEV video AUC replaces the best checkpoint;
      * equal values retain the earlier checkpoint;
      * held-out validation/test/external partitions are never used here.
    """

    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring="video_auc",
        time_now=None,
        swa_model=None,
    ):
        if config is None:
            raise ValueError(
                "config must be provided."
            )

        if model is None:
            raise ValueError(
                "model must be provided."
            )

        if optimizer is None:
            raise ValueError(
                "optimizer must be provided."
            )

        if logger is None:
            raise ValueError(
                "logger must be provided."
            )

        if not config.get(
            "study_controlled_training",
            False,
        ):
            raise ValueError(
                "Controlled Trainer may only be used when "
                "study_controlled_training=true."
            )

        model_name = config.get(
            "model_name"
        )

        if model_name not in (
            SUPPORTED_STUDY_MODELS
        ):
            raise ValueError(
                "Unsupported study model: "
                f"{model_name}"
            )

        if config.get(
            "ddp",
            False,
        ):
            raise ValueError(
                "DDP is not supported by the controlled study Trainer."
            )

        if scheduler is not None:
            raise ValueError(
                "The controlled study uses no learning-rate scheduler."
            )

        if config.get(
            "lr_scheduler"
        ) is not None:
            raise ValueError(
                "Study config must set lr_scheduler: null."
            )

        if (
            config.get(
                "optimizer",
                {},
            ).get(
                "type"
            )
            != "adam"
        ):
            raise ValueError(
                "The controlled study Trainer requires Adam."
            )

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.logger = logger

        # Retained only for compatibility with the current train.py API.
        # Checkpoint selection is intentionally fixed to video_auc.
        self.metric_scoring = (
            metric_scoring
        )

        self.selection_metric = (
            CHECKPOINT_SELECTION_METRIC
        )

        self.writers = {}

        if config.get(
            "cuda",
            False,
        ):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable."
                )

            self.device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
        else:
            self.device = torch.device(
                "cpu"
            )

        self.model.to(
            self.device
        )

        self.model.device = (
            self.device
        )

        if time_now is None:
            time_now = (
                datetime.datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S"
                )
            )

        self.timenow = (
            time_now
        )

        task_target = config.get(
            "task_target"
        )

        task_suffix = (
            f"_{task_target}"
            if task_target
            else ""
        )

        self.log_dir = os.path.join(
            config[
                "log_dir"
            ],
            (
                f"{model_name}"
                f"{task_suffix}_"
                f"{self.timenow}"
            ),
        )

        self.checkpoint_dir = (
            os.path.join(
                self.log_dir,
                "checkpoints",
            )
        )

        self.dev_dir = (
            os.path.join(
                self.log_dir,
                "dev",
            )
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.dev_dir,
            exist_ok=True,
        )

        self.best_video_auc = float(
            "-inf"
        )

        self.best_epoch = None

        self.best_metrics_all_time = {
            "DEV": {}
        }

        self.logger.info(
            "Controlled Trainer initialized."
        )

        self.logger.info(
            "Checkpoint selection metric: %s",
            self.selection_metric,
        )

        self.logger.info(
            "Checkpoint tie policy: retain earlier checkpoint."
        )

    def setTrain(
        self,
    ):
        self.model.train()

    def setEval(
        self,
    ):
        self.model.eval()

    def _move_batch_to_device(
        self,
        data_dict,
    ):
        moved = {}

        for (
            key,
            value,
        ) in data_dict.items():
            if torch.is_tensor(
                value
            ):
                moved[
                    key
                ] = value.to(
                    self.device,
                    non_blocking=True,
                )
            else:
                moved[
                    key
                ] = value

        return moved

    @staticmethod
    def _loss_to_float(
        value,
    ):
        if torch.is_tensor(
            value
        ):
            if value.numel() != 1:
                raise ValueError(
                    "Study losses must be scalar tensors."
                )

            return float(
                value.detach().cpu().item()
            )

        return float(
            value
        )

    def train_step(
        self,
        data_dict,
    ):
        predictions = self.model(
            data_dict
        )

        losses = self.model.get_losses(
            data_dict,
            predictions,
        )

        if not isinstance(
            losses,
            Mapping,
        ):
            raise TypeError(
                "Detector get_losses() must return a mapping."
            )

        if "overall" not in losses:
            raise KeyError(
                "Detector losses must contain 'overall'."
            )

        overall_loss = losses[
            "overall"
        ]

        if not torch.is_tensor(
            overall_loss
        ):
            raise TypeError(
                "'overall' loss must be a torch.Tensor."
            )

        if overall_loss.numel() != 1:
            raise ValueError(
                "'overall' loss must be scalar."
            )

        if not torch.isfinite(
            overall_loss
        ).item():
            raise ValueError(
                "Non-finite training loss encountered."
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        overall_loss.backward()

        self.optimizer.step()

        return losses

    @staticmethod
    def _extract_probability_vector(
        predictions,
    ):
        if not isinstance(
            predictions,
            Mapping,
        ):
            raise TypeError(
                "Study detector inference must return a mapping."
            )

        if "prob" not in predictions:
            raise KeyError(
                "Study detector inference must expose 'prob'."
            )

        probabilities = predictions[
            "prob"
        ]

        if not torch.is_tensor(
            probabilities
        ):
            raise TypeError(
                "predictions['prob'] must be a torch.Tensor."
            )

        if probabilities.ndim == 2:
            if probabilities.shape[
                1
            ] != 1:
                raise ValueError(
                    "Study detector 'prob' output must contain "
                    "one manipulation-oriented probability per frame."
                )

            probabilities = (
                probabilities[
                    :,
                    0
                ]
            )

        if probabilities.ndim != 1:
            raise ValueError(
                "Study detector 'prob' output must be one-dimensional."
            )

        if not torch.isfinite(
            probabilities
        ).all().item():
            raise ValueError(
                "Non-finite DEV detector probabilities encountered."
            )

        return probabilities

    @torch.no_grad()
    def evaluate_dev(
        self,
        test_data_loaders,
    ):
        if not isinstance(
            test_data_loaders,
            Mapping,
        ):
            raise TypeError(
                "DEV loaders must be provided as a mapping."
            )

        loader_keys = list(
            test_data_loaders.keys()
        )

        if len(
            loader_keys
        ) != 1:
            raise ValueError(
                "Controlled checkpoint selection requires exactly "
                "one aggregate DEV dataset."
            )

        configured_dev = (
            self.config.get(
                "test_dataset"
            )
        )

        if not isinstance(
            configured_dev,
            list,
        ):
            raise ValueError(
                "Study test_dataset must be a one-item list."
            )

        if len(
            configured_dev
        ) != 1:
            raise ValueError(
                "Study checkpoint selection requires exactly "
                "one DEV dataset."
            )

        dev_key = loader_keys[
            0
        ]

        if dev_key != configured_dev[
            0
        ]:
            raise ValueError(
                "DEV loader key does not match configured test_dataset: "
                f"{dev_key} != {configured_dev[0]}"
            )

        if dev_key != "FaceForensics++":
            raise ValueError(
                "Controlled training expects the aggregate "
                "FaceForensics++ DEV execution view."
            )

        data_loader = (
            test_data_loaders[
                dev_key
            ]
        )

        if not isinstance(
            data_loader.sampler,
            SequentialSampler,
        ):
            raise ValueError(
                "DEV loader must preserve deterministic dataset order."
            )

        if not hasattr(
            data_loader.dataset,
            "data_dict",
        ):
            raise ValueError(
                "DEV dataset must expose data_dict."
            )

        frame_paths = list(
            data_loader.dataset.data_dict[
                "image"
            ]
        )

        self.setEval()

        prediction_parts = []
        label_parts = []

        for data_dict in tqdm(
            data_loader,
            total=len(
                data_loader
            ),
            desc="DEV",
        ):
            data_dict = dict(
                data_dict
            )

            if "label_spe" in data_dict:
                data_dict.pop(
                    "label_spe"
                )

            if "label" not in data_dict:
                raise KeyError(
                    "DEV batch does not contain 'label'."
                )

            if not torch.is_tensor(
                data_dict[
                    "label"
                ]
            ):
                raise TypeError(
                    "DEV labels must be torch tensors."
                )

            data_dict[
                "label"
            ] = torch.where(
                data_dict[
                    "label"
                ]
                != 0,
                1,
                0,
            ).long()

            data_dict = (
                self._move_batch_to_device(
                    data_dict
                )
            )

            predictions = self.model(
                data_dict,
                inference=True,
            )

            probabilities = (
                self._extract_probability_vector(
                    predictions
                )
            )

            labels = data_dict[
                "label"
            ]

            if labels.ndim != 1:
                labels = labels.reshape(
                    -1
                )

            if (
                len(
                    probabilities
                )
                != len(
                    labels
                )
            ):
                raise ValueError(
                    "DEV prediction and label batch sizes do not match."
                )

            prediction_parts.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            label_parts.append(
                labels
                .detach()
                .cpu()
                .numpy()
            )

        if not prediction_parts:
            raise ValueError(
                "DEV loader produced no predictions."
            )

        frame_predictions = (
            np.concatenate(
                prediction_parts,
                axis=0,
            )
        )

        frame_labels = (
            np.concatenate(
                label_parts,
                axis=0,
            )
        )

        if len(
            frame_paths
        ) != len(
            frame_predictions
        ):
            raise ValueError(
                "DEV frame-path order does not align with predictions: "
                f"{len(frame_paths)} paths vs "
                f"{len(frame_predictions)} predictions."
            )

        dev_metrics = (
            compute_dev_metrics(
                frame_paths=frame_paths,
                frame_scores=frame_predictions,
                frame_labels=frame_labels,
            )
        )

        return (
            dev_key,
            dev_metrics,
        )

    def _write_json_atomic(
        self,
        destination,
        payload,
    ):
        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            destination.with_name(
                destination.name
                + ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                sort_keys=True,
            )

            file.write(
                "\n"
            )

        os.replace(
            temporary_path,
            destination,
        )

    def _append_dev_history(
        self,
        record,
    ):
        history_path = os.path.join(
            self.dev_dir,
            "dev_history.jsonl",
        )

        with open(
            history_path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )

            file.write(
                "\n"
            )

    def _save_best_checkpoint(
        self,
        metadata,
    ):
        if not self.config.get(
            "save_ckpt",
            True,
        ):
            self.logger.info(
                "Checkpoint saving disabled; "
                "best DEV state not written to disk."
            )

            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "best.pth",
        )

        temporary_checkpoint = (
            checkpoint_path
            + ".tmp"
        )

        torch.save(
            self.model.state_dict(),
            temporary_checkpoint,
        )

        os.replace(
            temporary_checkpoint,
            checkpoint_path,
        )

        metadata_path = os.path.join(
            self.checkpoint_dir,
            "best_metadata.json",
        )

        self._write_json_atomic(
            metadata_path,
            metadata,
        )

        self.logger.info(
            "Best checkpoint saved: %s",
            checkpoint_path,
        )

        self.logger.info(
            "Best checkpoint metadata saved: %s",
            metadata_path,
        )

    def _update_best(
        self,
        epoch,
        optimizer_steps_completed,
        dev_key,
        dev_metrics,
    ):
        current_video_auc = float(
            dev_metrics[
                "video_auc"
            ]
        )

        improved = (
            current_video_auc
            > self.best_video_auc
        )

        if improved:
            self.best_video_auc = (
                current_video_auc
            )

            self.best_epoch = int(
                epoch
            )

            best_summary = {
                "video_auc": current_video_auc,
                "frame_auc": float(
                    dev_metrics[
                        "frame_auc"
                    ]
                ),
                "completed_epoch": int(
                    epoch
                ),
                "optimizer_steps_completed": int(
                    optimizer_steps_completed
                ),
                "video_count": int(
                    dev_metrics[
                        "video_count"
                    ]
                ),
                "real_video_count": int(
                    dev_metrics[
                        "real_video_count"
                    ]
                ),
                "fake_video_count": int(
                    dev_metrics[
                        "fake_video_count"
                    ]
                ),
            }

            self.best_metrics_all_time = {
                "DEV": best_summary
            }

            metadata = {
                "model_name": self.config[
                    "model_name"
                ],
                "dev_dataset": dev_key,
                "selection_metric": (
                    self.selection_metric
                ),
                "selection_value": (
                    current_video_auc
                ),
                "selection_rule": (
                    "strictly_greater_video_auc_earliest_tie"
                ),
                "completed_epoch": int(
                    epoch
                ),
                "optimizer_steps_completed": int(
                    optimizer_steps_completed
                ),
                "manual_seed": int(
                    self.config[
                        "manualSeed"
                    ]
                ),
                "dev_metrics": {
                    key: (
                        float(
                            value
                        )
                        if isinstance(
                            value,
                            float,
                        )
                        else int(
                            value
                        )
                    )
                    for (
                        key,
                        value,
                    ) in dev_metrics.items()
                },
            }

            self._save_best_checkpoint(
                metadata
            )

        return improved

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
    ):
        if test_data_loaders is None:
            raise ValueError(
                "Controlled study training requires DEV evaluation "
                "after every completed epoch."
            )

        if len(
            train_data_loader
        ) <= 0:
            raise ValueError(
                "Training loader contains no optimizer steps."
            )

        self.logger.info(
            "===> Epoch[%s] start",
            epoch,
        )

        self.setTrain()

        loss_sums = defaultdict(
            float
        )

        rec_iter = max(
            1,
            int(
                self.config.get(
                    "rec_iter",
                    100,
                )
            ),
        )

        for (
            iteration,
            data_dict,
        ) in tqdm(
            enumerate(
                train_data_loader
            ),
            total=len(
                train_data_loader
            ),
            desc=f"TRAIN epoch {epoch}",
        ):
            data_dict = (
                self._move_batch_to_device(
                    data_dict
                )
            )

            losses = self.train_step(
                data_dict
            )

            for (
                loss_name,
                loss_value,
            ) in losses.items():
                loss_sums[
                    loss_name
                ] += self._loss_to_float(
                    loss_value
                )

            if (
                iteration == 0
                or (
                    iteration
                    + 1
                )
                % rec_iter
                == 0
                or (
                    iteration
                    + 1
                )
                == len(
                    train_data_loader
                )
            ):
                loss_text = ", ".join(
                    (
                        f"{name}="
                        f"{loss_sums[name] / (iteration + 1):.6f}"
                    )
                    for name in sorted(
                        loss_sums
                    )
                )

                self.logger.info(
                    "Epoch[%s] optimizer-step %s/%s: %s",
                    epoch,
                    iteration + 1,
                    len(
                        train_data_loader
                    ),
                    loss_text,
                )

        # This point is the only DEV entry point:
        # the complete optimizer epoch has finished.
        dev_key, dev_metrics = (
            self.evaluate_dev(
                test_data_loaders
            )
        )

        optimizer_steps_completed = (
            (
                int(
                    epoch
                )
                + 1
            )
            * len(
                train_data_loader
            )
        )

        improved = self._update_best(
            epoch=epoch,
            optimizer_steps_completed=(
                optimizer_steps_completed
            ),
            dev_key=dev_key,
            dev_metrics=dev_metrics,
        )

        history_record = {
            "model_name": self.config[
                "model_name"
            ],
            "dev_dataset": dev_key,
            "completed_epoch": int(
                epoch
            ),
            "optimizer_steps_completed": int(
                optimizer_steps_completed
            ),
            "selection_metric": (
                self.selection_metric
            ),
            "selection_value": float(
                dev_metrics[
                    "video_auc"
                ]
            ),
            "improved_best": bool(
                improved
            ),
            "best_epoch_after_evaluation": int(
                self.best_epoch
            ),
            "best_video_auc_after_evaluation": float(
                self.best_video_auc
            ),
            "dev_metrics": {
                key: (
                    float(
                        value
                    )
                    if isinstance(
                        value,
                        float,
                    )
                    else int(
                        value
                    )
                )
                for (
                    key,
                    value,
                ) in dev_metrics.items()
            },
        }

        self._append_dev_history(
            history_record
        )

        self.logger.info(
            (
                "===> Epoch[%s] completed; "
                "DEV frame_auc=%.6f, "
                "video_auc=%.6f, "
                "videos=%s, "
                "best=%s"
            ),
            epoch,
            dev_metrics[
                "frame_auc"
            ],
            dev_metrics[
                "video_auc"
            ],
            dev_metrics[
                "video_count"
            ],
            improved,
        )

        return self.best_metrics_all_time
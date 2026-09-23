from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TRAINING_ROOT = (
    PROJECT_ROOT
    / "training"
)

if str(
    TRAINING_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            TRAINING_ROOT
        ),
    )


from detectors import DETECTOR


BASE_CONFIG_PATH = (
    TRAINING_ROOT
    / "config"
    / "train_config.yaml"
)

DETECTOR_SPECS = {
    "xception": {
        "config_path": (
            TRAINING_ROOT
            / "config"
            / "detector"
            / "study_xception.yaml"
        ),
        "checkpoint_path": (
            TRAINING_ROOT
            / "weights"
            / "xception_ffpp_fit_best.pth"
        ),
        "checkpoint_sha256": (
            "bb32d52ae99d9a21d92205050e7654b"
            "41f70b285bb158860499b582a08cd2d48"
        ),
    },
    "ucf": {
        "config_path": (
            TRAINING_ROOT
            / "config"
            / "detector"
            / "study_ucf.yaml"
        ),
        "checkpoint_path": (
            TRAINING_ROOT
            / "weights"
            / "ucf_ffpp_fit_best.pth"
        ),
        "checkpoint_sha256": (
            "8fa10f38d8a877fedd4a35fc6265db"
            "6cd93d0fd4367648d29312f37225535e58"
        ),
    },
    "spsl": {
        "config_path": (
            TRAINING_ROOT
            / "config"
            / "detector"
            / "study_spsl.yaml"
        ),
        "checkpoint_path": (
            TRAINING_ROOT
            / "weights"
            / "spsl_ffpp_fit_best.pth"
        ),
        "checkpoint_sha256": (
            "eb539587bddb092334411af16f7f2d13"
            "3d8f6a2b99462d07fc7ebd2e1c936c9c"
        ),
    },
}


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def load_detector_config(
    model_name: str,
) -> dict:
    if (
        model_name
        not in DETECTOR_SPECS
    ):
        raise ValueError(
            "Unsupported study detector: {}".format(
                model_name
            )
        )

    spec = DETECTOR_SPECS[
        model_name
    ]

    with BASE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    with spec[
        "config_path"
    ].open(
        "r",
        encoding="utf-8",
    ) as file:
        config.update(
            yaml.safe_load(
                file
            )
        )

    if (
        config[
            "model_name"
        ]
        != model_name
    ):
        raise RuntimeError(
            "Detector config/model mismatch: "
            "{} != {}".format(
                config[
                    "model_name"
                ],
                model_name,
            )
        )

    if (
        int(
            config[
                "resolution"
            ]
        )
        != 256
    ):
        raise RuntimeError(
            "Study detector resolution must be 256."
        )

    pretrained = config.get(
        "pretrained"
    )

    if (
        isinstance(
            pretrained,
            str,
        )
        and pretrained
        and pretrained.lower()
        != "none"
    ):
        pretrained_path = Path(
            pretrained
        )

        if not pretrained_path.is_absolute():
            pretrained_path = (
                PROJECT_ROOT
                / pretrained_path
            ).resolve()

        config[
            "pretrained"
        ] = str(
            pretrained_path
        )

    #
    # Construct all models on CPU first.
    # The fully selected checkpoint is loaded after construction,
    # then the complete model is transferred to the inference device.
    #
    config[
        "cuda"
    ] = False

    config[
        "ddp"
    ] = False

    return config


def resolve_device(
    requested: str = "auto",
) -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return device


class FrozenDetectorScorer:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if (
            model_name
            not in DETECTOR_SPECS
        ):
            raise ValueError(
                "Unsupported detector: {}".format(
                    model_name
                )
            )

        self.model_name = (
            model_name
        )

        self.batch_size = (
            batch_size
        )

        self.device = (
            resolve_device(
                device
            )
        )

        self.spec = (
            DETECTOR_SPECS[
                model_name
            ]
        )

        self.config = (
            load_detector_config(
                model_name
            )
        )

        self.checkpoint_path = (
            self.spec[
                "checkpoint_path"
            ]
            .resolve()
        )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Frozen checkpoint does not exist: {}".format(
                    self.checkpoint_path
                )
            )

        self.checkpoint_sha256 = (
            sha256_file(
                self.checkpoint_path
            )
        )

        expected_sha = (
            self.spec[
                "checkpoint_sha256"
            ]
        )

        if (
            self.checkpoint_sha256
            != expected_sha
        ):
            raise RuntimeError(
                "{} checkpoint SHA-256 mismatch.\n"
                "Expected: {}\n"
                "Actual:   {}".format(
                    model_name,
                    expected_sha,
                    self.checkpoint_sha256,
                )
            )

        model_class = (
            DETECTOR[
                model_name
            ]
        )

        self.model = (
            model_class(
                self.config
            )
        )

        state_dict = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        mean = self.config[
            "mean"
        ]

        std = self.config[
            "std"
        ]

        self.to_tensor = (
            T.ToTensor()
        )

        self.normalize = (
            T.Normalize(
                mean=mean,
                std=std,
            )
        )

        self.resolution = int(
            self.config[
                "resolution"
            ]
        )

    def preprocess_bgr(
        self,
        aligned_bgr: np.ndarray,
    ) -> torch.Tensor:
        if not isinstance(
            aligned_bgr,
            np.ndarray,
        ):
            raise TypeError(
                "aligned_bgr must be a NumPy array."
            )

        if (
            aligned_bgr.ndim != 3
            or aligned_bgr.shape[2] != 3
        ):
            raise ValueError(
                "aligned_bgr must have shape HxWx3."
            )

        if (
            aligned_bgr.dtype
            != np.uint8
        ):
            raise ValueError(
                "aligned_bgr must have dtype uint8."
            )

        image_rgb = cv2.cvtColor(
            aligned_bgr,
            cv2.COLOR_BGR2RGB,
        )

        #
        # Preserve the same detector-input construction
        # used by DeepfakeAbstractBaseDataset.load_rgb().
        #
        image_rgb = cv2.resize(
            image_rgb,
            (
                self.resolution,
                self.resolution,
            ),
            interpolation=(
                cv2.INTER_CUBIC
            ),
        )

        image = Image.fromarray(
            np.array(
                image_rgb,
                dtype=np.uint8,
            )
        )

        tensor = self.to_tensor(
            image
        )

        tensor = self.normalize(
            tensor
        )

        expected_shape = (
            3,
            self.resolution,
            self.resolution,
        )

        if (
            tuple(
                tensor.shape
            )
            != expected_shape
        ):
            raise RuntimeError(
                "Unexpected detector-input shape: "
                "{} != {}".format(
                    tuple(
                        tensor.shape
                    ),
                    expected_shape,
                )
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError(
                "Detector input contains "
                "non-finite values."
            )

        return tensor

    def score(
        self,
        aligned_bgr_images: Sequence[
            np.ndarray
        ],
        study_labels: Sequence[int],
    ) -> list[float]:
        if (
            len(
                aligned_bgr_images
            )
            != len(
                study_labels
            )
        ):
            raise ValueError(
                "Image/label count mismatch."
            )

        if not aligned_bgr_images:
            return []

        scores = []

        for start in range(
            0,
            len(
                aligned_bgr_images
            ),
            self.batch_size,
        ):
            end = min(
                start
                + self.batch_size,
                len(
                    aligned_bgr_images
                ),
            )

            batch_images = [
                self.preprocess_bgr(
                    image
                )
                for image in (
                    aligned_bgr_images[
                        start:end
                    ]
                )
            ]

            batch = torch.stack(
                batch_images,
                dim=0,
            ).to(
                self.device,
                non_blocking=(
                    self.device.type
                    == "cuda"
                ),
            )

            labels = torch.tensor(
                study_labels[
                    start:end
                ],
                dtype=torch.long,
                device=self.device,
            )

            data_dict = {
                "image": batch,
                "label": labels,
            }

            with torch.inference_mode():
                predictions = self.model(
                    data_dict,
                    inference=True,
                )

            if not isinstance(
                predictions,
                Mapping,
            ):
                raise RuntimeError(
                    "{} inference output is not "
                    "a mapping.".format(
                        self.model_name
                    )
                )

            if (
                "prob"
                not in predictions
            ):
                raise RuntimeError(
                    "{} inference output does not "
                    "contain 'prob'.".format(
                        self.model_name
                    )
                )

            probability = (
                predictions[
                    "prob"
                ]
            )

            if not torch.is_tensor(
                probability
            ):
                raise RuntimeError(
                    "{} 'prob' output is not "
                    "a tensor.".format(
                        self.model_name
                    )
                )

            if (
                probability.ndim
                != 1
            ):
                raise RuntimeError(
                    "{} 'prob' must be one-dimensional; "
                    "got {}.".format(
                        self.model_name,
                        tuple(
                            probability.shape
                        ),
                    )
                )

            expected_batch = (
                end
                - start
            )

            if (
                probability.shape[0]
                != expected_batch
            ):
                raise RuntimeError(
                    "{} probability batch-size mismatch: "
                    "expected {}, got {}.".format(
                        self.model_name,
                        expected_batch,
                        probability.shape[
                            0
                        ],
                    )
                )

            if not torch.isfinite(
                probability
            ).all():
                raise RuntimeError(
                    "{} produced non-finite "
                    "frame scores.".format(
                        self.model_name
                    )
                )

            if (
                torch.any(
                    probability
                    < 0.0
                )
                or torch.any(
                    probability
                    > 1.0
                )
            ):
                raise RuntimeError(
                    "{} produced a score outside "
                    "[0, 1].".format(
                        self.model_name
                    )
                )

            scores.extend(
                float(
                    value
                )
                for value in (
                    probability
                    .detach()
                    .cpu()
                    .tolist()
                )
            )

        if (
            len(
                scores
            )
            != len(
                aligned_bgr_images
            )
        ):
            raise RuntimeError(
                "Final detector-score count mismatch."
            )

        return scores
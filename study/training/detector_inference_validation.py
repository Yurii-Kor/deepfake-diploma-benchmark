from collections.abc import Mapping
from pathlib import Path
import sys

import torch
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

DETECTOR_CONFIG_PATHS = {
    "xception": (
        TRAINING_ROOT
        / "config"
        / "detector"
        / "study_xception.yaml"
    ),
    "ucf": (
        TRAINING_ROOT
        / "config"
        / "detector"
        / "study_ucf.yaml"
    ),
    "spsl": (
        TRAINING_ROOT
        / "config"
        / "detector"
        / "study_spsl.yaml"
    ),
}


def _load_config(
    detector_path: Path,
):
    with BASE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    with detector_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config.update(
            yaml.safe_load(
                file
            )
        )

    config[
        "cuda"
    ] = False

    config[
        "ddp"
    ] = False

    return config


def _validate_detector(
    model_name: str,
) -> None:
    config = _load_config(
        DETECTOR_CONFIG_PATHS[
            model_name
        ]
    )

    if (
        config[
            "model_name"
        ]
        != model_name
    ):
        raise AssertionError(
            "model_name mismatch: "
            f"{config['model_name']} "
            f"!= {model_name}"
        )

    model_class = DETECTOR[
        model_name
    ]

    model = model_class(
        config
    )

    model.eval()

    batch_size = 2

    data_dict = {
        "image": torch.zeros(
            batch_size,
            3,
            256,
            256,
            dtype=torch.float32,
        ),
        "label": torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        ),
    }

    with torch.no_grad():
        predictions = model(
            data_dict,
            inference=True,
        )

    if not isinstance(
        predictions,
        Mapping,
    ):
        raise AssertionError(
            f"{model_name}: "
            "inference output must be a mapping"
        )

    if "prob" not in predictions:
        raise AssertionError(
            f"{model_name}: "
            "inference output must contain 'prob'"
        )

    probabilities = predictions[
        "prob"
    ]

    if not torch.is_tensor(
        probabilities
    ):
        raise AssertionError(
            f"{model_name}: "
            "'prob' must be a torch.Tensor"
        )

    if probabilities.ndim != 1:
        raise AssertionError(
            f"{model_name}: "
            "'prob' must be one-dimensional; "
            f"got shape {tuple(probabilities.shape)}"
        )

    if probabilities.shape[
        0
    ] != batch_size:
        raise AssertionError(
            f"{model_name}: "
            "'prob' batch dimension mismatch; "
            f"expected {batch_size}, "
            f"got {probabilities.shape[0]}"
        )

    if not torch.isfinite(
        probabilities
    ).all():
        raise AssertionError(
            f"{model_name}: "
            "'prob' contains non-finite values"
        )

    if torch.any(
        probabilities < 0.0
    ) or torch.any(
        probabilities > 1.0
    ):
        raise AssertionError(
            f"{model_name}: "
            "'prob' must contain probabilities "
            "in the [0, 1] interval"
        )

    print(
        "{:<9} inference contract: OK".format(
            model_name.upper()
        )
    )


def main() -> None:
    for model_name in (
        "xception",
        "ucf",
        "spsl",
    ):
        _validate_detector(
            model_name
        )

    print()
    print(
        "DETECTOR INFERENCE VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
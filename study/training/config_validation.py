from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]

BASE_CONFIG_PATH = (
    ROOT
    / "training"
    / "config"
    / "train_config.yaml"
)

STUDY_CONFIG_PATHS = {
    "xception": (
        ROOT
        / "training"
        / "config"
        / "detector"
        / "study_xception.yaml"
    ),
    "ucf": (
        ROOT
        / "training"
        / "config"
        / "detector"
        / "study_ucf.yaml"
    ),
    "spsl": (
        ROOT
        / "training"
        / "config"
        / "detector"
        / "study_spsl.yaml"
    ),
}


def _load_yaml(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def _merged_config(
    detector_path: Path,
):
    base_config = _load_yaml(
        BASE_CONFIG_PATH
    )

    detector_config = _load_yaml(
        detector_path
    )

    config = dict(
        base_config
    )

    config.update(
        detector_config
    )

    return config


def _validate_common(
    model_name: str,
    config: dict,
) -> None:
    assert (
        config[
            "study_controlled_training"
        ]
        is True
    )

    assert (
        config[
            "model_name"
        ]
        == model_name
    )

    assert (
        config[
            "lmdb"
        ]
        is False
    )

    assert (
        config[
            "compression"
        ]
        == "raw"
    )

    assert (
        config[
            "frame_num"
        ][
            "train"
        ]
        == 32
    )

    assert (
        config[
            "frame_num"
        ][
            "test"
        ]
        == 32
    )

    assert (
        config[
            "test_batchSize"
        ]
        == 32
    )

    assert (
        config[
            "nEpochs"
        ]
        == 10
    )

    assert (
        config[
            "start_epoch"
        ]
        == 0
    )

    assert (
        config[
            "manualSeed"
        ]
        == 1024
    )

    assert (
        config[
            "lr_scheduler"
        ]
        is None
    )

    assert (
        config[
            "optimizer"
        ][
            "type"
        ]
        == "adam"
    )

    adam = config[
        "optimizer"
    ][
        "adam"
    ]

    expected_adam = {
        "lr": 0.0002,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0005,
        "amsgrad": False,
    }

    assert (
        adam
        == expected_adam
    )

    assert (
        config[
            "use_data_augmentation"
        ]
        is False
    )

    assert (
        config[
            "dataset_json_folder"
        ]
        == "./study/training/materialized/dataset_json"
    )

    assert (
        config[
            "rgb_dir"
        ]
        == "./study/training/materialized/rgb"
    )

    assert (
        config[
            "metric_scoring"
        ]
        == "video_auc"
    )


def _validate_xception(
    config: dict,
) -> None:
    assert (
        config[
            "train_dataset"
        ]
        == [
            "FaceForensics++"
        ]
    )

    assert (
        config[
            "test_dataset"
        ]
        == [
            "FaceForensics++"
        ]
    )

    assert (
        config[
            "train_batchSize"
        ]
        == 32
    )

    assert (
        config[
            "backbone_config"
        ][
            "inc"
        ]
        == 3
    )

    assert (
        config[
            "loss_func"
        ]
        == "cross_entropy"
    )


def _validate_ucf(
    config: dict,
) -> None:
    assert (
        config[
            "train_dataset"
        ]
        == [
            "FF-F2F",
            "FF-DF",
            "FF-FS",
            "FF-NT",
        ]
    )

    assert (
        config[
            "test_dataset"
        ]
        == [
            "FaceForensics++"
        ]
    )

    assert (
        config[
            "dataset_type"
        ]
        == "pair"
    )

    assert (
        config[
            "train_batchSize"
        ]
        == 16
    )

    assert (
        config[
            "backbone_config"
        ][
            "inc"
        ]
        == 3
    )

    assert (
        config[
            "label_dict"
        ][
            "FF-real"
        ]
        == 0
    )

    assert (
        config[
            "label_dict"
        ][
            "FF-DF"
        ]
        == 1
    )

    assert (
        config[
            "label_dict"
        ][
            "FF-F2F"
        ]
        == 2
    )

    assert (
        config[
            "label_dict"
        ][
            "FF-FS"
        ]
        == 3
    )

    assert (
        config[
            "label_dict"
        ][
            "FF-NT"
        ]
        == 4
    )


def _validate_spsl(
    config: dict,
) -> None:
    assert (
        config[
            "train_dataset"
        ]
        == [
            "FaceForensics++"
        ]
    )

    assert (
        config[
            "test_dataset"
        ]
        == [
            "FaceForensics++"
        ]
    )

    assert (
        config[
            "train_batchSize"
        ]
        == 32
    )

    assert (
        config[
            "backbone_config"
        ][
            "mode"
        ]
        == "original"
    )

    assert (
        config[
            "backbone_config"
        ][
            "inc"
        ]
        == 4
    )

    assert (
        config[
            "loss_func"
        ]
        == "cross_entropy"
    )


def main() -> None:
    validators = {
        "xception": _validate_xception,
        "ucf": _validate_ucf,
        "spsl": _validate_spsl,
    }

    for (
        model_name,
        path,
    ) in (
        STUDY_CONFIG_PATHS.items()
    ):
        config = _merged_config(
            path
        )

        _validate_common(
            model_name=model_name,
            config=config,
        )

        validators[
            model_name
        ](
            config
        )

        print(
            "{:<8} merged config: OK".format(
                model_name.upper()
            )
        )

    print()
    print(
        "STUDY CONFIG VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
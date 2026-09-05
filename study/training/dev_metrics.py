from collections import OrderedDict

import numpy as np
from sklearn import metrics


def _normalize_frame_reference(
    frame_reference,
):
    if isinstance(
        frame_reference,
        (
            list,
            tuple,
        ),
    ):
        raise TypeError(
            "Study DEV expects frame-level image references, "
            "not list-based video entries."
        )

    normalized = str(
        frame_reference
    ).replace(
        "\\",
        "/",
    )

    if "/" not in normalized:
        raise ValueError(
            "Frame reference does not contain a parent video directory: "
            f"{frame_reference}"
        )

    parent_path, filename = (
        normalized.rsplit(
            "/",
            1,
        )
    )

    if (
        not parent_path
        or not filename
    ):
        raise ValueError(
            "Invalid frame reference: "
            f"{frame_reference}"
        )

    return parent_path


def _as_score_array(
    values,
):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError(
                "Expected scalar frame scores."
            )

        array = array[:, 0]

    if array.ndim != 1:
        raise ValueError(
            "Expected one-dimensional frame scores."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "DEV scores contain non-finite values."
        )

    return array


def _as_binary_label_array(
    values,
):
    array = np.asarray(
        values
    ).reshape(
        -1
    )

    binary = []

    for value in array:
        numeric_value = float(
            value
        )

        if numeric_value not in (
            0.0,
            1.0,
        ):
            raise ValueError(
                "Study DEV labels must be binary 0/1. "
                f"Received: {value}"
            )

        binary.append(
            int(
                numeric_value
            )
        )

    return np.asarray(
        binary,
        dtype=np.int64,
    )


def _require_two_classes(
    labels,
    level,
):
    classes = set(
        int(
            value
        )
        for value in labels
    )

    if classes != {
        0,
        1,
    }:
        raise ValueError(
            f"{level} AUC requires both real and fake classes. "
            f"Observed classes: {sorted(classes)}"
        )


def aggregate_frame_scores_to_videos(
    frame_paths,
    frame_scores,
    frame_labels,
):
    scores = _as_score_array(
        frame_scores
    )

    labels = _as_binary_label_array(
        frame_labels
    )

    frame_paths = list(
        frame_paths
    )

    if not (
        len(
            frame_paths
        )
        == len(
            scores
        )
        == len(
            labels
        )
    ):
        raise ValueError(
            "Frame references, scores, and labels "
            "must have identical lengths."
        )

    if len(
        scores
    ) == 0:
        raise ValueError(
            "Cannot aggregate an empty DEV result."
        )

    grouped = OrderedDict()

    for (
        frame_path,
        score,
        label,
    ) in zip(
        frame_paths,
        scores,
        labels,
    ):
        video_id = (
            _normalize_frame_reference(
                frame_path
            )
        )

        if video_id not in grouped:
            grouped[
                video_id
            ] = {
                "label": int(
                    label
                ),
                "scores": [],
            }

        elif (
            grouped[
                video_id
            ][
                "label"
            ]
            != int(
                label
            )
        ):
            raise ValueError(
                "Conflicting labels found inside one video: "
                f"{video_id}"
            )

        grouped[
            video_id
        ][
            "scores"
        ].append(
            float(
                score
            )
        )

    video_ids = []
    video_scores = []
    video_labels = []
    frame_counts = []

    for (
        video_id,
        video_data,
    ) in grouped.items():
        scores_one_video = (
            video_data[
                "scores"
            ]
        )

        if not scores_one_video:
            raise ValueError(
                "Video has no frame scores: "
                f"{video_id}"
            )

        video_ids.append(
            video_id
        )

        video_scores.append(
            float(
                np.mean(
                    scores_one_video
                )
            )
        )

        video_labels.append(
            video_data[
                "label"
            ]
        )

        frame_counts.append(
            len(
                scores_one_video
            )
        )

    video_scores = np.asarray(
        video_scores,
        dtype=np.float64,
    )

    video_labels = np.asarray(
        video_labels,
        dtype=np.int64,
    )

    _require_two_classes(
        video_labels,
        "Video-level",
    )

    return {
        "video_ids": tuple(
            video_ids
        ),
        "video_scores": video_scores,
        "video_labels": video_labels,
        "frame_counts": tuple(
            frame_counts
        ),
    }


def compute_dev_metrics(
    frame_paths,
    frame_scores,
    frame_labels,
):
    scores = _as_score_array(
        frame_scores
    )

    labels = _as_binary_label_array(
        frame_labels
    )

    if len(
        scores
    ) != len(
        labels
    ):
        raise ValueError(
            "DEV score and label counts do not match."
        )

    _require_two_classes(
        labels,
        "Frame-level",
    )

    frame_auc = float(
        metrics.roc_auc_score(
            labels,
            scores,
        )
    )

    aggregation = (
        aggregate_frame_scores_to_videos(
            frame_paths=frame_paths,
            frame_scores=scores,
            frame_labels=labels,
        )
    )

    video_scores = aggregation[
        "video_scores"
    ]

    video_labels = aggregation[
        "video_labels"
    ]

    video_auc = float(
        metrics.roc_auc_score(
            video_labels,
            video_scores,
        )
    )

    frame_counts = aggregation[
        "frame_counts"
    ]

    return {
        "frame_auc": frame_auc,
        "video_auc": video_auc,
        "frame_count": int(
            len(
                scores
            )
        ),
        "video_count": int(
            len(
                video_scores
            )
        ),
        "real_video_count": int(
            np.sum(
                video_labels
                == 0
            )
        ),
        "fake_video_count": int(
            np.sum(
                video_labels
                == 1
            )
        ),
        "min_frames_per_video": int(
            min(
                frame_counts
            )
        ),
        "max_frames_per_video": int(
            max(
                frame_counts
            )
        ),
    }
import math

from study.training.dev_metrics import (
    aggregate_frame_scores_to_videos,
    compute_dev_metrics,
)


def _expect_failure(
    function,
    expected_exception,
):
    try:
        function()
    except expected_exception:
        return

    raise AssertionError(
        "Expected failure was not raised."
    )


def validate_nominal_case():
    frame_paths = [
        "FaceForensics++/original/raw/frames/000/000.png",
        "FaceForensics++/original/raw/frames/000/001.png",
        "FaceForensics++/original/raw/frames/001/000.png",
        "FaceForensics++/original/raw/frames/001/001.png",

        # Same base-video directory name deliberately appears
        # under two different manipulation methods.
        "FaceForensics++/Deepfakes/raw/frames/000_003/000.png",
        "FaceForensics++/Deepfakes/raw/frames/000_003/001.png",
        "FaceForensics++/Face2Face/raw/frames/000_003/000.png",
        "FaceForensics++/Face2Face/raw/frames/000_003/001.png",
    ]

    frame_scores = [
        0.10,
        0.20,
        0.20,
        0.30,
        0.80,
        0.90,
        0.70,
        0.80,
    ]

    frame_labels = [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]

    aggregation = (
        aggregate_frame_scores_to_videos(
            frame_paths=frame_paths,
            frame_scores=frame_scores,
            frame_labels=frame_labels,
        )
    )

    assert (
        len(
            aggregation[
                "video_ids"
            ]
        )
        == 4
    )

    assert (
        len(
            set(
                aggregation[
                    "video_ids"
                ]
            )
        )
        == 4
    )

    metrics = compute_dev_metrics(
        frame_paths=frame_paths,
        frame_scores=frame_scores,
        frame_labels=frame_labels,
    )

    assert math.isclose(
        metrics[
            "frame_auc"
        ],
        1.0,
    )

    assert math.isclose(
        metrics[
            "video_auc"
        ],
        1.0,
    )

    assert (
        metrics[
            "frame_count"
        ]
        == 8
    )

    assert (
        metrics[
            "video_count"
        ]
        == 4
    )

    assert (
        metrics[
            "real_video_count"
        ]
        == 2
    )

    assert (
        metrics[
            "fake_video_count"
        ]
        == 2
    )

    assert (
        metrics[
            "min_frames_per_video"
        ]
        == 2
    )

    assert (
        metrics[
            "max_frames_per_video"
        ]
        == 2
    )

    print(
        "DEV VIDEO AGGREGATION"
    )

    print(
        "  frame-level AUC:          1.0"
    )

    print(
        "  video-level AUC:          1.0"
    )

    print(
        "  distinct videos:          4"
    )

    print(
        "  method basename collision: prevented"
    )


def validate_negative_cases():
    _expect_failure(
        lambda: compute_dev_metrics(
            frame_paths=[
                "a/video/0.png",
            ],
            frame_scores=[
                0.1,
                0.2,
            ],
            frame_labels=[
                0,
            ],
        ),
        ValueError,
    )

    _expect_failure(
        lambda: compute_dev_metrics(
            frame_paths=[
                "a/real/0.png",
                "a/fake/0.png",
            ],
            frame_scores=[
                0.1,
                float(
                    "nan"
                ),
            ],
            frame_labels=[
                0,
                1,
            ],
        ),
        ValueError,
    )

    _expect_failure(
        lambda: compute_dev_metrics(
            frame_paths=[
                "a/video1/0.png",
                "a/video2/0.png",
            ],
            frame_scores=[
                0.1,
                0.2,
            ],
            frame_labels=[
                0,
                0,
            ],
        ),
        ValueError,
    )

    _expect_failure(
        lambda: aggregate_frame_scores_to_videos(
            frame_paths=[
                "same/video/0.png",
                "same/video/1.png",
            ],
            frame_scores=[
                0.1,
                0.9,
            ],
            frame_labels=[
                0,
                1,
            ],
        ),
        ValueError,
    )

    _expect_failure(
        lambda: aggregate_frame_scores_to_videos(
            frame_paths=[
                [
                    "video/0.png",
                    "video/1.png",
                ],
                [
                    "video2/0.png",
                    "video2/1.png",
                ],
            ],
            frame_scores=[
                0.1,
                0.9,
            ],
            frame_labels=[
                0,
                1,
            ],
        ),
        TypeError,
    )

    print()
    print(
        "NEGATIVE / INVARIANT TESTS"
    )

    print(
        "  mismatched lengths:       rejected"
    )

    print(
        "  non-finite scores:        rejected"
    )

    print(
        "  single-class DEV:         rejected"
    )

    print(
        "  conflicting video labels: rejected"
    )

    print(
        "  video-list input:         rejected"
    )


def main():
    validate_nominal_case()
    validate_negative_cases()

    print()
    print(
        "DEV METRICS VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
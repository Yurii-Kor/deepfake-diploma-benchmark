import numpy as np

from study.materialization.face_alignment import (
    ALIGNMENT_METHOD,
    LANDMARK_INDICES,
    OUTPUT_SIZE,
    align_face_bgr,
    destination_points,
    estimate_similarity_matrix,
    extract_five_keypoints,
    rectangle_to_bbox,
    select_largest_face,
)


class FakeRectangle:
    def __init__(
        self,
        left,
        top,
        right,
        bottom,
    ):
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self):
        return self._left

    def top(self):
        return self._top

    def right(self):
        return self._right

    def bottom(self):
        return self._bottom

    def width(self):
        return (
            self._right
            - self._left
        )

    def height(self):
        return (
            self._bottom
            - self._top
        )


class FakePart:
    def __init__(
        self,
        x,
        y,
    ):
        self.x = x
        self.y = y


class FakeShape:
    def __init__(
        self,
        points,
    ):
        self._points = points

    def part(
        self,
        index,
    ):
        return self._points[
            index
        ]


class FakePredictor:
    def __init__(
        self,
        keypoints,
    ):
        self._points = {}

        for index in range(
            81
        ):
            self._points[
                index
            ] = FakePart(
                0,
                0,
            )

        for index, point in zip(
            LANDMARK_INDICES,
            keypoints,
        ):
            self._points[
                index
            ] = FakePart(
                int(
                    round(
                        float(
                            point[0]
                        )
                    )
                ),
                int(
                    round(
                        float(
                            point[1]
                        )
                    )
                ),
            )

    def __call__(
        self,
        image,
        face,
    ):
        return FakeShape(
            self._points
        )


def validate_destination_geometry():
    actual = destination_points()

    expected = np.array(
        [
            [96.86963, 120.43306],
            [158.82513, 120.09037],
            [128.04431, 155.66875],
            [102.59218, 191.93933],
            [153.89873, 191.65555],
        ],
        dtype=np.float32,
    )

    if actual.shape != (
        5,
        2,
    ):
        raise AssertionError(
            "unexpected destination shape"
        )

    if not np.allclose(
        actual,
        expected,
        atol=1e-4,
        rtol=0.0,
    ):
        raise AssertionError(
            "destination geometry changed"
        )


def validate_largest_face():
    small = FakeRectangle(
        0,
        0,
        10,
        10,
    )

    large = FakeRectangle(
        5,
        6,
        35,
        26,
    )

    selected = select_largest_face(
        [
            small,
            large,
        ]
    )

    if selected is not large:
        raise AssertionError(
            "largest face was not selected"
        )

    if rectangle_to_bbox(
        selected
    ) != (
        5,
        6,
        35,
        26,
    ):
        raise AssertionError(
            "bbox conversion changed"
        )


def validate_landmark_indices():
    expected = np.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
            [50.0, 51.0],
        ],
        dtype=np.float32,
    )

    predictor = FakePredictor(
        expected
    )

    image = np.zeros(
        (
            64,
            64,
            3,
        ),
        dtype=np.uint8,
    )

    face = FakeRectangle(
        0,
        0,
        63,
        63,
    )

    actual = extract_five_keypoints(
        image_rgb=image,
        face=face,
        predictor=predictor,
    )

    if not np.array_equal(
        actual,
        expected,
    ):
        raise AssertionError(
            "five-point landmark selection changed"
        )


def validate_identity_transform():
    points = destination_points()

    matrix = estimate_similarity_matrix(
        points
    )

    if matrix is None:
        raise AssertionError(
            "identity transform was not estimated"
        )

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    if not np.allclose(
        matrix,
        expected,
        atol=1e-5,
        rtol=0.0,
    ):
        raise AssertionError(
            "unexpected identity transform"
        )


def validate_no_face_failure():
    frame = np.zeros(
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    def detector(
        image,
        upsample,
    ):
        if upsample != 1:
            raise AssertionError(
                "detector upsample changed"
            )

        return []

    result = align_face_bgr(
        frame_bgr=frame,
        face_detector=detector,
        predictor=None,
    )

    if result.ok:
        raise AssertionError(
            "no-face frame unexpectedly succeeded"
        )

    if result.failure_stage != "face_detection":
        raise AssertionError(
            "unexpected no-face failure stage"
        )

    if result.face_count != 0:
        raise AssertionError(
            "unexpected no-face count"
        )


def validate_success_contract():
    frame = np.zeros(
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    target_points = destination_points()

    face = FakeRectangle(
        20,
        20,
        220,
        220,
    )

    def detector(
        image,
        upsample,
    ):
        if upsample != 1:
            raise AssertionError(
                "detector upsample changed"
            )

        return [
            face
        ]

    predictor = FakePredictor(
        target_points
    )

    result = align_face_bgr(
        frame_bgr=frame,
        face_detector=detector,
        predictor=predictor,
    )

    if not result.ok:
        raise AssertionError(
            "synthetic valid face failed: {}".format(
                result.failure_reason
            )
        )

    if result.aligned_bgr is None:
        raise AssertionError(
            "aligned face is missing"
        )

    if result.aligned_bgr.shape != (
        OUTPUT_SIZE,
        OUTPUT_SIZE,
        3,
    ):
        raise AssertionError(
            "unexpected aligned output shape"
        )

    if result.face_count != 1:
        raise AssertionError(
            "unexpected detected-face count"
        )

    if result.bbox != (
        20,
        20,
        220,
        220,
    ):
        raise AssertionError(
            "unexpected selected bbox"
        )

    if result.keypoints is None:
        raise AssertionError(
            "alignment keypoints are missing"
        )

    if result.affine_matrix is None:
        raise AssertionError(
            "affine matrix is missing"
        )


def main():
    validate_destination_geometry()
    validate_largest_face()
    validate_landmark_indices()
    validate_identity_transform()
    validate_no_face_failure()
    validate_success_contract()

    print(
        "FACE ALIGNMENT VALIDATION"
    )
    print(
        "  method:                    {}".format(
            ALIGNMENT_METHOD
        )
    )
    print(
        "  output size:               {}x{}".format(
            OUTPUT_SIZE,
            OUTPUT_SIZE,
        )
    )
    print(
        "  landmark indices:          {}".format(
            LANDMARK_INDICES
        )
    )
    print(
        "  largest-face selection:    preserved"
    )
    print(
        "  similarity geometry:       preserved"
    )
    print(
        "  no-face failure:           explicit"
    )
    print(
        "  synthetic success path:    passed"
    )
    print()
    print(
        "FACE ALIGNMENT VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()
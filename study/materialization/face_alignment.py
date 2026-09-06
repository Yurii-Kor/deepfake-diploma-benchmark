from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage import transform as trans


OUTPUT_SIZE = 256
ALIGNMENT_SCALE = 1.3
DETECTOR_UPSAMPLE = 1

LANDMARK_INDICES = (
    37,
    44,
    30,
    49,
    55,
)

REFERENCE_SIZE = 112

REFERENCE_POINTS = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)

ALIGNMENT_METHOD = "dfb_dlib_5pt_similarity_v1"


BBox = Tuple[int, int, int, int]


@dataclass
class FaceAlignmentResult:
    ok: bool
    aligned_bgr: Optional[np.ndarray]
    face_count: int
    bbox: Optional[BBox]
    keypoints: Optional[np.ndarray]
    affine_matrix: Optional[np.ndarray]
    failure_stage: Optional[str]
    failure_reason: Optional[str]


def load_dlib_face_components(
    predictor_path: Path,
):
    import dlib

    predictor_path = Path(
        predictor_path
    ).resolve()

    if not predictor_path.is_file():
        raise FileNotFoundError(
            "dlib predictor not found: {}".format(
                predictor_path
            )
        )

    face_detector = (
        dlib.get_frontal_face_detector()
    )

    predictor = dlib.shape_predictor(
        str(
            predictor_path
        )
    )

    return (
        face_detector,
        predictor,
    )


def destination_points(
    output_size: int = OUTPUT_SIZE,
    scale: float = ALIGNMENT_SCALE,
) -> np.ndarray:
    if output_size <= 0:
        raise ValueError(
            "output_size must be positive"
        )

    if scale <= 0:
        raise ValueError(
            "scale must be positive"
        )

    dst = REFERENCE_POINTS.copy()

    # Preserve the original DeepfakeBench
    # alignment geometry.
    dst[:, 0] += 8.0

    dst[:, 0] = (
        dst[:, 0]
        * output_size
        / REFERENCE_SIZE
    )

    dst[:, 1] = (
        dst[:, 1]
        * output_size
        / REFERENCE_SIZE
    )

    margin_rate = (
        scale
        - 1.0
    )

    x_margin = (
        output_size
        * margin_rate
        / 2.0
    )

    y_margin = (
        output_size
        * margin_rate
        / 2.0
    )

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= (
        output_size
        / (
            output_size
            + 2.0 * x_margin
        )
    )

    dst[:, 1] *= (
        output_size
        / (
            output_size
            + 2.0 * y_margin
        )
    )

    return dst.astype(
        np.float32
    )


def rectangle_to_bbox(
    face: Any,
) -> BBox:
    return (
        int(
            face.left()
        ),
        int(
            face.top()
        ),
        int(
            face.right()
        ),
        int(
            face.bottom()
        ),
    )


def select_largest_face(
    faces: Sequence[Any],
):
    faces = list(
        faces
    )

    if not faces:
        return None

    return max(
        faces,
        key=lambda rect: (
            rect.width()
            * rect.height()
        ),
    )


def extract_five_keypoints(
    image_rgb: np.ndarray,
    face: Any,
    predictor: Any,
) -> np.ndarray:
    shape = predictor(
        image_rgb,
        face,
    )

    points = []

    for index in LANDMARK_INDICES:
        part = shape.part(
            index
        )

        points.append(
            [
                float(
                    part.x
                ),
                float(
                    part.y
                ),
            ]
        )

    keypoints = np.asarray(
        points,
        dtype=np.float32,
    )

    if keypoints.shape != (
        5,
        2,
    ):
        raise ValueError(
            "expected five 2D alignment keypoints"
        )

    if not np.isfinite(
        keypoints
    ).all():
        raise ValueError(
            "alignment keypoints contain "
            "non-finite values"
        )

    return keypoints


def estimate_similarity_matrix(
    keypoints: np.ndarray,
    output_size: int = OUTPUT_SIZE,
    scale: float = ALIGNMENT_SCALE,
) -> Optional[np.ndarray]:
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.shape != (
        5,
        2,
    ):
        raise ValueError(
            "keypoints must have shape (5, 2)"
        )

    if not np.isfinite(
        keypoints
    ).all():
        raise ValueError(
            "keypoints contain non-finite values"
        )

    dst = destination_points(
        output_size=output_size,
        scale=scale,
    )

    transform = (
        trans.SimilarityTransform()
    )

    estimated = transform.estimate(
        keypoints,
        dst,
    )

    if not estimated:
        return None

    matrix = np.asarray(
        transform.params[
            0:2,
            :
        ],
        dtype=np.float64,
    )

    if matrix.shape != (
        2,
        3,
    ):
        return None

    if not np.isfinite(
        matrix
    ).all():
        return None

    return matrix


def warp_aligned_face(
    image_rgb: np.ndarray,
    affine_matrix: np.ndarray,
    output_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    matrix = np.asarray(
        affine_matrix,
        dtype=np.float64,
    )

    if matrix.shape != (
        2,
        3,
    ):
        raise ValueError(
            "affine_matrix must have shape (2, 3)"
        )

    aligned_rgb = cv2.warpAffine(
        image_rgb,
        matrix,
        (
            output_size,
            output_size,
        ),
    )

    # The original DeepfakeBench preprocessing
    # performs this resize after warpAffine.
    aligned_rgb = cv2.resize(
        aligned_rgb,
        (
            output_size,
            output_size,
        ),
    )

    if aligned_rgb.shape != (
        output_size,
        output_size,
        3,
    ):
        raise RuntimeError(
            "unexpected aligned face shape: {}".format(
                aligned_rgb.shape
            )
        )

    return aligned_rgb


def _failure(
    face_count: int,
    bbox: Optional[BBox],
    keypoints: Optional[np.ndarray],
    affine_matrix: Optional[np.ndarray],
    stage: str,
    reason: str,
) -> FaceAlignmentResult:
    return FaceAlignmentResult(
        ok=False,
        aligned_bgr=None,
        face_count=face_count,
        bbox=bbox,
        keypoints=keypoints,
        affine_matrix=affine_matrix,
        failure_stage=stage,
        failure_reason=reason,
    )


def align_face_bgr(
    frame_bgr: np.ndarray,
    face_detector: Any,
    predictor: Any,
    output_size: int = OUTPUT_SIZE,
) -> FaceAlignmentResult:
    if not isinstance(
        frame_bgr,
        np.ndarray,
    ):
        raise TypeError(
            "frame_bgr must be a NumPy array"
        )

    if (
        frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError(
            "frame_bgr must have shape HxWx3"
        )

    if frame_bgr.dtype != np.uint8:
        raise ValueError(
            "frame_bgr must have dtype uint8"
        )

    if output_size <= 0:
        raise ValueError(
            "output_size must be positive"
        )

    image_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    try:
        faces = list(
            face_detector(
                image_rgb,
                DETECTOR_UPSAMPLE,
            )
        )
    except Exception as exc:
        return _failure(
            face_count=0,
            bbox=None,
            keypoints=None,
            affine_matrix=None,
            stage="face_detection",
            reason=str(
                exc
            ),
        )

    face_count = len(
        faces
    )

    if face_count == 0:
        return _failure(
            face_count=0,
            bbox=None,
            keypoints=None,
            affine_matrix=None,
            stage="face_detection",
            reason="no face detected",
        )

    face = select_largest_face(
        faces
    )

    bbox = rectangle_to_bbox(
        face
    )

    try:
        keypoints = extract_five_keypoints(
            image_rgb=image_rgb,
            face=face,
            predictor=predictor,
        )
    except Exception as exc:
        return _failure(
            face_count=face_count,
            bbox=bbox,
            keypoints=None,
            affine_matrix=None,
            stage="landmarks",
            reason=str(
                exc
            ),
        )

    try:
        affine_matrix = (
            estimate_similarity_matrix(
                keypoints=keypoints,
                output_size=output_size,
                scale=ALIGNMENT_SCALE,
            )
        )
    except Exception as exc:
        return _failure(
            face_count=face_count,
            bbox=bbox,
            keypoints=keypoints,
            affine_matrix=None,
            stage="alignment_transform",
            reason=str(
                exc
            ),
        )

    if affine_matrix is None:
        return _failure(
            face_count=face_count,
            bbox=bbox,
            keypoints=keypoints,
            affine_matrix=None,
            stage="alignment_transform",
            reason=(
                "similarity transform "
                "could not be estimated"
            ),
        )

    try:
        aligned_rgb = warp_aligned_face(
            image_rgb=image_rgb,
            affine_matrix=affine_matrix,
            output_size=output_size,
        )
    except Exception as exc:
        return _failure(
            face_count=face_count,
            bbox=bbox,
            keypoints=keypoints,
            affine_matrix=affine_matrix,
            stage="alignment_warp",
            reason=str(
                exc
            ),
        )

    aligned_bgr = cv2.cvtColor(
        aligned_rgb,
        cv2.COLOR_RGB2BGR,
    )

    return FaceAlignmentResult(
        ok=True,
        aligned_bgr=aligned_bgr,
        face_count=face_count,
        bbox=bbox,
        keypoints=keypoints,
        affine_matrix=affine_matrix,
        failure_stage=None,
        failure_reason=None,
    )
"""
Factory for creating face detectors from config.
"""

from __future__ import annotations

from src.config import FaceBlurConfig
from src.detectors.base import FaceDetector


def make_detector(config: FaceBlurConfig) -> FaceDetector:
    """
    Instantiate and return the appropriate :class:`FaceDetector` based on
    ``config.detector_type``.

    Supported values
    ----------------
    * ``"retinaface"``  — insightface RetinaFace (default, recommended)
    * ``"mediapipe"``   — MediaPipe Face Detection (fallback)

    When ``config.use_tiled_detection`` is True the detector is wrapped in a
    :class:`TiledDetector` that runs detection on overlapping tiles of each
    frame, improving recall for small/background faces.

    Raises
    ------
    ValueError
        If ``config.detector_type`` is not a known type.
    """
    dtype = config.detector_type.lower()

    if dtype == "retinaface":
        from src.detectors.retinaface_detector import RetinaFaceDetector

        detector: FaceDetector = RetinaFaceDetector(
            confidence_threshold=config.detector_confidence_threshold,
            min_face_size=config.min_face_size,
        )

    elif dtype == "mediapipe":
        from src.detectors.mediapipe_detector import MediaPipeDetector

        detector = MediaPipeDetector(
            min_detection_confidence=config.detector_confidence_threshold,
            min_face_size=config.min_face_size,
        )

    else:
        raise ValueError(
            f"Unknown detector_type {config.detector_type!r}. "
            f"Choose 'retinaface' or 'mediapipe'."
        )

    if config.use_tiled_detection:
        from src.detectors.tiled_detector import TiledDetector

        return TiledDetector(
            base_detector=detector,
            tile_size=config.tile_size,
            overlap=config.tile_overlap,
            nms_iou_threshold=config.tile_nms_threshold,
        )

    return detector

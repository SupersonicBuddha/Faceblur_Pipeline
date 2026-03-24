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

    Raises
    ------
    ValueError
        If ``config.detector_type`` is not a known type.
    """
    dtype = config.detector_type.lower()

    if dtype == "retinaface":
        from src.detectors.retinaface_detector import RetinaFaceDetector

        return RetinaFaceDetector(
            confidence_threshold=config.detector_confidence_threshold,
            min_face_size=config.min_face_size,
        )

    if dtype == "mediapipe":
        from src.detectors.mediapipe_detector import MediaPipeDetector

        return MediaPipeDetector(
            min_detection_confidence=config.detector_confidence_threshold,
            min_face_size=config.min_face_size,
        )

    raise ValueError(
        f"Unknown detector_type {config.detector_type!r}. "
        f"Choose 'retinaface' or 'mediapipe'."
    )

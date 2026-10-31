"""MediaPipe Tasks Hand Landmarker adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import VisionPilotConfig
from .landmarks import HandObservation, extract_hand_observation


class HandLandmarkDetector:
    def __init__(self, model_path: Path, config: VisionPilotConfig) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        if not model_path.is_file():
            raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}")

        self._mp = mp
        self._config = config
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.detection_confidence,
            min_hand_presence_confidence=config.hand_presence_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(
        self, bgr_frame: Any, timestamp_ms: int, *, mirrored_input: bool = False
    ) -> HandObservation | None:
        if self._landmarker is None:
            raise RuntimeError("Detector is closed")

        cv2 = self._mp
        import cv2 as cv2_module

        rgb_frame = cv2_module.cvtColor(bgr_frame, cv2_module.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return extract_hand_observation(result, mirrored_input=mirrored_input)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

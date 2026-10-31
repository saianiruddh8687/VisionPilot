"""OpenCV webcam capture with Windows backend fallback."""

from __future__ import annotations

from typing import Any


class CameraError(RuntimeError):
    """Raised when the requested webcam cannot be opened or read."""


class CameraStream:
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        import cv2

        self._cv2 = cv2
        self._capture: Any = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise CameraError(
                f"Could not open camera index {camera_index}. "
                "Check camera permissions or try --camera 1."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Any:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Camera opened but a video frame could not be read.")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

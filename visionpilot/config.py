"""Validated configuration for VisionPilot's perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


@dataclass(frozen=True, slots=True)
class VisionPilotConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_hands: int = 1
    detection_confidence: float = 0.55
    hand_presence_confidence: float = 0.55
    tracking_confidence: float = 0.50
    mirror_preview: bool = True

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index must be >= 0")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.max_hands != 1:
            raise ValueError("Phase 1 supports exactly one hand")
        for name in (
            "detection_confidence",
            "hand_presence_confidence",
            "tracking_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

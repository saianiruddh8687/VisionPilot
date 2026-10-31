"""Small, dependency-light structures and drawing helpers for hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Landmark:
    """A normalized hand landmark; x/y are expected to be in the 0..1 range."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    """The first detected hand and its handedness label, when available."""

    landmarks: tuple[Landmark, ...]
    handedness: str = "Unknown"
    handedness_score: float | None = None


def extract_hand_observation(result: Any, *, mirrored_input: bool = False) -> HandObservation | None:
    """Convert a MediaPipe result to a stable app-level structure.

    The detector is intentionally configured for one hand. If no hand is
    detected, this function returns None. Mirroring the input image reverses
    MediaPipe's left/right label, so swap it back for the physical hand label.
    """

    hands = getattr(result, "hand_landmarks", None) or []
    if not hands:
        return None

    first_hand = hands[0]
    points = tuple(
        Landmark(x=float(point.x), y=float(point.y), z=float(point.z))
        for point in first_hand
    )
    if len(points) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(points)}")

    label = "Unknown"
    score: float | None = None
    handedness_groups = getattr(result, "handedness", None) or []
    if handedness_groups and handedness_groups[0]:
        category = handedness_groups[0][0]
        label = getattr(category, "category_name", None) or getattr(
            category, "categoryName", None
        ) or "Unknown"
        raw_score = getattr(category, "score", None)
        if raw_score is not None:
            score = float(raw_score)

    if mirrored_input:
        if label == "Left":
            label = "Right"
        elif label == "Right":
            label = "Left"

    return HandObservation(
        landmarks=points,
        handedness=str(label),
        handedness_score=score,
    )


# MediaPipe's 21-point hand topology, represented as index pairs.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def draw_hand_observation(frame: Any, observation: HandObservation) -> Any:
    """Draw the hand skeleton onto an OpenCV BGR frame and return the frame."""

    import cv2  # Imported only when rendering is requested.

    height, width = frame.shape[:2]
    pixel_points = [
        (int(round(point.x * width)), int(round(point.y * height)))
        for point in observation.landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixel_points[start], pixel_points[end], (80, 220, 120), 2)

    for index, (x, y) in enumerate(pixel_points):
        radius = 6 if index in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (x, y), radius, (30, 170, 255), -1)
        cv2.circle(frame, (x, y), radius, (20, 20, 20), 1)

    wrist_x, wrist_y = pixel_points[0]
    cv2.putText(
        frame,
        f"{observation.handedness} hand",
        (max(8, wrist_x - 25), max(28, wrist_y - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 220, 120),
        2,
        cv2.LINE_AA,
    )
    return frame

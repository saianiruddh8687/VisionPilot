"""Index-fingertip tracking, motion estimation, and loss recovery.

This module has no camera, MediaPipe, or desktop-automation dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

from .filters import PointSmoother
from .landmarks import HandObservation

INDEX_FINGER_TIP = 8


class TrackingPhase(str, Enum):
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"


class MovementDirection(str, Enum):
    STILL = "STILL"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


@dataclass(frozen=True, slots=True)
class Point2D:
    """Normalized camera coordinate; origin is the top-left corner."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Filter, direction dead-zone, and temporary-loss settings."""

    min_cutoff: float = 1.2
    beta: float = 0.035
    derivative_cutoff: float = 1.0
    movement_threshold: float = 0.055
    dropout_grace_period: float = 0.35

    def __post_init__(self) -> None:
        values = (
            self.min_cutoff,
            self.beta,
            self.derivative_cutoff,
            self.movement_threshold,
            self.dropout_grace_period,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("tracking configuration values must be finite")
        if self.min_cutoff <= 0 or self.derivative_cutoff <= 0:
            raise ValueError("filter cutoffs must be positive")
        if self.beta < 0 or self.movement_threshold < 0 or self.dropout_grace_period < 0:
            raise ValueError("beta, movement threshold, and grace period cannot be negative")


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    """One frame's tracking result. Live coordinates exist only when detected."""

    phase: TrackingPhase
    hand_detected: bool
    index_tip: Point2D | None
    raw_index_tip: Point2D | None
    last_known_position: Point2D | None
    velocity: Point2D
    speed: float
    direction: MovementDirection
    handedness: str
    loss_duration: float
    frames_since_seen: int
    timestamp: float

    @property
    def action_ready(self) -> bool:
        """A live target is available only for a detection in this frame."""

        return self.phase is TrackingPhase.TRACKING and self.index_tip is not None


class HandTracker:
    """Track landmark 8, smooth coordinates, estimate motion, and recover from loss.

    On a missed frame, index_tip becomes None immediately. The last-known
    coordinate is diagnostic only and is never marked action-ready. A short
    dropout preserves filter state; prolonged loss resets it before reacquiring.
    """

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._smoother = PointSmoother(
            self.config.min_cutoff,
            self.config.beta,
            self.config.derivative_cutoff,
        )
        self._phase = TrackingPhase.SEARCHING
        self._last_update_timestamp: float | None = None
        self._last_seen_timestamp: float | None = None
        self._last_position_timestamp: float | None = None
        self._last_known_position: Point2D | None = None
        self._last_handedness = "Unknown"
        self._frames_since_seen = 0

    @staticmethod
    def _clamp(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("landmark coordinates must be finite")
        return min(1.0, max(0.0, value))

    def _direction(self, velocity: Point2D) -> MovementDirection:
        threshold = self.config.movement_threshold
        horizontal = "RIGHT" if velocity.x > threshold else (
            "LEFT" if velocity.x < -threshold else ""
        )
        vertical = "DOWN" if velocity.y > threshold else (
            "UP" if velocity.y < -threshold else ""
        )
        if horizontal and vertical:
            return MovementDirection[f"{vertical}_{horizontal}"]
        if vertical:
            return MovementDirection[vertical]
        if horizontal:
            return MovementDirection[horizontal]
        return MovementDirection.STILL

    def update(
        self,
        observation: HandObservation | None,
        timestamp: float | None = None,
    ) -> TrackingUpdate:
        """Process one camera frame using a monotonic timestamp in seconds."""

        now = time.perf_counter() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._last_update_timestamp is not None and now <= self._last_update_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_update_timestamp = now

        if observation is None:
            self._frames_since_seen += 1
            if self._last_seen_timestamp is None:
                phase = TrackingPhase.SEARCHING
                loss_duration = 0.0
            else:
                loss_duration = max(0.0, now - self._last_seen_timestamp)
                if loss_duration <= self.config.dropout_grace_period:
                    phase = TrackingPhase.OCCLUDED
                else:
                    phase = TrackingPhase.LOST
                    if self._phase is not TrackingPhase.LOST:
                        self._smoother.reset()
                        self._last_position_timestamp = None
            self._phase = phase
            return TrackingUpdate(
                phase=phase,
                hand_detected=False,
                index_tip=None,
                raw_index_tip=None,
                last_known_position=self._last_known_position,
                velocity=Point2D(0.0, 0.0),
                speed=0.0,
                direction=MovementDirection.STILL,
                handedness=self._last_handedness,
                loss_duration=loss_duration,
                frames_since_seen=self._frames_since_seen,
                timestamp=now,
            )

        if len(observation.landmarks) <= INDEX_FINGER_TIP:
            raise ValueError("observation is missing index fingertip landmark 8")

        tip = observation.landmarks[INDEX_FINGER_TIP]
        raw = Point2D(self._clamp(float(tip.x)), self._clamp(float(tip.y)))
        reacquiring = (
            self._last_seen_timestamp is None
            or self._phase in (TrackingPhase.SEARCHING, TrackingPhase.LOST)
        )

        if reacquiring:
            self._smoother.reset()
            filtered_x, filtered_y = self._smoother.filter(raw.x, raw.y, now)
            velocity = Point2D(0.0, 0.0)
        else:
            assert self._last_known_position is not None
            assert self._last_position_timestamp is not None
            dt = now - self._last_position_timestamp
            if dt <= 0:
                raise ValueError("detected-frame timestamps must be strictly increasing")
            filtered_x, filtered_y = self._smoother.filter(raw.x, raw.y, now)
            velocity = Point2D(
                (filtered_x - self._last_known_position.x) / dt,
                (filtered_y - self._last_known_position.y) / dt,
            )

        filtered = Point2D(self._clamp(filtered_x), self._clamp(filtered_y))
        self._phase = TrackingPhase.TRACKING
        self._last_seen_timestamp = now
        self._last_position_timestamp = now
        self._last_known_position = filtered
        self._last_handedness = observation.handedness
        self._frames_since_seen = 0
        speed = math.hypot(velocity.x, velocity.y)

        return TrackingUpdate(
            phase=TrackingPhase.TRACKING,
            hand_detected=True,
            index_tip=filtered,
            raw_index_tip=raw,
            last_known_position=filtered,
            velocity=velocity,
            speed=speed,
            direction=self._direction(velocity),
            handedness=observation.handedness,
            loss_duration=0.0,
            frames_since_seen=0,
            timestamp=now,
        )

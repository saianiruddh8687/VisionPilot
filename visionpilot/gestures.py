"""Camera-independent geometric gesture classifier with debouncing and loss safety."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

from .landmarks import HandObservation


class Gesture(str, Enum):
    NONE = "NONE"
    PINCH = "PINCH"
    OPEN_PALM = "OPEN_PALM"
    TWO_FINGER = "TWO_FINGER"
    FIST = "FIST"
    INDEX_POINT = "INDEX_POINT"


class GestureEvent(str, Enum):
    NONE = "NONE"
    STARTED = "STARTED"
    ENDED = "ENDED"
    CHANGED = "CHANGED"


@dataclass(frozen=True, slots=True)
class GestureConfig:
    pinch_enter_ratio: float = 0.30
    pinch_release_ratio: float = 0.44
    finger_extension_angle: float = 155.0
    thumb_extension_angle: float = 145.0
    thumb_abduction_ratio: float = 0.45
    stable_frames: int = 3
    min_stable_seconds: float = 0.06
    scroll_deadzone: float = 0.003
    max_scroll_delta: float = 0.12

    def __post_init__(self) -> None:
        values = (
            self.pinch_enter_ratio, self.pinch_release_ratio,
            self.finger_extension_angle, self.thumb_extension_angle,
            self.thumb_abduction_ratio, self.min_stable_seconds,
            self.scroll_deadzone, self.max_scroll_delta,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gesture configuration values must be finite")
        if not 0 < self.pinch_enter_ratio < self.pinch_release_ratio:
            raise ValueError("pinch thresholds must satisfy 0 < enter < release")
        if not 0 < self.finger_extension_angle <= 180:
            raise ValueError("finger_extension_angle must be in (0, 180]")
        if not 0 < self.thumb_extension_angle <= 180:
            raise ValueError("thumb_extension_angle must be in (0, 180]")
        if self.thumb_abduction_ratio < 0 or self.stable_frames < 1:
            raise ValueError("thumb_abduction_ratio must be >= 0 and stable_frames >= 1")
        if self.min_stable_seconds < 0 or self.scroll_deadzone < 0 or self.max_scroll_delta <= 0:
            raise ValueError("invalid debounce or scroll configuration")


@dataclass(frozen=True, slots=True)
class GestureUpdate:
    """Stable gesture state and an edge event for one frame."""
    gesture: Gesture
    candidate: Gesture
    event: GestureEvent
    previous_gesture: Gesture
    hand_detected: bool
    stable_frames: int
    pinch_duration: float
    scroll_delta_y: float
    timestamp: float

    @property
    def action_ready(self) -> bool:
        # Do not authorize a downstream action while the pose is transitioning.
        return (
            self.hand_detected
            and self.gesture is not Gesture.NONE
            and self.candidate is self.gesture
        )


class GestureClassifier:
    """Classify pinch, open palm, two fingers, fist, and index-point poses.

    Gesture changes are debounced. A missing hand immediately emits ENDED for
    an active gesture and clears scroll state, so downstream controls can release
    any held action rather than using a stale gesture.
    """
    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self._gesture = Gesture.NONE
        self._candidate = Gesture.NONE
        self._candidate_since: float | None = None
        self._candidate_frames = 0
        self._last_timestamp: float | None = None
        self._pinch_started: float | None = None
        self._scroll_anchor_y: float | None = None

    @staticmethod
    def _distance(a, b) -> float:
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    @staticmethod
    def _angle(a, b, c) -> float:
        """Angle ABC in degrees; 180 degrees means a straight finger."""
        bax, bay, baz = a.x-b.x, a.y-b.y, a.z-b.z
        bcx, bcy, bcz = c.x-b.x, c.y-b.y, c.z-b.z
        left = math.sqrt(bax*bax + bay*bay + baz*baz)
        right = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)
        if left <= 1e-9 or right <= 1e-9:
            return 0.0
        cosine = (bax*bcx + bay*bcy + baz*bcz) / (left*right)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _classify(self, observation: HandObservation) -> tuple[Gesture, float]:
        points = observation.landmarks
        if len(points) != 21:
            raise ValueError(f"expected 21 hand landmarks, received {len(points)}")
        if not all(math.isfinite(c) for p in points for c in (p.x, p.y, p.z)):
            raise ValueError("landmark coordinates must be finite")

        palm_width = self._distance(points[5], points[17])
        if palm_width <= 1e-6:
            return Gesture.NONE, 1.0
        index_angle = self._angle(points[5], points[6], points[8])
        extended = {
            "index": index_angle >= self.config.finger_extension_angle,
            "middle": self._angle(points[9], points[10], points[12]) >= self.config.finger_extension_angle,
            "ring": self._angle(points[13], points[14], points[16]) >= self.config.finger_extension_angle,
            "pinky": self._angle(points[17], points[18], points[20]) >= self.config.finger_extension_angle,
        }
        thumb_angle = self._angle(points[2], points[3], points[4])
        thumb_abduction = self._distance(points[4], points[5]) / palm_width
        thumb_extended = (
            thumb_angle >= self.config.thumb_extension_angle
            and thumb_abduction >= self.config.thumb_abduction_ratio
        )

        pinch_ratio = self._distance(points[4], points[8]) / palm_width
        pinch_limit = (
            self.config.pinch_release_ratio
            if self._gesture is Gesture.PINCH or self._candidate is Gesture.PINCH
            else self.config.pinch_enter_ratio
        )
        # A pinch can bend the index finger, but a tightly curled index plus
        # all other folded fingers is more consistent with a closed fist.
        if pinch_ratio <= pinch_limit and index_angle >= 105.0:
            return Gesture.PINCH, pinch_ratio
        if not any(extended.values()) and not thumb_extended:
            return Gesture.FIST, 1.0
        if all(extended.values()) and thumb_extended:
            return Gesture.OPEN_PALM, min(thumb_abduction, thumb_angle / 180.0)
        if extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
            return Gesture.TWO_FINGER, 1.0
        if extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
            return Gesture.INDEX_POINT, 1.0
        return Gesture.NONE, 0.0

    def _candidate_update(self, candidate: Gesture, now: float) -> GestureEvent:
        if candidate is not self._candidate:
            self._candidate = candidate
            self._candidate_since = now
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        if (
            candidate is not self._gesture
            and self._candidate_frames >= self.config.stable_frames
            and self._candidate_since is not None
            and now - self._candidate_since >= self.config.min_stable_seconds
        ):
            previous = self._gesture
            self._gesture = candidate
            self._pinch_started = now if candidate is Gesture.PINCH else None
            if previous is Gesture.NONE and candidate is not Gesture.NONE:
                return GestureEvent.STARTED
            if candidate is Gesture.NONE:
                return GestureEvent.ENDED
            return GestureEvent.CHANGED
        return GestureEvent.NONE

    def update(self, observation: HandObservation | None, timestamp: float | None = None) -> GestureUpdate:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and now <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp = now

        previous = self._gesture
        if observation is None:
            event = GestureEvent.ENDED if previous is not Gesture.NONE else GestureEvent.NONE
            pinch_duration = (
                max(0.0, now - self._pinch_started)
                if previous is Gesture.PINCH and self._pinch_started is not None
                else 0.0
            )
            self._gesture = Gesture.NONE
            self._candidate = Gesture.NONE
            self._candidate_since = None
            self._candidate_frames = 0
            self._pinch_started = None
            self._scroll_anchor_y = None
            return GestureUpdate(
                gesture=Gesture.NONE, candidate=Gesture.NONE, event=event,
                previous_gesture=previous, hand_detected=False, stable_frames=0,
                pinch_duration=pinch_duration, scroll_delta_y=0.0, timestamp=now,
            )

        candidate, _metric = self._classify(observation)
        previous_pinch_started = self._pinch_started
        event = self._candidate_update(candidate, now)

        scroll_delta = 0.0
        if self._gesture is Gesture.TWO_FINGER:
            points = observation.landmarks
            centroid_y = (points[8].y + points[12].y) / 2.0
            if self._scroll_anchor_y is not None:
                delta = centroid_y - self._scroll_anchor_y
                if abs(delta) >= self.config.scroll_deadzone:
                    scroll_delta = max(
                        -self.config.max_scroll_delta,
                        min(self.config.max_scroll_delta, delta),
                    )
            self._scroll_anchor_y = centroid_y
        else:
            self._scroll_anchor_y = None

        if self._gesture is Gesture.PINCH and self._pinch_started is not None:
            pinch_duration = max(0.0, now - self._pinch_started)
        elif (
            previous is Gesture.PINCH
            and previous_pinch_started is not None
            and event in (GestureEvent.ENDED, GestureEvent.CHANGED)
        ):
            # Preserve hold time on release so Phase 4 can distinguish tap/drag.
            pinch_duration = max(0.0, now - previous_pinch_started)
        else:
            pinch_duration = 0.0

        return GestureUpdate(
            gesture=self._gesture, candidate=self._candidate, event=event,
            previous_gesture=previous, hand_detected=True,
            stable_frames=self._candidate_frames, pinch_duration=pinch_duration,
            scroll_delta_y=scroll_delta, timestamp=now,
        )

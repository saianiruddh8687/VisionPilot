"""Fail-safe desktop cursor and mouse-action layer for VisionPilot Phase 4.

The engine is disarmed by default. It accepts tracking and gesture updates but
does not act until explicitly armed by the user.
"""
from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from typing import Protocol

from .gestures import Gesture, GestureEvent, GestureUpdate
from .tracking import Point2D, TrackingUpdate


class MouseBackend(Protocol):
    @property
    def screen_size(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def click_left(self) -> None: ...
    def press_left(self) -> None: ...
    def release_left(self) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...


class PynputMouseBackend:
    """Real OS mouse backend. Imported only when desktop control is initialized."""

    def __init__(self) -> None:
        try:
            from pynput.mouse import Button, Controller
        except ImportError as exc:
            raise RuntimeError(
                "pynput is required for desktop control. Install requirements.txt."
            ) from exc
        self._button = Button.left
        self._controller = Controller()
        self._screen_size = self._get_screen_size()

    @staticmethod
    def _get_screen_size() -> tuple[int, int]:
        if sys.platform == "win32":
            import ctypes
            user32 = ctypes.windll.user32
            size = (int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1)))
        else:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
            root.destroy()
        if size[0] < 2 or size[1] < 2:
            raise RuntimeError(f"Invalid primary display dimensions: {size}")
        return size

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    def move_to(self, x: int, y: int) -> None:
        self._controller.position = (x, y)

    def click_left(self) -> None:
        self._controller.click(self._button, 1)

    def press_left(self) -> None:
        self._controller.press(self._button)

    def release_left(self) -> None:
        self._controller.release(self._button)

    def scroll(self, dx: int, dy: int) -> None:
        self._controller.scroll(dx, dy)


@dataclass(frozen=True, slots=True)
class DesktopControlConfig:
    drag_start_seconds: float = 0.45
    scroll_gain: float = 35.0
    edge_margin: float = 0.025

    def __post_init__(self) -> None:
        values = (self.drag_start_seconds, self.scroll_gain, self.edge_margin)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("desktop-control settings must be finite")
        if self.drag_start_seconds < 0 or self.scroll_gain <= 0:
            raise ValueError("drag threshold must be >= 0 and scroll gain must be > 0")
        if not 0 <= self.edge_margin < 0.25:
            raise ValueError("edge_margin must be between 0 and 0.25")


class ScreenMapper:
    """Map normalized camera coordinates onto a safe inset of the primary screen."""

    def __init__(self, width: int, height: int, edge_margin: float = 0.025) -> None:
        if width < 2 or height < 2:
            raise ValueError("screen dimensions must both be at least 2 pixels")
        if not math.isfinite(edge_margin) or not 0 <= edge_margin < 0.25:
            raise ValueError("edge_margin must be between 0 and 0.25")
        self.width = int(width)
        self.height = int(height)
        self.edge_margin = float(edge_margin)

    def map(self, point: Point2D) -> tuple[int, int]:
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise ValueError("cursor coordinates must be finite")
        x = min(1.0, max(0.0, point.x))
        y = min(1.0, max(0.0, point.y))
        mx = self.edge_margin * (self.width - 1)
        my = self.edge_margin * (self.height - 1)
        px = round(mx + x * ((self.width - 1) - 2 * mx))
        py = round(my + y * ((self.height - 1) - 2 * my))
        return (
            min(self.width - 1, max(0, int(px))),
            min(self.height - 1, max(0, int(py))),
        )


class DesktopControlEngine:
    """Translate validated hand-tracking/gesture states into OS mouse actions.

    Controls: INDEX_POINT moves; quick PINCH release clicks; sustained PINCH
    drags; TWO_FINGER scrolls; OPEN_PALM pauses. Tracking loss disarms.
    """

    def __init__(
        self,
        backend: MouseBackend,
        config: DesktopControlConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or DesktopControlConfig()
        width, height = backend.screen_size
        self.mapper = ScreenMapper(width, height, self.config.edge_margin)
        self._lock = threading.RLock()
        self._enabled = False
        self._paused = False
        self._left_down = False
        self._scroll_remainder = 0.0
        self._status = "DISARMED — press E to enable"
        self._last_error: str | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def toggle_enabled(self) -> bool:
        """Explicit keyboard arming/disarming; returns the new enabled state."""
        with self._lock:
            if self._enabled:
                self._disarm("DISARMED — press E to enable")
            else:
                self._last_error = None
                self._enabled = True
                self._paused = False
                self._scroll_remainder = 0.0
                self._status = "ARMED — INDEX POINT moves cursor"
            return self._enabled

    def emergency_stop(self) -> None:
        """Thread-safe global stop. User must explicitly press E to re-arm."""
        with self._lock:
            self._disarm("EMERGENCY STOP — press E to re-arm")

    def _release_left(self) -> None:
        if self._left_down:
            try:
                self.backend.release_left()
            finally:
                self._left_down = False

    def _disarm(self, reason: str) -> None:
        try:
            self._release_left()
        except Exception as exc:
            self._last_error = f"Mouse release failed: {exc}"
        self._enabled = False
        self._paused = False
        self._scroll_remainder = 0.0
        self._status = reason

    def _move_cursor(self, point: Point2D) -> None:
        x, y = self.mapper.map(point)
        self.backend.move_to(x, y)

    def _scroll(self, delta_y: float) -> None:
        if not math.isfinite(delta_y):
            return
        self._scroll_remainder += -delta_y * self.config.scroll_gain
        ticks = int(self._scroll_remainder)
        if ticks:
            self.backend.scroll(0, ticks)
            self._scroll_remainder -= ticks

    def update(self, tracking: TrackingUpdate, gesture: GestureUpdate) -> str:
        """Apply one frame of input. Fail closed on tracking loss or I/O error."""
        with self._lock:
            if not self._enabled:
                return self._status

            if not tracking.action_ready or not gesture.hand_detected:
                self._disarm("TRACKING LOST — control disarmed; press E to re-arm")
                return self._status

            try:
                if gesture.action_ready and gesture.gesture is Gesture.OPEN_PALM:
                    self._release_left()
                    self._paused = True
                    self._status = "PAUSED — open palm"
                    return self._status

                if self._paused:
                    if not gesture.action_ready or gesture.gesture is Gesture.OPEN_PALM:
                        self._status = "PAUSED — open palm"
                        return self._status
                    self._paused = False

                active = gesture.gesture
                # Move to the release-frame coordinate before a click/drop so
                # the final pinch position is used, not the previous frame.
                if gesture.action_ready and active in (Gesture.INDEX_POINT, Gesture.PINCH):
                    if tracking.index_tip is not None:
                        self._move_cursor(tracking.index_tip)

                # Release/click only on the debounced pinch-end event. Hand loss
                # follows the fail-safe disarm path above and never synthesizes a click.
                if (
                    gesture.previous_gesture is Gesture.PINCH
                    and gesture.event in (GestureEvent.CHANGED, GestureEvent.ENDED)
                ):
                    held_for = gesture.pinch_duration
                    if self._left_down:
                        self._release_left()
                    elif held_for < self.config.drag_start_seconds:
                        self.backend.click_left()

                if gesture.action_ready and active is Gesture.PINCH:
                    if (
                        gesture.pinch_duration >= self.config.drag_start_seconds
                        and not self._left_down
                    ):
                        self.backend.press_left()
                        self._left_down = True
                    self._status = "DRAGGING — hold pinch" if self._left_down else "PINCH — hold to drag"
                elif gesture.action_ready and active is Gesture.TWO_FINGER:
                    self._scroll(gesture.scroll_delta_y)
                    self._status = "SCROLLING — two-finger movement"
                elif gesture.action_ready and active is Gesture.INDEX_POINT:
                    self._status = "ARMED — cursor tracking"
                elif gesture.action_ready and active is Gesture.FIST:
                    self._release_left()
                    self._status = "ARMED — fist detected (no action)"
                else:
                    self._status = "ARMED — waiting for a supported gesture"
            except Exception as exc:
                self._last_error = str(exc)
                self._disarm("CONTROL ERROR — control disarmed")
            return self._status

    def close(self) -> None:
        """Release any held button and disarm during application shutdown."""
        with self._lock:
            self._disarm("CLOSED — mouse button released")


class EmergencyStopListener:
    """Global F12 emergency-stop listener powered by pynput."""

    def __init__(self, engine: DesktopControlEngine) -> None:
        self.engine = engine
        self._listener = None

    def start(self) -> bool:
        try:
            from pynput import keyboard
        except ImportError:
            return False

        def on_press(key) -> None:
            if key == keyboard.Key.f12:
                self.engine.emergency_stop()

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception:
            self._listener = None
            return False

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

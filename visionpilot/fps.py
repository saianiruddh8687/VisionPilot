"""A tiny exponential moving-average FPS meter."""

from __future__ import annotations

import time


class FPSCounter:
    def __init__(self, smoothing: float = 0.15) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in the interval (0, 1]")
        self.smoothing = smoothing
        self._last_time: float | None = None
        self._fps: float | None = None

    def tick(self, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else float(now)
        if self._last_time is not None:
            delta = current - self._last_time
            if delta > 0:
                instant_fps = 1.0 / delta
                self._fps = (
                    instant_fps
                    if self._fps is None
                    else self.smoothing * instant_fps + (1.0 - self.smoothing) * self._fps
                )
        self._last_time = current
        return self.value

    @property
    def value(self) -> float:
        return self._fps or 0.0

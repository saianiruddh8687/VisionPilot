"""Signal filters for stable hand-motion tracking."""

from __future__ import annotations

import math


class _LowPassFilter:
    def __init__(self) -> None:
        self.value: float | None = None

    def reset(self, value: float | None = None) -> None:
        self.value = value

    def apply(self, value: float, alpha: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """Adaptive low-pass filter: smoother at rest, more responsive in motion."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.025,
        derivative_cutoff: float = 1.0,
    ) -> None:
        values = (min_cutoff, beta, derivative_cutoff)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("filter parameters must be finite")
        if min_cutoff <= 0 or derivative_cutoff <= 0 or beta < 0:
            raise ValueError("cutoffs must be positive and beta must be non-negative")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._signal = _LowPassFilter()
        self._derivative = _LowPassFilter()
        self._last_raw: float | None = None
        self._last_timestamp: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._signal.reset()
        self._derivative.reset()
        self._last_raw = None
        self._last_timestamp = None

    def filter(self, value: float, timestamp: float) -> float:
        value = float(value)
        timestamp = float(timestamp)
        if not math.isfinite(value) or not math.isfinite(timestamp):
            raise ValueError("filter values and timestamps must be finite")

        if self._last_timestamp is None:
            self._last_raw = value
            self._last_timestamp = timestamp
            self._derivative.reset(0.0)
            return self._signal.apply(value, 1.0)

        dt = timestamp - self._last_timestamp
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")

        assert self._last_raw is not None
        raw_derivative = (value - self._last_raw) / dt
        derivative = self._derivative.apply(
            raw_derivative, self._alpha(self.derivative_cutoff, dt)
        )
        cutoff = self.min_cutoff + self.beta * abs(derivative)
        filtered = self._signal.apply(value, self._alpha(cutoff, dt))
        self._last_raw = value
        self._last_timestamp = timestamp
        return filtered


class PointSmoother:
    """Apply identical One Euro filters to normalized X and Y coordinates."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.025,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self._x = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._y = OneEuroFilter(min_cutoff, beta, derivative_cutoff)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        return (
            self._x.filter(x, timestamp),
            self._y.filter(y, timestamp),
        )

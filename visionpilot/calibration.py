"""Five-point camera-to-screen calibration for VisionPilot Phase 5."""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable

from .tracking import Point2D


class CalibrationError(ValueError):
    """Raised when calibration data is incomplete, unstable, or invalid."""


@dataclass(frozen=True, slots=True)
class CalibrationPair:
    camera: Point2D
    screen: Point2D


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    matrix: tuple[float, ...]
    rms_error: float
    screen_width: int
    screen_height: int
    point_count: int = 5
    version: int = 1

    def __post_init__(self) -> None:
        if len(self.matrix) != 9 or not all(math.isfinite(v) for v in self.matrix):
            raise CalibrationError("calibration matrix must contain nine finite values")
        if not math.isfinite(self.rms_error) or self.rms_error < 0:
            raise CalibrationError("calibration RMS error must be finite and non-negative")
        if self.screen_width < 2 or self.screen_height < 2:
            raise CalibrationError("screen dimensions must be at least 2 pixels")
        if self.point_count < 4:
            raise CalibrationError("at least four calibration points are required")
        if abs(self.matrix[8]) < 1e-10:
            raise CalibrationError("invalid homography normalization")
        if self.version != 1:
            raise CalibrationError(f"unsupported calibration profile version: {self.version}")

    def transform(self, point: Point2D) -> Point2D:
        """Transform normalized camera coordinates into normalized screen coordinates."""
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise CalibrationError("camera coordinates must be finite")
        h = self.matrix
        denominator = h[6] * point.x + h[7] * point.y + h[8]
        if abs(denominator) < 1e-9:
            raise CalibrationError("camera point maps to an invalid projective denominator")
        x = (h[0] * point.x + h[1] * point.y + h[2]) / denominator
        y = (h[3] * point.x + h[4] * point.y + h[5]) / denominator
        if not math.isfinite(x) or not math.isfinite(y):
            raise CalibrationError("calibration produced non-finite screen coordinates")
        return Point2D(x, y)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "matrix": list(self.matrix),
            "rms_error": self.rms_error,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "point_count": self.point_count,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibrationProfile":
        try:
            return cls(
                matrix=tuple(float(v) for v in payload["matrix"]),
                rms_error=float(payload["rms_error"]),
                screen_width=int(payload["screen_width"]),
                screen_height=int(payload["screen_height"]),
                point_count=int(payload.get("point_count", 5)),
                version=int(payload.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid calibration profile: {exc}") from exc


CALIBRATION_TARGETS: tuple[tuple[str, Point2D], ...] = (
    ("TOP LEFT", Point2D(0.05, 0.05)),
    ("TOP RIGHT", Point2D(0.95, 0.05)),
    ("BOTTOM RIGHT", Point2D(0.95, 0.95)),
    ("BOTTOM LEFT", Point2D(0.05, 0.95)),
    ("CENTER", Point2D(0.50, 0.50)),
)


def fit_calibration(
    pairs: Iterable[CalibrationPair],
    screen_width: int,
    screen_height: int,
    *,
    max_rms_error: float = 0.08,
) -> CalibrationProfile:
    """Fit a camera-to-screen homography from four or more paired points."""
    import numpy as np

    points = tuple(pairs)
    if len(points) < 4:
        raise CalibrationError("capture at least four calibration points")
    if screen_width < 2 or screen_height < 2:
        raise CalibrationError("screen dimensions must be at least 2 pixels")
    if not math.isfinite(max_rms_error) or max_rms_error <= 0:
        raise CalibrationError("max_rms_error must be positive and finite")

    rows: list[list[float]] = []
    values: list[float] = []
    for pair in points:
        x, y = pair.camera.x, pair.camera.y
        u, v = pair.screen.x, pair.screen.y
        if not all(math.isfinite(value) for value in (x, y, u, v)):
            raise CalibrationError("calibration point coordinates must be finite")
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u*x, -u*y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v*x, -v*y])
        values.append(v)

    matrix_a = np.asarray(rows, dtype=float)
    vector_b = np.asarray(values, dtype=float)
    solution, _residuals, rank, _singular = np.linalg.lstsq(matrix_a, vector_b, rcond=None)
    if rank < 8:
        raise CalibrationError("calibration points are degenerate; recapture all targets")

    homography = tuple(float(value) for value in (*solution, 1.0))
    provisional = CalibrationProfile(
        matrix=homography,
        rms_error=0.0,
        screen_width=screen_width,
        screen_height=screen_height,
        point_count=len(points),
    )
    errors = []
    for pair in points:
        mapped = provisional.transform(pair.camera)
        errors.append((mapped.x - pair.screen.x) ** 2 + (mapped.y - pair.screen.y) ** 2)
    rms = math.sqrt(fmean(errors))
    if rms > max_rms_error:
        raise CalibrationError(
            f"calibration fit error {rms:.4f} exceeds limit {max_rms_error:.4f}; recapture targets"
        )
    return CalibrationProfile(
        matrix=homography,
        rms_error=rms,
        screen_width=screen_width,
        screen_height=screen_height,
        point_count=len(points),
    )


class CalibrationSession:
    """Guided five-target capture session with a stability gate."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        *,
        min_samples: int = 20,
        max_samples: int = 20,
        max_stddev: float = 0.012,
    ) -> None:
        if screen_width < 2 or screen_height < 2:
            raise ValueError("screen dimensions must be at least 2 pixels")
        if min_samples < 5 or max_samples < min_samples:
            raise ValueError("sample limits must satisfy 5 <= min_samples <= max_samples")
        if not math.isfinite(max_stddev) or max_stddev <= 0:
            raise ValueError("max_stddev must be positive and finite")
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.max_stddev = max_stddev
        self._samples: deque[Point2D] = deque(maxlen=max_samples)
        self._pairs: list[CalibrationPair] = []
        self._profile: CalibrationProfile | None = None
        self._cancelled = False

    @property
    def progress(self) -> int:
        return len(self._pairs)

    @property
    def total_targets(self) -> int:
        return len(CALIBRATION_TARGETS)

    @property
    def current_target(self) -> tuple[str, Point2D] | None:
        if self.complete or self._cancelled:
            return None
        return CALIBRATION_TARGETS[self.progress]

    @property
    def complete(self) -> bool:
        return self._profile is not None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def profile(self) -> CalibrationProfile | None:
        return self._profile

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def stability(self) -> tuple[float, float] | None:
        if len(self._samples) < 2:
            return None
        xs = [point.x for point in self._samples]
        ys = [point.y for point in self._samples]
        return pstdev(xs), pstdev(ys)

    @property
    def can_capture(self) -> bool:
        stats = self.stability
        return (
            not self.complete
            and not self._cancelled
            and len(self._samples) >= self.min_samples
            and stats is not None
            and stats[0] <= self.max_stddev
            and stats[1] <= self.max_stddev
        )

    def observe(self, point: Point2D | None) -> None:
        if self.complete or self._cancelled:
            return
        if point is None:
            self._samples.clear()
            return
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            self._samples.clear()
            return
        self._samples.append(Point2D(
            min(1.0, max(0.0, point.x)),
            min(1.0, max(0.0, point.y)),
        ))

    def capture(self) -> bool:
        """Capture the current stable average. Return True after the final target."""
        if not self.can_capture:
            raise CalibrationError(
                f"need {self.min_samples} stable frames; hold index fingertip steady"
            )
        _label, target = CALIBRATION_TARGETS[self.progress]
        camera_point = Point2D(
            fmean(point.x for point in self._samples),
            fmean(point.y for point in self._samples),
        )
        self._pairs.append(CalibrationPair(camera=camera_point, screen=target))
        self._samples.clear()
        if self.progress == self.total_targets:
            try:
                self._profile = fit_calibration(
                    self._pairs, self.screen_width, self.screen_height
                )
            except CalibrationError as exc:
                self._pairs.clear()
                self._samples.clear()
                raise CalibrationError(
                    f"calibration fit failed ({exc}); all targets were reset, start again"
                ) from exc
            return True
        return False

    def cancel(self) -> None:
        self._cancelled = True
        self._samples.clear()


def default_calibration_path() -> Path:
    """Return a per-user profile location outside the source repository."""
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "VisionPilot" / "calibration.json"


def save_profile(profile: CalibrationProfile, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(path)


def load_profile(path: Path) -> CalibrationProfile | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CalibrationProfile.from_dict(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read calibration profile {path}: {exc}") from exc


class CalibratedScreenMapper:
    """Map smoothed camera points through a homography to screen pixels."""

    def __init__(
        self,
        width: int,
        height: int,
        profile: CalibrationProfile,
        edge_margin: float = 0.025,
    ) -> None:
        if (width, height) != (profile.screen_width, profile.screen_height):
            raise CalibrationError(
                "saved calibration screen dimensions do not match the current primary display; recalibrate"
            )
        if not math.isfinite(edge_margin) or not 0 <= edge_margin < 0.25:
            raise ValueError("edge_margin must be between 0 and 0.25")
        self.width = width
        self.height = height
        self.profile = profile
        self.edge_margin = edge_margin

    def map(self, point: Point2D) -> tuple[int, int]:
        normalized = self.profile.transform(point)
        x = min(1.0 - self.edge_margin, max(self.edge_margin, normalized.x))
        y = min(1.0 - self.edge_margin, max(self.edge_margin, normalized.y))
        return (
            int(round(x * (self.width - 1) + 1e-8)),
            int(round(y * (self.height - 1) + 1e-8)),
        )

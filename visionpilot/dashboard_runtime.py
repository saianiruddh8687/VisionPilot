"""Camera/inference worker for the PySide6 VisionPilot dashboard.

All camera, MediaPipe, calibration, and mouse-backend work stays on this thread.
The Qt window receives detached QImages and small telemetry dictionaries.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .calibration import (
    CalibrationError,
    CalibrationSession,
    CalibratedScreenMapper,
    default_calibration_path,
    load_profile,
    save_profile,
)
from .camera import CameraStream
from .config import DEFAULT_MODEL_PATH, VisionPilotConfig
from .dashboard_state import DashboardSafetyGate
from .desktop import (
    DesktopControlConfig,
    DesktopControlEngine,
    EmergencyStopListener,
    PynputMouseBackend,
    ScreenMapper,
)
from .detector import HandLandmarkDetector
from .fps import FPSCounter
from .gestures import Gesture, GestureClassifier
from .landmarks import draw_hand_observation
from .tracking import HandTracker, TrackingPhase


class DashboardWorker(QThread):
    """Own the real-time pipeline and accept non-blocking UI commands."""

    frame_ready = Signal(QImage)
    state_ready = Signal(dict)

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        mirror_preview: bool = True,
        calibration_file: Path | None = None,
        drag_threshold: float = 0.45,
        scroll_gain: float = 35.0,
        edge_margin: float = 0.025,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.camera_index = int(camera_index)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.mirror_preview = bool(mirror_preview)
        self.calibration_file = Path(calibration_file or default_calibration_path())
        self.drag_threshold = float(drag_threshold)
        self.scroll_gain = float(scroll_gain)
        self.edge_margin = float(edge_margin)

        self._commands: queue.Queue[tuple[str, tuple[Any, ...]]] = queue.Queue()
        self._stop_event = threading.Event()
        self._camera: CameraStream | None = None
        self._detector: HandLandmarkDetector | None = None
        self._desktop: DesktopControlEngine | None = None
        self._backend: PynputMouseBackend | None = None
        self._emergency_listener: EmergencyStopListener | None = None
        self._listener_available = False
        self._calibration: CalibrationSession | None = None
        self._profile = None
        self._gate = DashboardSafetyGate()
        self._tracker = HandTracker()
        self._gestures = GestureClassifier()
        self._fps = FPSCounter()
        self._last_control_status = "INITIALIZING"
        self._calibration_status = "Checking calibration profile…"
        self._last_error = ""
        self._frame_count = 0

    def command(self, name: str, *args: Any) -> None:
        """Queue a UI action; it is applied by the owner thread."""
        if not self._stop_event.is_set():
            self._commands.put((name, args))

    def request_stop(self) -> None:
        self._stop_event.set()
        self._commands.put(("stop", ()))

    def run(self) -> None:
        try:
            import cv2

            self._initialize_desktop()
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Hand Landmarker model not found: {self.model_path}. "
                    "Run python scripts/download_model.py."
                )
            config = self._vision_config()
            self._detector = HandLandmarkDetector(self.model_path, config)
            self._emit_state()
            while not self._stop_event.is_set():
                self._process_commands()
                if self._stop_event.is_set():
                    break
                if self._camera is None:
                    self._camera = CameraStream(
                        self.camera_index, self.frame_width, self.frame_height
                    )

                frame = self._camera.read()
                if self.mirror_preview:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                timestamp_ms = time.monotonic_ns() // 1_000_000
                observation = self._detector.detect(
                    frame, timestamp_ms, mirrored_input=self.mirror_preview
                )
                tracking = self._tracker.update(observation, timestamp=now)
                gesture = self._gestures.update(observation, timestamp=now)
                fps_value = self._fps.tick()

                if self._calibration is not None:
                    sample = None
                    if (
                        tracking.action_ready
                        and gesture.action_ready
                        and gesture.gesture is Gesture.INDEX_POINT
                    ):
                        sample = tracking.index_tip
                    self._calibration.observe(sample)
                    target = self._calibration.current_target
                    if target is not None:
                        label, point = target
                        marker = (
                            int(point.x * (frame.shape[1] - 1)),
                            int(point.y * (frame.shape[0] - 1)),
                        )
                        cv2.drawMarker(
                            frame, marker, (0, 220, 255), cv2.MARKER_CROSS,
                            36, 2, cv2.LINE_AA,
                        )
                        self._calibration_status = (
                            f"{label} · {self._calibration.sample_count}/"
                            f"{self._calibration.min_samples} stable samples"
                        )
                    self._last_control_status = "CALIBRATION — mouse control locked"
                elif self._desktop is not None:
                    self._last_control_status = self._desktop.update(tracking, gesture)
                    self._gate.armed = self._desktop.enabled
                else:
                    self._last_control_status = "PREVIEW ONLY — desktop control unavailable"

                if observation is not None:
                    draw_hand_observation(frame, observation)
                if tracking.phase is TrackingPhase.TRACKING and tracking.index_tip is not None:
                    tip = (
                        int(tracking.index_tip.x * (frame.shape[1] - 1)),
                        int(tracking.index_tip.y * (frame.shape[0] - 1)),
                    )
                    cv2.circle(frame, tip, 11, (40, 230, 235), 2, cv2.LINE_AA)
                    cv2.circle(frame, tip, 3, (40, 230, 235), -1, cv2.LINE_AA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    int(rgb.strides[0]),
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image)
                self._frame_count += 1
                self._emit_state(
                    tracking=tracking,
                    gesture=gesture,
                    fps=fps_value,
                )
        except Exception as exc:
            self._last_error = str(exc)
            self._last_control_status = "STOPPED — runtime error"
            self._emit_state()
        finally:
            self._safe_shutdown()

    def _vision_config(self) -> VisionPilotConfig:
        return VisionPilotConfig(
            camera_index=self.camera_index,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            mirror_preview=self.mirror_preview,
        )

    def _initialize_desktop(self) -> None:
        try:
            self._backend = PynputMouseBackend()
            settings = DesktopControlConfig(
                drag_start_seconds=self.drag_threshold,
                scroll_gain=self.scroll_gain,
                edge_margin=self.edge_margin,
            )
            self._desktop = DesktopControlEngine(self._backend, settings)
            self._gate.set_desktop_available(True)
            self._reload_profile()
            self._emergency_listener = EmergencyStopListener(self._desktop)
            self._listener_available = self._emergency_listener.start()
            if not self._listener_available:
                self._last_error = (
                    "Global F12 listener unavailable; use the dashboard Stop button."
                )
        except Exception as exc:
            self._desktop = None
            self._backend = None
            self._gate.set_desktop_available(False)
            self._calibration_status = "Desktop control unavailable"
            self._last_control_status = "PREVIEW ONLY"
            self._last_error = f"Desktop control unavailable: {exc}"

    def _reload_profile(self) -> None:
        self._profile = None
        self._gate.set_profile_valid(False)
        if self._desktop is None or self._backend is None:
            self._calibration_status = "Preview only — no desktop backend"
            return
        try:
            profile = load_profile(self.calibration_file)
            if profile is None:
                self._calibration_status = "Calibration required before arming"
                self._desktop.mapper = ScreenMapper(
                    *self._backend.screen_size, edge_margin=self.edge_margin
                )
                return
            expected = (
                self.camera_index, self.frame_width, self.frame_height,
                self.mirror_preview,
            )
            saved = (
                profile.camera_index, profile.camera_width, profile.camera_height,
                profile.mirror_preview,
            )
            if saved != expected:
                raise CalibrationError(
                    "profile camera or mirror settings do not match the current setup"
                )
            self._desktop.mapper = CalibratedScreenMapper(
                *self._backend.screen_size, profile, edge_margin=self.edge_margin
            )
            self._profile = profile
            self._gate.set_profile_valid(True)
            self._calibration_status = (
                f"Calibrated · RMS {profile.rms_error:.4f}"
            )
        except CalibrationError as exc:
            self._calibration_status = f"Recalibration required · {exc}"
            self._last_error = str(exc)
            self._desktop.mapper = ScreenMapper(
                *self._backend.screen_size, edge_margin=self.edge_margin
            )

    def _process_commands(self) -> None:
        while True:
            try:
                name, args = self._commands.get_nowait()
            except queue.Empty:
                return
            if name == "stop":
                self._stop_event.set()
                return
            try:
                if name == "arm":
                    self._toggle_arm()
                elif name == "emergency":
                    if self._desktop is not None:
                        self._desktop.emergency_stop()
                    self._gate.emergency_stop()
                    self._last_control_status = "EMERGENCY STOP — disarmed"
                elif name == "calibrate":
                    self._start_calibration()
                elif name == "capture":
                    self._capture_calibration()
                elif name == "cancel_calibration":
                    self._cancel_calibration()
                elif name == "camera":
                    self._change_camera(int(args[0]))
                elif name == "mirror":
                    self._change_mirror(bool(args[0]))
                elif name == "settings":
                    self._change_settings(
                        drag_threshold=float(args[0]),
                        scroll_gain=float(args[1]),
                        edge_margin=float(args[2]),
                    )
            except Exception as exc:
                self._last_error = str(exc)
                self._last_control_status = f"Action rejected · {exc}"
            self._emit_state()

    def _toggle_arm(self) -> None:
        if self._desktop is None:
            self._last_control_status = "Desktop control unavailable"
            return
        if self._desktop.enabled:
            self._desktop.toggle_enabled()
            self._gate.disarm()
            self._last_control_status = self._desktop.status
            return
        self._gate.disarm()
        if not self._gate.request_arm():
            self._last_control_status = "ARM BLOCKED — valid calibration is required"
            return
        self._desktop.toggle_enabled()
        self._last_control_status = self._desktop.status

    def _start_calibration(self) -> None:
        if self._desktop is None or self._backend is None:
            raise RuntimeError("desktop control is not available")
        if self._calibration is not None:
            raise RuntimeError("calibration is already active")
        if not self._gate.begin_calibration():
            raise RuntimeError("disarm desktop control before calibrating")
        self._calibration = CalibrationSession(
            *self._backend.screen_size,
            camera_index=self.camera_index,
            camera_width=self.frame_width,
            camera_height=self.frame_height,
            mirror_preview=self.mirror_preview,
        )
        self._calibration_status = "Calibration started — follow the five targets"

    def _capture_calibration(self) -> None:
        if self._calibration is None:
            return
        completed = self._calibration.capture()
        if completed and self._calibration.profile is not None:
            profile = self._calibration.profile
            try:
                save_profile(profile, self.calibration_file)
            except OSError:
                self._calibration = None
                self._gate.finish_calibration(valid_profile=self._profile is not None)
                raise
            self._profile = profile
            self._desktop.mapper = CalibratedScreenMapper(
                *self._backend.screen_size,
                profile,
                edge_margin=self.edge_margin,
            )
            self._calibration = None
            self._gate.finish_calibration(valid_profile=True)
            self._calibration_status = (
                f"Calibrated · RMS {profile.rms_error:.4f} · ready to arm"
            )
        else:
            self._calibration_status = (
                f"Captured {self._calibration.progress}/"
                f"{self._calibration.total_targets} targets"
            )

    def _cancel_calibration(self) -> None:
        if self._calibration is None:
            return
        self._calibration.cancel()
        self._calibration = None
        self._gate.finish_calibration(valid_profile=self._profile is not None)
        self._calibration_status = (
            "Calibration cancelled"
            if self._profile is None
            else "Calibration cancelled · previous profile retained"
        )

    def _change_camera(self, camera_index: int) -> None:
        if camera_index < 0:
            return
        if self._desktop is not None and self._desktop.enabled:
            raise RuntimeError("disarm before changing cameras")
        if self._calibration is not None:
            raise RuntimeError("cancel calibration before changing cameras")
        if camera_index == self.camera_index:
            return
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self.camera_index = camera_index
        self._reset_perception()
        self._reload_profile()
        self._last_error = ""

    def _change_mirror(self, mirrored: bool) -> None:
        if self._desktop is not None and self._desktop.enabled:
            raise RuntimeError("disarm before changing mirror mode")
        if self._calibration is not None:
            raise RuntimeError("cancel calibration before changing mirror mode")
        if mirrored == self.mirror_preview:
            return
        self.mirror_preview = mirrored
        self._reset_perception()
        self._reload_profile()

    def _change_settings(
        self, *, drag_threshold: float, scroll_gain: float, edge_margin: float
    ) -> None:
        if self._desktop is None or self._backend is None:
            return
        if self._desktop.enabled or self._calibration is not None:
            raise RuntimeError("disarm and finish calibration before changing settings")
        settings = DesktopControlConfig(
            drag_start_seconds=drag_threshold,
            scroll_gain=scroll_gain,
            edge_margin=edge_margin,
        )
        self.drag_threshold = drag_threshold
        self.scroll_gain = scroll_gain
        self.edge_margin = edge_margin
        self._desktop.config = settings
        if self._profile is not None:
            self._desktop.mapper = CalibratedScreenMapper(
                *self._backend.screen_size, self._profile, edge_margin=edge_margin
            )
        else:
            self._desktop.mapper = ScreenMapper(
                *self._backend.screen_size, edge_margin=edge_margin
            )

    def _reset_perception(self) -> None:
        self._tracker = HandTracker()
        self._gestures = GestureClassifier()
        self._fps = FPSCounter()

    def _emit_state(self, *, tracking=None, gesture=None, fps: float | None = None) -> None:
        phase = tracking.phase.name if tracking is not None else "SEARCHING"
        if tracking is not None and tracking.phase is TrackingPhase.TRACKING:
            phase = f"TRACKING · {tracking.handedness}"
        gesture_name = gesture.gesture.value if gesture is not None else "NONE"
        fps_value = fps if fps is not None else self._fps.value
        desktop_available = self._desktop is not None
        calibration_active = self._calibration is not None
        payload = {
            "camera_index": self.camera_index,
            "camera_open": self._camera is not None,
            "mirror_preview": self.mirror_preview,
            "tracking": phase,
            "gesture": gesture_name,
            "fps": float(fps_value),
            "control_status": self._last_control_status,
            "armed": bool(self._desktop and self._desktop.enabled),
            "desktop_available": desktop_available,
            "listener_available": self._listener_available,
            "profile_valid": self._profile is not None,
            "calibration_active": calibration_active,
            "calibration_status": self._calibration_status,
            "calibration_progress": (
                self._calibration.progress if calibration_active else
                (5 if self._profile is not None else 0)
            ),
            "calibration_total": 5,
            "sample_count": (
                self._calibration.sample_count if calibration_active else 0
            ),
            "can_capture": bool(
                self._calibration is not None and self._calibration.can_capture
            ),
            "calibration_target": (
                self._calibration.current_target[0]
                if self._calibration is not None
                and self._calibration.current_target is not None else ""
            ),
            "error": self._last_error,
            "frames": self._frame_count,
        }
        self.state_ready.emit(payload)

    def _safe_shutdown(self) -> None:
        if self._calibration is not None:
            self._calibration.cancel()
            self._calibration = None
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._emergency_listener is not None:
            self._emergency_listener.close()
            self._emergency_listener = None
        if self._desktop is not None:
            self._desktop.close()
            self._desktop = None
        self._gate.disarm()
        self._last_control_status = "STOPPED — mouse released"
        self._emit_state()

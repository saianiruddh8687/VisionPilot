"""PySide6 desktop dashboard for VisionPilot Phase 6."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .config import DEFAULT_MODEL_PATH
from .dashboard_runtime import DashboardWorker


DASHBOARD_STYLE = """
QMainWindow, QWidget#root { background: #0b1018; color: #e8edf5; }
QWidget { color: #e8edf5; font-family: "Segoe UI"; font-size: 10pt; }
QFrame#card { background: #121b28; border: 1px solid #253449; border-radius: 14px; }
QLabel#eyebrow { color: #7f94ae; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#muted { color: #8fa1b8; }
QLabel#value { color: #f2f6fc; font-size: 12pt; font-weight: 600; }
QLabel#brand { color: #f4f7fb; font-size: 21pt; font-weight: 800; letter-spacing: 2px; }
QLabel#subtitle { color: #8fa1b8; font-size: 9pt; }
QLabel#pill { background: #172c2c; color: #73e0b0; border: 1px solid #275344;
              border-radius: 10px; padding: 6px 10px; font-size: 8pt; font-weight: 700; }
QLabel#preview { background: #060a10; color: #7d8da3; border: 1px solid #28374a;
                 border-radius: 10px; font-size: 12pt; }
QLabel#preview_hint { color: #a5b4c7; background: transparent; }
QPushButton { background: #1d2d42; color: #eaf1fa; border: 1px solid #334b67;
              border-radius: 9px; padding: 10px 14px; font-weight: 600; }
QPushButton:hover { background: #29415e; }
QPushButton:disabled { color: #68788c; background: #141d29; border-color: #263243; }
QPushButton#primary { background: #2874d0; border-color: #3688e8; color: white; }
QPushButton#primary:hover { background: #3488ed; }
QPushButton#danger { background: #a9323e; border-color: #d04b58; color: white; }
QPushButton#danger:hover { background: #c6404c; }
QPushButton#armed { background: #8b303a; border-color: #bd4450; color: white; }
QComboBox, QDoubleSpinBox { background: #0e1723; color: #e8edf5;
                            border: 1px solid #304157; border-radius: 7px; padding: 7px; }
QComboBox::drop-down { border: 0; width: 24px; }
QCheckBox { color: #cbd6e4; spacing: 8px; }
QCheckBox::indicator { width: 17px; height: 17px; }
QProgressBar { background: #0c1420; border: 1px solid #2c3d53; border-radius: 6px;
               height: 10px; text-align: center; }
QProgressBar::chunk { background: #35b78b; border-radius: 5px; }
QScrollArea { border: 0; background: transparent; }
"""


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(10)
    heading = QLabel(title.upper())
    heading.setObjectName("eyebrow")
    layout.addWidget(heading)
    if subtitle:
        note = QLabel(subtitle)
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
    return frame, layout


def _metric(label: str) -> tuple[QWidget, QLabel]:
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    caption = QLabel(label.upper())
    caption.setObjectName("eyebrow")
    value = QLabel("—")
    value.setObjectName("value")
    value.setWordWrap(True)
    layout.addWidget(caption)
    layout.addWidget(value)
    return wrapper, value


class CameraPreview(QLabel):
    def __init__(self) -> None:
        super().__init__("Waiting for camera…")
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: QImage | None = None

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self._render()

    def _render(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()


class DashboardWindow(QMainWindow):
    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        camera_index: int = 0,
        calibration_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("VisionPilot — Desktop Dashboard")
        self.setMinimumSize(1120, 740)
        self.resize(1480, 900)
        self.setStyleSheet(DASHBOARD_STYLE)
        self._build_ui()

        self.worker = DashboardWorker(
            model_path=model_path,
            camera_index=camera_index,
            calibration_file=calibration_file,
        )
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.state_ready.connect(self._apply_state)
        self.worker.start()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(16)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        brand = QLabel("VISIONPILOT")
        brand.setObjectName("brand")
        subtitle = QLabel("GESTURE-DRIVEN DESKTOP CONTROL")
        subtitle.setObjectName("subtitle")
        title_stack.addWidget(brand)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch(1)
        local_badge = QLabel("●  LOCAL PROCESSING")
        local_badge.setObjectName("pill")
        header.addWidget(local_badge)
        self.emergency_button = QPushButton("EMERGENCY STOP")
        self.emergency_button.setObjectName("danger")
        self.emergency_button.clicked.connect(
            lambda: self.worker.command("emergency")
        )
        header.addWidget(self.emergency_button)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(16)

        preview_card, preview_layout = _card(
            "Camera preview",
            "Live hand landmarks and fingertip position. Camera frames remain on this device.",
        )
        self.preview = CameraPreview()
        preview_layout.addWidget(self.preview, 1)
        preview_footer = QHBoxLayout()
        self.preview_camera = QLabel("CAMERA 0 · CONNECTING")
        self.preview_camera.setObjectName("muted")
        self.preview_fps = QLabel("0.0 FPS")
        self.preview_fps.setObjectName("muted")
        preview_footer.addWidget(self.preview_camera)
        preview_footer.addStretch(1)
        preview_footer.addWidget(self.preview_fps)
        preview_layout.addLayout(preview_footer)
        body.addWidget(preview_card, 3)

        right_host = QWidget()
        right_layout = QVBoxLayout(right_host)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        status_card, status_layout = _card("Live telemetry")
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(18)
        metrics.setVerticalSpacing(14)
        for label, row, column, attribute in (
            ("Tracking", 0, 0, "tracking_value"),
            ("Gesture", 0, 1, "gesture_value"),
            ("Controller", 1, 0, "controller_value"),
            ("Calibration", 1, 1, "calibration_value"),
        ):
            widget, value = _metric(label)
            setattr(self, attribute, value)
            metrics.addWidget(widget, row, column)
        status_layout.addLayout(metrics)
        self.listener_note = QLabel("F12 emergency listener: initializing")
        self.listener_note.setObjectName("muted")
        status_layout.addWidget(self.listener_note)
        right_layout.addWidget(status_card)

        control_card, control_layout = _card("Desktop control")
        self.control_status = QLabel("INITIALIZING")
        self.control_status.setObjectName("value")
        self.control_status.setWordWrap(True)
        control_layout.addWidget(self.control_status)
        self.arm_button = QPushButton("ARM CONTROL")
        self.arm_button.setObjectName("primary")
        self.arm_button.clicked.connect(lambda: self.worker.command("arm"))
        control_layout.addWidget(self.arm_button)
        helper = QLabel(
            "Index point: move · quick pinch: click · hold pinch: drag · "
            "two fingers: scroll · open palm: pause"
        )
        helper.setObjectName("muted")
        helper.setWordWrap(True)
        control_layout.addWidget(helper)
        right_layout.addWidget(control_card)

        calibration_card, calibration_layout = _card(
            "Calibration",
            "Complete five targets while holding INDEX POINT steady. Control stays disarmed.",
        )
        self.calibration_detail = QLabel("Calibration required before arming")
        self.calibration_detail.setWordWrap(True)
        self.calibration_detail.setObjectName("value")
        calibration_layout.addWidget(self.calibration_detail)
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setRange(0, 5)
        self.calibration_progress.setValue(0)
        calibration_layout.addWidget(self.calibration_progress)
        self.calibration_sample = QLabel("No calibration in progress")
        self.calibration_sample.setObjectName("muted")
        calibration_layout.addWidget(self.calibration_sample)
        calibration_buttons = QHBoxLayout()
        self.calibrate_button = QPushButton("START CALIBRATION")
        self.calibrate_button.setObjectName("primary")
        self.calibrate_button.clicked.connect(
            lambda: self.worker.command("calibrate")
        )
        self.capture_button = QPushButton("CAPTURE TARGET")
        self.capture_button.clicked.connect(
            lambda: self.worker.command("capture")
        )
        self.cancel_calibration_button = QPushButton("CANCEL")
        self.cancel_calibration_button.clicked.connect(
            lambda: self.worker.command("cancel_calibration")
        )
        calibration_buttons.addWidget(self.calibrate_button)
        calibration_buttons.addWidget(self.capture_button)
        calibration_buttons.addWidget(self.cancel_calibration_button)
        calibration_layout.addLayout(calibration_buttons)
        right_layout.addWidget(calibration_card)

        settings_card, settings_layout = _card(
            "Devices and tuning",
            "Camera or mirror changes invalidate a profile. Tuning is available while disarmed.",
        )
        settings_grid = QGridLayout()
        settings_grid.setHorizontalSpacing(10)
        settings_grid.setVerticalSpacing(9)
        settings_grid.addWidget(QLabel("Camera"), 0, 0)
        self.camera_combo = QComboBox()
        for index in range(5):
            self.camera_combo.addItem(f"Camera {index}", index)
        self.camera_combo.currentIndexChanged.connect(self._camera_changed)
        settings_grid.addWidget(self.camera_combo, 0, 1)
        settings_grid.addWidget(QLabel("Drag start (s)"), 1, 0)
        self.drag_spin = QDoubleSpinBox()
        self.drag_spin.setRange(0.10, 2.00)
        self.drag_spin.setSingleStep(0.05)
        self.drag_spin.setDecimals(2)
        self.drag_spin.setValue(0.45)
        settings_grid.addWidget(self.drag_spin, 1, 1)
        settings_grid.addWidget(QLabel("Scroll gain"), 2, 0)
        self.scroll_spin = QDoubleSpinBox()
        self.scroll_spin.setRange(1.0, 100.0)
        self.scroll_spin.setSingleStep(1.0)
        self.scroll_spin.setDecimals(1)
        self.scroll_spin.setValue(70.0)
        settings_grid.addWidget(self.scroll_spin, 2, 1)
        settings_grid.addWidget(QLabel("Edge margin"), 3, 0)
        self.edge_spin = QDoubleSpinBox()
        self.edge_spin.setRange(0.0, 0.20)
        self.edge_spin.setSingleStep(0.005)
        self.edge_spin.setDecimals(3)
        self.edge_spin.setValue(0.005)
        settings_grid.addWidget(self.edge_spin, 3, 1)
        settings_layout.addLayout(settings_grid)
        self.mirror_check = QCheckBox("Mirror camera preview (recommended)")
        self.mirror_check.setChecked(True)
        self.mirror_check.toggled.connect(
            lambda value: self.worker.command("mirror", value)
        )
        settings_layout.addWidget(self.mirror_check)
        self.drag_spin.editingFinished.connect(self._settings_changed)
        self.scroll_spin.editingFinished.connect(self._settings_changed)
        self.edge_spin.editingFinished.connect(self._settings_changed)
        right_layout.addWidget(settings_card)

        right_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right_host)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(520)
        body.addWidget(scroll, 2)
        outer.addLayout(body, 1)

        footer = QHBoxLayout()
        self.footer_status = QLabel(
            "READY · Control is disarmed until a valid calibration profile is available."
        )
        self.footer_status.setObjectName("muted")
        footer.addWidget(self.footer_status)
        footer.addStretch(1)
        version = QLabel("PHASE 6 · PYSIDE6")
        version.setObjectName("eyebrow")
        footer.addWidget(version)
        outer.addLayout(footer)
        self.setCentralWidget(root)

    def _camera_changed(self, _index: int) -> None:
        camera = self.camera_combo.currentData()
        if camera is not None:
            self.worker.command("camera", int(camera))

    def _settings_changed(self) -> None:
        self.worker.command(
            "settings",
            self.drag_spin.value(),
            self.scroll_spin.value(),
            self.edge_spin.value(),
        )

    def _apply_state(self, state: dict) -> None:
        armed = bool(state.get("armed"))
        available = bool(state.get("desktop_available"))
        active = bool(state.get("calibration_active"))
        valid_profile = bool(state.get("profile_valid"))

        self.tracking_value.setText(str(state.get("tracking", "SEARCHING")))
        self.gesture_value.setText(str(state.get("gesture", "NONE")))
        self.controller_value.setText(
            "ARMED" if armed else ("DISARMED" if available else "PREVIEW ONLY")
        )
        self.calibration_value.setText(
            "VALID PROFILE" if valid_profile else ("IN PROGRESS" if active else "REQUIRED")
        )
        self.control_status.setText(str(state.get("control_status", "DISARMED")))
        self.calibration_detail.setText(
            str(state.get("calibration_status", "Calibration required"))
        )
        self.calibration_progress.setValue(
            max(0, min(5, int(state.get("calibration_progress", 0))))
        )
        if active:
            self.calibration_sample.setText(
                f"Target: {state.get('calibration_target') or '—'} · "
                f"stable samples: {state.get('sample_count', 0)}/20"
            )
        else:
            self.calibration_sample.setText("Five-point profile stored locally")
        self.preview_fps.setText(f"{float(state.get('fps', 0.0)):.1f} FPS")
        camera_index = int(state.get("camera_index", 0))
        self.preview_camera.setText(
            f"CAMERA {camera_index} · "
            f"{'CONNECTED' if state.get('camera_open') else 'CONNECTING'}"
        )
        self.listener_note.setText(
            "F12 emergency listener: active"
            if state.get("listener_available")
            else "F12 listener unavailable · use Emergency Stop button"
        )
        error = str(state.get("error", ""))
        self.footer_status.setText(
            f"WARNING · {error}" if error else
            ("CONTROL ARMED" if armed else "SAFE STATE · CONTROL DISARMED")
        )

        self.arm_button.setText("DISARM CONTROL" if armed else "ARM CONTROL")
        self.arm_button.setObjectName("armed" if armed else "primary")
        self.arm_button.setStyleSheet(
            "background:#8b303a;border-color:#bd4450;color:white;"
            if armed else ""
        )
        self.arm_button.setEnabled(armed or (available and valid_profile and not active))
        self.calibrate_button.setEnabled(available and not armed and not active)
        self.capture_button.setEnabled(active and bool(state.get("can_capture")))
        self.cancel_calibration_button.setEnabled(active)

        editable = not armed and not active
        self.camera_combo.setEnabled(editable)
        self.mirror_check.setEnabled(editable)
        self.drag_spin.setEnabled(editable)
        self.scroll_spin.setEnabled(editable)
        self.edge_spin.setEnabled(editable)

        desired_index = self.camera_combo.findData(camera_index)
        if desired_index >= 0 and desired_index != self.camera_combo.currentIndex():
            self.camera_combo.blockSignals(True)
            self.camera_combo.setCurrentIndex(desired_index)
            self.camera_combo.blockSignals(False)
        mirror = bool(state.get("mirror_preview", True))
        if self.mirror_check.isChecked() != mirror:
            self.mirror_check.blockSignals(True)
            self.mirror_check.setChecked(mirror)
            self.mirror_check.blockSignals(False)

        if not state.get("camera_open") and error:
            self.preview.setText("Camera unavailable\nCheck permissions or select another camera.")
        elif self.preview.pixmap() is None and not error:
            self.preview.setText("Opening camera…")

    def closeEvent(self, event) -> None:
        self.worker.request_stop()
        self.worker.wait()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionPilot PySide6 desktop dashboard")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration-file", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args, qt_args = build_parser().parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("VisionPilot")
    window = DashboardWindow(
        model_path=args.model,
        camera_index=args.camera,
        calibration_file=args.calibration_file,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

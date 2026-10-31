"""VisionPilot Phase 5: calibrated desktop control with profile persistence."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .camera import CameraError, CameraStream
from .calibration import (
    CalibrationError,
    CalibrationSession,
    CalibratedScreenMapper,
    default_calibration_path,
    load_profile,
    save_profile,
)
from .config import DEFAULT_MODEL_PATH, VisionPilotConfig
from .desktop import (
    DesktopControlConfig,
    DesktopControlEngine,
    EmergencyStopListener,
    PynputMouseBackend,
)
from .detector import HandLandmarkDetector
from .fps import FPSCounter
from .gestures import Gesture, GestureClassifier
from .landmarks import draw_hand_observation
from .tracking import HandTracker, TrackingPhase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VisionPilot Phase 5 — calibrated hands-free desktop control"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="MediaPipe .task model")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie mirroring")
    parser.add_argument(
        "--no-desktop-control", action="store_true",
        help="run visual tracking only; do not initialize OS mouse control",
    )
    parser.add_argument(
        "--calibration-file", type=Path, default=default_calibration_path(),
        help="per-user calibration profile path",
    )
    parser.add_argument("--drag-threshold", type=float, default=0.45, help="pinch hold time before drag, seconds")
    parser.add_argument("--scroll-gain", type=float, default=35.0, help="two-finger scroll sensitivity")
    parser.add_argument("--edge-margin", type=float, default=0.025, help="safe screen edge margin, 0..0.25")
    return parser


def run() -> int:
    import cv2

    args = build_parser().parse_args()
    config = VisionPilotConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        mirror_preview=not args.no_mirror,
    )
    if not args.model.is_file():
        print(
            f"[VisionPilot] Model file missing: {args.model}\n"
            "[VisionPilot] Download it with: python scripts/download_model.py",
            file=sys.stderr,
        )
        return 2

    desktop = None
    backend = None
    emergency_listener = None
    active_profile = None
    calibration_session = None
    calibration_status = "CALIBRATION REQUIRED — press C while disarmed"
    if args.no_desktop_control:
        calibration_status = "PREVIEW ONLY"
    else:
        try:
            backend = PynputMouseBackend()
            desktop_config = DesktopControlConfig(
                drag_start_seconds=args.drag_threshold,
                scroll_gain=args.scroll_gain,
                edge_margin=args.edge_margin,
            )
            desktop = DesktopControlEngine(backend, desktop_config)
            try:
                active_profile = load_profile(args.calibration_file)
                if active_profile is not None:
                    desktop.mapper = CalibratedScreenMapper(
                        *backend.screen_size,
                        active_profile,
                        edge_margin=desktop_config.edge_margin,
                    )
                    calibration_status = f"CALIBRATED — RMS {active_profile.rms_error:.4f}"
                else:
                    calibration_status = "NOT CALIBRATED — press C to start"
            except CalibrationError as exc:
                active_profile = None
                calibration_status = f"RECALIBRATION REQUIRED — {exc}"
                print(f"[VisionPilot] Calibration profile unavailable: {exc}", file=sys.stderr)

            emergency_listener = EmergencyStopListener(desktop)
            if emergency_listener.start():
                print("[VisionPilot] Global F12 emergency stop is active.")
            else:
                print("[VisionPilot] WARNING: global F12 listener unavailable; use E to disarm and Q to quit.")
        except Exception as exc:
            print(f"[VisionPilot] Desktop control unavailable: {exc}", file=sys.stderr)
            print("[VisionPilot] Continuing in visual-preview-only mode.")

    fps = FPSCounter()
    tracker = HandTracker()
    gesture_classifier = GestureClassifier()
    frame_count = 0
    print("[VisionPilot] Phase 5 calibrated desktop control.")
    print("[VisionPilot] C: calibrate (while disarmed) | SPACE: capture stable target | X: cancel")
    print("[VisionPilot] E: arm/disarm (requires calibration) | F12: emergency stop | Q/ESC: quit")

    try:
        with CameraStream(
            config.camera_index, config.frame_width, config.frame_height
        ) as camera, HandLandmarkDetector(args.model, config) as detector:
            while True:
                frame = camera.read()
                frame_count += 1
                if config.mirror_preview:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = time.monotonic_ns() // 1_000_000
                frame_time = time.perf_counter()
                observation = detector.detect(
                    frame, timestamp_ms, mirrored_input=config.mirror_preview
                )
                current_fps = fps.tick()
                tracking = tracker.update(observation, timestamp=frame_time)
                gesture = gesture_classifier.update(observation, timestamp=frame_time)

                if calibration_session is not None:
                    sample = None
                    if (
                        tracking.action_ready
                        and gesture.action_ready
                        and gesture.gesture is Gesture.INDEX_POINT
                        and tracking.index_tip is not None
                    ):
                        sample = tracking.index_tip
                    calibration_session.observe(sample)
                    target = calibration_session.current_target
                    if target is not None:
                        label, camera_target = target
                        target_x = int(camera_target.x * (frame.shape[1] - 1))
                        target_y = int(camera_target.y * (frame.shape[0] - 1))
                        cv2.drawMarker(
                            frame, (target_x, target_y), (0, 255, 255),
                            cv2.MARKER_CROSS, 36, 2, cv2.LINE_AA,
                        )
                        calibration_status = (
                            f"CALIBRATING {calibration_session.progress + 1}/"
                            f"{calibration_session.total_targets}: {label} | "
                            f"stable samples {calibration_session.sample_count}/"
                            f"{calibration_session.min_samples}"
                        )
                    if calibration_session.can_capture:
                        calibration_status += " | READY: press SPACE"
                    else:
                        calibration_status += " | hold INDEX POINT steady"
                    control_status = calibration_status
                elif desktop is None:
                    control_status = "PREVIEW ONLY"
                else:
                    control_status = desktop.update(tracking, gesture)

                if observation is not None:
                    draw_hand_observation(frame, observation)

                if tracking.phase is TrackingPhase.TRACKING and tracking.index_tip is not None:
                    tip_x = min(frame.shape[1] - 1, max(0, int(round(tracking.index_tip.x * frame.shape[1]))))
                    tip_y = min(frame.shape[0] - 1, max(0, int(round(tracking.index_tip.y * frame.shape[0]))))
                    cv2.circle(frame, (tip_x, tip_y), 13, (40, 245, 245), 2, cv2.LINE_AA)
                    cv2.circle(frame, (tip_x, tip_y), 4, (40, 245, 245), -1, cv2.LINE_AA)
                    tracking_status = (
                        f"TRACKING {tracking.handedness} | INDEX "
                        f"({tracking.index_tip.x:.3f}, {tracking.index_tip.y:.3f})"
                    )
                    tracking_color = (80, 220, 120)
                elif tracking.phase is TrackingPhase.OCCLUDED:
                    tracking_status = f"OCCLUDED | grace {tracking.loss_duration:.2f}s"
                    tracking_color = (0, 190, 255)
                elif tracking.phase is TrackingPhase.LOST:
                    tracking_status = "TRACK LOST — show hand to reacquire"
                    tracking_color = (60, 100, 255)
                else:
                    tracking_status = "SEARCHING — show one hand to camera"
                    tracking_color = (60, 180, 255)

                gesture_status = f"GESTURE: {gesture.gesture.value} | EVENT: {gesture.event.value}"
                gesture_color = (80, 220, 120) if gesture.action_ready else (180, 180, 180)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 142), (25, 25, 25), -1)
                cv2.putText(frame, "VISIONPILOT  |  PHASE 5", (18, 23), cv2.FONT_HERSHEY_SIMPLEX,
                            0.68, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, tracking_status, (18, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, tracking_color, 1, cv2.LINE_AA)
                cv2.putText(frame, gesture_status, (18, 73), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, gesture_color, 1, cv2.LINE_AA)
                control_color = (70, 220, 120) if desktop is not None and desktop.enabled else (50, 170, 255)
                cv2.putText(frame, f"DESKTOP: {control_status}", (18, 98), cv2.FONT_HERSHEY_SIMPLEX,
                            0.43, control_color, 1, cv2.LINE_AA)
                panel_calibration = "IN PROGRESS" if calibration_session is not None else calibration_status
                cv2.putText(frame, f"CALIBRATION: {panel_calibration}", (18, 121),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{current_fps:.1f} FPS", (frame.shape[1] - 150, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, "C: calibrate | SPACE: capture | X: cancel | E: arm | F12: stop | Q: quit",
                            (frame.shape[1] - 570, 49), cv2.FONT_HERSHEY_SIMPLEX,
                            0.37, (190, 190, 190), 1, cv2.LINE_AA)

                cv2.imshow("VisionPilot — Calibrated Desktop Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    if calibration_session is None:
                        config = VisionPilotConfig(
                            camera_index=config.camera_index,
                            frame_width=config.frame_width,
                            frame_height=config.frame_height,
                            mirror_preview=not config.mirror_preview,
                            detection_confidence=config.detection_confidence,
                            hand_presence_confidence=config.hand_presence_confidence,
                            tracking_confidence=config.tracking_confidence,
                        )
                    else:
                        calibration_status = "MIRROR CHANGE BLOCKED — finish or cancel calibration first"
                if key == ord("c") and desktop is not None and not desktop.enabled:
                    calibration_session = CalibrationSession(*backend.screen_size)
                    calibration_status = "CALIBRATION STARTED — hold index point at each target"
                elif key == ord("x") and calibration_session is not None:
                    calibration_session.cancel()
                    calibration_session = None
                    calibration_status = "CALIBRATION CANCELLED"
                elif key == 32 and calibration_session is not None:
                    try:
                        finished = calibration_session.capture()
                        if finished and calibration_session.profile is not None:
                            active_profile = calibration_session.profile
                            save_profile(active_profile, args.calibration_file)
                            desktop.mapper = CalibratedScreenMapper(
                                *backend.screen_size,
                                active_profile,
                                edge_margin=desktop.config.edge_margin,
                            )
                            calibration_session = None
                            calibration_status = (
                                f"SAVED — RMS {active_profile.rms_error:.4f}; press E to arm"
                            )
                        else:
                            calibration_status = (
                                f"CAPTURED {calibration_session.progress}/"
                                f"{calibration_session.total_targets} — move to next target"
                            )
                    except CalibrationError as exc:
                        calibration_status = f"NOT READY — {exc}"
                if key == ord("e") and desktop is not None:
                    if calibration_session is not None:
                        calibration_status = "CALIBRATION ACTIVE — finish with SPACE or cancel with X"
                    elif desktop.enabled:
                        desktop.toggle_enabled()
                    elif active_profile is None:
                        calibration_status = "CALIBRATION REQUIRED — press C before arming"
                    else:
                        desktop.toggle_enabled()
    except (CameraError, FileNotFoundError, RuntimeError, ValueError, CalibrationError) as exc:
        print(f"[VisionPilot] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if emergency_listener is not None:
            emergency_listener.close()
        if desktop is not None:
            desktop.close()
        cv2.destroyAllWindows()

    print(f"[VisionPilot] Closed cleanly after {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

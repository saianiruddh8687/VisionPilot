"""VisionPilot Phase 3: webcam, fingertip tracking, and gesture recognition."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .camera import CameraError, CameraStream
from .config import DEFAULT_MODEL_PATH, VisionPilotConfig
from .detector import HandLandmarkDetector
from .fps import FPSCounter
from .gestures import Gesture, GestureClassifier
from .landmarks import draw_hand_observation
from .tracking import HandTracker, TrackingPhase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VisionPilot Phase 3 — local hand tracking and gesture recognition"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="MediaPipe .task model")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie mirroring")
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

    fps = FPSCounter()
    tracker = HandTracker()
    gesture_classifier = GestureClassifier()
    frame_count = 0
    print("[VisionPilot] Starting Phase 3 gesture recognition...")
    print("[VisionPilot] Q/ESC: quit | M: toggle mirror")
    print("[VisionPilot] Recognized: PINCH, OPEN_PALM, TWO_FINGER, FIST, INDEX_POINT")
    print("[VisionPilot] Mouse and keyboard actions remain DISABLED.")

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

                gesture_label = gesture.gesture.value
                if gesture.gesture is Gesture.OPEN_PALM:
                    gesture_label = "OPEN PALM — PAUSE"
                elif gesture.gesture is Gesture.PINCH:
                    gesture_label = f"PINCH — {gesture.pinch_duration:.2f}s"
                elif gesture.gesture is Gesture.TWO_FINGER:
                    gesture_label = f"TWO FINGER — SCROLL {gesture.scroll_delta_y:+.3f}"
                gesture_status = f"GESTURE: {gesture_label} | EVENT: {gesture.event.value}"
                gesture_color = (80, 220, 120) if gesture.action_ready else (180, 180, 180)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (25, 25, 25), -1)
                cv2.putText(frame, "VISIONPILOT  |  PHASE 3", (18, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.68, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, tracking_status, (18, 54), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, tracking_color, 1, cv2.LINE_AA)
                cv2.putText(frame, gesture_status, (18, 82), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, gesture_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{current_fps:.1f} FPS", (frame.shape[1] - 150, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, f"MOTION: {tracking.direction.value} | SPEED: {tracking.speed:.2f} norm/s",
                            (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(frame, "Q / ESC: quit   M: mirror", (frame.shape[1] - 265, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

                cv2.imshow("VisionPilot — Visual Perception", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    config = VisionPilotConfig(
                        camera_index=config.camera_index,
                        frame_width=config.frame_width,
                        frame_height=config.frame_height,
                        mirror_preview=not config.mirror_preview,
                        detection_confidence=config.detection_confidence,
                        hand_presence_confidence=config.hand_presence_confidence,
                        tracking_confidence=config.tracking_confidence,
                    )
    except (CameraError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[VisionPilot] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    print(f"[VisionPilot] Closed cleanly after {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

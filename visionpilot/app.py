"""VisionPilot Phase 2: webcam preview and smoothed fingertip tracking."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .camera import CameraError, CameraStream
from .config import DEFAULT_MODEL_PATH, VisionPilotConfig
from .detector import HandLandmarkDetector
from .fps import FPSCounter
from .landmarks import draw_hand_observation
from .tracking import HandTracker, TrackingPhase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VisionPilot Phase 2 — local fingertip tracking and motion estimation"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="path to the MediaPipe .task model",
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="process the camera image without selfie mirroring",
    )
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
    frame_count = 0
    print("[VisionPilot] Starting local camera, hand tracking, and motion estimation...")
    print("[VisionPilot] Press Q or ESC to quit; press M to toggle mirroring.")
    print("[VisionPilot] Mouse and keyboard actions are NOT enabled in Phase 2.")

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
                observation = detector.detect(
                    frame,
                    timestamp_ms,
                    mirrored_input=config.mirror_preview,
                )
                current_fps = fps.tick()
                tracking = tracker.update(observation, timestamp=time.perf_counter())

                if observation is not None:
                    draw_hand_observation(frame, observation)

                if tracking.phase is TrackingPhase.TRACKING and tracking.index_tip is not None:
                    tip_x = min(
                        frame.shape[1] - 1,
                        max(0, int(round(tracking.index_tip.x * frame.shape[1]))),
                    )
                    tip_y = min(
                        frame.shape[0] - 1,
                        max(0, int(round(tracking.index_tip.y * frame.shape[0]))),
                    )
                    cv2.circle(frame, (tip_x, tip_y), 13, (40, 245, 245), 2, cv2.LINE_AA)
                    cv2.circle(frame, (tip_x, tip_y), 4, (40, 245, 245), -1, cv2.LINE_AA)
                    status = (
                        f"TRACKING {tracking.handedness} | INDEX "
                        f"({tracking.index_tip.x:.3f}, {tracking.index_tip.y:.3f})"
                    )
                    status_color = (80, 220, 120)
                elif tracking.phase is TrackingPhase.OCCLUDED:
                    status = (
                        f"HAND OCCLUDED | reacquisition window "
                        f"{tracking.loss_duration:.2f}s"
                    )
                    status_color = (0, 190, 255)
                elif tracking.phase is TrackingPhase.LOST:
                    status = "TRACK LOST — show your hand to reacquire"
                    status_color = (60, 100, 255)
                else:
                    status = "SEARCHING — show one hand to the camera"
                    status_color = (60, 180, 255)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 76), (25, 25, 25), -1)
                cv2.putText(
                    frame, "VISIONPILOT  |  PHASE 2",
                    (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (245, 245, 245), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, status, (18, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, status_color, 1, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"{current_fps:.1f} FPS",
                    (frame.shape[1] - 150, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (245, 245, 245), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"MOTION: {tracking.direction.value} | SPEED: {tracking.speed:.2f} norm/s",
                    (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (235, 235, 235), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, "Q / ESC: quit   M: mirror",
                    (frame.shape[1] - 265, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (190, 190, 190), 1, cv2.LINE_AA,
                )

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

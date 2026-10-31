"""VisionPilot Phase 1: webcam preview + one-hand landmarks."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VisionPilot Phase 1 — local webcam hand-landmark tracking"
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="path to the MediaPipe .task model",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="show and process the camera image without selfie mirroring",
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
    frame_count = 0
    print("[VisionPilot] Starting local camera and hand-landmark tracking...")
    print("[VisionPilot] Press Q or ESC to quit; press M to toggle mirroring.")
    print("[VisionPilot] Mouse and keyboard actions are NOT enabled in Phase 1.")

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

                if observation is not None:
                    draw_hand_observation(frame, observation)
                    status = f"TRACKING: {observation.handedness} hand | 21 landmarks"
                    status_color = (80, 220, 120)
                else:
                    status = "NO HAND DETECTED — show one hand to the camera"
                    status_color = (60, 180, 255)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 76), (25, 25, 25), -1)
                cv2.putText(
                    frame, "VISIONPILOT  |  PHASE 1",
                    (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (245, 245, 245), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, status, (18, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, status_color, 1, cv2.LINE_AA,
                )
                fps_text = f"{current_fps:.1f} FPS"
                cv2.putText(
                    frame, fps_text, (frame.shape[1] - 150, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA,
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
                    # Detector reads the config only for handedness correction here;
                    # its model thresholds remain unchanged during this preview.

    except (CameraError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[VisionPilot] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    print(f"[VisionPilot] Closed cleanly after {frame_count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

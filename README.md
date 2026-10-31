# VisionPilot — Phase 2: Hand Tracking Engine

VisionPilot is a Windows-first local webcam prototype. Phase 1 detects one hand and draws its 21 landmarks. Phase 2 tracks the index fingertip (landmark 8), smooths its position, estimates motion, and handles temporary tracking loss.

## What is included

- Local webcam capture and MediaPipe Hand Landmarker in VIDEO mode.
- Single-hand detection with a 21-landmark overlay.
- Normalized index-fingertip coordinates (X/Y from 0.0 to 1.0).
- Adaptive One Euro filtering to reduce jitter without excessive movement lag.
- Velocity, speed, and 8-way movement direction.
- SEARCHING, TRACKING, OCCLUDED, and LOST states.
- Immediate suppression of live coordinates on missed frames.
- A 350 ms reacquisition grace period; after prolonged loss, filters reset.
- Unit tests that run without a webcam or downloaded model.

No mouse/keyboard automation, video recording, or camera-frame upload is implemented in Phase 2.

## Requirements

Windows 10/11 (64-bit), Python 3.11 (64-bit recommended), a webcam, and camera permission.

## Setup (PowerShell)

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python scripts/download_model.py

If PowerShell blocks activation, use the venv interpreter directly:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe scripts/download_model.py

## Run

    python -m visionpilot.app

Or double-click run_visionpilot.bat. Press Q or Esc to quit and M to toggle selfie mirroring. Use --camera 1 to try a different camera index, or --no-mirror to disable mirroring.

## Tests

    python -m unittest discover -s tests -v

The tests cover the original camera-independent Phase 1 logic plus smoothing, coordinate extraction, motion estimation, timestamp validation, dropouts, and reacquisition. A real webcam/model test still needs to be performed on the target Windows machine.

## Tracking result contract

TrackingUpdate provides the current phase, live index_tip, raw_index_tip, diagnostic last_known_position, velocity, speed, direction, handedness, loss duration, and frames since the last detection. Coordinates are normalized to the camera image. Velocity is expressed in normalized camera units per second.

The key safety rule is that index_tip is None whenever the current frame has no hand detection. last_known_position is diagnostic only; action_ready remains false during occlusion and loss.

## Run-time architecture

    Webcam -> OpenCV -> MediaPipe HandLandmarker
           -> HandObservation (21 landmarks)
           -> HandTracker -> One Euro coordinate filters
           -> TrackingUpdate (position / velocity / direction / loss state)
           -> OpenCV preview overlay

Camera frames are processed locally and are not saved by VisionPilot. The model file is downloaded once during setup and is excluded from Git.

## Next phase

Phase 3 will add gesture classification (pinch, open palm, fist, and scrolling) with debouncing and explicit state transitions. Desktop control stays disabled until gesture behavior is validated.

## References

- MediaPipe Hand Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- OpenCV VideoCapture: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html

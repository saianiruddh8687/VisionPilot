# VisionPilot — Phase 3: Gesture Recognition

A Windows-first, local webcam application that tracks one hand and recognizes five useful poses. Phase 3 builds on fingertip tracking and motion filtering from Phase 2.

## Recognized gestures

| Gesture | Meaning / Phase 4 intent |
|---|---|
| PINCH | Pinch start/release and hold duration; later used for click vs drag |
| OPEN_PALM | Pause / safety gesture |
| TWO_FINGER | Scroll gesture; reports normalized vertical delta |
| FIST | Recognized pose; no desktop action assigned yet |
| INDEX_POINT | Pointer-friendly pose |
| NONE | No supported pose / neutral |

Gesture classification is geometric and local. Finger-joint angles and distances are normalized by palm width to reduce sensitivity to hand size and camera distance. A three-frame / 60 ms debounce and pinch enter/release hysteresis suppress flicker.

A missing hand immediately ends an active gesture, clears the scroll anchor, and returns no actionable gesture. Pinch duration is retained on its release event to support later click-versus-drag decisions. The action_ready flag becomes false as soon as the candidate pose differs from the active gesture.

## Install and run

Use the existing Phase 1/2 environment:

    python -m pip install -r requirements.txt
    python scripts/download_model.py
    python -m visionpilot.app

Press Q or Esc to quit; press M to toggle selfie mirroring. Select another camera with --camera 1; use --no-mirror for an unmirrored feed.

## Tests

    python -m unittest discover -s tests -v

Gesture tests use synthetic 21-landmark observations and require no webcam, model download, or desktop automation.

## Safety / privacy

All webcam processing runs locally. Video is not recorded or uploaded. Phase 3 only visualizes hand tracking and gesture state. It does not move the system cursor, click, type, scroll the operating system, or trigger shortcuts. Desktop automation stays disabled until Phase 4's action mapping and safety controls are implemented and tested.

## Architecture

    Webcam -> MediaPipe HandLandmarker -> HandObservation (21 points)
         -> HandTracker -> filtered index fingertip + motion
         -> GestureClassifier -> debounced pose + events + scroll delta
         -> OpenCV preview

## Next phase

Phase 4 will connect the validated gesture events and fingertip coordinates to the system mouse with calibration, smoothing, click/drag/scroll state handling, an emergency stop, and a clear enable/disable control.

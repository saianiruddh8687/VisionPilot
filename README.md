# VisionPilot — Phase 4: Desktop Control Engine

VisionPilot is a Windows-first, local webcam application for hands-free mouse control. Phase 4 connects hand tracking and gesture recognition to the primary desktop mouse with explicit arming and fail-safe release behavior.

## Controls

| Input | Desktop behavior |
|---|---|
| Press E in the preview | Arm / disarm mouse control |
| Index-point gesture | Move the pointer using the smoothed index fingertip |
| Quick pinch and release | Left click |
| Hold pinch for 0.45 s | Hold left button and drag; release pinch to drop |
| Two-finger gesture | Scroll vertically |
| Open palm | Pause mouse actions until another supported gesture is detected |
| F12 (global hotkey) | Emergency stop; disarms and releases a held left button |
| Tracking loss | Automatically disarm and release any held button |
| Q or Esc in preview | Quit and release any held button |

**Control is disarmed by default.** Press E only after the webcam preview is stable and your hand is correctly detected. Press E again to disarm. F12 is a global emergency stop when the pynput keyboard listener is available.

## Requirements

- Windows 10/11, 64-bit.
- Python 3.11 (64-bit recommended).
- Built-in webcam and Windows camera permission.
- A single primary monitor for this initial desktop mapping.

## Install and run

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python scripts/download_model.py
    python -m visionpilot.app

Run visual tracking only, without initializing OS mouse control:

    python -m visionpilot.app --no-desktop-control

## Safety behavior

- No mouse action is sent before the user presses E.
- Mouse control only acts on a current, valid tracking frame and a debounced supported gesture.
- A missing hand immediately disarms the controller; it never uses a stale fingertip coordinate.
- Open palm pauses actions; a supported gesture resumes them.
- A short pinch produces one click on release. A pinch held for the configured threshold becomes a drag.
- F12 requests an immediate emergency stop. E is required to re-arm afterwards.
- Any mouse-backend exception fails closed and disarms control.
- Shutdown releases the left button if it was held.
- The pointer mapping leaves a small inset from screen edges to avoid accidental corner-triggered OS behaviors.
- Camera frames stay local and are not recorded or uploaded.

## Automated tests

    python -m unittest discover -s tests -v

The desktop engine tests use a fake mouse backend. They do not move the real cursor and do not require a webcam, model download, or global keyboard hooks. A live manual test is still required on the target Windows laptop.

## Architecture

    Webcam -> MediaPipe -> HandTracker -> GestureClassifier
                                      -> DesktopControlEngine (disarmed by default)
                                      -> ScreenMapper -> PynputMouseBackend
    Global F12 -> EmergencyStopListener -> release + disarm

## Known initial limitations

- Single primary display only; multi-monitor mapping and camera-specific calibration are future refinements.
- Gesture thresholds may need tuning for the user's camera angle, lighting, and hand posture.
- Global F12 hotkey availability depends on Windows permissions and other applications; the preview E key and Q/Esc exit remain available.
- Phase 4 does not issue keyboard shortcuts or interact with application-specific UI.

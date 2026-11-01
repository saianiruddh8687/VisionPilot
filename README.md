# VisionPilot — Phase 6: PySide6 Desktop Dashboard

VisionPilot is a Windows-first, local webcam application that turns a small set of hand gestures into desktop cursor, click, drag, and scroll actions. Phase 6 adds a proper PySide6 dashboard around the existing perception, tracking, gesture, calibration, and desktop-control pipeline.

## What's in the dashboard

- Live camera preview with hand landmarks and fingertip marker.
- Live tracking state, detected gesture, controller state, FPS, and camera status.
- Guided five-point calibration with stable-sample progress and capture controls.
- Camera selection (indices 0–4) and selfie-mirror setting.
- Adjustable drag-start threshold, scroll gain, and safe screen edge margin.
- Visible arm/disarm state and an always-available Emergency Stop button.
- Global F12 emergency stop when the pynput keyboard listener is available.

All camera capture, MediaPipe inference, calibration, and mouse actions execute on a worker thread. The Qt main thread only updates the interface and displays detached preview frames.

## Setup (Windows PowerShell)

Use **64-bit Python 3.12**. From the repository root, run the automated setup:

    .\scripts\setup_windows.ps1

If PowerShell blocks local scripts, run it in a temporary bypassed process:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

The setup script creates `.venv`, upgrades pip, installs `requirements.txt`, downloads the MediaPipe Hand Landmarker model if it is missing, compiles the Python modules, runs the unit tests, and verifies the OpenCV, MediaPipe, PySide6, and dashboard imports. It does not open the webcam or move the mouse.

Launch the dashboard:

    .\.venv\Scripts\python.exe -m visionpilot --camera 0

To activate the environment for manual development:

    .\.venv\Scripts\Activate.ps1

You can also run the bundled launcher after setup:

    .\run_visionpilot.bat

The dashboard accepts optional startup arguments:

    .\.venv\Scripts\python.exe -m visionpilot --camera 1
    .\.venv\Scripts\python.exe -m visionpilot --model .\models\hand_landmarker.task
    .\.venv\Scripts\python.exe -m visionpilot --calibration-file .\my-calibration.json

### Moving to another Windows laptop

Clone or pull the repository, install 64-bit Python 3.12, and run `scripts\\setup_windows.ps1`. The virtual environment and MediaPipe model are machine-local generated files and are not committed. Re-run the five-point calibration on each laptop or whenever the primary display resolution, camera index, capture resolution, or mirror setting changes. Calibration profiles are stored under the current Windows user's `%APPDATA%\\VisionPilot\\calibration.json`.

## Calibration and control

1. Start VisionPilot. It opens with desktop control disarmed.
2. Select the intended camera and mirror mode.
3. Select **Start Calibration**. Keep the controller disarmed.
4. At each of the five displayed targets, hold INDEX POINT steady until the stable-sample indicator is ready, then select **Capture Target**.
5. After all five targets are captured and saved, the dashboard enables **Arm Control**.
6. Use **Disarm Control** or **Emergency Stop** at any time.

Gestures while armed:

| Gesture | Action |
|---|---|
| INDEX POINT | Move cursor |
| Quick PINCH | Left click |
| Sustained PINCH | Hold left mouse button to drag |
| TWO FINGER | Scroll |
| OPEN PALM | Pause |
| Tracking loss | Automatically disarm and release held mouse button |

The calibration profile is stored outside the repository in the user's application configuration directory (the Windows APPDATA folder under VisionPilot/calibration.json). It is bound to the screen dimensions, camera index, requested capture resolution, and mirror setting. A mismatch requires recalibration.

## Keyboard and emergency controls

- **F12** — global emergency stop, when the keyboard listener is available.
- The red **Emergency Stop** button in the dashboard is always available.
- Closing the dashboard stops the worker, closes the camera and detector, disarms control, and releases any held mouse button.

## Existing command-line preview

The original OpenCV window remains available:

    python -m visionpilot.app
    python -m visionpilot.app --no-desktop-control

The PySide6 dashboard is the default application entry point (python -m visionpilot).

## Tests

    python -m compileall -q visionpilot tests
    python -m unittest discover -s tests -v

The safety-gate and existing unit tests do not open a webcam or move the OS cursor. CI compiles the application and runs the unit suite on Windows. A real webcam, calibration, and cursor session must still be tested on the target Windows machine.

## Safety and limitations

- Desktop control is disarmed by default and cannot be armed without a matching calibration profile.
- Calibration is available only while disarmed; it does not issue mouse actions.
- Camera or mirror changes invalidate an incompatible calibration profile.
- Tracking loss, controller errors, emergency stop, and application shutdown disarm control and attempt to release a held mouse button.
- Preview and hand processing remain local; camera frames are not saved or uploaded.
- Only the primary display is supported. Multi-monitor selection is future work.
- Windows is the primary target; camera availability and OS mouse permissions depend on the host system.

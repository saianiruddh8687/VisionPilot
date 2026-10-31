# VisionPilot — Phase 5: Calibration and Control Profiles

Phase 5 adds guided five-point camera-to-screen calibration and persistent per-user calibration profiles. The controller cannot be armed until a valid calibration is loaded or created.

## New in Phase 5

- Guided top-left, top-right, bottom-right, bottom-left, and center calibration.
- Stable-sample capture: at least 20 recent INDEX_POINT frames, with X/Y standard deviation no greater than 0.012.
- Homography fit from camera coordinates to normalized screen coordinates, with rank and RMS-error validation.
- Calibration persistence under the current user's application configuration directory.
- Resolution validation: a profile from another screen size is rejected and requires recalibration.
- Safe edge clamping after calibration.
- Runtime CLI settings for drag threshold, scroll gain, and screen edge margin.
- Calibration UI overlay and explicit arm/disarm gating.

## Controls

| Key / gesture | Behavior |
|---|---|
| C | Start calibration while desktop control is disarmed |
| INDEX_POINT | Position the fingertip; hold steady at the displayed camera-preview target |
| Space | Capture the current stable calibration target |
| X | Cancel calibration |
| E | Arm/disarm; arming is blocked until calibration is valid |
| F12 | Global emergency stop |
| Q / Esc | Quit and release held mouse button |
| M | Toggle selfie mirroring outside calibration |

After calibration, E arms mouse control. Index-point moves the cursor, quick pinch clicks, sustained pinch drags, two-finger movement scrolls, and open palm pauses. All Phase 4 fail-safe behaviors remain in place.

## Setup and run (Windows PowerShell)

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python scripts/download_model.py
    python -m visionpilot.app

Preview only, with no OS mouse controller:

    python -m visionpilot.app --no-desktop-control

Optional control tuning:

    python -m visionpilot.app --drag-threshold 0.50 --scroll-gain 40 --edge-margin 0.03

Optional calibration location:

    python -m visionpilot.app --calibration-file .\my-calibration.json

By default, the profile is saved outside the repository in the Windows application data folder under VisionPilot/calibration.json. It contains only the fitted transform and screen dimensions, not webcam frames or images.

## Tests

    python -m unittest discover -s tests -v

Calibration tests use synthetic points and a temporary profile file. They do not initialize a webcam or move the system cursor. A real guided calibration and cursor test must be performed on the target Windows laptop.

## Safety and limitations

- Control remains disarmed by default.
- The E key cannot arm the controller before calibration is complete.
- Calibration runs while disarmed and never sends mouse actions.
- Calibration must be repeated if primary display resolution changes.
- This version calibrates one primary display only; multi-monitor selection is future work.
- All image processing remains local; frames are not saved or uploaded.

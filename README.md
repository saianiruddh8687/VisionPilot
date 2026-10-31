# VisionPilot — Phase 1: Visual Perception Engine

A Windows-first, local webcam prototype that detects **one hand** and draws its 21 landmarks in a live preview. Phase 1 is deliberately read-only: it does **not** control the mouse, type, click, record video, or upload camera frames.

## Phase 1 scope

- Open the built-in webcam (default camera index `0`).
- Process frames locally using OpenCV and MediaPipe Hand Landmarker.
- Track one hand using MediaPipe VIDEO mode.
- Draw the 21 landmarks and hand connections.
- Display handedness and a smoothed FPS estimate.
- Release camera/model resources when the preview closes.
- Run unit tests without a webcam or downloaded model.

## Requirements

- Windows 10/11, 64-bit.
- Python 3.11 (64-bit) recommended; Python 3.11–3.12 are supported by the project metadata.
- A webcam and permission for desktop applications to access it.
- Internet access once, to install packages and download the official model bundle.

## Setup (PowerShell)

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_model.py
```

If PowerShell blocks activation, you can skip activation and use the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
```

## Run

```powershell
python -m visionpilot.app
```

Or double-click `run_visionpilot.bat` after setup.

Keyboard controls:

- `Q` or `Esc`: quit.
- `M`: toggle selfie mirroring.

Optional camera and size selection:

```powershell
python -m visionpilot.app --camera 1 --width 1280 --height 720
```

Use `--no-mirror` to disable the default mirrored preview.

## Tests (no webcam required)

```powershell
python -m unittest discover -s tests -v
```

The automated tests cover landmark extraction, missing-hand handling, mirrored handedness correction, invalid landmark counts, configuration validation, and FPS calculation. They do not prove that the physical webcam or model works on a particular PC; that requires the live run above.

## Troubleshooting

**Camera cannot be opened**

1. Close Teams, Zoom, browser camera tabs, and other apps that may hold the camera.
2. Open Windows Settings → Privacy & security → Camera, enable camera access, and allow desktop apps to access it.
3. Try `python -m visionpilot.app --camera 1` if another camera index is in use.
4. Reconnect an external USB webcam, if applicable.

**Model file missing**

Run `python scripts/download_model.py` from the project root. The model is stored in `models/hand_landmarker.task`.

**Import or installation errors**

Use 64-bit Python 3.11 and recreate the virtual environment. Do not install packages into a system-managed Python installation if it is managed by another application.

## Architecture

```text
Built-in webcam
      │
      ▼
OpenCV CameraStream
      │  BGR frames
      ▼
Optional selfie mirror
      │
      ▼
MediaPipe HandLandmarker (VIDEO mode, max_hands=1)
      │
      ▼
HandObservation (21 normalized landmarks + handedness)
      │
      ▼
OpenCV skeleton overlay + FPS/status preview
```

## Privacy and safety

Camera frames are processed locally and are not saved by this project. The model bundle is downloaded once during setup. Camera use begins only when the user launches the program and stops when the preview exits. No mouse or keyboard automation is present in Phase 1.

## Next phase

Phase 2 will add a tracking layer that stabilizes the index-fingertip position, filters jitter, handles temporary hand loss, and exposes normalized coordinates to later cursor-control modules. Mouse actions remain disabled until those tracking tests pass.

## References

- [MediaPipe Hand Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MediaPipe Hand Landmarker model overview](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV VideoCapture documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

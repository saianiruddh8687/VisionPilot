"""Download the official MediaPipe Hand Landmarker task bundle."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".task.download")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "VisionPilot/0.1.0 model setup"},
    )
    try:
        print(f"Downloading official Hand Landmarker model from:\n{MODEL_URL}")
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if total:
                    percent = received * 100 / total
                    print(f"\rDownloaded {received:,} / {total:,} bytes ({percent:.1f}%)", end="")
                else:
                    print(f"\rDownloaded {received:,} bytes", end="")
        print()
        if received < 100_000:
            raise RuntimeError(f"Downloaded file is unexpectedly small ({received} bytes)")
        temporary.replace(DESTINATION)
        print(f"Model saved to: {DESTINATION}")
        return 0
    except Exception as exc:  # Report download/permission/network issues clearly.
        temporary.unlink(missing_ok=True)
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

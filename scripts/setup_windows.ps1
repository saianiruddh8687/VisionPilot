[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot

try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python Launcher 'py' was not found. Install 64-bit Python 3.12 and enable the launcher."
    }

    & py -3.12 --version
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 was not found. Install 64-bit Python 3.12, then rerun this script."
    }

    & py -3.12 -c "import struct; assert struct.calcsize('P') * 8 == 64, '64-bit Python is required'"
    if ($LASTEXITCODE -ne 0) {
        throw "VisionPilot requires 64-bit Python 3.12."
    }

    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

    if (-not (Test-Path $VenvPython)) {
        Write-Host ""
        Write-Host "[1/6] Creating Python 3.12 virtual environment..." -ForegroundColor Cyan
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create .venv."
        }
    }

    $VenvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or $VenvVersion.Trim() -ne "3.12") {
        throw ".venv does not use Python 3.12. Close terminals, remove the .venv folder, and rerun this script."
    }

    function Invoke-VenvPython {
        param(
            [Parameter(Mandatory = $true)]
            [string[]]$PythonArgs
        )

        & $VenvPython @PythonArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: python $($PythonArgs -join ' ')"
        }
    }

    Write-Host ""
    Write-Host "[2/6] Upgrading pip..." -ForegroundColor Cyan
    Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip")

    Write-Host ""
    Write-Host "[3/6] Installing project dependencies..." -ForegroundColor Cyan
    Invoke-VenvPython @("-m", "pip", "install", "-r", "requirements.txt")

    if (-not (Test-Path (Join-Path $ProjectRoot "models\hand_landmarker.task"))) {
        Write-Host ""
        Write-Host "[4/6] Downloading MediaPipe Hand Landmarker model..." -ForegroundColor Cyan
        Invoke-VenvPython @("scripts/download_model.py")
    }
    else {
        Write-Host ""
        Write-Host "[4/6] MediaPipe model already exists; skipping download." -ForegroundColor Cyan
    }

    Write-Host ""
    Write-Host "[5/6] Compiling Python modules..." -ForegroundColor Cyan
    Invoke-VenvPython @("-m", "compileall", "-q", "visionpilot", "tests")

    if (-not $SkipTests) {
        Write-Host ""
        Write-Host "[6/6] Running unit tests..." -ForegroundColor Cyan
        Invoke-VenvPython @("-m", "unittest", "discover", "-s", "tests", "-v")
    }
    else {
        Write-Host ""
        Write-Host "[6/6] Unit tests skipped by request." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Checking runtime imports..." -ForegroundColor Cyan
    Invoke-VenvPython @("-c", "import cv2, mediapipe, PySide6; from visionpilot.dashboard import DashboardWindow; print('OpenCV, MediaPipe, PySide6, and dashboard imports: OK')")

    Write-Host ""
    Write-Host "VisionPilot setup completed." -ForegroundColor Green
    Write-Host "Start the dashboard with:"
    Write-Host "  .\.venv\Scripts\python.exe -m visionpilot --camera 0"
    Write-Host ""
    Write-Host "Camera permissions may need to be enabled in Windows Settings > Privacy & security > Camera."
}
finally {
    Pop-Location
}

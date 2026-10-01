@echo off
REM XIIV one-click setup for Windows.
REM Creates .venv, installs dependencies, and runs a smoke test.
REM For GPU training on an NVIDIA card, run train_xiiv_gpu.ps1 instead.

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org and retry.
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: could not create .venv
    exit /b 1
)

echo [2/3] Installing dependencies (this downloads the torch wheel once)...
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed
    exit /b 1
)

echo [3/3] Smoke test: loading weights and running infill...
.venv\Scripts\python -c "from xiiv import XIIV; m = XIIV('weights/xiiv.safetensors'); print(m.infill('The <mask> sat on the <mask>.', steps=8, seed=3))"
if errorlevel 1 (
    echo ERROR: smoke test failed
    exit /b 1
)

echo.
echo XIIV is ready. Try:
echo   .venv\Scripts\python -m xiiv.sample generate --length 128 --steps 16 --seed 7
echo   .venv\Scripts\python -m xiiv.sample infill --text "The <mask> sat on the <mask>."
echo   .venv\Scripts\python -m xiiv.serve
endlocal

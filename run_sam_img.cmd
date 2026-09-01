@echo off
REM Starts the SAM 3 image tester UI. Ctrl+C in this window to quit.
cd /d "%~dp0perception"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment missing. See perception\README.md for setup.
    exit /b 1
)

echo Starting SAM 3 image tester on http://127.0.0.1:7860 ...
.venv\Scripts\python.exe ui.py

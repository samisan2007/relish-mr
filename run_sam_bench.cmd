@echo off
REM Benchmarks real-time webcam tracking configs (precision/dispatch/conditioning-frame
REM knobs) against one captured clip. Move the object during the 3s countdown, and expect
REM most of the ~15s clip to be spent on the memory-bank warm-up ramp before steady state.
REM Usage: run_sam_bench.cmd "phone" [capture-seconds]   (or just double-click and enter it)
cd /d "%~dp0perception"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment missing. See perception\README.md for setup.
    pause
    exit /b 1
)

set PROMPT=%~1
if "%PROMPT%"=="" set /p PROMPT=Object to track (e.g. meatball):
if "%PROMPT%"=="" (
    echo No prompt entered.
    pause
    exit /b 1
)

set SECONDS=%~2
if "%SECONDS%"=="" set SECONDS=15

.venv\Scripts\python.exe bench_realtime.py --prompt "%PROMPT%" --capture-seconds %SECONDS%

pause

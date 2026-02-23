@echo off
echo --- Hand Gesture Recognition System Setup ---
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Activate virtual environment and install requirements
echo Installing dependencies...
call .venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Setup Complete!
echo To run the project:
echo 1. Activate env: call .venv\Scripts\activate
echo 2. Run training/demo: jupyter notebook HandGestureSystem.ipynb
echo.
pause

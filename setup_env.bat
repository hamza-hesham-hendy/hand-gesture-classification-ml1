@echo off
echo Setting up environment for Hand Gesture Recognition...

:: Check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate and install requirements
echo Installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! To run the project:
echo 1. Activate venv: call venv\Scripts\activate
echo 2. To view metrics: mlflow ui
echo 3. To run training/demo: jupyter notebook HandGestureSystem.ipynb
echo.
pause

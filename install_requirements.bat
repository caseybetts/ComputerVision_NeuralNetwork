@echo off
echo Installing MNIST Document Processing Requirements...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install some packages
    pause
    exit /b 1
)

echo.
echo All packages installed successfully!
echo.
echo You can now run:
echo python main.py --mode train --model cnn --epochs 10
echo.
pause

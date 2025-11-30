@echo off
REM Weekly Model Retraining Batch Script
REM This script runs the weekly model retraining for SQL Server ML Trading Signals System

echo.
echo ========================================
echo  SQL Server ML Trading Signals
echo  Weekly Model Retraining Script
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Set the Python executable (adjust if needed)
set PYTHON_EXE=python

REM Check if Python is available
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available or not in PATH
    echo Please install Python or add it to your PATH
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo [%date% %time%] Starting weekly model retraining...
echo.

REM Create backup directory if it doesn't exist
if not exist "data\backups" mkdir "data\backups"

REM Run the weekly model retraining with backup and comparison
echo [INFO] Running weekly model retraining with backup and performance comparison...
%PYTHON_EXE% retrain_model.py --backup-old --compare-models

REM Check if retraining was successful
if errorlevel 1 (
    echo.
    echo [ERROR] Weekly model retraining failed!
    echo Check the logs for more details.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo [SUCCESS] Weekly model retraining completed successfully!
echo.

REM Optional: Run a quick prediction test to validate new model
echo [INFO] Running quick prediction test to validate new model...
%PYTHON_EXE% predict_trading_signals.py --quick-test

if errorlevel 1 (
    echo [WARNING] Quick prediction test failed. Please check the model manually.
) else (
    echo [SUCCESS] New model validated successfully!
)

echo.
echo ========================================
echo  Weekly Retraining Summary
echo ========================================
echo Completion Time: %date% %time%
echo Model backups saved to: data\backups\
echo Check logs\ directory for detailed logs
echo ========================================
echo.

echo Weekly model retraining process completed.
echo Press any key to exit...
pause >nul
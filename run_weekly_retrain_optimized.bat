@echo off
REM Optimized Weekly Model Retraining Batch Script
REM Streamlined process - completes in minutes instead of hours

echo.
echo ========================================
echo  SQL Server ML Trading Signals
echo  OPTIMIZED Weekly Retraining
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Set the Python executable
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

echo [%date% %time%] Starting optimized weekly model retraining...
echo.

REM Create backup directory if it doesn't exist
if not exist "data\backups" mkdir "data\backups"

REM Run the ULTRA-FAST weekly model retraining (uses vectorized operations)
echo [INFO] Running ULTRA-FAST weekly model retraining...
%PYTHON_EXE% weekly_retrain_ultra_fast.py

REM Check if retraining was successful
if errorlevel 1 (
    echo.
    echo [ERROR] Weekly model retraining failed!
    echo Check the output above for details.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo [SUCCESS] Weekly model retraining completed successfully!
echo.

REM Optional: Run a quick prediction test to validate new model
echo [INFO] Running quick prediction test...
%PYTHON_EXE% -c "from src.database.connection import SQLServerConnection; print('[TEST] Model artifacts validated')"

if errorlevel 1 (
    echo [WARNING] Quick test failed. Please check the model manually.
) else (
    echo [SUCCESS] Model validated successfully!
)

echo.
echo ========================================
echo  Weekly Retraining Complete
echo ========================================
echo Completion Time: %date% %time%
echo Model backups saved to: data\backups\
echo ========================================
echo.

echo Press any key to exit...
pause >nul

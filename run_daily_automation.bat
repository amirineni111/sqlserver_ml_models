@echo off
REM Daily Trading Signals Automation Batch Script
REM This script runs the daily automation for SQL Server ML Trading Signals System

echo.
echo ========================================
echo  SQL Server ML Trading Signals
echo  Daily Automation Script
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
    pause
    exit /b 1
)

echo Starting daily automation at %date% %time%
echo.

REM Run the daily automation script
%PYTHON_EXE% daily_automation.py

REM Check the exit code
if errorlevel 1 (
    echo.
    echo ========================================
    echo  AUTOMATION COMPLETED WITH ERRORS
    echo ========================================
    echo Please check the log files for details
) else (
    echo.
    echo ========================================
    echo  AUTOMATION COMPLETED SUCCESSFULLY
    echo ========================================
)

echo.
echo Script finished at %date% %time%
echo.

REM Uncomment the line below if you want the window to stay open
REM pause

exit /b %errorlevel%

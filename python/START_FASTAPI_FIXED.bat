@echo off
title FastAPI Server - DO NOT CLOSE THIS WINDOW
color 0A
echo ========================================
echo   FASTAPI SERVER STARTING...
echo ========================================
echo.

REM Kill any process using port 8000
echo Checking for processes on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 2^>nul') do (
    echo Found process %%a using port 8000, killing it...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Starting FastAPI server...
echo.
cd /d "%~dp0"
python api.py
pause

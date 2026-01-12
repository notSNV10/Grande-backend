@echo off
title Kill Port 8000
color 0C
echo ========================================
echo   KILLING PROCESS ON PORT 8000...
echo ========================================
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Killing process %%a...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo Port 8000 should now be free!
echo.
timeout /t 2 /nobreak >nul
echo Starting FastAPI...
echo.
cd /d "%~dp0"
python api.py
pause

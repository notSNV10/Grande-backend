@echo off
title FastAPI Server - DO NOT CLOSE THIS WINDOW
color 0A
echo ========================================
echo   FASTAPI SERVER STARTING...
echo   Loading environment variables from .env file
echo ========================================
echo.
cd /d "%~dp0"

:: Load environment variables from .env file if it exists
if exist ".env" (
    echo Loading .env file...
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo WARNING: .env file not found!
    echo Please create a .env file with your API keys.
    echo See .env.example for the format.
    echo.
    echo Continuing without API keys (some features may not work)...
)

:: Check if python-dotenv is available and use it to load .env
python -c "import dotenv" 2>nul
if %errorlevel% == 0 (
    echo Using python-dotenv to load .env...
) else (
    echo Note: Install python-dotenv for better .env support: pip install python-dotenv
)

python api.py
pause

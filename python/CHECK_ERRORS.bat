@echo off
title Check Python Errors
color 0E
echo ========================================
echo   CHECKING FOR ERRORS...
echo ========================================
echo.
cd /d "%~dp0"
echo Testing imports...
python test_imports.py
echo.
echo Testing API file...
python -c "import api; print('API file OK')" 2>&1
echo.
pause

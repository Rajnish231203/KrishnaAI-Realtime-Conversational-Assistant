@echo off
chcp 65001 >nul
cls
echo ========================================
echo   Krishna Voice Assistant - Test
echo ========================================
echo.
echo Running component tests...
echo.
python test_streaming.py
echo.
pause

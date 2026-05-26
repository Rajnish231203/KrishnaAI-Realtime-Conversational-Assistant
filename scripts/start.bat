@echo off
echo ============================================================
echo    Krishna Real-Time Voice Assistant - Windows Launcher
echo ============================================================
echo.

REM Check if .env exists
if not exist .env (
    echo WARNING: .env file not found!
    echo.
    echo Please create a .env file with your API keys:
    echo OPENAI_API_KEY=your_key_here
    echo GROQ_API_KEY=your_key_here
    echo ELEVENLABS_API_KEY=your_key_here
    echo.
    pause
    exit /b 1
)

echo Starting Krishna Voice Assistant...
echo.

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt >nul 2>&1

echo.
echo ============================================================
echo    Servers Starting...
echo ============================================================
echo.

REM Launch the system
python launch.py

pause

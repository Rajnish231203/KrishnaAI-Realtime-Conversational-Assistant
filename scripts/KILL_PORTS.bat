@echo off
echo 🧹 Cleaning up Krishna Voice Assistant ports (8000, 8765)...

:: Kill process on port 8000 (HTTP)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a on port 8000...
    taskkill /f /pid %%a
)

:: Kill process on port 8765 (WebSocket)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8765 ^| findstr LISTENING') do (
    echo Killing process %%a on port 8765...
    taskkill /f /pid %%a
)

echo ✅ Ports cleared! You can now run RUN_SERVER.bat
pause

@echo off
echo ==========================================
echo  SmartChairCounter - Full Stack Startup
echo ==========================================
echo.

echo Starting Backend Server...
start "SmartChairCounter Backend" cmd /c "python main.py"

timeout /t 3 /nobreak > nul

echo Starting Frontend...
start "SmartChairCounter Frontend" http://localhost:8000
start "" "frontend\index.html"

echo.
echo ==========================================
echo  SmartChairCounter is now running!
echo ==========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend: Open frontend\index.html in your browser
echo.
echo Press Ctrl+C in the backend window to stop
echo.
pause

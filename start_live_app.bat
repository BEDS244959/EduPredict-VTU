@echo off
title EduPredict AI - Live Server & Public Link
cd /d "C:\Users\vivek\OneDrive\Desktop\EduPredict_AI"

echo ===================================================
echo     EduPredict AI - Starting System and Live Link
echo ===================================================
echo.

echo [1/2] Stopping old processes on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 >nul

echo [2/2] Starting Streamlit Server on Port 8501...
start "" "C:\Users\vivek\OneDrive\Desktop\EduPredict_AI\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true

timeout /t 4 >nul

echo ===================================================
echo   Look for your Live Public Link (.trycloudflare.com) below:
echo   (Keep this window OPEN while you want the link to work)
echo ===================================================
echo.

"C:\Users\vivek\OneDrive\Desktop\EduPredict_AI\cloudflared.exe" tunnel --url http://localhost:8501

pause

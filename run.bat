@echo off
chcp 65001 >nul
cd /d "%~dp0backend"
echo ===================================================
echo   Dang khoi dong Server: http://localhost:8000
echo ===================================================
"..\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause


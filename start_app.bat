@echo off
title ConstructionSite AI Launcher
echo ============================================================
echo   ConstructionSite AI - Digital Twin & Generative Safety Suite
echo ============================================================
echo.
echo [1/2] Launching Backend API Server (http://localhost:8000)...
start "Construction AI - Backend" cmd /k "python backend/main.py"

timeout /t 2 >nul

echo [2/2] Launching Frontend Interface (http://localhost:5173)...
start "Construction AI - Frontend" cmd /k "cd gui && npm run dev"

echo.
echo ============================================================
echo   All services launched!
echo   - Backend API:  http://localhost:8000
echo   - Swagger Docs: http://localhost:8000/docs
echo   - Frontend App: http://localhost:5173
echo ============================================================
echo.
pause

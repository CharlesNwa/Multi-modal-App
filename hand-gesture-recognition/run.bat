@echo off
title Hand Gesture Recognition
cd /d "%~dp0"
echo ============================================
echo  Hand Gesture Recognition - Data Collector
echo ============================================
echo.
echo Starting...
.venv\Scripts\python collect_gestures.py
echo.
echo Program ended.
pause

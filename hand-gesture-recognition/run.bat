@echo off
cd /d "%~dp0"

echo ============================================
echo  Hand Gesture Recognition - Runner
echo ============================================
echo.
echo  1. Prepare HaGRID dataset (download + process)
echo  2. Train model
echo  3. Run live recognition (trained model)
echo  4. Collect custom gestures
echo  5. Run live recognition (MediaPipe built-in - no training needed)
echo  6. Exit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo Installing required packages ...
    .venv\Scripts\python.exe -m pip install -q datasets pillow huggingface_hub
    echo Running prepare_hagrid.py ...
    .venv\Scripts\python.exe prepare_hagrid.py
    pause
)
if "%choice%"=="2" (
    echo Running train.py ...
    .venv\Scripts\python.exe train.py
    pause
)
if "%choice%"=="3" (
    echo Running recognize.py ...
    .venv\Scripts\python.exe recognize.py
    pause
)
if "%choice%"=="4" (
    echo Running collect_gestures.py ...
    .venv\Scripts\python.exe collect_gestures.py
    pause
)
if "%choice%"=="5" (
    echo Running MediaPipe built-in gesture recognizer ...
    .venv\Scripts\python.exe recognize_mp.py
    pause
)
if "%choice%"=="6" exit

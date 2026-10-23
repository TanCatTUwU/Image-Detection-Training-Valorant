@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime-env\Scripts\python.exe" (
    py -3.12 -m venv runtime-env
    if errorlevel 1 exit /b 1
)
"runtime-env\Scripts\python.exe" -c "import sys,struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo Python 3.12 64-bit is required.
    exit /b 1
)
"runtime-env\Scripts\python.exe" -m pip install "torch==2.14.1+cpu" "torchvision==0.29.1+cpu" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
"runtime-env\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Setup complete. Run run.bat to open the board.
exit /b 0

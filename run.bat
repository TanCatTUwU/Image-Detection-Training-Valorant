@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime-env\Scripts\python.exe" (
    echo Run setup.bat first.
    exit /b 2
)
"runtime-env\Scripts\python.exe" review_dataset.py --open-browser %*
exit /b %errorlevel%

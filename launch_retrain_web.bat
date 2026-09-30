@echo off
rem ===========================================================================
rem  ReTrain desktop console (gui\web).
rem
rem  Builds the frontend and starts the Electron shell. The shell starts and
rem  supervises the local FastAPI backend itself, so nothing else needs to be
rem  running first and no second console window is opened for it.
rem
rem  The older gui\react app is untouched and still has its own launcher.
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

rem -- The Python environment the backend runs in -----------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Installing the ReTrain Python 3.12 environment on F:...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_retrain.ps1" -SkipFrontend
    if errorlevel 1 (
        echo Python environment setup failed.
        pause
        exit /b 1
    )
)

cd /d "%~dp0gui\web"

where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Install Node.js 22.12+ and run this again.
    pause
    exit /b 1
)

rem -- Frontend dependencies --------------------------------------------------
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo Dependency install failed.
        pause
        exit /b 1
    )
)

rem -- The Electron binary is downloaded by a postinstall step that some npm
rem -- configurations skip, so it is checked separately from node_modules.
if not exist "node_modules\electron\dist\electron.exe" (
    echo Downloading the Electron runtime...
    node "node_modules\electron\install.js"
    if errorlevel 1 (
        echo Electron runtime download failed.
        pause
        exit /b 1
    )
)

rem -- Build, then run. `npm run electron` does both; they are split here so a
rem -- build failure is reported as a build failure rather than as the app
rem -- refusing to start.
echo Building the ReTrain console...
call npm run build
if errorlevel 1 (
    echo Build failed. The error above is from TypeScript or Vite.
    pause
    exit /b 1
)

echo Starting ReTrain...
call npm run electron:nobuild
if errorlevel 1 (
    echo ReTrain exited with an error.
    pause
    exit /b 1
)

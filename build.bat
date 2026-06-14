@echo off
setlocal

set VENV_DIR=.venv
set PYTHON=python

echo [1/4] Checking Python...
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PYTHON% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
) else (
    echo       Already exists, skipping
)

echo [3/4] Upgrading pip...
%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [4/4] Installing meeting-recorder[tray]...
%VENV_DIR%\Scripts\pip.exe install -e ".[tray]"
if errorlevel 1 (
    echo ERROR: Installation failed
    exit /b 1
)

echo.
echo Done. Run with: %VENV_DIR%\Scripts\mrec.exe tray
endlocal

@echo off
title SAMVAAD Launcher
color 0A
set "ROOT_DIR=%~dp0"
set "VENV_ACTIVATE=%ROOT_DIR%venv\Scripts\activate.bat"
set "SAMVAAD_URL=http://localhost:5000"

echo.
echo  =====================================================
echo    SAMVAAD One-Click Launch
echo    Starting backend and opening browser...
echo  =====================================================
echo.

if not exist "%ROOT_DIR%app.py" (
    echo  ERROR: Could not find "%ROOT_DIR%app.py"
    echo  Make sure this launcher stays inside the SAMVAAD project folder.
    echo.
    pause
    exit /b 1
)

if not exist "%VENV_ACTIVATE%" (
    echo  ERROR: Could not find virtual environment activation script:
    echo    "%VENV_ACTIVATE%"
    echo  Create the venv first or update this launcher.
    echo.
    pause
    exit /b 1
)

:: Start backend in a separate window so this launcher can continue
start "SAMVAAD Server" cmd /k "cd /d ""%ROOT_DIR%"" && call ""%VENV_ACTIVATE%"" && python app.py"

:: Give Flask a moment to come up, then open the browser
timeout /t 3 /nobreak >nul
start "" "%SAMVAAD_URL%"

echo  SAMVAAD should now open in your browser:
echo    %SAMVAAD_URL%
echo.
echo  If the page does not load immediately, wait a few seconds and refresh once.
echo.
pause

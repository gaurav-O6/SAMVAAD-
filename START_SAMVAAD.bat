@echo off
title SAMVAAD Server
color 0A
set "ROOT_DIR=%~dp0"
set "TEMPLATES_DIR=%ROOT_DIR%templates"
set "VENV_ACTIVATE=%ROOT_DIR%venv\Scripts\activate.bat"

echo.
echo  =====================================================
echo    SAMVAAD Backend Starting...
echo    Keep this window open while using the site!
echo  =====================================================
echo.

if not exist "%TEMPLATES_DIR%\app.py" (
    echo  ERROR: Could not find "%TEMPLATES_DIR%\app.py"
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

:: Activate virtual environment and start the server
cd /d "%TEMPLATES_DIR%"
call "%VENV_ACTIVATE%"
python app.py

:: If server crashes, show error and wait
echo.
echo  Server stopped. Press any key to close...
pause

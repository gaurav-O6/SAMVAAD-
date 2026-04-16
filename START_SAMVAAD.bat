@echo off
title SAMVAAD Server
color 0A
echo.
echo  =====================================================
echo    SAMVAAD Backend Starting...
echo    Keep this window open while using the site!
echo  =====================================================
echo.

:: Activate virtual environment and start the server
cd /d D:\SAMVAAD\templates
call D:\SAMVAAD\venv\Scripts\activate.bat
python app.py

:: If server crashes, show error and wait
echo.
echo  Server stopped. Press any key to close...
pause

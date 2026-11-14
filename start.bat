@echo off
setlocal EnableExtensions
title InsightRAG
cd /d "%~dp0"

rem Python prints UTF-8; switch this console to UTF-8 so hints are readable.
chcp 65001 >nul 2>&1

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo   [X] Python venv not found:
    echo       %VENV_PY%
    echo.
    echo       Build it first - see ENVIRONMENT.md section 2.
    echo.
    pause
    exit /b 1
)

"%VENV_PY%" -X utf8 "%~dp0tools\launch.py" %*
set "RC=%ERRORLEVEL%"

rem Always hold the window, even on success: a double-clicked console vanishes
rem instantly otherwise, and "the already-running instance" path exits in ~0.2s.
echo.
if not "%RC%"=="0" (
    echo   [X] exited with code %RC% - see the messages above.
) else (
    echo   Done.
)
echo.
pause

endlocal & exit /b %RC%

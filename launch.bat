@echo off
setlocal

:: ============================================================
::  1. Change to script directory (works on double-click)
:: ============================================================
cd /d "%~dp0"

:: ============================================================
::  Banner
:: ============================================================
echo.
echo   ==============================================================
echo     MemeTalk - Meme Semantic Search (Local)
echo   ==============================================================
echo.

:: ============================================================
::  2. Detect Python
:: ============================================================
echo [1/5] Detecting Python ...

set "PY="
where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :found_py
)
where py >nul 2>&1
if not errorlevel 1 (
    set "PY=py"
    goto :found_py
)
where python3 >nul 2>&1
if not errorlevel 1 (
    set "PY=python3"
    goto :found_py
)

echo   [ERROR] Python not found. Please install Python 3.11+.
echo          https://www.python.org/downloads/
goto :error

:found_py
echo       Found: %PY%

:: ============================================================
::  3. Detect / create virtualenv
:: ============================================================
echo [2/5] Detecting virtualenv ...

set "VENV_DIR="
if exist ".venv\Scripts\activate.bat" (
    set "VENV_DIR=.venv"
    echo       Found existing virtualenv: .venv
    goto :has_venv
)
if exist "venv\Scripts\activate.bat" (
    set "VENV_DIR=venv"
    echo       Found existing virtualenv: venv
    goto :has_venv
)
if exist "env\Scripts\activate.bat" (
    set "VENV_DIR=env"
    echo       Found existing virtualenv: env
    goto :has_venv
)

echo       No virtualenv found, creating .venv ...
%PY% -m venv .venv
if errorlevel 1 (
    echo   [ERROR] Failed to create virtualenv.
    goto :error
)
set "VENV_DIR=.venv"
echo       .venv created.

:has_venv

:: ============================================================
::  4. Activate virtualenv
:: ============================================================
echo [3/5] Activating virtualenv (%VENV_DIR%) ...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo   [ERROR] Failed to activate virtualenv.
    goto :error
)
echo       Virtualenv activated.

:: ============================================================
::  5. Install / update dependencies
:: ============================================================
echo [4/5] Installing dependencies ...
echo       pip install -e .[openai,chroma]
echo.
pip install -e .[openai,chroma] --quiet --disable-pip-version-check
if errorlevel 1 (
    echo   [ERROR] Dependency installation failed.
    goto :error
)
echo.
echo       Dependencies ready.

:: ============================================================
::  6. Launch Streamlit
:: ============================================================
echo [5/5] Launching MemeTalk UI ...
echo.
echo   ---------------------------------------------------------------
echo   Browser will open automatically, or go to http://localhost:8501
echo   Press Ctrl+C to stop
echo   ---------------------------------------------------------------
echo.

streamlit run streamlit_app.py
if errorlevel 1 (
    echo.
    echo   [ERROR] Streamlit failed to start.
    goto :error
)
goto :end

:error
echo.
echo   ==================================================
echo   Launch failed. Check the error messages above.
echo   ==================================================
echo.
pause
exit /b 1

:end
endlocal

@echo off
REM One-click launcher for Windows: activates the project's virtual environment (creating one on
REM first run if it does not exist yet), installs dependencies if needed, starts the server, and
REM opens the dashboard in your default browser.
REM
REM Usage: double-click this file, or run it from PowerShell/cmd with:  .\run.bat

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [run.bat] No virtual environment found -- creating one in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [run.bat] Could not create a virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo [run.bat] Checking dependencies (this is quick if they are already installed) ...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo [run.bat] Dependency installation failed. See the error above.
    pause
    exit /b 1
)

echo [run.bat] Starting the server at http://127.0.0.1:8000 ...
echo [run.bat] Press Ctrl+C in this window to stop it.

REM Open the browser only once the server actually answers on port 8000, instead of firing
REM "start" immediately -- "start" does not wait for uvicorn, so the old version would open the
REM browser before the server was listening, showing "This site can't be reached". This polls
REM the port in the background (hidden window) for up to 20 seconds and then opens the browser.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command ^
  "$ok=$false; for ($i=0; $i -lt 40 -and -not $ok; $i++) { try { (New-Object Net.Sockets.TcpClient).Connect('127.0.0.1', 8000); $ok=$true } catch { Start-Sleep -Milliseconds 500 } }; if ($ok) { Start-Process 'http://127.0.0.1:8000' }"

uvicorn backend.main:app --reload

endlocal

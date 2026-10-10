@echo off
setlocal
cd /d "%~dp0"
set "PY="
call :find
if defined PY goto run

echo rAidiology needs Python 3.9 or newer, and it was not found on this computer.
echo It is free and installs for your user only.
where winget >nul 2>&1
if errorlevel 1 goto manual
choice /c YN /m "Install Python now with winget"
if errorlevel 2 goto manual
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :find
if defined PY goto run
echo Python was installed but could not be found yet. Close this window and double-click this file again.
goto fail

:manual
echo Opening the Python download page. Install Python, then double-click this file again.
start "" https://www.python.org/downloads/
goto fail

:run
%PY% reading\local\launcher.py %*
if errorlevel 1 (
  echo.
  echo Something went wrong. The message above says what.
  pause
)
exit /b

:fail
pause
exit /b 1

:find
for %%C in ("py -3" "python" "python3") do (
  if not defined PY (
    %%~C -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=%%~C"
  )
)
if defined PY exit /b
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
  if not defined PY if exist "%%~D\python.exe" (
    "%%~D\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)" >nul 2>&1
    if not errorlevel 1 set PY="%%~D\python.exe"
  )
)
exit /b

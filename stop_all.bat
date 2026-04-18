@echo off
setlocal EnableDelayedExpansion

echo [INFO] Stopping backend/frontend ports...
call :KillPort 8000
call :KillPort 5173
echo [DONE] Stop script completed.
endlocal
exit /b 0

:KillPort
set "PORT=%~1"
set "FOUND=0"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  set "FOUND=1"
  echo [INFO] Killing PID %%P on port %PORT%
  taskkill /PID %%P /F >nul 2>&1
)
if "!FOUND!"=="0" (
  echo [INFO] No LISTENING process found on port %PORT%
)
exit /b 0


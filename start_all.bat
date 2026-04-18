@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "CONDA_BAT=D:\anaconda\Scripts\activate.bat"
set "APROJECT_PY=D:\anaconda\envs\Aproject\python.exe"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

echo [INFO] Project root: %ROOT%

if not exist "%CONDA_BAT%" (
  echo [ERROR] Conda activate script not found: %CONDA_BAT%
  echo [HINT] Edit start_all.bat and update CONDA_BAT path.
  pause
  exit /b 1
)

echo [INFO] Cleaning old ports if occupied...
call "%ROOT%stop_all.bat" >nul 2>&1

echo [INFO] Starting backend on 127.0.0.1:%BACKEND_PORT% ...
if exist "%APROJECT_PY%" (
  start "backend-8000" cmd /k "cd /d ""%ROOT%backend"" && ""%APROJECT_PY%"" -m uvicorn app.main:app --host 127.0.0.1 --port %BACKEND_PORT%"
) else (
  start "backend-8000" cmd /k "cd /d ""%ROOT%backend"" && call ""%CONDA_BAT%"" Aproject && python -m uvicorn app.main:app --host 127.0.0.1 --port %BACKEND_PORT%"
)

echo [INFO] Starting frontend on localhost:%FRONTEND_PORT% ...
start "frontend-5173" cmd /k "cd /d ""%ROOT%frontend"" && npm.cmd run dev -- --host=0.0.0.0 --port=%FRONTEND_PORT%"

echo [INFO] Waiting services to boot...
ping 127.0.0.1 -n 3 >nul

echo [DONE] start_all finished.
echo [TIP] Open http://127.0.0.1:%BACKEND_PORT% and http://localhost:%FRONTEND_PORT%
echo [TIP] Stop all with: .\stop_all.bat
endlocal
exit /b 0


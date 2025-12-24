@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "HOST=127.0.0.1"
set "PORT=7860"
if not "%SHARP_HOST%"=="" set "HOST=%SHARP_HOST%"
if not "%SHARP_PORT%"=="" set "PORT=%SHARP_PORT%"

if not exist "%VENV_PY%" (
  echo 未检测到虚拟环境 .venv。
  choice /M "是否现在运行初始化（init.bat）"
  if errorlevel 2 (
    echo 已取消，请先运行 init.bat 后再启动。
    pause
    exit /b 1
  )
  call "%~dp0init.bat"
)

if not exist "%VENV_PY%" (
  echo 虚拟环境仍未就绪，请检查初始化过程。
  pause
  exit /b 1
)

echo 正在启动 SHARP Web...
echo 地址: http://%HOST%:%PORT%/
start "" "http://%HOST%:%PORT%/"
"%VENV_PY%" app\server.py
echo.
echo SHARP Web 已停止。
pause

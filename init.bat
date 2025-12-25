@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo "未检测到 .venv 目录，正在创建虚拟环境..."
  python -m venv .venv 2>nul
  if not exist "%VENV_PY%" (
    py -3 -m venv .venv 2>nul
  )
)

if not exist "%VENV_PY%" (
  echo "虚拟环境创建失败，请先安装 Python 3.9+ 并重试。"
  pause
  exit /b 1
)

echo "正在检查环境..."
"%VENV_PY%" check_env.py
echo.
echo "正在初始化环境..."
"%VENV_PY%" init_env.py
echo.
echo "初始化完成。"
echo Done
pause
